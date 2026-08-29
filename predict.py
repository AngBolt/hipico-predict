"""Predictor de apuestas hipicas (Gijon - Equipe).

Modalidades:
- Ganador de serie: mejor binomio de cada serie (~6 caballos por orden de salida).
- Gemela: los 2 mejores de la serie, sin importar orden. Una gemela (1 pareja)
  cuesta 2 EUR. Una "combinada de 3" (3 caballos combinados entre si = 3 parejas)
  cuesta 6 EUR.
- Ganador de la prueba: mejor de toda la prueba.
- Triple gemela: SOLO en las 3 ultimas series de la ultima prueba del dia.
  Consiste en acertar la gemela de esas 3 series. Se pueden marcar varias
  gemelas por serie y el coste del boleto se multiplica
  (p.ej. 3 x 3 x 3 = 27 combinaciones x 0.30 EUR = 8.10 EUR).

Uso: python predict.py   -> genera index.html con predicciones y backtest.
"""
import html
import json
import re
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "index.html"

SERIE_SIZE = 10         # caballos por serie (consecutivos por orden de salida)
MAX_SERIES = 4          # maximo de series con apuestas por prueba
BET_LEVELS = {"CSI4*", "CSI2*"}   # los CSIYH1* (caballos jovenes) no tienen apuestas
GEMELA_COST = 2.0       # coste de una gemela (1 pareja)
COMBINADA = 3           # caballos en la combinada de gemela -> C(3,2)=3 parejas = 6 EUR
TRIPLE_UNIT = 0.30      # coste por combinacion del boleto de triple gemela
GEMELAS_MARCADAS = 3    # gemelas marcadas por serie en la triple gemela
DECAY = 0.6             # peso por dia de antiguedad
LEVEL_BONUS = 1.0       # peso extra si el historial es del mismo nivel (CSI4*, etc.)
W_COMBO, W_RIDER, W_HORSE = 0.4, 0.1, 0.5   # mezcla binomio / jinete / caballo
K_SMOOTH = 0.5          # suavizado bayesiano hacia el prior (nº obs. equivalentes)
W_CLEAR = 0.5           # peso del componente "cero faltas" en el rendimiento
PRIOR = 0.5             # puntuacion para binomios sin ningun historial
CONF_MIN = 0.025        # diferencia top1-top2 para considerar serie "fiable" (apostar)


def level_of(name):
    m = re.search(r"(CSIYH1\*|CSI4\*|CSI2\*)", name or "")
    return m.group(1) if m else "?"


def load_classes():
    schedule = json.loads((DATA / "schedule.json").read_text(encoding="utf-8"))
    classes = []
    for mc in schedule["meeting_classes"]:
        no = mc.get("class_no")
        if not no or not re.fullmatch(r"\d+", no):
            continue
        for cs in mc.get("class_sections", []):
            f = DATA / f"cs_{cs['id']}.json"
            if not f.exists():
                continue
            sec = json.loads(f.read_text(encoding="utf-8"))
            starts = sorted(sec.get("starts", []), key=lambda s: s.get("position", 0))
            entries = []
            for s in starts:
                faults = None
                if s.get("results"):
                    faults = sum((r.get("faults") or 0) + (r.get("time_faults") or 0)
                                 for r in s["results"] if isinstance(r, dict))
                entries.append({
                    "rider_id": s["rider_id"],
                    "horse_id": s["horse_id"],
                    "rider": s.get("rider_name", "?"),
                    "horse": s.get("horse_name", "?"),
                    "start_no": s.get("start_no", ""),
                    "rank": s.get("rank"),
                    "ridden": s.get("ridden", False),
                    "faults": faults,
                    "country": s.get("logo_id", ""),
                })
            classes.append({
                "class_no": no,
                "name": mc["name"].strip(),
                "date": mc["date"],
                "start_at": mc.get("start_at", ""),
                "level": level_of(mc["name"]),
                "state": sec.get("state"),
                "entries": entries,
            })
    classes.sort(key=lambda c: (c["date"], c["start_at"]))
    return classes


def split_series(entries):
    """Series oficiales de apuestas: bloques de SERIE_SIZE consecutivos por orden
    de salida, anclados al FINAL de la lista (max MAX_SERIES series). Los primeros
    de la lista que sobran quedan fuera de las apuestas.

    Calibrado con el programa real (GP CSI2* 29/08: 57 inscritos -> 4 series
    empezando en el dorsal 18). Devuelve (series, excluidos).
    """
    n = len(entries)
    if n == 0:
        return [], []
    k = min(MAX_SERIES, n // SERIE_SIZE)
    if k == 0:
        return [entries], []
    start = n - k * SERIE_SIZE
    series = [entries[start + i * SERIE_SIZE: start + (i + 1) * SERIE_SIZE]
              for i in range(k)]
    return series, entries[:start]


def perf_of(rank, n):
    """Rendimiento en [0,1]: 1 = gano, 0 = ultimo/eliminado."""
    if not rank or n <= 1:
        return 0.0
    return max(0.0, 1.0 - (rank - 1) / (n - 1))


class Model:
    """Puntuacion por historial ponderado (recencia + nivel) dentro del concurso."""

    def __init__(self):
        self.combo = {}   # (rider_id, horse_id) -> [(date, level, perf)]
        self.rider = {}   # rider_id -> [(date, level, perf)]
        self.horse = {}   # horse_id -> [(date, level, perf)]

    def learn(self, cls):
        n = len([e for e in cls["entries"] if e["ridden"] or e["rank"]])
        d = date.fromisoformat(cls["date"])
        for e in cls["entries"]:
            p = perf_of(e["rank"], n)
            clear = 1.0 if (e["rank"] and e.get("faults") == 0) else 0.0
            p = (1 - W_CLEAR) * p + W_CLEAR * clear
            rec = (d, cls["level"], p)
            self.combo.setdefault((e["rider_id"], e["horse_id"]), []).append(rec)
            self.rider.setdefault(e["rider_id"], []).append(rec)
            self.horse.setdefault(e["horse_id"], []).append(rec)

    @staticmethod
    def _smooth(records, today, level):
        """Media ponderada con suavizado bayesiano hacia PRIOR."""
        num, den = K_SMOOTH * PRIOR, K_SMOOTH
        for d, lvl, p in records:
            w = DECAY ** max(0, (today - d).days)
            if lvl == level:
                w *= LEVEL_BONUS
            num += w * p
            den += w
        return num / den

    def score(self, e, cls):
        today = date.fromisoformat(cls["date"])
        c = self._smooth(self.combo.get((e["rider_id"], e["horse_id"]), []), today, cls["level"])
        r = self._smooth(self.rider.get(e["rider_id"], []), today, cls["level"])
        h = self._smooth(self.horse.get(e["horse_id"], []), today, cls["level"])
        return W_COMBO * c + W_RIDER * r + W_HORSE * h


def predict_class(model, cls):
    """Devuelve series con binomios puntuados y ordenados (mejor primero)."""
    scored = []
    for e in cls["entries"]:
        e = dict(e)
        e["score"] = model.score(e, cls)
        scored.append(e)
    series = []
    blocks, excluded = split_series(scored)
    for serie in blocks:
        ranked = sorted(serie, key=lambda x: -x["score"])
        pairs = [frozenset([(a["rider_id"], a["horse_id"]), (b["rider_id"], b["horse_id"])])
                 for a, b in combinations(ranked[:GEMELAS_MARCADAS], 2)]
        conf = (ranked[0]["score"] - ranked[1]["score"]) if len(ranked) > 1 else 0.0
        series.append({"entries": serie, "ranked": ranked, "gemelas": pairs, "conf": conf})
    winner = max(scored, key=lambda x: x["score"]) if scored else None
    return {"series": series, "winner": winner, "excluded": excluded}


def actual_serie_order(serie):
    """Orden real de la serie por clasificacion de la prueba (sin rank = ultimo)."""
    return sorted(serie, key=lambda e: e["rank"] if e["rank"] else 10**6)


def key_of(e):
    return (e["rider_id"], e["horse_id"])


def day_serie_offsets(classes):
    """Numeracion oficial: las series se numeran de forma continua durante el dia
    (p.ej. prueba de las 14:00 = series 1-4, prueba de las 18:30 = series 5-7).
    Devuelve {(date, class_no): offset}."""
    offsets, counters = {}, {}
    for c in classes:
        if c["level"] not in BET_LEVELS or not c["entries"]:
            continue
        offsets[(c["date"], c["class_no"])] = counters.get(c["date"], 0)
        blocks, _ = split_series(c["entries"])
        counters[c["date"]] = counters.get(c["date"], 0) + len(blocks)
    return offsets


def backtest(classes):
    model = Model()
    rows, day_last = [], {}
    offsets = day_serie_offsets(classes)
    finished = [c for c in classes if c["state"] == "results"]
    for c in finished:
        if c["level"] in BET_LEVELS:
            day_last[c["date"]] = c["class_no"]  # ultima prueba con apuestas del dia

    triple_days = {}
    for cls in finished:
        if cls["level"] not in BET_LEVELS:
            model.learn(cls)   # los CSIYH no tienen apuestas, pero si aportan historial
            continue
        off = offsets.get((cls["date"], cls["class_no"]), 0)
        pred = predict_class(model, cls)
        n_series = len(pred["series"])
        win_hits = gem1_hits = gem3_hits = 0
        serie_detail = []
        gem3_all = True
        last3 = []
        for idx, s in enumerate(pred["series"]):
            order = actual_serie_order(s["entries"])
            real_win, real_2nd = order[0], (order[1] if len(order) > 1 else None)
            pick = s["ranked"][0]
            hit_w = key_of(pick) == key_of(real_win)
            win_hits += hit_w
            hit_g1 = hit_g3 = False
            if real_2nd and real_2nd["rank"]:
                real_pair = frozenset([key_of(real_win), key_of(real_2nd)])
                pred_pair = frozenset([key_of(s["ranked"][0]), key_of(s["ranked"][1])])
                hit_g1 = pred_pair == real_pair
                hit_g3 = real_pair in s["gemelas"]
            gem1_hits += hit_g1
            gem3_hits += hit_g3
            serie_detail.append({
                "num": off + idx + 1, "pick": pick, "real": real_win,
                "marks": s["ranked"][:COMBINADA], "conf": s["conf"],
                "hit_w": hit_w, "hit_g1": hit_g1, "hit_g3": hit_g3,
            })
            if idx >= n_series - 3:
                last3.append(hit_g3)
        gem3_all = len(last3) == 3 and all(last3)

        real_ranked = actual_serie_order(cls["entries"])
        hit_class_winner = pred["winner"] and real_ranked and \
            key_of(pred["winner"]) == key_of(real_ranked[0])

        row = {
            "cls": cls, "n_series": n_series,
            "win_hits": win_hits, "gem1_hits": gem1_hits, "gem3_hits": gem3_hits,
            "hit_class_winner": bool(hit_class_winner),
            "pred_winner": pred["winner"], "real_winner": real_ranked[0] if real_ranked else None,
            "series": serie_detail,
        }
        rows.append(row)
        if day_last.get(cls["date"]) == cls["class_no"]:
            cost = (GEMELAS_MARCADAS ** 3) * TRIPLE_UNIT
            triple_days[cls["date"]] = {"hit": gem3_all, "cost": cost, "class_no": cls["class_no"]}
        model.learn(cls)
    return rows, triple_days, model


def esc(s):
    return html.escape(str(s))


def fmt_combo(e):
    return f"{esc(e['rider'])} / {esc(e['horse'])}"


def render(classes, rows, triples, model):
    upcoming = [c for c in classes if c["state"] != "results" and c["entries"]
                and c["level"] in BET_LEVELS]
    offsets = day_serie_offsets(classes)
    day_last_up = {}
    for c in upcoming:
        day_last_up[c["date"]] = c["class_no"]

    css = """
    body{font-family:Segoe UI,Arial,sans-serif;background:#14172b;color:#e8e8ef;margin:0;padding:0 1rem 3rem}
    .wrap{max-width:1100px;margin:0 auto}
    h1{color:#ffd166}h2{color:#7ecbff;border-bottom:1px solid #2c3054;padding-bottom:.3rem;margin-top:2.2rem}
    h3{color:#ffd166;margin-bottom:.3rem}
    table{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.92rem}
    th,td{border:1px solid #2c3054;padding:.35rem .55rem;text-align:left}
    th{background:#1d2140}tr:nth-child(even){background:#191d38}
    .ok{color:#6fe38f;font-weight:700}.ko{color:#ff7b7b}
    .pill{display:inline-block;background:#26305c;border-radius:1rem;padding:.05rem .6rem;margin:.1rem;font-size:.85rem}
    .serie{background:#1a1e3c;border:1px solid #2c3054;border-radius:.6rem;padding:.6rem .9rem;margin:.6rem 0}
    .muted{color:#9aa0c0;font-size:.85rem}
    .big{font-size:1.05rem;font-weight:600}
    .triple{background:#26224a;border:1px solid #574fd6;border-radius:.6rem;padding:.8rem 1rem;margin:1rem 0}
    """

    p = []
    p.append(f"<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Predicciones Hipico Gijon</title><style>{css}</style></head><body><div class='wrap'>")
    p.append("<h1>&#127943; Predicciones apuestas &mdash; Hipico de Gijon (Las Mestas)</h1>")
    p.append(f"<p class='muted'>Generado: {datetime.now():%Y-%m-%d %H:%M} &middot; "
             f"Datos: online.equipe.com &middot; Series oficiales: bloques de {SERIE_SIZE} "
             f"por orden de salida (m&aacute;x. {MAX_SERIES} series, ancladas al final de la lista; "
             "CSIYH1* sin apuestas) &middot; "
             f"Gemela: {GEMELA_COST:.0f}&euro; &middot; Combinada de {COMBINADA} caballos = "
             f"{COMBINADA*(COMBINADA-1)//2} gemelas = {COMBINADA*(COMBINADA-1)//2*GEMELA_COST:.0f}&euro; &middot; "
             f"Triple gemela (solo 3 &uacute;ltimas series de la &uacute;ltima prueba del d&iacute;a): "
             f"{GEMELAS_MARCADAS}x{GEMELAS_MARCADAS}x{GEMELAS_MARCADAS} = "
             f"{GEMELAS_MARCADAS**3} combinaciones x {TRIPLE_UNIT:.2f}&euro; = "
             f"{GEMELAS_MARCADAS**3*TRIPLE_UNIT:.2f}&euro;</p>")

    # ---- Predicciones proximas pruebas ----
    p.append("<h2>Predicciones &mdash; pr&oacute;ximas pruebas</h2>")
    if not upcoming:
        p.append("<p>No hay listas de salida publicadas. Ejecuta <code>python fetch_data.py</code> y regenera.</p>")
    for cls in upcoming:
        pred = predict_class(model, cls)
        off = offsets.get((cls["date"], cls["class_no"]), 0)
        is_last_of_day = day_last_up.get(cls["date"]) == cls["class_no"]
        n_s = len(pred["series"])
        triple_series = set(range(off + n_s - 2, off + n_s + 1)) if (is_last_of_day and n_s >= 3) else set()
        p.append(f"<h3>Prueba {esc(cls['class_no'])} &middot; {esc(cls['date'])} &middot; {esc(cls['name'])}</h3>")
        if pred["winner"]:
            p.append(f"<p class='big'>&#127942; Ganador de la prueba: {fmt_combo(pred['winner'])} "
                     f"<span class='muted'>(score {pred['winner']['score']:.3f})</span></p>")
        if pred["excluded"]:
            p.append(f"<p class='muted'>Sin apuestas (primeros {len(pred['excluded'])} del orden de salida): "
                     f"dorsales {esc(pred['excluded'][0]['start_no'])}&ndash;{esc(pred['excluded'][-1]['start_no'])}</p>")
        for i, s in enumerate(pred["series"], off + 1):
            r = s["ranked"]
            bet = s["conf"] >= CONF_MIN
            badge = ("<span class='ok'>&#11088; FIABLE &mdash; apostar</span>" if bet
                     else "<span class='muted'>&#9888; serie igualada &mdash; mejor no apostar</span>")
            p.append("<div class='serie'>")
            p.append(f"<b>Serie {i}</b> <span class='muted'>({len(s['entries'])} binomios, "
                     f"dorsales {esc(s['entries'][0]['start_no'])}&ndash;{esc(s['entries'][-1]['start_no'])})</span> "
                     f"&middot; confianza {s['conf']:.3f} &middot; {badge}<br>")
            p.append(f"&#129351; Ganador serie: <b>{fmt_combo(r[0])}</b> <span class='muted'>({r[0]['score']:.3f})</span><br>")
            if len(r) > 1:
                p.append(f"&#128111; Gemela ({GEMELA_COST:.0f}&euro;): <b>{fmt_combo(r[0])}</b> + <b>{fmt_combo(r[1])}</b><br>")
            if len(r) >= COMBINADA:
                marks = r[:COMBINADA]
                n_pairs = COMBINADA * (COMBINADA - 1) // 2
                p.append(f"&#127922; Combinada de {COMBINADA} ({n_pairs} gemelas, "
                         f"{n_pairs*GEMELA_COST:.0f}&euro;) &mdash; los 3 participantes: " +
                         " ".join(f"<span class='pill'>{esc(e['start_no'])}. {fmt_combo(e)}</span>" for e in marks))
            if i in triple_series:
                p.append("<br><span class='muted'>&#127919; Esta serie entra en la TRIPLE GEMELA: "
                         "marcar esos mismos 3 caballos como gemelas.</span>")
            p.append("</div>")
        if triple_series:
            ts = sorted(triple_series)
            p.append("<div class='triple'><b>&#127919; TRIPLE GEMELA del d&iacute;a "
                     f"{esc(cls['date'])}</b> &mdash; solo en las 3 &uacute;ltimas series de esta prueba "
                     f"(series {ts[0]}, {ts[1]} y {ts[2]}, numeraci&oacute;n continua del d&iacute;a). "
                     f"Marcando {GEMELAS_MARCADAS} gemelas por serie: "
                     f"{GEMELAS_MARCADAS}x{GEMELAS_MARCADAS}x{GEMELAS_MARCADAS} = {GEMELAS_MARCADAS**3} "
                     f"combinaciones x {TRIPLE_UNIT:.2f}&euro; = boleto de "
                     f"<b>{GEMELAS_MARCADAS**3*TRIPLE_UNIT:.2f}&euro;</b></div>")

    # ---- Backtest ----
    p.append("<h2>Backtest &mdash; resultados reales vs predicci&oacute;n</h2>")
    p.append("<p class='muted'>La predicci&oacute;n de cada prueba usa solo informaci&oacute;n "
             "de pruebas anteriores. Las primeras pruebas del martes no tienen historial "
             "(el modelo apenas puede acertar; sirven de l&iacute;nea base).</p>")
    tw = sum(r["n_series"] for r in rows)
    p.append("<table><tr><th>Prueba</th><th>Fecha</th><th>Series</th>"
             f"<th>Ganador serie</th><th>Gemela simple ({GEMELA_COST:.0f}&euro;)</th>"
             f"<th>Combinada de {COMBINADA} ({COMBINADA*(COMBINADA-1)//2*GEMELA_COST:.0f}&euro;)</th>"
             "<th>Ganador prueba</th></tr>")
    tot_w = tot_g1 = tot_g3 = tot_cw = 0
    for r in rows:
        c = r["cls"]
        tot_w += r["win_hits"]; tot_g1 += r["gem1_hits"]; tot_g3 += r["gem3_hits"]
        tot_cw += r["hit_class_winner"]
        cw = "<span class='ok'>SI</span>" if r["hit_class_winner"] else "<span class='ko'>no</span>"
        p.append(f"<tr><td>{esc(c['class_no'])} {esc(c['name'][:38])}&hellip;</td><td>{esc(c['date'])}</td>"
                 f"<td>{r['n_series']}</td><td>{r['win_hits']}/{r['n_series']}</td>"
                 f"<td>{r['gem1_hits']}/{r['n_series']}</td><td>{r['gem3_hits']}/{r['n_series']}</td><td>{cw}</td></tr>")
    if tw:
        p.append(f"<tr><th>TOTAL</th><th></th><th>{tw}</th>"
                 f"<th>{tot_w}/{tw} ({100*tot_w/tw:.0f}%)</th>"
                 f"<th>{tot_g1}/{tw} ({100*tot_g1/tw:.0f}%)</th>"
                 f"<th>{tot_g3}/{tw} ({100*tot_g3/tw:.0f}%)</th>"
                 f"<th>{tot_cw}/{len(rows)}</th></tr>")
        hi = [s for r in rows if r["cls"]["date"] > "2026-08-25" for s in r["series"]
              if s["conf"] >= CONF_MIN]
        if hi:
            hw = sum(s["hit_w"] for s in hi)
            hg1 = sum(s["hit_g1"] for s in hi)
            hg3 = sum(s["hit_g3"] for s in hi)
            nh = len(hi)
            p.append(f"<tr><th>Solo series FIABLES (conf &ge; {CONF_MIN}, desde mi&eacute;.)</th><th></th><th>{nh}</th>"
                     f"<th>{hw}/{nh} ({100*hw/nh:.0f}%)</th>"
                     f"<th>{hg1}/{nh} ({100*hg1/nh:.0f}%)</th>"
                     f"<th>{hg3}/{nh} ({100*hg3/nh:.0f}%)</th><th>&mdash;</th></tr>")
    p.append("</table>")
    p.append("<div class='triple'><b>&#128176; Estrategia de banca</b>: con series de 10 caballos, "
             "la fuerza del modelo est&aacute; en la <b>gemela</b>: acierto ~11% frente al ~2% del azar (x5), "
             "y combinada de 3 ~19% frente al 6,7% del azar (x3). El ganador de serie a pelo es poco fiable "
             "(1 entre 10). Recomendaci&oacute;n: jugar gemela/combinada en las series &#11088; FIABLES "
             f"(confianza &ge; {CONF_MIN}) y evitar el resto, salvo la triple gemela que obliga a marcar "
             "las 3 series. Backtest corto (27 series): tomar los porcentajes con cautela.</div>")

    p.append("<h3>Triple gemela (3 &uacute;ltimas series de la &uacute;ltima prueba de cada d&iacute;a, "
             f"{GEMELAS_MARCADAS} gemelas marcadas por serie)</h3><table>"
             "<tr><th>D&iacute;a</th><th>Prueba</th><th>Coste boleto</th><th>Acertada</th></tr>")
    for d, t in sorted(triples.items()):
        ok = "<span class='ok'>SI &#127881;</span>" if t["hit"] else "<span class='ko'>no</span>"
        p.append(f"<tr><td>{esc(d)}</td><td>{esc(t['class_no'])}</td>"
                 f"<td>{t['cost']:.2f}&euro;</td><td>{ok}</td></tr>")
    p.append("</table>")

    # ---- Detalle por serie ----
    p.append("<h2>Detalle del backtest por serie</h2>")
    for r in rows:
        c = r["cls"]
        p.append(f"<h3>Prueba {esc(c['class_no'])} &middot; {esc(c['date'])} &middot; {esc(c['name'][:60])}</h3>")
        pw = fmt_combo(r["pred_winner"]) if r["pred_winner"] else "-"
        rw = fmt_combo(r["real_winner"]) if r["real_winner"] else "-"
        mark = "ok" if r["hit_class_winner"] else "ko"
        p.append(f"<p>Ganador prueba &mdash; predicho: <b>{pw}</b> &middot; real: "
                 f"<b class='{mark}'>{rw}</b></p>")
        p.append("<table><tr><th>Serie</th><th>Conf.</th><th>Predicho (ganador)</th>"
                 "<th>Marcados combinada de 3</th><th>Ganador real</th>"
                 "<th>Ganador</th><th>Gemela simple</th><th>Combinada de 3</th></tr>")
        for s in r["series"]:
            f1 = "<span class='ok'>&#10004;</span>" if s["hit_w"] else "<span class='ko'>&#10008;</span>"
            f2 = "<span class='ok'>&#10004;</span>" if s["hit_g1"] else "<span class='ko'>&#10008;</span>"
            f3 = "<span class='ok'>&#10004;</span>" if s["hit_g3"] else "<span class='ko'>&#10008;</span>"
            marks = "<br>".join(f"{j}. {fmt_combo(e)}" for j, e in enumerate(s.get("marks", []), 1))
            conf = f"{s.get('conf', 0):.3f}" + (" &#11088;" if s.get("conf", 0) >= CONF_MIN else "")
            p.append(f"<tr><td>{s['num']}</td><td>{conf}</td><td>{fmt_combo(s['pick'])}</td>"
                     f"<td>{marks}</td>"
                     f"<td>{fmt_combo(s['real'])}</td><td>{f1}</td><td>{f2}</td><td>{f3}</td></tr>")
        p.append("</table>")

    p.append("<h2>Metodolog&iacute;a</h2><ul>"
             "<li>Puntuaci&oacute;n de cada binomio = historial ponderado en el propio concurso: "
             "percentil de clasificaci&oacute;n + componente cero-faltas "
             f"({W_CLEAR:.0%}), con decaimiento temporal ({DECAY}/d&iacute;a) y bonus x{LEVEL_BONUS} "
             "si el historial es del mismo nivel (CSI4*/CSI2*/CSIYH1*).</li>"
             f"<li>Mezcla: {W_COMBO:.0%} binomio jinete+caballo, {W_RIDER:.0%} jinete (otros caballos), "
             f"{W_HORSE:.0%} caballo (otros jinetes). Suavizado bayesiano hacia prior {PRIOR} "
             f"(k={K_SMOOTH}).</li>"
             f"<li>Confianza de serie = score(top1) &minus; score(top2); &ge; {CONF_MIN} = fiable.</li>"
             f"<li>Series: bloques de {SERIE_SIZE} consecutivos por orden de salida, m&aacute;ximo "
             f"{MAX_SERIES} series ancladas al final de la lista; los primeros sobrantes quedan fuera "
             "de apuestas. CSIYH1* sin apuestas (solo aporta historial). "
             "(configurable en <code>predict.py</code>).</li>"
             "<li>Algoritmo completo documentado en <code>README.md</code>.</li>"
             "</ul>")
    p.append("</div></body></html>")
    OUT.write_text("\n".join(p), encoding="utf-8")
    print(f"OK -> {OUT}")


def main():
    classes = load_classes()
    rows, triples, model = backtest(classes)
    render(classes, rows, triples, model)
    tw = sum(r["n_series"] for r in rows)
    if tw:
        print(f"Backtest: {len(rows)} pruebas, {tw} series | "
              f"ganador serie {sum(r['win_hits'] for r in rows)}/{tw} | "
              f"gemela x1 {sum(r['gem1_hits'] for r in rows)}/{tw} | "
              f"gemela x{GEMELAS_MARCADAS} {sum(r['gem3_hits'] for r in rows)}/{tw} | "
              f"ganador prueba {sum(r['hit_class_winner'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
