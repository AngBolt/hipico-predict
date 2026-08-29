"""Predictor de apuestas hipicas (Gijon - Equipe).

Modalidades:
- Ganador de serie: mejor binomio de cada serie (~6 caballos por orden de salida).
- Gemela: los 2 mejores de la serie, sin importar orden.
- Ganador de la prueba: mejor de toda la prueba.
- Triple gemela: acertar la gemela de las 3 ultimas series del ultimo trofeo
  del dia. Se pueden marcar varias gemelas por serie; el coste se multiplica
  (p.ej. 3 x 3 x 3 combinaciones x 0.30 EUR = 8.10 EUR).

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

SERIE_TARGET = 6        # tamano objetivo de cada serie
TICKET_UNIT = 0.30      # coste por combinacion de triple gemela
GEMELAS_MARCADAS = 3    # gemelas marcadas por serie en la triple gemela
DECAY = 0.7             # peso por dia de antiguedad
LEVEL_BONUS = 1.0       # peso extra si el historial es del mismo nivel (CSI4*, etc.)
W_COMBO, W_RIDER = 0.8, 0.2
W_CLEAR = 0.35          # peso del componente "cero faltas" en el rendimiento
PRIOR = 0.5             # puntuacion para binomios sin ningun historial


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
    """Divide la lista de salida en series de ~SERIE_TARGET por orden de salida."""
    n = len(entries)
    if n == 0:
        return []
    n_series = max(1, round(n / SERIE_TARGET))
    base, extra = divmod(n, n_series)
    series, i = [], 0
    for k in range(n_series):
        size = base + (1 if k < extra else 0)
        series.append(entries[i:i + size])
        i += size
    return series


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

    @staticmethod
    def _wavg(records, today, level):
        num = den = 0.0
        for d, lvl, p in records:
            w = DECAY ** max(0, (today - d).days)
            if lvl == level:
                w *= LEVEL_BONUS
            num += w * p
            den += w
        return num / den if den else None

    def score(self, e, cls):
        today = date.fromisoformat(cls["date"])
        c = self._wavg(self.combo.get((e["rider_id"], e["horse_id"]), []), today, cls["level"])
        r = self._wavg(self.rider.get(e["rider_id"], []), today, cls["level"])
        if c is not None and r is not None:
            return W_COMBO * c + W_RIDER * r
        if r is not None:
            return 0.85 * r
        return PRIOR


def predict_class(model, cls):
    """Devuelve series con binomios puntuados y ordenados (mejor primero)."""
    scored = []
    for e in cls["entries"]:
        e = dict(e)
        e["score"] = model.score(e, cls)
        scored.append(e)
    series = []
    for serie in split_series(scored):
        ranked = sorted(serie, key=lambda x: -x["score"])
        pairs = [frozenset([(a["rider_id"], a["horse_id"]), (b["rider_id"], b["horse_id"])])
                 for a, b in combinations(ranked[:GEMELAS_MARCADAS], 2)]
        series.append({"entries": serie, "ranked": ranked, "gemelas": pairs})
    winner = max(scored, key=lambda x: x["score"]) if scored else None
    return {"series": series, "winner": winner}


def actual_serie_order(serie):
    """Orden real de la serie por clasificacion de la prueba (sin rank = ultimo)."""
    return sorted(serie, key=lambda e: e["rank"] if e["rank"] else 10**6)


def key_of(e):
    return (e["rider_id"], e["horse_id"])


def backtest(classes):
    model = Model()
    rows, day_last = [], {}
    finished = [c for c in classes if c["state"] == "results"]
    for c in finished:
        day_last[c["date"]] = c["class_no"]  # ultimo trofeo (cronologico) de cada dia

    triple_days = {}
    for cls in finished:
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
                "num": idx + 1, "pick": pick, "real": real_win,
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
            cost = (GEMELAS_MARCADAS ** 3) * TICKET_UNIT
            triple_days[cls["date"]] = {"hit": gem3_all, "cost": cost, "class_no": cls["class_no"]}
        model.learn(cls)
    return rows, triple_days, model


def esc(s):
    return html.escape(str(s))


def fmt_combo(e):
    return f"{esc(e['rider'])} / {esc(e['horse'])}"


def render(classes, rows, triples, model):
    upcoming = [c for c in classes if c["state"] != "results" and c["entries"]]
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
             f"Datos: online.equipe.com &middot; Series de ~{SERIE_TARGET} por orden de salida &middot; "
             f"Triple gemela: {GEMELAS_MARCADAS}x{GEMELAS_MARCADAS}x{GEMELAS_MARCADAS} = "
             f"{GEMELAS_MARCADAS**3} combinaciones x {TICKET_UNIT:.2f}&euro; = "
             f"{GEMELAS_MARCADAS**3*TICKET_UNIT:.2f}&euro;</p>")

    # ---- Predicciones proximas pruebas ----
    p.append("<h2>Predicciones &mdash; pr&oacute;ximas pruebas</h2>")
    if not upcoming:
        p.append("<p>No hay listas de salida publicadas. Ejecuta <code>python fetch_data.py</code> y regenera.</p>")
    for cls in upcoming:
        pred = predict_class(model, cls)
        p.append(f"<h3>Prueba {esc(cls['class_no'])} &middot; {esc(cls['date'])} &middot; {esc(cls['name'])}</h3>")
        if pred["winner"]:
            p.append(f"<p class='big'>&#127942; Ganador de la prueba: {fmt_combo(pred['winner'])} "
                     f"<span class='muted'>(score {pred['winner']['score']:.3f})</span></p>")
        for i, s in enumerate(pred["series"], 1):
            r = s["ranked"]
            p.append("<div class='serie'>")
            p.append(f"<b>Serie {i}</b> <span class='muted'>({len(s['entries'])} binomios, "
                     f"dorsales {esc(s['entries'][0]['start_no'])}&ndash;{esc(s['entries'][-1]['start_no'])})</span><br>")
            p.append(f"&#129351; Ganador serie: <b>{fmt_combo(r[0])}</b> <span class='muted'>({r[0]['score']:.3f})</span><br>")
            if len(r) > 1:
                p.append(f"&#128111; Gemela: <b>{fmt_combo(r[0])}</b> + <b>{fmt_combo(r[1])}</b><br>")
            marks = r[:GEMELAS_MARCADAS]
            p.append("Marcar (triple gemela): " +
                     " ".join(f"<span class='pill'>{fmt_combo(e)}</span>" for e in marks))
            p.append("</div>")
        if day_last_up.get(cls["date"]) == cls["class_no"] and len(pred["series"]) >= 3:
            p.append("<div class='triple'><b>&#127919; TRIPLE GEMELA del d&iacute;a "
                     f"{esc(cls['date'])}</b> (3 &uacute;ltimas series de esta prueba): "
                     f"marcar las {GEMELAS_MARCADAS} combinaciones indicadas en las series "
                     f"{len(pred['series'])-2}, {len(pred['series'])-1} y {len(pred['series'])}. "
                     f"Coste boleto: <b>{GEMELAS_MARCADAS**3*TICKET_UNIT:.2f}&euro;</b></div>")

    # ---- Backtest ----
    p.append("<h2>Backtest &mdash; resultados reales vs predicci&oacute;n</h2>")
    p.append("<p class='muted'>La predicci&oacute;n de cada prueba usa solo informaci&oacute;n "
             "de pruebas anteriores. Las primeras pruebas del martes no tienen historial "
             "(el modelo apenas puede acertar; sirven de l&iacute;nea base).</p>")
    tw = sum(r["n_series"] for r in rows)
    p.append("<table><tr><th>Prueba</th><th>Fecha</th><th>Series</th>"
             "<th>Ganador serie</th><th>Gemela (1 marca)</th>"
             f"<th>Gemela ({GEMELAS_MARCADAS} marcas)</th><th>Ganador prueba</th></tr>")
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
    p.append("</table>")

    p.append("<h3>Triple gemela (&uacute;ltimo trofeo de cada d&iacute;a)</h3><table>"
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
        p.append("<table><tr><th>Serie</th><th>Predicho</th><th>Ganador real</th>"
                 "<th>Ganador</th><th>Gemela 1</th><th>Gemela 3</th></tr>")
        for s in r["series"]:
            f1 = "<span class='ok'>&#10004;</span>" if s["hit_w"] else "<span class='ko'>&#10008;</span>"
            f2 = "<span class='ok'>&#10004;</span>" if s["hit_g1"] else "<span class='ko'>&#10008;</span>"
            f3 = "<span class='ok'>&#10004;</span>" if s["hit_g3"] else "<span class='ko'>&#10008;</span>"
            p.append(f"<tr><td>{s['num']}</td><td>{fmt_combo(s['pick'])}</td>"
                     f"<td>{fmt_combo(s['real'])}</td><td>{f1}</td><td>{f2}</td><td>{f3}</td></tr>")
        p.append("</table>")

    p.append("<h2>Metodolog&iacute;a</h2><ul>"
             "<li>Puntuaci&oacute;n de cada binomio = historial ponderado en el propio concurso: "
             "percentil de clasificaci&oacute;n en pruebas anteriores, con decaimiento temporal "
             f"({DECAY}/d&iacute;a) y bonus x{LEVEL_BONUS} si es del mismo nivel (CSI4*/CSI2*/CSIYH1*).</li>"
             f"<li>Mezcla: {W_COMBO:.0%} historial jinete+caballo, {W_RIDER:.0%} historial del jinete "
             f"(cualquier caballo). Sin historial: prior {PRIOR}.</li>"
             f"<li>Series: divisi&oacute;n de la lista de salida en grupos de ~{SERIE_TARGET} por orden de salida "
             "(configurable en <code>predict.py</code>).</li>"
             "<li>Mejora futura: integrar ranking FEI/Longines como prior para el primer d&iacute;a.</li>"
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
