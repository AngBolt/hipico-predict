"""Descarga datos del concurso desde la API de Equipe.

Uso: python fetch_data.py [meeting_id]   (por defecto 81588)
Guarda schedule.json y cs_<id>.json en data/.
"""
import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://online.equipe.com/api/v1"
DATA = Path(__file__).parent / "data"


def get_json(url):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    meeting_id = sys.argv[1] if len(sys.argv) > 1 else "81588"
    DATA.mkdir(exist_ok=True)

    schedule = get_json(f"{BASE}/meetings/{meeting_id}/schedule")
    (DATA / "schedule.json").write_text(
        json.dumps(schedule, ensure_ascii=False), encoding="utf-8"
    )
    print(f"schedule: {schedule.get('display_name')}")

    for mc in schedule.get("meeting_classes", []):
        if not mc.get("class_no"):
            continue
        for cs in mc.get("class_sections", []):
            cs_id = cs["id"]
            try:
                data = get_json(f"{BASE}/class_sections/{cs_id}")
            except Exception as e:  # noqa: BLE001
                print(f"  cs {cs_id}: ERROR {e}")
                continue
            (DATA / f"cs_{cs_id}.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"  clase {mc.get('class_no')} cs {cs_id}: "
                f"{data.get('state')} ({data.get('total')} binomios)"
            )


if __name__ == "__main__":
    main()
