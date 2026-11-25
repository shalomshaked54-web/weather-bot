"""3.1: הרצה חוזרת של הניתוח על נתוני מכ"ם שנשמרו (קבצי npz מה-artifact של הריצה).

    python -m rainbot.replay out/alert_20261125_1830.npz [עוד קבצים...] [--png-dir dir]

מדפיס לכל קובץ מה הבוט היה מחליט היום, כך שאפשר לבדוק שינוי בספים מול אירועים אמיתיים.
"""
import argparse
import json
import os
from datetime import datetime

from .alerts import apply_sources, qualifies
from .analysis import analyze
from .config import TZ
from .messages import build_message
from .render import render_png
from .storage import load_snapshot


def replay_file(path):
    fields, times, meta = load_snapshot(path)
    radar_time = datetime.fromtimestamp(times[-1], TZ)
    res = analyze(fields, times)
    apply_sources(res, None, None, radar_time)
    out = {"file": os.path.basename(path), "radar": radar_time.isoformat(timespec="minutes"),
           "then": {k: meta.get(k) for k in ("stage", "src", "sent", "v")},
           "now": {"stage": res["stage"], "src": res["src"], "score": res["score"], "min_score": res["min_score"],
                   "eta": res["eta"], "trend": res.get("trend"), "alert": qualifies(res), "cell": res["cell_dbg"]}}
    msg = build_message(res, radar_time, radar_time) if qualifies(res) else None
    return out, msg, res, fields[-1], radar_time


def main(argv=None):
    ap = argparse.ArgumentParser(description="הרצה חוזרת על נתוני מכ\"ם שמורים")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--png-dir", help="שמירת תמונת מכ\"ם לכל קובץ")
    a = ap.parse_args(argv)
    for p in a.files:
        out, msg, res, raw, radar_time = replay_file(p)
        print(json.dumps(out, ensure_ascii=False))
        if msg:
            print(msg)
        if a.png_dir:
            os.makedirs(a.png_dir, exist_ok=True)
            name = os.path.splitext(os.path.basename(p))[0] + "_replay.png"
            with open(os.path.join(a.png_dir, name), "wb") as f:
                f.write(render_png(raw, res, radar_time))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
