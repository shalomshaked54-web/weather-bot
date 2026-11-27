"""3.2: בדיקה לאחור על אירועי גשם אמיתיים מהארכיון של מכ"ם השירות המטאורולוגי.

    python -m rainbot.backtest --start 2026-11-07T06:00 --end 2026-11-08T16:00 [--out out/backtest.json]

מריץ את הבוט כאילו היה חי, פריים אחרי פריים (כל 10 דקות), כולל ההחלטות על התראות ומערכת
הלמידה, ומדפיס: מתי הגשם באמת התחיל אצלך, כמה זמן מראש הייתה התראה, וכמה החיזוי טעה בכל טווח.
--seed-state state/state.json שומר את מה שנלמד כנקודת התחלה לבוט החי (כיול לפני הסערה).
"""
import argparse
import json
import os
import statistics
from datetime import datetime, timedelta

import requests

from . import alerts, learn
from .analysis import analyze
from .config import TZ
from .ims_radar import HEADERS, IMS_RADAR_BASE, decode_image
from .radar import http_get

PATH = "/sites/default/files/ims_data/map_images/IMSRadar4GIS/IMSRadar4GIS_{:%Y%m%d%H%M}_0.png"


class Frames:
    def __init__(self, session):
        self.session, self.cache = session, {}

    def get(self, t):
        if t not in self.cache:
            try:
                self.cache[t] = decode_image(http_get(self.session, IMS_RADAR_BASE + PATH.format(t), tries=2,
                                                      headers=HEADERS).content)
            except (RuntimeError, OSError, ValueError):
                self.cache[t] = None
        return self.cache[t]


def run(start, end, step=10, state=None, frames=None, log=print):
    state = state if state is not None else {}
    rows = []
    t = start
    with requests.Session() as session:
        frames = frames or Frames(session)
        while t <= end:
            ts = [t - timedelta(minutes=step * k) for k in (3, 2, 1, 0)]
            fields = [frames.get(x) for x in ts]
            if any(f is None for f in fields):
                t += timedelta(minutes=step)
                continue
            times = [int(x.timestamp()) for x in ts]
            res = analyze(fields, times, spread=learn.params(state)["system"]["spread"])
            learn.apply(res, state)
            alerts.apply_sources(res, None, None, t)
            sent, why = False, None
            if alerts.qualifies(res):
                go, why = alerts.send_decision(state, res["stage"], t, t.isoformat(timespec="seconds"),
                                               res.get("peak_dbz"))
                sent = go
            alerts.update_state(state, res, t, t, sent)
            alert_times = [learn.parse_ts(a["t"]) for a in state.get("alerts", [])]
            ev = learn.observe(state, fields[-1], t, alert_times)
            learn.record(state, res, t, sent)
            nc = res.get("nowcast") or {}
            row = {"t": t.strftime("%m-%d %H:%M"), "stage": res["stage"], "src": res["src"], "eta": res["eta"],
                   "p": nc.get("p"), "at": learn.point_wet(fields[-1]), "sent": why if sent else None}
            rows.append(row)
            if sent or ev:
                log(json.dumps({**row, "event": ev}, ensure_ascii=False))
            t += timedelta(minutes=step)
    return state, rows


def report(state):
    lr = state.get("learn") or {}
    ss = lr.get("samples", [])
    out = {"events": lr.get("events", []), "params": learn.params(state),
           "false_alarms": lr.get("false_alarms", 0), "misses": lr.get("misses", 0), "by_lead": {}}
    for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 60), (60, 120)):
        e = [s["err"] for s in ss if lo <= s["obs"] < hi]
        if e:
            out["by_lead"][f"{lo}-{hi}"] = {"n": len(e), "median_err": round(statistics.median(e), 1),
                                            "median_abs_err": round(statistics.median(abs(x) for x in e), 1),
                                            "inside": round(sum(s["inside"] for s in ss if lo <= s["obs"] < hi)
                                                            / len(e), 2)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="בדיקה לאחור על הארכיון של מכ\"ם השירות המטאורולוגי")
    ap.add_argument("--start", required=True, help="זמן מקומי, למשל 2026-11-07T06:00")
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", help="קובץ JSON לדוח")
    ap.add_argument("--seed-state", help="שמירת מה שנלמד לקובץ מצב (נקודת התחלה לבוט החי)")
    a = ap.parse_args(argv)
    start = datetime.fromisoformat(a.start).replace(tzinfo=TZ)
    end = datetime.fromisoformat(a.end).replace(tzinfo=TZ)
    state, rows = run(start, end)
    rep = report(state)
    rep["frames"] = len(rows)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"report": rep, "rows": rows}, f, ensure_ascii=False)
    if a.seed_state:
        seed = {"learn": {k: state["learn"].get(k) for k in ("params", "samples", "events")}}
        seed["learn"]["seeded_from"] = f"{a.start}..{a.end}"
        os.makedirs(os.path.dirname(a.seed_state) or ".", exist_ok=True)
        with open(a.seed_state, "w") as f:
            json.dump(seed, f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
