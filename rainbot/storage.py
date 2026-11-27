"""קבצים: מצב, לוג, status.json לדשבורד, ושמירת נתוני מכ"ם להרצה חוזרת (3.1)"""
import json
import math
import os
from datetime import datetime

import numpy as np

from . import config


def atomic_write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def load_state():
    try:
        with open(config.STATE_FILE) as f:
            st = json.load(f)
            return st if isinstance(st, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state):
    atomic_write(config.STATE_FILE, json.dumps(state, ensure_ascii=False))


def append_history(entry):
    os.makedirs(config.STATE_DIR, exist_ok=True)
    try:
        with open(config.HISTORY_FILE) as f:
            lines = f.read().splitlines()
    except OSError:
        lines = []
    lines.append(json.dumps(entry, ensure_ascii=False))
    atomic_write(config.HISTORY_FILE, "\n".join(lines[-config.HISTORY_MAX_LINES:]) + "\n")


def parse_ts(s):
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def read_trend(n=24):
    trend = []
    try:
        with open(config.HISTORY_FILE) as f:
            for line in f.read().splitlines()[-n:]:
                h = json.loads(line)
                trend.append({"t": h["t"], "n15": h.get("nearest15_km"), "m10": h.get("max_dbz_10km")})
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return trend


def watch_ok(res):
    from .alerts import watch_ok as _ok   # ייבוא מאוחר: alerts מייבא את storage
    return _ok(res)


def write_status(res, summary, state, radar_time, now, sources=None, active=False):
    """status.json: מה שהדשבורד מציג. השדות של 3.0 נשמרים בדיוק; שדות 3.1 כולם אופציונליים:
    sources, intensity_trend, forecast_agree, boost, radar_down_since, rain_ended_at.
    res=None כשאין מכ"ם (הריצה נכשלה)"""
    mu = res.get("motion_used") if res else None
    on = bool(res) and active
    status = {"updated": now.isoformat(timespec="seconds"),
              "radar": radar_time.isoformat(timespec="seconds") if radar_time else None,
              "version": config.VERSION,
              "stage": res["stage"] if on else None, "src": res["src"] if on else None,
              "eta": res["eta"] if on else None, "score": res["score"] if on else None,
              "peak_dbz": res.get("peak_dbz") if on else None,
              "front_km": res.get("front_km") if on else None,
              "speed": round(mu["speed"]) if (on and mu) else None,
              "heading": round(math.degrees(math.atan2(mu["vx"], mu["vy"])) % 360) if (on and mu) else None,
              "nearest15_km": summary.get("nearest15_km"), "max_dbz_10km": summary.get("max_dbz_10km"),
              "alerts": state.get("alerts", []), "trend": read_trend(),
              # 3.1
              "sources": sources or {},
              "intensity_trend": res.get("trend") if on else None,
              "forecast_agree": res.get("forecast_agree") if on else None,
              "boost": res.get("boost") if on else None,
              "radar_down_since": state.get("radar_fail_since"),
              "rain_ended_at": None,
              # 3.2: מערכת שבמעקב (45-90 דקות), רק כשאין התראה פעילה ועומדת בכל התנאים
              "watch": None,
              # 3.2: חיזוי לנקודה {"p", "eta", "dur"} או null
              "nowcast": ({k: res["nowcast"][k] for k in ("p", "eta", "dur")}
                          if (res and res.get("nowcast")) else None)}
    if res and not on and res.get("watch") and watch_ok(res):
        w = res["watch"]
        status["watch"] = {"eta": w["eta"], "peak_dbz": w["peak_dbz"], "front_km": round(w["front_km"], 1)}
    # 3.2: הלמידה (גם גיבוי: אם ה-cache נמחק, היא משוחזרת מכאן)
    lr = state.get("learn")
    status["learning"] = ({k: lr.get(k) for k in ("params", "samples", "events", "false_alarms", "misses")}
                          if lr else None)
    ended, last = parse_ts(state.get("rain_ended_at")), parse_ts(state.get("last_alert"))
    if ended and (last is None or ended > last):
        status["rain_ended_at"] = state["rain_ended_at"]
    atomic_write(os.path.join(config.STATUS_DIR, "status.json"), json.dumps(status, ensure_ascii=False))
    return status


def save_snapshot(path, fields, times, meta=None):
    """3.1: שומר את שדות ה-dBZ (קובץ npz קטן) כדי שאפשר יהיה להריץ את הניתוח שוב על אירוע אמיתי:
    python -m rainbot.replay out/frames_*.npz"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, fields=np.stack(fields).astype(np.float16), times=np.array(times, np.int64),
                        meta=json.dumps(meta or {}, ensure_ascii=False))


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as z:
        fields = [f.astype(np.float32) for f in z["fields"]]
        times = [int(t) for t in z["times"]]
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
    return fields, times, meta
