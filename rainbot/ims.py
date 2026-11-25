"""3.1: מד הגשם של השירות המטאורולוגי (ברירת מחדל: בית דגן, כ-6 ק"מ מהגבעה).

פעיל רק אם הוגדר הסוד IMS_TOKEN (מפתח חינמי שמבקשים מהשירות המטאורולוגי).
בלי מפתח הבוט עובד בדיוק כמו קודם. מד הגשם הוא "אמת קרקע": אם הוא מודד גשם עכשיו,
התראת "כעת" בוודאות גבולית מקבלת אישור.
"""
from datetime import datetime, timedelta

import requests

from .config import IMS_MAX_AGE_MIN, IMS_STATION, IMS_STATION_NAMES, IMS_TOKEN, IMS_URL


def parse_latest(data, now, station=IMS_STATION):
    """תשובת /data/latest -> {"ok", "mm", "t", "station"}"""
    rec = data["data"][-1]
    t = datetime.fromisoformat(rec["datetime"])
    rain = next((c for c in rec["channels"] if c.get("name") == "Rain"), None)
    out = {"ok": False, "mm": None, "t": t.isoformat(timespec="minutes"),
           "station": IMS_STATION_NAMES.get(str(station), str(station))}
    if rain is None or not rain.get("valid", True) or rain.get("value") is None:
        out["err"] = "no rain channel"
        return out
    if now - t > timedelta(minutes=IMS_MAX_AGE_MIN):
        out["err"] = "stale"
        return out
    out.update(ok=True, mm=round(float(rain["value"]), 2))
    return out


def fetch_gauge(session, now, token=IMS_TOKEN, station=IMS_STATION):
    if not token:
        return None
    url = IMS_URL.format(station=station)
    try:
        res = session.get(url, headers={"Authorization": f"ApiToken {token}"}, timeout=(5, 8))
        if res.status_code != 200:
            return {"ok": False, "err": f"HTTP {res.status_code}", "mm": None}
        return parse_latest(res.json(), now, station)
    except requests.RequestException as e:
        return {"ok": False, "err": type(e).__name__, "mm": None}   # בלי ההודעה המלאה: אולי כוללת את הכתובת
    except (KeyError, IndexError, TypeError, ValueError) as e:
        return {"ok": False, "err": f"bad response: {type(e).__name__}", "mm": None}


def is_wet(gauge):
    return bool(gauge and gauge.get("ok") and (gauge.get("mm") or 0) > 0)
