"""3.1: תחזית מודלים מ-Open-Meteo (חינמי, בלי מפתח). "דעה שנייה" למכ"ם.

התחזית לבד אף פעם לא שולחת התראת מכ"ם. היא משמשת ל:
  - חיזוק ודאות של מערכת בציון 2 כשהיא צופה גשם בחלון ההגעה (alerts.apply_sources)
  - הודעת "צפוי גשם בשעות הקרובות" (לכל היותר פעם ב-12 שעות)
  - שורת מידע בהודעה ובדשבורד
"""
from datetime import datetime, timedelta

import requests

from .config import (FORECAST_CACHE_MIN, FORECAST_ENABLED, FORECAST_RAIN_MM, FORECAST_URL, HEADSUP_HOURS,
                     TARGET_LAT, TARGET_LON, TZ)

SLOT_MIN = 15


def _parse_local(s):
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=TZ)


def parse_response(data, now):
    """תשובת Open-Meteo -> {"ok", "t", "slots": [[iso, מ"מ]], "hourly": [[iso, סיכוי, מ"מ]]}"""
    m, h = data["minutely_15"], data["hourly"]
    slots = [[t, float(v)] for t, v in zip(m["time"], m["precipitation"]) if v is not None]
    hourly = [[t, int(p) if p is not None else None, float(v) if v is not None else 0.0]
              for t, p, v in zip(h["time"], h["precipitation_probability"], h["precipitation"])]
    if not slots and not hourly:
        raise ValueError("תחזית ריקה")
    return {"ok": True, "t": now.isoformat(timespec="seconds"), "slots": slots, "hourly": hourly}


def fetch_forecast(session, now):
    params = {"latitude": TARGET_LAT, "longitude": TARGET_LON,
              "minutely_15": "precipitation", "forecast_minutely_15": 4 * 6,
              "hourly": "precipitation_probability,precipitation", "forecast_hours": 6,
              "timezone": "Asia/Jerusalem"}
    try:
        res = session.get(FORECAST_URL, params=params, timeout=(5, 8))
        if res.status_code != 200:
            return {"ok": False, "err": f"HTTP {res.status_code}"}
        return parse_response(res.json(), now)
    except requests.RequestException as e:
        return {"ok": False, "err": type(e).__name__}
    except (KeyError, TypeError, ValueError) as e:
        return {"ok": False, "err": f"bad response: {type(e).__name__}"}


def get_forecast(session, state, now):
    """תחזית עם מטמון של 15 דקות בתוך קובץ המצב (כדי לא להעמיס על השירות)"""
    if not FORECAST_ENABLED:
        return None
    cached = state.get("forecast")
    if cached and cached.get("ok"):
        try:
            if now - datetime.fromisoformat(cached["t"]) < timedelta(minutes=FORECAST_CACHE_MIN):
                return cached
        except (KeyError, TypeError, ValueError):
            pass
    fc = fetch_forecast(session, now)
    if fc.get("ok"):
        state["forecast"] = fc
    elif cached and cached.get("ok"):
        try:   # תחזית של עד שעתיים עדיפה על כלום
            if now - datetime.fromisoformat(cached["t"]) < timedelta(hours=2):
                return {**cached, "stale": True}
        except (KeyError, TypeError, ValueError):
            pass
    return fc


def rain_between(fc, t0, t1, mm=FORECAST_RAIN_MM, prob=None):
    """האם התחזית צופה גשם בין t0 ל-t1 (עם מרווח של משבצת אחת לכל צד).
    prob: אם נתון, נדרש גם סיכוי שעתי של לפחות prob באחת השעות של החלון"""
    if not fc or not fc.get("ok"):
        return None
    lo, hi = t0 - timedelta(minutes=SLOT_MIN), t1 + timedelta(minutes=2 * SLOT_MIN)
    wet = any(lo <= _parse_local(t) <= hi and v >= mm for t, v in fc.get("slots", []))
    if not wet or prob is None:
        return wet
    hours = [p for t, p, _ in fc.get("hourly", [])
             if p is not None and lo - timedelta(hours=1) <= _parse_local(t) <= hi]
    return bool(hours) and max(hours) >= prob


def summary(fc, now, hours=HEADSUP_HOURS):
    """סיכום לשעות הקרובות: {"ok", "prob", "mm_3h", "next_rain"} (או None אם אין תחזית)"""
    if fc is None:
        return None
    if not fc.get("ok"):
        return {"ok": False, "prob": None, "mm_3h": None, "next_rain": None}
    end = now + timedelta(hours=hours)
    slots = [(_parse_local(t), v) for t, v in fc.get("slots", [])]
    future = [(t, v) for t, v in slots if now < t <= end + timedelta(minutes=SLOT_MIN)]
    mm = round(sum(v for _, v in future), 1) if future else None
    nxt = next((t - timedelta(minutes=SLOT_MIN) for t, v in slots
                if t > now and v >= FORECAST_RAIN_MM), None)
    probs = [p for t, p, _ in fc.get("hourly", [])
             if p is not None and now - timedelta(hours=1) < _parse_local(t) <= end]
    if mm is None:   # בלי משבצות של 15 דקות: לפי הכמות השעתית
        hv = [v for t, _, v in fc.get("hourly", []) if now - timedelta(hours=1) < _parse_local(t) <= end]
        mm = round(sum(hv), 1) if hv else None
    return {"ok": True, "prob": max(probs) if probs else None, "mm_3h": mm,
            "next_rain": max(nxt, now).isoformat(timespec="minutes") if nxt else None}
