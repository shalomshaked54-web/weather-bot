"""3.2: הדיווחים שלך מטלגרם - "האמת מהשטח" למערכת הלמידה.

המכ"ם רואה גשם בגובה, ולפעמים הגשם מגיע לקרקע כמה דקות אחרי (או לפני). כשאתה כותב לבוט בצ'אט:
  "גשם" / "יורד" / "מטפטף"   -> התחיל לרדת אצלך עכשיו. החיזויים נבדקים מול השעה שלך.
  "יבש" / "לא יורד"           -> לא יורד אצלך. אם הייתה התראה בשעה האחרונה, נספר כהתראת שווא.
הבוט קורא את ההודעות בכל ריצה (getUpdates), לפי השעה שבה שלחת אותן, ועונה בהודעה שקטה.
רק הודעות מהצ'אט של הבוט (TELEGRAM_CHAT_ID) נספרות.
"""
from datetime import datetime, timedelta

import requests

from . import config, learn

RAIN_WORDS = ("גשם", "יורד", "מטפטף", "טפטוף", "rain")
DRY_WORDS = ("יבש", "לא יורד", "אין גשם", "dry")
MAX_AGE_MIN = 120

REPLIES = {
    "onset": "☔ נרשם: התחיל לרדת אצלך ב-{t}. החיזויים ייבדקו מול השעה שלך.",
    "radar_already": "☔ נרשם ({t}). המכ\"ם כבר ראה גשם אצלך, נרשם ההפרש בין המכ\"ם למציאות.",
    "false_alarm": "🌤️ נרשם ({t}): התראת שווא. הבוט ילמד מזה.",
    "noted": "🌤️ נרשם ({t}): יבש אצלך.",
}


def classify(text):
    t = (text or "").strip().lower()
    if not t or len(t) > 40:
        return None
    if any(w in t for w in DRY_WORDS):       # קודם "לא יורד", לפני "יורד"
        return "dry"
    if any(w in t for w in RAIN_WORDS):
        return "rain"
    return None


def _api(method, **payload):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/{method}"
    try:
        r = requests.post(url, json=payload, timeout=(5, 15))
    except requests.RequestException as e:
        print(f"טלגרם {method}: {type(e).__name__}")
        return None
    if r.status_code != 200:
        print(f"טלגרם {method}: {r.status_code} {(getattr(r, 'text', '') or '')[:120]}")
        return None
    try:
        return r.json().get("result")
    except ValueError:
        return None


def parse_updates(updates, chat_id, now):
    """רשימת עדכונים -> (דיווחים [(kind, datetime)], offset הבא)"""
    reports, offset = [], None
    for u in updates or []:
        offset = max(offset or 0, int(u.get("update_id", 0)) + 1)
        m = u.get("message") or u.get("channel_post") or {}
        if str((m.get("chat") or {}).get("id")) != str(chat_id):
            continue
        kind = classify(m.get("text"))
        if not kind or not m.get("date"):
            continue
        t = datetime.fromtimestamp(int(m["date"]), config.TZ)
        if timedelta(0) <= now - t <= timedelta(minutes=MAX_AGE_MIN):
            reports.append((kind, t))
    return reports, offset


def poll(state, now, alert_times=(), fetch=None, reply=None):
    """קורא דיווחים חדשים ומעביר ללמידה. מחזיר רשימת [(kind, תוצאה)]"""
    fetch = fetch or (lambda off: _api("getUpdates", offset=off, timeout=0,
                                       allowed_updates=["message", "channel_post"]))
    reply = reply or (lambda text: _api("sendMessage", chat_id=config.TELEGRAM_CHAT_ID, text=text,
                                        disable_notification=True))
    updates = fetch(state.get("tg_offset"))
    if not isinstance(updates, list):
        return []
    reports, offset = parse_updates(updates, config.TELEGRAM_CHAT_ID, now)
    if offset:
        state["tg_offset"] = offset
    done = []
    for kind, t in sorted(reports, key=lambda r: r[1]):
        what = learn.user_report(state, kind, t, alert_times)
        done.append((kind, what))
        if what in REPLIES:
            reply(REPLIES[what].format(t=t.strftime("%H:%M")))
    return done
