"""טלגרם. תוצאות שליחה: "ok" נשלח, "maybe" כנראה נשלח (ReadTimeout: טלגרם קיבל את הבקשה אבל
התשובה לא חזרה בזמן), "fail" לא נשלח. על "maybe" לא שולחים שוב"""
import html
import re
import time

import requests

from . import config

MAX_RETRY_AFTER_S = 15


def _post_telegram(method, label, **kw):
    """מחזיר ok / maybe / fail, או bad_html כשטלגרם לא הצליח לפענח את התגיות (3.2).
    3.2: על 429 (יותר מדי הודעות) מחכים את הזמן שטלגרם מבקש, עד 15 שניות, ומנסים פעם אחת נוספת"""
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/{method}"
    for attempt in range(2):
        try:
            res = requests.post(url, **kw)
        except requests.exceptions.ReadTimeout:
            print(f"ב{label}: ReadTimeout - ההודעה כנראה נשלחה, לא שולחים שוב")
            return "maybe"
        except requests.RequestException as e:
            print(f"שגיאה ב{label}: {type(e).__name__}")   # ההודעה המלאה כוללת את הטוקן
            return "fail"
        if res.status_code == 200:
            return "ok"
        text = getattr(res, "text", "") or ""
        print(f"שגיאה ב{label}: {res.status_code} {text[:200]}")
        if res.status_code == 429 and attempt == 0:
            wait = _retry_after(res)
            if wait is not None and wait <= MAX_RETRY_AFTER_S:
                time.sleep(wait)
                continue
        if res.status_code == 400 and "parse entities" in text:
            return "bad_html"
        return "fail"
    return "fail"


def _retry_after(res):
    try:
        return float(res.json()["parameters"]["retry_after"])
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def plain_text(message):
    """HTML של טלגרם -> טקסט רגיל (גיבוי כשהתגיות שבורות)"""
    return html.unescape(re.sub(r"<[^>]+>", "", message))


def send_telegram(message):
    payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    r = _post_telegram("sendMessage", "שליחה לטלגרם", json=payload, timeout=(5, 20))
    if r == "bad_html":   # 3.2: לא מוותרים על ההתראה בגלל תגית שבורה
        payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": plain_text(message),
                   "disable_web_page_preview": True}
        r = _post_telegram("sendMessage", "שליחה לטלגרם (טקסט)", json=payload, timeout=(5, 20))
    return "fail" if r == "bad_html" else r


def send_telegram_photo(png_bytes, caption=""):
    """שולח תמונה עם כיתוב. מחזיר "fail" גם אם הכיתוב ארוך מדי, ואז שולחים הודעת טקסט רגילה"""
    if len(caption) > 1000:
        return "fail"
    data = {"chat_id": config.TELEGRAM_CHAT_ID}
    if caption:
        data.update(caption=caption, parse_mode="HTML")
    r = _post_telegram("sendPhoto", "שליחת תמונה לטלגרם", data=data,
                       files={"photo": ("radar.png", png_bytes, "image/png")}, timeout=20)
    return "fail" if r == "bad_html" else r


def deliver(png_bytes, msg):
    """תמונה עם כיתוב, ואם נכשלה בוודאות - טקסט. מחזיר ok / maybe / fail"""
    r = send_telegram_photo(png_bytes, msg)
    return r if r != "fail" else send_telegram(msg)

