"""טלגרם. תוצאות שליחה: "ok" נשלח, "maybe" כנראה נשלח (ReadTimeout: טלגרם קיבל את הבקשה אבל
התשובה לא חזרה בזמן), "fail" לא נשלח. על "maybe" לא שולחים שוב"""
import requests

from . import config


def _post_telegram(method, label, **kw):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/{method}"
    try:
        res = requests.post(url, **kw)
        if res.status_code != 200:
            print(f"שגיאה ב{label}: {res.status_code} {res.text[:200]}")
            return "fail"
        return "ok"
    except requests.exceptions.ReadTimeout:
        print(f"ב{label}: ReadTimeout - ההודעה כנראה נשלחה, לא שולחים שוב")
        return "maybe"
    except requests.RequestException as e:
        print(f"שגיאה ב{label}: {type(e).__name__}")   # ההודעה המלאה כוללת את הטוקן
        return "fail"


def send_telegram(message):
    payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    return _post_telegram("sendMessage", "שליחה לטלגרם", json=payload, timeout=10)


def send_telegram_photo(png_bytes, caption=""):
    """שולח תמונה עם כיתוב. מחזיר "fail" גם אם הכיתוב ארוך מדי, ואז שולחים הודעת טקסט רגילה"""
    if len(caption) > 1000:
        return "fail"
    data = {"chat_id": config.TELEGRAM_CHAT_ID}
    if caption:
        data.update(caption=caption, parse_mode="HTML")
    return _post_telegram("sendPhoto", "שליחת תמונה לטלגרם", data=data,
                          files={"photo": ("radar.png", png_bytes, "image/png")}, timeout=20)


def deliver(png_bytes, msg):
    """תמונה עם כיתוב, ואם נכשלה בוודאות - טקסט. מחזיר ok / maybe / fail"""
    r = send_telegram_photo(png_bytes, msg)
    return r if r != "fail" else send_telegram(msg)

