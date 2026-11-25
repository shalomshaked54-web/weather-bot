"""נוסחי ההודעות בטלגרם"""
import math
from datetime import datetime, timedelta

from .analysis import compass, dbz_to_mmh, intensity_label
from .config import THRESH_DBZ, VERSION

FOOTER_LINK = "<a href=\"https://www.rainviewer.com\">Weather data by Rain Viewer</a>"
TREND_TEXT = {"up": "📈 <b>מגמה:</b> מתחזק בדרך", "down": "📉 <b>מגמה:</b> נחלש בדרך",
              "steady": "➖ <b>מגמה:</b> יציב"}
TREND_TEXT_NOW = {"up": "📈 <b>מגמה:</b> מתחזק", "down": "📉 <b>מגמה:</b> נחלש",
                  "steady": "➖ <b>מגמה:</b> יציב"}


def arrival_text(now, lo, hi):
    """חלון ההגעה בשעון: "בין 21:00 ל-21:20", או "סביב 21:05" כשהטווח צר"""
    a, b = (now + timedelta(minutes=lo)).strftime("%H:%M"), (now + timedelta(minutes=hi)).strftime("%H:%M")
    if abs(hi - lo) < 5:
        return f"סביב {(now + timedelta(minutes=(lo + hi) / 2)).strftime('%H:%M')}"
    return f"בין {a} ל-{b}"


def _hhmm(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%H:%M")
    except (TypeError, ValueError):
        return None


def build_message(res, radar_time, now, gauge=None):
    src = res.get("src") or "system"
    cell = res.get("cell") or {}
    popup = src == "cell" and cell.get("popup")
    m = res.get("motion_used")
    age_min = max(0.0, (now - radar_time).total_seconds() / 60.0)
    if popup:
        headlines = {"imminent": "⚠️ <b>תא גשם התפתח ממש ליד גבעת שמואל</b>",
                     "now": "☔ <b>תא גשם התפתח מעל גבעת שמואל</b>"}
    elif src == "cell":
        headlines = {"early": "🌧️ <b>תא גשם מתקרב לגבעת שמואל</b>",
                     "imminent": "⚠️ <b>תא גשם מגיע - בעוד דקות ספורות</b>",
                     "now": "☔ <b>תא גשם מעל גבעת שמואל</b>"}
    else:
        headlines = {"early": "🌧️ <b>גשם בדרך לגבעת שמואל</b>",
                     "imminent": "⚠️ <b>הגשם מגיע - בעוד דקות ספורות</b>",
                     "now": "☔ <b>יורד גשם באזור גבעת שמואל</b>"}
    lines = [headlines.get(res["stage"]) or headlines["now"], ""]
    if popup and res["stage"] != "now":
        lines += ["⏱️ <b>זמן הגעה:</b> עלול להתחיל בכל רגע (התא נוצר במקום, לא הגיע מהים)",
                  f"📏 <b>מרחק התא:</b> כ-{res['front_km']:.0f} ק\"מ מהגבעה"]
    elif res["stage"] != "now":
        lo, hi = (max(0, int(round(x - age_min))) for x in res["eta"])
        eta = f"כ-<b>{lo} דקות</b>" if abs(hi - lo) < 5 else f"כ-<b>{lo}-{hi} דקות</b>"
        arrival = arrival_text(now, lo, hi)
        frm = compass((math.degrees(math.atan2(m["vx"], m["vy"])) + 180) % 360)
        label = "מרחק התא" if src == "cell" else "קצה החזית"
        lines += [f"⏱️ <b>זמן הגעה:</b> {eta} ({arrival})",
                  f"🧭 <b>מגיע מכיוון:</b> {frm}, ~{m['speed']:.0f} קמ\"ש",
                  f"📏 <b>{label}:</b> כ-{res['front_km']:.0f} ק\"מ מהגבעה"]
    dbz = res.get("peak_dbz", THRESH_DBZ)
    lines.append(f"💧 <b>עוצמה:</b> {intensity_label(dbz)} (~{dbz_to_mmh(dbz):.1f} מ\"מ/שעה)")
    trend = res.get("trend")
    if trend:
        lines.append((TREND_TEXT_NOW if res["stage"] == "now" else TREND_TEXT)[trend])
    conf = {3: "גבוהה 🟢", 2: "בינונית 🟡"}.get(res["score"], "נמוכה 🔴")
    ok = res.get("checks") or {}
    boost = {"forecast": " · אושר בתחזית המודלים", "ims": " · אושר במד הגשם"}.get(res.get("boost"), "")
    if popup:
        lines.append(f"🎯 <b>ודאות:</b> {conf}  (תא חדש שנצפה ב-{cell.get('n', 2)} פריימים{boost})")
    elif src == "cell":
        lines.append(f"🎯 <b>ודאות:</b> {conf}  (תא שנצפה ב-{cell.get('n', '?')} פריימים רצופים · "
                     f"התראה קצרה{boost})")
    elif ok:
        lines.append(f"🎯 <b>ודאות:</b> {conf}  "
                     f"(תנועה {'✓' if ok['motion'] else '✗'} · גודל {'✓' if ok['size'] else '✗'} · "
                     f"יציבות {'✓' if ok['persist'] else '✗'}{boost})")
    agree = res.get("forecast_agree")
    if agree is True:
        lines.append("📡 <b>תחזית המודלים:</b> מסכימה, צפוי גשם באותו זמן")
    elif agree is False:
        lines.append("📡 <b>תחזית המודלים:</b> לא צופה גשם באותו זמן (המכ\"ם קובע)")
    if gauge and gauge.get("ok") and (gauge.get("mm") or 0) > 0:
        lines.append(f"🌧️ <b>מד הגשם ב{gauge['station']}:</b> {gauge['mm']:.1f} מ\"מ ב-10 הדקות האחרונות")
    lines += ["", f"🛰️ <i>פריים מכ\"ם: {radar_time.strftime('%H:%M')} · {FOOTER_LINK}</i>"]
    return "\n".join(lines)


def radar_down_message(since, err):
    t = since.strftime("%H:%M") if since else "?"
    return ("⚠️ <b>הבוט לא רואה מכ\"ם כרגע</b>\n\n"
            f"מאז {t} אין נתוני מכ\"ם תקינים ({err}). עד שהמכ\"ם יחזור <b>לא יישלחו התראות גשם</b>, "
            "גם אם יורד גשם. כדאי להציץ בשמיים או באתר השירות המטאורולוגי.\n"
            "תגיע הודעה כשהמכ\"ם יחזור.")


def radar_up_message(radar_time):
    return (f"✅ <b>המכ\"ם חזר לפעול</b>\n\nהבוט שוב רואה מכ\"ם (פריים {radar_time.strftime('%H:%M')}) "
            "וממשיך לשמור על גבעת שמואל.")


def headsup_message(fsum):
    start = _hhmm(fsum.get("next_rain"))
    when = f", החל מסביבות {start}" if start else ""
    return ("🌦️ <b>שימו לב: צפוי גשם בגבעת שמואל בשעות הקרובות</b>\n\n"
            f"לפי תחזית המודלים: סיכוי של {fsum['prob']}%, כ-{fsum['mm_3h']:.1f} מ\"מ ב-3 השעות הקרובות{when}.\n"
            "זו עדיין לא התראת מכ\"ם. כשהגשם יתקרב תגיע התראה נפרדת עם זמן הגעה מדויק.\n\n"
            "<i>תחזית: Open-Meteo</i>")


def rain_end_message(fsum):
    lines = ["🌤️ <b>הגשם בגבעת שמואל הפסיק</b>", "",
             "לפי המכ\"ם אין כרגע גשם בטווח 10 ק\"מ, ואין גשם שמתקרב."]
    if fsum and fsum.get("ok") and (fsum.get("prob") or 0) >= 50:
        lines.append(f"📡 תחזית המודלים: ייתכן גשם נוסף בשעות הקרובות ({fsum['prob']}%).")
    lines.append("אם יתקרב גל נוסף תגיע התראה חדשה.")
    return "\n".join(lines)


def test_message():
    return f"✅ בדיקת חיבור: בוט הגשם {VERSION} מחובר לטלגרם."
