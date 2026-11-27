"""נוסחי ההודעות בטלגרם (3.2: נוסח חדש, קצר וברור).

מבנה קבוע לכל התראה:
  שורה 1  כותרת: מה קורה (אימוג'י + משפט קצר)
  שורה 2  מתי: שעה בשעון, ובסוגריים בעוד כמה דקות
  שורה 3  כמה חזק ולאן זה הולך (מתחזק / נחלש)
  שורה 4  מאיפה ובאיזו מהירות
  שורה 5  ודאות, ואישור ממקור נוסף אם יש
  שורה 6  מה כדאי לעשות (רק כשזה רלוונטי)
"""
import math
from datetime import datetime, timedelta

from .analysis import compass, dbz_to_mmh, intensity_label
from .config import NC_MISS_P, THRESH_DBZ, VERSION

FOOTER_LINK = "<a href=\"https://www.rainviewer.com\">Weather data by Rain Viewer</a>"
TREND_WORD = {"up": "ומתחזק 📈", "down": "ונחלש 📉", "steady": "ויציב"}
CONF_WORD = {3: "ודאות גבוהה 🟢", 2: "ודאות בינונית 🟡"}
BOOST_WORD = {"forecast": "אושר בתחזית", "ims": "אושר במד הגשם"}
TIP = {"early": "💡 יש זמן להתארגן: כביסה, חלונות, מטרייה.",
       "imminent": "☂️ זה הזמן להכניס כביסה ולקחת מטרייה."}


def arrival_text(now, lo, hi):
    """חלון ההגעה בשעון: "בין 21:00 ל-21:20", או "סביב 21:05" כשהטווח צר"""
    a, b = (now + timedelta(minutes=lo)).strftime("%H:%M"), (now + timedelta(minutes=hi)).strftime("%H:%M")
    if abs(hi - lo) < 5:
        return f"סביב {(now + timedelta(minutes=(lo + hi) / 2)).strftime('%H:%M')}"
    return f"בין {a} ל-{b}"


def minutes_text(lo, hi):
    if hi <= 1:
        return "ממש עכשיו"
    if abs(hi - lo) < 5:
        return f"בעוד כ-{max(1, round((lo + hi) / 2))} דק'"
    return f"בעוד {lo}-{hi} דק'"


def _hhmm(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%H:%M")
    except (TypeError, ValueError):
        return None


def _footer(radar_time):
    return f"<i>🛰️ מכ\"ם {radar_time.strftime('%H:%M')} · {FOOTER_LINK}</i>"


def intensity_line(dbz, trend=None):
    t = f" {TREND_WORD[trend]}" if trend in TREND_WORD else ""
    return f"💧 {intensity_label(dbz)}{t} (~{dbz_to_mmh(dbz):.0f} מ\"מ/שעה)" if dbz_to_mmh(dbz) >= 1.5 \
        else f"💧 {intensity_label(dbz)}{t}"


def direction_line(m, dist_km, cell=False):
    frm = compass((math.degrees(math.atan2(m["vx"], m["vy"])) + 180) % 360)
    what = "התא" if cell else "הגשם"
    return f"🧭 {what} מגיע מ{frm}, ~{m['speed']:.0f} קמ\"ש · {dist_km:.0f} ק\"מ מכאן"


def confidence_line(score, boost=None, agree=None, note=None):
    parts = [CONF_WORD.get(score, "ודאות נמוכה 🔴")]
    if boost in BOOST_WORD:
        parts.append(BOOST_WORD[boost])
    elif agree is True:
        parts.append("התחזית מסכימה")
    elif agree is False:
        parts.append("התחזית לא צופה גשם, המכ\"ם קובע")
    if note:
        parts.append(note)
    return "🎯 " + " · ".join(parts)


def gauge_line(gauge):
    if gauge and gauge.get("ok") and (gauge.get("mm") or 0) > 0:
        return f"🌧️ מד הגשם ב{gauge['station']} מודד גשם: {gauge['mm']:.1f} מ\"מ ב-10 הדקות האחרונות"
    return None


def nowcast_line(nc):
    """3.2: שורת החיזוי לנקודה: הסיכוי שיירד גשם אצלך, וכמה זמן הוא צפוי להימשך"""
    if not nc:
        return None
    p = int(round(nc["p"] * 100))
    if nc["p"] < NC_MISS_P:
        return f"📍 ייתכן שהגשם יחלוף לידך (סיכוי {p}% שיגיע ממש אלייך)"
    line = f"📍 סיכוי {p}% שיירד ממש אצלך"
    if nc.get("dur") and nc["dur"] >= 5:
        dur = nc["dur"]
        how_long = "יותר משעה" if dur >= 60 else f"כ-{max(5, int(round(dur / 5) * 5))} דק'"
        line += f" · צפוי להימשך {how_long}"
    return line


def build_message(res, radar_time, now, gauge=None, why=None):
    """התראת גשם (מוקדמת / מיידית / כעת, ממערכת או מתא). why="intensify" = הגשם הפך לחזק"""
    src = res.get("src") or "system"
    cell = res.get("cell") or {}
    is_cell = src == "cell"
    popup = is_cell and cell.get("popup")
    stage = res["stage"]
    m = res.get("motion_used")
    dbz = res.get("peak_dbz", THRESH_DBZ)
    age_min = max(0.0, (now - radar_time).total_seconds() / 60.0)

    if why == "intensify":
        head = "⛈️ <b>הגשם מתחזק: גשם חזק עכשיו בגבעת שמואל</b>"
    elif popup:
        head = ("☔ <b>תא גשם נוצר עכשיו מעל גבעת שמואל</b>" if stage == "now"
                else "⚡ <b>תא גשם נוצר ממש ליד גבעת שמואל</b>")
    elif stage == "now":
        head = "☔ <b>יורד גשם בגבעת שמואל</b>" if not is_cell else "☔ <b>תא גשם מעל גבעת שמואל</b>"
    elif stage == "imminent":
        head = "⚠️ <b>גשם בעוד כמה דקות</b>" if not is_cell else "⚠️ <b>תא גשם מגיע בעוד כמה דקות</b>"
    else:
        head = "🌧️ <b>גשם בדרך לגבעת שמואל</b>" if not is_cell else "🌦️ <b>תא גשם מתקרב לגבעת שמואל</b>"
    lines = [head, ""]

    if popup and stage != "now":
        lines.append(f"⏱️ <b>עלול להתחיל בכל רגע</b> · התא נמצא {res['front_km']:.0f} ק\"מ מכאן")
    elif stage != "now":
        lo, hi = (max(0, int(round(x - age_min))) for x in res["eta"])
        lines.append(f"⏱️ מגיע <b>{arrival_text(now, lo, hi)}</b> ({minutes_text(lo, hi)})")
    lines.append(intensity_line(dbz, res.get("trend")))
    if stage != "now" and not popup and m:
        lines.append(direction_line(m, res["front_km"], is_cell))
    if stage != "now" and not is_cell:
        ncl = nowcast_line(res.get("nowcast"))
        if ncl:
            lines.append(ncl)
    g = gauge_line(gauge)
    if g:
        lines.append(g)
    note = None
    if popup:
        note = f"תא חדש, נצפה ב-{cell.get('n', 2)} פריימים"
    elif is_cell:
        note = "תא מקומי, הגשם יכול להיות קצר"
    agree = res.get("forecast_agree") if stage != "now" else None
    lines.append(confidence_line(res["score"], res.get("boost"), agree, note))
    tip = TIP.get(stage) if why != "intensify" else None
    if tip:
        lines.append(tip)
    if why == "intensify":
        lines.append("⚡ ייתכנו הצפות מקומיות. כדאי להישאר בפנים עד שיחלש.")
    lines += ["", _footer(radar_time)]
    return "\n".join(lines)


def watch_message(res, radar_time, now):
    """3.2: "מעקב" - מערכת גשם בים, כשעה לפני שהיא אמורה להגיע"""
    w = res["watch"]
    age_min = max(0.0, (now - radar_time).total_seconds() / 60.0)
    lo, hi = (max(0, int(round(x - age_min))) for x in w["eta"])
    lines = ["🔭 <b>מעקב: מערכת גשם בדרך לגבעת שמואל</b>", "",
             f"⏱️ אם תמשיך כך, תגיע <b>{arrival_text(now, lo, hi)}</b> ({minutes_text(lo, hi)})",
             intensity_line(w["peak_dbz"], w.get("trend"))]
    if res.get("motion"):
        lines.append(direction_line(res["motion"], w["front_km"]))
    ncl = nowcast_line(res.get("nowcast"))
    if ncl:
        lines.append(ncl)
    lines += ["📡 תחזית המודלים מסכימה שיירד גשם באותו זמן",
              "ℹ️ זו התראה מוקדמת. כשהגשם יתקרב תגיע התראה עם זמן מדויק.", "", _footer(radar_time)]
    return "\n".join(lines)


def radar_down_message(since, err):
    t = since.strftime("%H:%M") if since else "?"
    return ("🚫 <b>הבוט לא רואה מכ\"ם כרגע</b>\n\n"
            f"מאז {t} אין נתוני מכ\"ם תקינים ({err}).\n"
            "⚠️ עד שהמכ\"ם יחזור <b>לא יישלחו התראות גשם</b>, גם אם יורד גשם.\n"
            "💡 בינתיים כדאי להציץ בשמיים או באתר השירות המטאורולוגי.\n"
            "תגיע הודעה כשהמכ\"ם יחזור.")


def radar_up_message(radar_time):
    return (f"✅ <b>המכ\"ם חזר</b>\n\nהבוט שוב רואה מכ\"ם (פריים {radar_time.strftime('%H:%M')}) "
            "וחוזר לשמור על גבעת שמואל.")


def headsup_message(fsum):
    start = _hhmm(fsum.get("next_rain"))
    lines = ["🌦️ <b>צפוי גשם בשעות הקרובות</b>", ""]
    if start:
        lines.append(f"⏱️ לפי התחזית, החל מסביבות <b>{start}</b>")
    lines += [f"📊 סיכוי {fsum['prob']}% · כ-{fsum['mm_3h']:.1f} מ\"מ ב-3 השעות הקרובות",
              "ℹ️ זו תחזית, לא מכ\"ם. כשהגשם יתקרב תגיע התראה עם זמן מדויק.", "",
              "<i>📡 תחזית: Open-Meteo</i>"]
    return "\n".join(lines)


def rain_end_message(fsum):
    lines = ["🌤️ <b>הגשם בגבעת שמואל נגמר</b>", "",
             "✅ אין גשם בטווח 10 ק\"מ, ואין גשם שמתקרב."]
    if fsum and fsum.get("ok") and (fsum.get("prob") or 0) >= 50:
        lines.append(f"📡 שימו לב: התחזית צופה עוד גשם בשעות הקרובות ({fsum['prob']}%).")
    lines.append("אם יגיע גל נוסף, תגיע התראה חדשה.")
    return "\n".join(lines)


def test_message():
    return f"✅ <b>בדיקת חיבור</b>\n\nבוט הגשם {VERSION} מחובר לטלגרם ושומר על גבעת שמואל."
