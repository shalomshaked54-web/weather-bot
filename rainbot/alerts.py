"""החלטות: מתי שולחים התראה, השתקה, חיזוק ממקורות נוספים, והודעות מערכת (3.1)"""
from datetime import timedelta

from . import forecast as fcst
from . import ims
from .config import (CELL_MIN_DBZ, COOLDOWN_MIN, EPISODE_GAP_MIN, FORECAST_BOOST_MM, FORECAST_BOOST_PROB,
                     HEADSUP_AFTER_ALERT_H, HEADSUP_GAP_H, HEADSUP_MM, HEADSUP_PROB, NOW_CLEAR_MIN,
                     RADAR_DOWN_ALERT_MIN, RAIN_END_CLEAR_MIN, RAIN_END_MAX_AGE_H, STAGE_RANK)
from .storage import parse_ts


def qualifies(res):
    return res["stage"] is not None and res["score"] >= res["min_score"]


def send_decision(state, stage, now, radar_iso=None):
    """האם לשלוח התראה בשלב stage. מחזיר (שולחים?, סיבה). הסיבה נרשמת בלוג:
    first / new_episode / escalation / new_wave / cooldown_over  -> שולחים
    same_frame / cooldown                                      -> לא שולחים"""
    last = parse_ts(state.get("last_alert"))
    if last is None:
        return True, "first"
    last_stage = state.get("last_stage")
    rank, last_rank = STAGE_RANK[stage], STAGE_RANK.get(last_stage, 0)
    if radar_iso and state.get("last_alert_radar") == radar_iso and rank <= last_rank:
        return False, "same_frame"                    # ריצה כפולה על אותו פריים
    seen = parse_ts(state.get("last_rain_seen"))
    if seen is None or now - seen > timedelta(minutes=EPISODE_GAP_MIN):
        return True, "new_episode"
    if rank > last_rank:
        return True, "escalation"                     # החמרה: early -> imminent -> now
    if last_stage == "now" and stage != "now":
        # הגשם מעל הגבעה נגמר וגל חדש מתקרב: לא מחכים לסוף ההשתקה
        now_seen = parse_ts(state.get("last_now_seen")) or last
        if now - now_seen >= timedelta(minutes=NOW_CLEAR_MIN):
            return True, "new_wave"
    if now - last >= timedelta(minutes=COOLDOWN_MIN):
        return True, "cooldown_over"
    return False, "cooldown"


def update_state(state, res, now, radar_time, sent, delivery=None):
    """מעדכן את המצב אחרי ריצה (בלי לשמור לקובץ)"""
    stage = res["stage"]
    if qualifies(res):
        state["last_rain_seen"] = now.isoformat()    # רק זיהוי שעובר את הסף ממשיך את האירוע
        if stage == "now":
            state["last_now_seen"] = now.isoformat()
    if sent:
        state.update(last_alert=now.isoformat(), last_stage=stage,
                     last_alert_radar=radar_time.isoformat(timespec="seconds"))
        if stage == "now":
            state["last_now_alert"] = now.isoformat()
        alert = {"t": now.isoformat(timespec="seconds"), "stage": stage,
                 "src": res["src"], "eta": res["eta"], "score": res["score"]}
        if res.get("boost"):
            alert["boost"] = res["boost"]
        if delivery and delivery != "ok":
            alert["delivery"] = delivery
        state["alerts"] = (state.get("alerts", []) + [alert])[-15:]
    return state


# ------------------------------------------------------------------ מקורות נוספים

def apply_sources(res, fc, gauge, radar_time):
    """3.1: מצרף לתוצאה את התחזית ומד הגשם, ומחזק ודאות במקרים מוגדרים בלבד:
      - "כעת" שחסרה לו נקודת ודאות אחת, ומד הגשם מודד גשם עכשיו  -> boost="ims"
      - מערכת בציון 2 עם תנועה עקבית (חסר רק גודל או יציבות), שהתחזית צופה לה גשם
        בחלון ההגעה (0.2+ מ"מ ו-60%+)                             -> boost="forecast"
    הציון עצמו נשאר אמיתי, רק הסף יורד (min_score), כדי שההודעה תציג ודאות כנה"""
    res["boost"] = None
    res["forecast_agree"] = None
    res["gauge_wet"] = ims.is_wet(gauge)
    stage = res["stage"]
    if stage is None:
        return res
    if stage != "now" and res.get("eta"):
        lo, hi = res["eta"]
        t0, t1 = radar_time + timedelta(minutes=lo), radar_time + timedelta(minutes=hi)
        res["forecast_agree"] = fcst.rain_between(fc, t0, t1)
    if qualifies(res):
        return res
    if stage == "now" and res["gauge_wet"] and res["score"] >= res["min_score"] - 1:
        res.update(min_score=res["score"], boost="ims")
    elif (stage != "now" and res["src"] == "system" and res["score"] == 2
          and res.get("checks", {}).get("motion") and res.get("eta")):
        lo, hi = res["eta"]
        t0, t1 = radar_time + timedelta(minutes=lo), radar_time + timedelta(minutes=hi)
        if fcst.rain_between(fc, t0, t1, mm=FORECAST_BOOST_MM, prob=FORECAST_BOOST_PROB):
            res.update(min_score=res["score"], boost="forecast")
    return res


# ------------------------------------------------------------------ הודעות מערכת

def radar_health(state, ok, now):
    """3.1: מחזיר "down" (צריך להודיע שהמכ"ם לא זמין), "up" (חזר), או None.
    המצב מסומן רק כשההודעה באמת נשלחה (mark_health)"""
    if ok:
        state.pop("radar_fail_since", None)
        return "up" if state.get("radar_down_notified") else None
    since = parse_ts(state.get("radar_fail_since"))
    if since is None:
        state["radar_fail_since"] = now.isoformat()
        since = now
    if not state.get("radar_down_notified") and now - since >= timedelta(minutes=RADAR_DOWN_ALERT_MIN):
        return "down"
    return None


def mark_health(state, kind, delivered):
    if not delivered:
        return
    if kind == "down":
        state["radar_down_notified"] = True
    elif kind == "up":
        state.pop("radar_down_notified", None)


def headsup_decision(state, fsum, now, res):
    """3.1: "צפוי גשם בשעות הקרובות" לפי התחזית. לא אם כבר יש התראת מכ"ם (עכשיו או ב-3 השעות
    האחרונות), ולכל היותר פעם ב-12 שעות"""
    if not fsum or not fsum.get("ok") or qualifies(res):
        return False
    if (fsum.get("prob") or 0) < HEADSUP_PROB or (fsum.get("mm_3h") or 0) < HEADSUP_MM:
        return False
    last_h = parse_ts(state.get("last_headsup"))
    if last_h and now - last_h < timedelta(hours=HEADSUP_GAP_H):
        return False
    last_a = parse_ts(state.get("last_alert"))
    if last_a and now - last_a < timedelta(hours=HEADSUP_AFTER_ALERT_H):
        return False
    return True


def rain_end_decision(state, res, now):
    """3.1: "הגשם עבר" - פעם אחת אחרי התראת "כעת": אין הד ב-10 ק"מ, שום גשם לא מתקרב,
    ו-25 דקות לא נצפה גשם מעל הגבעה"""
    now_alert = parse_ts(state.get("last_now_alert"))
    now_seen = parse_ts(state.get("last_now_seen"))
    if now_alert is None or now_seen is None or now_seen < now_alert - timedelta(minutes=1):
        return False
    if state.get("rain_end_for") == state.get("last_now_seen"):
        return False
    if not (timedelta(minutes=RAIN_END_CLEAR_MIN) <= now - now_seen <= timedelta(hours=RAIN_END_MAX_AGE_H)):
        return False
    if res["stage"] is not None or res["diag"]["max_dbz_10km"] >= CELL_MIN_DBZ:
        return False
    return True


def mark_rain_end(state, now, delivered):
    if delivered:
        state["rain_end_for"] = state.get("last_now_seen")
        state["rain_ended_at"] = now.isoformat(timespec="seconds")
