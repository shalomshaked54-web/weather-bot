"""3.2: הודעה חיה - עדכון ההתראה האחרונה בטלגרם בכל פריים חדש, במקום הודעות חדשות.

state["live"] = {"id", "kind", "t", "radar", "stage", "miss", "cancelled"}
"""
from datetime import timedelta

from . import messages, telegram
from .alerts import qualifies
from .config import LIVE_CANCEL_FRAMES, LIVE_MAX_MIN, NC_MISS_P
from .storage import parse_ts

APPROACH = ("early", "imminent")


def remember(state, res, now, radar_iso, sent):
    """אחרי התראה שנשלחה: שומרים את ההודעה לעדכון אם היא על גשם שמתקרב"""
    if not sent:
        return
    if res["stage"] in APPROACH and telegram.LAST.get("id"):
        state["live"] = {"id": telegram.LAST["id"], "kind": telegram.LAST.get("kind", "photo"),
                         "t": now.isoformat(timespec="seconds"), "radar": radar_iso, "stage": res["stage"],
                         "miss": 0, "cancelled": False}
    else:
        state.pop("live", None)       # "כעת" או בלי מזהה: אין מה לעדכן


def active(state, now):
    live = state.get("live")
    t = parse_ts((live or {}).get("t"))
    if not live or t is None or now - t > timedelta(minutes=LIVE_MAX_MIN):
        state.pop("live", None)
        return None
    return live


def resume(state, now, why):
    """אחרי "כבר לא צפוי", גשם שחוזר מקבל התראה חדשה עם צלצול (ולא רק עריכה שקטה)"""
    live = active(state, now)
    return bool(live and live.get("cancelled") and why == "cooldown")


def no_rain_coming(res):
    nc = res.get("nowcast")
    return res["stage"] is None and not res.get("watch") and not (nc and nc["p"] >= NC_MISS_P)


def stamp(radar_time, now):
    return f"\n🔄 <i>עודכן {now.strftime('%H:%M')} (מכ\"ם {radar_time.strftime('%H:%M')})</i>"


def update(state, res, radar_time, now, radar_iso, gauge=None, send=None):
    """מעדכן את ההודעה החיה לפי הפריים הנוכחי. מחזיר "edit" / "cancel" / None (לא נעשה כלום)"""
    live = active(state, now)
    if not live or live.get("radar") == radar_iso:
        return None
    send = send or telegram.edit_message
    if qualifies(res) and res["stage"] in APPROACH:
        live["miss"] = 0
        if live.get("cancelled"):
            return None               # תטופל כהתראה חדשה (resume)
        msg = messages.build_message(res, radar_time, now, gauge) + stamp(radar_time, now)
        if send(live["id"], live["kind"], msg) in ("ok", "maybe"):
            live.update(radar=radar_iso, stage=res["stage"])
            return "edit"
        return None
    if no_rain_coming(res) and not live.get("cancelled"):
        live["miss"] = live.get("miss", 0) + 1
        live["radar"] = radar_iso
        if live["miss"] >= LIVE_CANCEL_FRAMES:
            if send(live["id"], live["kind"], messages.live_cancel_message(radar_time, now)) in ("ok", "maybe"):
                live["cancelled"] = True
                return "cancel"
    return None
