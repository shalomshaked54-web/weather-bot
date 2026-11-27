"""ריצה אחת של הבוט: מכ"ם + מקורות נוספים -> ניתוח -> החלטה -> טלגרם -> מצב, לוג ודשבורד"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import requests

from . import alerts, forecast, ims, live, messages, storage, telegram
from .analysis import analyze
from .config import (MAX_FRAME_AGE_MIN, OUT_DIR, STATUS_PUBLISH, STORM_HOLD_MIN, STORM_MIN_RUN_GAP_S,
                     STORM_POLL_S, STORM_RADIUS_KM, STORM_TAIL_MIN, SUSPICIOUS_MIN_CELLS, TZ, VERSION)
from .radar import XX, fetch_frames_index, load_fields
from .render import render_png

DELIVERED = ("ok", "maybe")


def get_radar(session, now):
    """מחזיר (שדות, פריימים, None) או (None, None, הודעת שגיאה)"""
    try:
        host, frames = fetch_frames_index(session)
        fields, frames = load_fields(session, host, frames)
        age = (now.timestamp() - frames[-1]["time"]) / 60.0
        if age > MAX_FRAME_AGE_MIN:
            raise RuntimeError(f"הפריים האחרון ישן ({age:.0f} דקות)")
        return fields, frames, None
    except (RuntimeError, KeyError, ValueError, TypeError, IndexError, requests.RequestException) as e:
        return None, None, str(e) or type(e).__name__


def sources_status(radar_ok, age_min, fsum, gauge):
    g = None
    if gauge is not None:
        g = {"ok": bool(gauge.get("ok")), "mm_10min": gauge.get("mm"), "station": gauge.get("station"),
             "t": gauge.get("t")}
    return {"radar": {"ok": radar_ok, "age_min": None if age_min is None else round(age_min, 1)},
            "forecast": fsum, "ims": g}


class Sender:
    """שולח לטלגרם, או במצב dry-run רק מדפיס. מחזיר ok / maybe / fail / dry"""

    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.sent = []

    def text(self, kind, msg):
        if self.dry_run:
            print(f"[dry-run] הודעה ({kind}):\n{msg}")
            return "dry"
        d = telegram.send_telegram(msg)
        self.sent.append(kind if d in DELIVERED else f"{kind}:fail")
        return d


def run(dry_run=False, png=None, save_frames=None):
    if not dry_run:
        missing = [n for n in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not getattr(telegram.config, n)]
        if missing:
            print(f"חסרים משתני סביבה: {', '.join(missing)}")
            return 1

    now = datetime.now(TZ)
    state = storage.load_state()
    with requests.Session() as session:
        fields, frames, radar_err = get_radar(session, now)
        fc = forecast.get_forecast(session, state, now)
        gauge = ims.fetch_gauge(session, now)
    fsum = forecast.summary(fc, now)
    out = Sender(dry_run)
    health = alerts.radar_health(state, radar_err is None, now)

    # ---------------------------------------------------------- אין מכ"ם
    if radar_err:
        print(f"שגיאה בקבלת נתוני מכ\"ם: {radar_err}")
        if health == "down":
            d = out.text("radar_down", messages.radar_down_message(storage.parse_ts(state.get("radar_fail_since")),
                                                                    radar_err))
            alerts.mark_health(state, "down", d in DELIVERED)
        summary = {"v": VERSION, "t": now.isoformat(timespec="seconds"), "radar_err": radar_err,
                   "fc_prob": (fsum or {}).get("prob"), "ims_mm": (gauge or {}).get("mm"), "extra": out.sent}
        print(json.dumps(summary, ensure_ascii=False))
        if not dry_run:
            storage.save_state(state)
            storage.append_history(summary)
            radar_time = storage.parse_ts(state.get("last_radar"))
            storage.write_status(None, {}, state, radar_time, now, sources_status(False, None, fsum, gauge))
        # תקלת מכ"ם היא מצב צפוי שהבוט מטפל בו בעצמו (נרשם בלוג ובדשבורד, והודעה בטלגרם אחרי
        # RADAR_DOWN_ALERT_MIN דקות). לא מסמנים את הריצה כנכשלת, כדי שתקלה רגעית אצל RainViewer
        # לא תשלח מייל "Run failed" מ-GitHub
        return 0

    # ---------------------------------------------------------- ניתוח
    times = [f["time"] for f in frames]
    radar_time = datetime.fromtimestamp(times[-1], TZ)
    radar_iso = radar_time.isoformat(timespec="seconds")
    age_min = (now - radar_time).total_seconds() / 60.0
    res = analyze(fields, times)
    alerts.apply_sources(res, fc, gauge, radar_time)
    raw = fields[-1]

    if health == "up":
        alerts.mark_health(state, "up", out.text("radar_up", messages.radar_up_message(radar_time)) in DELIVERED)

    sent = False
    stage = res["stage"]
    png_bytes = None
    delivery, hold, sent_why = None, None, None
    if stage is None:
        hold = "no_rain"
    elif not alerts.qualifies(res):
        hold = "low_score"
    else:
        go, why = alerts.send_decision(state, stage, now, radar_iso, res.get("peak_dbz"))
        if not go and live.resume(state, now, why):
            go, why = True, "resumed"     # 3.2: הגשם חזר אחרי "כבר לא צפוי"
        msg = messages.build_message(res, radar_time, now, gauge, why)
        if dry_run:
            print(msg)
            print(f"[dry-run] לפי המצב השמור ההתראה {'הייתה נשלחת' if go else 'לא הייתה נשלחת'} ({why})")
            hold = "dry_run"
        elif not go:
            hold = why
        else:
            png_bytes = render_png(raw, res, radar_time)
            delivery = telegram.deliver(png_bytes, msg)
            sent = delivery != "fail"
            sent_why, hold = (why, None) if sent else (None, "send_failed")

    # ---------------------------------------------------------- הודעה חיה (3.2)
    live_act = None
    if not dry_run:
        if sent:
            live.remember(state, res, now, radar_iso, True)
        else:
            live_act = live.update(state, res, radar_time, now, radar_iso, gauge)

    # ---------------------------------------------------------- הודעות נוספות (3.1)
    if not sent and alerts.rain_end_decision(state, res, now):
        alerts.mark_rain_end(state, now, out.text("rain_end", messages.rain_end_message(fsum)) in DELIVERED)
    elif not sent and alerts.watch_decision(state, res, now):
        if out.text("watch", messages.watch_message(res, radar_time, now)) in DELIVERED:
            state["last_watch"] = now.isoformat(timespec="seconds")
    elif not sent and alerts.headsup_decision(state, fsum, now, res):
        if out.text("headsup", messages.headsup_message(fsum)) in DELIVERED:
            state["last_headsup"] = now.isoformat(timespec="seconds")

    # ---------------------------------------------------------- תמונות ונתונים לבדיקה
    suspicious = (not sent) and res["diag"]["raw20_10km"] >= SUSPICIOUS_MIN_CELLS
    meta = {"radar": radar_iso, "stage": stage, "src": res["src"], "sent": sent, "v": VERSION}
    if png:
        with open(png, "wb") as f:
            f.write(render_png(raw, res, radar_time))
    if save_frames:
        storage.save_snapshot(save_frames, fields, times, meta)
    if not dry_run and (sent or suspicious):
        os.makedirs(OUT_DIR, exist_ok=True)
        base = f"{'alert' if sent else 'suspicious'}_{radar_time:%Y%m%d_%H%M}"
        with open(os.path.join(OUT_DIR, base + ".png"), "wb") as f:
            f.write(png_bytes or render_png(raw, res, radar_time))
        storage.save_snapshot(os.path.join(OUT_DIR, base + ".npz"), fields, times, meta)

    m = res["motion"]
    summary = {"v": VERSION, "t": now.isoformat(timespec="seconds"), "radar": radar_iso,
               "rain_cells": res["rain_cells"], "stage": stage, "src": res["src"], "eta": res["eta"],
               "score": res["score"],
               "speed": round(m["speed"], 1) if m else None,
               "heading": round(math.degrees(math.atan2(m["vx"], m["vy"])) % 360) if m else None,
               **res["diag"], "cell": res["cell_dbg"], "sent": sent, "hold": hold, "sent_why": sent_why,
               "trend": res.get("trend"), "boost": res.get("boost"), "fc_agree": res.get("forecast_agree"),
               "fc_prob": (fsum or {}).get("prob"), "ims_mm": (gauge or {}).get("mm"),
               "frames": len(frames),
               # 3.2: חיזוי לנקודה - סיכוי, טווח זמן ההתחלה, ומספר האזורים עם תנועה מקומית
               "nc": ({k: res["nowcast"][k] for k in ("p", "eta", "eta_mid", "dur", "blocks")}
                      if res.get("nowcast") else None),
               "watch": ({"eta": res["watch"]["eta"], "score": res["watch"]["score"],
                          "fc": res["watch"].get("forecast_agree"), "ok": alerts.watch_ok(res)}
                         if res.get("watch") else None)}
    if live_act:
        summary["live"] = live_act
    if out.sent:
        summary["extra"] = out.sent
    if delivery and delivery != "ok":
        summary["delivery"] = delivery
    print(json.dumps(summary, ensure_ascii=False))

    if not dry_run:
        alerts.update_state(state, res, now, radar_time, sent, delivery)
        if storm_weather(res):
            state["storm_seen"] = now.isoformat(timespec="seconds")
        state["last_radar"] = radar_iso
        storage.save_state(state)
        storage.append_history(summary)
        storage.write_status(res, summary, state, radar_time, now,
                             sources_status(True, age_min, fsum, gauge), alerts.qualifies(res))
    return 0


# ------------------------------------------------------------------ מצב סערה (3.2)

def storm_weather(res):
    """יש גשם עד 100 ק"מ, התראה, או מועמד למעקב"""
    near = res["diag"].get("nearest_km")
    return bool(alerts.qualifies(res) or res.get("watch") or (near is not None and near <= STORM_RADIUS_KM))


def storm_active(state, now):
    seen = storage.parse_ts(state.get("storm_seen"))
    return bool(seen and now - seen <= timedelta(minutes=STORM_HOLD_MIN))


def newest_frame_time(session):
    """זמן הפריים האחרון באינדקס של RainViewer (בקשה קטנה), או None בתקלה"""
    try:
        _, frames = fetch_frames_index(session)
        return int(frames[-1]["time"])
    except (RuntimeError, KeyError, ValueError, TypeError, IndexError, requests.RequestException):
        return None


def publish_status():
    if not STATUS_PUBLISH:
        return
    try:
        subprocess.run(["bash", os.path.join(os.path.dirname(__file__), "..", "scripts", "publish_status.sh")],
                       timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"פרסום הדשבורד נכשל: {type(e).__name__}")


def storm_loop(budget_min, run_once=None, clock=time.monotonic, sleep=time.sleep, newest=None, publish=None):
    """ריצה רגילה, ואם יש מצב סערה: ממשיכים לחכות לפריימים חדשים עד budget_min דקות"""
    run_once = run_once or (lambda: run())
    publish = publish or publish_status
    start = clock()
    rc = run_once()
    last_run = clock()
    runs = 1
    if storm_active(storage.load_state(), datetime.now(TZ)):
        publish()          # בסערה הדשבורד מתעדכן אחרי כל פריים, לא רק בסוף הריצה
    with requests.Session() as session:
        newest = newest or (lambda: newest_frame_time(session))
        while True:
            state = storage.load_state()
            if not storm_active(state, datetime.now(TZ)):
                break
            last = storage.parse_ts(state.get("last_radar"))
            last_ts = last.timestamp() if last else 0
            if (clock() - start) / 60.0 > budget_min - STORM_TAIL_MIN:
                break
            sleep(STORM_POLL_S)
            t = newest()
            if t is None or t <= last_ts or clock() - last_run < STORM_MIN_RUN_GAP_S:
                continue
            rc = run_once()
            last_run = clock()
            runs += 1
            publish()
    print(f"מצב סערה: {runs} ניתוחים בריצה הזו")
    return rc


def test_telegram():
    """בדיקת חיבור. 3.2: ReadTimeout ("maybe") = טלגרם קיבל את הבקשה אבל ענה לאט. זו לא תקלה:
    ההודעה כמעט תמיד מגיעה, ובהתראות אמיתיות הבוט מתייחס לזה כנשלח. לכן גם כאן זה לא נכשל"""
    word = {"ok": "נשלחה", "maybe": "כנראה נשלחה (טלגרם ענה לאט)", "fail": "נכשלה"}
    r_msg = telegram.send_telegram(messages.test_message())
    r_img = telegram.send_telegram_photo(render_png(np.zeros(XX.shape, np.float32), {}, None),
                                         "🖼️ בדיקת שליחת תמונה")
    print(f"ההודעה {word.get(r_msg, r_msg)} | התמונה {word.get(r_img, r_img)}")
    return 0 if (r_msg in DELIVERED and r_img in DELIVERED) else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="בלי שליחה ובלי שמירת מצב")
    ap.add_argument("--png", help="שמירת תמונת בקרה (למשל out.png)")
    ap.add_argument("--save-frames", help="שמירת נתוני המכ\"ם (npz) להרצה חוזרת")
    ap.add_argument("--test-telegram", action="store_true", help="שולח הודעת בדיקה ותמונת בדיקה וסיים")
    ap.add_argument("--storm", type=float, default=0, metavar="MIN",
                    help="מצב סערה: כשיש גשם בטווח, להמשיך לנתח כל פריים חדש עד MIN דקות")
    a = ap.parse_args(argv)
    if a.test_telegram:
        sys.exit(test_telegram())
    if a.storm > 0 and not a.dry_run:
        sys.exit(storm_loop(a.storm))
    sys.exit(run(a.dry_run, a.png, a.save_frames))
