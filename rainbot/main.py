"""ריצה אחת של הבוט: מכ"ם + מקורות נוספים -> ניתוח -> החלטה -> טלגרם -> מצב, לוג ודשבורד"""
import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import requests

from . import alerts, forecast, ims, messages, storage, telegram
from .analysis import analyze
from .config import MAX_FRAME_AGE_MIN, OUT_DIR, SUSPICIOUS_MIN_CELLS, TZ, VERSION
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
               "watch": ({"eta": res["watch"]["eta"], "score": res["watch"]["score"],
                          "fc": res["watch"].get("forecast_agree"), "ok": alerts.watch_ok(res)}
                         if res.get("watch") else None)}
    if out.sent:
        summary["extra"] = out.sent
    if delivery and delivery != "ok":
        summary["delivery"] = delivery
    print(json.dumps(summary, ensure_ascii=False))

    if not dry_run:
        alerts.update_state(state, res, now, radar_time, sent, delivery)
        state["last_radar"] = radar_iso
        storage.save_state(state)
        storage.append_history(summary)
        storage.write_status(res, summary, state, radar_time, now,
                             sources_status(True, age_min, fsum, gauge), alerts.qualifies(res))
    return 0


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
    a = ap.parse_args(argv)
    if a.test_telegram:
        sys.exit(test_telegram())
    sys.exit(run(a.dry_run, a.png, a.save_frames))
