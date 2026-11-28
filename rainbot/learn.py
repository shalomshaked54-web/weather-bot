"""3.2: מערכת הלמידה - הבוט משווה כל חיזוי למה שקרה בפועל, ומכייל את עצמו לסבב הבא.

איך זה עובד:
  1. בכל פריים חדש עם חיזוי לגבעה (זמן התחלה), החיזוי נרשם כ"ממתין" (state["learn"]["pending"]).
  2. "אמת": גשם (20+ dBZ) במרחק 2.5 ק"מ מהגבעה בפריים המכ"ם, אחרי 30+ דקות יבשות = התחלת גשם.
     זמן ההתחלה = אמצע הדרך בין הפריים היבש האחרון לפריים הרטוב הראשון.
  3. בהתחלת גשם: כל חיזוי ממתין מ-2.5 השעות האחרונות נבדק: כמה דקות טעה, והאם נפל בטווח שנמסר.
     התראה שלא הגיע אחריה גשם = התראת שווא. גשם שהגיע בלי התראה שעה לפני = פספוס.
  4. מהדגימות מחושבים שני פרמטרים (בנפרד למערכות ולתאים):
       eta_scale  - תיקון לזמני ההגעה (למשל 1.1 = הגשם מגיע בממוצע 10% מאוחר מהחיזוי).
       spread     - רוחב הטווח: גדל אם הגשם נופל מחוץ לטווח יותר מדי, ומצטמצם אם הטווח רחב מדי.
     הלמידה זהירה: חציון (לא ממוצע) כדי שאירוע חריג לא ישבש, "כיווץ" לכיוון 1 כשיש מעט דגימות,
     וגבולות קשיחים (0.7-1.4 ו-0.7-2.0).
  5. אחרי כל אירוע: סיכום בהודעת "הגשם נגמר" (מה נחזה, מה קרה, מה השתנה).

המצב נשמר ב-state.json, ומגובה גם ב-status.json (ענף status), כך שאם ה-cache של GitHub נמחק,
הלמידה משוחזרת משם (restore_from_status).
"""
import json
import statistics
from datetime import timedelta

import numpy as np

from .config import (IMMINENT_MIN, LEARN_DRY_MIN, LEARN_SEED, LEARN_FALSE_AFTER_MIN, LEARN_MATCH_MIN, LEARN_MAX_SAMPLES, LEARN_MIN_SAMPLES,
                     LEARN_PRIOR_N, LEARN_SCALE_RANGE, LEARN_SPREAD_RANGE, LEARN_WINDOW_MIN, NC_HIT_KM, NC_MIN_P,
                     THRESH_DBZ)
from . import verify
from .config import LEARN_ROT_PRIOR_N, LEARN_ROT_RANGE, USER_MATCH_MIN, VERIFY_WEIGHT
from .radar import XX, YY
from .storage import parse_ts

POINT = (XX ** 2 + YY ** 2) <= NC_HIT_KM ** 2


def params(state):
    """הפרמטרים הנוכחיים {"system": {"eta_scale", "spread"}, "cell": {...}}"""
    lp = (state.get("learn") or {}).get("params") or {}
    out = {}
    for src in ("system", "cell"):
        p = lp.get(src) or {}
        seed = LEARN_SEED.get(src, {})
        out[src] = {"eta_scale": float(p.get("eta_scale", seed.get("eta_scale", 1.0))),
                    "spread": float(p.get("spread", seed.get("spread", 1.0))), "n": int(p.get("n", 0)),
                    "rot": float(p.get("rot", 0.0)), "nv": int(p.get("nv", 0))}
    return out


def _learn(state):
    lr = state.setdefault("learn", {})
    for k, v in (("pending", []), ("samples", []), ("events", []), ("params", {}),
                 ("false_alarms", 0), ("misses", 0), ("verify", []), ("frames", []), ("user", [])):
        lr.setdefault(k, v)
    return lr


def point_wet(raw):
    return bool((raw[POINT] >= THRESH_DBZ).any())


# ------------------------------------------------------------------ החלת הכיול

def apply(res, state):
    """מתקן את זמני ההגעה בתוצאה לפי מה שנלמד. הערכים המקוריים נשמרים ב-eta_raw"""
    p = params(state)
    src = res.get("src") or "system"
    scale = p.get(src, p["system"])["eta_scale"]
    res["learn_scale"] = round(scale, 3)
    if abs(scale - 1.0) < 1e-3:
        return res
    if res.get("eta") and res["stage"] != "now" and not (res.get("cell") or {}).get("popup"):
        res["eta_raw"] = res["eta"]
        res["eta"] = tuple(int(round(x * scale)) for x in res["eta"])
        if res.get("eta_mid") is not None:
            res["eta_mid_raw"] = res["eta_mid"]
            res["eta_mid"] = int(round(res["eta_mid"] * scale))
            if src == "system" and res["stage"] in ("early", "imminent"):
                res["stage"] = "imminent" if res["eta_mid"] <= IMMINENT_MIN else "early"
    w = res.get("watch")
    if w and w.get("eta"):
        w["eta"] = tuple(int(round(x * p["system"]["eta_scale"])) for x in w["eta"])
    return res


# ------------------------------------------------------------------ רישום חיזויים

def _prediction(res):
    """החיזוי לגבעה מהתוצאה: (src, דקות חציון, (lo, hi), סיכוי) או None"""
    nc = res.get("nowcast")
    if res["stage"] in ("early", "imminent") and res.get("eta") and res.get("eta_mid") is not None:
        p = nc["p"] if (nc and res.get("src") == "system") else 1.0
        return res.get("src") or "system", res["eta_mid"], tuple(res["eta"]), p, res.get("eta_mid_raw")
    if nc and nc["p"] >= NC_MIN_P and nc.get("eta_mid") is not None:
        return "system", nc["eta_mid"], tuple(nc["eta"]), nc["p"], None
    return None


def record(state, res, radar_time, alerted):
    """רושם חיזוי ממתין (פעם אחת לכל פריים)"""
    lr = _learn(state)
    pr = _prediction(res)
    iso = radar_time.isoformat(timespec="seconds")
    if pr is None or any(x["m"] == iso for x in lr["pending"]):
        return None
    src, mid, (lo, hi), p, raw_mid = pr
    entry = {"m": iso, "src": src, "lead": mid, "raw": raw_mid if raw_mid is not None else mid,
             "lo": lo, "hi": hi, "p": round(float(p), 2), "alert": res["stage"] if alerted else None}
    lr["pending"] = (lr["pending"] + [entry])[-60:]
    return entry


# ------------------------------------------------------------------ אמת: מתי התחיל גשם

def observe(state, raw, radar_time, alert_times=()):
    """מעדכן לפי פריים חדש. מחזיר את אירוע הלמידה אם זוהתה עכשיו התחלת גשם, אחרת None.
    alert_times: זמני ההתראות (מוקדמת/מיידית/כעת) שנשלחו, לבדיקת פספוסים"""
    lr = _learn(state)
    iso = radar_time.isoformat(timespec="seconds")
    if lr.get("last_frame") == iso:
        return None
    wet = point_wet(raw)
    prev_t = parse_ts(lr.get("last_frame"))
    lr["last_frame"] = iso
    event = None
    if wet:
        dry_since = parse_ts(lr.get("dry_since"))
        if lr.get("wet") is False and dry_since and radar_time - dry_since >= timedelta(minutes=LEARN_DRY_MIN):
            onset = radar_time - (radar_time - prev_t) / 2 if prev_t else radar_time
            u = _recent_user(lr, "rain", onset)
            if u is not None and u.get("evaluated"):
                pass                    # כבר נבדק לפי הדיווח שלך: לא סופרים פעמיים
            else:
                if u is not None:       # הדיווח שלך מדויק יותר מהמכ"ם: הוא קובע את זמן ההתחלה
                    onset = parse_ts(u["t"])
                    u["evaluated"] = True
                event = _evaluate(lr, onset, alert_times)
                event["by"] = "user" if u is not None else "radar"
        lr["wet"] = True
        lr.pop("dry_since", None)
    else:
        if lr.get("wet") is not False or not lr.get("dry_since"):
            lr["dry_since"] = iso
        lr["wet"] = False
    _expire(lr, radar_time)
    return event


def _expire(lr, now):
    """חיזויים שעבר זמנם בלי גשם: התראה כזו = התראת שווא"""
    keep = []
    for x in lr["pending"]:
        made = parse_ts(x["m"])
        if made is None:
            continue
        if now - made > timedelta(minutes=x["hi"] + LEARN_FALSE_AFTER_MIN):
            if x.get("alert"):
                lr["false_alarms"] += 1
                lr["fa_recent"] = (lr.get("fa_recent", []) + [x["m"]])[-20:]
            continue
        keep.append(x)
    lr["pending"] = keep


def _evaluate(lr, onset, alert_times):
    samples = []
    matched = []
    for x in lr["pending"]:
        made = parse_ts(x["m"])
        if made is None or not (timedelta(minutes=5) <= onset - made <= timedelta(minutes=LEARN_WINDOW_MIN)):
            continue
        pred = made + timedelta(minutes=x["lead"])
        if abs((onset - pred).total_seconds()) > LEARN_MATCH_MIN * 60:
            continue        # כנראה חיזוי של מערכת אחרת
        obs = (onset - made).total_seconds() / 60.0
        samples.append({"m": x["m"], "src": x["src"], "obs": round(obs, 1), "pred": x["lead"], "raw": x["raw"],
                        "err": round(obs - x["lead"], 1), "inside": x["lo"] - 3 <= obs <= x["hi"] + 3,
                        "alert": x.get("alert")})
        matched.append(x)
    lr["pending"] = [x for x in lr["pending"] if parse_ts(x["m"]) and parse_ts(x["m"]) > onset]
    lr["samples"] = (lr["samples"] + samples)[-LEARN_MAX_SAMPLES:]
    alerted = [t for t in alert_times if t and timedelta(0) <= onset - t <= timedelta(minutes=90)]
    lead_alert = round((onset - min(alerted)).total_seconds() / 60.0) if alerted else None
    if lead_alert is None:
        lr["misses"] += 1
    before = params({"learn": lr})
    _refit(lr)
    after = params({"learn": lr})
    errs = [s["err"] for s in samples if 10 <= s["pred"] <= 60]
    ev = {"onset": onset.isoformat(timespec="minutes"), "n": len(samples),
          "err_med": round(statistics.median(errs), 1) if errs else None,
          "abs_err_med": round(statistics.median([abs(e) for e in errs]), 1) if errs else None,
          "inside": round(float(np.mean([s["inside"] for s in samples])), 2) if samples else None,
          "alert_lead": lead_alert,
          "scale": [round(before["system"]["eta_scale"], 3), round(after["system"]["eta_scale"], 3)],
          "spread": [round(before["system"]["spread"], 3), round(after["system"]["spread"], 3)]}
    lr["events"] = (lr["events"] + [ev])[-30:]
    lr["last_event"] = ev
    return ev


def _refit(lr, update_spread=True):
    """מחשב מחדש את הפרמטרים מהדגימות (חציון משוקלל, כיווץ לכיוון נקודת ההתחלה, גבולות).
    זמני הגעה: דגימות "הגשם התחיל אצלך" (משקל 1) + דגימות אימות מכל פריים (משקל 0.25, רק למערכות).
    כיוון: רק מדגימות האימות. רוחב הטווח: רק כשהגשם התחיל אצלך (update_spread)"""
    for src in ("system", "cell"):
        ss = [s for s in lr["samples"] if s["src"] == src and s["raw"] and s["raw"] >= 10][-60:]
        vs = [v for v in lr.get("verify", []) if v.get("f")][-120:] if src == "system" else []
        seed = LEARN_SEED.get(src, {})
        prior = float(seed.get("eta_scale", 1.0))
        cur = (lr["params"].get(src) or {})
        spread = float(cur.get("spread", seed.get("spread", 1.0)))
        scale = float(cur.get("eta_scale", prior))
        rot = float(cur.get("rot", 0.0))
        vals = [s["obs"] / s["raw"] for s in ss] + [1.0 / v["f"] for v in vs]
        wts = [1.0] * len(ss) + [VERIFY_WEIGHT] * len(vs)
        n_eff = sum(wts)
        if n_eff >= LEARN_MIN_SAMPLES:
            ratio = verify.weighted_median(vals, wts)
            w = n_eff / (n_eff + LEARN_PRIOR_N)
            scale = float(np.clip(prior + (ratio - prior) * w, *LEARN_SCALE_RANGE))
        if len(vs) >= LEARN_MIN_SAMPLES:
            a = verify.weighted_median([v["a"] for v in vs], [v["q"] for v in vs])
            rot = float(np.clip(a * len(vs) / (len(vs) + LEARN_ROT_PRIOR_N), *LEARN_ROT_RANGE))
        if update_spread and len(ss) >= LEARN_MIN_SAMPLES:
            cov = float(np.mean([s["inside"] for s in ss[-40:]]))
            if cov < 0.7:
                spread *= 1.15
            elif cov > 0.92:
                spread *= 0.95
            spread = float(np.clip(spread, *LEARN_SPREAD_RANGE))
        lr["params"][src] = {"eta_scale": round(scale, 4), "spread": round(spread, 4), "n": len(ss),
                             "rot": round(rot, 1), "nv": len(vs)}


# ------------------------------------------------------------------ למידה מכל פריים (verify.py)

def verify_step(state, raw, radar_time, motion):
    """אימות התנועה של לפני 30 דקות מול הפריים הנוכחי. מחזיר את הדגימה (או None)"""
    lr = _learn(state)
    sample = verify.step(lr, raw, radar_time, motion)
    if sample:
        _refit(lr, update_spread=False)
    return sample


# ------------------------------------------------------------------ דיווחים שלך מטלגרם

def _recent_user(lr, kind, t, minutes=USER_MATCH_MIN):
    for u in reversed(lr.get("user", [])):
        ut = parse_ts(u.get("t"))
        if u.get("kind") == kind and ut and abs((t - ut).total_seconds()) <= minutes * 60:
            return u
    return None


def user_report(state, kind, t, alert_times=()):
    """דיווח שלך: "rain" (התחיל לרדת אצלי) או "dry" (לא יורד אצלי). מחזיר תיאור קצר למה שנעשה.
    rain: אם המכ"ם עוד לא ראה גשם אצלך, זו התחלת הגשם לכל דבר (החיזויים נבדקים מול השעה שלך).
          אם המכ"ם כבר ראה, נרשם ההפרש בין המכ"ם למציאות.
    dry:  נספר כהתראת שווא אם הייתה התראה בשעה האחרונה"""
    lr = _learn(state)
    entry = {"t": t.isoformat(timespec="seconds"), "kind": kind}
    lr["user"] = (lr["user"] + [entry])[-100:]
    if kind == "rain":
        if lr.get("wet"):
            ev = lr.get("last_event")
            on = parse_ts(ev["onset"]) if ev else None
            if on and timedelta(0) <= t - on <= timedelta(minutes=90):
                entry["radar_lag"] = round((t - on).total_seconds() / 60.0, 1)
            entry["evaluated"] = True
            return "radar_already"
        ev = _evaluate(lr, t, alert_times)
        ev["by"] = "user"
        entry["evaluated"] = True
        return "onset"
    if kind == "dry":
        recent = [a for a in alert_times if a and timedelta(0) <= t - a <= timedelta(minutes=60)]
        if recent:
            lr["user_false"] = lr.get("user_false", 0) + 1
            return "false_alarm"
        return "noted"
    return "unknown"


# ------------------------------------------------------------------ סיכום וגיבוי

def summary_lines(state):
    """שורות לסיכום אחרי אירוע (בהודעת "הגשם נגמר")"""
    lr = state.get("learn") or {}
    ev = lr.get("last_event")
    if not ev or lr.get("reported") == ev["onset"]:
        return []
    lines = ["", "📊 <b>מה למדנו מהאירוע</b>"]
    if ev.get("alert_lead") is not None:
        lines.append(f"• ההתראה הראשונה יצאה {ev['alert_lead']} דק' לפני שהגשם התחיל אצלך")
    else:
        lines.append("• הגשם התחיל בלי התראה מוקדמת (נרשם כפספוס)")
    if ev.get("abs_err_med") is not None:
        side = "מוקדם" if ev["err_med"] < 0 else "מאוחר"
        lines.append(f"• טעות חציונית בזמן ההגעה: {ev['abs_err_med']:.0f} דק'"
                     + (f" (הגשם הגיע {side} מהחיזוי)" if abs(ev["err_med"]) >= 2 else ""))
    if ev.get("inside") is not None:
        lines.append(f"• {int(round(ev['inside'] * 100))}% מהחיזויים פגעו בטווח שנמסר")
    s0, s1 = ev["scale"]
    if abs(s1 - s0) >= 0.01:
        lines.append(f"• כיול זמני ההגעה עודכן: {s0:.2f} ← {s1:.2f}")
    if lr.get("false_alarms"):
        lines.append(f"• התראות שווא העונה: {lr['false_alarms']} · פספוסים: {lr.get('misses', 0)}")
    return lines


def mark_reported(state):
    lr = state.get("learn") or {}
    if lr.get("last_event"):
        lr["reported"] = lr["last_event"]["onset"]


def public(state):
    """מה שנשמר ב-status.json (גיבוי + דשבורד): הפרמטרים, הסטטיסטיקה והאירועים, בלי הממתינים"""
    lr = state.get("learn") or {}
    return {k: lr.get(k) for k in ("params", "samples", "events", "false_alarms", "misses", "verify", "user",
                                    "user_false")} if lr else None


def restore_from_status(state, path):
    """אם הלמידה חסרה במצב (cache נמחק), משחזרים אותה מ-status.json שפורסם"""
    if (state.get("learn") or {}).get("params"):
        return False
    try:
        with open(path) as f:
            saved = json.load(f).get("learning")
    except (OSError, ValueError, AttributeError):
        return False
    if not isinstance(saved, dict) or not saved.get("params"):
        return False
    lr = _learn(state)
    for k in ("params", "samples", "events", "false_alarms", "misses", "verify", "user", "user_false"):
        if saved.get(k) is not None:
            lr[k] = saved[k]
    return True
