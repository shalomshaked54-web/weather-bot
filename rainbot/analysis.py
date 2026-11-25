"""ניתוח המכ"ם: ניקוי, תנועה, מסלול המערכות, מסלול התאים ומגמת עוצמה"""
import math

import numpy as np

from .config import (BASE_RADIUS_KM, CELL_DECAY_DBZ, CELL_DECAY_MAX_PEAK, CELL_HIT_BASE_KM,
                     CELL_HIT_SPREAD, CELL_KM, CELL_MAX_CELLS, CELL_MAX_ETA_MIN, CELL_MAX_SPEED_KMH,
                     CELL_MIN_APPROACH_KMH, CELL_MIN_CELLS, CELL_MIN_DBZ, CELL_NOW_STRONG_DBZ,
                     CELL_PEAK_DBZ, CELL_SANITY_ANGLE_DEG, CELL_SANITY_SPEED_RATIO, CELL_ZONE_KM,
                     EARLY_MIN, IMMINENT_MIN, MAX_LEAD_MIN, MAX_SPEED_KMH, MIN_ALERT_AREA_KM2,
                     MIN_CELLS, MIN_CORR, MIN_SCORE_CELL, MIN_SCORE_CELL_NOW_STRONG, MIN_SCORE_SYSTEM,
                     MIN_SPEED_KMH, NOW_RADIUS_KM, PERSIST_MIN_CELLS, POPUP_CLEAR_DBZ, POPUP_EDGE_KM,
                     POPUP_MAX_SPEED_KMH, POPUP_PEAK_DBZ, SPEED_UNCERTAINTY, SPREAD_PER_KM, STAGE_RANK,
                     STEP_MIN, SUSPICIOUS_MIN_DBZ, SUSPICIOUS_RADIUS_KM, THRESH_DBZ, TREND_DBZ)
from .radar import XX, YY


# ------------------------------------------------------------------ עיבוד שדה

def label_clusters(mask):
    ny, nx = mask.shape
    labels = np.zeros((ny, nx), np.int32)
    sizes = [0]
    cur = 0
    for j, i in zip(*np.nonzero(mask)):
        if labels[j, i]:
            continue
        cur += 1
        labels[j, i] = cur
        stack, n = [(j, i)], 0
        while stack:
            y, x = stack.pop()
            n += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx] and not labels[yy, xx]:
                        labels[yy, xx] = cur
                        stack.append((yy, xx))
        sizes.append(n)
    return labels, np.array(sizes)


def clean_field(field):
    """משאיר רק אשכולות גשם בגודל מינימלי. מחזיר (שדה נקי, תוויות, גדלים)"""
    labels, sizes = label_clusters(field >= THRESH_DBZ)
    keep = sizes >= MIN_CELLS
    keep[0] = False
    return np.where(keep[labels], field, 0.0), labels, sizes


def motion_weights(cleaned):
    return np.where(cleaned > 0, np.clip(cleaned - 15.0, 5.0, 25.0), 0.0)


# ------------------------------------------------------------------ תנועה

def xcorr_shift(a, b, max_shift):
    """באיזה היסט (dy, dx בתאים) b הוא a מוזז. מחזיר (dy, dx, איכות 0-1)"""
    ny, nx = a.shape
    sy, sx = 2 * ny, 2 * nx
    c = np.fft.irfft2(np.fft.rfft2(b, (sy, sx)) * np.conj(np.fft.rfft2(a, (sy, sx))), (sy, sx))
    r = np.arange(-max_shift, max_shift + 1)
    sub = c[np.ix_(r % sy, r % sx)]
    j, i = np.unravel_index(np.argmax(sub), sub.shape)

    def refine(l, m, rr):
        den = l - 2 * m + rr
        return 0.0 if den >= -1e-9 else 0.5 * (l - rr) / den

    dy, dx = float(j - max_shift), float(i - max_shift)
    if 0 < j < sub.shape[0] - 1:
        dy += refine(sub[j - 1, i], sub[j, i], sub[j + 1, i])
    if 0 < i < sub.shape[1] - 1:
        dx += refine(sub[j, i - 1], sub[j, i], sub[j, i + 1])
    quality = float(sub[j, i] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return dy, dx, quality


def estimate_motion(weights, times):
    """וקטור תנועה גלובלי (קמ"ש מזרחה/צפונה) מכל זוגות (פריים ישן, פריים אחרון)"""
    est = []
    for k in range(len(weights) - 1):
        dt_h = (times[-1] - times[k]) / 3600.0
        if dt_h <= 0 or not weights[k].any() or not weights[-1].any():
            continue
        m = int(min(MAX_SPEED_KMH * dt_h / CELL_KM + 2, 40))
        dy, dx, q = xcorr_shift(weights[k], weights[-1], m)
        if q >= MIN_CORR:
            est.append((dx * CELL_KM / dt_h, dy * CELL_KM / dt_h, q, dt_h))
    if not est:
        return None
    w = np.array([e[2] * e[3] for e in est])
    vx = float(np.average([e[0] for e in est], weights=w))
    vy = float(np.average([e[1] for e in est], weights=w))
    speed = math.hypot(vx, vy)
    consistent = speed >= MIN_SPEED_KMH
    if consistent:
        for ex, ey, _, _ in est:
            s = math.hypot(ex, ey)
            ang = abs((math.degrees(math.atan2(ex, ey) - math.atan2(vx, vy)) + 180) % 360 - 180)
            if abs(s - speed) > max(10.0, 0.35 * speed) or (s >= 10.0 and ang > 35.0):
                consistent = False
    return {"vx": vx, "vy": vy, "speed": speed, "quality": float(np.mean([e[2] for e in est])),
            "consistent": consistent, "n": len(est)}


# ------------------------------------------------------------------ חיזוי

def disc(cx, cy, r):
    return (XX - cx) ** 2 + (YY - cy) ** 2 <= r * r


def predict_eta(cleaned, motion, speed_factor=1.0):
    """הרצה אחורה (upwind): מחזיר (דקות, מרחק ק"מ) לראשון שבו נמצא גשם, או None"""
    speed = motion["speed"] * speed_factor
    ux, uy = motion["vx"] / motion["speed"], motion["vy"] / motion["speed"]
    for tau in range(STEP_MIN, MAX_LEAD_MIN + 1, STEP_MIN):
        d = speed * tau / 60.0
        if (cleaned[disc(-ux * d, -uy * d, BASE_RADIUS_KM + SPREAD_PER_KM * d)] > 0).any():
            return tau, d
    return None


def compass(bearing):
    names = ["צפון", "צפון-מזרח", "מזרח", "דרום-מזרח", "דרום", "דרום-מערב", "מערב", "צפון-מערב"]
    return names[int(((bearing + 22.5) % 360) // 45)]


def dbz_to_mmh(dbz):
    return (10 ** (dbz / 10.0) / 200.0) ** (1 / 1.6)   # Marshall-Palmer, בקירוב


def intensity_label(dbz):
    if dbz >= 46:
        return "⛈️ עז מאוד (ייתכנו רעמים/ברד)"
    if dbz >= 38:
        return "🌧️ חזק"
    if dbz >= 28:
        return "🌦️ בינוני"
    return "🌦️ קל"


# ------------------------------------------------------------------ מגמת עוצמה (3.1)

def classify_trend(peaks):
    """רשימת שיאים (ישן->חדש) -> "up" / "down" / "steady", או None אם אין מספיק נתונים"""
    vals = [p for p in peaks if p is not None]
    if len(vals) < 2 or len(peaks) < 2 or peaks[-1] is None:
        return None
    ref = float(np.mean([p for p in peaks[:-1] if p is not None] or [0.0]))
    d = peaks[-1] - ref
    return "up" if d >= TREND_DBZ else ("down" if d <= -TREND_DBZ else "steady")


def _peak_in(field, mask):
    return float(field[mask].max()) if mask.any() else 0.0


def system_peaks(fields, times, motion, tau0):
    """השיא של הגשם שצפוי לפגוע בגבעה, בכל פריים, לאורך מסלול התנועה (לגרנז'יאני):
    הגשם שיגיע בעוד tau דקות היה לפני dt רחוק יותר ב-speed*dt"""
    ux, uy = motion["vx"] / motion["speed"], motion["vy"] / motion["speed"]
    peaks = []
    for k, f in enumerate(fields):
        back_h = (times[-1] - times[k]) / 3600.0
        pk = 0.0
        for tau in range(tau0, tau0 + 31, STEP_MIN):
            d = motion["speed"] * (tau / 60.0 + back_h)
            pk = max(pk, _peak_in(f, disc(-ux * d, -uy * d, BASE_RADIUS_KM + SPREAD_PER_KM * d)))
        peaks.append(pk)
    return peaks


def analyze_system(fields, times):
    """מסלול המערכות: שדות dBZ גולמיים (ישן->חדש) וזמנים -> תוצאה מובנית"""
    cleaned, labels, sizes = [], None, None
    for f in fields:
        c, lab, sz = clean_field(f)
        cleaned.append(c)
        labels, sizes = lab, sz   # תוויות הפריים האחרון
    latest = cleaned[-1]

    res = {"rain_cells": int((latest > 0).sum()), "stage": None, "motion": None,
           "eta": None, "score": 0, "checks": {}}

    dist = np.hypot(XX, YY)
    raw = fields[-1]

    def nearest(field, thr):
        m = field >= thr
        return round(float(dist[m].min()), 1) if m.any() else None

    # נתוני אבחון שנשמרים בלוג (איפה בדיוק הגשם ומה העוצמה)
    res["diag"] = {"nearest_km": nearest(latest, THRESH_DBZ),
                   "nearest15_km": nearest(raw, CELL_MIN_DBZ),
                   "max_dbz_10km": round(float(raw[dist <= 10].max()), 1),
                   "cells_20km": int(((latest > 0) & (dist <= 20)).sum()),
                   "raw20_10km": int(((raw >= SUSPICIOUS_MIN_DBZ) & (dist <= SUSPICIOUS_RADIUS_KM)).sum())}

    now_mask = disc(0, 0, NOW_RADIUS_KM)
    if (latest[now_mask] > 0).any():
        res["stage"] = "now"
        res["peak_dbz"] = float(latest[now_mask].max())
        res["eta"] = (0, 0)
        res["front_km"] = 0.0
        res["cluster_km2"] = float(sizes[np.unique(labels[now_mask & (latest > 0)])].sum() * CELL_KM ** 2)

    motion = estimate_motion([motion_weights(c) for c in cleaned], times)
    res["motion"] = motion

    if res["stage"] is None and motion and motion["speed"] >= MIN_SPEED_KMH:
        mid = predict_eta(latest, motion)
        if mid:
            etas = [mid[0]] + [e[0] for e in (predict_eta(latest, motion, f) for f in SPEED_UNCERTAINTY) if e]
            res["eta"] = (min(etas), max(etas))
            res["eta_mid"] = mid[0]
            res["front_km"] = float(mid[1])
            ux, uy = motion["vx"] / motion["speed"], motion["vy"] / motion["speed"]
            peaks, hit_labels = [0.0], set()
            for tau in range(mid[0], mid[0] + 31, STEP_MIN):
                d = motion["speed"] * tau / 60.0
                m = disc(-ux * d, -uy * d, BASE_RADIUS_KM + SPREAD_PER_KM * d)
                if (latest[m] > 0).any():
                    peaks.append(float(latest[m].max()))
                    hit_labels.update(np.unique(labels[m & (latest > 0)]).tolist())
            res["peak_dbz"] = max(peaks)
            res["cluster_km2"] = float(sum(sizes[l] for l in hit_labels if l) * CELL_KM ** 2)
            res["stage"] = "imminent" if mid[0] <= IMMINENT_MIN else ("early" if mid[0] <= EARLY_MIN else None)

    # 3.1: מגמת עוצמה של הגשם הרלוונטי (מעל הגבעה, או זה שבדרך אליה)
    if res["stage"] == "now":
        near = disc(0, 0, NOW_RADIUS_KM + 4.0)
        res["peaks"] = [_peak_in(f, near) for f in fields]
    elif res.get("eta_mid") is not None:
        res["peaks"] = system_peaks(fields, times, motion, res["eta_mid"])
    if res.get("peaks"):
        res["trend"] = classify_trend(res["peaks"])

    if res["stage"]:
        persistent = sum(1 for c in cleaned if (c > 0).sum() >= PERSIST_MIN_CELLS)
        res["checks"] = {
            "motion": bool(res["stage"] == "now" or (motion and motion["consistent"] and motion["quality"] >= 0.4)),
            "size": res["cluster_km2"] >= MIN_ALERT_AREA_KM2,
            "persist": persistent >= 3,
        }
        res["score"] = sum(res["checks"].values())

    return res


# ------------------------------------------------------------------ מסלול תאים

def extract_cells(field):
    """תאי גשם קומפקטיים מהשדה הגולמי. תא קטן וחזק לא נמחק כמו במסלול המערכות"""
    labels, sizes = label_clusters(field >= CELL_MIN_DBZ)
    cells = []
    for lab in range(1, len(sizes)):
        n = int(sizes[lab])
        if n < CELL_MIN_CELLS or n > CELL_MAX_CELLS:
            continue
        m = labels == lab
        x, y = float(XX[m].mean()), float(YY[m].mean())
        if math.hypot(x, y) > CELL_ZONE_KM:
            continue
        cells.append({"x": x, "y": y, "area": n, "peak": float(field[m].max()),
                      "r": math.sqrt(n * CELL_KM ** 2 / math.pi)})
    return cells


def track_cells(cells_pf, times):
    """מחבר תאים בין פריימים עוקבים (הקרוב ביותר קודם). מחזיר מסלולים שמסתיימים בפריים האחרון"""
    tracks, prev_map = [], {}
    for i in range(len(cells_pf)):
        cur_map = {}
        if i > 0:
            gate = CELL_MAX_SPEED_KMH * (times[i] - times[i - 1]) / 3600.0
            prev, cells = cells_pf[i - 1], cells_pf[i]
            pairs = sorted((math.hypot(c["x"] - p["x"], c["y"] - p["y"]), ci, pi)
                           for ci, c in enumerate(cells) for pi, p in enumerate(prev))
            used_c, used_p = set(), set()
            for d, ci, pi in pairs:
                if d > gate:
                    break
                if ci in used_c or pi in used_p:
                    continue
                used_c.add(ci)
                used_p.add(pi)
                tr = prev_map.get(pi)
                if tr is None:
                    tr = [(i - 1, prev[pi])]
                    tracks.append(tr)
                tr.append((i, cells[ci]))
                cur_map[ci] = tr
        prev_map = cur_map
    last = len(cells_pf) - 1
    return [t for t in tracks if t[-1][0] == last]


def angle_diff(ax, ay, bx, by):
    """הפרש כיוונים במעלות (0-180) בין שני וקטורים"""
    return abs((math.degrees(math.atan2(ax, ay) - math.atan2(bx, by)) + 180) % 360 - 180)


def reliable_motion(motion):
    """תנועה כללית שאפשר לסמוך עליה לבדיקת סבירות של תאים"""
    return bool(motion and motion["consistent"] and motion["quality"] >= 0.4
                and motion["speed"] >= MIN_SPEED_KMH)


def evaluate_track(tr, times, gmotion=None):
    """בודק מסלול תא אחד. info["why"] הוא None אם התא מתאים להתראה, אחרת סיבת הדחייה.
    gmotion: התנועה הכללית ממסלול המערכות (לבדיקת סבירות), או None"""
    (i0, c0), (i1, c1) = tr[0], tr[-1]
    dt_h = (times[i1] - times[i0]) / 3600.0
    vx, vy = (c1["x"] - c0["x"]) / dt_h, (c1["y"] - c0["y"]) / dt_h
    speed = math.hypot(vx, vy)
    dist = math.hypot(c1["x"], c1["y"])
    peaks = [c["peak"] for _, c in tr]
    info = {"why": None, "dist": dist, "peak": c1["peak"], "area": c1["area"], "n": len(tr),
            "speed": speed, "vx": vx, "vy": vy, "r": c1["r"], "eta_min": None, "adj": False,
            "trend": classify_trend(peaks), "peaks": peaks}
    if c1["peak"] < CELL_PEAK_DBZ:
        info["why"] = "weak"
        return info
    if max(peaks[:-1]) - c1["peak"] >= CELL_DECAY_DBZ and c1["peak"] < CELL_DECAY_MAX_PEAK:
        info["why"] = "decaying"       # 3.1: תא קטן שגווע בדרך
        return info
    if speed < CELL_MIN_APPROACH_KMH:
        info["why"] = "still"          # תא עומד: הגנה מהדי קרקע, גם כשהמערכת זזה
        return info
    if reliable_motion(gmotion):
        gs = gmotion["speed"]
        ratio = speed / gs
        if (ratio > CELL_SANITY_SPEED_RATIO or ratio < 1.0 / CELL_SANITY_SPEED_RATIO
                or angle_diff(vx, vy, gmotion["vx"], gmotion["vy"]) > CELL_SANITY_ANGLE_DEG):
            vx, vy, speed = gmotion["vx"], gmotion["vy"], gs
            info.update(vx=vx, vy=vy, speed=speed, adj=True)
    if dist <= c1["r"] + NOW_RADIUS_KM:
        info["eta_min"] = 0.0                      # התא כבר מעל הגבעה
    else:
        approach = -(c1["x"] * vx + c1["y"] * vy) / dist
        if approach < CELL_MIN_APPROACH_KMH:
            info["why"] = "away"
            return info
        along = -(c1["x"] * vx + c1["y"] * vy) / speed      # מרחק לאורך הכיוון עד הנקודה הקרובה
        d_min = math.sqrt(max(dist ** 2 - along ** 2, 0.0))
        if d_min > c1["r"] + CELL_HIT_BASE_KM + CELL_HIT_SPREAD * dist:
            info["why"] = "miss"
            return info
        chord = math.sqrt(max(c1["r"] ** 2 - d_min ** 2, 0.0))
        info["eta_min"] = 60.0 * max(0.0, along - chord) / speed
        if info["eta_min"] > CELL_MAX_ETA_MIN:
            info["why"] = "far"
            return info
    # ציון: 3 אם המסלול נצפה ב-3+ פריימים והתנועה האחרונה תואמת לכללית.
    # תא שהמהירות שלו הוחלפה בזו של המערכת: 3 אם נצפה ב-3+ פריימים (התנועה כבר עקבית)
    info["score"] = 2
    if info["adj"]:
        info["score"] = 3 if len(tr) >= 3 else 2
    elif len(tr) >= 3:
        (ia, ca), (ib, cb) = tr[-2], tr[-1]
        h = (times[ib] - times[ia]) / 3600.0
        lx, ly = (cb["x"] - ca["x"]) / h, (cb["y"] - ca["y"]) / h
        ls = math.hypot(lx, ly)
        ang = abs((math.degrees(math.atan2(lx, ly) - math.atan2(vx, vy)) + 180) % 360 - 180)
        if abs(ls - speed) <= max(8.0, 0.4 * speed) and (ls < 10.0 or ang <= 45.0):
            info["score"] = 3
    return info


def cell_min_score(stage, peak):
    """הציון הנדרש להתראת תא לפי השלב"""
    if stage == "now" and peak >= CELL_NOW_STRONG_DBZ:
        return MIN_SCORE_CELL_NOW_STRONG
    return MIN_SCORE_CELL


def cell_stage(eta):
    return "now" if eta == 0 else ("imminent" if eta <= IMMINENT_MIN else "early")


def find_popup(fields, times, tracks, infos):
    """3.1: תא שהתפתח במקום ממש ליד הגבעה. מסלול התאים רגיל דוחה תא שלא זז ("still") כדי
    להגן מהדי קרקע; כאן הוא מתקבל רק אם לפני 30 דקות לא היה שם כלום (הד קרקע קבוע)"""
    best = None
    for t, inf in zip(tracks, infos):
        if len(t) < 2 or t[-2][0] != len(fields) - 2:
            continue
        (_, cp), (_, c) = t[-2], t[-1]
        edge = max(0.0, math.hypot(c["x"], c["y"]) - c["r"])
        if edge > POPUP_EDGE_KM or min(cp["peak"], c["peak"]) < POPUP_PEAK_DBZ - 2 or c["peak"] < POPUP_PEAK_DBZ:
            continue
        h = (times[t[-1][0]] - times[t[-2][0]]) / 3600.0
        if math.hypot(c["x"] - cp["x"], c["y"] - cp["y"]) / h > POPUP_MAX_SPEED_KMH:
            continue                   # תא שזז מטופל במסלול הרגיל
        if t[0][0] == 0:
            continue                   # נצפה כבר לפני 30 דקות: לא "התפתח עכשיו"
        old = disc(c["x"], c["y"], c["r"] + 4.0)
        if _peak_in(fields[0], old) >= POPUP_CLEAR_DBZ:
            continue
        if best is None or edge < best[0]:
            best = (edge, c, inf)
    if best is None:
        return None
    edge, c, inf = best
    over = math.hypot(c["x"], c["y"]) <= c["r"] + NOW_RADIUS_KM
    stage = "now" if over else "imminent"
    return {"stage": stage, "min_score": MIN_SCORE_CELL, "eta": (0, 0) if over else (0, 15),
            "eta_mid": 0 if over else 5, "score": 3, "dist": math.hypot(c["x"], c["y"]), "edge_km": edge,
            "peak": c["peak"], "area_km2": c["area"] * CELL_KM ** 2, "vx": inf["vx"], "vy": inf["vy"],
            "speed": inf["speed"], "n": inf["n"], "adj": inf["adj"], "popup": True,
            "trend": inf["trend"], "peaks": inf["peaks"]}


def analyze_cells(fields, times, gmotion=None):
    """מסלול התאים: מחזיר {"alert": התראה או None, "dbg": סיכום ללוג, "draw": תאים לציור}.
    alert מוחזר רק אם התא עובר את הציון הנדרש לשלב שלו"""
    cells_pf = [extract_cells(f) for f in fields]
    out = {"alert": None, "dbg": None, "draw": []}
    latest = cells_pf[-1]
    if not latest:
        return out
    tracks = track_cells(cells_pf, times)
    infos = [evaluate_track(t, times, gmotion) for t in tracks]
    tracked_ids = {id(t[-1][1]) for t in tracks}
    out["draw"] = ([{"x": i["x"], "y": i["y"], "r": i["r"], "vx": i["vx"], "vy": i["vy"]}
                    for i in ({**inf, "x": t[-1][1]["x"], "y": t[-1][1]["y"]} for inf, t in zip(infos, tracks))]
                   + [{"x": c["x"], "y": c["y"], "r": c["r"], "vx": None, "vy": None}
                      for c in latest if id(c) not in tracked_ids])
    for i in infos:
        if i["why"] is None:
            i["stage"] = cell_stage(i["eta_min"])
            i["min_score"] = cell_min_score(i["stage"], i["peak"])
            if i["score"] < i["min_score"]:
                i["why"] = "low_score"
    good = [i for i in infos if i["why"] is None]
    if good:
        best = min(good, key=lambda i: i["eta_min"])
        eta = best["eta_min"]
        lo, hi = int(round(eta / SPEED_UNCERTAINTY[0])), int(round(eta / SPEED_UNCERTAINTY[1]))
        out["alert"] = {"stage": best["stage"], "min_score": best["min_score"],
                        "eta": (lo, hi), "eta_mid": int(round(eta)), "score": best["score"],
                        "dist": best["dist"], "edge_km": max(0.0, best["dist"] - best["r"]),
                        "peak": best["peak"], "area_km2": best["area"] * CELL_KM ** 2,
                        "vx": best["vx"], "vy": best["vy"], "speed": best["speed"], "n": best["n"],
                        "adj": best["adj"], "popup": False, "trend": best["trend"], "peaks": best["peaks"]}
        out["dbg"] = {"why": "ok", "d": round(best["dist"], 1), "pk": round(best["peak"]), "n": best["n"],
                      "v": round(best["speed"]), "adj": best["adj"]}
        return out
    popup = find_popup(fields, times, tracks, infos)
    if popup:
        out["alert"] = popup
        out["dbg"] = {"why": "popup", "d": round(popup["dist"], 1), "pk": round(popup["peak"]), "n": popup["n"],
                      "v": round(popup["speed"]), "adj": popup["adj"]}
        return out
    if infos:
        near = min(infos, key=lambda i: i["dist"])
        out["dbg"] = {"why": near["why"], "d": round(near["dist"], 1), "pk": round(near["peak"]),
                      "n": near["n"], "v": round(near["speed"]), "adj": near["adj"]}
    else:
        c = min(latest, key=lambda c: math.hypot(c["x"], c["y"]))
        out["dbg"] = {"why": "no_track", "d": round(math.hypot(c["x"], c["y"]), 1), "pk": round(c["peak"]),
                      "n": 1, "v": None}
    return out


def analyze(fields, times):
    """מאחד את שני המסלולים. התראה נשלחת אם qualifies(res)"""
    res = analyze_system(fields, times)
    res.setdefault("trend", None)
    res.update(src="system" if res["stage"] else None, min_score=MIN_SCORE_SYSTEM, boost=None,
               motion_used=res["motion"], cell=None)
    cell = analyze_cells(fields, times, res["motion"])
    res["cell_dbg"], res["cell_draw"] = cell["dbg"], cell["draw"]
    c = cell["alert"]
    if c:
        sys_ok = res["stage"] is not None and res["score"] >= MIN_SCORE_SYSTEM
        if not sys_ok or STAGE_RANK[c["stage"]] > STAGE_RANK[res["stage"]]:
            res.update(src="cell", stage=c["stage"], eta=c["eta"], eta_mid=c["eta_mid"],
                       front_km=c["edge_km"], peak_dbz=c["peak"], cluster_km2=c["area_km2"],
                       score=c["score"], checks={}, min_score=c["min_score"], cell=c,
                       trend=c["trend"], peaks=c["peaks"],
                       motion_used={"vx": c["vx"], "vy": c["vy"], "speed": c["speed"]})
    return res
