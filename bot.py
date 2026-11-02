#!/usr/bin/env python3
"""
בוט התראות גשם לגבעת שמואל - גרסה 2.6 בטא (מבוסס מכ"ם)

זרימה:
  1. מושך מ-RainViewer את 4 פריימי המכ"ם האחרונים (10 דקות בין פריים לפריים).
     ההיסטוריה נמצאת ב-API עצמו, ולכן חישוב התנועה לא תלוי בקובץ מצב ולא
     מושפע מעיכובים של GitHub Actions.
  2. מפענח את אריחי ה-PNG (סכמת Universal Blue) ל-dBZ ומדגם לרשת מקומית
     של 2 ק"מ סביב גבעת שמואל (הרבה יותר רחוק מערבה מעל הים).
  3. מנקה רעש (אשכולות קטנים), מעריך וקטור תנועה גלובלי בין הפריימים
     (cross-correlation) ו"מריץ אחורה" את השדה כדי לחשב מתי הגשם מגיע.
  4. מחשב ודאות לפי עקביות התנועה, גודל האשכול ויציבות ההד לאורך זמן.
  5. מסלול תאים (חדש ב-2.6): תאי גשם קטנים נעקבים בנפרד בין פריימים, על השדה
     הגולמי (בלי מחיקת תאים קטנים), ומקבלים וקטור תנועה וזמן הגעה משלהם.
  6. שולח התראה בטלגרם עם תמונת מכ"ם, לפי שלבים (early / imminent / now),
     פעם אחת לכל אירוע. מסלול המערכות (ים) מחמיר, מסלול התאים קצר ורגיש יותר.

שימוש:
  python bot.py                 # ריצה רגילה
  python bot.py --dry-run       # בלי שליחה ובלי עדכון מצב, מדפיס את ההודעה
  python bot.py --dry-run --png out.png   # שומר גם תמונת בקרה להשוואה מול ה-IMS

Weather data by Rain Viewer (https://www.rainviewer.com)
"""
import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import io

import numpy as np
import requests
from PIL import Image, ImageDraw

# ------------------------------------------------------------------ הגדרות

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
STATE_DIR = os.environ.get("STATE_DIR", "state")
OUT_DIR = os.environ.get("OUT_DIR", "out")   # תמונות התראה / חשד לפספוס (מועלות כ-artifact)
STATE_FILE = os.path.join(STATE_DIR, "state.json")
HISTORY_FILE = os.path.join(STATE_DIR, "history.jsonl")
HISTORY_MAX_LINES = 3000

VERSION = "2.6-beta"
TZ = ZoneInfo("Asia/Jerusalem")

TARGET_LAT, TARGET_LON = 32.0780, 34.8470

# RainViewer
API_URL = "https://api.rainviewer.com/public/weather-maps.json"
ZOOM = 7            # מקסימום שמותר ב-API החינמי
TILE_SIZE = 512
COLOR = 2           # Universal Blue - סכמת הצבעים היחידה שמתועדת כזמינה
N_FRAMES = 4        # 4 פריימים = 30 דקות של היסטוריה
MAX_FRAME_AGE_MIN = 45

# רשת עבודה מקומית (ק"מ ממזרח / מצפון לגבעת שמואל)
CELL_KM = 2.0
X_RANGE = (-140.0, 30.0)
Y_RANGE = (-80.0, 80.0)
KM_PER_DEG = 111.195

# זיהוי גשם
THRESH_DBZ = 20.0       # ~0.65 מ"מ/שעה - "גשם קל, יותר מטפטוף"
MIN_CLUSTER_KM2 = 20.0  # אשכול קטן מזה נחשב רעש
MIN_CELLS = int(MIN_CLUSTER_KM2 / CELL_KM ** 2)

# תנועה וחיזוי
MIN_CORR = 0.25          # מתחת לזה אומדן תנועה נפסל
MIN_SPEED_KMH = 5.0
MAX_SPEED_KMH = 120.0
MAX_LEAD_MIN = 120
STEP_MIN = 5
NOW_RADIUS_KM = 4.0
BASE_RADIUS_KM = 5.0     # רדיוס "פגיעה" סביב הגבעה
SPREAD_PER_KM = 0.08     # הרדיוס גדל עם המרחק (אי-ודאות בכיוון)
SPEED_UNCERTAINTY = (1.25, 0.8)  # טווח ETA: מהיר / איטי

# התראות - מסלול המערכות (גשם שמגיע מהים). מחמיר, לפי ניתוח ההתראות של 2.5:
# כל התראות ציון 2 וכל התראות early עם ETA של 45+ דקות נכשלו.
EARLY_MIN = 45           # התראה מוקדמת כש-ETA <= 45 דקות (היה 60)
IMMINENT_MIN = 20        # התראה "מגיע עכשיו" כש-ETA <= 20 דקות
MIN_SCORE_SYSTEM = 3     # 0-3. נדרש 3 (גבוהה) למסלול המערכות (היה 2)
MIN_ALERT_AREA_KM2 = 100.0
PERSIST_MIN_CELLS = 3 * MIN_CELLS
STAGE_RANK = {"early": 1, "imminent": 2, "now": 3}

# מסלול תאים (חדש ב-2.6): תאים קטנים וקומפקטיים קרוב לגבעה, נעקבים על השדה הגולמי
CELL_ZONE_KM = 60.0       # מחוץ לטווח הזה התאים שייכים למסלול המערכות
CELL_MIN_DBZ = 15.0       # סף לזיהוי גבולות התא
CELL_PEAK_DBZ = 22.0      # עוצמת שיא מינימלית כדי להתריע (~0.9 מ"מ/שעה)
CELL_MIN_CELLS = 3        # שטח מינימלי ~12 קמ"ר (נגד רעש של פיקסל בודד)
CELL_MAX_CELLS = 60       # גדול מזה = מערכת, לא תא
CELL_MAX_ETA_MIN = 30     # מעבר לזה אין התראת תא (המערכות מטופלות במסלול האחר)
CELL_MAX_SPEED_KMH = 90.0 # מגבלת התאמה בין פריימים
CELL_MIN_APPROACH_KMH = 5.0
CELL_HIT_BASE_KM = 4.0    # "פגיעה": מרחק מסלול התא מהגבעה <= רדיוס התא + זה + תוספת
CELL_HIT_SPREAD = 0.10    # תוספת אי-ודאות לפי המרחק
MIN_SCORE_CELL = 2        # תא שנצפה בשני פריימים = 2, שלושה פריימים עקביים = 3

# חשד לפספוס: הד קרוב בלי התראה => נשמרת תמונה לבדיקה
SUSPICIOUS_RADIUS_KM = 10.0
SUSPICIOUS_MIN_CELLS = 2
SUSPICIOUS_MIN_DBZ = 20.0
COOLDOWN_MIN = 90        # חזרה על אותו שלב באותו אירוע
EPISODE_GAP_MIN = 60     # בלי גשם חזוי/קיים כל כך הרבה זמן = אירוע חדש

# סכמת Universal Blue (גשם), dBZ 15..64, ועוד 65 ו-75+
_UB_HEX = (
    "88ddee 6cd1eb 51c5e8 36bae5 1baee2 00a3e0 009ad5 0091ca 0088bf 007fb4 "
    "0077aa 0070a3 00699c 006295 005b8e 005588 005180 004e78 004a70 004768 "
    "ffee00 ffe000 ffd200 ffc500 ffb700 ffaa00 ff9f00 ff9500 ff8b00 ff8100 "
    "ff4400 f23600 e62800 d91b00 cd0d00 c10000 a80000 8f0000 760000 5d0000 "
    "ffaaff ff9fff ff95ff ff8bff ff81ff ff77ff ff6cff ff62ff ff58ff ff4eff"
).split()
LUT_DBZ = np.array(list(range(15, 65)) + [65, 75], dtype=np.float32)
LUT_RGB = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in _UB_HEX + ["ffffff", "00ff00"]],
                   dtype=np.int32)


# ------------------------------------------------------------------ גיאומטריה

def build_grid():
    xs = np.arange(X_RANGE[0] + CELL_KM / 2, X_RANGE[1], CELL_KM)
    ys = np.arange(Y_RANGE[0] + CELL_KM / 2, Y_RANGE[1], CELL_KM)
    xx, yy = np.meshgrid(xs, ys)  # שורות עולות צפונה, עמודות עולות מזרחה
    lon = TARGET_LON + xx / (KM_PER_DEG * math.cos(math.radians(TARGET_LAT)))
    lat = TARGET_LAT + yy / KM_PER_DEG
    n = (2 ** ZOOM) * TILE_SIZE
    px = ((lon + 180.0) / 360.0 * n).astype(np.int64)
    py = ((1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n).astype(np.int64)
    return xx, yy, px, py


XX, YY, PX, PY = build_grid()
TX0, TX1 = int(PX.min() // TILE_SIZE), int(PX.max() // TILE_SIZE)
TY0, TY1 = int(PY.min() // TILE_SIZE), int(PY.max() // TILE_SIZE)


# ------------------------------------------------------------------ RainViewer

def http_get(session, url, tries=3):
    last = "unknown"
    for attempt in range(tries):
        try:
            res = session.get(url, timeout=(5, 15))
            if res.status_code == 200:
                return res
            last = f"HTTP {res.status_code}"
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"הורדה נכשלה ({last})")


def decode_tile(png_bytes):
    """PNG של RainViewer -> מטריצת dBZ (0 = אין גשם / מתחת ל-15 dBZ)"""
    img = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))
    out = np.zeros(img.shape[:2], dtype=np.float32)
    opaque = img[..., 3] >= 250      # מתחת ל-15 dBZ הסכמה שקופה חלקית
    if opaque.any():
        px = img[..., :3][opaque].astype(np.int32)
        uniq, inv = np.unique(px, axis=0, return_inverse=True)
        d = ((uniq[:, None, :] - LUT_RGB[None, :, :]) ** 2).sum(-1)
        best = d.argmin(1)
        vals = np.where(d.min(1) <= 30 ** 2, LUT_DBZ[best], 0.0)
        out[opaque] = vals[np.asarray(inv).ravel()]
    return out


def fetch_frames_index(session):
    data = http_get(session, API_URL).json()
    past = data["radar"]["past"]
    if len(past) < 3:
        raise RuntimeError("פחות מ-3 פריימי מכ\"ם זמינים")
    return data["host"], past[-N_FRAMES:]


def load_fields(session, host, frames):
    """מחזיר רשימת שדות dBZ על הרשת המקומית, מהישן לחדש"""
    jobs = [(f, tx, ty) for f in range(len(frames))
            for tx in range(TX0, TX1 + 1) for ty in range(TY0, TY1 + 1)]

    def get(job):
        f, tx, ty = job
        url = f"{host}{frames[f]['path']}/{TILE_SIZE}/{ZOOM}/{tx}/{ty}/{COLOR}/0_0.png"
        return decode_tile(http_get(session, url).content)

    with ThreadPoolExecutor(max_workers=8) as pool:
        tiles = list(pool.map(get, jobs))

    h, w = (TY1 - TY0 + 1) * TILE_SIZE, (TX1 - TX0 + 1) * TILE_SIZE
    mosaics = [np.zeros((h, w), np.float32) for _ in frames]
    for (f, tx, ty), tile in zip(jobs, tiles):
        y0, x0 = (ty - TY0) * TILE_SIZE, (tx - TX0) * TILE_SIZE
        mosaics[f][y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] = tile
    iy, ix = PY - TY0 * TILE_SIZE, PX - TX0 * TILE_SIZE
    return [m[iy, ix] for m in mosaics]


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


def evaluate_track(tr, times):
    """בודק מסלול תא אחד. info["why"] הוא None אם התא מתאים להתראה, אחרת סיבת הדחייה"""
    (i0, c0), (i1, c1) = tr[0], tr[-1]
    dt_h = (times[i1] - times[i0]) / 3600.0
    vx, vy = (c1["x"] - c0["x"]) / dt_h, (c1["y"] - c0["y"]) / dt_h
    speed = math.hypot(vx, vy)
    dist = math.hypot(c1["x"], c1["y"])
    info = {"why": None, "dist": dist, "peak": c1["peak"], "area": c1["area"], "n": len(tr),
            "speed": speed, "vx": vx, "vy": vy, "r": c1["r"], "eta_min": None}
    if c1["peak"] < CELL_PEAK_DBZ:
        info["why"] = "weak"
        return info
    if speed < CELL_MIN_APPROACH_KMH:
        info["why"] = "still"
        return info
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
    # ציון: 3 אם המסלול נצפה ב-3+ פריימים והתנועה האחרונה תואמת לכללית
    info["score"] = 2
    if len(tr) >= 3:
        (ia, ca), (ib, cb) = tr[-2], tr[-1]
        h = (times[ib] - times[ia]) / 3600.0
        lx, ly = (cb["x"] - ca["x"]) / h, (cb["y"] - ca["y"]) / h
        ls = math.hypot(lx, ly)
        ang = abs((math.degrees(math.atan2(lx, ly) - math.atan2(vx, vy)) + 180) % 360 - 180)
        if abs(ls - speed) <= max(8.0, 0.4 * speed) and (ls < 10.0 or ang <= 45.0):
            info["score"] = 3
    return info


def analyze_cells(fields, times):
    """מסלול התאים: מחזיר {"alert": התראה או None, "dbg": סיכום ללוג, "draw": תאים לציור}"""
    cells_pf = [extract_cells(f) for f in fields]
    out = {"alert": None, "dbg": None, "draw": []}
    latest = cells_pf[-1]
    if not latest:
        return out
    tracks = track_cells(cells_pf, times)
    infos = [evaluate_track(t, times) for t in tracks]
    tracked_ids = {id(t[-1][1]) for t in tracks}
    out["draw"] = ([{"x": i["x"], "y": i["y"], "r": i["r"], "vx": i["vx"], "vy": i["vy"]}
                    for i in ({**inf, "x": t[-1][1]["x"], "y": t[-1][1]["y"]} for inf, t in zip(infos, tracks))]
                   + [{"x": c["x"], "y": c["y"], "r": c["r"], "vx": None, "vy": None}
                      for c in latest if id(c) not in tracked_ids])
    good = [i for i in infos if i["why"] is None]
    if good:
        best = min(good, key=lambda i: i["eta_min"])
        eta = best["eta_min"]
        lo, hi = int(round(eta / SPEED_UNCERTAINTY[0])), int(round(eta / SPEED_UNCERTAINTY[1]))
        out["alert"] = {"stage": "now" if eta == 0 else ("imminent" if eta <= IMMINENT_MIN else "early"),
                        "eta": (lo, hi), "eta_mid": int(round(eta)), "score": best["score"],
                        "dist": best["dist"], "edge_km": max(0.0, best["dist"] - best["r"]),
                        "peak": best["peak"], "area_km2": best["area"] * CELL_KM ** 2,
                        "vx": best["vx"], "vy": best["vy"], "speed": best["speed"], "n": best["n"]}
        out["dbg"] = {"why": "ok", "d": round(best["dist"], 1), "pk": round(best["peak"]), "n": best["n"],
                      "v": round(best["speed"])}
        return out
    if infos:
        near = min(infos, key=lambda i: i["dist"])
        out["dbg"] = {"why": near["why"], "d": round(near["dist"], 1), "pk": round(near["peak"]),
                      "n": near["n"], "v": round(near["speed"])}
    else:
        c = min(latest, key=lambda c: math.hypot(c["x"], c["y"]))
        out["dbg"] = {"why": "no_track", "d": round(math.hypot(c["x"], c["y"]), 1), "pk": round(c["peak"]),
                      "n": 1, "v": None}
    return out


def analyze(fields, times):
    """מאחד את שני המסלולים. התראה נשלחת אם qualifies(res)"""
    res = analyze_system(fields, times)
    res.update(src="system" if res["stage"] else None, min_score=MIN_SCORE_SYSTEM,
               motion_used=res["motion"], cell=None)
    cell = analyze_cells(fields, times)
    res["cell_dbg"], res["cell_draw"] = cell["dbg"], cell["draw"]
    c = cell["alert"]
    if c:
        sys_ok = res["stage"] is not None and res["score"] >= MIN_SCORE_SYSTEM
        if not sys_ok or STAGE_RANK[c["stage"]] > STAGE_RANK[res["stage"]]:
            res.update(src="cell", stage=c["stage"], eta=c["eta"], eta_mid=c["eta_mid"],
                       front_km=c["edge_km"], peak_dbz=c["peak"], cluster_km2=c["area_km2"],
                       score=c["score"], checks={}, min_score=MIN_SCORE_CELL, cell=c,
                       motion_used={"vx": c["vx"], "vy": c["vy"], "speed": c["speed"]})
    return res


# ------------------------------------------------------------------ מצב והודעות

def atomic_write(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def append_history(entry):
    os.makedirs(STATE_DIR, exist_ok=True)
    try:
        with open(HISTORY_FILE) as f:
            lines = f.read().splitlines()
    except OSError:
        lines = []
    lines.append(json.dumps(entry, ensure_ascii=False))
    
    # שומר את קובץ ה-JSON Lines הקיים
    atomic_write(HISTORY_FILE, "\n".join(lines[-HISTORY_MAX_LINES:]) + "\n")
    
    # מייצר גם קובץ history.json מסודר עבור הדשבורד
    json_entries = [json.loads(line) for line in lines if line.strip()]
    json_path = os.path.join(STATE_DIR, "history.json")
    atomic_write(json_path, json.dumps(json_entries, ensure_ascii=False, indent=2))


def parse_ts(s):
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def should_send(state, stage, now):
    last = parse_ts(state.get("last_alert"))
    if last is None:
        return True
    seen = parse_ts(state.get("last_rain_seen"))
    if seen is None or now - seen > timedelta(minutes=EPISODE_GAP_MIN):
        return True                                   # אירוע חדש
    if STAGE_RANK[stage] > STAGE_RANK.get(state.get("last_stage"), 0):
        return True                                   # החמרה: early -> imminent -> now
    return now - last >= timedelta(minutes=COOLDOWN_MIN)


def qualifies(res):
    return res["stage"] is not None and res["score"] >= res["min_score"]


def build_message(res, radar_time, now):
    src = res.get("src") or "system"
    m = res.get("motion_used")
    age_min = max(0.0, (now - radar_time).total_seconds() / 60.0)
    if src == "cell":
        headlines = {"early": "🌧️ <b>תא גשם מתקרב לגבעת שמואל</b>",
                     "imminent": "⚠️ <b>תא גשם מגיע - בעוד דקות ספורות</b>",
                     "now": "☔ <b>תא גשם מעל גבעת שמואל</b>"}
    else:
        headlines = {"early": "🌧️ <b>גשם בדרך לגבעת שמואל</b>",
                     "imminent": "⚠️ <b>הגשם מגיע - בעוד דקות ספורות</b>",
                     "now": "☔ <b>יורד גשם באזור גבעת שמואל</b>"}
    lines = [headlines[res["stage"]], ""]
    if res["stage"] != "now":
        lo, hi = (max(0, int(round(x - age_min))) for x in res["eta"])
        eta = f"כ-<b>{lo} דקות</b>" if abs(hi - lo) < 5 else f"כ-<b>{lo}-{hi} דקות</b>"
        arrival = (now + timedelta(minutes=(lo + hi) / 2)).strftime("%H:%M")
        frm = compass((math.degrees(math.atan2(m["vx"], m["vy"])) + 180) % 360)
        label = "מרחק התא" if src == "cell" else "קצה החזית"
        lines += [f"⏱️ <b>זמן הגעה:</b> {eta} (סביב {arrival})",
                  f"🧭 <b>מגיע מכיוון:</b> {frm}, ~{m['speed']:.0f} קמ\"ש",
                  f"📏 <b>{label}:</b> כ-{res['front_km']:.0f} ק\"מ מהגבעה"]
    dbz = res.get("peak_dbz", THRESH_DBZ)
    lines.append(f"💧 <b>עוצמה:</b> {intensity_label(dbz)} (~{dbz_to_mmh(dbz):.1f} מ\"מ/שעה)")
    conf = {3: "גבוהה 🟢", 2: "בינונית 🟡"}.get(res["score"], "נמוכה 🔴")
    ok = res["checks"]
    if src == "cell":
        lines.append(f"🎯 <b>ודאות:</b> {conf}  (תא שנצפה ב-{res['cell']['n']} פריימים רצופים · התראה קצרה)")
    elif ok:
        lines.append(f"🎯 <b>ודאות:</b> {conf}  "
                     f"(תנועה {'✓' if ok['motion'] else '✗'} · גודל {'✓' if ok['size'] else '✗'} · "
                     f"יציבות {'✓' if ok['persist'] else '✗'})")
    lines += ["", f"🛰️ <i>פריים מכ\"ם: {radar_time.strftime('%H:%M')} · "
                  f"<a href=\"https://www.rainviewer.com\">Weather data by Rain Viewer</a></i>"]
    return "\n".join(lines)


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            print(f"שגיאה בשליחה לטלגרם: {res.status_code} {res.text[:200]}")
            return False
        return True
    except requests.RequestException as e:
        print(f"שגיאה בשליחה לטלגרם: {type(e).__name__}")   # ההודעה המלאה כוללת את הטוקן
        return False


def send_telegram_photo(png_bytes, caption=""):
    """שולח תמונה עם כיתוב. מחזיר False גם אם הכיתוב ארוך מדי, ואז שולחים הודעת טקסט רגילה"""
    if len(caption) > 1000:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    data = {"chat_id": TELEGRAM_CHAT_ID}
    if caption:
        data.update(caption=caption, parse_mode="HTML")
    try:
        res = requests.post(url, data=data, files={"photo": ("radar.png", png_bytes, "image/png")}, timeout=20)
        if res.status_code != 200:
            print(f"שגיאה בשליחת תמונה לטלגרם: {res.status_code} {res.text[:200]}")
            return False
        return True
    except requests.RequestException as e:
        print(f"שגיאה בשליחת תמונה לטלגרם: {type(e).__name__}")
        return False


# ------------------------------------------------------------------ תמונת מכ"ם

# קו חוף משוער (קו רוחב, קו אורך), רק לכיוון בתמונה
COAST = [(32.82, 34.99), (32.69, 34.94), (32.50, 34.89), (32.33, 34.85), (32.17, 34.79),
         (32.08, 34.76), (32.05, 34.75), (32.02, 34.74), (31.97, 34.72), (31.93, 34.69),
         (31.88, 34.68), (31.80, 34.64), (31.67, 34.55)]
DBZ_COLORS = [(15, (90, 200, 255)), (20, (40, 120, 255)), (30, (60, 220, 90)),
              (38, (255, 220, 0)), (45, (255, 90, 0)), (55, (255, 0, 160))]


def render_png(raw, res, radar_time=None, scale=6):
    """תמונת מכ"ם גולמית (15+ dBZ) עם קו חוף, טבעות 10/20/30 ק"מ, תאים וחצי תנועה. מחזיר bytes"""
    ny, nx = raw.shape
    rgb = np.full((ny, nx, 3), 38, np.uint8)
    for lo, col in DBZ_COLORS:
        rgb[raw >= lo] = col
    img = Image.fromarray(np.flipud(rgb)).resize((nx * scale, ny * scale), Image.NEAREST)
    dr = ImageDraw.Draw(img)

    def pt(x, y):
        return ((x - X_RANGE[0]) / CELL_KM * scale, (Y_RANGE[1] - y) / CELL_KM * scale)

    cos0 = math.cos(math.radians(TARGET_LAT))
    dr.line([pt((lon - TARGET_LON) * KM_PER_DEG * cos0, (lat - TARGET_LAT) * KM_PER_DEG) for lat, lon in COAST],
            fill=(170, 170, 170), width=2)
    cx, cy = pt(0, 0)
    for rkm in (10, 20, 30):
        rr = rkm / CELL_KM * scale
        dr.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(110, 110, 110))
        dr.text((cx - 6, cy - rr - 11), str(rkm), fill=(190, 190, 190))
    for c in res.get("cell_draw") or []:
        px, py = pt(c["x"], c["y"])
        rr = max(c["r"] / CELL_KM * scale, 3)
        dr.ellipse([px - rr, py - rr, px + rr, py + rr], outline="white")
        if c["vx"] is not None:
            dr.line([(px, py), pt(c["x"] + c["vx"] * 0.5, c["y"] + c["vy"] * 0.5)], fill="white", width=2)
    m = res.get("motion")
    if m and m["speed"] >= MIN_SPEED_KMH:
        ux, uy = m["vx"] / m["speed"], m["vy"] / m["speed"]
        d = m["speed"] * MAX_LEAD_MIN / 60.0
        dr.line([pt(0, 0), pt(-ux * d, -uy * d)], fill=(255, 255, 0), width=1)
    dr.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline="white", width=2)
    dr.rectangle([0, 0, 235, 14], fill=(0, 0, 0))
    when = radar_time.strftime("%H:%M") if radar_time else "test"
    dr.text((3, 2), f"radar {when}  v{VERSION}  {res.get('src') or '-'}/{res.get('stage') or '-'}",
            fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ------------------------------------------------------------------ ראשי

def run(dry_run=False, png=None):
    if not dry_run:
        missing = [n for n, v in [("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
                                  ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)] if not v]
        if missing:
            print(f"חסרים משתני סביבה: {', '.join(missing)}")
            return 1

    now = datetime.now(TZ)
    try:
        with requests.Session() as session:
            host, frames = fetch_frames_index(session)
            age = (now.timestamp() - frames[-1]["time"]) / 60.0
            if age > MAX_FRAME_AGE_MIN:
                print(f"הפריים האחרון ישן ({age:.0f} דקות). מדלג.")
                return 1
            fields = load_fields(session, host, frames)
    except (RuntimeError, KeyError, ValueError) as e:
        print(f"שגיאה בקבלת נתוני מכ\"ם: {e}")
        return 1

    times = [f["time"] for f in frames]
    radar_time = datetime.fromtimestamp(times[-1], TZ)
    res = analyze(fields, times)
    raw = fields[-1]

    state = load_state()
    sent = False
    stage = res["stage"]
    png_bytes = None
    if qualifies(res):
        msg = build_message(res, radar_time, now)
        if dry_run:
            print(msg)
        elif should_send(state, stage, now):
            png_bytes = render_png(raw, res, radar_time)
            sent = send_telegram_photo(png_bytes, msg) or send_telegram(msg)

    # תמונה לבדיקה: בכל התראה שנשלחה, ובכל חשד לפספוס (הד קרוב בלי התראה)
    suspicious = (not sent) and res["diag"]["raw20_10km"] >= SUSPICIOUS_MIN_CELLS
    if png:
        with open(png, "wb") as f:
            f.write(render_png(raw, res, radar_time))
    if not dry_run and (sent or suspicious):
        os.makedirs(OUT_DIR, exist_ok=True)
        name = f"{'alert' if sent else 'suspicious'}_{radar_time:%Y%m%d_%H%M}.png"
        with open(os.path.join(OUT_DIR, name), "wb") as f:
            f.write(png_bytes or render_png(raw, res, radar_time))

    m = res["motion"]
    summary = {"v": VERSION, "t": now.isoformat(timespec="seconds"), "radar": radar_time.isoformat(timespec="seconds"),
               "rain_cells": res["rain_cells"], "stage": stage, "src": res["src"], "eta": res["eta"],
               "score": res["score"],
               "speed": round(m["speed"], 1) if m else None,
               "heading": round(math.degrees(math.atan2(m["vx"], m["vy"])) % 360) if m else None,
               **res["diag"], "cell": res["cell_dbg"], "sent": sent}
    print(json.dumps(summary, ensure_ascii=False))

    if not dry_run:
        if stage is not None:
            state["last_rain_seen"] = now.isoformat()
        if sent:
            state.update(last_alert=now.isoformat(), last_stage=stage)
        atomic_write(STATE_FILE, json.dumps(state))
        append_history(summary)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="בלי שליחה ובלי שמירת מצב")
    ap.add_argument("--png", help="שמירת תמונת בקרה (למשל out.png)")
    ap.add_argument("--test-telegram", action="store_true", help="שולח הודעת בדיקה ותמונת בדיקה וסיים")
    a = ap.parse_args()
    if a.test_telegram:
        ok = send_telegram(f"✅ בדיקת חיבור: בוט הגשם {VERSION} מחובר לטלגרם.")
        ok_img = send_telegram_photo(render_png(np.zeros(XX.shape, np.float32), {}, None), "🖼️ בדיקת שליחת תמונה")
        print("ההודעה נשלחה" if ok else "שליחת ההודעה נכשלה", "| התמונה נשלחה" if ok_img else "| שליחת התמונה נכשלה")
        sys.exit(0 if (ok and ok_img) else 1)
    sys.exit(run(a.dry_run, a.png))
