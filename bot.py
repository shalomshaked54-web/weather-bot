#!/usr/bin/env python3
"""
בוט התראות גשם לגבעת שמואל - גרסה 2.4 (מבוסס מכ"ם)

זרימה:
  1. מושך מ-RainViewer את 4 פריימי המכ"ם האחרונים (10 דקות בין פריים לפריים).
     ההיסטוריה נמצאת ב-API עצמו, ולכן חישוב התנועה לא תלוי בקובץ מצב ולא
     מושפע מעיכובים של GitHub Actions.
  2. מפענח את אריחי ה-PNG (סכמת Universal Blue) ל-dBZ ומדגם לרשת מקומית
     של 2 ק"מ סביב גבעת שמואל (הרבה יותר רחוק מערבה מעל הים).
  3. מנקה רעש (אשכולות קטנים), מעריך וקטור תנועה גלובלי בין הפריימים
     (cross-correlation) ו"מריץ אחורה" את השדה כדי לחשב מתי הגשם מגיע.
  4. מחשב ודאות לפי עקביות התנועה, גודל האשכול ויציבות ההד לאורך זמן.
  5. שולח התראה בטלגרם לפי שלבים (early / imminent / now), פעם אחת לכל אירוע.

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
STATE_FILE = os.path.join(STATE_DIR, "state.json")
HISTORY_FILE = os.path.join(STATE_DIR, "history.jsonl")
HISTORY_MAX_LINES = 3000

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
THRESH_DBZ = 20.0       # ~0.65 מ"מ/ש' - "גשם קל, יותר מטפטוף"
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

# התראות
EARLY_MIN = 60           # התראה מוקדמת כש-ETA <= 60 דקות
IMMINENT_MIN = 20        # התראה "מגיע עכשיו" כש-ETA <= 20 דקות
MIN_SCORE = 2            # 0-3. 3 = גבוהה, 2 = בינונית. כדאי להעלות ל-3 אם יש יותר מדי התראות
MIN_ALERT_AREA_KM2 = 100.0
PERSIST_MIN_CELLS = 3 * MIN_CELLS
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


def analyze(fields, times):
    """הליבה: שדות dBZ גולמיים (ישן->חדש) וזמנים -> תוצאה מובנית"""
    cleaned, labels, sizes = [], None, None
    for f in fields:
        c, lab, sz = clean_field(f)
        cleaned.append(c)
        labels, sizes = lab, sz   # תוויות הפריים האחרון
    latest = cleaned[-1]

    res = {"rain_cells": int((latest > 0).sum()), "stage": None, "motion": None,
           "eta": None, "score": 0, "checks": {}}

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
    atomic_write(HISTORY_FILE, "\n".join(lines[-HISTORY_MAX_LINES:]) + "\n")


def parse_ts(s):
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


STAGE_RANK = {"early": 1, "imminent": 2, "now": 3}


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


def build_message(res, radar_time, now):
    m = res["motion"]
    age_min = max(0.0, (now - radar_time).total_seconds() / 60.0)
    headline = {"early": "🌧️ <b>גשם בדרך לגבעת שמואל</b>",
                "imminent": "⚠️ <b>הגשם מגיע - בעוד דקות ספורות</b>",
                "now": "☔ <b>יורד גשם באזור גבעת שמואל</b>"}[res["stage"]]
    lines = [headline, ""]
    if res["stage"] != "now":
        lo, hi = (max(0, int(round(x - age_min))) for x in res["eta"])
        eta = f"כ-<b>{lo} דקות</b>" if abs(hi - lo) < 5 else f"כ-<b>{lo}-{hi} דקות</b>"
        arrival = (now + timedelta(minutes=(lo + hi) / 2)).strftime("%H:%M")
        frm = compass((math.degrees(math.atan2(m["vx"], m["vy"])) + 180) % 360)
        lines += [f"⏱️ <b>זמן הגעה:</b> {eta} (סביב {arrival})",
                  f"🧭 <b>מגיע מכיוון:</b> {frm}, ~{m['speed']:.0f} קמ\"ש",
                  f"📏 <b>קצה החזית:</b> כ-{res['front_km']:.0f} ק\"מ מהגבעה"]
    dbz = res.get("peak_dbz", THRESH_DBZ)
    lines.append(f"💧 <b>עוצמה:</b> {intensity_label(dbz)} (~{dbz_to_mmh(dbz):.1f} מ\"מ/ש')")
    conf = {3: "גבוהה 🟢", 2: "בינונית 🟡"}.get(res["score"], "נמוכה 🔴")
    ok = res["checks"]
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


# ------------------------------------------------------------------ תמונת בקרה

def save_png(field, res, path, scale=6):
    ny, nx = field.shape
    rgb = np.zeros((ny, nx, 3), np.uint8)
    for lo, col in [(THRESH_DBZ, (80, 170, 255)), (30, (60, 220, 90)), (38, (255, 220, 0)),
                    (45, (255, 90, 0)), (55, (255, 0, 160))]:
        rgb[field >= lo] = col
    img = Image.fromarray(np.flipud(rgb)).resize((nx * scale, ny * scale), Image.NEAREST)
    dr = ImageDraw.Draw(img)

    def pt(x, y):
        return ((x - X_RANGE[0]) / CELL_KM * scale, (Y_RANGE[1] - y) / CELL_KM * scale)

    tx, ty = pt(0, 0)
    dr.ellipse([tx - 5, ty - 5, tx + 5, ty + 5], outline="white", width=2)
    m = res.get("motion")
    if m and m["speed"] >= MIN_SPEED_KMH:
        ux, uy = m["vx"] / m["speed"], m["vy"] / m["speed"]
        d = m["speed"] * MAX_LEAD_MIN / 60.0
        dr.line([pt(0, 0), pt(-ux * d, -uy * d)], fill="white", width=1)
    img.save(path)


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
    if png:
        save_png(clean_field(fields[-1])[0], res, png)

    state = load_state()
    sent = False
    msg = None
    stage = res["stage"]
    qualifies = stage is not None and res["score"] >= MIN_SCORE
    if qualifies:
        msg = build_message(res, radar_time, now)
        if dry_run:
            print(msg)
        elif should_send(state, stage, now):
            sent = send_telegram(msg)

    m = res["motion"]
    summary = {"t": now.isoformat(timespec="seconds"), "radar": radar_time.isoformat(timespec="seconds"),
               "rain_cells": res["rain_cells"], "stage": stage, "eta": res["eta"], "score": res["score"],
               "speed": round(m["speed"], 1) if m else None, "sent": sent}
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
       ap.add_argument("--test-telegram", action="store_true")
    a = ap.parse_args()
    if a.test_telegram:
        ok = send_telegram("✅ בדיקת חיבור: בוט הגשם מחובר לטלגרם.")
        print("ההודעה נשלחה" if ok else "השליחה נכשלה")
        sys.exit(0 if ok else 1)
    sys.exit(run(a.dry_run, a.png))
