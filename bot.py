#!/usr/bin/env python3
"""
התראת גשם לגבעת שמואל על בסיס מכ"ם RainViewer (שמקורו, בישראל, ב-IMS).

עיקרון העבודה:
  1. מורידים את 4 פריימי המכ"ם האחרונים (מרווח של 10 דקות) כאריחי PNG.
  2. מפענחים את הצבעים (סכמת Universal Blue) חזרה ל-dBZ.
  3. בונים רשת מקומית של 200x200 ק"מ סביב גבעת שמואל.
  4. מעריכים את כיוון ומהירות התנועה של שדה המשקעים (קורלציה בין פריימים).
  5. מקרינים את התנועה "אחורה" מהאתר ובודקים מתי ההד הקרוב ביותר צפוי להגיע (ETA).

תלויות:  pip install requests numpy pillow
הרצה:   python rain_radar_alert.py --dry-run --debug-dir debug    (בדיקה, בלי שליחה)
         python rain_radar_alert.py                                (ריצה רגילה, מומלץ כל 10 דקות)

משתני סביבה: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, (אופציונלי) STATE_DIR
"""
import argparse
import io
import json
import math
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import requests
from PIL import Image, ImageDraw

# ------------------------------------------------------------------ הגדרות
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SITE_LAT, SITE_LON = 32.0780, 34.8470  # גבעת שמואל
TZ_ISRAEL = ZoneInfo("Asia/Jerusalem")

STATE_DIR = os.environ.get("STATE_DIR", ".")
STATE_FILE = os.path.join(STATE_DIR, "rain_alert_state.json")
COOLDOWN_MINUTES = 90

# --- RainViewer ---
WEATHER_MAPS_URL = "https://api.rainviewer.com/public/weather-maps.json"
ZOOM = 7               # המקסימום שהחבילה החינמית מאפשרת
TILE_SIZE = 256
COLOR_SCHEME = 2       # Universal Blue - הסכמה היחידה שנשארה בחינם
TILE_OPTIONS = "0_0"   # בלי החלקה ובלי שלג, כדי שהצבעים יהיו מדויקים לפלטה
FRAMES_N = 4           # כמה פריימים אחרונים לנתח (40 דקות)
MAX_FRAME_AGE_MIN = 40 # פריים אחרון ישן מזה נחשב לא רלוונטי

# --- רשת מקומית ---
GRID_RADIUS_KM = 100
GRID_STEP_KM = 2
KM_PER_DEG_LAT = 111.2

# --- ספי החלטה ---
DBZ_MIN = 20.0           # מתחת לזה לא נחשב גשם (בערך 0.6 מ"מ/ש')
MIN_ECHO_CELLS = 15      # מינימום תאי רשת עם הד כדי למדוד תנועה (15 תאים = 60 קמ"ר)
MAX_SPEED_KMH = 90.0
MIN_SPEED_KMH = 5.0
MIN_CORR = 0.35          # מתחת לזה מדידת התנועה לא אמינה
CORRIDOR_HALF_WIDTH_KM = 6.0  # רוחב המסדרון מצד לצד שבו בודקים אם ההד "פוגע" באתר
ALERT_MAX_ETA_MIN = 60   # שולחים התראה רק אם ה-ETA עד שעה
NEAR_ECHO_KM = 15.0      # גיבוי: הד קרוב מזה מתריע גם בלי מדידת תנועה (ביטחון נמוך)

# --- פלטת Universal Blue (גשם) לפי הטבלה הרשמית של RainViewer ---
_UB_15_TO_64 = [
    "88ddee", "6cd1eb", "51c5e8", "36bae5", "1baee2",  # 15-19
    "00a3e0", "009ad5", "0091ca", "0088bf", "007fb4",  # 20-24
    "0077aa", "0070a3", "00699c", "006295", "005b8e",  # 25-29
    "005588", "005180", "004e78", "004a70", "004768",  # 30-34
    "ffee00", "ffe000", "ffd200", "ffc500", "ffb700",  # 35-39
    "ffaa00", "ff9f00", "ff9500", "ff8b00", "ff8100",  # 40-44
    "ff4400", "f23600", "e62800", "d91b00", "cd0d00",  # 45-49
    "c10000", "a80000", "8f0000", "760000", "5d0000",  # 50-54
    "ffaaff", "ff9fff", "ff95ff", "ff8bff", "ff81ff",  # 55-59
    "ff77ff", "ff6cff", "ff62ff", "ff58ff", "ff4eff",  # 60-64
]
PALETTE_HEX = {15 + i: h for i, h in enumerate(_UB_15_TO_64)}
PALETTE_HEX.update({d: "ffffff" for d in range(65, 75)})
PALETTE_HEX.update({d: "00ff00" for d in range(75, 96)})
PAL_DBZ = np.array(sorted(PALETTE_HEX), dtype=np.float32)
PAL_RGB = np.array([[int(PALETTE_HEX[d][i:i + 2], 16) for i in (0, 2, 4)] for d in sorted(PALETTE_HEX)],
                   dtype=np.int32)
ALPHA_MIN = 200          # פיקסלים שקופים-למחצה (מתחת ל-15 dBZ) מתעלמים מהם
PAL_TOLERANCE = 20       # מרחק צבע מקסימלי להתאמה לפלטה (RGB)

COMPASS_HE = ["צפון", "צפון-מזרח", "מזרח", "דרום-מזרח", "דרום", "דרום-מערב", "מערב", "צפון-מערב"]


class AnalysisFailed(Exception):
    """חריגה ייעודית לכשל בעיבוד או בתקשורת"""
    pass


# ------------------------------------------------------------------ ניהול State
def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"שגיאה בטעינת קובץ state: {e}")
        return {}


def save_state(state):
    tmp = f"{STATE_FILE}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        print(f"שגיאה בשמירת קובץ state: {e}")


def severity(result):
    dbz = result.get("core_dbz", DBZ_MIN)
    if dbz >= 40:
        return 3
    if dbz >= 30:
        return 2
    return 1


def should_send(state, result, now):
    last_str = state.get("last_alert_time")
    if not last_str:
        return True, "התראה ראשונה"

    try:
        last_time = datetime.fromisoformat(last_str)
    except ValueError:
        return True, "זמן התראה קודמת לא תקין"

    elapsed = (now - last_time).total_seconds() / 60.0
    if elapsed >= COOLDOWN_MINUTES:
        return True, f"עברו {elapsed:.0f} דקות (מעבר ל-Cooldown)"

    last_sev = state.get("last_alert_severity", 1)
    curr_sev = severity(result)
    if curr_sev > last_sev:
        return True, f"חומרת הגשם עלתה ({last_sev} -> {curr_sev})"

    return False, f"בתוך Cooldown ({elapsed:.0f}/{COOLDOWN_MINUTES} דקות)"


def handle_success(state):
    state["consecutive_failures"] = 0


def handle_failure(state, err_msg):
    fails = state.get("consecutive_failures", 0) + 1
    state["consecutive_failures"] = fails
    state["last_failure_reason"] = err_msg
    save_state(state)


def note_quiet(state, now):
    if "quiet_since" not in state:
        state["quiet_since"] = now.isoformat()


# ------------------------------------------------------------------ הורדה ופענוח
def new_session():
    s = requests.Session()
    s.headers["User-Agent"] = "givat-shmuel-rain-alert/2.4"
    return s


def get_frames(session):
    try:
        res = session.get(WEATHER_MAPS_URL, timeout=(5, 15))
        if res.status_code != 200:
            raise AnalysisFailed(f"שגיאה בקבלת רשימת הפריימים: סטטוס {res.status_code}")
        data = res.json()
        return data["host"], data["radar"]["past"]
    except Exception as e:
        if isinstance(e, AnalysisFailed):
            raise
        raise AnalysisFailed(f"שגיאה בתקשורת לקבלת פריימים: {type(e).__name__}")


def fetch_tile(session, url):
    try:
        res = session.get(url, timeout=(5, 15))
        if res.status_code != 200:
            return None
        return Image.open(io.BytesIO(res.content)).convert("RGBA")
    except Exception:
        return None


def decode_tile(img):
    arr = np.asarray(img)
    alpha = arr[..., 3]
    dbz = np.full(alpha.shape, np.nan, dtype=np.float32)
    valid = alpha >= ALPHA_MIN
    if not valid.any():
        return dbz, 0, 0

    rgb = arr[..., :3].astype(np.int32)
    packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    uniq, inv = np.unique(packed[valid], return_inverse=True)
    inv = inv.ravel()
    ucolors = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1)
    dist2 = ((ucolors[:, None, :] - PAL_RGB[None, :, :]) ** 2).sum(axis=2)
    best = dist2.argmin(axis=1)
    bad_colors = dist2.min(axis=1) > PAL_TOLERANCE ** 2
    dbz[valid] = PAL_DBZ[best][inv]
    counts = np.bincount(inv, minlength=len(uniq))
    return dbz, int(valid.sum()), int(counts[bad_colors].sum())


def build_grid_geometry():
    n = 2 * (GRID_RADIUS_KM // GRID_STEP_KM) + 1
    east = (np.arange(n) - n // 2) * GRID_STEP_KM
    north = (n // 2 - np.arange(n)) * GRID_STEP_KM
    E, N = np.meshgrid(east, north)
    lat = SITE_LAT + N / KM_PER_DEG_LAT
    lon = SITE_LON + E / (KM_PER_DEG_LAT * math.cos(math.radians(SITE_LAT)))
    scale = TILE_SIZE * 2 ** ZOOM
    px = (lon + 180.0) / 360.0 * scale
    s = np.sin(np.radians(lat))
    py = (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * scale
    return E, N, px, py


def frame_to_grid(session, host, frame, px, py):
    tx = np.floor(px / TILE_SIZE).astype(int)
    ty = np.floor(py / TILE_SIZE).astype(int)
    grid = np.full(px.shape, np.nan, dtype=np.float32)
    total_valid = total_bad = 0
    for x, y in sorted(set(zip(tx.ravel().tolist(), ty.ravel().tolist()))):
        url = f"{host}{frame['path']}/{TILE_SIZE}/{ZOOM}/{x}/{y}/{COLOR_SCHEME}/{TILE_OPTIONS}.png"
        img = fetch_tile(session, url)
        if img is None:
            return None
        dbz, valid, bad = decode_tile(img)
        total_valid += valid
        total_bad += bad
        h, w = dbz.shape
        m = (tx == x) & (ty == y)
        ix = np.clip(((px[m] - x * TILE_SIZE) / TILE_SIZE * w).astype(int), 0, w - 1)
        iy = np.clip(((py[m] - y * TILE_SIZE) / TILE_SIZE * h).astype(int), 0, h - 1)
        grid[m] = dbz[iy, ix]
    return grid, total_valid, total_bad


# ------------------------------------------------------------------ ניתוח
def echo_mask(dbz):
    echo = np.nan_to_num(dbz, nan=-99.0) >= DBZ_MIN
    p = np.pad(echo, 1)
    n0, n1 = echo.shape
    cnt = sum(p[1 + dy:1 + dy + n0, 1 + dx:1 + dx + n1].astype(int)
              for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return echo & (cnt >= 3)


def _field(dbz, mask):
    return np.where(mask, np.clip(np.nan_to_num(dbz, nan=0.0) - DBZ_MIN + 1.0, 1.0, 20.0), 0.0)


def _ncc(a_full, b_full, sy, sx):
    n = a_full.shape[0]
    y0, y1 = max(0, -sy), min(n, n - sy)
    x0, x1 = max(0, -sx), min(n, n - sx)
    a = a_full[y0:y1, x0:x1]
    b = b_full[y0 + sy:y1 + sy, x0 + sx:x1 + sx]
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / den if den > 0 else 0.0


def _refine(sub, iy, ix):
    def off(m1, c, p1):
        d = m1 - 2 * c + p1
        return 0.0 if d >= 0 else float(np.clip(0.5 * (m1 - p1) / d, -0.5, 0.5))
    oy = off(sub[iy - 1, ix], sub[iy, ix], sub[iy + 1, ix]) if 0 < iy < sub.shape[0] - 1 else 0.0
    ox = off(sub[iy, ix - 1], sub[iy, ix], sub[iy, ix + 1]) if 0 < ix < sub.shape[1] - 1 else 0.0
    return oy, ox


def estimate_motion(old_dbz, old_mask, new_dbz, new_mask, dt_hours):
    if old_mask.sum() < MIN_ECHO_CELLS or new_mask.sum() < MIN_ECHO_CELLS:
        return None
    A, B = _field(old_dbz, old_mask), _field(new_dbz, new_mask)
    n = A.shape[0]
    m = min(int(math.ceil(MAX_SPEED_KMH * dt_hours / GRID_STEP_KM)), n // 4)
    size = 2 * n
    c = np.fft.irfft2(np.conj(np.fft.rfft2(A, s=(size, size))) * np.fft.rfft2(B, s=(size, size)),
                      s=(size, size))
    sh = np.arange(-m, m + 1)
    sub = c[np.ix_(sh % size, sh % size)] / np.outer(n - np.abs(sh), n - np.abs(sh))
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    oy, ox = _refine(sub, iy, ix)
    sy, sx = sh[iy] + oy, sh[ix] + ox
    corr = _ncc(A, B, int(round(sh[iy])), int(round(sh[ix])))
    u = sx * GRID_STEP_KM / dt_hours
    v = -sy * GRID_STEP_KM / dt_hours
    return u, v, corr


def _idx(e_km, n_km, n):
    col = int(round(e_km / GRID_STEP_KM)) + n // 2
    row = n // 2 - int(round(n_km / GRID_STEP_KM))
    return (row, col) if (0 <= row < n and 0 <= col < n) else None


def project_eta(mask, dbz, u, v):
    speed = math.hypot(u, v)
    mx, my = u / speed, v / speed
    qx, qy = -my, mx
    n = mask.shape[0]
    offsets = np.linspace(-CORRIDOR_HALF_WIDTH_KM, CORRIDOR_HALF_WIDTH_KM, 5)
    d_max = min(GRID_RADIUS_KM * 0.95, speed * ALERT_MAX_ETA_MIN / 60.0 + 10.0)

    def points(d):
        for o in offsets:
            yield _idx(-mx * d + qx * o, -my * d + qy * o, n)

    d_front = None
    for d in np.arange(0, d_max + 1e-9, GRID_STEP_KM):
        if any(i is not None and mask[i] for i in points(d)):
            d_front = float(d)
            break
    if d_front is None:
        return None

    vals = [dbz[i] for d in np.arange(d_front, d_front + 20.0 + 1e-9, GRID_STEP_KM)
            for i in points(d) if i is not None and mask[i]]
    core = float(np.percentile(vals, 90)) if vals else DBZ_MIN
    return d_front, core, d_front / speed * 60.0


def dbz_to_rain_mmh(dbz):
    return (10 ** (dbz / 10.0) / 200.0) ** (1 / 1.6)


def compass(bearing):
    return COMPASS_HE[int(((bearing % 360) + 22.5) // 45) % 8]


def nearest_echo(mask, E, N):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    dist = np.hypot(E[ys, xs], N[ys, xs])
    k = int(dist.argmin())
    bearing = math.degrees(math.atan2(E[ys[k], xs[k]], N[ys[k], xs[k]])) % 360
    return float(dist[k]), bearing


def confidence_from(corr, lag):
    level = 2 if corr >= 0.7 else 1 if corr >= 0.5 else 0
    if lag < 2:
        level -= 1
    return ["נמוכה 🔴", "בינונית 🟡", "גבוהה 🟢"][max(level, 0)]


# ------------------------------------------------------------------ הודעה ושליחה
def intensity_text(core_dbz):
    rate = dbz_to_rain_mmh(core_dbz)
    label = ("🌧️ <b>גשם חזק</b>" if core_dbz >= 40 else
             "🌧️ <b>גשם בינוני</b>" if core_dbz >= 30 else "🌦️ <b>גשם קל</b>")
    text = f"{label} (בליבה כ-{rate:.1f} מ\"מ/ש', הערכה גסה)"
    if core_dbz >= 45:
        text += " ⛈️ ייתכנו רעמים"
    return text


def build_message(r):
    if r["kind"] == "approach":
        if r["eta_min"] <= 2:
            eta = "<b>כבר עכשיו / בדקות הקרובות</b>"
        else:
            eta = f"כ-<b>{r['eta_min']} דקות</b> (סביב {r['arrival']})"
        motion = f"מגיע מכיוון {r['from_dir']}, כ-{r['speed']:.0f} קמ\"ש"
        title = "🚨 <b>גשם מתקרב לגבעת שמואל!</b>"
    else:
        eta = "לא ידוע (כיוון התנועה לא נמדד)"
        motion = f"ההד נמצא בכיוון {r['from_dir']}; מגמת התנועה לא ברורה"
        title = "⚠️ <b>הד גשם קרוב לגבעת שמואל</b>"

    return (
        f"{title}\n\n"
        f"📊 <b>עוצמה:</b> {intensity_text(r['core_dbz'])}\n"
        f"📏 <b>מרחק חזית:</b> כ-<b>{r['front_km']:.0f} ק\"מ</b>\n"
        f"🧭 <b>תנועה:</b> {motion}\n"
        f"⏱️ <b>זמן הגעה (ETA):</b> {eta}\n"
        f"🎯 <b>רמת ביטחון:</b> {r['confidence']}\n\n"
        f"🛰️ <i>מכ\"ם RainViewer, פריים של {r['frame_time']}. "
        f"השתקה ל-{COOLDOWN_MINUTES} דקות.</i>"
    )


def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            print(f"שגיאה בשליחה לטלגרם: {res.status_code} {res.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"שגיאה בשליחה לטלגרם: {type(e).__name__}")
        return False


def save_debug_image(path, dbz, mask, velocity=None):
    n, k = dbz.shape[0], 4
    v = np.nan_to_num(dbz, nan=-99.0)
    rgb = np.full((n, n, 3), 25, np.uint8)
    rgb[(v >= 15) & ~mask] = (70, 70, 95)
    rgb[mask & (v < 30)] = (80, 170, 255)
    rgb[mask & (v >= 30) & (v < 40)] = (255, 220, 0)
    rgb[mask & (v >= 40)] = (255, 60, 0)
    img = Image.fromarray(rgb).resize((n * k, n * k), Image.NEAREST)
    d, c = ImageDraw.Draw(img), (n // 2) * k + k // 2
    for r_km in (20, 40, 60, 80):
        r = r_km / GRID_STEP_KM * k
        d.ellipse([c - r, c - r, c + r, c + r], outline=(255, 255, 255))
    d.line([c - 8, c, c + 8, c], fill=(0, 255, 0), width=3)
    d.line([c, c - 8, c, c + 8], fill=(0, 255, 0), width=3)
    if velocity:
        sp = math.hypot(*velocity)
        if sp > 0:
            L = 60 / GRID_STEP_KM * k
            d.line([c, c, c - velocity[0] / sp * L, c + velocity[1] / sp * L], fill=(0, 255, 0), width=2)
    img.save(path)
    print(f"נשמרה תמונת דיבוג: {path}")


# ------------------------------------------------------------------ ראשי
def analyze(session, args):
    host, past = get_frames(session)
    frames = past[-FRAMES_N:]
    if len(frames) < 2:
        print("אין מספיק פריימים.")
        return None

    now_ts = datetime.now(TZ_ISRAEL).timestamp()
    age_min = (now_ts - frames[-1]["time"]) / 60
    if age_min > MAX_FRAME_AGE_MIN:
        print(f"הפריים האחרון ישן ({age_min:.0f} דקות). מדלג.")
        return None

    E, N, px, py = build_grid_geometry()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda f: frame_to_grid(session, host, f, px, py), frames))

    if results[-1] is None:
        raise AnalysisFailed("הורדת הפריים האחרון נכשלה.")

    good = [(f, r) for f, r in zip(frames, results) if r is not None]
    if len(good) < len(frames):
        print(f"{len(frames) - len(good)} פריימים לא ירדו, ממשיך עם {len(good)}.")

    grids = [r[0] for _, r in good]
    masks = [echo_mask(g) for g in grids]
    new_dbz, new_mask = grids[-1], masks[-1]

    _, valid_px, bad_px = good[-1][1]
    if valid_px and bad_px / valid_px > 0.3:
        print(f"אזהרה: {bad_px / valid_px:.0%} מהפיקסלים לא תואמים לפלטה. ייתכן שהסכמה השתנתה.")

    echo_cells = int(new_mask.sum())
    near = nearest_echo(new_mask, E, N)
    print(f"פריים אחרון: {datetime.fromtimestamp(good[-1][0]['time'], TZ_ISRAEL):%H:%M} | "
          f"תאי הד: {echo_cells} | הד קרוב: "
          f"{'אין' if near is None else f'{near[0]:.0f} ק\"מ'}")

    velocity = None
    motion = None
    lag = 0
    if echo_cells >= MIN_ECHO_CELLS and len(good) >= 2:
        lag = min(3, len(good) - 1)
        t_old, t_new = good[-1 - lag][0]["time"], good[-1][0]["time"]
        dt_h = (t_new - t_old) / 3600.0
        if dt_h <= 0:
            dt_h = lag * 10 / 60.0
        motion = estimate_motion(grids[-1 - lag], masks[-1 - lag], new_dbz, new_mask, dt_h)
        if motion:
            u, v, corr = motion
            print(f"תנועה: {math.hypot(u, v):.0f} קמ\"ש לכיוון {compass(math.degrees(math.atan2(u, v)))} | "
                  f"מתאם {corr:.2f} | פער {lag * 10} דק'")
            if corr >= MIN_CORR and math.hypot(u, v) >= MIN_SPEED_KMH:
                velocity = (u, v)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        save_debug_image(os.path.join(args.debug_dir, "latest.png"), new_dbz, new_mask, velocity)

    frame_time = f"{datetime.fromtimestamp(good[-1][0]['time'], TZ_ISRAEL):%H:%M}"

    if velocity:
        u, v = velocity
        speed = math.hypot(u, v)
        proj = project_eta(new_mask, new_dbz, u, v)
        if proj is None:
            print("ההד לא מתקדם לעבר גבעת שמואל בטווח הזמן הרלוונטי.")
            return None
        front_km, core, raw_eta_min = proj
        
        # פיצוי על גיל הפריים עבור ה-ETA
        adjusted_eta_min = raw_eta_min - age_min
        if adjusted_eta_min <= 0:
            adjusted_eta_min = 0

        if adjusted_eta_min > ALERT_MAX_ETA_MIN:
            print(f"ETA מתוקן של {adjusted_eta_min:.0f} דקות, מעבר לסף.")
            return None
            
        eta_min = int(round(adjusted_eta_min))
        arrival_dt = datetime.now(TZ_ISRAEL) + timedelta(minutes=eta_min)
        
        return {
            "kind": "approach", "front_km": front_km, "core_dbz": core, "eta_min": eta_min,
            "arrival": arrival_dt.strftime("%H:%M"),
            "speed": speed, "from_dir": compass(math.degrees(math.atan2(u, v)) + 180),
            "confidence": confidence_from(motion[2], lag), "frame_time": frame_time,
        }

    if near and near[0] <= NEAR_ECHO_KM:
        vals = new_dbz[new_mask & (np.hypot(E, N) <= near[0] + 20)]
        core = float(np.percentile(vals, 90)) if vals.size else DBZ_MIN
        return {"kind": "near", "front_km": near[0], "core_dbz": core, "from_dir": compass(near[1]),
                "confidence": "נמוכה 🔴", "frame_time": frame_time}

    print("אין התראה: אין הד שמתקרב, ואין הד קרוב מאוד.")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="מנתח ומדפיס, בלי לשלוח הודעה ובלי להפעיל השתקה")
    ap.add_argument("--debug-dir", help="תיקייה לשמירת תמונת דיבוג של הפריים האחרון")
    args = ap.parse_args(argv)

    if not args.dry_run:
        missing = [n for n, v in [("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
                                  ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)] if not v]
        if missing:
            print(f"חסרים משתני סביבה: {', '.join(missing)}")
            sys.exit(1)

    state = load_state()
    now = datetime.now(TZ_ISRAEL)

    with new_session() as session:
        try:
            result = analyze(session, args)
            handle_success(state)
        except AnalysisFailed as e:
            print(f"ניתוח נכשל: {e}")
            if not args.dry_run:
                handle_failure(state, str(e))
            sys.exit(1)
        except Exception as e:
            print(f"חריגה לא צפויה: {type(e).__name__}: {e}")
            traceback.print_exc()
            if not args.dry_run:
                handle_failure(state, f"{type(e).__name__}: {e}")
            sys.exit(1)

    if result is None:
        note_quiet(state, now)
        if not args.dry_run:
            save_state(state)
        return

    msg = build_message(result)
    print("\n--- הודעה מתוכננת ---")
    print(msg)
    print("----------------------\n")

    if args.dry_run:
        print("מצב --dry-run פעיל: ההודעה לא נשלחה והמצב לא נשמר.")
        return

    send_it, reason = should_send(state, result, now)
    if send_it:
        print(f"שולח התראה לטלגרם (סיבה: {reason})...")
        if send_telegram_message(msg):
            state["last_alert_time"] = now.isoformat()
            state["last_alert_severity"] = severity(result)
            state.pop("quiet_since", None)
            save_state(state)
            print("ההתראה נשלחה בהצלחה.")
        else:
            print("נכשלה שליחת ההתראה לטלגרם.")
    else:
        print(f"התראה דולגה ({reason}).")
        save_state(state)


if __name__ == "__main__":
    main()
