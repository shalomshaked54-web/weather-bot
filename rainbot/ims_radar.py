"""3.2: מקור מכ"ם שני - מכ"ם השירות המטאורולוגי (ims.gov.il).

המכ"ם הישראלי עצמו, פריים כל 5 דקות (RainViewer: כל 10). תמונת PNG אחת לכל פריים (940x940),
מוצגת באתר מעל מפה ב-Leaflet בגבולות קבועים, ולכן לינארית בהיטל מרקטור. הצבעים הם מעבר רציף בין
נקודות סרגל הצבעים של האתר (מ"מ/שעה), ואנחנו מתרגמים צבע -> מ"מ/שעה -> dBZ (Marshall-Palmer),
כך שכל שאר הבוט עובד בדיוק כמו עם RainViewer.

ברירת מחדל: גיבוי. כש-RainViewer לא זמין או ישן, הבוט ממשיך עם המכ"ם הזה.
RADAR_PRIMARY=ims הופך אותו למקור הראשי. בכל ריצה נרשמת השוואה קצרה בין שני המקורות (ims_cmp),
כדי שנוכל להחליט על סמך נתונים איזה מהם טוב יותר.
"""
import io
import math
from datetime import datetime

import numpy as np
from PIL import Image

from .config import (IMS_RADAR_BASE, IMS_RADAR_BOUNDS, IMS_RADAR_INDEX, IMS_RADAR_STEP_MIN, MIN_FRAMES, N_FRAMES,
                     TZ)
from .radar import XX, YY, http_get

# סרגל הצבעים של האתר (radar_colorbar): מ"מ/שעה -> צבע, מהחלש לחזק
COLORBAR = [(0.1, "0165ef"), (0.2, "00c3c9"), (0.7, "00a09b"), (1.2, "008c4d"), (2.0, "00b02c"), (4.0, "01d31c"),
            (6.0, "12f218"), (9.0, "7cff21"), (13.0, "fefd19"), (18.0, "ffcf00"), (24.0, "ffa800"),
            (30.0, "ff7d01"), (40.0, "fb3f00"), (50.0, "e10a12"), (100.0, "d00078"), (200.0, "ff00fe")]
MAX_COLOR_DIST = 45.0
HEADERS = {"User-Agent": "Mozilla/5.0 (rain-alert-bot)", "Accept": "application/json, image/png, */*",
           "Referer": "https://ims.gov.il/he/RadarSatellite"}


def _rgb(h):
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def _build_lut(samples=40):
    """נקודות לאורך הסרגל (מעבר לינארי בצבע, לוגריתמי בעוצמה) -> (צבעים, מ"מ/שעה)"""
    cols, vals = [], []
    for (v0, c0), (v1, c1) in zip(COLORBAR[:-1], COLORBAR[1:]):
        a, b = np.array(_rgb(c0), float), np.array(_rgb(c1), float)
        for k in range(samples):
            f = k / samples
            cols.append(a + (b - a) * f)
            vals.append(math.exp(math.log(v0) + (math.log(v1) - math.log(v0)) * f))
    cols.append(np.array(_rgb(COLORBAR[-1][1]), float))
    vals.append(COLORBAR[-1][0])
    return np.array(cols), np.array(vals)


LUT_COLORS, LUT_MMH = _build_lut()


def mmh_to_dbz(r):
    return 10.0 * np.log10(200.0 * np.power(np.maximum(r, 1e-6), 1.6))


def _merc(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def grid_pixels(w, h, bounds=IMS_RADAR_BOUNDS):
    """אינדקסי הפיקסלים של כל נקודה ברשת העבודה (או -1 מחוץ לתמונה)"""
    from .config import KM_PER_DEG, TARGET_LAT, TARGET_LON
    (lat_a, lon_a), (lat_b, lon_b) = bounds
    lat0, lat1 = min(lat_a, lat_b), max(lat_a, lat_b)
    lon0, lon1 = min(lon_a, lon_b), max(lon_a, lon_b)
    lon = TARGET_LON + XX / (KM_PER_DEG * math.cos(math.radians(TARGET_LAT)))
    lat = TARGET_LAT + YY / KM_PER_DEG
    px = np.floor((lon - lon0) / (lon1 - lon0) * w).astype(np.int64)
    py = np.floor((_merc(lat1) - _merc(lat)) / (_merc(lat1) - _merc(lat0)) * h).astype(np.int64)
    bad = (px < 0) | (px >= w) | (py < 0) | (py >= h)
    px[bad], py[bad] = -1, -1
    return py, px


def decode_image(png_bytes):
    """PNG של המכ"ם -> שדה dBZ על רשת העבודה (0 = אין גשם)"""
    img = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))
    h, w = img.shape[:2]
    py, px = grid_pixels(w, h)
    inside = px >= 0
    out = np.zeros(XX.shape, np.float32)
    pix = img[py[inside], px[inside]]
    opaque = pix[:, 3] >= 128
    if not opaque.any():
        return out
    rgb = pix[opaque, :3].astype(float)
    uniq, inv = np.unique(rgb, axis=0, return_inverse=True)
    d = ((uniq[:, None, :] - LUT_COLORS[None, :, :]) ** 2).sum(-1)
    best = d.argmin(1)
    dbz = np.where(d.min(1) <= MAX_COLOR_DIST ** 2, mmh_to_dbz(LUT_MMH[best]), 0.0)
    vals = np.zeros(len(pix), np.float32)
    vals[np.nonzero(opaque)[0]] = dbz[np.asarray(inv).ravel()]
    out[inside] = vals
    return out


def parse_index(data):
    """תשובת radar_satellite -> רשימת פריימים [{"time": epoch, "path"}] מהישן לחדש"""
    items = data["data"]["types"]["IMSRadar"]
    frames = {}
    for it in items or []:
        fn = it.get("file_name") or ""
        if not fn.endswith("_0.png"):
            continue
        t = datetime.strptime(it["forecast_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
        frames[int(t.timestamp())] = {"time": int(t.timestamp()), "path": fn}
    return [frames[k] for k in sorted(frames)]


def pick_frames(frames, step_min=IMS_RADAR_STEP_MIN, n=N_FRAMES):
    """הפריים האחרון ועוד n-1 פריימים כל step_min דקות אחורה (כמו RainViewer: 4 פריימים על 30 דקות)"""
    if not frames:
        return []
    by_t = {f["time"]: f for f in frames}
    last = frames[-1]["time"]
    out = [by_t[last - 60 * step_min * k] for k in range(n) if last - 60 * step_min * k in by_t]
    return sorted(out, key=lambda f: f["time"])


def get_fields(session):
    """מחזיר (שדות מהישן לחדש, פריימים). זורק RuntimeError בתקלה"""
    data = http_get(session, IMS_RADAR_INDEX, headers=HEADERS).json()
    frames = pick_frames(parse_index(data))
    if len(frames) < MIN_FRAMES:
        raise RuntimeError("פחות מ-3 פריימים במכ\"ם השירות המטאורולוגי")
    fields = [decode_image(http_get(session, IMS_RADAR_BASE + f["path"], headers=HEADERS).content) for f in frames]
    return fields, frames
