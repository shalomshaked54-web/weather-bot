"""RainViewer: רשת העבודה, הורדת אריחים ופענוח שלהם ל-dBZ"""
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

from .config import (API_URL, CELL_KM, COLOR, KM_PER_DEG, LUT_DBZ, LUT_RGB, MIN_FRAMES, N_FRAMES,
                     TARGET_LAT, TARGET_LON, TILE_SIZE, X_RANGE, Y_RANGE, ZOOM)


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


def http_get(session, url, tries=3, timeout=(5, 15)):
    last = "unknown"
    for attempt in range(tries):
        try:
            res = session.get(url, timeout=timeout)
            if res.status_code == 200:
                return res
            last = f"HTTP {res.status_code}"
        except requests.RequestException as e:
            last = type(e).__name__
        if attempt < tries - 1:
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
    if len(past) < MIN_FRAMES:
        raise RuntimeError("פחות מ-3 פריימי מכ\"ם זמינים")
    return data["host"], past[-N_FRAMES:]


def assemble(frames_tiles, n_frames):
    """{(f, tx, ty): tile או None} -> (שדות על הרשת המקומית, אינדקסים של הפריימים שנשמרו).
    3.1: פריים שחסר לו אריח אחד נזרק כולו (שדה חלקי היה משבש את חישוב התנועה)"""
    bad = {f for (f, _, _), t in frames_tiles.items() if t is None}
    keep = [f for f in range(n_frames) if f not in bad]
    if len(keep) < MIN_FRAMES:
        raise RuntimeError(f"רק {len(keep)} פריימים שלמים (חסרים אריחים)")
    h, w = (TY1 - TY0 + 1) * TILE_SIZE, (TX1 - TX0 + 1) * TILE_SIZE
    iy, ix = PY - TY0 * TILE_SIZE, PX - TX0 * TILE_SIZE
    fields = []
    for f in keep:
        mosaic = np.zeros((h, w), np.float32)
        for (ff, tx, ty), tile in frames_tiles.items():
            if ff == f:
                y0, x0 = (ty - TY0) * TILE_SIZE, (tx - TX0) * TILE_SIZE
                mosaic[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] = tile
        fields.append(mosaic[iy, ix])
    return fields, keep


def load_fields(session, host, frames):
    """מחזיר (שדות dBZ מהישן לחדש, הפריימים שנשמרו)"""
    jobs = [(f, tx, ty) for f in range(len(frames))
            for tx in range(TX0, TX1 + 1) for ty in range(TY0, TY1 + 1)]

    def get(job):
        f, tx, ty = job
        url = f"{host}{frames[f]['path']}/{TILE_SIZE}/{ZOOM}/{tx}/{ty}/{COLOR}/0_0.png"
        try:
            tile = decode_tile(http_get(session, url).content)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"אריח {tx}/{ty} בפריים {f} לא נטען: {e}")
            return None
        return tile if tile.shape == (TILE_SIZE, TILE_SIZE) else None

    with ThreadPoolExecutor(max_workers=8) as pool:
        tiles = list(pool.map(get, jobs))
    fields, keep = assemble(dict(zip(jobs, tiles)), len(frames))
    return fields, [frames[f] for f in keep]
