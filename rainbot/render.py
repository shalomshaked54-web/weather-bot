"""תמונת מכ"ם לטלגרם ולבדיקה"""
import io
import math

import numpy as np
from PIL import Image, ImageDraw

from .config import CELL_KM, KM_PER_DEG, MAX_LEAD_MIN, MIN_SPEED_KMH, TARGET_LAT, TARGET_LON, VERSION, X_RANGE, Y_RANGE

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
