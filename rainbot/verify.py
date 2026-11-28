"""3.2: למידה מכל פריים - גם כשהגשם עוד לא הגיע אלייך, וגם מגשם חלש.

הלמידה ב-learn.py לומדת רק כשהגשם מתחיל ממש אצלך, וזה קורה כמה פעמים בסערה. כאן הבוט לומד בכל
פריים שיש בו גשם בטווח (גם 15 dBZ, טפטוף):
  1. כל פריים נשמר בקטן במצב (dBZ שלם, דחוס), יחד עם וקטור התנועה שהבוט חישב באותו רגע.
  2. אחרי 30 דקות, הבוט לוקח את הפריים הישן, מזיז אותו 30 דקות קדימה לפי התנועה שחישב אז, ובודק
     כמה זה דומה למה שהמכ"ם רואה עכשיו. הוא מנסה גם מהירויות וכיוונים קצת אחרים (60%-150%,
     ±20 מעלות), ומוצא מה היה נותן את ההתאמה הטובה ביותר.
  3. אם בעקביות "מהר יותר ב-10%" מתאים יותר, הבוט לומד שהתנועה שלו איטית מדי, וזמני ההגעה
     יתקצרו. אם בעקביות "10 מעלות ימינה", החיזוי לנקודה יסובב את התנועה בהתאם.
כל דגימה כזו שווה רבע מדגימה של "הגשם התחיל אצלך" (היא מודדת תנועה, לא את הגשם אצלך בדיוק).
"""
import base64
import math
import zlib
from datetime import datetime

import numpy as np

from .config import (CELL_KM, VERIFY_ANGLES, VERIFY_FACTORS, VERIFY_KEEP_FRAMES, VERIFY_LEAD_MIN, VERIFY_MIN_CELLS,
                     VERIFY_MIN_DBZ, VERIFY_MIN_Q, TZ)
from .radar import XX


def pack(field):
    q = np.clip(np.round(field), 0, 255).astype(np.uint8)
    return base64.b64encode(zlib.compress(q.tobytes(), 9)).decode()


def unpack(s):
    return np.frombuffer(zlib.decompress(base64.b64decode(s)), np.uint8).reshape(XX.shape).astype(np.float32)


def _weights(field):
    return np.where(field >= VERIFY_MIN_DBZ, np.clip(field - 10.0, 5.0, 25.0), 0.0)


def shift_bilinear(f, dx, dy):
    """מזיז שדה ב-(dx, dy) תאים (לא שלמים), אינטרפולציה בילינארית, אפסים מחוץ לרשת"""
    ny, nx = f.shape
    jj, ii = np.mgrid[0:ny, 0:nx].astype(np.float64)
    sx, sy = ii - dx, jj - dy
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = sx - x0, sy - y0
    out = np.zeros_like(f, dtype=np.float64)
    for oy, wy in ((0, 1 - fy), (1, fy)):
        for ox, wx in ((0, 1 - fx), (1, fx)):
            yy, xx = y0 + oy, x0 + ox
            ok = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
            v = np.zeros_like(out)
            v[ok] = f[yy[ok], xx[ok]]
            out += v * wy * wx
    return out


def _corr(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a * b).sum() / (na * nb)) if na > 0 and nb > 0 else 0.0


def best_fit(old, new, vx, vy, minutes):
    """איזו מהירות (פקטור) וכיוון (מעלות) היו מזיזים את old הכי קרוב ל-new. מחזיר (f, a, q, q0)"""
    a0, b = _weights(old), _weights(new)
    best = (1.0, 0.0, -1.0)
    q0 = None
    for ang in VERIFY_ANGLES:
        r = math.radians(ang)
        c, s = math.cos(r), math.sin(r)
        rx, ry = c * vx + s * vy, -s * vx + c * vy
        for f in VERIFY_FACTORS:
            dx = rx * f * minutes / 60.0 / CELL_KM
            dy = ry * f * minutes / 60.0 / CELL_KM
            q = _corr(shift_bilinear(a0, dx, dy), b)
            if abs(f - 1.0) < 1e-9 and ang == 0:
                q0 = q
            if q > best[2]:
                best = (f, float(ang), q)
    return best[0], best[1], best[2], q0


def step(lr, raw, radar_time, motion):
    """נקרא בכל פריים חדש. שומר את הפריים, ואם יש פריים מלפני 30 דקות - מחזיר דגימת אימות (או None)"""
    t = int(radar_time.timestamp())
    frames = [f for f in lr.get("frames", []) if 0 < t - f["t"] <= 60 * (VERIFY_LEAD_MIN + 40)]
    if any(f["t"] == t for f in lr.get("frames", [])):
        return None
    sample = None
    target = t - 60 * VERIFY_LEAD_MIN
    old = min((f for f in frames if abs(f["t"] - target) <= 180 and f.get("vx") is not None),
              key=lambda f: abs(f["t"] - target), default=None)
    if old is not None and (raw >= VERIFY_MIN_DBZ).sum() >= VERIFY_MIN_CELLS:
        f_old = unpack(old["f"])
        if (f_old >= VERIFY_MIN_DBZ).sum() >= VERIFY_MIN_CELLS and math.hypot(old["vx"], old["vy"]) >= 5.0:
            minutes = (t - old["t"]) / 60.0
            fac, ang, q, q0 = best_fit(f_old, raw, old["vx"], old["vy"], minutes)
            edge = fac in (VERIFY_FACTORS[0], VERIFY_FACTORS[-1])
            if q >= VERIFY_MIN_Q and not edge:
                sample = {"t": datetime.fromtimestamp(t, TZ).isoformat(timespec="minutes"), "f": round(fac, 2),
                          "a": ang, "q": round(q, 2), "q0": round(q0 or 0.0, 2), "v": round(math.hypot(
                              old["vx"], old["vy"]), 1)}
    keep = frames + [{"t": t, "f": pack(raw),
                      "vx": round(motion["vx"], 2) if motion else None,
                      "vy": round(motion["vy"], 2) if motion else None}]
    lr["frames"] = keep[-VERIFY_KEEP_FRAMES:]
    if sample:
        lr["verify"] = (lr.get("verify", []) + [sample])[-200:]
    return sample


def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2.0)])
