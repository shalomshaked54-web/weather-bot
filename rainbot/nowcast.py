"""3.2: חיזוי לנקודה - "בעוד כמה דקות יתחיל לרדת גשם אצלי".

עד 3.2 זמן ההגעה חושב לפי הגשם הראשון שנכנס לרדיוס של 5+ ק"מ מהגבעה, עם וקטור תנועה אחד לכל המכ"ם.
כאן:
  1. שדה תנועה מקומי: וקטור לכל אזור של ~48 ק"מ (מערכת גדולה לא זזה באותה מהירות בכל חלקיה).
     אזור שהאומדן שלו לא אמין או סותר את התנועה הכללית מקבל את התנועה הכללית.
  2. הרצה אחורה (semi-Lagrangian) מהנקודה עצמה: איפה נמצא עכשיו הגשם שיהיה מעל הגבעה בעוד t דקות.
  3. אנסמבל: 40 תרחישים עם שינויים קטנים במהירות ובכיוון (יותר פיזור כשהתנועה פחות אמינה).
     התוצאה: סיכוי לגשם בשעתיים הקרובות, טווח זמני ההתחלה (10%-90%), וכמה זמן הגשם יימשך.
"""
import math

import numpy as np

from .config import (CELL_KM, MAX_LEAD_MIN, NC_ANGLE_SIGMA_DEG, NC_BLOCK_CELLS, NC_BLOCK_MAX_ANGLE_DEG,
                     NC_BLOCK_MAX_DEV_KMH, NC_BLOCK_MAX_DEV_RATIO, NC_BLOCK_MIN_RAIN,
                     NC_BLOCK_MIN_Q, NC_BLOCK_SIGMA_KM, NC_BLOCK_STRIDE, NC_GLOBAL_WEIGHT, NC_HIT_KM, NC_MEMBERS,
                     NC_SPEED_SIGMA, X_RANGE, Y_RANGE)
from .radar import XX, YY

NY, NX = XX.shape


def _idx(x, y):
    i = int(round((x - X_RANGE[0] - CELL_KM / 2) / CELL_KM))
    j = int(round((y - Y_RANGE[0] - CELL_KM / 2) / CELL_KM))
    return j, i


def _inside(j, i):
    return 0 <= j < NY and 0 <= i < NX


def dilate(mask, radius_km):
    """mask מורחב: True בכל תא שיש בו גשם במרחק radius_km"""
    r = int(math.ceil(radius_km / CELL_KM))
    out = np.zeros_like(mask)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx * dx + dy * dy) * CELL_KM ** 2 > radius_km ** 2 + 1e-6:
                continue
            ys = slice(max(dy, 0), NY + min(dy, 0))
            yd = slice(max(-dy, 0), NY + min(-dy, 0))
            xs = slice(max(dx, 0), NX + min(dx, 0))
            xd = slice(max(-dx, 0), NX + min(-dx, 0))
            out[yd, xd] |= mask[ys, xs]
    return out


def dilate_max(field, radius_km):
    """המקסימום של field ברדיוס radius_km מכל תא"""
    r = int(math.ceil(radius_km / CELL_KM))
    out = np.zeros_like(field)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx * dx + dy * dy) * CELL_KM ** 2 > radius_km ** 2 + 1e-6:
                continue
            ys = slice(max(dy, 0), NY + min(dy, 0))
            yd = slice(max(-dy, 0), NY + min(-dy, 0))
            xs = slice(max(dx, 0), NX + min(dx, 0))
            xd = slice(max(-dx, 0), NX + min(-dx, 0))
            np.maximum(out[yd, xd], field[ys, xs], out=out[yd, xd])
    return out


# ------------------------------------------------------------------ שדה תנועה מקומי

def _block_shift(a, b, max_shift):
    from .analysis import xcorr_shift   # ייבוא מאוחר: analysis מייבא את המודול הזה
    return xcorr_shift(a, b, max_shift)


def _shift(field, dy, dx):
    """מזיז שדה ב-(dy, dx) תאים שלמים, עם אפסים בשוליים"""
    out = np.zeros_like(field)
    ys, yd = slice(max(-dy, 0), NY + min(-dy, 0)), slice(max(dy, 0), NY + min(dy, 0))
    xs, xd = slice(max(-dx, 0), NX + min(-dx, 0)), slice(max(dx, 0), NX + min(dx, 0))
    out[yd, xd] = field[ys, xs]
    return out


def local_motion(weights, times, gmotion):
    """וקטורי תנועה לאזורים. מחזיר רשימה של (x, y, vx, vy, q) לאזורים שהאומדן שלהם התקבל.
    הפריים הישן מוזז קודם לפי התנועה הכללית, ובכל אזור מחפשים רק סטייה מוגבלת ממנה. כך אזור לא
    "נתפס" למערכת אחרת שנמצאת לידו"""
    if not gmotion or len(weights) < 2:
        return []
    old, new = weights[0], weights[-1]
    dt_h = (times[-1] - times[0]) / 3600.0
    if dt_h <= 0:
        return []
    gs = gmotion["speed"]
    gdx, gdy = gmotion["vx"] * dt_h / CELL_KM, gmotion["vy"] * dt_h / CELL_KM
    sdx, sdy = int(round(gdx)), int(round(gdy))
    pre = _shift(old, sdy, sdx)
    dev = int(math.ceil(max(NC_BLOCK_MAX_DEV_KMH, NC_BLOCK_MAX_DEV_RATIO * gs) * dt_h / CELL_KM)) + 1
    out = []
    half = NC_BLOCK_CELLS // 2
    for j0 in range(0, NY - half, NC_BLOCK_STRIDE):
        for i0 in range(0, NX - half, NC_BLOCK_STRIDE):
            j1, i1 = min(j0 + NC_BLOCK_CELLS, NY), min(i0 + NC_BLOCK_CELLS, NX)
            b = np.zeros_like(new)
            b[j0:j1, i0:i1] = new[j0:j1, i0:i1]
            if (b > 0).sum() < NC_BLOCK_MIN_RAIN:
                continue
            a = np.zeros_like(pre)
            ja, jb, ia, ib = max(j0 - dev, 0), min(j1 + dev, NY), max(i0 - dev, 0), min(i1 + dev, NX)
            a[ja:jb, ia:ib] = pre[ja:jb, ia:ib]
            if (a > 0).sum() < NC_BLOCK_MIN_RAIN:
                continue
            dy, dx, q = _block_shift(a, b, dev)
            if q < NC_BLOCK_MIN_Q or abs(dy) >= dev or abs(dx) >= dev:
                continue      # התאמה חלשה, או שהשיא בקצה החיפוש (כנראה לא אמיתי)
            vx, vy = (sdx + dx) * CELL_KM / dt_h, (sdy + dy) * CELL_KM / dt_h
            if gs >= 10.0 and math.hypot(vx, vy) >= 10.0:
                ang = abs((math.degrees(math.atan2(vx, vy) - math.atan2(gmotion["vx"], gmotion["vy"])) + 180)
                          % 360 - 180)
                if ang > NC_BLOCK_MAX_ANGLE_DEG:
                    continue
            jc, ic = (j0 + j1 - 1) / 2.0, (i0 + i1 - 1) / 2.0
            x = X_RANGE[0] + CELL_KM / 2 + ic * CELL_KM
            y = Y_RANGE[0] + CELL_KM / 2 + jc * CELL_KM
            out.append((x, y, vx, vy, q))
    return out


def motion_field(blocks, gmotion):
    """שדה תנועה על כל הרשת: ממוצע משוקלל (גאוסיאני) של האזורים, עם התנועה הכללית כבסיס"""
    w0 = NC_GLOBAL_WEIGHT
    sx = np.full(XX.shape, w0 * gmotion["vx"])
    sy = np.full(XX.shape, w0 * gmotion["vy"])
    sw = np.full(XX.shape, w0)
    for x, y, vx, vy, q in blocks:
        w = q * np.exp(-((XX - x) ** 2 + (YY - y) ** 2) / (2 * NC_BLOCK_SIGMA_KM ** 2))
        sx += w * vx
        sy += w * vy
        sw += w
    return sx / sw, sy / sw


# ------------------------------------------------------------------ אנסמבל

def _trace(wet, dbz, vxf, vyf, factor, rot, lead_min):
    """הרצה אחורה של תרחיש אחד מהנקודה (0, 0). מחזיר רשימה לכל דקה: dBZ (0 = יבש), או None מחוץ לרשת"""
    c, s = math.cos(rot), math.sin(rot)
    x = y = 0.0
    out = []
    for _ in range(lead_min):
        j, i = _idx(x, y)
        if not _inside(j, i):
            out.append(None)
            break
        vx, vy = vxf[j, i] * factor, vyf[j, i] * factor
        vx, vy = c * vx + s * vy, -s * vx + c * vy     # סיבוב בזווית rot (חיובי = עם כיוון השעון)
        x -= vx / 60.0
        y -= vy / 60.0
        j, i = _idx(x, y)
        if not _inside(j, i):
            out.append(None)
            break
        out.append(float(dbz[j, i]) if wet[j, i] else 0.0)
    return out


def point_nowcast(cleaned, weights, times, gmotion, lead_min=MAX_LEAD_MIN, members=NC_MEMBERS, seed=7, spread=1.0,
                  rot_deg=0.0):
    """חיזוי לגבעה עצמה. cleaned: השדה הנקי של הפריים האחרון. מחזיר None אם אין תנועה אמינה, אחרת:
    {"p": סיכוי לגשם בתוך lead_min דקות, "eta": (p10, p90), "eta_mid": חציון, "dur": משך חציוני בדקות,
     "peak_dbz": עוצמה חציונית בחצי השעה הראשונה, "n": תרחישים, "blocks": אזורים עם תנועה מקומית}"""
    if not gmotion or gmotion["speed"] < 5.0:
        return None
    blocks = local_motion(weights, times, gmotion)
    vxf, vyf = motion_field(blocks, gmotion)
    wet = dilate(cleaned > 0, NC_HIT_KM)
    dbz = dilate_max(cleaned, NC_HIT_KM)
    q = float(gmotion.get("quality", 0.5))
    if not gmotion.get("consistent"):
        q *= 0.5
    sp_sig = (NC_SPEED_SIGMA + 0.25 * (1 - q)) * spread         # spread: כיול מהלמידה (learn.py)
    an_sig = math.radians(NC_ANGLE_SIGMA_DEG + 12.0 * (1 - q)) * spread
    rot0 = math.radians(rot_deg)        # 3.2: סטייה קבועה בכיוון התנועה שנלמדה (verify.py)
    rng = np.random.default_rng(seed)
    onsets, durs, peaks = [], [], []
    for k in range(members):
        if k == 0:
            f, r = 1.0, rot0      # התרחיש הראשון: בלי שינוי (מלבד הסיבוב שנלמד)
        else:
            f = float(np.clip(rng.normal(1.0, sp_sig), 0.5, 1.6))
            r = float(rng.normal(rot0, an_sig))
        tr = _trace(wet, dbz, vxf, vyf, f, r, lead_min)
        on = next((t for t, v in enumerate(tr, 1) if v), None)
        if on is None:
            continue
        onsets.append(on)
        after = tr[on - 1:]
        dry_run, dur = 0, 0
        for t, v in enumerate(after):     # משך: עד הפסקה של 10 דקות רצופות (או סוף החיזוי)
            if v:
                dur, dry_run = t + 1, 0
            else:
                dry_run += 1
                if dry_run >= 10:
                    break
        durs.append(dur)
        peaks.append(max((v or 0.0) for v in after[:30]))
    out = {"n": members, "blocks": len(blocks), "p": round(len(onsets) / members, 2),
           "eta": None, "eta_mid": None, "dur": None, "peak_dbz": None}
    if onsets:
        o = np.array(onsets, float)
        out.update(eta=(int(np.percentile(o, 10)), int(math.ceil(np.percentile(o, 90)))),
                   eta_mid=int(round(np.median(o))), dur=int(round(np.median(durs))),
                   peak_dbz=float(np.median(peaks)))
    return out
