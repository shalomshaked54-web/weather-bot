import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

# --- הגדרות קבועות ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

OWM_URL = "https://api.openweathermap.org/data/2.5/weather"

GIVAT_SHMUEL_LAT = 32.0780
GIVAT_SHMUEL_LON = 34.8470

# רשת ממוקדת: ים גוש דן ותל אביב (20 נקודות)
# 20 נקודות x ריצה כל 30 דקות = 960 קריאות ביום (מתחת למכסת ה-1,000)
# הערה: שלוש הנקודות האחרונות בקו הצמוד לחוף הוזזו מערבה, כי בקואורדינטות
# המקוריות (34.76) הן נפלו ביבשה (בת ים / ראשון לציון). מומלץ לוודא על מפה.
SEA_GRID = [
    # קו עמוק (כ-30-35 ק"מ מהחוף)
    (32.12, 34.48), (32.07, 34.48), (32.02, 34.48), (31.96, 34.48),

    # קו ביניים (כ-20-25 ק"מ מהחוף)
    (32.18, 34.58), (32.12, 34.58), (32.07, 34.58), (32.02, 34.58), (31.96, 34.58),

    # קו ימי קרוב (כ-10-12 ק"מ מהחוף)
    (32.18, 34.68), (32.12, 34.68), (32.07, 34.68), (32.02, 34.68), (31.96, 34.68),

    # קו צמוד לחוף (כ-1-5 ק"מ מהחוף)
    (32.15, 34.76), (32.10, 34.76), (32.07, 34.76),
    (32.03, 34.72), (31.98, 34.69), (31.93, 34.66),
]

# --- קבצי מצב ---
# ב-GitHub Actions / סביבה אפמרית יש להצביע על תיקייה שנשמרת בין ריצות (cache / volume)
STATE_DIR = os.environ.get("STATE_DIR", ".")
COOLDOWN_FILE = os.path.join(STATE_DIR, "last_trigger.txt")
PREVIOUS_STATE_FILE = os.path.join(STATE_DIR, "previous_state.json")

COOLDOWN_MINUTES = 90
TZ_ISRAEL = ZoneInfo("Asia/Jerusalem")

# --- ספי החלטה ---
# הנקודה הקרובה ביותר ברשת נמצאת ~8.3 ק"מ מגבעת שמואל, לכן הסף חייב להיות מעליה
CLOSE_POINT_KM = 10.0
# התראה המבוססת על רוח בלבד (בלי תנועה מדודה) נשלחת רק אם החזית בטווח הזה
WIND_ONLY_MAX_FRONT_KM = 20.0
# מעקב תנועה: חלון זמן תקף בין שתי ריצות (שעות), והתקרבות מינימלית של החזית
MIN_TREND_HOURS = 0.25
MAX_TREND_HOURS = 1.25
APPROACH_MIN_KM = 3.0        # בערך צעד אחד ברשת; מתחת לזה זה רעש
MIN_SPEED_KMH = 5.0
WIND_ANGLE_TOLERANCE = 50    # מעלות
HEAVY_RADIUS_KM = 20.0       # עוצמה נמדדת רק בנקודות הקרובות
HEAVY_MM_PER_HOUR = 2.5
# אם יותר מ-25% מהקריאות נכשלו, הנתונים לא אמינים ולא נוגעים במצב השמור
MAX_FAILED_FRACTION = 0.25
MAX_WORKERS = 5              # מתחת ל-60 קריאות בדקה של החבילה החינמית

# טפטוף (3xx) יוצר הרבה התראות שווא בנתוני מודל מעל הים. שנה ל-True כדי לכלול אותו.
INCLUDE_DRIZZLE = False

RAIN_CODES = set(range(200, 300)) | set(range(500, 600))
if INCLUDE_DRIZZLE:
    RAIN_CODES |= set(range(300, 400))
HEAVY_CODES = {501, 502, 503, 504, 521, 522, 531}


# ---------------------------------------------------------------- גיאוגרפיה

def haversine(lat1, lon1, lat2, lon2):
    """חישוב מרחק בקילומטרים"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_bearing(lat1, lon1, lat2, lon2):
    """חישוב אזימוט מנקודה 1 לנקודה 2 (0-360 מעלות)"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ---------------------------------------------------------------- מצב וקבצים

def _atomic_write(path, text):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def is_in_cooldown():
    if not os.path.exists(COOLDOWN_FILE):
        return False
    try:
        with open(COOLDOWN_FILE, "r") as f:
            last_trigger = datetime.fromisoformat(f.read().strip())
        return datetime.now(TZ_ISRAEL) - last_trigger < timedelta(minutes=COOLDOWN_MINUTES)
    except Exception as e:
        print(f"שגיאה בקריאת cooldown: {type(e).__name__}")
        return False


def update_cooldown():
    _atomic_write(COOLDOWN_FILE, datetime.now(TZ_ISRAEL).isoformat())


def load_previous_state():
    if os.path.exists(PREVIOUS_STATE_FILE):
        try:
            with open(PREVIOUS_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"שגיאה בטעינת מצב קודם: {type(e).__name__}")
    return None


def save_current_state(center_lat, center_lon, front_dist_km, active_count):
    data = {
        "timestamp": datetime.now(TZ_ISRAEL).isoformat(),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "front_dist_km": front_dist_km,
        "active_count": active_count,
    }
    _atomic_write(PREVIOUS_STATE_FILE, json.dumps(data))


# ---------------------------------------------------------------- תקשורת

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code != 200:
            # גוף התשובה של טלגרם לא כולל את הטוקן; ה-URL כן, ולכן לא מדפיסים אותו
            print(f"שגיאה בשליחה לטלגרם: {res.status_code} {res.text[:200]}")
            return False
        return True
    except Exception as e:
        # הודעת חריגה של requests כוללת את ה-URL (ואיתו הטוקן), לכן מדפיסים רק את סוג השגיאה
        print(f"שגיאה בשליחה לטלגרם: {type(e).__name__}")
        return False


def fetch_point(session, lat, lon):
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        res = session.get(OWM_URL, params=params, timeout=(5, 10))
        if res.status_code != 200:
            print(f"שגיאת API בנקודה ({lat}, {lon}): סטטוס {res.status_code}")
            return None
        return res.json()
    except Exception as e:
        print(f"שגיאת API בנקודה ({lat}, {lon}): {type(e).__name__}")
        return None


def parse_point(lat, lon, data):
    """מחזיר dict אם יש משקעים בנקודה, אחרת None"""
    weather = data.get("weather") or []
    if not weather:
        return None
    weather_id = weather[0]["id"]
    if weather_id not in RAIN_CODES:
        return None

    wind = data.get("wind", {})
    rain = data.get("rain", {})
    return {
        "lat": lat, "lon": lon,
        "id": weather_id,
        "dist": haversine(lat, lon, GIVAT_SHMUEL_LAT, GIVAT_SHMUEL_LON),
        "bearing_to_target": calculate_bearing(lat, lon, GIVAT_SHMUEL_LAT, GIVAT_SHMUEL_LON),
        "wind_speed": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "rain_1h": rain.get("1h"),
    }


def scan_grid():
    """קריאה מקבילית לכל נקודות הרשת. מחזיר (נקודות פעילות, מספר קריאות שנכשלו)"""
    with requests.Session() as session, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(lambda pt: fetch_point(session, *pt), SEA_GRID))

    failed = sum(1 for r in results if r is None)
    active = []
    for (lat, lon), data in zip(SEA_GRID, results):
        if data is None:
            continue
        point = parse_point(lat, lon, data)
        if point:
            active.append(point)
    return active, failed


# ---------------------------------------------------------------- ניתוח

def is_thunder(p):
    return 200 <= p["id"] < 300


def is_heavy(p):
    if is_thunder(p) or p["id"] in HEAVY_CODES:
        return True
    return p["rain_1h"] is not None and p["rain_1h"] >= HEAVY_MM_PER_HOUR


def wind_is_supporting(points):
    """האם רוח פני השטח (הפוכה מ-FROM ל-TO) מצביעה לכיוון גבעת שמואל ברוב הנקודות"""
    valid = [p for p in points if p["wind_speed"] is not None and p["wind_deg"] is not None]
    if not valid:
        return False
    supporting = 0
    for p in valid:
        wind_towards = (p["wind_deg"] + 180) % 360
        diff = abs((wind_towards - p["bearing_to_target"] + 180) % 360 - 180)
        if diff <= WIND_ANGLE_TOLERANCE:
            supporting += 1
    return supporting / len(valid) >= 0.5


def detect_approach(prev_state, front_dist, center_dist, now):
    """
    מעקב התקרבות: החזית (הנקודה הפעילה הקרובה ביותר) חייבת להתקרב לפחות
    APPROACH_MIN_KM, ומרכז הכובד חייב להתקרב גם הוא (אישור נוסף).
    מחזיר (approaching, speed_kmh)
    """
    if not prev_state:
        return False, None
    if prev_state.get("center_lat") is None or prev_state.get("front_dist_km") is None:
        return False, None

    try:
        prev_time = datetime.fromisoformat(prev_state["timestamp"])
    except (KeyError, ValueError):
        return False, None

    hours = (now - prev_time).total_seconds() / 3600.0
    if not (MIN_TREND_HOURS <= hours <= MAX_TREND_HOURS):
        return False, None

    prev_center_dist = haversine(prev_state["center_lat"], prev_state["center_lon"],
                                 GIVAT_SHMUEL_LAT, GIVAT_SHMUEL_LON)
    front_moved = prev_state["front_dist_km"] - front_dist   # חיובי = התקרבות
    center_moved = prev_center_dist - center_dist
    speed = front_moved / hours

    if front_moved >= APPROACH_MIN_KM and center_moved > 0 and speed >= MIN_SPEED_KMH:
        return True, speed
    return False, None


# ---------------------------------------------------------------- ראשי

def check_weather():
    missing = [name for name, val in [
        ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
        ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
        ("OPENWEATHER_API_KEY", OPENWEATHER_API_KEY),
    ] if not val]
    if missing:
        print(f"חסרים משתני סביבה: {', '.join(missing)}")
        sys.exit(1)

    now = datetime.now(TZ_ISRAEL)
    active_points, failed = scan_grid()

    if failed > len(SEA_GRID) * MAX_FAILED_FRACTION:
        print(f"{failed} מתוך {len(SEA_GRID)} קריאות נכשלו. הנתונים לא אמינים, מדלג בלי לעדכן מצב.")
        return

    if not active_points:
        save_current_state(None, None, None, 0)
        print("אין משקעים פעילים בים מול גוש דן.")
        return

    # החזית = הנקודה הפעילה הקרובה ביותר לגבעת שמואל
    front_dist = min(p["dist"] for p in active_points)

    # סינון רעש: מינימום 2 נקודות, או נקודה אחת קרובה מאוד
    if len(active_points) < 2 and front_dist > CLOSE_POINT_KM:
        print("נקודת גשם בודדת ורחוקה. מסונן כרעש מקומי.")
        save_current_state(None, None, None, len(active_points))
        return

    avg_lat = sum(p["lat"] for p in active_points) / len(active_points)
    avg_lon = sum(p["lon"] for p in active_points) / len(active_points)
    center_dist = haversine(avg_lat, avg_lon, GIVAT_SHMUEL_LAT, GIVAT_SHMUEL_LON)

    wind_supporting = wind_is_supporting(active_points)

    prev_state = load_previous_state()
    approaching, speed_kmh = detect_approach(prev_state, front_dist, center_dist, now)

    # המצב נשמר בכל ריצה, גם בזמן השתקה, כדי שמעקב התנועה ימשיך לעבוד
    save_current_state(avg_lat, avg_lon, front_dist, len(active_points))

    if is_in_cooldown():
        print("המערכת בהשתקה (Cooldown). המצב עודכן, לא נשלחת הודעה.")
        return

    # תנאי סף: תנועה מדודה, או רוח תומכת + מספיק נקודות / נקודה קרובה
    # (רוח מערבית היא ברירת המחדל בחוף, לכן בלי מגבלת מרחק כל 3 נקודות בים היו מפעילות התראה)
    wind_trigger = (wind_supporting
                    and front_dist <= WIND_ONLY_MAX_FRONT_KM
                    and (len(active_points) >= 3 or front_dist <= CLOSE_POINT_KM))
    if not approaching and not wind_trigger:
        print("זיהוי גשם בים, אך ללא אינדיקציה מספקת להתקרבות לגבעת שמואל.")
        return

    # רמת ביטחון: רוח פני השטח היא אינדיקציה חלשה, ולכן לבדה נותנת "נמוכה"
    if approaching and wind_supporting:
        confidence = "גבוהה 🟢"
    elif approaching:
        confidence = "בינונית 🟡"
    else:
        confidence = "נמוכה 🔴"

    if approaching:
        eta_min = int((front_dist / speed_kmh) * 60)
        arrival = (now + timedelta(minutes=eta_min)).strftime("%H:%M")
        eta_text = f"כ-<b>{eta_min} דקות</b> (סביב {arrival}) - לפי תנועה שנמדדה"
        trend_text = f"התקרבות שנמדדה (~{speed_kmh:.0f} קמ\"ש)"
    else:
        eta_text = "לא ניתן להעריך זמן הגעה"
        trend_text = "כיוון רוח פני השטח תומך (אינדיקציה חלשה)"

    # עוצמה: רק מהנקודות הקרובות (אם אין כאלה, מכולן)
    near_points = [p for p in active_points if p["dist"] <= HEAVY_RADIUS_KM] or active_points
    heavy_count = sum(1 for p in near_points if is_heavy(p))
    if heavy_count >= 2:
        intensity = "🌧️ <b>גשם משמעותי/חזק</b>"
    else:
        intensity = "🌦️ <b>גשם קל / טפטופים</b>"
    if any(is_thunder(p) for p in near_points):
        intensity += " ⛈️ עם סימני רעמים"

    msg = (
        "🚨 <b>מערכת גשם מתקרבת לגבעת שמואל!</b>\n\n"
        f"📊 <b>עוצמה:</b> {intensity}\n"
        f"📍 <b>נקודות פעילות בים:</b> {len(active_points)} מתוך {len(SEA_GRID)}\n"
        f"📏 <b>מרחק חזית קרובה:</b> כ-<b>{front_dist:.1f} ק\"מ</b>\n"
        f"🧭 <b>מגמת תנועה:</b> {trend_text}\n"
        f"⏱️ <b>זמן הגעה (ETA):</b> {eta_text}\n"
        f"🎯 <b>רמת ביטחון:</b> {confidence}\n\n"
        f"🤫 <i>מנגנון השתקה הופעל ל-{COOLDOWN_MINUTES} דקות.</i>"
    )

    if send_telegram_message(msg):
        update_cooldown()


if __name__ == "__main__":
    check_weather()
