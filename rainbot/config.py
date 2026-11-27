"""הגדרות הבוט: משתני סביבה, ספים וקבועים. כל הקבצים האחרים קוראים מכאן"""
import os
from zoneinfo import ZoneInfo

import numpy as np

VERSION = "3.2-beta"
TZ = ZoneInfo("Asia/Jerusalem")

# ------------------------------------------------------------------ סביבה

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
STATE_DIR = os.environ.get("STATE_DIR", "state")
STATUS_DIR = os.environ.get("STATUS_DIR", "status")   # status.json לדשבורד
OUT_DIR = os.environ.get("OUT_DIR", "out")   # תמונות ונתוני מכ"ם של התראה / חשד לפספוס (artifact)
STATE_FILE = os.path.join(STATE_DIR, "state.json")
HISTORY_FILE = os.path.join(STATE_DIR, "history.jsonl")
HISTORY_MAX_LINES = 3000

TARGET_LAT, TARGET_LON = 32.0780, 34.8470

# ------------------------------------------------------------------ RainViewer

API_URL = "https://api.rainviewer.com/public/weather-maps.json"
ZOOM = 7            # מקסימום שמותר ב-API החינמי
TILE_SIZE = 512
COLOR = 2           # Universal Blue - סכמת הצבעים היחידה שמתועדת כזמינה
N_FRAMES = 4        # 4 פריימים = 30 דקות של היסטוריה
MIN_FRAMES = 3      # 3.1: פריים שחסר לו אריח נזרק, וממשיכים כל עוד נשארו 3
MAX_FRAME_AGE_MIN = 45

# רשת עבודה מקומית (ק"מ ממזרח / מצפון לגבעת שמואל)
CELL_KM = 2.0
X_RANGE = (-140.0, 30.0)
Y_RANGE = (-80.0, 80.0)
KM_PER_DEG = 111.195

# ------------------------------------------------------------------ זיהוי גשם

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

# התראות - מסלול המערכות (גשם שמגיע מהים)
EARLY_MIN = 45           # התראה מוקדמת כש-ETA <= 45 דקות
IMMINENT_MIN = 20        # התראה "מגיע עכשיו" כש-ETA <= 20 דקות
MIN_SCORE_SYSTEM = 3     # 0-3. נדרש 3 (גבוהה) למסלול המערכות
MIN_ALERT_AREA_KM2 = 100.0
PERSIST_MIN_CELLS = 3 * MIN_CELLS
STAGE_RANK = {"early": 1, "imminent": 2, "now": 3}

# מסלול תאים: תאים קטנים וקומפקטיים קרוב לגבעה, נעקבים על השדה הגולמי
CELL_ZONE_KM = 60.0
CELL_MIN_DBZ = 15.0
CELL_PEAK_DBZ = 22.0
CELL_MIN_CELLS = 3
CELL_MAX_CELLS = 60
CELL_MAX_ETA_MIN = 30
CELL_MAX_SPEED_KMH = 90.0
CELL_MIN_APPROACH_KMH = 5.0
CELL_HIT_BASE_KM = 4.0
CELL_HIT_SPREAD = 0.10
MIN_SCORE_CELL = 3
MIN_SCORE_CELL_NOW_STRONG = 2
CELL_NOW_STRONG_DBZ = 28.0
CELL_SANITY_SPEED_RATIO = 1.6
CELL_SANITY_ANGLE_DEG = 60.0

# 3.1: מגמת עוצמה. הפרש של 4 dBZ ומעלה בין הפריים האחרון לממוצע הקודמים = מתחזק / נחלש
TREND_DBZ = 4.0
# 3.1: תא שגווע (ירד 6+ dBZ מהשיא שלו ועכשיו חלש מ-26) לא מתריע - תאים קטנים כאלה מתפוגגים
CELL_DECAY_DBZ = 6.0
CELL_DECAY_MAX_PEAK = 26.0

# 3.1: תא שהתפתח במקום ממש ליד הגבעה (לא הגיע מהים). דרישות:
# נצפה בשני הפריימים האחרונים, שיא 30+ dBZ, כמעט לא זז, לא היה שם לפני 30 דקות (לא הד קרקע)
POPUP_EDGE_KM = 6.0
POPUP_PEAK_DBZ = 30.0
POPUP_MAX_SPEED_KMH = 10.0
POPUP_CLEAR_DBZ = 20.0

# 3.2: התראת "מעקב" - מערכת גשם בים שתגיע בעוד 45-90 דקות. רק בתנאים מחמירים:
# ציון 3, תנועה איכותית, מערכת גדולה, לא נחלשת, והתחזית מסכימה שיירד גשם באותו זמן
WATCH_MAX_MIN = 90
WATCH_MIN_QUALITY = 0.5
WATCH_MIN_AREA_KM2 = 200.0
WATCH_GAP_MIN = 180          # לכל היותר פעם ב-3 שעות
WATCH_AFTER_ALERT_MIN = 90   # לא אם כבר נשלחה התראה רגילה לאחרונה

# 3.2: "הגשם מתחזק" - בזמן גשם, מעבר לגשם חזק (40+ dBZ) כשההתראה האחרונה הייתה על גשם חלש/בינוני
HEAVY_DBZ = 40.0
HEAVY_PREV_MAX_DBZ = 35.0

# 3.2: חיזוי לנקודה (nowcast.py) - הרצה אחורה מהגבעה על שדה תנועה מקומי, עם אנסמבל
NC_MEMBERS = 40              # מספר התרחישים
NC_HIT_KM = 2.5              # "גשם אצלי" = גשם במרחק 2.5 ק"מ מהנקודה
NC_SPEED_SIGMA = 0.12        # פיזור המהירות בין התרחישים (12%), גדל כשהתנועה פחות אמינה
NC_ANGLE_SIGMA_DEG = 6.0     # פיזור הכיוון (מעלות), גדל כשהתנועה פחות אמינה
NC_BLOCK_CELLS = 24          # אזור לתנועה מקומית: 24 תאים = 48 ק"מ
NC_BLOCK_STRIDE = 12         # חפיפה של חצי אזור
NC_BLOCK_MIN_RAIN = 25       # מינימום תאי גשם באזור כדי לאמוד בו תנועה
NC_BLOCK_MIN_Q = 0.45        # איכות מינימלית של אומדן אזורי
NC_BLOCK_MAX_DEV_KMH = 15.0  # סטייה מותרת של אזור מהתנועה הכללית: 15 קמ"ש,
NC_BLOCK_MAX_DEV_RATIO = 0.6 # או 60% מהמהירות הכללית (הגדול מביניהם)
NC_BLOCK_MAX_ANGLE_DEG = 45.0
NC_BLOCK_SIGMA_KM = 20.0     # רדיוס ההשפעה של אזור
NC_GLOBAL_WEIGHT = 0.15      # משקל התנועה הכללית (בסיס, בעיקר איפה שאין אזורים)
NC_MIN_P = 0.5               # מעל הסיכוי הזה החיזוי לנקודה קובע את זמן ההגעה בהודעה
NC_MISS_P = 0.25             # מתחת לזה: "ייתכן שיחלוף לידך" בהודעה

# חשד לפספוס: הד קרוב בלי התראה => נשמרים תמונה ונתונים לבדיקה
SUSPICIOUS_RADIUS_KM = 10.0
SUSPICIOUS_MIN_CELLS = 2
SUSPICIOUS_MIN_DBZ = 20.0

# השתקה
COOLDOWN_MIN = 90        # חזרה על אותו שלב באותו אירוע
EPISODE_GAP_MIN = 60     # בלי זיהוי שעובר את סף ההתראה כל כך הרבה זמן = אירוע חדש
NOW_CLEAR_MIN = 20       # אחרי "כעת", כך הרבה זמן בלי גשם מעל הגבעה = גל חדש מותר להתריע

# 3.1: "הגשם עבר" - נשלח פעם אחת אחרי התראת "כעת", כשאין הד ב-10 ק"מ ואין גשם בדרך
RAIN_END_CLEAR_MIN = 25
RAIN_END_MAX_AGE_H = 3

# 3.1: בריאות המכ"ם - הודעה אחת אם אין מכ"ם תקין כל כך הרבה דקות, והודעה כשהוא חוזר
RADAR_DOWN_ALERT_MIN = 30

# 3.2: מצב סערה. כשיש גשם עד 100 ק"מ (או התראה / מעקב), הריצה לא נגמרת אחרי פריים אחד: היא נשארת
# פתוחה עד STORM_BUDGET_MIN דקות, בודקת כל 45 שניות אם RainViewer פרסם פריים חדש, ומנתחת אותו מיד.
# כך התראה יוצאת דקה-שתיים אחרי שהפריים קיים, ולא עד 9 דקות אחריו
STORM_RADIUS_KM = 100.0
STORM_HOLD_MIN = 30          # מצב הסערה נשאר 30 דקות אחרי הפעם האחרונה שנראה גשם בטווח
STORM_POLL_S = 45
STORM_MIN_RUN_GAP_S = 90     # לא יותר מניתוח אחד כל 90 שניות (גם כשהפריים החדש לא נטען)
STORM_TAIL_MIN = 3           # לא מתחילים ניתוח חדש פחות מ-3 דקות לפני סוף התקציב
STATUS_PUBLISH = os.environ.get("STATUS_PUBLISH") == "1"   # פרסום הדשבורד אחרי כל פריים (ב-workflow)

# 3.2: הודעה חיה. אחרי התראה מוקדמת / מיידית, אותה הודעה מתעדכנת בכל פריים חדש (בלי צלצול).
# מגיע הגשם -> התראת "כעת" רגילה (עם צלצול). שני פריימים ברצף בלי גשם בדרך -> ההודעה מתעדכנת
# ל"כבר לא צפוי", ואם הגשם חוזר נשלחת התראה חדשה עם צלצול
LIVE_MAX_MIN = 90
LIVE_CANCEL_FRAMES = 2

# ------------------------------------------------------------------ Open-Meteo (תחזית)

FORECAST_ENABLED = os.environ.get("FORECAST", "1") != "0"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_CACHE_MIN = 15
FORECAST_RAIN_MM = 0.1        # משבצת של 15 דקות עם 0.1 מ"מ ומעלה = "התחזית צופה גשם"
# חיזוק ודאות: מערכת בציון 2 עם תנועה עקבית, כשהתחזית צופה גשם בחלון ההגעה
FORECAST_BOOST_MM = 0.2
FORECAST_BOOST_PROB = 60
# הודעת "צפוי גשם בשעות הקרובות"
HEADSUP_PROB = 70
HEADSUP_MM = 1.0
HEADSUP_HOURS = 3
HEADSUP_GAP_H = 12            # לכל היותר פעם ב-12 שעות
HEADSUP_AFTER_ALERT_H = 3     # לא אחרי התראת מכ"ם (כבר יודעים שיורד גשם)

# ------------------------------------------------------------------ השירות המטאורולוגי (מד גשם)

IMS_TOKEN = os.environ.get("IMS_TOKEN") or None
IMS_STATION = os.environ.get("IMS_STATION", "54")    # 54 = בית דגן, כ-6 ק"מ מהגבעה
IMS_STATION_NAMES = {"54": "בית דגן"}
IMS_URL = "https://api.ims.gov.il/v1/envista/stations/{station}/data/latest"
IMS_MAX_AGE_MIN = 30

# ------------------------------------------------------------------ סכמת הצבעים של RainViewer

# Universal Blue (גשם), dBZ 15..64, ועוד 65 ו-75+
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
