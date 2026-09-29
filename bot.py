import math
import os
from datetime import datetime, timezone, timedelta
import requests

# --- שליפת משתני סביבה ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# --- מיקום היעד: גבעת שמואל ---
GIVAT_SHMUEL = {"lat": 32.0780, "lon": 34.8470}

# --- 1. הקמת רשת מסיבית בים מול גוש דן (18 נקודות בים) ---
SEA_POINTS = []
latitudes = [31.95, 32.03, 32.10, 32.17, 32.25]  # מראשל"צ ועד נתניה
longitudes = [34.50, 34.60, 34.68, 34.73]         # עומק הים מול החוף

for lat in latitudes:
  for lon in longitudes:
    SEA_POINTS.append({
        "name": f"ים ({lat:.2f}, {lon:.2f})",
        "lat": lat,
        "lon": lon,
    })


def calculate_distance(lat1, lon1, lat2, lon2):
  """חישוב מרחק בקילומטרים מנקודה בים לגבעת שמואל"""
  R = 6371.0
  dlat = math.radians(lat2 - lat1)
  dlon = math.radians(lon2 - lon1)
  a = (
      math.sin(dlat / 2) ** 2
      + math.cos(math.radians(lat1))
      * math.cos(math.radians(lat2))
      * math.sin(dlon / 2) ** 2
  )
  return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def send_telegram_alert(message):
  url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
  payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
  try:
    requests.post(url, json=payload)
  except Exception as e:
    print(f"שגיאה בשליחת הודעה: {e}")


def check_weather():
  rainy_points = []
  min_distance = float("inf")

  # --- 2. סריקת רשת הנקודות הימיות ---
  for point in SEA_POINTS:
    url = (
        "https://api.openweathermap.org/data/2.5/weather?lat="
        f"{point['lat']}&lon={point['lon']}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    try:
      res = requests.get(url, timeout=10)
      data = res.json()
      if res.status_code == 200:
        weather_code = data["weather"][0]["id"]
        weather_desc = data["weather"][0]["description"]

        if weather_code < 700:  # גשם / משקעים
          dist = calculate_distance(
              point["lat"],
              point["lon"],
              GIVAT_SHMUEL["lat"],
              GIVAT_SHMUEL["lon"],
          )
          if dist < min_distance:
            min_distance = dist
          rainy_points.append(f"• {point['name']} ({weather_desc})")
    except Exception as e:
      print(f"שגיאה בנקודה {point['name']}: {e}")

  active_count = len(rainy_points)
  print(f"נמצאו {active_count} נקודות גשומות בים.")

  # --- 3. לוגיקת התראות וזמן הגעה משוער (ETA) ---
  if active_count > 0:
    # חישוב ETA (לפי מהירות התקדמות עננים ממוצעת של 35 קמ"ש)
    eta_minutes = int((min_distance / 35.0) * 60)
    israel_time = datetime.now(timezone.utc) + timedelta(hours=3)
    arrival_time = (israel_time + timedelta(minutes=eta_minutes)).strftime(
        "%H:%M"
    )

    # --- 4. מנגנון השתקה (אם יותר מ-2 נקודות) ---
    cooldown_note = ""
    if active_count > 2:
      cooldown_note = (
          "\n\n🛑 **מנגנון השתקה הופעל:** עקב זיהוי נרחב, לא יישלחו התראות"
          " נוספות בשעה הקרובה."
      )

    locations_text = "\n".join(rainy_points[:5])  # מציג עד 5 נקודות
    if active_count > 5:
      locations_text += f"\n• ועוד {active_count - 5} נקודות בים..."

    msg = (
        f"🚨 **התראת ענני גשם בים (מול גוש דן)!**\n\n"
        f"🌧️ זוהו משקעים ב-{active_count} נקודות ימיות.\n"
        f"📏 מרחק קו חזית ממוקד: כ-{int(min_distance)} ק\"מ מגבעת שמואל.\n"
        f"⏱️ **זמן הגעה משוער לגבעת שמואל:** כ-{eta_minutes} דקות (סביב"
        f" {arrival_time}).\n\n"
        f"📍 **מיקומים בים:**\n{locations_text}"
        f"{cooldown_note}"
    )

    send_telegram_alert(msg)


if __name__ == "__main__":
  check_weather()
