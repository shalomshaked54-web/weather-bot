import os
import requests

# --- שליפת מפתחות ממשתני הסביבה של GitHub Actions ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# רשימת נקודות לאורך החוף
COAST_POINTS = [
    {"name": "חוף צפוני (חיפה)", "lat": 32.819, "lon": 34.989},
    {"name": "חוף מרכזי (תל אביב)", "lat": 32.085, "lon": 34.781},
    {"name": "חוף דרומי (אשדוד)", "lat": 31.804, "lon": 34.655},
]


def send_telegram_alert(message):
  url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
  payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
  try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
  except Exception as e:
    print(f"שגיאה בשליחת הודעה לטלגרם: {e}")


def check_weather():
  for point in COAST_POINTS:
    url = (
        "https://api.openweathermap.org/data/2.5/weather?lat="
        f"{point['lat']}&lon={point['lon']}&appid={OPENWEATHER_API_KEY}&units=metric"
    )

    try:
      response = requests.get(url)
      data = response.json()

      if response.status_code == 200:
        weather_code = data["weather"][0]["id"]
        weather_desc = data["weather"][0]["description"]

        print(f"בודק את {point['name']}: קוד {weather_code} ({weather_desc})")

        # קוד מתחת ל-700 מעיד על משקעים
        if weather_code < 700:
          alert_msg = (
              f"🚨 **התראת גשם!**\n"
              f"📍 אזור: {point['name']}\n"
              f"🌧️ מצב: {weather_desc} (קוד {weather_code})"
          )
          send_telegram_alert(alert_msg)
      else:
        print(f"שגיאה בקבלת נתונים עבור {point['name']}: {data.get('message')}")

    except Exception as e:
      print(f"שגיאת תקשורת מול ה-API עבור {point['name']}: {e}")


if __name__ == "__main__":
  check_weather()
