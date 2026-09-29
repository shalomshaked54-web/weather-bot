import os
import requests

# --- שליפת מפתחות ממשתני הסביבה של GitHub Actions ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# --- כיסוי מפלצתי וצפוף מאוד לאזור תל אביב וגוש דן ---
COAST_POINTS = [
    # אזור תל אביב מרכז והחוף
    {"name": "תל אביב - חוף הים", "lat": 32.085, "lon": 34.781},
    {"name": "תל אביב - צפון הירקון", "lat": 32.105, "lon": 34.785},
    {"name": "תל אביב - לנצקי/נמל", "lat": 32.095, "lon": 34.775},
    {"name": "תל אביב - דרום/יפו", "lat": 32.055, "lon": 34.755},
    {"name": "תל אביב - מזרח (עזריאלי)", "lat": 32.075, "lon": 34.792},
    # ממערב לחוף (בים - התראה מוקדמת לעננים מתקרבים)
    {"name": "ים - מערב תל אביב 1", "lat": 32.085, "lon": 34.730},
    {"name": "ים - מערב תל אביב 2", "lat": 32.105, "lon": 34.730},
    {"name": "ים - מערב תל אביב 3", "lat": 32.065, "lon": 34.730},
    # טבעת פנימית וצפונית לגוש דן
    {"name": "הרצליה - מערב וחוף", "lat": 32.162, "lon": 34.791},
    {"name": "רמת השרון", "lat": 32.144, "lon": 34.838},
    {"name": "בת ים - חוף", "lat": 32.023, "lon": 34.746},
    {"name": "חולון", "lat": 32.016, "lon": 34.787},
    {"name": "רמת גן - גבעתיים", "lat": 32.068, "lon": 34.829},
    {"name": "פתח תקווה", "lat": 32.087, "lon": 34.887},
    {"name": "ראשון לציון - מערב וחוף", "lat": 31.973, "lon": 34.745},
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
  rainy_points = []

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

        # קוד מתחת ל-700 מעיד על משקעים (גשם/ערפל/ברד)
        if weather_code < 700:
          rainy_points.append(f"• {point['name']} ({weather_desc})")
      else:
        print(f"שגיאה בקבלת נתונים עבור {point['name']}: {data.get('message')}")

    except Exception as e:
      print(f"שגיאת תקשורת מול ה-API עבור {point['name']}: {e}")

  # --- לוגיקה: אם נמצאו אזורים גשומים, שלח התראה אחת מאוחדת בלבד ---
  if len(rainy_points) > 0:
    # ניצור רשימה של המקומות שבהם זוהה גשם
    locations_text = "\n".join(rainy_points)
    alert_msg = (
        f"🚨 **התראת גשם בגוש דן!**\n"
        f"🌧️ זוהו משקעים ב-{len(rainy_points)} נקודות באזור:\n"
        f"{locations_text}"
    )
    send_telegram_alert(alert_msg)
    print("נשלחה התראה מאוחדת לטלגרם!")
  else:
    print("הכל יבש, אין צורך בהתראה כרגע.")


if __name__ == "__main__":
  check_weather()
