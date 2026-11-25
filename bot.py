#!/usr/bin/env python3
"""
בוט התראות גשם לגבעת שמואל - גרסה 3.1 בטא

נקודת הכניסה. הקוד עצמו נמצא בתיקייה rainbot/, מחולק לפי נושא:
  config.py    הגדרות וספים
  radar.py     RainViewer: הורדה ופענוח של אריחי המכ"ם
  analysis.py  ניתוח: תנועה, מסלול המערכות, מסלול התאים, מגמת עוצמה, תאים שמתפתחים במקום
  forecast.py  תחזית מודלים מ-Open-Meteo ("דעה שנייה")
  ims.py       מד הגשם של השירות המטאורולוגי (פעיל רק עם IMS_TOKEN)
  alerts.py    החלטות: השתקה, חיזוק ממקורות נוספים, "הגשם עבר", "צפוי גשם", בריאות המכ"ם
  messages.py  נוסחי ההודעות
  telegram.py  שליחה לטלגרם
  render.py    תמונת המכ"ם
  storage.py   מצב, לוג, status.json, שמירת נתוני מכ"ם
  replay.py    הרצה חוזרת על אירועים שמורים
  main.py      ריצה אחת מקצה לקצה

שימוש:
  python bot.py                 # ריצה רגילה
  python bot.py --dry-run       # בלי שליחה ובלי עדכון מצב, מדפיס את ההודעה
  python bot.py --dry-run --png out.png --save-frames out.npz
  python -m rainbot.replay out.npz

Weather data by Rain Viewer (https://www.rainviewer.com). Forecast by Open-Meteo (https://open-meteo.com).
"""
import requests  # noqa: F401  (הבדיקות מחליפות את bot.requests.post)

from rainbot.config import *  # noqa: F401,F403
from rainbot.radar import *  # noqa: F401,F403
from rainbot.analysis import *  # noqa: F401,F403
from rainbot.storage import *  # noqa: F401,F403
from rainbot.alerts import *  # noqa: F401,F403
from rainbot.messages import *  # noqa: F401,F403
from rainbot.telegram import *  # noqa: F401,F403
from rainbot.render import *  # noqa: F401,F403
from rainbot.main import main, run  # noqa: F401

if __name__ == "__main__":
    main()
