#!/usr/bin/env bash
# פרסום status.json לענף status (commit יחיד שנדרס בכל פעם). רץ מה-workflow, ובמצב סערה גם מתוך הבוט
# אחרי כל פריים, כדי שהדשבורד יתעדכן בלי לחכות לסוף הריצה. צריך GH_TOKEN ו-GITHUB_REPOSITORY
set -euo pipefail
DIR="${STATUS_DIR:-status}"
[ -f "$DIR/status.json" ] || exit 0
cd "$DIR"
rm -rf .git
git init -q -b status
git config user.name "rain-bot"
git config user.email "rain-bot@users.noreply.github.com"
git add status.json
git commit -q -m "status"
git push -q -f "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" status
