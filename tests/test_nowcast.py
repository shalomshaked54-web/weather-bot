"""בדיקות לחיזוי לנקודה (nowcast.py, 3.2-beta). בלי רשת"""
import os
import sys
import unittest
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import bot  # noqa: E402
from rainbot import messages, nowcast  # noqa: E402
from rainbot.analysis import analyze, clean_field, motion_weights  # noqa: E402

T0 = 1_760_000_000
TIMES = [T0 + 600 * k for k in range(4)]


def blob(cx, cy, r, dbz):
    f = np.zeros(bot.XX.shape, np.float32)
    f[(bot.XX - cx) ** 2 + (bot.YY - cy) ** 2 <= r * r] = dbz
    return f


def moving(cx, cy, r, v, dbz=35):
    """מערכת שזזה מזרחה במהירות v קמ"ש. cx הוא המיקום בפריים האחרון"""
    return [blob(cx - v * (3 - k) / 6, cy, r, dbz) for k in range(4)]


def truth_min(cx, cy, r, v):
    edge = abs(cx) - np.sqrt(r ** 2 - cy ** 2)
    return (edge - 2.5) / v * 60


class PointNowcastTests(unittest.TestCase):

    def test_eta_close_to_truth(self):
        for cx, cy, r, v in [(-37, 0, 12, 30), (-30, 0, 15, 40), (-50, 8, 12, 30), (-40, 0, 25, 30)]:
            res = analyze(moving(cx, cy, r, v), TIMES)
            nc = res["nowcast"]
            self.assertGreaterEqual(nc["p"], 0.9)
            self.assertLessEqual(abs(nc["eta_mid"] - truth_min(cx, cy, r, v)), 6, (cx, cy, nc))
            lo, hi = nc["eta"]
            self.assertLessEqual(lo, nc["eta_mid"])
            self.assertGreaterEqual(hi, nc["eta_mid"])

    def test_point_eta_replaces_5km_disc_eta(self):
        res = analyze(moving(-37, 0, 12, 30), TIMES)
        self.assertEqual(res["eta"], res["nowcast"]["eta"])
        self.assertLess(res["eta_disc"][0], res["eta"][0])   # הדיסק של 5 ק"מ מקדים את הגשם אצלך

    def test_stage_follows_point_eta(self):
        # גשם ברדיוס 5 ק"מ בעוד ~15 דקות, אבל אצלך רק בעוד ~22: מוקדמת ולא מיידית
        res = analyze(moving(-30, 0, 15, 40), TIMES)
        self.assertGreater(res["nowcast"]["eta_mid"], 20)
        self.assertEqual(res["stage"], "early")

    def test_system_passing_by_has_low_probability(self):
        res = analyze(moving(-40, 30, 10, 30), TIMES)
        nc = res.get("nowcast")
        self.assertTrue(nc is None or nc["p"] < 0.25)

    def test_local_motion_separates_two_systems(self):
        # מערכת איטית (25 קמ"ש) מולך, ומערכת גדולה ומהירה (55) בצפון שמושכת את התנועה הכללית
        fields = [a + b for a, b in zip(moving(-40, 0, 10, 25), moving(-60, 55, 14, 55))]
        res = analyze(fields, TIMES)
        self.assertGreater(res["motion"]["speed"], 40)
        w = [motion_weights(clean_field(f)[0]) for f in fields]
        blocks = nowcast.local_motion(w, TIMES, res["motion"])
        south = [b for b in blocks if b[1] < 25]
        self.assertTrue(south)
        for _, _, vx, _, _ in south:
            self.assertAlmostEqual(vx, 25, delta=5)
        self.assertLessEqual(abs(res["nowcast"]["eta_mid"] - 66), 10)

    def test_no_motion_no_nowcast(self):
        self.assertIsNone(nowcast.point_nowcast(np.zeros(bot.XX.shape, np.float32), [], TIMES, None))
        self.assertIsNone(analyze([np.zeros(bot.XX.shape, np.float32)] * 4, TIMES).get("nowcast"))

    def test_deterministic(self):
        a = analyze(moving(-37, 0, 12, 30), TIMES)["nowcast"]
        b = analyze(moving(-37, 0, 12, 30), TIMES)["nowcast"]
        self.assertEqual(a, b)


class NowcastMessageTests(unittest.TestCase):

    def test_line(self):
        self.assertIn("סיכוי 85%", messages.nowcast_line({"p": 0.85, "dur": 40}))
        self.assertIn("כ-40 דק'", messages.nowcast_line({"p": 0.85, "dur": 40}))
        self.assertIn("יותר משעה", messages.nowcast_line({"p": 1.0, "dur": 90}))
        self.assertIn("יחלוף לידך", messages.nowcast_line({"p": 0.1, "dur": 5}))
        self.assertIsNone(messages.nowcast_line(None))

    def test_alert_message_includes_point_line(self):
        res = analyze(moving(-37, 0, 12, 30), TIMES)
        bot.apply_sources(res, None, None, datetime.fromtimestamp(TIMES[-1], bot.TZ))
        t = datetime.fromtimestamp(TIMES[-1], bot.TZ)
        msg = bot.build_message(res, t, t)
        self.assertIn("📍", msg)


if __name__ == "__main__":
    unittest.main()
