"""בדיקות למערכת הלמידה (3.2-beta). בלי רשת"""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import bot  # noqa: E402
from rainbot import learn  # noqa: E402
from rainbot.config import TZ  # noqa: E402

T = datetime(2026, 12, 1, 18, 0, tzinfo=TZ)
DRY = np.zeros(bot.XX.shape, np.float32)
WET = np.where(bot.XX ** 2 + bot.YY ** 2 <= 9, 30.0, 0.0).astype(np.float32)


def res_pred(eta_mid, lo=None, hi=None, stage="early", src="system"):
    lo = eta_mid - 5 if lo is None else lo
    hi = eta_mid + 5 if hi is None else hi
    return {"stage": stage, "src": src, "eta": (lo, hi), "eta_mid": eta_mid, "nowcast": {"p": 0.9, "eta_mid": eta_mid,
                                                                                        "eta": (lo, hi)}}


def run_event(state, start, predicted_onset_min, actual_onset_min, alert=True):
    """פריים כל 10 דקות מ-start: חיזוי להתחלה ב-start+predicted, וגשם אמיתי מ-start+actual"""
    ev = None
    for k in range(0, 13):
        t = start + timedelta(minutes=10 * k)
        wet = 10 * k >= actual_onset_min
        alerts = [start] if alert else []
        e = learn.observe(state, WET if wet else DRY, t, alerts)
        ev = ev or e
        left = predicted_onset_min - 10 * k
        if not wet and left > 0:
            learn.record(state, res_pred(left), t, alert and k == 0)
    # אחרי האירוע: יבש שוב
    for k in range(13, 20):
        learn.observe(state, DRY, start + timedelta(minutes=10 * k), [])
    return ev


class LearnTests(unittest.TestCase):

    def test_onset_detected_and_scored(self):
        st = {}
        ev = run_event(st, T, 60, 70)
        self.assertIsNotNone(ev)
        self.assertEqual(ev["alert_lead"], 65)      # אמצע הדרך בין 60 ל-70
        self.assertGreater(ev["err_med"], 0)       # הגשם הגיע מאוחר מהחיזוי
        self.assertGreater(ev["n"], 3)

    def test_scale_learns_late_rain_and_is_bounded(self):
        st = {}
        for d in range(6):
            run_event(st, T + timedelta(days=d), 50, 70)
        p = learn.params(st)["system"]
        self.assertGreater(p["eta_scale"], 1.05)
        self.assertLessEqual(p["eta_scale"], 1.4)
        self.assertEqual(learn.params(st)["cell"]["eta_scale"], 1.0)

    def test_starts_from_backtest_seed(self):
        self.assertEqual(learn.params({})["system"]["eta_scale"], learn.LEARN_SEED["system"]["eta_scale"])

    def test_scale_unchanged_when_accurate(self):
        st = {}
        for d in range(6):
            run_event(st, T + timedelta(days=d), 65, 70)
        self.assertAlmostEqual(learn.params(st)["system"]["eta_scale"], 1.0, delta=0.05)

    def test_one_odd_event_does_not_change_much(self):
        st = {}
        run_event(st, T, 30, 110)
        # מעט דגימות: נשארים בנקודת ההתחלה
        self.assertEqual(learn.params(st)["system"]["eta_scale"], learn.LEARN_SEED["system"]["eta_scale"])

    def test_false_alarm_counted(self):
        st = {}
        learn.observe(st, DRY, T, [])
        learn.record(st, res_pred(20), T, True)
        for k in range(1, 10):
            learn.observe(st, DRY, T + timedelta(minutes=10 * k), [])
        self.assertEqual(st["learn"]["false_alarms"], 1)
        self.assertEqual(st["learn"]["pending"], [])

    def test_miss_counted(self):
        st = {}
        run_event(st, T, 1000, 60, alert=False)
        self.assertEqual(st["learn"]["misses"], 1)

    def test_apply_scales_eta(self):
        st = {"learn": {"params": {"system": {"eta_scale": 1.2, "spread": 1.0}}}}
        r = {"stage": "early", "src": "system", "eta": (20, 30), "eta_mid": 25, "watch": None}
        learn.apply(r, st)
        self.assertEqual(r["eta"], (24, 36))
        self.assertEqual(r["eta_mid"], 30)
        self.assertEqual(r["eta_raw"], (20, 30))

    def test_summary_once(self):
        st = {}
        run_event(st, T, 60, 70)
        lines = learn.summary_lines(st)
        self.assertTrue(any("מה למדנו" in x for x in lines))
        learn.mark_reported(st)
        self.assertEqual(learn.summary_lines(st), [])

    def test_restore_from_status(self):
        st = {}
        for d in range(6):
            run_event(st, T + timedelta(days=d), 50, 70)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            with open(path, "w") as f:
                json.dump({"learning": learn.public(st)}, f)
            fresh = {}
            self.assertTrue(learn.restore_from_status(fresh, path))
            self.assertEqual(learn.params(fresh), learn.params(st))
            self.assertFalse(learn.restore_from_status(fresh, path))   # כבר יש: לא דורסים
            self.assertFalse(learn.restore_from_status({}, os.path.join(d, "missing.json")))


if __name__ == "__main__":
    unittest.main()


class BacktestTests(unittest.TestCase):

    def test_backtest_on_synthetic_frames(self):
        from rainbot import backtest

        start = datetime(2026, 12, 1, 12, 0, tzinfo=TZ)

        class Fake:
            def get(self, t):
                k = (t - start).total_seconds() / 3600.0          # שעות מההתחלה
                cx = -60 + 30 * k                                  # חזית שזזה מזרחה ב-30 קמ"ש
                f = np.zeros(bot.XX.shape, np.float32)
                f[(bot.XX > cx - 40) & (bot.XX < cx) & (np.abs(bot.YY) < 40)] = 35.0
                return f
        state, rows = backtest.run(start, start + timedelta(hours=3), frames=Fake(), log=lambda *_: None)
        rep = backtest.report(state)
        self.assertEqual(len(rep["events"]), 1)
        ev = rep["events"][0]
        self.assertIsNotNone(ev["alert_lead"])
        self.assertGreaterEqual(ev["alert_lead"], 30)          # התראה לפחות חצי שעה מראש
        self.assertLessEqual(abs(ev["err_med"]), 8)
