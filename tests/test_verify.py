"""בדיקות ללמידה מכל פריים (verify.py) ולדיווחים מטלגרם (feedback.py). בלי רשת"""
import math
import os
import sys
import unittest
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainbot import feedback, learn, verify  # noqa: E402
from rainbot.config import CELL_KM, TZ  # noqa: E402
from rainbot.radar import XX, YY  # noqa: E402

T = datetime(2026, 11, 29, 9, 0, tzinfo=TZ)


def blob(cx, cy, dbz=25.0, r=14.0):
    """אזור גשם חלש (15-25 dBZ) עם מבנה, כדי שההתאמה תהיה חדה"""
    d2 = (XX - cx) ** 2 + (YY - cy) ** 2
    f = np.where(d2 <= r * r, dbz - 8.0 * np.sqrt(d2) / r, 0.0)
    f += np.where((XX - cx - 6) ** 2 + (YY - cy + 4) ** 2 <= 16, 8.0, 0.0)
    return f.astype(np.float32)


def motion(vx, vy):
    return {"vx": vx, "vy": vy, "speed": math.hypot(vx, vy)}


class TestVerify(unittest.TestCase):
    def test_pack_roundtrip(self):
        f = blob(0, 0)
        self.assertTrue(np.allclose(verify.unpack(verify.pack(f)), np.round(f)))

    def test_learns_that_motion_was_too_slow(self):
        # הגשם באמת זז 40 קמ"ש מזרחה, אבל הבוט חשב 32 (איטי ב-20%) -> פקטור ~1.25
        true_v, est_v = 40.0, 32.0
        lr = {}
        for k in range(4):
            t = T + timedelta(minutes=10 * k)
            x = -40 + true_v * (10 * k) / 60.0
            s = verify.step(lr, blob(x, 0), t, motion(est_v, 0.0))
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s["f"], 1.25, delta=0.1)
        self.assertLessEqual(abs(s["a"]), 5)

    def test_learns_rotation(self):
        # התנועה האמיתית מסובבת 15 מעלות עם כיוון השעון ממה שהבוט חשב
        v, rot = 40.0, math.radians(15)
        vx, vy = v, 0.0
        tvx, tvy = math.cos(rot) * vx + math.sin(rot) * vy, -math.sin(rot) * vx + math.cos(rot) * vy
        lr = {}
        for k in range(4):
            t = T + timedelta(minutes=10 * k)
            m = 10 * k / 60.0
            s = verify.step(lr, blob(-40 + tvx * m, 10 + tvy * m), t, motion(vx, vy))
        self.assertIsNotNone(s)
        self.assertAlmostEqual(s["a"], 15, delta=5)

    def test_no_sample_without_rain_or_motion(self):
        lr = {}
        for k in range(4):
            s = verify.step(lr, np.zeros(XX.shape, np.float32), T + timedelta(minutes=10 * k), motion(30, 0))
            self.assertIsNone(s)
        lr = {}
        for k in range(4):
            s = verify.step(lr, blob(0, 0), T + timedelta(minutes=10 * k), None)
        self.assertIsNone(s)

    def test_same_frame_twice_is_ignored(self):
        lr = {}
        verify.step(lr, blob(0, 0), T, motion(30, 0))
        verify.step(lr, blob(0, 0), T, motion(30, 0))
        self.assertEqual(len(lr["frames"]), 1)

    def test_refit_moves_eta_scale_and_rot(self):
        # הרבה דגימות: הגשם זז מהר ב-25% מהחישוב וסובב 10 מעלות
        state = {}
        lr = learn._learn(state)
        lr["verify"] = [{"t": T.isoformat(), "f": 1.25, "a": 10.0, "q": 0.7, "q0": 0.5, "v": 30.0}] * 60
        learn._refit(lr, update_spread=False)
        p = learn.params(state)["system"]
        self.assertGreater(p["nv"], 5)
        self.assertLess(p["eta_scale"], 0.90)      # תנועה מהירה יותר -> זמני הגעה קצרים יותר
        self.assertGreater(p["rot"], 1.0)
        self.assertLessEqual(p["rot"], 10.0)

    def test_weighted_median(self):
        self.assertEqual(verify.weighted_median([1, 2, 3], [1, 1, 10]), 3)
        self.assertEqual(verify.weighted_median([1, 2, 3], [1, 1, 1]), 2)


class TestFeedback(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(feedback.classify("גשם"), "rain")
        self.assertEqual(feedback.classify("יורד אצלי"), "rain")
        self.assertEqual(feedback.classify("לא יורד"), "dry")
        self.assertEqual(feedback.classify("יבש"), "dry")
        self.assertIsNone(feedback.classify("מה המצב?"))
        self.assertIsNone(feedback.classify(None))

    def _upd(self, uid, text, t, chat="123"):
        return {"update_id": uid, "message": {"chat": {"id": int(chat)}, "text": text, "date": int(t.timestamp())}}

    def test_parse_filters_chat_and_age(self):
        now = T
        ups = [self._upd(5, "גשם", now - timedelta(minutes=3)),
               self._upd(6, "גשם", now - timedelta(minutes=3), chat="999"),
               self._upd(7, "יבש", now - timedelta(hours=5)),
               {"update_id": 8, "channel_post": {"chat": {"id": 123}, "text": "יבש",
                                                 "date": int((now - timedelta(minutes=1)).timestamp())}}]
        reps, off = feedback.parse_updates(ups, "123", now)
        self.assertEqual(off, 9)
        self.assertEqual([k for k, _ in reps], ["rain", "dry"])

    def test_poll_rain_is_onset_and_radar_does_not_double_count(self):
        from rainbot import config
        old = config.TELEGRAM_CHAT_ID
        config.TELEGRAM_CHAT_ID = "123"
        try:
            state, sent = {}, []
            on = T + timedelta(minutes=40)
            learn.record(state, {"stage": "early", "src": "system", "eta": (25, 35), "eta_mid": 30,
                                 "nowcast": {"p": 0.9}}, T, True)
            dry = np.zeros(XX.shape, np.float32)
            for k in range(4):
                learn.observe(state, dry, T + timedelta(minutes=10 * k), [T])
            done = feedback.poll(state, on + timedelta(minutes=2), [T],
                                 fetch=lambda off: [self._upd(1, "גשם", on)], reply=sent.append)
            self.assertEqual(done, [("rain", "onset")])
            self.assertEqual(state["tg_offset"], 2)
            self.assertEqual(len(sent), 1)
            ev = state["learn"]["events"][-1]
            self.assertEqual(ev.get("by"), "user")
            n = len(state["learn"]["events"])
            wet = np.where(XX ** 2 + YY ** 2 <= 9, 30.0, 0.0).astype(np.float32)
            learn.observe(state, wet, on + timedelta(minutes=5), [T])
            self.assertEqual(len(state["learn"]["events"]), n)    # המכ"ם לא סופר שוב את אותו אירוע
        finally:
            config.TELEGRAM_CHAT_ID = old

    def test_dry_after_alert_is_false_alarm(self):
        state = {}
        self.assertEqual(learn.user_report(state, "dry", T, [T - timedelta(minutes=20)]), "false_alarm")
        self.assertEqual(state["learn"]["user_false"], 1)
        self.assertEqual(learn.user_report(state, "dry", T, []), "noted")

    def test_poll_survives_telegram_failure(self):
        self.assertEqual(feedback.poll({}, T, fetch=lambda off: None, reply=lambda x: None), [])


if __name__ == "__main__":
    unittest.main()
