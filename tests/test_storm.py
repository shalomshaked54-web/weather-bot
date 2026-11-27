"""בדיקות למצב סערה (3.2-beta). בלי רשת"""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainbot import config, main, storage  # noqa: E402

TZ = config.TZ


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class StormLoopTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = mock.patch.object(config, "STATE_FILE", os.path.join(self.tmp.name, "state.json"))
        self.p.start()
        self.clock = FakeClock()
        self.frame = int(datetime.now(TZ).timestamp()) - 120
        self.runs = 0

    def tearDown(self):
        self.p.stop()
        self.tmp.cleanup()

    def newest(self):
        """פריים חדש כל 10 דקות של שעון הבדיקה"""
        return self.frame + 600 * int(self.clock.t // 600)

    def run_once(self, storm=True):
        def f():
            self.runs += 1
            st = storage.load_state()
            st["last_radar"] = datetime.fromtimestamp(self.newest(), TZ).isoformat()
            if storm:
                st["storm_seen"] = datetime.now(TZ).isoformat()
            else:
                st.pop("storm_seen", None)
            storage.save_state(st)
            self.clock.t += 30          # ניתוח לוקח חצי דקה
            return 0
        return f

    def loop(self, budget, storm=True, publish=None, run_once=None):
        return main.storm_loop(budget, run_once=run_once or self.run_once(storm), clock=self.clock,
                               sleep=self.clock.sleep, newest=self.newest, publish=publish or (lambda: None))

    def test_no_storm_runs_once(self):
        self.loop(30, storm=False)
        self.assertEqual(self.runs, 1)
        self.assertLess(self.clock.t, 60)

    def test_storm_analyses_every_new_frame_within_budget(self):
        pubs = []
        self.loop(30, publish=lambda: pubs.append(self.clock.t))
        # פריימים ב-0, 10, 20 דקות: 3 ניתוחים, והאחרון לא יותר מ-27 דקות מההתחלה
        self.assertEqual(self.runs, 3)
        self.assertEqual(len(pubs), 3)
        self.assertLessEqual(self.clock.t / 60, 30)

    def test_new_frame_is_picked_up_within_a_minute(self):
        starts = []
        orig = self.run_once()

        def tracked():
            starts.append(self.clock.t)
            return orig()
        self.loop(25, run_once=tracked)
        for k, s in enumerate(starts[1:], 1):
            self.assertLess(s - 600 * k, 60)

    def test_storm_active_window(self):
        now = datetime.now(TZ)
        self.assertTrue(main.storm_active({"storm_seen": (now - timedelta(minutes=10)).isoformat()}, now))
        self.assertFalse(main.storm_active({"storm_seen": (now - timedelta(minutes=45)).isoformat()}, now))
        self.assertFalse(main.storm_active({}, now))

    def test_storm_weather(self):
        base = {"stage": None, "score": 0, "min_score": 3, "watch": None}
        self.assertTrue(main.storm_weather({**base, "diag": {"nearest_km": 80.0}}))
        self.assertFalse(main.storm_weather({**base, "diag": {"nearest_km": 130.0}}))
        self.assertFalse(main.storm_weather({**base, "diag": {"nearest_km": None}}))


if __name__ == "__main__":
    unittest.main()
