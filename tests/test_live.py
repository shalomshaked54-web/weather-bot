"""בדיקות להודעה החיה (3.2-beta). בלי רשת"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainbot import live, telegram  # noqa: E402
from rainbot.config import TZ  # noqa: E402

T = datetime(2026, 12, 1, 18, 0, tzinfo=TZ)


def res(stage, score=3, nc_p=None, watch=None):
    r = {"stage": stage, "score": score, "min_score": 3, "src": "system", "eta": (15, 25), "eta_mid": 20,
         "peak_dbz": 33.0, "front_km": 10.0, "trend": None, "motion_used": {"vx": 30.0, "vy": 0.0, "speed": 30.0},
         "watch": watch, "nowcast": None if nc_p is None else {"p": nc_p, "dur": 30}}
    if stage is None:
        r["eta"] = None
    return r


class LiveTests(unittest.TestCase):

    def setUp(self):
        telegram.LAST.clear()
        telegram.LAST.update(id=42, kind="photo")
        self.state = {}
        live.remember(self.state, res("early"), T, "r0", True)
        self.calls = []

    def send(self, *a):
        self.calls.append(a)
        return "ok"

    def test_remember_only_approaching(self):
        self.assertEqual(self.state["live"]["id"], 42)
        st = {}
        live.remember(st, res("now"), T, "r0", True)
        self.assertNotIn("live", st)

    def test_edit_on_new_frame_once(self):
        t = T + timedelta(minutes=10)
        self.assertEqual(live.update(self.state, res("early"), t, t, "r1", send=self.send), "edit")
        self.assertEqual(live.update(self.state, res("early"), t, t, "r1", send=self.send), None)  # אותו פריים
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][:2], (42, "photo"))
        self.assertIn("🔄", self.calls[0][2])

    def test_cancel_after_two_dry_frames_then_resume(self):
        t1, t2 = T + timedelta(minutes=10), T + timedelta(minutes=20)
        self.assertIsNone(live.update(self.state, res(None), t1, t1, "r1", send=self.send))
        self.assertEqual(live.update(self.state, res(None), t2, t2, "r2", send=self.send), "cancel")
        self.assertIn("כבר לא צפוי", self.calls[-1][2])
        self.assertTrue(live.resume(self.state, t2, "cooldown"))
        self.assertFalse(live.resume(self.state, t2, "same_frame"))

    def test_single_dry_frame_does_not_cancel(self):
        t1, t2 = T + timedelta(minutes=10), T + timedelta(minutes=20)
        live.update(self.state, res(None), t1, t1, "r1", send=self.send)
        live.update(self.state, res("early"), t2, t2, "r2", send=self.send)
        self.assertEqual(self.state["live"]["miss"], 0)
        self.assertFalse(self.state["live"]["cancelled"])

    def test_no_cancel_while_watch_or_nowcast_says_rain(self):
        for r in (res(None, watch={"eta": (50, 70)}), res(None, nc_p=0.6)):
            st = {}
            live.remember(st, res("early"), T, "r0", True)
            for k in (1, 2, 3):
                t = T + timedelta(minutes=10 * k)
                self.assertIsNone(live.update(st, r, t, t, f"r{k}", send=self.send))

    def test_expires(self):
        t = T + timedelta(minutes=120)
        self.assertIsNone(live.update(self.state, res("early"), t, t, "r9", send=self.send))
        self.assertNotIn("live", self.state)

    def test_not_modified_is_ok(self):
        class R:
            status_code = 400
            text = "Bad Request: message is not modified"
        with mock.patch.object(telegram.requests, "post", return_value=R()):
            self.assertEqual(telegram.edit_message(1, "text", "x"), "ok")

    def test_send_remembers_message_id(self):
        class R:
            status_code = 200

            def json(self):
                return {"result": {"message_id": 77}}
        with mock.patch.object(telegram.requests, "post", return_value=R()):
            self.assertEqual(telegram.deliver(b"png", "hi"), "ok")
        self.assertEqual(telegram.LAST, {"id": 77, "kind": "photo"})


if __name__ == "__main__":
    unittest.main()
