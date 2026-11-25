"""בדיקות לבוט הגשם. רצות בלי רשת ובלי טלגרם:

    python -m unittest discover -s tests -v

רצפי הזמנים בבדיקות ההשתקה לקוחים מרשם הגלאי (האקסל), מספר השורה מופיע בשם הבדיקה.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import bot  # noqa: E402

TZ = bot.TZ
T0 = 1_760_000_000
TIMES = [T0 + 600 * k for k in range(4)]   # 4 פריימים, 10 דקות ביניהם


def at(s):
    return datetime.fromisoformat(s).replace(tzinfo=TZ)


def blob(cx, cy, r, dbz):
    f = np.zeros(bot.XX.shape, np.float32)
    f[(bot.XX - cx) ** 2 + (bot.YY - cy) ** 2 <= r * r] = dbz
    return f


def res_for(stage, score=3, min_score=3, src="system"):
    return {"stage": stage, "score": score, "min_score": min_score, "src": src,
            "eta": (0, 0) if stage == "now" else (10, 20)}


def after_alert(stage, t, radar=None):
    """מצב כפי שנשמר אחרי התראה שנשלחה בזמן t"""
    state = {}
    bot.update_state(state, res_for(stage), t, radar or t.replace(second=0), sent=True)
    return state


class SendDecisionTests(unittest.TestCase):

    def test_first_alert_is_sent(self):
        self.assertEqual(bot.send_decision({}, "early", at("2026-11-07T08:00:00")), (True, "first"))

    def test_escalation_is_sent(self):
        st = after_alert("early", at("2026-11-07T08:20:29"))
        self.assertEqual(bot.send_decision(st, "imminent", at("2026-11-07T08:45:24")), (True, "escalation"))

    def test_same_stage_within_cooldown_is_held(self):
        t = at("2026-11-07T08:00:00")
        st = after_alert("early", t)
        bot.update_state(st, res_for("early"), t + timedelta(minutes=30), t, sent=False)
        self.assertEqual(bot.send_decision(st, "early", t + timedelta(minutes=35)), (False, "cooldown"))

    def test_row_16_17_same_radar_frame_not_sent_twice(self):
        radar = at("2026-10-27T19:20:00")
        st = after_alert("early", at("2026-10-27T19:20:29"), radar)
        go, why = bot.send_decision(st, "early", at("2026-10-27T19:21:29"), radar.isoformat(timespec="seconds"))
        self.assertEqual((go, why), (False, "same_frame"))

    def test_same_frame_still_allows_escalation(self):
        radar = at("2026-10-27T19:20:00")
        st = after_alert("early", at("2026-10-27T19:20:29"), radar)
        go, _ = bot.send_decision(st, "now", at("2026-10-27T19:21:29"), radar.isoformat(timespec="seconds"))
        self.assertTrue(go)

    def test_row_24_new_wave_after_now_is_not_silenced(self):
        # 01:15 "כעת", הגשם מעל הגבעה נגמר ב-01:30, מערכת חדשה מתקרבת ב-02:05.
        # ב-2.7 ההתראה נחסמה עד 02:45 (90 דקות). ב-3.0 היא יוצאת מיד.
        t_now = at("2026-11-07T01:15:27")
        st = after_alert("now", t_now)
        for m in (5, 10, 15):                     # "כעת" ממשיך להיות מזוהה עד 01:30
            bot.update_state(st, res_for("now"), t_now + timedelta(minutes=m), t_now, sent=False)
        for m in (20, 25, 30, 35, 40, 45):        # ואז גשם רחוק שעובר את הסף, בלי התראה
            bot.update_state(st, res_for("early"), t_now + timedelta(minutes=m), t_now, sent=False)
        t = at("2026-11-07T02:05:27")
        self.assertEqual(bot.send_decision(st, "early", t), (True, "new_wave"))

    def test_new_wave_waits_until_rain_over_target_ends(self):
        t_now = at("2026-11-07T01:15:27")
        st = after_alert("now", t_now)
        bot.update_state(st, res_for("now"), t_now + timedelta(minutes=10), t_now, sent=False)
        # רק 15 דקות מאז "כעת" האחרון: עדיין באותו גל
        self.assertEqual(bot.send_decision(st, "imminent", t_now + timedelta(minutes=25)), (False, "cooldown"))

    def test_rows_22_23_cooldown_boundary_unchanged(self):
        # 89:59 דקות אחרי "כעת" עם "כעת" חוזר: עדיין מושתק, כמו ב-2.7
        t = at("2026-11-06T22:10:29")
        st = after_alert("now", t)
        bot.update_state(st, res_for("now"), at("2026-11-06T23:35:28"), t, sent=False)
        self.assertEqual(bot.send_decision(st, "now", at("2026-11-06T23:40:28")), (False, "cooldown"))
        self.assertEqual(bot.send_decision(st, "now", at("2026-11-06T23:40:30")), (True, "cooldown_over"))

    def test_low_score_detection_does_not_extend_episode(self):
        # ב-2.7 כל זיהוי (גם בציון 2) האריך את האירוע. ב-3.0 רק זיהוי שעובר את הסף.
        t = at("2026-11-07T10:00:00")
        st = after_alert("early", t)
        for m in range(5, 70, 5):
            bot.update_state(st, res_for("early", score=2), t + timedelta(minutes=m), t, sent=False)
        self.assertEqual(bot.send_decision(st, "early", t + timedelta(minutes=70)), (True, "new_episode"))

    def test_rows_25_26_success_sequence_still_sent(self):
        st = after_alert("now", at("2026-11-07T02:45:30"))
        self.assertTrue(bot.send_decision(st, "imminent", at("2026-11-07T04:20:25"))[0])
        bot.update_state(st, res_for("imminent"), at("2026-11-07T04:20:25"), at("2026-11-07T04:20:00"), sent=True)
        self.assertEqual(bot.send_decision(st, "now", at("2026-11-07T04:40:24")), (True, "escalation"))

    def test_old_state_file_without_new_fields(self):
        st = {"last_alert": "2026-11-07T01:15:27+02:00", "last_stage": "now",
              "last_rain_seen": "2026-11-07T01:50:00+02:00"}
        go, _ = bot.send_decision(st, "early", at("2026-11-07T02:00:00"))
        self.assertTrue(go)                       # נופל חזרה לזמן ההתראה האחרונה


class DeliveryTests(unittest.TestCase):

    def test_read_timeout_on_photo_does_not_send_text_again(self):
        with mock.patch.object(bot.requests, "post", side_effect=requests.exceptions.ReadTimeout()) as post:
            self.assertEqual(bot.deliver(b"png", "msg"), "maybe")
        self.assertEqual(post.call_count, 1)

    def test_connection_error_falls_back_to_text(self):
        ok = mock.Mock(status_code=200)
        with mock.patch.object(bot.requests, "post",
                               side_effect=[requests.exceptions.ConnectionError(), ok]) as post:
            self.assertEqual(bot.deliver(b"png", "msg"), "ok")
        self.assertEqual(post.call_count, 2)

    def test_long_caption_goes_as_text(self):
        ok = mock.Mock(status_code=200)
        with mock.patch.object(bot.requests, "post", return_value=ok) as post:
            self.assertEqual(bot.deliver(b"png", "x" * 1200), "ok")
        self.assertEqual(post.call_count, 1)


class AnalysisTests(unittest.TestCase):

    def test_system_from_sea_gives_early_alert(self):
        v = 30.0
        fields = [blob(-37 - v * (3 - k) / 6, 0, 12, 35) for k in range(4)]
        r = bot.analyze(fields, TIMES)
        self.assertEqual((r["stage"], r["src"], r["score"]), ("early", "system", 3))
        self.assertTrue(bot.qualifies(r))
        self.assertAlmostEqual(r["motion"]["speed"], v, delta=2)

    def test_no_rain_no_alert(self):
        r = bot.analyze([np.zeros(bot.XX.shape, np.float32)] * 4, TIMES)
        self.assertIsNone(r["stage"])
        self.assertFalse(bot.qualifies(r))

    def test_cell_tracked_in_four_frames_alerts(self):
        fields = [blob(-14 - 6 * (3 - k), 2, 3, 32) for k in range(4)]
        r = bot.analyze(fields, TIMES)
        self.assertEqual(r["src"], "cell")
        self.assertIn(r["stage"], ("early", "imminent"))
        self.assertTrue(bot.qualifies(r))

    def test_row_20_cell_seen_in_two_frames_does_not_alert(self):
        empty = np.zeros(bot.XX.shape, np.float32)
        fields = [empty, empty, blob(-16, 2, 3, 32), blob(-10, 2, 3, 32)]
        r = bot.analyze(fields, TIMES)
        self.assertFalse(bot.qualifies(r))
        self.assertEqual(r["cell_dbg"]["why"], "low_score")

    def test_row_21_weak_now_cell_two_frames_does_not_alert(self):
        empty = np.zeros(bot.XX.shape, np.float32)
        fields = [empty, empty, blob(-4, 1, 2.5, 23), blob(-1.5, 1, 2.5, 23)]
        r = bot.analyze(fields, TIMES)
        self.assertFalse(bot.qualifies(r))

    def test_strong_now_cell_two_frames_alerts(self):
        empty = np.zeros(bot.XX.shape, np.float32)
        fields = [empty, empty, blob(-4, 1, 2.5, 33), blob(-1.5, 1, 2.5, 33)]
        r = bot.analyze(fields, TIMES)
        self.assertEqual((r["stage"], r["src"]), ("now", "cell"))
        self.assertTrue(bot.qualifies(r))

    def test_cell_velocity_replaced_when_it_contradicts_system(self):
        # התא "רץ" פי 2 מהמערכת (שורה 20): הערכה מחדש לפי וקטור המערכת
        g = {"vx": 20.0, "vy": 0.0, "speed": 20.0, "quality": 0.9, "consistent": True}
        c = lambda x: {"x": x, "y": 0.0, "area": 6, "peak": 32.0, "r": 2.8}
        track = [(1, c(-30.0)), (2, c(-23.3)), (3, c(-16.6))]       # 40 קמ"ש
        info = bot.evaluate_track(track, TIMES, g)
        self.assertTrue(info["adj"])
        self.assertAlmostEqual(info["speed"], 20.0)
        self.assertGreater(info["eta_min"], 35)                      # במקום ~21 דקות

    def test_cell_velocity_kept_when_consistent(self):
        g = {"vx": 20.0, "vy": 0.0, "speed": 20.0, "quality": 0.9, "consistent": True}
        c = lambda x: {"x": x, "y": 0.0, "area": 6, "peak": 32.0, "r": 2.8}
        info = bot.evaluate_track([(1, c(-30.0)), (2, c(-26.0)), (3, c(-22.0))], TIMES, g)  # 24 קמ"ש
        self.assertFalse(info["adj"])

    def test_still_cell_is_not_moved_by_system_motion(self):
        g = {"vx": 20.0, "vy": 0.0, "speed": 20.0, "quality": 0.9, "consistent": True}
        c = lambda x: {"x": x, "y": 0.0, "area": 6, "peak": 32.0, "r": 2.8}
        info = bot.evaluate_track([(2, c(-10.0)), (3, c(-10.1))], TIMES, g)
        self.assertEqual(info["why"], "still")


class MessageTests(unittest.TestCase):

    def test_arrival_window(self):
        now = at("2026-11-07T21:00:00")
        self.assertEqual(bot.arrival_text(now, 10, 30), "בין 21:10 ל-21:30")
        self.assertEqual(bot.arrival_text(now, 10, 12), "סביב 21:11")

    def test_message_builds_for_every_stage(self):
        now = at("2026-11-07T21:00:00")
        for stage in ("early", "imminent", "now"):
            res = {"stage": stage, "src": "system", "eta": (0, 0) if stage == "now" else (15, 25),
                   "front_km": 9.0, "peak_dbz": 35.0, "score": 3,
                   "checks": {"motion": True, "size": True, "persist": True},
                   "motion_used": {"vx": 20.0, "vy": 0.0, "speed": 20.0}}
            msg = bot.build_message(res, now, now)
            self.assertIn("Rain Viewer", msg)
            if stage != "now":
                self.assertIn("בין 21:15 ל-21:25", msg)

    def test_render_png(self):
        png = bot.render_png(np.zeros(bot.XX.shape, np.float32), {}, None)
        self.assertTrue(png.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
