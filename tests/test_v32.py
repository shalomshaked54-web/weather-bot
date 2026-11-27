"""בדיקות ל-3.2: התראת "מעקב", "הגשם מתחזק", חוסן בטלגרם ובמכ"ם, ונוסח ההודעות החדש.
רצות בלי רשת ובלי טלגרם."""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import bot  # noqa: E402
from rainbot import alerts, config, learn, main, messages, radar, storage, telegram  # noqa: E402

TZ = bot.TZ
T0 = 1_760_000_000
TIMES = [T0 + 600 * k for k in range(4)]
EMPTY = np.zeros(bot.XX.shape, np.float32)
RT = datetime.fromtimestamp(TIMES[-1], TZ)


def at(s):
    return datetime.fromisoformat(s).replace(tzinfo=TZ)


def blob(cx, cy, r, dbz):
    f = np.zeros(bot.XX.shape, np.float32)
    f[(bot.XX - cx) ** 2 + (bot.YY - cy) ** 2 <= r * r] = dbz
    return f


def far_system(x=-55, dbz=35, r=12):
    """מערכת גדולה בים, ~75 דקות מהגבעה, נעה מזרחה ב-30 קמ"ש"""
    return [blob(x - 30 * (3 - k) / 6, 0, r, dbz) for k in range(4)]


def wet_forecast(start=RT, prob=80):
    slots = [[(start + timedelta(minutes=15 * k)).strftime("%Y-%m-%dT%H:%M"), 0.5] for k in range(24)]
    hourly = [[(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00"), prob, 2.0] for h in range(6)]
    return {"ok": True, "t": start.isoformat(), "slots": slots, "hourly": hourly}


def dry_forecast(start=RT):
    fc = wet_forecast(start)
    fc["slots"] = [[t, 0.0] for t, _ in fc["slots"]]
    return fc


def res_now(peak, score=3):
    return {"stage": "now", "score": score, "min_score": 3, "src": "system", "eta": (0, 0), "peak_dbz": peak,
            "checks": {"motion": True, "size": True, "persist": True}, "diag": {"max_dbz_10km": peak}}


# ------------------------------------------------------------------ מעקב

class WatchTests(unittest.TestCase):

    def analyzed(self, fields, fc):
        r = bot.analyze(fields, TIMES)
        alerts.apply_sources(r, fc, None, RT)
        return r

    def test_far_system_is_watch_candidate_not_alert(self):
        r = self.analyzed(far_system(), wet_forecast())
        self.assertIsNone(r["stage"])
        self.assertIsNotNone(r["watch"])
        self.assertGreater(r["watch"]["eta_mid"], config.EARLY_MIN)
        self.assertTrue(alerts.watch_ok(r))
        self.assertTrue(alerts.watch_decision({}, r, RT))

    def test_watch_needs_forecast_agreement(self):
        self.assertFalse(alerts.watch_ok(self.analyzed(far_system(), dry_forecast())))
        self.assertFalse(alerts.watch_ok(self.analyzed(far_system(), None)))

    def test_watch_rejects_weakening_system(self):
        fields = [blob(-55 - 30 * (3 - k) / 6, 0, 12, d) for k, d in enumerate((42, 39, 35, 30))]
        r = self.analyzed(fields, wet_forecast())
        self.assertEqual(r["watch"]["trend"], "down")
        self.assertFalse(alerts.watch_ok(r))

    def test_watch_rejects_small_system(self):
        self.assertFalse(alerts.watch_ok(self.analyzed(far_system(r=7), wet_forecast())))

    def test_too_far_is_not_watch(self):
        self.assertIsNone(self.analyzed(far_system(x=-80), wet_forecast())["watch"])

    def test_watch_at_most_every_3_hours(self):
        r = self.analyzed(far_system(), wet_forecast())
        st = {"last_watch": (RT - timedelta(hours=2)).isoformat()}
        self.assertFalse(alerts.watch_decision(st, r, RT))
        st = {"last_watch": (RT - timedelta(hours=4)).isoformat()}
        self.assertTrue(alerts.watch_decision(st, r, RT))

    def test_no_watch_right_after_alert(self):
        r = self.analyzed(far_system(), wet_forecast())
        self.assertFalse(alerts.watch_decision({"last_alert": (RT - timedelta(minutes=30)).isoformat()}, r, RT))

    def test_close_system_is_alert_not_watch(self):
        r = self.analyzed(far_system(x=-37), wet_forecast())
        self.assertEqual(r["stage"], "early")
        self.assertFalse(alerts.watch_decision({}, r, RT))

    def test_watch_message(self):
        r = self.analyzed(far_system(), wet_forecast())
        msg = messages.watch_message(r, RT, RT)
        self.assertIn("מעקב", msg)
        self.assertIn("ממערב", msg)
        self.assertIn("התראה מוקדמת", msg)
        self.assertLess(len(msg), 1000)

    def test_headsup_is_skipped_after_watch(self):
        fsum = {"ok": True, "prob": 90, "mm_3h": 5.0, "next_rain": None}
        st = {"last_watch": (RT - timedelta(hours=1)).isoformat()}
        self.assertFalse(alerts.headsup_decision(st, fsum, RT, {"stage": None, "score": 0, "min_score": 3}))


# ------------------------------------------------------------------ הגשם מתחזק

class IntensifyTests(unittest.TestCase):
    T = at("2026-11-25T18:00:00")

    def after_now_alert(self, peak):
        st = {}
        alerts.update_state(st, res_now(peak), self.T, self.T, sent=True)
        alerts.update_state(st, res_now(peak), self.T + timedelta(minutes=10), self.T, sent=False)
        return st

    def test_heavy_rain_after_light_alert_is_sent(self):
        st = self.after_now_alert(28.0)
        self.assertEqual(st["last_alert_peak"], 28.0)
        self.assertEqual(alerts.send_decision(st, "now", self.T + timedelta(minutes=15), peak=44.0),
                         (True, "intensify"))

    def test_only_once(self):
        st = self.after_now_alert(28.0)
        t = self.T + timedelta(minutes=15)
        alerts.update_state(st, res_now(44.0), t, t, sent=True)
        self.assertEqual(alerts.send_decision(st, "now", t + timedelta(minutes=10), peak=46.0),
                         (False, "cooldown"))

    def test_not_when_already_heavy(self):
        st = self.after_now_alert(41.0)
        self.assertEqual(alerts.send_decision(st, "now", self.T + timedelta(minutes=15), peak=45.0),
                         (False, "cooldown"))

    def test_moderate_increase_is_silent(self):
        st = self.after_now_alert(28.0)
        self.assertEqual(alerts.send_decision(st, "now", self.T + timedelta(minutes=15), peak=36.0),
                         (False, "cooldown"))

    def test_same_frame_still_blocks(self):
        st = self.after_now_alert(28.0)
        radar = self.T.isoformat(timespec="seconds")
        st["last_alert_radar"] = radar
        self.assertEqual(alerts.send_decision(st, "now", self.T + timedelta(minutes=1), radar, peak=45.0)[1],
                         "same_frame")

    def test_intensify_message(self):
        r = {**res_now(45.0), "trend": "up", "motion_used": None}
        msg = messages.build_message(r, self.T, self.T, why="intensify")
        self.assertIn("מתחזק", msg)
        self.assertIn("גשם חזק", msg)
        self.assertNotIn("כביסה", msg)


# ------------------------------------------------------------------ חוסן

class TelegramRobustnessTests(unittest.TestCase):

    def setUp(self):
        self.p = [mock.patch.object(config, "TELEGRAM_BOT_TOKEN", "t"),
                  mock.patch.object(config, "TELEGRAM_CHAT_ID", "c")]
        for p in self.p:
            p.start()

    def tearDown(self):
        for p in self.p:
            p.stop()

    def test_rate_limit_waits_and_retries(self):
        limited = mock.Mock(status_code=429, text="Too Many Requests",
                            json=lambda: {"parameters": {"retry_after": 3}})
        ok = mock.Mock(status_code=200)
        with mock.patch.object(bot.requests, "post", side_effect=[limited, ok]) as post, \
                mock.patch.object(telegram.time, "sleep") as sleep:
            self.assertEqual(telegram.send_telegram("hi"), "ok")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(3.0)

    def test_long_rate_limit_is_not_waited(self):
        limited = mock.Mock(status_code=429, text="x", json=lambda: {"parameters": {"retry_after": 300}})
        with mock.patch.object(bot.requests, "post", return_value=limited) as post, \
                mock.patch.object(telegram.time, "sleep") as sleep:
            self.assertEqual(telegram.send_telegram("hi"), "fail")
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_broken_html_falls_back_to_plain_text(self):
        bad = mock.Mock(status_code=400, text="Bad Request: can't parse entities")
        ok = mock.Mock(status_code=200)
        with mock.patch.object(bot.requests, "post", side_effect=[bad, ok]) as post:
            self.assertEqual(telegram.send_telegram("<b>גשם &amp; רוח"), "ok")
        second = post.call_args_list[1].kwargs["json"]
        self.assertEqual(second["text"], "גשם & רוח")
        self.assertNotIn("parse_mode", second)

    def test_photo_with_broken_html_goes_to_text(self):
        bad = mock.Mock(status_code=400, text="Bad Request: can't parse entities")
        ok = mock.Mock(status_code=200)
        with mock.patch.object(bot.requests, "post", side_effect=[bad, bad, ok]) as post:
            self.assertEqual(telegram.deliver(b"png", "<b>x"), "ok")
        self.assertEqual(post.call_count, 3)


class TestTelegramCommandTests(unittest.TestCase):
    """--test-telegram: ReadTimeout (כנראה נשלח) לא מפיל את הריצה (ריצה 13405)"""

    def run_with(self, *responses):
        with mock.patch.object(config, "TELEGRAM_BOT_TOKEN", "t"), \
                mock.patch.object(config, "TELEGRAM_CHAT_ID", "c"), \
                mock.patch.object(bot.requests, "post", side_effect=list(responses)):
            return main.test_telegram()

    def test_slow_telegram_is_not_a_failure(self):
        self.assertEqual(self.run_with(bot.requests.exceptions.ReadTimeout(), mock.Mock(status_code=200)), 0)

    def test_all_ok(self):
        self.assertEqual(self.run_with(mock.Mock(status_code=200), mock.Mock(status_code=200)), 0)

    def test_real_failure_still_fails(self):
        self.assertEqual(self.run_with(mock.Mock(status_code=401, text="Unauthorized"),
                                       mock.Mock(status_code=200)), 1)


class RadarIndexTests(unittest.TestCase):

    def test_frames_sorted_and_deduplicated(self):
        data = {"host": "h", "radar": {"past": [{"time": 3, "path": "/c"}, {"time": 1, "path": "/a"},
                                                {"time": 2, "path": "/b"}, {"time": 3, "path": "/c"},
                                                {"time": 4, "path": "/d"}, {"time": 5}]}}
        with mock.patch.object(radar, "http_get", return_value=mock.Mock(json=lambda: data)):
            host, frames = radar.fetch_frames_index(mock.Mock())
        self.assertEqual([f["time"] for f in frames], [1, 2, 3, 4])


# ------------------------------------------------------------------ נוסח ההודעות

class WordingTests(unittest.TestCase):
    NOW = at("2026-11-25T20:00:00")
    M = {"vx": 30.0, "vy": 0.0, "speed": 30.0}

    def res(self, stage, src="system", **kw):
        r = {"stage": stage, "src": src, "eta": (0, 0) if stage == "now" else (15, 30), "front_km": 12.0,
             "peak_dbz": 33.0, "score": 3, "checks": {"motion": True, "size": True, "persist": True},
             "motion_used": self.M, "trend": "up", "forecast_agree": True}
        r.update(kw)
        return r

    def all_messages(self):
        gauge = {"ok": True, "mm": 0.6, "station": "בית דגן"}
        out = {}
        for stage in ("early", "imminent", "now"):
            out[f"system_{stage}"] = messages.build_message(self.res(stage), self.NOW, self.NOW, gauge)
            out[f"cell_{stage}"] = messages.build_message(self.res(stage, "cell", cell={"n": 3}), self.NOW, self.NOW)
        out["popup"] = messages.build_message(self.res("imminent", "cell", cell={"n": 2, "popup": True}),
                                              self.NOW, self.NOW)
        out["intensify"] = messages.build_message(self.res("now", peak_dbz=45.0), self.NOW, self.NOW, why="intensify")
        out["watch"] = messages.watch_message({"watch": {"eta": (60, 90), "peak_dbz": 35.0, "front_km": 45.0,
                                                         "trend": "steady"}, "motion": self.M}, self.NOW, self.NOW)
        out["radar_down"] = messages.radar_down_message(self.NOW, "HTTP 503")
        out["radar_up"] = messages.radar_up_message(self.NOW)
        out["headsup"] = messages.headsup_message({"ok": True, "prob": 85, "mm_3h": 4.2,
                                                   "next_rain": "2026-11-25T21:15+02:00"})
        out["rain_end"] = messages.rain_end_message({"ok": True, "prob": 60})
        out["test"] = messages.test_message()
        return out

    def test_every_message_is_short_and_has_a_bold_headline(self):
        for kind, msg in self.all_messages().items():
            with self.subTest(kind=kind):
                self.assertLess(len(msg), 700)
                self.assertIn("<b>", msg.splitlines()[0])
                self.assertEqual(msg.count("<b>"), msg.count("</b>"))
                self.assertEqual(msg.count("<i>"), msg.count("</i>"))

    def test_alerts_show_clock_time_and_minutes(self):
        msg = self.all_messages()["system_early"]
        self.assertIn("בין 20:15 ל-20:30", msg)
        self.assertIn("בעוד 15-30 דק'", msg)
        self.assertIn("ממערב", msg)
        self.assertIn("מתחזק", msg)
        self.assertIn("ודאות גבוהה", msg)

    def test_imminent_has_practical_tip(self):
        self.assertIn("מטרייה", self.all_messages()["system_imminent"])

    def test_cell_messages_say_it_may_be_short(self):
        self.assertIn("יכול להיות קצר", self.all_messages()["cell_early"])

    def test_now_has_no_arrival_line(self):
        msg = self.all_messages()["system_now"]
        self.assertNotIn("מגיע <b>", msg)
        self.assertIn("בית דגן", msg)

    def test_plain_text_fallback_is_readable(self):
        for kind, msg in self.all_messages().items():
            with self.subTest(kind=kind):
                self.assertNotIn("<", telegram.plain_text(msg))

    def test_write_examples_file(self):
        """שומר דוגמאות לכל ההודעות (לבדיקה בעין): tests/out/messages.txt אם WRITE_EXAMPLES=1"""
        if os.environ.get("WRITE_EXAMPLES") != "1":
            self.skipTest("WRITE_EXAMPLES לא מוגדר")
        path = os.path.join(os.path.dirname(__file__), "out", "messages.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for kind, msg in self.all_messages().items():
                f.write(f"===== {kind}\n{msg}\n\n")


# ------------------------------------------------------------------ ריצה מלאה

class RunWatchTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.patches = [mock.patch.object(config, "STATE_DIR", self.dir),
                        mock.patch.object(config, "STATE_FILE", os.path.join(self.dir, "state.json")),
                        mock.patch.object(config, "HISTORY_FILE", os.path.join(self.dir, "history.jsonl")),
                        mock.patch.object(config, "STATUS_DIR", os.path.join(self.dir, "status")),
                        mock.patch.object(config, "TELEGRAM_BOT_TOKEN", "t"),
                        mock.patch.object(config, "TELEGRAM_CHAT_ID", "c"),
                        mock.patch.object(main, "OUT_DIR", os.path.join(self.dir, "out")),
                        mock.patch.object(main.forecast, "get_forecast", return_value=wet_forecast()),
                        mock.patch.object(main.ims, "fetch_gauge", return_value=None),
                        mock.patch.object(main, "get_radar",
                                          return_value=(far_system(), [{"time": t} for t in TIMES], None))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_watch_sent_once_and_shown_on_dashboard(self):
        ok = mock.Mock(status_code=200)
        with mock.patch.object(bot.requests, "post", return_value=ok) as post:
            self.assertEqual(main.run(), 0)
            self.assertEqual(main.run(), 0)
        self.assertEqual(post.call_count, 1)
        self.assertIn("מעקב", post.call_args.kwargs["json"]["text"])
        st = storage.load_state()
        self.assertIn("last_watch", st)
        self.assertNotIn("last_alert", st)          # מעקב לא נחשב התראה ולא משפיע על ההשתקה
        with open(os.path.join(config.STATUS_DIR, "status.json")) as f:
            status = json.load(f)
        self.assertIsNone(status["stage"])
        # 3.2: זמני ההגעה עוברים כיול לפי הלמידה (learn.apply), גם במעקב
        expected = bot.analyze(far_system(), TIMES, spread=learn.params({})["system"]["spread"])
        learn.apply(expected, {})
        self.assertEqual(status["watch"]["eta"], list(expected["watch"]["eta"]))
        with open(config.HISTORY_FILE) as f:
            last = json.loads(f.read().splitlines()[0])
        self.assertEqual(last["extra"], ["watch"])
        self.assertTrue(last["watch"]["ok"])


if __name__ == "__main__":
    unittest.main()
