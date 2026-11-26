"""בדיקות לחידושים של 3.1: מקורות נוספים, אמינות, מגמה, תאים שמתפתחים במקום, הודעות מערכת.
רצות בלי רשת ובלי טלגרם."""
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import bot  # noqa: E402
from rainbot import alerts, analysis, config, forecast, ims, main, messages, radar, storage  # noqa: E402

TZ = bot.TZ
T0 = 1_760_000_000
TIMES = [T0 + 600 * k for k in range(4)]
EMPTY = np.zeros(bot.XX.shape, np.float32)


def at(s):
    return datetime.fromisoformat(s).replace(tzinfo=TZ)


def blob(cx, cy, r, dbz):
    f = np.zeros(bot.XX.shape, np.float32)
    f[(bot.XX - cx) ** 2 + (bot.YY - cy) ** 2 <= r * r] = dbz
    return f


def fc_with(start, mm_by_slot, prob=80):
    """תחזית מדומה: משבצות של 15 דקות מ-start"""
    slots = [[(start + timedelta(minutes=15 * k)).strftime("%Y-%m-%dT%H:%M"), v] for k, v in enumerate(mm_by_slot)]
    hourly = [[(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00"), prob, 1.0] for h in range(6)]
    return {"ok": True, "t": start.isoformat(), "slots": slots, "hourly": hourly}


def res_for(stage, score=3, min_score=3, src="system", eta=None, checks=None):
    return {"stage": stage, "score": score, "min_score": min_score, "src": src,
            "eta": eta or ((0, 0) if stage == "now" else (10, 20)),
            "checks": checks if checks is not None else {"motion": True, "size": True, "persist": True},
            "diag": {"max_dbz_10km": 0.0}}


# ------------------------------------------------------------------ מכ"ם

class RadarTests(unittest.TestCase):

    def jobs(self, n, bad=()):
        tile = np.zeros((config.TILE_SIZE, config.TILE_SIZE), np.float32)
        return {(f, tx, ty): (None if (f, tx, ty) in bad else tile)
                for f in range(n) for tx in range(radar.TX0, radar.TX1 + 1) for ty in range(radar.TY0, radar.TY1 + 1)}

    def test_all_frames_kept(self):
        fields, keep = radar.assemble(self.jobs(4), 4)
        self.assertEqual(keep, [0, 1, 2, 3])
        self.assertEqual(fields[0].shape, bot.XX.shape)

    def test_frame_with_missing_tile_is_dropped(self):
        fields, keep = radar.assemble(self.jobs(4, bad={(1, radar.TX0, radar.TY0)}), 4)
        self.assertEqual(keep, [0, 2, 3])
        self.assertEqual(len(fields), 3)

    def test_too_many_broken_frames_fails(self):
        bad = {(0, radar.TX0, radar.TY0), (3, radar.TX0, radar.TY0)}
        with self.assertRaises(RuntimeError):
            radar.assemble(self.jobs(4, bad=bad), 4)

    def test_http_get_does_not_sleep_after_last_try(self):
        session = mock.Mock()
        session.get.side_effect = requests.exceptions.ConnectionError()
        with mock.patch.object(radar.time, "sleep") as sleep, self.assertRaises(RuntimeError):
            radar.http_get(session, "http://x", tries=3)
        self.assertEqual(sleep.call_count, 2)

    def test_get_radar_reports_stale_frame(self):
        now = datetime.fromtimestamp(TIMES[-1], TZ) + timedelta(minutes=60)
        with mock.patch.object(main, "fetch_frames_index", return_value=("h", [{"time": t} for t in TIMES])), \
                mock.patch.object(main, "load_fields", return_value=([EMPTY] * 4, [{"time": t} for t in TIMES])):
            fields, frames, err = main.get_radar(mock.Mock(), now)
        self.assertIsNone(fields)
        self.assertIn("ישן", err)


# ------------------------------------------------------------------ ניתוח

class TrendTests(unittest.TestCase):

    def test_classify(self):
        self.assertEqual(analysis.classify_trend([25, 26, 31]), "up")
        self.assertEqual(analysis.classify_trend([35, 34, 28]), "down")
        self.assertEqual(analysis.classify_trend([30, 31, 30]), "steady")
        self.assertIsNone(analysis.classify_trend([30]))

    def test_system_trend_up_when_intensifying(self):
        v = 30.0
        fields = [blob(-37 - v * (3 - k) / 6, 0, 12, 30 + 4 * k) for k in range(4)]
        r = bot.analyze(fields, TIMES)
        self.assertEqual(r["stage"], "early")
        self.assertEqual(r["trend"], "up")

    def test_system_trend_steady(self):
        fields = [blob(-37 - 30 * (3 - k) / 6, 0, 12, 35) for k in range(4)]
        self.assertEqual(bot.analyze(fields, TIMES)["trend"], "steady")

    def test_decaying_cell_does_not_alert(self):
        fields = [blob(-14 - 6 * (3 - k), 2, 3, dbz) for k, dbz in enumerate((34, 32, 29, 24))]
        r = bot.analyze(fields, TIMES)
        self.assertFalse(bot.qualifies(r))
        self.assertEqual(r["cell_dbg"]["why"], "decaying")

    def test_cell_alert_carries_trend(self):
        fields = [blob(-14 - 6 * (3 - k), 2, 3, dbz) for k, dbz in enumerate((26, 28, 31, 35))]
        r = bot.analyze(fields, TIMES)
        self.assertTrue(bot.qualifies(r))
        self.assertEqual(r["trend"], "up")


class PopupTests(unittest.TestCase):

    def test_cell_forming_over_target_alerts_now(self):
        fields = [EMPTY, EMPTY, blob(-1, 1, 3, 33), blob(-1, 1, 3, 36)]
        r = bot.analyze(fields, TIMES)
        self.assertEqual((r["stage"], r["src"]), ("now", "cell"))
        self.assertTrue(r["cell"]["popup"])
        self.assertTrue(bot.qualifies(r))

    def test_cell_forming_next_to_target_alerts_imminent(self):
        fields = [EMPTY, EMPTY, blob(-9, 0, 3, 34), blob(-9, 0, 3, 35)]
        r = bot.analyze(fields, TIMES)
        self.assertEqual(r["stage"], "imminent")
        self.assertEqual(r["cell_dbg"]["why"], "popup")

    def test_persistent_echo_is_clutter_not_popup(self):
        f = blob(-1, 1, 3, 35)
        r = bot.analyze([f, f, f, f], TIMES)
        self.assertFalse(bot.qualifies(r))

    def test_weak_new_cell_is_not_popup(self):
        r = bot.analyze([EMPTY, EMPTY, blob(-1, 1, 3, 25), blob(-1, 1, 3, 26)], TIMES)
        self.assertFalse(bot.qualifies(r))

    def test_far_new_cell_is_not_popup(self):
        r = bot.analyze([EMPTY, EMPTY, blob(-25, 0, 3, 35), blob(-25, 0, 3, 35)], TIMES)
        self.assertFalse(bot.qualifies(r))

    def test_single_frame_cell_is_not_popup(self):
        r = bot.analyze([EMPTY, EMPTY, EMPTY, blob(-1, 1, 3, 40)], TIMES)
        self.assertFalse(bot.qualifies(r))


# ------------------------------------------------------------------ תחזית

class ForecastTests(unittest.TestCase):
    NOW = at("2026-11-25T18:00:00")

    def test_parse_response(self):
        data = {"minutely_15": {"time": ["2026-11-25T18:15", "2026-11-25T18:30"], "precipitation": [0.0, 0.4]},
                "hourly": {"time": ["2026-11-25T18:00"], "precipitation_probability": [70], "precipitation": [0.6]}}
        fc = forecast.parse_response(data, self.NOW)
        self.assertTrue(fc["ok"])
        self.assertEqual(fc["slots"][1], ["2026-11-25T18:30", 0.4])

    def test_rain_between(self):
        fc = fc_with(self.NOW, [0, 0, 0.5, 0, 0, 0, 0, 0])      # גשם במשבצת 18:30
        self.assertTrue(forecast.rain_between(fc, self.NOW + timedelta(minutes=20), self.NOW + timedelta(minutes=30)))
        self.assertFalse(forecast.rain_between(fc, self.NOW + timedelta(minutes=80), self.NOW + timedelta(minutes=95)))
        self.assertIsNone(forecast.rain_between(None, self.NOW, self.NOW))

    def test_rain_between_with_probability(self):
        fc = fc_with(self.NOW, [0, 0.5, 0.5, 0], prob=40)
        self.assertFalse(forecast.rain_between(fc, self.NOW, self.NOW + timedelta(minutes=30), mm=0.2, prob=60))

    def test_summary(self):
        fc = fc_with(self.NOW, [0, 0, 0.5, 0.7, 0.3, 0, 0, 0], prob=85)
        s = forecast.summary(fc, self.NOW)
        self.assertEqual((s["ok"], s["prob"], s["mm_3h"]), (True, 85, 1.5))
        self.assertEqual(s["next_rain"], "2026-11-25T18:15+02:00")

    def test_cache_is_used(self):
        state = {"forecast": {**fc_with(self.NOW, [0]), "t": self.NOW.isoformat()}}
        session = mock.Mock()
        with mock.patch.object(forecast, "FORECAST_ENABLED", True):
            fc = forecast.get_forecast(session, state, self.NOW + timedelta(minutes=5))
        session.get.assert_not_called()
        self.assertTrue(fc["ok"])

    def test_failure_falls_back_to_recent_cache(self):
        state = {"forecast": {**fc_with(self.NOW, [0]), "t": self.NOW.isoformat()}}
        session = mock.Mock()
        session.get.side_effect = requests.exceptions.ConnectionError()
        with mock.patch.object(forecast, "FORECAST_ENABLED", True):
            fc = forecast.get_forecast(session, state, self.NOW + timedelta(minutes=40))
        self.assertTrue(fc["ok"] and fc["stale"])

    def test_failure_without_cache(self):
        session = mock.Mock()
        session.get.return_value = mock.Mock(status_code=503)
        with mock.patch.object(forecast, "FORECAST_ENABLED", True):
            fc = forecast.get_forecast(session, {}, self.NOW)
        self.assertEqual(fc, {"ok": False, "err": "HTTP 503"})
        self.assertFalse(forecast.summary(fc, self.NOW)["ok"])


# ------------------------------------------------------------------ מד גשם

class ImsTests(unittest.TestCase):
    NOW = at("2026-11-25T18:12:00")

    def data(self, t="2026-11-25T18:10:00+02:00", rain=0.3):
        return {"stationId": 54, "data": [{"datetime": t, "channels": [
            {"name": "TD", "value": 15.1, "valid": True}, {"name": "Rain", "value": rain, "valid": True}]}]}

    def test_parse_latest(self):
        g = ims.parse_latest(self.data(), self.NOW, "54")
        self.assertEqual((g["ok"], g["mm"], g["station"]), (True, 0.3, "בית דגן"))
        self.assertTrue(ims.is_wet(g))

    def test_stale_reading_is_ignored(self):
        g = ims.parse_latest(self.data(t="2026-11-25T16:00:00+02:00"), self.NOW)
        self.assertFalse(g["ok"])
        self.assertFalse(ims.is_wet(g))

    def test_dry_gauge(self):
        self.assertFalse(ims.is_wet(ims.parse_latest(self.data(rain=0.0), self.NOW)))

    def test_no_token_means_disabled(self):
        self.assertIsNone(ims.fetch_gauge(mock.Mock(), self.NOW, token=None))

    def test_fetch_error_is_not_fatal(self):
        session = mock.Mock()
        session.get.side_effect = requests.exceptions.ReadTimeout()
        g = ims.fetch_gauge(session, self.NOW, token="secret")
        self.assertFalse(g["ok"])
        self.assertNotIn("secret", json.dumps(g))

    def test_fetch_sends_token_header(self):
        session = mock.Mock()
        session.get.return_value = mock.Mock(status_code=200, json=lambda: self.data())
        g = ims.fetch_gauge(session, self.NOW, token="abc", station="54")
        self.assertTrue(g["ok"])
        self.assertEqual(session.get.call_args.kwargs["headers"], {"Authorization": "ApiToken abc"})


# ------------------------------------------------------------------ חיזוק ודאות

class SourceBoostTests(unittest.TestCase):
    RT = at("2026-11-25T18:00:00")
    WET = {"ok": True, "mm": 0.2, "station": "בית דגן"}

    def test_gauge_confirms_borderline_now(self):
        r = alerts.apply_sources(res_for("now", score=2, src="cell"), None, self.WET, self.RT)
        self.assertEqual(r["boost"], "ims")
        self.assertTrue(alerts.qualifies(r))
        self.assertEqual(r["score"], 2)            # הציון נשאר כן

    def test_dry_gauge_does_not_boost(self):
        r = alerts.apply_sources(res_for("now", score=2), None, {"ok": True, "mm": 0.0}, self.RT)
        self.assertFalse(alerts.qualifies(r))

    def test_gauge_does_not_rescue_score_1(self):
        r = alerts.apply_sources(res_for("now", score=1), None, self.WET, self.RT)
        self.assertFalse(alerts.qualifies(r))

    def test_gauge_does_not_boost_approaching_rain(self):
        r = alerts.apply_sources(res_for("early", score=2), None, self.WET, self.RT)
        self.assertIsNone(r["boost"])

    def test_forecast_confirms_system_score_2(self):
        fc = fc_with(self.RT, [0, 0.4, 0.6, 0.3, 0, 0, 0, 0], prob=80)
        r = res_for("early", score=2, eta=(15, 30), checks={"motion": True, "size": False, "persist": True})
        alerts.apply_sources(r, fc, None, self.RT)
        self.assertEqual(r["boost"], "forecast")
        self.assertTrue(r["forecast_agree"])
        self.assertTrue(alerts.qualifies(r))

    def test_forecast_never_rescues_bad_motion(self):
        fc = fc_with(self.RT, [0, 0.4, 0.6, 0.3], prob=80)
        r = res_for("early", score=2, eta=(15, 30), checks={"motion": False, "size": True, "persist": True})
        self.assertFalse(alerts.qualifies(alerts.apply_sources(r, fc, None, self.RT)))

    def test_forecast_disagreeing_does_not_boost(self):
        fc = fc_with(self.RT, [0] * 8, prob=80)
        r = res_for("early", score=2, eta=(15, 30), checks={"motion": True, "size": False, "persist": True})
        alerts.apply_sources(r, fc, None, self.RT)
        self.assertFalse(alerts.qualifies(r))
        self.assertFalse(r["forecast_agree"])

    def test_forecast_does_not_boost_cells(self):
        fc = fc_with(self.RT, [0, 0.4, 0.6, 0.3], prob=80)
        r = res_for("early", score=2, src="cell", eta=(15, 30), checks={})
        self.assertFalse(alerts.qualifies(alerts.apply_sources(r, fc, None, self.RT)))

    def test_qualifying_alert_untouched(self):
        r = alerts.apply_sources(res_for("imminent"), None, None, self.RT)
        self.assertIsNone(r["boost"])
        self.assertIsNone(r["forecast_agree"])


# ------------------------------------------------------------------ הודעות מערכת

class HealthTests(unittest.TestCase):

    def test_down_after_30_minutes_once_then_up(self):
        st, t = {}, at("2026-11-25T18:00:00")
        self.assertIsNone(alerts.radar_health(st, False, t))
        self.assertIsNone(alerts.radar_health(st, False, t + timedelta(minutes=25)))
        self.assertEqual(alerts.radar_health(st, False, t + timedelta(minutes=30)), "down")
        alerts.mark_health(st, "down", True)
        self.assertIsNone(alerts.radar_health(st, False, t + timedelta(minutes=35)))
        self.assertEqual(alerts.radar_health(st, True, t + timedelta(minutes=40)), "up")
        alerts.mark_health(st, "up", True)
        self.assertIsNone(alerts.radar_health(st, True, t + timedelta(minutes=45)))
        self.assertNotIn("radar_fail_since", st)

    def test_short_outage_is_silent(self):
        st, t = {}, at("2026-11-25T18:00:00")
        alerts.radar_health(st, False, t)
        self.assertIsNone(alerts.radar_health(st, True, t + timedelta(minutes=10)))

    def test_failed_down_message_is_retried(self):
        st, t = {}, at("2026-11-25T18:00:00")
        alerts.radar_health(st, False, t)
        self.assertEqual(alerts.radar_health(st, False, t + timedelta(minutes=30)), "down")
        alerts.mark_health(st, "down", False)
        self.assertEqual(alerts.radar_health(st, False, t + timedelta(minutes=35)), "down")


class HeadsupTests(unittest.TestCase):
    NOW = at("2026-11-25T18:00:00")
    WET = {"ok": True, "prob": 85, "mm_3h": 3.2, "next_rain": "2026-11-25T19:15+02:00"}

    def test_sent_when_forecast_is_wet(self):
        self.assertTrue(alerts.headsup_decision({}, self.WET, self.NOW, res_for(None)))

    def test_not_twice_in_12_hours(self):
        st = {"last_headsup": (self.NOW - timedelta(hours=6)).isoformat()}
        self.assertFalse(alerts.headsup_decision(st, self.WET, self.NOW, res_for(None)))

    def test_not_after_recent_radar_alert(self):
        st = {"last_alert": (self.NOW - timedelta(hours=1)).isoformat()}
        self.assertFalse(alerts.headsup_decision(st, self.WET, self.NOW, res_for(None)))

    def test_not_when_forecast_is_weak(self):
        self.assertFalse(alerts.headsup_decision({}, {**self.WET, "prob": 50}, self.NOW, res_for(None)))
        self.assertFalse(alerts.headsup_decision({}, {**self.WET, "mm_3h": 0.4}, self.NOW, res_for(None)))
        self.assertFalse(alerts.headsup_decision({}, None, self.NOW, res_for(None)))

    def test_not_when_radar_alert_is_active(self):
        self.assertFalse(alerts.headsup_decision({}, self.WET, self.NOW, res_for("early")))


class RainEndTests(unittest.TestCase):
    T = at("2026-11-25T18:00:00")

    def after_now_alert(self, last_now_seen_min=10):
        st = {}
        alerts.update_state(st, res_for("now"), self.T, self.T, sent=True)
        alerts.update_state(st, res_for("now"), self.T + timedelta(minutes=last_now_seen_min), self.T, sent=False)
        return st

    def test_sent_once_when_rain_is_over(self):
        st = self.after_now_alert()
        dry = res_for(None)
        self.assertFalse(alerts.rain_end_decision(st, dry, self.T + timedelta(minutes=30)))   # רק 20 דקות
        self.assertTrue(alerts.rain_end_decision(st, dry, self.T + timedelta(minutes=35)))
        alerts.mark_rain_end(st, self.T + timedelta(minutes=35), True)
        self.assertFalse(alerts.rain_end_decision(st, dry, self.T + timedelta(minutes=40)))

    def test_not_while_echo_nearby(self):
        st = self.after_now_alert()
        r = res_for(None)
        r["diag"]["max_dbz_10km"] = 22.0
        self.assertFalse(alerts.rain_end_decision(st, r, self.T + timedelta(minutes=40)))

    def test_not_while_more_rain_approaches(self):
        st = self.after_now_alert()
        self.assertFalse(alerts.rain_end_decision(st, res_for("early", score=2), self.T + timedelta(minutes=40)))

    def test_not_without_now_alert(self):
        st = {}
        alerts.update_state(st, res_for("early"), self.T, self.T, sent=True)
        self.assertFalse(alerts.rain_end_decision(st, res_for(None), self.T + timedelta(minutes=60)))

    def test_new_now_after_end_allows_another_end(self):
        st = self.after_now_alert()
        alerts.mark_rain_end(st, self.T + timedelta(minutes=35), True)
        t2 = self.T + timedelta(hours=2)
        alerts.update_state(st, res_for("now"), t2, t2, sent=True)
        self.assertTrue(alerts.rain_end_decision(st, res_for(None), t2 + timedelta(minutes=30)))


# ------------------------------------------------------------------ קבצים

class StorageTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.patches = [mock.patch.object(config, "STATE_DIR", self.dir),
                        mock.patch.object(config, "STATE_FILE", os.path.join(self.dir, "state.json")),
                        mock.patch.object(config, "HISTORY_FILE", os.path.join(self.dir, "history.jsonl")),
                        mock.patch.object(config, "STATUS_DIR", os.path.join(self.dir, "status"))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_snapshot_roundtrip(self):
        fields = [blob(-10, 0, 5, 33.0), blob(-5, 0, 5, 41.0), EMPTY]
        path = os.path.join(self.dir, "x.npz")
        storage.save_snapshot(path, fields, TIMES[:3], {"stage": "early"})
        f2, t2, meta = storage.load_snapshot(path)
        self.assertEqual(t2, TIMES[:3])
        self.assertEqual(meta["stage"], "early")
        np.testing.assert_array_equal(f2[1], fields[1])

    def test_status_keeps_30_fields_and_adds_31(self):
        fields = [blob(-37 - 30 * (3 - k) / 6, 0, 12, 35) for k in range(4)]
        r = bot.analyze(fields, TIMES)
        alerts.apply_sources(r, None, None, at("2026-11-25T18:00:00"))
        now = at("2026-11-25T18:03:00")
        st = storage.write_status(r, {"nearest15_km": 20.0, "max_dbz_10km": 0.0}, {"alerts": []}, now, now,
                                  {"radar": {"ok": True, "age_min": 3.0}, "forecast": None, "ims": None}, True)
        for k in ("updated", "radar", "version", "stage", "src", "eta", "score", "peak_dbz", "front_km", "speed",
                  "heading", "nearest15_km", "max_dbz_10km", "alerts", "trend"):
            self.assertIn(k, st)
        for k in ("sources", "intensity_trend", "forecast_agree", "boost", "radar_down_since", "rain_ended_at"):
            self.assertIn(k, st)
        self.assertEqual(st["stage"], "early")
        self.assertEqual(st["intensity_trend"], "steady")
        with open(os.path.join(config.STATUS_DIR, "status.json")) as f:
            self.assertEqual(json.load(f)["version"], config.VERSION)

    def test_status_when_radar_is_down(self):
        now = at("2026-11-25T18:03:00")
        st = storage.write_status(None, {}, {"radar_fail_since": "2026-11-25T17:30:00+02:00"}, None, now)
        self.assertIsNone(st["stage"])
        self.assertEqual(st["radar_down_since"], "2026-11-25T17:30:00+02:00")

    def test_rain_ended_hidden_after_new_alert(self):
        now = at("2026-11-25T18:03:00")
        st = {"rain_ended_at": "2026-11-25T17:00:00+02:00", "last_alert": "2026-11-25T17:30:00+02:00"}
        self.assertIsNone(storage.write_status(None, {}, st, None, now)["rain_ended_at"])
        st["last_alert"] = "2026-11-25T16:30:00+02:00"
        self.assertEqual(storage.write_status(None, {}, st, None, now)["rain_ended_at"], st["rain_ended_at"])

    def test_corrupt_state_file(self):
        with open(config.STATE_FILE, "w") as f:
            f.write("[1, 2")
        self.assertEqual(storage.load_state(), {})


# ------------------------------------------------------------------ הודעות

class MessageTests(unittest.TestCase):
    NOW = at("2026-11-07T21:00:00")

    def system_res(self, **kw):
        r = {"stage": "early", "src": "system", "eta": (15, 25), "front_km": 9.0, "peak_dbz": 35.0, "score": 3,
             "checks": {"motion": True, "size": True, "persist": True},
             "motion_used": {"vx": 20.0, "vy": 0.0, "speed": 20.0}}
        r.update(kw)
        return r

    def test_trend_forecast_and_gauge_lines(self):
        gauge = {"ok": True, "mm": 0.4, "station": "בית דגן"}
        msg = messages.build_message(self.system_res(trend="up", forecast_agree=True), self.NOW, self.NOW, gauge)
        self.assertIn("מתחזק", msg)
        self.assertIn("התחזית מסכימה", msg)
        self.assertIn("בית דגן", msg)

    def test_boost_is_shown(self):
        msg = messages.build_message(self.system_res(score=2, boost="forecast"), self.NOW, self.NOW)
        self.assertIn("בינונית", msg)
        self.assertIn("אושר בתחזית", msg)

    def test_popup_message(self):
        r = self.system_res(stage="imminent", src="cell", front_km=4.0,
                            cell={"popup": True, "n": 2}, motion_used={"vx": 0.0, "vy": 0.0, "speed": 0.0})
        msg = messages.build_message(r, self.NOW, self.NOW)
        self.assertIn("נוצר", msg)
        self.assertNotIn("מגיע מ", msg)

    def test_worst_case_caption_fits_telegram(self):
        r = self.system_res(stage="imminent", src="cell", cell={"n": 4}, trend="down", forecast_agree=False,
                            boost="ims")
        msg = messages.build_message(r, self.NOW, self.NOW, {"ok": True, "mm": 12.3, "station": "בית דגן"})
        self.assertLess(len(msg), 1000)

    def test_system_messages(self):
        self.assertIn("לא רואה מכ", messages.radar_down_message(self.NOW, "HTTP 503"))
        self.assertIn("חזר", messages.radar_up_message(self.NOW))
        h = messages.headsup_message({"ok": True, "prob": 85, "mm_3h": 3.2, "next_rain": "2026-11-07T22:15+02:00"})
        self.assertIn("85%", h)
        self.assertIn("22:15", h)
        self.assertIn("נגמר", messages.rain_end_message({"ok": True, "prob": 70}))
        self.assertIn("70%", messages.rain_end_message({"ok": True, "prob": 70}))


# ------------------------------------------------------------------ ריצה מלאה (בלי רשת)

class RunTests(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.patches = [mock.patch.object(config, "STATE_DIR", self.dir),
                        mock.patch.object(config, "STATE_FILE", os.path.join(self.dir, "state.json")),
                        mock.patch.object(config, "HISTORY_FILE", os.path.join(self.dir, "history.jsonl")),
                        mock.patch.object(config, "STATUS_DIR", os.path.join(self.dir, "status")),
                        mock.patch.object(config, "TELEGRAM_BOT_TOKEN", "t"),
                        mock.patch.object(config, "TELEGRAM_CHAT_ID", "c"),
                        mock.patch.object(main, "OUT_DIR", os.path.join(self.dir, "out")),
                        mock.patch.object(main.forecast, "get_forecast", return_value=None),
                        mock.patch.object(main.ims, "fetch_gauge", return_value=None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def radar(self, fields):
        now = datetime.fromtimestamp(TIMES[-1], TZ) + timedelta(minutes=3)
        frames = [{"time": t} for t in TIMES]
        return now, mock.patch.object(main, "get_radar", return_value=(fields, frames, None))

    def history(self):
        with open(config.HISTORY_FILE) as f:
            return [json.loads(x) for x in f.read().splitlines()]

    def test_alert_is_sent_and_logged(self):
        fields = [blob(-37 - 30 * (3 - k) / 6, 0, 12, 35) for k in range(4)]
        _, p = self.radar(fields)
        ok = mock.Mock(status_code=200)
        with p, mock.patch.object(bot.requests, "post", return_value=ok) as post:
            self.assertEqual(main.run(), 0)
        self.assertEqual(post.call_count, 1)
        h = self.history()[-1]
        self.assertEqual((h["sent"], h["sent_why"], h["stage"]), (True, "first", "early"))
        self.assertTrue(any(n.endswith(".npz") for n in os.listdir(os.path.join(self.dir, "out"))))
        with open(os.path.join(config.STATUS_DIR, "status.json")) as f:
            self.assertEqual(json.load(f)["stage"], "early")

    def test_no_rain_run(self):
        _, p = self.radar([EMPTY] * 4)
        with p, mock.patch.object(bot.requests, "post") as post:
            self.assertEqual(main.run(), 0)
        post.assert_not_called()
        self.assertEqual(self.history()[-1]["hold"], "no_rain")

    def test_brief_radar_outage_is_not_a_failed_run(self):
        """תקלה רגעית בהורדת המכ"ם: הריצה מצליחה (בלי מייל שגיאה מ-GitHub) ובלי הודעה בטלגרם"""
        with mock.patch.object(main, "get_radar", return_value=(None, None, "הורדה נכשלה (ConnectionError)")), \
                mock.patch.object(bot.requests, "post") as post:
            self.assertEqual(main.run(), 0)
        post.assert_not_called()
        self.assertIn("radar_fail_since", storage.load_state())

    def test_radar_down_message_after_30_minutes(self):
        ok = mock.Mock(status_code=200)
        storage.save_state({"radar_fail_since": (datetime.now(TZ) - timedelta(minutes=40)).isoformat()})
        with mock.patch.object(main, "get_radar", return_value=(None, None, "HTTP 503")), \
                mock.patch.object(bot.requests, "post", return_value=ok) as post:
            self.assertEqual(main.run(), 0)
        self.assertEqual(post.call_count, 1)
        self.assertIn("לא רואה מכ", post.call_args.kwargs["json"]["text"])
        self.assertTrue(storage.load_state()["radar_down_notified"])
        with open(os.path.join(config.STATUS_DIR, "status.json")) as f:
            self.assertIsNotNone(json.load(f)["radar_down_since"])

    def test_dry_run_saves_frames_without_state(self):
        fields = [blob(-37 - 30 * (3 - k) / 6, 0, 12, 35) for k in range(4)]
        _, p = self.radar(fields)
        path = os.path.join(self.dir, "f.npz")
        with p, mock.patch.object(bot.requests, "post") as post:
            self.assertEqual(main.run(dry_run=True, save_frames=path), 0)
        post.assert_not_called()
        self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(config.STATE_FILE))

    def test_replay_reproduces_decision(self):
        from rainbot import replay
        fields = [blob(-37 - 30 * (3 - k) / 6, 0, 12, 35) for k in range(4)]
        path = os.path.join(self.dir, "r.npz")
        storage.save_snapshot(path, fields, TIMES, {"stage": "early"})
        out, msg, *_ = replay.replay_file(path)
        self.assertEqual(out["now"]["stage"], "early")
        self.assertTrue(out["now"]["alert"])
        self.assertIn("גשם בדרך", msg)


if __name__ == "__main__":
    unittest.main()
