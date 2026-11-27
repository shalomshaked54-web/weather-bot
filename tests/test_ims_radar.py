"""בדיקות למכ"ם השירות המטאורולוגי ולבחירת המקור (3.2-beta). בלי רשת"""
import io
import os
import sys
import unittest
from datetime import datetime
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rainbot import ims_radar, main, messages  # noqa: E402
from rainbot.config import TZ  # noqa: E402
from rainbot.radar import XX, YY  # noqa: E402


def png(color_at_target=None, size=940):
    img = np.zeros((size, size, 4), np.uint8)
    if color_at_target:
        py, px = ims_radar.grid_pixels(size, size)
        j, i = np.unravel_index(np.argmin(XX ** 2 + YY ** 2), XX.shape)
        y, x = py[j, i], px[j, i]
        img[y - 3:y + 4, x - 3:x + 4] = list(color_at_target) + [255]
    b = io.BytesIO()
    Image.fromarray(img, "RGBA").save(b, "PNG")
    return b.getvalue()


class DecodeTests(unittest.TestCase):

    def test_whole_grid_inside_image(self):
        py, px = ims_radar.grid_pixels(940, 940)
        self.assertTrue((px >= 0).all())

    def test_target_pixel_matches_site(self):
        # נמדד באתר: גבעת שמואל היא פיקסל (474, 460) בתמונה של 940x940
        py, px = ims_radar.grid_pixels(940, 940)
        j, i = np.unravel_index(np.argmin(XX ** 2 + YY ** 2), XX.shape)
        self.assertLessEqual(abs(px[j, i] - 474), 2)
        self.assertLessEqual(abs(py[j, i] - 460), 2)

    def test_colors_to_dbz(self):
        f = ims_radar.decode_image(png((0x01, 0xd3, 0x1c)))     # 4 מ"מ/שעה
        self.assertAlmostEqual(float(f.max()), ims_radar.mmh_to_dbz(4.0), delta=0.5)
        self.assertGreater(f[np.hypot(XX, YY) <= 2].max(), 30)
        f = ims_radar.decode_image(png((0, 183, 185)))          # צבע אמיתי מהאתר, ~0.3 מ"מ/שעה
        self.assertTrue(10 < f.max() < 20)
        self.assertEqual(float(ims_radar.decode_image(png()).max()), 0.0)

    def test_unknown_color_ignored(self):
        self.assertEqual(float(ims_radar.decode_image(png((120, 120, 120))).max()), 0.0)

    def test_parse_and_pick(self):
        items = [{"forecast_time": f"2026-11-27 13:{m:02d}:00",
                  "file_name": f"/x/IMSRadar4GIS_2026112713{m:02d}_0.png"} for m in range(0, 60, 5)]
        items.append({"forecast_time": "2026-11-27 13:57:00", "file_name": "/x/bad_1.png"})
        frames = ims_radar.parse_index({"data": {"types": {"IMSRadar": items}}})
        self.assertEqual(len(frames), 12)
        picked = ims_radar.pick_frames(frames)
        self.assertEqual([datetime.fromtimestamp(f["time"], TZ).minute for f in picked], [25, 35, 45, 55])


class SourceChoiceTests(unittest.TestCase):

    def choose(self, rv, ims, primary="rainviewer"):
        now = datetime(2026, 12, 1, 18, 0, tzinfo=TZ)
        z = [np.zeros(XX.shape, np.float32)] * 4

        def res(age):
            if age is None:
                return None, None, "HTTP 503"
            return z, [{"time": int(now.timestamp()) - 60 * age}] * 4, None
        with mock.patch.object(main, "get_rainviewer", return_value=res(rv)), \
                mock.patch.object(main, "get_ims_radar", return_value=res(ims)), \
                mock.patch.object(main, "RADAR_PRIMARY", primary):
            fields, frames, err = main.get_radar(main.requests.Session(), now)
        return main.RADAR_INFO["src"], err

    def test_rainviewer_primary(self):
        self.assertEqual(self.choose(5, 8), ("rainviewer", None))

    def test_fallback_when_rainviewer_down(self):
        self.assertEqual(self.choose(None, 8), ("ims", None))

    def test_switch_when_rainviewer_stale(self):
        self.assertEqual(self.choose(30, 8), ("ims", None))

    def test_ims_primary(self):
        self.assertEqual(self.choose(5, 8, "ims"), ("ims", None))

    def test_both_down(self):
        src, err = self.choose(None, None)
        self.assertIsNone(src)
        self.assertTrue(err)

    def test_footer_follows_source(self):
        t = datetime(2026, 12, 1, 18, 0, tzinfo=TZ)
        messages.set_source("ims")
        self.assertIn("השירות המטאורולוגי", messages._footer(t))
        messages.set_source("rainviewer")
        self.assertIn("Rain Viewer", messages._footer(t))


if __name__ == "__main__":
    unittest.main()
