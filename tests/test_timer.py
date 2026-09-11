from datetime import datetime, time
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alert_timer import alert_window, is_alert_window_open, normalize_duration_min, parse_target_time
from detector import Detector, DetectorConfig
from profiles import GameProfile, TemplateItem, normalize_profile_id
from roi import RoiRel


class TimerTests(unittest.TestCase):
    def test_parse_target_time_accepts_single_digit_hour(self) -> None:
        self.assertEqual(parse_target_time("8:05"), time(8, 5))

    def test_parse_target_time_rejects_invalid_values(self) -> None:
        self.assertIsNone(parse_target_time("24:00"))
        self.assertIsNone(parse_target_time("08:60"))
        self.assertIsNone(parse_target_time("08"))

    def test_alert_window_opens_at_target_minus_duration(self) -> None:
        target = time(8, 0)
        before = datetime(2026, 9, 11, 7, 29, 59)
        opening = datetime(2026, 9, 11, 7, 30, 0)
        at_target = datetime(2026, 9, 11, 8, 0, 0)
        self.assertFalse(is_alert_window_open(before, target, 30))
        self.assertTrue(is_alert_window_open(opening, target, 30))
        self.assertTrue(is_alert_window_open(at_target, target, 30))

    def test_alert_window_stays_open_after_target(self) -> None:
        self.assertTrue(is_alert_window_open(datetime(2026, 9, 11, 8, 1), time(8, 0), 30))

    def test_alert_window_can_cross_midnight(self) -> None:
        window = alert_window(datetime(2026, 9, 11, 0, 10), time(0, 10), 30)
        self.assertEqual(window.opens_at, datetime(2026, 9, 10, 23, 40))

    def test_duration_is_bounded(self) -> None:
        self.assertEqual(normalize_duration_min(-1), 0)
        self.assertEqual(normalize_duration_min(2000), 1440)

    def test_custom_profile_id_is_safe_for_chinese_game_name(self) -> None:
        profile_id = normalize_profile_id("自定义游戏")
        self.assertRegex(profile_id, r"^game_[0-9a-f]{8}$")

    def test_detector_accepts_empty_custom_template_slots(self) -> None:
        profile = GameProfile(
            id="custom_game",
            display_name="Custom Game",
            estimated_duration_min=30,
            roi_rel=RoiRel(x=0.0, y=0.0, w=0.1, h=0.1),
            templates=[
                TemplateItem(
                    id="custom_game_win",
                    label="胜利",
                    path="assets/templates/custom_game/not_yet_captured.png",
                )
            ],
        )
        detector = Detector(DetectorConfig(), profile, on_match=lambda result: None)
        self.assertEqual(detector._tpls, {})


if __name__ == "__main__":
    unittest.main()
