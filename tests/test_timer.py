from datetime import datetime, time
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alert_timer import alert_window, countdown_window, is_alert_window_open, normalize_duration_min, parse_target_time, parse_countdown_hours
from detector import Detector, DetectorConfig
from profiles import GameProfile, TemplateItem, normalize_profile_id
from roi import RoiRel


class TimerTests(unittest.TestCase):
    def test_countdown_starts_scanning_before_target(self) -> None:
        started = datetime(2026, 10, 14, 22, 45)
        window = countdown_window(started, 2, 30)
        self.assertEqual(window.opens_at, datetime(2026, 10, 15, 0, 15))
        self.assertEqual(window.target_at, datetime(2026, 10, 15, 0, 45))

    def test_short_countdown_scans_immediately(self) -> None:
        started = datetime(2026, 10, 14, 22, 45)
        self.assertEqual(countdown_window(started, 0.25, 30).opens_at, started)

    def test_countdown_input_range(self) -> None:
        self.assertEqual(parse_countdown_hours("1.5"), 1.5)
        for value in ("0", "-1", "25", "nan", "bad"):
            self.assertIsNone(parse_countdown_hours(value))
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

    def test_detector_does_not_capture_before_countdown_opens(self) -> None:
        profile = GameProfile("test", "Test", 30, RoiRel(0, 0, 0.1, 0.1), [])
        allowed = Mock(return_value=False)
        detector = Detector(DetectorConfig(), profile, on_match=Mock(), scan_allowed=allowed)
        detector._stop_evt = Mock()
        detector._stop_evt.is_set.side_effect = [False, True]
        with patch.object(detector, "_get_screen_size", return_value=(1920, 1080)), \
             patch.object(detector, "_grab_roi") as grab:
            detector._run()
        allowed.assert_called_once()
        grab.assert_not_called()


if __name__ == "__main__":
    unittest.main()
