from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui_main import MainWindow
from ui_tray import TrayApp


class AutoStartTests(unittest.TestCase):
    def test_switch_persists_value(self):
        window = MainWindow.__new__(MainWindow)
        window.app = Mock()
        window.app._cfg = {}
        window.auto_start = Mock()
        for value in (True, False):
            window.auto_start.get.return_value = value
            window._set_auto_start()
            self.assertIs(window.app._cfg["auto_start_detection"], value)
        self.assertEqual(window.app._persist.call_count, 2)

    def test_startup_respects_setting_and_starts_countdown(self):
        for setting in (None, False, True):
            with self.subTest(setting=setting), patch("ui_tray.MainWindow"):
                app = TrayApp.__new__(TrayApp)
                app._cfg = {} if setting is None else {"auto_start_detection": setting}
                app._dlg = Mock()
                app._icon = Mock()
                app._detector = Mock()
                app._detector.is_running.return_value = False
                app._timer_mode = "countdown"
                app._countdown_started_at = None
                app._instance = None
                app._rebuild_menu = Mock()
                app.run()
                self.assertEqual(app._detector.start.call_count, int(setting is True))
                self.assertEqual(app._countdown_started_at is not None, setting is True)
                app._icon.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
