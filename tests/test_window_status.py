from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ui_main import MainWindow
from ui_tray import TrayApp


class WindowStatusTests(unittest.TestCase):
    def test_start_and_stop_update_status_from_detector(self):
        app = Mock()
        app._timer_mode = "always"
        app._detector.is_running.return_value = False
        app._detector.start.side_effect = lambda: setattr(app._detector.is_running, "return_value", True)
        app._detector.stop.side_effect = lambda: setattr(app._detector.is_running, "return_value", False)
        app._action_start.side_effect = lambda *args: TrayApp._action_start(app, *args)
        app._action_stop.side_effect = lambda *args: TrayApp._action_stop(app, *args)
        window = MainWindow.__new__(MainWindow)
        window.app = app
        window.timer = Mock()
        window.timer.get.return_value = "always"
        window.toggle = Mock()
        window.status_label = Mock()
        window.status_description = Mock()
        window.refresh = window._refresh_status

        window._toggle_detection()
        self.assertEqual(window.toggle.config.call_args.kwargs["text"], "停止检测")
        self.assertEqual(window.status_label.config.call_args.kwargs["text"], "正在检测")
        window._toggle_detection()
        self.assertEqual(window.toggle.config.call_args.kwargs["text"], "开始检测")
        self.assertEqual(window.status_label.config.call_args.kwargs["text"], "准备就绪")

    def test_background_poll_does_not_replace_unsaved_fields(self):
        window = MainWindow.__new__(MainWindow)
        window.window = Mock()
        window._refresh_status = Mock()
        window.refresh = Mock()
        window._poll_status()
        window._refresh_status.assert_called_once()
        window.refresh.assert_not_called()
        window.window.after.assert_called_once_with(500, window._poll_status)


if __name__ == "__main__":
    unittest.main()
