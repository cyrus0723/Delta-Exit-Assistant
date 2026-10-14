from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from detector import MatchResult
from notify import Notifier, NotifySettings
from profiles import GameProfile
from roi import RoiRel


class NotificationModeTests(unittest.TestCase):
    def setUp(self):
        profile = GameProfile("delta", "三角洲行动", 30, RoiRel(0, 0, 0.1, 0.1), [])
        self.popup = Mock()
        self.settings = NotifySettings()
        self.notifier = Notifier("test", profile, self.settings, show_popup=self.popup)
        self.result = MatchResult("撤离成功", "success", 0.9)

    def test_default_popup_does_not_send_windows_notification(self):
        with patch("notify.datetime") as clock, patch.object(self.notifier, "_show_toast") as toast, patch.object(self.notifier, "play_sound") as sound:
            clock.now.return_value.strftime.return_value = "23:47"
            self.notifier.notify(self.result)
        self.popup.assert_called_once_with("三角洲行动：该下机了", "现在是“23:47”，是不是该去做别的事啦？")
        toast.assert_not_called()
        sound.assert_called_once()

    def test_native_notification_does_not_open_popup(self):
        self.settings.mode = "native"
        with patch.object(self.notifier, "_show_toast") as toast, patch.object(self.notifier, "play_sound") as sound:
            self.notifier.notify(self.result)
        self.popup.assert_not_called()
        toast.assert_called_once()
        sound.assert_called_once()

    def test_custom_text_is_used_for_popup(self):
        self.settings.title_tpl = "{game} 结束"
        self.settings.msg_tpl = "{label}，请休息"
        with patch.object(self.notifier, "play_sound"):
            self.notifier.notify(self.result)
        self.popup.assert_called_once_with("三角洲行动 结束", "撤离成功，请休息")


if __name__ == "__main__":
    unittest.main()
