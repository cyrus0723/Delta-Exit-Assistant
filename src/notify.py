# src/notify.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional

import winsound
from winotify import Notification, audio

from detector import MatchResult
from profiles import GameProfile, resolve_resource_path


VALID_NOTIFY_MODES = {"popup", "native", "native_silent", "sound"}
DEFAULT_SOUND_PATH = "assets/sounds/default.wav"


@dataclass
class NotifySettings:
    title_tpl: str = "{game}：该下机了"
    msg_tpl: str = "现在是“{time}”，是不是该去做别的事啦？"
    mode: str = "popup"
    sound_path: str = ""  # Empty uses the bundled default WAV.


class Notifier:
    def __init__(self, app_name: str, profile: GameProfile, settings: NotifySettings,
                 show_popup: Optional[Callable[[str, str], None]] = None):
        self.app_name = app_name
        self.profile = profile
        self.settings = settings
        self._show_popup = show_popup

    def set_profile(self, profile: GameProfile) -> None:
        self.profile = profile

    def _format(self, tpl: str, result: MatchResult) -> str:
        data: Dict[str, object] = {
            "game": self.profile.display_name,
            "label": result.label,
            "score": result.score,
            "id": result.template_id,
            "time": datetime.now().strftime("%H:%M"),
        }
        try:
            return tpl.format(**data)
        except Exception:
            return tpl

    def _show_toast(self, title: str, msg: str) -> None:
        toast = Notification(
            app_id=self.app_name,
            title=title,
            msg=msg,
            icon=resolve_resource_path("assets/icon.ico"),
        )
        toast.set_audio(audio.Silent, loop=False)
        toast.show()

    def _beep(self) -> None:
        for sound_path in ((self.settings.sound_path or "").strip(), DEFAULT_SOUND_PATH):
            if not sound_path:
                continue
            try:
                path = resolve_resource_path(sound_path)
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            except Exception as e:
                print("Sound error:", repr(e))
        winsound.MessageBeep(winsound.MB_ICONASTERISK)

    def play_sound(self) -> None:
        """Play the configured sound, falling back to the system beep."""
        try:
            self._beep()
        except Exception:
            pass

    def notify(self, result: MatchResult) -> None:
        title = self._format(self.settings.title_tpl, result)
        msg = self._format(self.settings.msg_tpl, result)

        mode = (self.settings.mode or "popup").lower()
        if mode not in VALID_NOTIFY_MODES:
            mode = "popup"

        if mode == "popup" and self._show_popup:
            self._show_popup(title, msg)

        if mode in ("native", "native_silent"):
            try:
                self._show_toast(title, msg)
            except Exception as e:
                print("Toast error:", repr(e))

        if mode in ("popup", "native", "sound"):
            self.play_sound()
