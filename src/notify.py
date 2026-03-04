# src/notify.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import winsound
from winotify import Notification, audio

from detector import MatchResult
from profiles import GameProfile, resolve_resource_path


VALID_NOTIFY_MODES = {"both", "toast", "sound"}


@dataclass
class NotifySettings:
    title_tpl: str = "{game} 结算检测"
    msg_tpl: str = "{label}（score={score:.3f}）"
    mode: str = "both"  # both/toast/sound


class Notifier:
    def __init__(self, app_name: str, profile: GameProfile, settings: NotifySettings):
        self.app_name = app_name
        self.profile = profile
        self.settings = settings

    def set_profile(self, profile: GameProfile) -> None:
        self.profile = profile

    def _format(self, tpl: str, result: MatchResult) -> str:
        data: Dict[str, object] = {
            "game": self.profile.display_name,
            "label": result.label,
            "score": result.score,
            "id": result.template_id,
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
        toast.set_audio(audio.Default, loop=False)
        toast.show()

    def _beep(self) -> None:
        winsound.MessageBeep(winsound.MB_ICONASTERISK)

    def notify(self, result: MatchResult) -> None:
        title = self._format(self.settings.title_tpl, result)
        msg = self._format(self.settings.msg_tpl, result)

        mode = (self.settings.mode or "both").lower()
        if mode not in VALID_NOTIFY_MODES:
            mode = "both"

        if mode in ("both", "toast"):
            try:
                self._show_toast(title, msg)
            except Exception as e:
                print("Toast error:", repr(e))

        if mode in ("both", "sound"):
            try:
                self._beep()
            except Exception:
                pass