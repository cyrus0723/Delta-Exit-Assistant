# src/app.py
from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from winotify import Notification, audio

import pystray
from PIL import Image

from detector import Detector, DetectorConfig, MatchResult
from profiles import (
    GameProfile,
    fallback_delta_profile_from_legacy_config,
    load_profiles_from_assets,
    pick_profile,
    resolve_resource_path,
)


import ctypes

def enable_dpi_awareness():
    # Windows only
    try:
        # Per-monitor DPI aware (best on Win10/11)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            # System DPI aware fallback
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

enable_dpi_awareness()

# Windows 提示音
import winsound


APP_NAME = "Delta Exit Assistant"


def exe_dir() -> Path:
    """
    Put config.json next to the exe (or project root in dev).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def config_path() -> Path:
    return exe_dir() / "config.json"


def load_config() -> Dict:
    p = config_path()
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg: Dict) -> None:
    p = config_path()
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


class TrayApp:
    def __init__(self) -> None:
        self._cfg = load_config()

        # Load profiles
        self._profiles: List[GameProfile] = load_profiles_from_assets()
        if not self._profiles:
            # fallback to a single delta profile if no profiles exist
            self._profiles = [fallback_delta_profile_from_legacy_config(self._cfg)]

        selected_id = str(self._cfg.get("selected_profile_id", "")).strip()
        if not selected_id:
            selected_id = self._profiles[0].id
        self._profile: GameProfile = pick_profile(self._profiles, selected_id)

        # Detector config from config.json (with defaults)
        det_cfg = DetectorConfig(
            threshold=float(self._cfg.get("threshold", 0.82)),
            scan_interval_sec=float(self._cfg.get("scan_interval_sec", 0.20)),
            cooldown_sec=float(self._cfg.get("cooldown_sec", 3.0)),
        )

        self._detector = Detector(det_cfg, self._profile, on_match=self._on_match)

        icon_path = resolve_resource_path("assets/icon.ico")
        try:
            image = Image.open(icon_path)
        except Exception:
            # fallback: create a blank image
            image = Image.new("RGB", (64, 64), color=(0, 0, 0))

        self._icon = pystray.Icon(APP_NAME, image, APP_NAME)
        self._icon.menu = self._build_menu()

    # -------------------------
    # Notifications / Sound
    # -------------------------
    def _on_match(self, result: MatchResult) -> None:
        # Toast (Win11)
        try:
            toast = Notification(
                app_id=APP_NAME,  # 这里可以用你的名字
                title=f"{self._profile.display_name} 检测到结算",
                msg=f"{result.label}  (score={result.score:.3f})",
                icon=resolve_resource_path("assets/icon.ico"),
            )
            # 可选：系统提示音（不一定需要）
            toast.set_audio(audio.Default, loop=False)
            toast.show()
        except Exception as e:
            print("Toast error:", repr(e))

        # Sound (fallback)
        try:
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

    # -------------------------
    # Menu building
    # -------------------------
    def _is_running(self, item: pystray.MenuItem) -> bool:
        return self._detector.is_running()

    def _is_profile_selected(self, profile_id: str):
        def _inner(item: pystray.MenuItem) -> bool:
            return self._profile.id == profile_id
        return _inner

    def _action_start(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._detector.start()

    def _action_stop(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._detector.stop()

    def _action_reload_templates(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._detector.reload_templates()

    def _action_select_profile(self, profile_id: str):
        def _inner(icon: pystray.Icon, item: pystray.MenuItem) -> None:
            # Switch in memory
            for p in self._profiles:
                if p.id == profile_id:
                    self._profile = p
                    break
            self._detector.set_profile(self._profile)

            # Persist
            self._cfg["selected_profile_id"] = self._profile.id
            # also persist detector params in case user tuned config manually
            self._cfg["threshold"] = self._detector.cfg.threshold
            self._cfg["scan_interval_sec"] = self._detector.cfg.scan_interval_sec
            self._cfg["cooldown_sec"] = self._detector.cfg.cooldown_sec
            save_config(self._cfg)

            # Rebuild menu so the radio check updates
            self._icon.menu = self._build_menu()
            self._icon.update_menu()
        return _inner

    def _action_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        try:
            self._detector.stop()
        except Exception:
            pass
        icon.stop()

    def _build_menu(self) -> pystray.Menu:
        # Profiles submenu (radio-like)
        profile_items = []
        for p in self._profiles:
            profile_items.append(
                pystray.MenuItem(
                    p.display_name,
                    self._action_select_profile(p.id),
                    checked=self._is_profile_selected(p.id),
                    radio=True,
                )
            )

        menu = pystray.Menu(
            pystray.MenuItem("启动检测", self._action_start, enabled=lambda item: not self._detector.is_running()),
            pystray.MenuItem("停止检测", self._action_stop, enabled=lambda item: self._detector.is_running()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("选择游戏", pystray.Menu(*profile_items)),
            pystray.MenuItem("重载模板", self._action_reload_templates),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self._action_quit),
        )
        return menu

    def run(self) -> None:
        # Save initial selected id
        self._cfg["selected_profile_id"] = self._profile.id
        save_config(self._cfg)
        self._icon.run()


def main() -> None:
    TrayApp().run()


if __name__ == "__main__":
    main()