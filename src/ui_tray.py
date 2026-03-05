from __future__ import annotations

import ctypes
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import pystray
from PIL import Image

from detector import Detector, DetectorConfig, MatchResult
from profiles import (
    GameProfile,
    TemplateItem,
    fallback_delta_profile_from_legacy_config,
    load_profiles_from_assets,
    pick_profile,
    resolve_resource_path,
    normalize_profile_id,
    save_profile,
    profile_file_path,
)
from ui_dialogs import TkDialogService
from notify import Notifier, NotifySettings, VALID_NOTIFY_MODES
from capture import capture_to_template, grab_profile_roi_bgr
from config_store import load_config, save_config

from roi_tuner import (
    RoiTuner,
    load_roi_override,
    save_roi_override,
    clear_roi_override,
)
from roi_selector import select_roi_fullscreen
from roi import RoiRel

APP_NAME = "Delta Exit Assistant"


def enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


enable_dpi_awareness()


class TrayApp:
    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = load_config()
        self._dlg = TkDialogService()

        self._profiles: List[GameProfile] = load_profiles_from_assets()
        if not self._profiles:
            self._profiles = [fallback_delta_profile_from_legacy_config(self._cfg)]

        selected_id = str(self._cfg.get("selected_profile_id", "")).strip() or self._profiles[0].id
        self._profile_base: GameProfile = pick_profile(self._profiles, selected_id)
        self._profile: GameProfile = self._apply_roi_override(self._profile_base)

        det_cfg = DetectorConfig(
            threshold=float(self._cfg.get("threshold", 0.82)),
            hysteresis=float(self._cfg.get("hysteresis", 0.12)),
            scan_interval_sec=float(self._cfg.get("scan_interval_sec", 0.20)),
            cooldown_sec=float(self._cfg.get("cooldown_sec", 3.0)),
        )

        title_tpl = str(self._cfg.get("title_tpl", "{game} 结算检测")).strip() or "{game} 结算检测"
        msg_tpl = str(self._cfg.get("msg_tpl", "{label}（score={score:.3f}）")).strip() or "{label}（score={score:.3f}）"
        mode = str(self._cfg.get("notify_mode", "both")).strip().lower()
        if mode not in VALID_NOTIFY_MODES:
            mode = "both"

        self._notify_settings = NotifySettings(title_tpl=title_tpl, msg_tpl=msg_tpl, mode=mode)
        self._notifier = Notifier(APP_NAME, self._profile, self._notify_settings)

        self._detector = Detector(det_cfg, self._profile, on_match=self._on_match)

        icon_path = resolve_resource_path("assets/icon.ico")
        try:
            image = Image.open(icon_path)
        except Exception:
            image = Image.new("RGB", (64, 64), color=(0, 0, 0))

        self._icon = pystray.Icon(APP_NAME, image, APP_NAME)
        self._icon.menu = self._build_menu()

        self._persist()

    def _apply_roi_override(self, base: GameProfile) -> GameProfile:
        override = load_roi_override(self._cfg, base.id)
        if override is None:
            return base
        return replace(base, roi_rel=override)

    def _on_match(self, result: MatchResult) -> None:
        self._notifier.notify(result)

    def _persist(self) -> None:
        self._cfg["selected_profile_id"] = self._profile_base.id
        self._cfg["threshold"] = self._detector.cfg.threshold
        self._cfg["hysteresis"] = self._detector.cfg.hysteresis
        self._cfg["scan_interval_sec"] = self._detector.cfg.scan_interval_sec
        self._cfg["cooldown_sec"] = self._detector.cfg.cooldown_sec
        self._cfg["title_tpl"] = self._notify_settings.title_tpl
        self._cfg["msg_tpl"] = self._notify_settings.msg_tpl
        self._cfg["notify_mode"] = self._notify_settings.mode
        save_config(self._cfg)

    def _force_tray_refresh(self) -> None:
        """
        Windows tray menu sometimes caches submenu handles.
        Toggling visibility forces rebuild in practice.
        """
        try:
            self._icon.visible = False
            self._icon.visible = True
        except Exception:
            pass

    def _rebuild_menu(self) -> None:
        self._icon.menu = self._build_menu()
        try:
            self._icon.update_menu()
        except Exception:
            # pystray versions differ; fallback
            pass
        self._force_tray_refresh()

    # -------------------------
    def _action_start(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._detector.start()

    def _action_stop(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._detector.stop()

    def _action_reload_templates(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._detector.reload_templates()

    def _action_quit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        try:
            self._detector.stop()
        except Exception:
            pass
        icon.stop()

    # -------------------------
    def _is_profile_selected(self, profile_id: str):
        def _inner(item: pystray.MenuItem) -> bool:
            return self._profile_base.id == profile_id
        return _inner

    def _action_select_profile(self, profile_id: str):
        def _inner(icon: pystray.Icon, item: pystray.MenuItem) -> None:
            for p in self._profiles:
                if p.id == profile_id:
                    self._profile_base = p
                    break

            self._profile = self._apply_roi_override(self._profile_base)
            self._detector.set_profile(self._profile)
            self._notifier.set_profile(self._profile)

            self._persist()
            self._rebuild_menu()
        return _inner

    def _action_refresh_profiles(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        cur = self._profile_base.id
        self._profiles = load_profiles_from_assets()
        if not self._profiles:
            self._profiles = [fallback_delta_profile_from_legacy_config(self._cfg)]
        self._profile_base = pick_profile(self._profiles, cur)
        self._profile = self._apply_roi_override(self._profile_base)
        self._detector.set_profile(self._profile)
        self._notifier.set_profile(self._profile)
        self._persist()
        self._rebuild_menu()

    # -------------------------
    # New Game
    def _action_new_game(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        name = self._dlg.ask_str("新建游戏", "请输入游戏名称（例如：OW2 / Overwatch 2）", "")
        if not name:
            return

        pid = normalize_profile_id(name)
        fp = profile_file_path(pid)

        if fp.exists():
            ok = self._dlg.confirm("已存在同名游戏", f"已存在 {pid}.json\n是否覆盖？")
            if not ok:
                return

        self._dlg.info("选择 ROI", "拖拽框选『结算标题/结果文字』所在区域。\nEnter 确认，ESC 取消。")
        roi_rel: RoiRel | None = self._dlg.run_in_tk(lambda root: select_roi_fullscreen(root, title=f"选择 ROI - {name}"))
        if roi_rel is None:
            return

        templates = [
            TemplateItem(id=f"{pid}_win", label="胜利", path=f"assets/templates/{pid}/win.png"),
            TemplateItem(id=f"{pid}_lose", label="败北", path=f"assets/templates/{pid}/lose.png"),
            TemplateItem(id=f"{pid}_draw", label="平局", path=f"assets/templates/{pid}/draw.png"),
        ]

        profile = GameProfile(id=pid, display_name=str(name).strip(), roi_rel=roi_rel, templates=templates)

        try:
            out = save_profile(profile)
        except Exception as e:
            self._dlg.info("创建失败", repr(e))
            return

        # ✅关键：重新加载列表 + 立刻切换
        self._profiles = load_profiles_from_assets()
        if not any(p.id == pid for p in self._profiles):
            # fallback: 如果读取失败，至少把它放入内存
            self._profiles.append(profile)

        self._profile_base = pick_profile(self._profiles, pid)
        self._profile = self._apply_roi_override(self._profile_base)

        self._detector.set_profile(self._profile)
        self._notifier.set_profile(self._profile)

        self._persist()
        self._rebuild_menu()

        self._dlg.info("创建成功", f"已创建：\n{out}\n\n下一步：到结算界面 → 抓取模板 → 抓取：胜利/败北/平局。")

    # -------------------------
    # Capture templates
    def _action_capture_template(self, tpl: TemplateItem):
        def _inner(icon: pystray.Icon, item: pystray.MenuItem) -> None:
            try:
                saved = capture_to_template(self._profile, tpl)
                self._detector.reload_templates()
                self._dlg.info("抓取模板成功", f"已保存：\n{saved}")
            except Exception as e:
                self._dlg.info(
                    "抓取模板失败",
                    f"{repr(e)}\n\n模板路径：{tpl.path}\n游戏：{self._profile.display_name} ({self._profile.id})",
                )
        return _inner

    # -------------------------
    def _build_menu(self) -> pystray.Menu:
        profile_items = [
            pystray.MenuItem("新建游戏…", self._action_new_game),
            pystray.MenuItem("刷新游戏列表", self._action_refresh_profiles),
            pystray.Menu.SEPARATOR,
        ]
        profile_items.extend(
            [
                pystray.MenuItem(
                    p.display_name,
                    self._action_select_profile(p.id),
                    checked=self._is_profile_selected(p.id),
                    radio=True,
                )
                for p in self._profiles
            ]
        )

        cap_items = [pystray.MenuItem(f"抓取：{t.label}", self._action_capture_template(t)) for t in self._profile.templates]
        capture_menu = pystray.Menu(*cap_items) if cap_items else pystray.Menu(
            pystray.MenuItem("（当前游戏无模板定义）", lambda i, it: None)
        )

        return pystray.Menu(
            pystray.MenuItem("启动检测", self._action_start, enabled=lambda item: not self._detector.is_running()),
            pystray.MenuItem("停止检测", self._action_stop, enabled=lambda item: self._detector.is_running()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("选择游戏", pystray.Menu(*profile_items)),
            pystray.MenuItem("抓取模板", capture_menu),
            pystray.MenuItem("重载模板", self._action_reload_templates),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self._action_quit),
        )

    def run(self) -> None:
        self._icon.run()