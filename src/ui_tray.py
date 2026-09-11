# src/ui_tray.py
from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
import wave
from dataclasses import replace
from datetime import datetime
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
    normalize_profile_id,
    pick_profile,
    profile_file_path,
    resolve_resource_path,
    save_profile,
)
from ui_dialogs import TkDialogService
from notify import Notifier, NotifySettings, VALID_NOTIFY_MODES
from capture import capture_to_template, grab_profile_roi_bgr
from config_store import exe_dir, load_config, save_config
from alert_timer import (
    DEFAULT_ESTIMATED_DURATION_MIN,
    alert_window,
    is_alert_window_open,
    normalize_duration_min,
    parse_target_time,
)

from roi_tuner import (
    RoiTuner,
    load_roi_override,
    save_roi_override,
    clear_roi_override,
)
from roi_selector import select_roi_fullscreen

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

        # profiles
        self._profiles: List[GameProfile] = load_profiles_from_assets()
        if not self._profiles:
            self._profiles = [fallback_delta_profile_from_legacy_config(self._cfg)]

        selected_id = str(self._cfg.get("selected_profile_id", "")).strip() or self._profiles[0].id
        self._profile_base: GameProfile = pick_profile(self._profiles, selected_id)

        # apply roi override if exists
        self._profile: GameProfile = self._apply_roi_override(self._profile_base)

        # Timer settings. Per-game overrides stay in config.json so packaged
        # profile files remain read-only defaults.
        self._timer_enabled = bool(self._cfg.get("timer_enabled", False))
        raw_target_time = str(self._cfg.get("timer_target_time", "08:00"))
        self._timer_target_time = parse_target_time(raw_target_time) or parse_target_time("08:00")

        # detector config
        det_cfg = DetectorConfig(
            threshold=float(self._cfg.get("threshold", 0.82)),
            hysteresis=float(self._cfg.get("hysteresis", 0.12)),
            scan_interval_sec=float(self._cfg.get("scan_interval_sec", 0.20)),
            cooldown_sec=float(self._cfg.get("cooldown_sec", 3.0)),
        )

        # notifier settings
        title_tpl = str(self._cfg.get("title_tpl", "{game} 结算检测")).strip() or "{game} 结算检测"
        msg_tpl = str(self._cfg.get("msg_tpl", "{label}（score={score:.3f}）")).strip() or "{label}（score={score:.3f}）"
        mode = str(self._cfg.get("notify_mode", "both")).strip().lower()
        if mode not in VALID_NOTIFY_MODES:
            mode = "both"
        sound_path = str(self._cfg.get("sound_path", "")).strip()

        self._notify_settings = NotifySettings(
            title_tpl=title_tpl, msg_tpl=msg_tpl, mode=mode, sound_path=sound_path
        )
        self._notifier = Notifier(APP_NAME, self._profile, self._notify_settings)

        # detector
        self._detector = Detector(
            det_cfg,
            self._profile,
            on_match=self._on_match,
            alerts_allowed=self._alerts_allowed_for_profile,
        )

        # tray icon
        icon_path = resolve_resource_path("assets/icon.ico")
        try:
            image = Image.open(icon_path)
        except Exception:
            image = Image.new("RGB", (64, 64), color=(0, 0, 0))

        self._icon = pystray.Icon(APP_NAME, image, APP_NAME)
        self._icon.menu = self._build_menu()

        # persist once
        self._persist()

    # -------------------------
    # ROI override helpers
    # -------------------------
    def _apply_roi_override(self, base: GameProfile) -> GameProfile:
        override = load_roi_override(self._cfg, base.id)
        if override is None:
            return base
        # GameProfile is frozen dataclass in profiles.py, so use dataclasses.replace
        return replace(base, roi_rel=override)

    def _current_tuner(self) -> RoiTuner:
        # operate on the currently effective roi (override or base)
        step = float(self._cfg.get("roi_step", 0.005) or 0.005)
        return RoiTuner(roi=self._profile.roi_rel, step=step)

    def _save_tuner(self, tuner: RoiTuner) -> None:
        # save override to config for current profile id
        save_roi_override(self._cfg, self._profile_base.id, tuner.roi)
        self._cfg["roi_step"] = tuner.step

        # refresh effective profile / detector / notifier
        self._profile = replace(self._profile_base, roi_rel=tuner.roi)
        self._detector.set_profile(self._profile)
        self._notifier.set_profile(self._profile)

        self._persist()
        self._rebuild_menu()

    # -------------------------
    # Callback from detector
    # -------------------------
    def _on_match(self, result: MatchResult) -> None:
        self._notifier.notify(result)

    # -------------------------
    # Timer helpers
    # -------------------------
    def _game_duration_min(self, profile: GameProfile) -> int:
        overrides = self._cfg.get("timer_game_durations")
        if isinstance(overrides, dict) and profile.id in overrides:
            return normalize_duration_min(overrides[profile.id], profile.estimated_duration_min)
        return normalize_duration_min(profile.estimated_duration_min, DEFAULT_ESTIMATED_DURATION_MIN)

    def _alerts_allowed_for_profile(self, profile: GameProfile) -> bool:
        if not self._timer_enabled:
            return True
        return is_alert_window_open(
            datetime.now(), self._timer_target_time, self._game_duration_min(profile)
        )

    def _timer_window_label(self) -> str:
        duration = self._game_duration_min(self._profile_base)
        window = alert_window(datetime.now(), self._timer_target_time, duration)
        return f"{window.opens_at:%H:%M} 起（目标 {window.target_at:%H:%M}）"

    # -------------------------
    # Persistence + menu refresh
    # -------------------------
    def _persist(self) -> None:
        self._cfg["selected_profile_id"] = self._profile_base.id
        self._cfg["threshold"] = self._detector.cfg.threshold
        self._cfg["hysteresis"] = self._detector.cfg.hysteresis
        self._cfg["scan_interval_sec"] = self._detector.cfg.scan_interval_sec
        self._cfg["cooldown_sec"] = self._detector.cfg.cooldown_sec
        self._cfg["title_tpl"] = self._notify_settings.title_tpl
        self._cfg["msg_tpl"] = self._notify_settings.msg_tpl
        self._cfg["notify_mode"] = self._notify_settings.mode
        self._cfg["sound_path"] = self._notify_settings.sound_path
        self._cfg["timer_enabled"] = self._timer_enabled
        self._cfg["timer_target_time"] = self._timer_target_time.strftime("%H:%M")
        save_config(self._cfg)

    def _rebuild_menu(self) -> None:
        self._icon.menu = self._build_menu()
        self._icon.update_menu()

    # -------------------------
    # Actions: start/stop
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
    # Actions: select profile
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
        selected_id = self._profile_base.id
        self._profiles = load_profiles_from_assets()
        if not self._profiles:
            self._profiles = [fallback_delta_profile_from_legacy_config(self._cfg)]
        self._profile_base = pick_profile(self._profiles, selected_id)
        self._profile = self._apply_roi_override(self._profile_base)
        self._detector.set_profile(self._profile)
        self._notifier.set_profile(self._profile)
        self._persist()
        self._rebuild_menu()

    def _action_new_game(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        name = self._dlg.ask_str("新建游戏", "请输入游戏名称，例如 OW2 或 Overwatch 2", "")
        if name is None or not name.strip():
            return

        display_name = name.strip()
        profile_id = normalize_profile_id(display_name)
        profile_path = profile_file_path(profile_id)
        if profile_path.exists() and not self._dlg.confirm(
            "已存在同名游戏", f"已存在 {profile_path.name}。是否覆盖其配置？"
        ):
            return

        self._dlg.info("选择检测区域", "拖拽框选结算标题或结果文字区域。\n按 Enter 确认，按 Esc 取消。")
        roi_rel = self._dlg.run_in_tk(
            lambda root: select_roi_fullscreen(root, f"选择 ROI - {display_name}")
        )
        if roi_rel is None:
            return

        templates = [
            TemplateItem(id=f"{profile_id}_win", label="胜利", path=f"assets/templates/{profile_id}/win.png"),
            TemplateItem(id=f"{profile_id}_lose", label="失败", path=f"assets/templates/{profile_id}/lose.png"),
            TemplateItem(id=f"{profile_id}_draw", label="平局", path=f"assets/templates/{profile_id}/draw.png"),
        ]
        profile = GameProfile(
            id=profile_id,
            display_name=display_name,
            estimated_duration_min=DEFAULT_ESTIMATED_DURATION_MIN,
            roi_rel=roi_rel,
            templates=templates,
        )

        try:
            saved_profile = save_profile(profile)
            (saved_profile.parent.parent / "templates" / profile_id).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._dlg.info("创建失败", repr(e))
            return

        self._profiles = load_profiles_from_assets()
        self._profile_base = pick_profile(self._profiles, profile_id)
        self._profile = self._apply_roi_override(self._profile_base)
        self._detector.set_profile(self._profile)
        self._notifier.set_profile(self._profile)
        self._persist()
        self._rebuild_menu()
        self._dlg.info(
            "创建成功",
            "已创建游戏配置并切换到该游戏。\n\n"
            "进入结算界面后，依次选择：\n抓取模板 -> 抓取：胜利 / 失败 / 平局。",
        )

    # -------------------------
    # Actions: tuning detector
    # -------------------------
    def _action_set_threshold(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        v = self._dlg.ask_float("设置匹配阈值", "threshold（建议 0.70 ~ 0.90）", self._detector.cfg.threshold)
        if v is None:
            return
        self._detector.cfg.threshold = max(0.0, min(1.0, float(v)))
        self._persist()
        self._rebuild_menu()

    def _action_set_hysteresis(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        v = self._dlg.ask_float("设置回落差值", "hysteresis（建议 0.08 ~ 0.20）\n越大越不容易重复提示", self._detector.cfg.hysteresis)
        if v is None:
            return
        self._detector.cfg.hysteresis = max(0.0, min(1.0, float(v)))
        self._persist()
        self._rebuild_menu()

    def _action_set_cooldown(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        v = self._dlg.ask_float("设置冷却时间", "cooldown_sec（秒，建议 1.0 ~ 10.0）", self._detector.cfg.cooldown_sec)
        if v is None:
            return
        self._detector.cfg.cooldown_sec = max(0.0, float(v))
        self._persist()
        self._rebuild_menu()

    def _action_set_interval(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        v = self._dlg.ask_float("设置扫描间隔", "scan_interval_sec（秒，建议 0.10 ~ 0.50）\n越小越灵敏但更耗资源", self._detector.cfg.scan_interval_sec)
        if v is None:
            return
        self._detector.cfg.scan_interval_sec = max(0.05, float(v))
        self._persist()
        self._rebuild_menu()

    # -------------------------
    # Actions: timer
    # -------------------------
    def _is_timer_enabled(self, item: pystray.MenuItem) -> bool:
        return self._timer_enabled

    def _action_toggle_timer(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._timer_enabled = not self._timer_enabled
        self._persist()
        self._rebuild_menu()

    def _action_set_timer_target(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        raw = self._dlg.ask_str(
            "设置目标时间",
            "使用 24 小时制 HH:MM，例如 08:00。\n"
            "提醒会从目标时间减去当前游戏大概时长时开始。",
            self._timer_target_time.strftime("%H:%M"),
        )
        if raw is None:
            return
        target = parse_target_time(raw)
        if target is None:
            self._dlg.info("时间格式错误", "请输入有效的 24 小时制时间，例如 08:00。")
            return
        self._timer_target_time = target
        self._persist()
        self._rebuild_menu()

    def _action_set_game_duration(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        profile = self._profile_base
        current = self._game_duration_min(profile)
        value = self._dlg.ask_float(
            "设置游戏大概时长",
            f"{profile.display_name} 的单局大概时长（分钟，0 ~ 1440）。\n"
            "程序会从目标时间减去该时长时开始允许结算提醒。",
            float(current),
        )
        if value is None:
            return

        duration = normalize_duration_min(value, current)
        overrides = self._cfg.get("timer_game_durations")
        if not isinstance(overrides, dict):
            overrides = {}
            self._cfg["timer_game_durations"] = overrides
        overrides[profile.id] = duration
        self._persist()
        self._rebuild_menu()

    # -------------------------
    # Actions: notify mode
    # -------------------------
    def _is_mode(self, mode: str):
        def _inner(item: pystray.MenuItem) -> bool:
            return self._notify_settings.mode == mode
        return _inner

    def _action_set_mode(self, mode: str):
        def _inner(icon: pystray.Icon, item: pystray.MenuItem) -> None:
            if mode not in VALID_NOTIFY_MODES:
                return
            self._notify_settings.mode = mode
            self._persist()
            self._rebuild_menu()
        return _inner

    # -------------------------
    # Actions: edit notification text
    # -------------------------
    def _action_edit_text(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._dlg.info(
            "可用占位符",
            "你可以在标题/正文里使用：\n"
            "  {game}  当前游戏名\n"
            "  {label} 结果标签（来自 profile.json 的 templates[].label）\n"
            "  {score} 匹配分数（支持格式：{score:.3f}）\n"
            "  {id}    模板ID\n",
        )

        title = self._dlg.ask_str("编辑通知标题", "例如：{game} 结算检测", self._notify_settings.title_tpl)
        if title is None:
            return
        msg = self._dlg.ask_str("编辑通知正文", "例如：{label}（score={score:.3f}）", self._notify_settings.msg_tpl)
        if msg is None:
            return

        if title.strip():
            self._notify_settings.title_tpl = title.strip()
        if msg.strip():
            self._notify_settings.msg_tpl = msg.strip()

        self._persist()
        self._rebuild_menu()

    def _action_test_notify(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._notifier.notify(MatchResult(label="测试成功", template_id="test", score=0.999))

    def _action_select_sound(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        selected = self._dlg.ask_open_file("选择提示音（仅支持 WAV）", [("WAV 音频", "*.wav"), ("所有文件", "*.*")])
        if not selected:
            return
        source = Path(selected)
        if source.suffix.lower() != ".wav":
            self._dlg.info("提示音设置", "请选择 WAV 格式的音频文件。")
            return
        try:
            if source.stat().st_size > 10 * 1024 * 1024:
                self._dlg.info("提示音设置", "音频文件不能超过 10 MB。")
                return
            try:
                with wave.open(str(source), "rb"):
                    pass
            except (wave.Error, EOFError, OSError):
                self._dlg.info("提示音设置", "所选文件不是有效的 WAV 音频。")
                return
            sounds_dir = exe_dir() / "sounds"
            sounds_dir.mkdir(parents=True, exist_ok=True)
            target = sounds_dir / "custom.wav"
            temp_target = sounds_dir / "custom.wav.tmp"
            shutil.copyfile(source, temp_target)
            temp_target.replace(target)
            self._notify_settings.sound_path = "sounds/custom.wav"
            self._persist()
            self._rebuild_menu()
            self._notifier.play_sound()
        except Exception as e:
            self._dlg.info("提示音设置失败", repr(e))

    def _action_reset_sound(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._notify_settings.sound_path = ""
        self._persist()
        self._rebuild_menu()

    def _action_test_sound(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._notifier.play_sound()

    # -------------------------
    # Actions: capture template
    # -------------------------
    def _action_capture_template(self, tpl: TemplateItem):
        def _inner(icon: pystray.Icon, item: pystray.MenuItem) -> None:
            try:
                saved = capture_to_template(self._profile, tpl)
                self._detector.reload_templates()
                self._dlg.info("抓取模板成功", f"已保存：\n{saved}")
            except Exception as e:
                self._dlg.info("抓取模板失败", repr(e))
        return _inner

    # -------------------------
    # ROI tuner actions
    # -------------------------
    def _action_set_roi_step(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        cur = float(self._cfg.get("roi_step", 0.005) or 0.005)
        v = self._dlg.ask_float("设置 ROI 步长", "建议：0.002 ~ 0.020\n(0.005 约等于 1920 宽下 9~10 像素)", cur)
        if v is None:
            return
        self._cfg["roi_step"] = max(0.0005, float(v))
        self._persist()
        self._rebuild_menu()

    def _action_preview_roi(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        try:
            bgr = grab_profile_roi_bgr(self._profile)
            # save to temp and open
            tmp = Path(tempfile.gettempdir()) / f"roi_preview_{self._profile_base.id}.png"
            # reuse cv2.imencode for chinese path safety
            import cv2
            ok, buf = cv2.imencode(".png", bgr)
            if ok:
                tmp.write_bytes(buf.tobytes())
                os.startfile(str(tmp))  # Windows
        except Exception as e:
            self._dlg.info("预览失败", repr(e))

    def _action_roi_move(self, direction: str):
        def _inner(icon: pystray.Icon, item: pystray.MenuItem) -> None:
            t = self._current_tuner()
            if direction == "left":
                t.left()
            elif direction == "right":
                t.right()
            elif direction == "up":
                t.up()
            elif direction == "down":
                t.down()
            self._save_tuner(t)
        return _inner

    def _action_roi_resize(self, which: str):
        def _inner(icon: pystray.Icon, item: pystray.MenuItem) -> None:
            t = self._current_tuner()
            if which == "wider":
                t.wider()
            elif which == "narrower":
                t.narrower()
            elif which == "taller":
                t.taller()
            elif which == "shorter":
                t.shorter()
            self._save_tuner(t)
        return _inner

    def _action_roi_reset(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        # clear override for this profile
        clear_roi_override(self._cfg, self._profile_base.id)
        self._profile = self._profile_base
        self._detector.set_profile(self._profile)
        self._notifier.set_profile(self._profile)
        self._persist()
        self._rebuild_menu()

    # -------------------------
    # Build menu
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

        mode_menu = pystray.Menu(
            pystray.MenuItem("都要（弹窗 + 响铃）", self._action_set_mode("both"), checked=self._is_mode("both"), radio=True),
            pystray.MenuItem("只弹窗", self._action_set_mode("toast"), checked=self._is_mode("toast"), radio=True),
            pystray.MenuItem("只响铃", self._action_set_mode("sound"), checked=self._is_mode("sound"), radio=True),
        )

        timer_menu = pystray.Menu(
            pystray.MenuItem("启用计时提醒", self._action_toggle_timer, checked=self._is_timer_enabled),
            pystray.MenuItem(
                f"目标时间 = {self._timer_target_time:%H:%M}", self._action_set_timer_target
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                f"当前游戏大概时长 = {self._game_duration_min(self._profile_base)} 分钟",
                self._action_set_game_duration,
            ),
            pystray.MenuItem(
                f"当前提醒窗口：{self._timer_window_label()}",
                lambda icon, item: None,
                enabled=False,
            ),
        )

        settings_menu = pystray.Menu(
            pystray.MenuItem(f"匹配阈值 threshold = {self._detector.cfg.threshold:.3f}", self._action_set_threshold),
            pystray.MenuItem(f"回落差值 hysteresis = {self._detector.cfg.hysteresis:.3f}", self._action_set_hysteresis),
            pystray.MenuItem(f"冷却时间 cooldown_sec = {self._detector.cfg.cooldown_sec:.2f}s", self._action_set_cooldown),
            pystray.MenuItem(f"扫描间隔 scan_interval = {self._detector.cfg.scan_interval_sec:.2f}s", self._action_set_interval),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("提醒方式（弹窗/响铃）", mode_menu),
            pystray.MenuItem(
                f"当前提示音：{Path(self._notify_settings.sound_path).name if self._notify_settings.sound_path else '系统默认'}",
                pystray.Menu(
                    pystray.MenuItem("选择提示音…", self._action_select_sound),
                    pystray.MenuItem("测试当前提示音", self._action_test_sound),
                    pystray.MenuItem("恢复系统默认提示音", self._action_reset_sound),
                ),
            ),
            pystray.MenuItem("计时提醒", timer_menu),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("编辑通知文本…", self._action_edit_text),
            pystray.MenuItem("发送测试通知", self._action_test_notify),
        )

        cap_items = [pystray.MenuItem(f"抓取：{t.label}", self._action_capture_template(t)) for t in self._profile.templates]
        capture_menu = pystray.Menu(*cap_items) if cap_items else pystray.Menu(
            pystray.MenuItem("（当前游戏无模板定义）", lambda i, it: None)
        )

        # ROI tuner menu
        step = float(self._cfg.get("roi_step", 0.005) or 0.005)
        roi_menu = pystray.Menu(
            pystray.MenuItem(f"预览 ROI（打开图片）", self._action_preview_roi),
            pystray.MenuItem(f"设置步长 step = {step:.4f}", self._action_set_roi_step),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("↑ 上移", self._action_roi_move("up")),
            pystray.MenuItem("↓ 下移", self._action_roi_move("down")),
            pystray.MenuItem("← 左移", self._action_roi_move("left")),
            pystray.MenuItem("→ 右移", self._action_roi_move("right")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("加宽", self._action_roi_resize("wider")),
            pystray.MenuItem("变窄", self._action_roi_resize("narrower")),
            pystray.MenuItem("加高", self._action_roi_resize("taller")),
            pystray.MenuItem("变矮", self._action_roi_resize("shorter")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("恢复为默认 ROI", self._action_roi_reset),
        )

        return pystray.Menu(
            pystray.MenuItem("启动检测", self._action_start, enabled=lambda item: not self._detector.is_running()),
            pystray.MenuItem("停止检测", self._action_stop, enabled=lambda item: self._detector.is_running()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("选择游戏", pystray.Menu(*profile_items)),
            pystray.MenuItem("ROI 调整", roi_menu),
            pystray.MenuItem("抓取模板", capture_menu),
            pystray.MenuItem("设置", settings_menu),
            pystray.MenuItem("重载模板", self._action_reload_templates),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self._action_quit),
        )

    def run(self) -> None:
        self._icon.run()
