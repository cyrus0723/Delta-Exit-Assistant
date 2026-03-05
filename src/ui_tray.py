# src/ui_tray.py
from __future__ import annotations

import ctypes
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import pystray
from PIL import Image

from capture import (
    capture_to_game_folder_auto,
    capture_to_template,
    grab_profile_roi_bgr,
    select_roi_rel_interactive,
)
from config_store import load_config, save_config
from detector import Detector, DetectorConfig, MatchResult
from notify import Notifier, NotifySettings, VALID_NOTIFY_MODES
from profiles import (
    GameProfile,
    TemplateItem,
    create_new_game_skeleton,
    fallback_delta_profile_from_legacy_config,
    is_valid_game_id,
    load_profiles_from_assets,
    pick_profile,
    resolve_resource_path,
    save_profile_json,
)
from roi_tuner import RoiTuner, clear_roi_override, load_roi_override, save_roi_override
from ui_dialogs import TkDialogService

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
        self._notify_settings = NotifySettings(title_tpl=title_tpl, msg_tpl=msg_tpl, mode=mode)
        self._notifier = Notifier(APP_NAME, self._profile, self._notify_settings)

        # detector
        self._detector = Detector(det_cfg, self._profile, on_match=self._on_match)

        # tray icon
        icon_path = resolve_resource_path("assets/icon.ico")
        try:
            image = Image.open(icon_path)
        except Exception:
            image = Image.new("RGB", (64, 64), color=(0, 0, 0))
        self._icon = pystray.Icon(APP_NAME, image, APP_NAME)
        self._icon.menu = self._build_menu()

        self._persist()

    # -------------------------
    # ROI override helpers
    # -------------------------
    def _apply_roi_override(self, base: GameProfile) -> GameProfile:
        override = load_roi_override(self._cfg, base.id)
        if override is None:
            return base
        return replace(base, roi_rel=override)

    def _current_tuner(self) -> RoiTuner:
        step = float(self._cfg.get("roi_step", 0.005) or 0.005)
        return RoiTuner(roi=self._profile.roi_rel, step=step)

    def _save_tuner(self, tuner: RoiTuner) -> None:
        save_roi_override(self._cfg, self._profile_base.id, tuner.roi)
        self._cfg["roi_step"] = tuner.step

        self._profile = replace(self._profile_base, roi_rel=tuner.roi)
        self._detector.set_profile(self._profile)
        self._notifier.set_profile(self._profile)

        self._persist()
        self._rebuild_menu()

    # -------------------------
    # Profiles refresh
    # -------------------------
    def _refresh_profiles(self, keep_selected_id: str | None = None) -> None:
        cur_id = keep_selected_id or self._profile_base.id
        self._profiles = load_profiles_from_assets() or self._profiles
        self._profile_base = pick_profile(self._profiles, cur_id) if self._profiles else self._profile_base
        self._profile = self._apply_roi_override(self._profile_base)
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
        # Since templates can be added dynamically, refresh profiles first
        self._refresh_profiles()
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

    # -------------------------
    # Action: create new game workflow
    # -------------------------
    def _action_create_new_game(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        game_id = self._dlg.ask_str("新建游戏", "请输入新游戏名称（建议：ow2 / apex / cs2）\n仅允许字母数字 _ -", "")
        if game_id is None:
            return
        game_id = game_id.strip()

        if not is_valid_game_id(game_id):
            self._dlg.info("名称不合法", "请使用 1~32 位：字母/数字/_/-")
            return

        # already exists?
        if any(p.id == game_id for p in self._profiles):
            self._dlg.info("已存在", f"游戏 {game_id} 已存在，将直接切换到它。")
            self._refresh_profiles(keep_selected_id=game_id)
            return

        # create skeleton files
        prof_path, tpl_dir = create_new_game_skeleton(game_id)

        # ROI selection
        self._dlg.info(
            "下一步：框选 ROI",
            "即将进入 ROI 框选。\n\n操作：拖拽框选区域 → 回车确认 / ESC 取消\n\n建议：框选结算标题/结算关键信息所在区域。",
        )
        roi_rel = select_roi_rel_interactive()
        if roi_rel is None:
            self._dlg.info("已取消", "ROI 框选已取消；已创建的文件不会删除，你可以下次再设置。")
            return

        # persist roi into json (templates keep empty; runtime will auto-scan templates/<id>/)
        save_profile_json(game_id, game_id, roi_rel, templates=[])

        # refresh and switch to new profile
        self._refresh_profiles(keep_selected_id=game_id)
        self._dlg.info("新建完成", f"已创建：\n{prof_path}\n{tpl_dir}\n\n并已切换到：{game_id}")

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
        v = self._dlg.ask_float(
            "设置回落差值",
            "hysteresis（建议 0.08 ~ 0.20）\n越大越不容易重复提示",
            self._detector.cfg.hysteresis,
        )
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
        v = self._dlg.ask_float(
            "设置扫描间隔",
            "scan_interval_sec（秒，建议 0.10 ~ 0.50）\n越小越灵敏但更耗资源",
            self._detector.cfg.scan_interval_sec,
        )
        if v is None:
            return
        self._detector.cfg.scan_interval_sec = max(0.05, float(v))
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
            " {game} 当前游戏名\n"
            " {label} 结果标签（来自模板名/或 profile.json 的 templates[].label）\n"
            " {score} 匹配分数（支持格式：{score:.3f}）\n"
            " {id} 模板ID\n",
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

    # -------------------------
    # Actions: capture template
    # -------------------------
    def _action_capture_template(self, tpl: TemplateItem):
        def _inner(icon: pystray.Icon, item: pystray.MenuItem) -> None:
            try:
                saved = capture_to_template(self._profile, tpl)
                # templates might have been overwritten; just reload
                self._detector.reload_templates()
                self._dlg.info("抓取模板成功", f"已保存：\n{saved}")
            except Exception as e:
                self._dlg.info("抓取模板失败", repr(e))

        return _inner

    def _action_capture_auto(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        try:
            saved = capture_to_game_folder_auto(self._profile)
            # refresh profiles so auto-scanned templates include the new file
            self._refresh_profiles(keep_selected_id=self._profile_base.id)
            self._detector.reload_templates()
            self._dlg.info("抓取模板成功", f"已保存：\n{saved}")
        except Exception as e:
            self._dlg.info("抓取模板失败", repr(e))

    # -------------------------
    # ROI tuner actions
    # -------------------------
    def _action_set_roi_step(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        cur = float(self._cfg.get("roi_step", 0.005) or 0.005)
        v = self._dlg.ask_float(
            "设置 ROI 步长",
            "建议：0.002 ~ 0.020\n(0.005 约等于 1920 宽下 9~10 像素)",
            cur,
        )
        if v is None:
            return
        self._cfg["roi_step"] = max(0.0005, float(v))
        self._persist()
        self._rebuild_menu()

    def _action_preview_roi(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        try:
            bgr = grab_profile_roi_bgr(self._profile)
            tmp = Path(tempfile.gettempdir()) / f"roi_preview_{self._profile_base.id}.png"
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
        # profiles submenu + create new game entry
        profile_items = [
            pystray.MenuItem(
                p.display_name,
                self._action_select_profile(p.id),
                checked=self._is_profile_selected(p.id),
                radio=True,
            )
            for p in self._profiles
        ]
        profile_items = [
            pystray.MenuItem("➕ 新建游戏…", self._action_create_new_game),
            pystray.Menu.SEPARATOR,
            *profile_items,
        ]

        mode_menu = pystray.Menu(
            pystray.MenuItem("都要（弹窗 + 响铃）", self._action_set_mode("both"), checked=self._is_mode("both"), radio=True),
            pystray.MenuItem("只弹窗", self._action_set_mode("toast"), checked=self._is_mode("toast"), radio=True),
            pystray.MenuItem("只响铃", self._action_set_mode("sound"), checked=self._is_mode("sound"), radio=True),
        )

        settings_menu = pystray.Menu(
            pystray.MenuItem(f"匹配阈值 threshold = {self._detector.cfg.threshold:.3f}", self._action_set_threshold),
            pystray.MenuItem(f"回落差值 hysteresis = {self._detector.cfg.hysteresis:.3f}", self._action_set_hysteresis),
            pystray.MenuItem(f"冷却时间 cooldown_sec = {self._detector.cfg.cooldown_sec:.2f}s", self._action_set_cooldown),
            pystray.MenuItem(f"扫描间隔 scan_interval = {self._detector.cfg.scan_interval_sec:.2f}s", self._action_set_interval),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("提醒方式（弹窗/响铃）", mode_menu),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("编辑通知文本…", self._action_edit_text),
            pystray.MenuItem("发送测试通知", self._action_test_notify),
        )

        # capture menu:
        # - if profile has predefined templates -> keep old behavior
        # - otherwise offer "auto capture"
        if self._profile.templates:
            cap_items = [pystray.MenuItem(f"抓取：{t.label}", self._action_capture_template(t)) for t in self._profile.templates]
            capture_menu = pystray.Menu(*cap_items)
        else:
            capture_menu = pystray.Menu(
                pystray.MenuItem("抓取当前画面（自动命名）", self._action_capture_auto),
                pystray.MenuItem("（提示：抓取后会自动加入模板库）", lambda i, it: None),
            )

        # ROI tuner menu
        step = float(self._cfg.get("roi_step", 0.005) or 0.005)
        roi_menu = pystray.Menu(
            pystray.MenuItem("预览 ROI（打开图片）", self._action_preview_roi),
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
            pystray.MenuItem("重载模板/刷新资源", self._action_reload_templates),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self._action_quit),
        )

    def run(self) -> None:
        self._icon.run()