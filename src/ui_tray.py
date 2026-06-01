# src/ui_tray.py
from __future__ import annotations

import ctypes
import os
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import pystray
from PIL import Image

from paths import ensure_assets_dirs, assets_dir
from detector import Detector, DetectorConfig, MatchResult
from profiles import (
    GameProfile,
    TemplateItem,
    load_profiles_from_assets,
    pick_profile,
    normalize_profile_id,
    save_profile,
    profile_file_path,
    templates_dir,
    resolve_path,
)
from ui_dialogs import TkDialogService
from roi_selector import select_roi_fullscreen
from roi import RoiRel

from notify import Notifier, NotifySettings, VALID_NOTIFY_MODES
from sleep_reminder import SleepReminder, SleepReminderConfig, normalize_bed_time
from i18n import (
    LANGUAGES,
    LANGUAGE_NAMES,
    default_notify_templates,
    default_sleep_templates,
    normalize_language,
    tr,
)
from capture import capture_to_template, grab_profile_roi_bgr
from config_store import load_config, save_config

from roi_tuner import (
    RoiTuner,
    load_roi_override,
    save_roi_override,
    clear_roi_override,
)

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
        ensure_assets_dirs()

        self._cfg: Dict[str, Any] = load_config()
        self._ui_language = normalize_language(self._cfg.get("ui_language", "zh"))
        self._dlg = TkDialogService()

        self._profiles: List[GameProfile] = load_profiles_from_assets()
        if not self._profiles:
            # If user deleted bundled profiles, create a minimal placeholder.
            # (But in your new packaging, assets/profiles should exist.)
            self._profiles = []

        if not self._profiles:
            self._dlg.info(self._t("missing_profiles_title"), self._t("missing_profiles_msg"))
            raise SystemExit(1)

        selected_id = str(self._cfg.get("selected_profile_id", "")).strip() or self._profiles[0].id
        self._profile_base: GameProfile = pick_profile(self._profiles, selected_id)
        self._profile: GameProfile = self._apply_roi_override(self._profile_base)

        det_cfg = DetectorConfig(
            threshold=float(self._cfg.get("threshold", 0.82)),
            hysteresis=float(self._cfg.get("hysteresis", 0.12)),
            scan_interval_sec=float(self._cfg.get("scan_interval_sec", 0.20)),
            cooldown_sec=float(self._cfg.get("cooldown_sec", 3.0)),
        )

        default_title_tpl, default_msg_tpl = default_notify_templates(self._ui_language)
        title_tpl = str(self._cfg.get("title_tpl", default_title_tpl)).strip() or default_title_tpl
        msg_tpl = str(self._cfg.get("msg_tpl", default_msg_tpl)).strip() or default_msg_tpl
        mode = str(self._cfg.get("notify_mode", "both")).strip().lower()
        if mode not in VALID_NOTIFY_MODES:
            mode = "both"

        self._notify_settings = NotifySettings(title_tpl=title_tpl, msg_tpl=msg_tpl, mode=mode)
        self._notifier = Notifier(APP_NAME, self._profile, self._notify_settings)

        default_sleep_title, default_sleep_msg = default_sleep_templates(self._ui_language)
        self._sleep_reminder = SleepReminder(
            SleepReminderConfig(
                enabled=bool(self._cfg.get("sleep_reminder_enabled", True)),
                bed_time=str(self._cfg.get("sleep_bed_time", "22:00")).strip() or "22:00",
                stop_after_bed_time=bool(self._cfg.get("sleep_stop_after_bed_time", True)),
                auto_start_detection=bool(self._cfg.get("sleep_auto_start_detection", True)),
                title_tpl=str(self._cfg.get("sleep_title_tpl", default_sleep_title)).strip() or default_sleep_title,
                msg_tpl=str(self._cfg.get("sleep_msg_tpl", default_sleep_msg)).strip() or default_sleep_msg,
            )
        )

        self._detector = Detector(det_cfg, self._profile, on_match=self._on_match)
        self._sleep_window_active = False
        self._sleep_monitor_stop = threading.Event()
        self._sleep_monitor_thread = threading.Thread(target=self._sleep_monitor_loop, daemon=True)
        self._refresh_sleep_window_state()
        self._sleep_monitor_thread.start()

        icon_path = resolve_path("assets/icon.ico")
        try:
            image = Image.open(icon_path)
        except Exception:
            image = Image.new("RGB", (64, 64), color=(0, 0, 0))

        self._icon = pystray.Icon(APP_NAME, image, APP_NAME)
        self._icon.menu = self._build_menu()
        self._persist()

    def _t(self, key: str, **kwargs: object) -> str:
        return tr(self._ui_language, key, **kwargs)

    # ---------- ROI override ----------
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

    # ---------- Detector callback ----------
    def _on_match(self, result: MatchResult) -> None:
        if self._sleep_reminder.is_active_now(self._profile):
            cfg = self._sleep_reminder.cfg
            self._notifier.notify_sleep(result, cfg.title_tpl, cfg.msg_tpl)
        else:
            self._notifier.notify(result)

    # ---------- sleep reminder ----------
    def _refresh_sleep_window_state(self, force_auto_start: bool = False) -> None:
        active = self._sleep_reminder.is_active_now(self._profile)
        if active and (force_auto_start or not self._sleep_window_active) and self._sleep_reminder.should_auto_start_detection(self._profile):
            self._detector.start()
        self._sleep_window_active = active

    def _sleep_monitor_loop(self) -> None:
        while not self._sleep_monitor_stop.wait(15.0):
            self._refresh_sleep_window_state()

    # ---------- persistence ----------
    def _persist(self) -> None:
        self._cfg["selected_profile_id"] = self._profile_base.id
        self._cfg["ui_language"] = self._ui_language
        self._cfg["threshold"] = self._detector.cfg.threshold
        self._cfg["hysteresis"] = self._detector.cfg.hysteresis
        self._cfg["scan_interval_sec"] = self._detector.cfg.scan_interval_sec
        self._cfg["cooldown_sec"] = self._detector.cfg.cooldown_sec
        self._cfg["title_tpl"] = self._notify_settings.title_tpl
        self._cfg["msg_tpl"] = self._notify_settings.msg_tpl
        self._cfg["notify_mode"] = self._notify_settings.mode
        sleep_cfg = self._sleep_reminder.cfg
        self._cfg["sleep_reminder_enabled"] = sleep_cfg.enabled
        self._cfg["sleep_bed_time"] = sleep_cfg.bed_time
        self._cfg["sleep_stop_after_bed_time"] = sleep_cfg.stop_after_bed_time
        self._cfg["sleep_auto_start_detection"] = sleep_cfg.auto_start_detection
        self._cfg["sleep_title_tpl"] = sleep_cfg.title_tpl
        self._cfg["sleep_msg_tpl"] = sleep_cfg.msg_tpl
        save_config(self._cfg)

    def _rebuild_menu(self) -> None:
        self._icon.menu = self._build_menu()
        self._icon.update_menu()

    # ---------- basic actions ----------
    def _action_start(self, icon, item) -> None:
        self._detector.start()

    def _action_stop(self, icon, item) -> None:
        self._detector.stop()

    def _action_reload_profiles(self, icon, item) -> None:
        self._profiles = load_profiles_from_assets()
        # keep current selection if possible
        self._profile_base = pick_profile(self._profiles, self._profile_base.id)
        self._profile = self._apply_roi_override(self._profile_base)
        self._detector.set_profile(self._profile)
        self._notifier.set_profile(self._profile)
        self._refresh_sleep_window_state()
        self._persist()
        self._rebuild_menu()

    def _action_reload_templates(self, icon, item) -> None:
        self._detector.reload_templates()

    def _action_quit(self, icon, item) -> None:
        self._sleep_monitor_stop.set()
        try:
            self._detector.stop()
        except Exception:
            pass
        icon.stop()

    # ---------- select profile ----------
    def _is_profile_selected(self, pid: str):
        def _inner(_):
            return self._profile_base.id == pid
        return _inner

    def _action_select_profile(self, pid: str):
        def _inner(icon, item):
            for p in self._profiles:
                if p.id == pid:
                    self._profile_base = p
                    break
            self._profile = self._apply_roi_override(self._profile_base)
            self._detector.set_profile(self._profile)
            self._notifier.set_profile(self._profile)
            self._refresh_sleep_window_state(force_auto_start=True)
            self._persist()
            self._rebuild_menu()
        return _inner

    # ---------- NEW GAME WORKFLOW ----------
    def _action_new_game(self, icon, item) -> None:
        name = self._dlg.ask_str(self._t("new_game_title"), self._t("new_game_prompt"), "")
        if not name or not name.strip():
            return

        pid = normalize_profile_id(name)
        if not pid:
            return

        fp = profile_file_path(pid)
        if fp.exists():
            ok = self._dlg.confirm(self._t("existing_game_title"), self._t("existing_game_msg", profile_id=pid))
            if not ok:
                return

        # ensure templates folder exists now (prevents capture FileNotFound surprises)
        templates_dir(pid)

        self._dlg.info(self._t("select_roi_title"), self._t("select_roi_msg"))
        roi_rel: RoiRel | None = self._dlg.run_in_tk(
            lambda root: select_roi_fullscreen(
                root,
                self._t("select_roi_window", name=name),
                self._t("select_roi_instruction"),
            )
        )
        if roi_rel is None:
            return

        # Pre-create template definitions (files may not exist yet; detector will skip missing)
        tpls = [
            TemplateItem(id=f"{pid}_win", label=self._t("tpl_win"), path=f"assets/templates/{pid}/win.png"),
            TemplateItem(id=f"{pid}_lose", label=self._t("tpl_lose"), path=f"assets/templates/{pid}/lose.png"),
            TemplateItem(id=f"{pid}_draw", label=self._t("tpl_draw"), path=f"assets/templates/{pid}/draw.png"),
        ]
        profile = GameProfile(id=pid, display_name=name.strip(), roi_rel=roi_rel, templates=tpls)

        try:
            out = save_profile(profile)
        except Exception as e:
            self._dlg.info(self._t("create_failed"), repr(e))
            return

        # reload and switch
        self._profiles = load_profiles_from_assets()
        self._profile_base = pick_profile(self._profiles, pid)
        self._profile = self._apply_roi_override(self._profile_base)
        self._detector.set_profile(self._profile)
        self._notifier.set_profile(self._profile)
        self._refresh_sleep_window_state(force_auto_start=True)

        self._persist()
        self._rebuild_menu()

        self._dlg.info(
            self._t("create_success"),
            self._t("create_success_msg", path=out),
        )

    # ---------- capture template ----------
    def _action_capture_template(self, tpl: TemplateItem):
        def _inner(icon, item):
            try:
                saved = capture_to_template(self._profile, tpl)
                self._detector.reload_templates()
                self._dlg.info(self._t("capture_success"), self._t("capture_success_msg", path=saved))
            except Exception as e:
                self._dlg.info(self._t("capture_failed"), repr(e))
        return _inner

    # ---------- ROI tuner (optional keep) ----------
    def _action_preview_roi(self, icon, item) -> None:
        try:
            bgr = grab_profile_roi_bgr(self._profile)
            tmp = Path(tempfile.gettempdir()) / f"roi_preview_{self._profile_base.id}.png"
            import cv2
            ok, buf = cv2.imencode(".png", bgr)
            if ok:
                tmp.write_bytes(buf.tobytes())
                os.startfile(str(tmp))
        except Exception as e:
            self._dlg.info(self._t("preview_failed"), repr(e))

    def _action_set_roi_step(self, icon, item) -> None:
        cur = float(self._cfg.get("roi_step", 0.005) or 0.005)
        v = self._dlg.ask_float(self._t("set_roi_step_title"), self._t("set_roi_step_prompt"), cur)
        if v is None:
            return
        self._cfg["roi_step"] = max(0.0005, float(v))
        self._persist()
        self._rebuild_menu()

    def _action_roi_move(self, direction: str):
        def _inner(icon, item):
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
        def _inner(icon, item):
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

    def _action_roi_reset(self, icon, item) -> None:
        clear_roi_override(self._cfg, self._profile_base.id)
        self._profile = self._profile_base
        self._detector.set_profile(self._profile)
        self._notifier.set_profile(self._profile)
        self._persist()
        self._rebuild_menu()

    # ---------- settings ----------
    def _action_set_threshold(self, icon, item) -> None:
        v = self._dlg.ask_float(self._t("threshold_title"), self._t("threshold_prompt"), self._detector.cfg.threshold)
        if v is None:
            return
        self._detector.cfg.threshold = max(0.0, min(1.0, float(v)))
        self._persist()
        self._rebuild_menu()

    def _action_set_hysteresis(self, icon, item) -> None:
        v = self._dlg.ask_float(self._t("hysteresis_title"), self._t("hysteresis_prompt"), self._detector.cfg.hysteresis)
        if v is None:
            return
        self._detector.cfg.hysteresis = max(0.0, min(1.0, float(v)))
        self._persist()
        self._rebuild_menu()

    def _action_set_cooldown(self, icon, item) -> None:
        v = self._dlg.ask_float(self._t("cooldown_title"), self._t("cooldown_prompt"), self._detector.cfg.cooldown_sec)
        if v is None:
            return
        self._detector.cfg.cooldown_sec = max(0.0, float(v))
        self._persist()
        self._rebuild_menu()

    def _action_set_interval(self, icon, item) -> None:
        v = self._dlg.ask_float(self._t("interval_title"), self._t("interval_prompt"), self._detector.cfg.scan_interval_sec)
        if v is None:
            return
        self._detector.cfg.scan_interval_sec = max(0.05, float(v))
        self._persist()
        self._rebuild_menu()

    def _is_mode(self, mode: str):
        def _inner(_):
            return self._notify_settings.mode == mode
        return _inner

    def _action_set_mode(self, mode: str):
        def _inner(icon, item):
            if mode not in VALID_NOTIFY_MODES:
                return
            self._notify_settings.mode = mode
            self._persist()
            self._rebuild_menu()
        return _inner

    def _action_edit_text(self, icon, item) -> None:
        self._dlg.info(self._t("placeholders_title"), self._t("placeholders_msg"))
        title = self._dlg.ask_str(self._t("edit_title"), self._t("edit_title_prompt"), self._notify_settings.title_tpl)
        if title is None:
            return
        msg = self._dlg.ask_str(self._t("edit_msg"), self._t("edit_msg_prompt"), self._notify_settings.msg_tpl)
        if msg is None:
            return
        if title.strip():
            self._notify_settings.title_tpl = title.strip()
        if msg.strip():
            self._notify_settings.msg_tpl = msg.strip()
        self._persist()
        self._rebuild_menu()

    # ---------- sleep reminder settings ----------
    def _is_sleep_flag_enabled(self, attr: str):
        def _inner(_):
            return bool(getattr(self._sleep_reminder.cfg, attr))
        return _inner

    def _action_toggle_sleep_flag(self, attr: str):
        def _inner(icon, item):
            cfg = self._sleep_reminder.cfg
            setattr(cfg, attr, not bool(getattr(cfg, attr)))
            self._refresh_sleep_window_state(force_auto_start=True)
            self._persist()
            self._rebuild_menu()
        return _inner

    def _action_set_sleep_bed_time(self, icon, item) -> None:
        cfg = self._sleep_reminder.cfg
        value = self._dlg.ask_str(self._t("set_bed_time_title"), self._t("set_bed_time_prompt"), cfg.bed_time)
        if value is None:
            return
        try:
            cfg.bed_time = normalize_bed_time(value)
        except ValueError:
            self._dlg.info(self._t("invalid_time_title"), self._t("invalid_time_msg"))
            return
        self._refresh_sleep_window_state(force_auto_start=True)
        self._persist()
        self._rebuild_menu()

    def _action_set_sleep_lead_minutes(self, icon, item) -> None:
        value = self._dlg.ask_float(
            self._t("set_lead_title"),
            self._t("set_lead_prompt", game=self._profile_base.display_name),
            float(self._profile_base.sleep_lead_minutes),
        )
        if value is None:
            return

        profile = replace(self._profile_base, sleep_lead_minutes=max(0, int(value)))
        try:
            save_profile(profile)
        except Exception as e:
            self._dlg.info(self._t("save_failed"), repr(e))
            return

        self._profiles = [profile if p.id == profile.id else p for p in self._profiles]
        self._profile_base = profile
        self._profile = self._apply_roi_override(profile)
        self._detector.set_profile(self._profile)
        self._notifier.set_profile(self._profile)
        self._refresh_sleep_window_state(force_auto_start=True)
        self._persist()
        self._rebuild_menu()

    def _action_edit_sleep_text(self, icon, item) -> None:
        cfg = self._sleep_reminder.cfg
        title = self._dlg.ask_str(self._t("edit_sleep_title"), self._t("edit_sleep_title_prompt"), cfg.title_tpl)
        if title is None:
            return
        msg = self._dlg.ask_str(self._t("edit_sleep_msg"), self._t("edit_sleep_msg_prompt"), cfg.msg_tpl)
        if msg is None:
            return
        if title.strip():
            cfg.title_tpl = title.strip()
        if msg.strip():
            cfg.msg_tpl = msg.strip()
        self._persist()
        self._rebuild_menu()

    # ---------- UI language ----------
    def _is_language(self, language: str):
        def _inner(_):
            return self._ui_language == language
        return _inner

    def _action_set_language(self, language: str):
        def _inner(icon, item):
            old_notify_defaults = default_notify_templates(self._ui_language)
            old_sleep_defaults = default_sleep_templates(self._ui_language)
            self._ui_language = normalize_language(language)
            new_notify_defaults = default_notify_templates(self._ui_language)
            new_sleep_defaults = default_sleep_templates(self._ui_language)

            if (self._notify_settings.title_tpl, self._notify_settings.msg_tpl) == old_notify_defaults:
                self._notify_settings.title_tpl, self._notify_settings.msg_tpl = new_notify_defaults
            sleep_cfg = self._sleep_reminder.cfg
            if (sleep_cfg.title_tpl, sleep_cfg.msg_tpl) == old_sleep_defaults:
                sleep_cfg.title_tpl, sleep_cfg.msg_tpl = new_sleep_defaults

            self._persist()
            self._rebuild_menu()
        return _inner

    # ---------- build menu ----------
    def _build_menu(self) -> pystray.Menu:
        # Profiles menu also exposes the add-game workflow.
        profile_items = [pystray.MenuItem(self._t("new_game"), self._action_new_game), pystray.Menu.SEPARATOR]
        profile_items.extend(
            pystray.MenuItem(p.display_name, self._action_select_profile(p.id), checked=self._is_profile_selected(p.id), radio=True)
            for p in self._profiles
        )

        cap_items = [pystray.MenuItem(self._t("capture_label", label=t.label), self._action_capture_template(t)) for t in self._profile.templates]
        capture_menu = pystray.Menu(*cap_items) if cap_items else pystray.Menu(pystray.MenuItem(self._t("no_templates"), lambda i, it: None))

        step = float(self._cfg.get("roi_step", 0.005) or 0.005)
        roi_menu = pystray.Menu(
            pystray.MenuItem(self._t("preview_roi"), self._action_preview_roi),
            pystray.MenuItem(self._t("set_roi_step", step=step), self._action_set_roi_step),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(self._t("move_up"), self._action_roi_move("up")),
            pystray.MenuItem(self._t("move_down"), self._action_roi_move("down")),
            pystray.MenuItem(self._t("move_left"), self._action_roi_move("left")),
            pystray.MenuItem(self._t("move_right"), self._action_roi_move("right")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(self._t("wider"), self._action_roi_resize("wider")),
            pystray.MenuItem(self._t("narrower"), self._action_roi_resize("narrower")),
            pystray.MenuItem(self._t("taller"), self._action_roi_resize("taller")),
            pystray.MenuItem(self._t("shorter"), self._action_roi_resize("shorter")),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(self._t("reset_roi"), self._action_roi_reset),
        )

        mode_menu = pystray.Menu(
            pystray.MenuItem(self._t("mode_both"), self._action_set_mode("both"), checked=self._is_mode("both"), radio=True),
            pystray.MenuItem(self._t("mode_toast"), self._action_set_mode("toast"), checked=self._is_mode("toast"), radio=True),
            pystray.MenuItem(self._t("mode_sound"), self._action_set_mode("sound"), checked=self._is_mode("sound"), radio=True),
        )

        sleep_cfg = self._sleep_reminder.cfg
        sleep_menu = pystray.Menu(
            pystray.MenuItem(
                self._t("enable_sleep_reminder"),
                self._action_toggle_sleep_flag("enabled"),
                checked=self._is_sleep_flag_enabled("enabled"),
            ),
            pystray.MenuItem(self._t("sleep_bed_time", time=sleep_cfg.bed_time), self._action_set_sleep_bed_time),
            pystray.MenuItem(
                self._t("sleep_lead_minutes", minutes=self._profile_base.sleep_lead_minutes),
                self._action_set_sleep_lead_minutes,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                self._t("stop_after_bed_time"),
                self._action_toggle_sleep_flag("stop_after_bed_time"),
                checked=self._is_sleep_flag_enabled("stop_after_bed_time"),
            ),
            pystray.MenuItem(
                self._t("auto_start_detection"),
                self._action_toggle_sleep_flag("auto_start_detection"),
                checked=self._is_sleep_flag_enabled("auto_start_detection"),
            ),
            pystray.MenuItem(self._t("edit_sleep_text"), self._action_edit_sleep_text),
        )

        language_menu = pystray.Menu(
            *(
                pystray.MenuItem(
                    LANGUAGE_NAMES[language],
                    self._action_set_language(language),
                    checked=self._is_language(language),
                    radio=True,
                )
                for language in LANGUAGES
            )
        )

        settings_menu = pystray.Menu(
            pystray.MenuItem(f"threshold={self._detector.cfg.threshold:.3f}", self._action_set_threshold),
            pystray.MenuItem(f"hysteresis={self._detector.cfg.hysteresis:.3f}", self._action_set_hysteresis),
            pystray.MenuItem(f"cooldown={self._detector.cfg.cooldown_sec:.2f}s", self._action_set_cooldown),
            pystray.MenuItem(f"interval={self._detector.cfg.scan_interval_sec:.2f}s", self._action_set_interval),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(self._t("notify_mode"), mode_menu),
            pystray.MenuItem(self._t("edit_notify_text"), self._action_edit_text),
            pystray.MenuItem(self._t("sleep_reminder"), sleep_menu),
            pystray.MenuItem(self._t("language"), language_menu),
        )

        return pystray.Menu(
            pystray.MenuItem(self._t("start_detection"), self._action_start, enabled=lambda _: not self._detector.is_running()),
            pystray.MenuItem(self._t("stop_detection"), self._action_stop, enabled=lambda _: self._detector.is_running()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(self._t("new_game"), self._action_new_game),
            pystray.MenuItem(self._t("select_game"), pystray.Menu(*profile_items)),
            pystray.MenuItem(self._t("roi_adjust"), roi_menu),
            pystray.MenuItem(self._t("capture_template"), capture_menu),
            pystray.MenuItem(self._t("settings"), settings_menu),
            pystray.MenuItem(self._t("reload_templates"), self._action_reload_templates),
            pystray.MenuItem(self._t("reload_profiles"), self._action_reload_profiles),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(self._t("quit"), self._action_quit),
        )

    def run(self) -> None:
        self._icon.run()
