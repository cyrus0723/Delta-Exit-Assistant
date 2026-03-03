# src/app.py
from __future__ import annotations

import ctypes
import json
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pystray
from PIL import Image
import winsound

from winotify import Notification, audio

from detector import Detector, DetectorConfig, MatchResult
from profiles import (
    GameProfile,
    fallback_delta_profile_from_legacy_config,
    load_profiles_from_assets,
    pick_profile,
    resolve_resource_path,
)

# tkinter must run in a dedicated thread with mainloop
import tkinter as tk
from tkinter import messagebox, simpledialog

APP_NAME = "Delta Exit Assistant"


# -------------------------
# DPI Awareness (Win10/11)
# -------------------------
def enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


enable_dpi_awareness()


# -------------------------
# config.json (same dir as exe for simplicity)
# -------------------------
def exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def config_path() -> Path:
    return exe_dir() / "config.json"


def load_config() -> Dict[str, Any]:
    p = config_path()
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg: Dict[str, Any]) -> None:
    p = config_path()
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# -------------------------
# Tk dialog service (thread-safe)
# -------------------------
@dataclass
class _TkJob:
    fn: Callable[[], Any]
    done: threading.Event
    out: Dict[str, Any]


class TkDialogService:
    """
    Runs tkinter mainloop on a dedicated thread.
    All dialogs are executed on that thread via a job queue.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[_TkJob]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ready = threading.Event()
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.attributes("-topmost", True)
        self._ready.set()

        def poll():
            try:
                while True:
                    job = self._q.get_nowait()
                    try:
                        job.out["value"] = job.fn()
                    except Exception as e:
                        job.out["error"] = e
                        job.out["value"] = None
                    finally:
                        job.done.set()
            except queue.Empty:
                pass
            self._root.after(50, poll)

        self._root.after(50, poll)
        self._root.mainloop()

    def _call(self, fn: Callable[[], Any], timeout: float = 60.0) -> Any:
        job = _TkJob(fn=fn, done=threading.Event(), out={})
        self._q.put(job)
        job.done.wait(timeout=timeout)
        return job.out.get("value", None)

    def info(self, title: str, msg: str) -> None:
        def _f():
            messagebox.showinfo(title, msg, parent=self._root)
            return None

        self._call(_f)

    def ask_float(self, title: str, prompt: str, initial: float) -> Optional[float]:
        def _f():
            return simpledialog.askfloat(
                title, prompt, initialvalue=initial, parent=self._root
            )

        v = self._call(_f)
        return None if v is None else float(v)

    def ask_str(self, title: str, prompt: str, initial: str) -> Optional[str]:
        def _f():
            return simpledialog.askstring(
                title, prompt, initialvalue=initial, parent=self._root
            )

        v = self._call(_f)
        return None if v is None else str(v)


# -------------------------
# Notify mode
# -------------------------
# "both" -> toast + sound
# "toast" -> only toast
# "sound" -> only sound
VALID_NOTIFY_MODES = {"both", "toast", "sound"}


class TrayApp:
    def __init__(self) -> None:
        self._cfg = load_config()
        self._dlg = TkDialogService()

        # Profiles
        self._profiles: List[GameProfile] = load_profiles_from_assets()
        if not self._profiles:
            self._profiles = [fallback_delta_profile_from_legacy_config(self._cfg)]

        selected_id = str(self._cfg.get("selected_profile_id", "")).strip() or self._profiles[0].id
        self._profile: GameProfile = pick_profile(self._profiles, selected_id)

        # Detector config
        det_cfg = DetectorConfig(
            threshold=float(self._cfg.get("threshold", 0.82)),
            hysteresis=float(self._cfg.get("hysteresis", 0.12)),
            scan_interval_sec=float(self._cfg.get("scan_interval_sec", 0.20)),
            cooldown_sec=float(self._cfg.get("cooldown_sec", 3.0)),
        )

        # Notification templates
        self._title_tpl: str = str(self._cfg.get("title_tpl", "{game} 结算检测")).strip() or "{game} 结算检测"
        self._msg_tpl: str = str(self._cfg.get("msg_tpl", "{label}（score={score:.3f}）")).strip() or "{label}（score={score:.3f}）"

        # Notify mode
        mode = str(self._cfg.get("notify_mode", "both")).strip().lower()
        self._notify_mode: str = mode if mode in VALID_NOTIFY_MODES else "both"

        self._detector = Detector(det_cfg, self._profile, on_match=self._on_match)

        # tray icon
        icon_path = resolve_resource_path("assets/icon.ico")
        try:
            image = Image.open(icon_path)
        except Exception:
            image = Image.new("RGB", (64, 64), color=(0, 0, 0))

        self._icon = pystray.Icon(APP_NAME, image, APP_NAME)
        self._icon.menu = self._build_menu()

        # persist once (you said ok with config.json)
        self._persist()

    # -------------------------
    # Formatting
    # -------------------------
    def _format_text(self, tpl: str, result: MatchResult) -> str:
        data = {
            "game": self._profile.display_name,
            "label": result.label,
            "score": result.score,
            "id": result.template_id,
        }
        try:
            return tpl.format(**data)
        except Exception:
            return tpl

    # -------------------------
    # Notification + Sound
    # -------------------------
    def _show_toast(self, title: str, msg: str) -> None:
        toast = Notification(
            app_id=APP_NAME,
            title=title,
            msg=msg,
            icon=resolve_resource_path("assets/icon.ico"),
        )
        toast.set_audio(audio.Default, loop=False)
        toast.show()

    def _beep(self) -> None:
        winsound.MessageBeep(winsound.MB_ICONASTERISK)

    def _on_match(self, result: MatchResult) -> None:
        title = self._format_text(self._title_tpl, result)
        msg = self._format_text(self._msg_tpl, result)

        # toast
        if self._notify_mode in ("both", "toast"):
            try:
                self._show_toast(title, msg)
            except Exception as e:
                print("Toast error:", repr(e))

        # sound
        if self._notify_mode in ("both", "sound"):
            try:
                self._beep()
            except Exception:
                pass

    # -------------------------
    # Persistence
    # -------------------------
    def _persist(self) -> None:
        self._cfg["selected_profile_id"] = self._profile.id
        self._cfg["threshold"] = self._detector.cfg.threshold
        self._cfg["hysteresis"] = self._detector.cfg.hysteresis
        self._cfg["scan_interval_sec"] = self._detector.cfg.scan_interval_sec
        self._cfg["cooldown_sec"] = self._detector.cfg.cooldown_sec
        self._cfg["title_tpl"] = self._title_tpl
        self._cfg["msg_tpl"] = self._msg_tpl
        self._cfg["notify_mode"] = self._notify_mode
        save_config(self._cfg)

    def _rebuild_menu(self) -> None:
        self._icon.menu = self._build_menu()
        self._icon.update_menu()

    # -------------------------
    # Menu actions: start/stop
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
    # Menu actions: profiles
    # -------------------------
    def _is_profile_selected(self, profile_id: str):
        def _inner(item: pystray.MenuItem) -> bool:
            return self._profile.id == profile_id
        return _inner

    def _action_select_profile(self, profile_id: str):
        def _inner(icon: pystray.Icon, item: pystray.MenuItem) -> None:
            for p in self._profiles:
                if p.id == profile_id:
                    self._profile = p
                    break
            self._detector.set_profile(self._profile)
            self._persist()
            self._rebuild_menu()
        return _inner

    # -------------------------
    # Menu actions: tuning
    # -------------------------
    def _action_set_threshold(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        v = self._dlg.ask_float("设置匹配阈值", "threshold（建议 0.70 ~ 0.90）", self._detector.cfg.threshold)
        if v is None:
            return
        v = max(0.0, min(1.0, float(v)))
        self._detector.cfg.threshold = v
        self._persist()
        self._rebuild_menu()

    def _action_set_hysteresis(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        v = self._dlg.ask_float("设置回落差值", "hysteresis（建议 0.08 ~ 0.20）\n越大越不容易重复提示", self._detector.cfg.hysteresis)
        if v is None:
            return
        v = max(0.0, min(1.0, float(v)))
        self._detector.cfg.hysteresis = v
        self._persist()
        self._rebuild_menu()

    def _action_set_cooldown(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        v = self._dlg.ask_float("设置冷却时间", "cooldown_sec（秒，建议 1.0 ~ 10.0）", self._detector.cfg.cooldown_sec)
        if v is None:
            return
        v = max(0.0, float(v))
        self._detector.cfg.cooldown_sec = v
        self._persist()
        self._rebuild_menu()

    def _action_set_interval(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        v = self._dlg.ask_float("设置扫描间隔", "scan_interval_sec（秒，建议 0.10 ~ 0.50）\n越小越灵敏但更耗资源", self._detector.cfg.scan_interval_sec)
        if v is None:
            return
        v = max(0.05, float(v))
        self._detector.cfg.scan_interval_sec = v
        self._persist()
        self._rebuild_menu()

    # -------------------------
    # Menu actions: notify mode
    # -------------------------
    def _is_mode(self, mode: str):
        def _inner(item: pystray.MenuItem) -> bool:
            return self._notify_mode == mode
        return _inner

    def _action_set_mode(self, mode: str):
        def _inner(icon: pystray.Icon, item: pystray.MenuItem) -> None:
            if mode not in VALID_NOTIFY_MODES:
                return
            self._notify_mode = mode
            self._persist()
            self._rebuild_menu()
        return _inner

    # -------------------------
    # Menu actions: text templates
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

        title = self._dlg.ask_str("编辑通知标题", "例如：{game} 结算检测", self._title_tpl)
        if title is None:
            return
        msg = self._dlg.ask_str("编辑通知正文", "例如：{label}（score={score:.3f}）", self._msg_tpl)
        if msg is None:
            return

        title = title.strip()
        msg = msg.strip()
        if title:
            self._title_tpl = title
        if msg:
            self._msg_tpl = msg

        self._persist()
        self._rebuild_menu()

    def _action_test_notify(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        fake = MatchResult(label="测试成功", template_id="test", score=0.999)
        self._on_match(fake)

    # -------------------------
    # Menu
    # -------------------------
    def _build_menu(self) -> pystray.Menu:
        profile_items = [
            pystray.MenuItem(
                p.display_name,
                self._action_select_profile(p.id),
                checked=self._is_profile_selected(p.id),
                radio=True,
            )
            for p in self._profiles
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

        return pystray.Menu(
            pystray.MenuItem("启动检测", self._action_start, enabled=lambda item: not self._detector.is_running()),
            pystray.MenuItem("停止检测", self._action_stop, enabled=lambda item: self._detector.is_running()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("选择游戏", pystray.Menu(*profile_items)),
            pystray.MenuItem("设置", settings_menu),
            pystray.MenuItem("重载模板", self._action_reload_templates),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self._action_quit),
        )

    def run(self) -> None:
        self._icon.run()


def main() -> None:
    TrayApp().run()


if __name__ == "__main__":
    main()