# src/app.py
import json
import sys
import threading
from pathlib import Path

import winsound

import pystray
from PIL import Image

from detector import DeltaResultDetector, DetectorConfig, NotifyMode, assets_dir


APP_NAME = "三角洲结算提醒"
CONFIG_FILE_NAME = "config.json"


def set_dpi_awareness():
    """避免 Win 缩放/DPI 导致 mss 截图坐标与实际屏幕不一致"""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        try:
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            return
        except Exception:
            pass
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


def base_dir() -> Path:
    # 打包后：exe所在目录；开发：项目根目录
    if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def config_path() -> Path:
    return base_dir() / CONFIG_FILE_NAME


def load_config() -> DetectorConfig:
    cfg = DetectorConfig()
    p = config_path()
    if not p.exists():
        return cfg

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        cfg.roi_left = int(data.get("roi_left", cfg.roi_left))
        cfg.roi_top = int(data.get("roi_top", cfg.roi_top))
        cfg.roi_width = int(data.get("roi_width", cfg.roi_width))
        cfg.roi_height = int(data.get("roi_height", cfg.roi_height))
        cfg.threshold = float(data.get("threshold", cfg.threshold))
        cfg.hysteresis = float(data.get("hysteresis", cfg.hysteresis))
        cfg.scan_interval = float(data.get("scan_interval", cfg.scan_interval))
        cfg.cooldown_sec = float(data.get("cooldown_sec", cfg.cooldown_sec))

        mode = data.get("mode", cfg.mode.value)
        if mode in (NotifyMode.BOTH.value, NotifyMode.SUCCESS.value, NotifyMode.FAIL.value):
            cfg.mode = NotifyMode(mode)
    except Exception:
        pass

    return cfg


def save_config(cfg: DetectorConfig) -> None:
    data = {
        "roi_left": cfg.roi_left,
        "roi_top": cfg.roi_top,
        "roi_width": cfg.roi_width,
        "roi_height": cfg.roi_height,
        "threshold": cfg.threshold,
        "hysteresis": cfg.hysteresis,
        "scan_interval": cfg.scan_interval,
        "cooldown_sec": cfg.cooldown_sec,
        "mode": cfg.mode.value,
    }
    config_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def toast_and_beep(result: str, score: float) -> None:
    # 1) 声音兜底（即使全屏/专注助手压制通知也能听到）
    try:
        winsound.Beep(1200, 180)
        winsound.Beep(900, 180)
    except Exception:
        try:
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except Exception:
            pass

    # 2) Win11 原生 Toast：winotify
    try:
        from winotify import Notification, audio

        ico = assets_dir() / "icon.ico"
        toast = Notification(
            app_id=APP_NAME,
            title="三角洲行动：检测到结算",
            msg=f"{result}（匹配度 {score:.2f}）\n打完这把就停手？",
            icon=str(ico) if ico.exists() else None,
        )
        toast.set_audio(audio.Default, loop=False)
        toast.show()
    except Exception:
        # 正式版：通知失败不落盘、不弹错（保持安静）
        pass


class TrayApp:
    def __init__(self):
        self.cfg = load_config()

        self.detector = DeltaResultDetector(
            config=self.cfg,
            on_result=self._on_result,
            on_status=self._on_status,
        )

        self._icon: pystray.Icon | None = None

    def _on_result(self, result: str, score: float) -> None:
        toast_and_beep(result, score)

    def _on_status(self, msg: str) -> None:
        # 正式版不输出日志；需要的话你可以自己 print(msg)
        pass

    # ---------- 菜单动作 ----------
    def action_start(self, icon, item):
        try:
            self.detector.start()
        except FileNotFoundError:
            # 模板缺失：用 toast 尝试提示一次（如果 toast 被压制至少有蜂鸣）
            toast_and_beep("缺少模板（请放置 success.png / fail.png）", 0.0)

    def action_stop(self, icon, item):
        self.detector.stop()

    def action_reload_templates(self, icon, item):
        try:
            self.detector.reload_templates()
        except FileNotFoundError:
            toast_and_beep("缺少模板（请放置 success.png / fail.png）", 0.0)

    def action_set_mode_both(self, icon, item):
        self.detector.set_mode(NotifyMode.BOTH)
        save_config(self.cfg)

    def action_set_mode_success(self, icon, item):
        self.detector.set_mode(NotifyMode.SUCCESS)
        save_config(self.cfg)

    def action_set_mode_fail(self, icon, item):
        self.detector.set_mode(NotifyMode.FAIL)
        save_config(self.cfg)

    def action_exit(self, icon, item):
        try:
            self.detector.stop()
        finally:
            save_config(self.cfg)
            if self._icon:
                self._icon.stop()

    # ---------- 菜单状态 ----------
    def checked_mode_both(self, item):
        return self.cfg.mode == NotifyMode.BOTH

    def checked_mode_success(self, item):
        return self.cfg.mode == NotifyMode.SUCCESS

    def checked_mode_fail(self, item):
        return self.cfg.mode == NotifyMode.FAIL

    def build_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem("启动检测", self.action_start, default=True),
            pystray.MenuItem("停止检测", self.action_stop),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "提醒模式",
                pystray.Menu(
                    pystray.MenuItem("两者都提醒", self.action_set_mode_both, checked=self.checked_mode_both, radio=True),
                    pystray.MenuItem("只提醒成功", self.action_set_mode_success, checked=self.checked_mode_success, radio=True),
                    pystray.MenuItem("只提醒失败", self.action_set_mode_fail, checked=self.checked_mode_fail, radio=True),
                ),
            ),
            pystray.MenuItem("重新加载模板", self.action_reload_templates),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("退出", self.action_exit),
        )

    def run(self):
        ico_path = assets_dir() / "icon.ico"
        image = Image.open(str(ico_path))
        self._icon = pystray.Icon(APP_NAME, image, APP_NAME, self.build_menu())

        # 启动后自动开始检测（不想自动启动就删掉这行）
        threading.Timer(0.2, lambda: self.action_start(None, None)).start()

        self._icon.run()


def main():
    set_dpi_awareness()
    TrayApp().run()


if __name__ == "__main__":
    main()