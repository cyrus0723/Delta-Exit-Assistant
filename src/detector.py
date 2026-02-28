# src/detector.py
import sys
import time
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from mss import mss


class NotifyMode(str, Enum):
    BOTH = "both"       # 成功+失败都提醒
    SUCCESS = "success" # 只提醒成功
    FAIL = "fail"       # 只提醒失败


@dataclass
class DetectorConfig:
    roi_left: int = 150
    roi_top: int = 150
    roi_width: int = 500
    roi_height: int = 160

    threshold: float = 0.82
    hysteresis: float = 0.08
    scan_interval: float = 0.20

    # 提醒冷却（建议 6~12 秒，避免结算页动画导致重复触发）
    cooldown_sec: float = 8.0

    mode: NotifyMode = NotifyMode.BOTH


def _resource_base_dir() -> Path:
    """
    资源目录：开发 / PyInstaller onefile / onedir 兼容（只读）
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def assets_dir() -> Path:
    return _resource_base_dir() / "assets"


def templates_dir() -> Path:
    return assets_dir() / "templates"


def grab_roi(sct: mss, roi: dict) -> np.ndarray:
    shot = sct.grab(roi)
    frame = np.array(shot)[:, :, :3]  # BGRA -> BGR
    return frame


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def load_gray(path: Path) -> np.ndarray:
    """
    兼容中文路径：np.fromfile + cv2.imdecode
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"模板不存在：{p}")

    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"读不到模板（可能损坏/格式不支持）：{p}")
    return img


def match_score(screen_gray: np.ndarray, templ_gray: np.ndarray) -> float:
    res = cv2.matchTemplate(screen_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
    return float(res.max())


class DeltaResultDetector:
    """
    后台线程检测器：
    - 边沿触发：进入结算页只触发一次
    - 回滞：离开结算页才允许下一次触发
    - 冷却：避免结算动画导致短时间重复提醒
    """

    def __init__(
        self,
        config: DetectorConfig,
        on_result: Callable[[str, float], None],
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        self.on_result = on_result
        self.on_status = on_status or (lambda _: None)

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._templ_win: Optional[np.ndarray] = None
        self._templ_lose: Optional[np.ndarray] = None

        self._in_result_screen = False
        self._last_notify_ts = 0.0

    def _roi_dict(self) -> dict:
        return {
            "left": self.config.roi_left,
            "top": self.config.roi_top,
            "width": self.config.roi_width,
            "height": self.config.roi_height,
        }

    def reload_templates(self) -> Tuple[Path, Path]:
        t_win = templates_dir() / "success.png"
        t_lose = templates_dir() / "fail.png"
        self._templ_win = load_gray(t_win)
        self._templ_lose = load_gray(t_lose)
        self.on_status("模板加载成功")
        return t_win, t_lose

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self.reload_templates()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="detector", daemon=True)
        self._thread.start()
        self.on_status("检测已启动")

    def stop(self) -> None:
        self._stop_evt.set()
        self.on_status("检测已停止")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def set_mode(self, mode: NotifyMode) -> None:
        self.config.mode = mode
        self.on_status(f"提醒模式：{mode.value}")

    def _should_notify_for_result(self, result: str) -> bool:
        if self.config.mode == NotifyMode.BOTH:
            return True
        if self.config.mode == NotifyMode.SUCCESS:
            return result == "撤离成功"
        if self.config.mode == NotifyMode.FAIL:
            return result == "撤离失败"
        return True

    def _cooldown_ok(self) -> bool:
        return (time.time() - self._last_notify_ts) >= max(0.0, self.config.cooldown_sec)

    def _mark_notified(self) -> None:
        self._last_notify_ts = time.time()

    def _run(self) -> None:
        roi = self._roi_dict()

        if self._templ_win is None or self._templ_lose is None:
            self.on_status("模板未加载，停止检测")
            return

        with mss() as sct:
            while not self._stop_evt.is_set():
                frame = grab_roi(sct, roi)
                gray = preprocess(frame)

                s_win = match_score(gray, self._templ_win)
                s_lose = match_score(gray, self._templ_lose)
                best = max(s_win, s_lose)

                # 进入结算：边沿触发
                if best >= self.config.threshold and not self._in_result_screen:
                    result = "撤离成功" if s_win >= s_lose else "撤离失败"
                    self._in_result_screen = True

                    if self._should_notify_for_result(result) and self._cooldown_ok():
                        self.on_result(result, best)
                        self._mark_notified()

                # 离开结算：回滞
                elif self._in_result_screen and best < (self.config.threshold - self.config.hysteresis):
                    self._in_result_screen = False

                time.sleep(self.config.scan_interval)