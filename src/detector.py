# src/detector.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from mss import mss

from profiles import GameProfile, resolve_resource_path
from roi import RoiPx, rel_to_px


# -------------------------
# 中文路径兼容：cv2.imdecode
# -------------------------
def load_gray_compat(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


@dataclass
class DetectorConfig:
    threshold: float = 0.82
    # 重新武装阈值：必须跌破这个值，才允许下一次触发
    # 建议 0.10~0.20
    hysteresis: float = 0.12

    scan_interval_sec: float = 0.20
    cooldown_sec: float = 3.0
    match_method: int = cv2.TM_CCOEFF_NORMED


@dataclass
class MatchResult:
    label: str
    template_id: str
    score: float


class Detector:
    """
    Multi-profile detector with:
    - cooldown
    - edge trigger (armed / disarmed) using hysteresis
    """

    def __init__(
        self,
        cfg: DetectorConfig,
        profile: GameProfile,
        on_match: Callable[[MatchResult], None],
    ):
        self.cfg = cfg
        self._profile_lock = threading.Lock()
        self._profile: GameProfile = profile

        self._on_match = on_match

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._tpls: Dict[str, Tuple[str, np.ndarray]] = {}
        self.reload_templates()

        self._last_fire_ts = 0.0

        # 关键状态：同一张结算界面只触发一次
        self._armed = True

    # -------------------------
    # Public controls
    # -------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop_evt.is_set())

    def set_profile(self, profile: GameProfile) -> None:
        with self._profile_lock:
            self._profile = profile
        self.reload_templates()
        # 切换游戏时重新武装
        self._armed = True

    def get_profile(self) -> GameProfile:
        with self._profile_lock:
            return self._profile

    def reload_templates(self) -> None:
        profile = self.get_profile()
        new_tpls: Dict[str, Tuple[str, np.ndarray]] = {}
        for t in profile.templates:
            try:
                abs_path = resolve_resource_path(t.path)
                gray = load_gray_compat(abs_path)
                new_tpls[t.id] = (t.label, gray)
            except Exception:
                # 模板不存在/读失败：跳过
                continue
        self._tpls = new_tpls

    # -------------------------
    # Internal loop
    # -------------------------
    def _get_screen_size(self) -> Tuple[int, int]:
        with mss() as sct:
            mon = sct.monitors[1]  # primary
            return int(mon["width"]), int(mon["height"])

    def _grab_roi(self, roi: RoiPx) -> np.ndarray:
        with mss() as sct:
            monitor = {"left": roi.left, "top": roi.top, "width": roi.width, "height": roi.height}
            img = np.array(sct.grab(monitor))  # BGRA
            return img[:, :, :3]  # BGR

    def _best_match(self, roi_bgr: np.ndarray) -> Optional[MatchResult]:
        if not self._tpls:
            return None

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        best: Optional[MatchResult] = None
        for tid, (label, templ_gray) in self._tpls.items():
            th, tw = templ_gray.shape[:2]
            rh, rw = gray.shape[:2]
            if th > rh or tw > rw:
                continue

            res = cv2.matchTemplate(gray, templ_gray, self.cfg.match_method)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            score = float(max_val)

            if best is None or score > best.score:
                best = MatchResult(label=label, template_id=tid, score=score)

        return best

    def _cooldown_ok(self) -> bool:
        return (time.time() - self._last_fire_ts) >= self.cfg.cooldown_sec

    def _fire(self, result: MatchResult) -> None:
        self._last_fire_ts = time.time()
        try:
            self._on_match(result)
        except Exception:
            pass

    def _run(self) -> None:
        screen_w, screen_h = self._get_screen_size()
        reset_threshold = max(0.0, self.cfg.threshold - self.cfg.hysteresis)

        while not self._stop_evt.is_set():
            try:
                profile = self.get_profile()
                roi_px = rel_to_px(profile.roi_rel, screen_w, screen_h)
                roi_bgr = self._grab_roi(roi_px)
                best = self._best_match(roi_bgr)

                if best is None:
                    # 匹配不到：重新武装
                    self._armed = True
                else:
                    # 跌破 reset_threshold：重新武装（意味着结算界面离开/变化足够大）
                    if best.score < reset_threshold:
                        self._armed = True

                    # 达到 threshold 且 armed 且 cooldown：触发一次并解除武装
                    if self._armed and best.score >= self.cfg.threshold and self._cooldown_ok():
                        self._fire(best)
                        self._armed = False

            except Exception as e:
                # 调试版可看到错误；-w 时不会显示
                print("Detector loop error:", repr(e))

            time.sleep(self.cfg.scan_interval_sec)