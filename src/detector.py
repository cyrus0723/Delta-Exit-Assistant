# src/detector.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from mss import mss

from profiles import GameProfile, resolve_resource_path
from roi import RoiPx, rel_to_px


# -------------------------
# 中文路径兼容：cv2.imdecode
# -------------------------
def load_gray_compat(path: str) -> np.ndarray:
    """
    Load image as grayscale with Chinese path compatibility.
    """
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
    scan_interval_sec: float = 0.20
    cooldown_sec: float = 3.0
    # match method: TM_CCOEFF_NORMED is a good default for UI templates
    match_method: int = cv2.TM_CCOEFF_NORMED


@dataclass
class MatchResult:
    label: str
    template_id: str
    score: float


class Detector:
    """
    Minimal viable multi-profile detector.

    Responsibilities:
    - Keep current GameProfile
    - Load templates for that profile
    - Each loop: ROI from profile -> screenshot -> best match among templates
    - Trigger callback when best score >= threshold and cooldown passed
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

        # loaded template grayscale cache: (template_id -> (label, gray_img))
        self._tpls: Dict[str, Tuple[str, np.ndarray]] = {}
        self.reload_templates()

        self._last_fire_ts = 0.0

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

    def get_profile(self) -> GameProfile:
        with self._profile_lock:
            return self._profile

    def reload_templates(self) -> None:
        profile = self.get_profile()
        new_tpls: Dict[str, Tuple[str, np.ndarray]] = {}
        for t in profile.templates:
            abs_path = resolve_resource_path(t.path)
            gray = load_gray_compat(abs_path)
            new_tpls[t.id] = (t.label, gray)
        self._tpls = new_tpls

    # -------------------------
    # Internal loop
    # -------------------------
    def _get_screen_size(self) -> Tuple[int, int]:
        """
        Get primary monitor size.
        """
        with mss() as sct:
            mon = sct.monitors[1]  # primary
            return int(mon["width"]), int(mon["height"])

    def _grab_roi(self, roi: RoiPx) -> np.ndarray:
        with mss() as sct:
            monitor = {"left": roi.left, "top": roi.top, "width": roi.width, "height": roi.height}
            img = np.array(sct.grab(monitor))  # BGRA
            # to BGR
            bgr = img[:, :, :3]
            return bgr

    def _best_match(self, roi_bgr: np.ndarray) -> Optional[MatchResult]:
        if not self._tpls:
            return None

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        best: Optional[MatchResult] = None
        for tid, (label, templ_gray) in self._tpls.items():
            # template must be <= roi
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

    def _should_fire(self) -> bool:
        now = time.time()
        return (now - self._last_fire_ts) >= self.cfg.cooldown_sec

    def _fire(self, result: MatchResult) -> None:
        self._last_fire_ts = time.time()
        try:
            self._on_match(result)
        except Exception:
            # swallow callback errors to keep detector alive
            pass

    def _run(self) -> None:
        # cache screen size to reduce overhead; refresh if needed in future
        screen_w, screen_h = self._get_screen_size()

        while not self._stop_evt.is_set():
            try:
                profile = self.get_profile()
                roi_px = rel_to_px(profile.roi_rel, screen_w, screen_h)

                roi_bgr = self._grab_roi(roi_px)
                best = self._best_match(roi_bgr)

                if best and best.score >= self.cfg.threshold and self._should_fire():
                    self._fire(best)

            except Exception:
                # keep loop alive
                pass

            time.sleep(self.cfg.scan_interval_sec)