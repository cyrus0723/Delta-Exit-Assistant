# src/detector.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from capture import grab_profile_roi_bgr
from profiles import GameProfile, TemplateItem, resolve_path


@dataclass
class MatchResult:
    label: str
    template_id: str
    score: float


@dataclass
class DetectorConfig:
    threshold: float = 0.82
    hysteresis: float = 0.12
    scan_interval_sec: float = 0.20
    cooldown_sec: float = 3.0


@dataclass
class _LoadedTemplate:
    template_id: str
    label: str
    path: str
    gray: np.ndarray


class Detector:
    def __init__(self, cfg: DetectorConfig, profile: GameProfile, on_match: Callable[[MatchResult], None]):
        self.cfg = cfg
        self._profile = profile
        self._on_match = on_match

        self._running = False
        self._th: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._templates: List[_LoadedTemplate] = []
        self._reload_templates_locked()

        # anti-spam
        self._armed = True
        self._last_fire_ts = 0.0

    def is_running(self) -> bool:
        return self._running

    def set_profile(self, profile: GameProfile) -> None:
        with self._lock:
            self._profile = profile
            self._reload_templates_locked()
            self._armed = True
            self._last_fire_ts = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._running = False

    def reload_templates(self) -> None:
        with self._lock:
            self._reload_templates_locked()

    def _reload_templates_locked(self) -> None:
        self._templates = []
        for t in self._profile.templates:
            p = resolve_path(t.path)
            try:
                img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self._templates.append(_LoadedTemplate(template_id=t.id, label=t.label, path=p, gray=gray))
            except FileNotFoundError:
                # missing template is OK (e.g. draw.png not captured yet)
                continue
            except Exception:
                continue

    def _best_match(self, roi_bgr: np.ndarray) -> Optional[MatchResult]:
        if not self._templates:
            return None

        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        best: Optional[Tuple[str, str, float]] = None  # (id, label, score)
        for t in self._templates:
            if roi_gray.shape[0] < t.gray.shape[0] or roi_gray.shape[1] < t.gray.shape[1]:
                continue
            res = cv2.matchTemplate(roi_gray, t.gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if best is None or max_val > best[2]:
                best = (t.template_id, t.label, float(max_val))

        if best is None:
            return None
        return MatchResult(label=best[1], template_id=best[0], score=best[2])

    def _loop(self) -> None:
        while self._running:
            time.sleep(max(0.05, float(self.cfg.scan_interval_sec)))

            with self._lock:
                profile = self._profile
                templates_count = len(self._templates)

            if templates_count == 0:
                continue

            try:
                roi = grab_profile_roi_bgr(profile)
            except Exception:
                continue

            r = self._best_match(roi)
            if r is None:
                continue

            now = time.time()

            # cooldown gate
            if now - self._last_fire_ts < float(self.cfg.cooldown_sec):
                continue

            if self._armed:
                if r.score >= float(self.cfg.threshold):
                    self._last_fire_ts = now
                    self._armed = False
                    try:
                        self._on_match(r)
                    except Exception:
                        pass
            else:
                # re-arm when score drops below threshold - hysteresis
                if r.score < float(self.cfg.threshold) - float(self.cfg.hysteresis):
                    self._armed = True