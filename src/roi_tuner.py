# src/roi_tuner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from roi import RoiRel


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class RoiTuner:
    """
    Holds current ROI override for a profile.
    Persist is handled by caller (ui_tray).
    """
    roi: RoiRel
    step: float = 0.005

    def set_step(self, step: float) -> None:
        self.step = max(0.0005, float(step))

    def move(self, dx: float, dy: float) -> None:
        x = _clamp(self.roi.x + dx)
        y = _clamp(self.roi.y + dy)
        # keep inside bounds
        x = _clamp(x, 0.0, 1.0 - self.roi.w)
        y = _clamp(y, 0.0, 1.0 - self.roi.h)
        self.roi = RoiRel(x=x, y=y, w=self.roi.w, h=self.roi.h)

    def resize(self, dw: float, dh: float) -> None:
        w = _clamp(self.roi.w + dw, 0.01, 1.0)
        h = _clamp(self.roi.h + dh, 0.01, 1.0)
        # keep top-left fixed, clamp width/height so ROI stays on screen
        w = min(w, 1.0 - self.roi.x)
        h = min(h, 1.0 - self.roi.y)
        self.roi = RoiRel(x=self.roi.x, y=self.roi.y, w=w, h=h)

    # convenient actions (by step)
    def left(self) -> None: self.move(-self.step, 0.0)
    def right(self) -> None: self.move(+self.step, 0.0)
    def up(self) -> None: self.move(0.0, -self.step)
    def down(self) -> None: self.move(0.0, +self.step)

    def wider(self) -> None: self.resize(+self.step, 0.0)
    def narrower(self) -> None: self.resize(-self.step, 0.0)
    def taller(self) -> None: self.resize(0.0, +self.step)
    def shorter(self) -> None: self.resize(0.0, -self.step)


def load_roi_override(cfg: Dict, profile_id: str) -> Optional[RoiRel]:
    ro = (cfg.get("roi_overrides") or {}).get(profile_id)
    if not isinstance(ro, dict):
        return None
    try:
        return RoiRel(
            x=float(ro["x"]),
            y=float(ro["y"]),
            w=float(ro["w"]),
            h=float(ro["h"]),
        )
    except Exception:
        return None


def save_roi_override(cfg: Dict, profile_id: str, roi: RoiRel) -> None:
    if "roi_overrides" not in cfg or not isinstance(cfg["roi_overrides"], dict):
        cfg["roi_overrides"] = {}
    cfg["roi_overrides"][profile_id] = {"x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h}


def clear_roi_override(cfg: Dict, profile_id: str) -> None:
    ro = cfg.get("roi_overrides")
    if isinstance(ro, dict) and profile_id in ro:
        del ro[profile_id]