# 新：相对ROI→像素ROI，DPI/多屏处理的集中地
# src/roi.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoiRel:
    """ROI in relative coordinates (0~1)"""
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class RoiPx:
    """ROI in pixel coordinates"""
    left: int
    top: int
    width: int
    height: int


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rel_to_px(roi: RoiRel, screen_w: int, screen_h: int) -> RoiPx:
    """
    Convert relative ROI (0~1) to pixel ROI on a screen of (screen_w, screen_h).
    Clamp to valid range.
    """
    x = _clamp(roi.x, 0.0, 1.0)
    y = _clamp(roi.y, 0.0, 1.0)
    w = _clamp(roi.w, 0.0, 1.0)
    h = _clamp(roi.h, 0.0, 1.0)

    left = int(round(x * screen_w))
    top = int(round(y * screen_h))
    width = int(round(w * screen_w))
    height = int(round(h * screen_h))

    # ensure >= 1
    width = max(1, width)
    height = max(1, height)

    # clamp to screen bounds
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if left + width > screen_w:
        width = max(1, screen_w - left)
    if top + height > screen_h:
        height = max(1, screen_h - top)

    return RoiPx(left=left, top=top, width=width, height=height)