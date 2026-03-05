from __future__ import annotations

import traceback
from pathlib import Path

import cv2
import numpy as np
from mss import mss

from config_store import app_root
from profiles import GameProfile, TemplateItem
from roi import rel_to_px


def _save_png_compat(path: Path, bgr_img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        raise RuntimeError("cv2.imencode('.png') failed")
    path.write_bytes(buf.tobytes())


def _get_primary_screen_size() -> tuple[int, int]:
    with mss() as sct:
        mon = sct.monitors[1]
        return int(mon["width"]), int(mon["height"])


def grab_profile_roi_bgr(profile: GameProfile) -> np.ndarray:
    w, h = _get_primary_screen_size()
    roi_px = rel_to_px(profile.roi_rel, w, h)

    if roi_px.width <= 2 or roi_px.height <= 2:
        raise ValueError(f"ROI too small: {roi_px}")

    with mss() as sct:
        monitor = {"left": roi_px.left, "top": roi_px.top, "width": roi_px.width, "height": roi_px.height}
        img = np.array(sct.grab(monitor))  # BGRA
        return img[:, :, :3]  # BGR


def capture_to_template(profile: GameProfile, tpl: TemplateItem) -> str:
    """
    Save to: <app_root>/<tpl.path>
    """
    out_abs = (app_root() / tpl.path.replace("\\", "/")).resolve()
    try:
        bgr = grab_profile_roi_bgr(profile)
        _save_png_compat(out_abs, bgr)
        return str(out_abs)
    except Exception:
        print("capture_to_template failed:")
        print(" profile:", profile.id, profile.display_name)
        print(" tpl:", tpl.id, tpl.label, tpl.path)
        print(" out_abs:", str(out_abs))
        print(traceback.format_exc())
        raise