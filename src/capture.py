# src/capture.py
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from mss import mss

from profiles import GameProfile, TemplateItem
from roi import rel_to_px
from paths import app_dir


def _save_png_compat(path: str, bgr_img: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    with open(str(p), "wb") as f:
        f.write(buf.tobytes())


def _get_primary_screen_size() -> tuple[int, int]:
    with mss() as sct:
        mon = sct.monitors[1]
        return int(mon["width"]), int(mon["height"])


def grab_profile_roi_bgr(profile: GameProfile) -> np.ndarray:
    w, h = _get_primary_screen_size()
    roi_px = rel_to_px(profile.roi_rel, w, h)
    with mss() as sct:
        monitor = {"left": roi_px.left, "top": roi_px.top, "width": roi_px.width, "height": roi_px.height}
        img = np.array(sct.grab(monitor))  # BGRA
        return img[:, :, :3]  # BGR


def capture_to_template(profile: GameProfile, tpl: TemplateItem) -> str:
    bgr = grab_profile_roi_bgr(profile)

    # tpl.path is like "assets/templates/ow2/win.png"
    out_path = str((app_dir() / tpl.path.replace("\\", "/")).resolve())
    _save_png_compat(out_path, bgr)
    return out_path