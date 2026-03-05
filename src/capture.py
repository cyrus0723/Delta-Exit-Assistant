# src/capture.py（抓取 ROI 保存为模板 + ROI 框选工具）
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from mss import mss

from config_store import exe_dir
from profiles import GameProfile, TemplateItem
from roi import RoiRel, rel_to_px


def _save_png_compat(path: str, bgr_img: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    with open(str(p), "wb") as f:
        f.write(buf.tobytes())


def _get_primary_monitor() -> dict:
    with mss() as sct:
        return dict(sct.monitors[1])


def _grab_fullscreen_bgr() -> Tuple[np.ndarray, int, int]:
    mon = _get_primary_monitor()
    w, h = int(mon["width"]), int(mon["height"])
    with mss() as sct:
        img = np.array(sct.grab({"left": 0, "top": 0, "width": w, "height": h}))  # BGRA
    return img[:, :, :3], w, h  # BGR


def grab_profile_roi_bgr(profile: GameProfile) -> np.ndarray:
    bgr, sw, sh = _grab_fullscreen_bgr()
    roi_px = rel_to_px(profile.roi_rel, sw, sh)
    x1, y1 = max(0, roi_px.left), max(0, roi_px.top)
    x2 = min(sw, roi_px.left + roi_px.width)
    y2 = min(sh, roi_px.top + roi_px.height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI is empty; please set ROI first.")
    return bgr[y1:y2, x1:x2].copy()


def capture_to_template(profile: GameProfile, tpl: TemplateItem) -> str:
    """
    Capture current profile ROI and save to writable runtime path: exe_dir()/tpl.path
    Return absolute saved path.
    """
    bgr = grab_profile_roi_bgr(profile)
    out_path = str(exe_dir() / tpl.path.replace("\\", "/"))
    _save_png_compat(out_path, bgr)
    return out_path


def capture_to_game_folder_auto(profile: GameProfile) -> str:
    """
    For "custom game" workflow:
    Capture ROI and save to assets/templates/<game>/cap_YYYYmmdd_HHMMSS.png
    Return absolute saved path.
    """
    bgr = grab_profile_roi_bgr(profile)
    ts = time.strftime("%Y%m%d_%H%M%S")
    rel = f"assets/templates/{profile.id}/cap_{ts}.png"
    out_path = str(exe_dir() / rel)
    _save_png_compat(out_path, bgr)
    return out_path


# -------------------------
# ROI selector (interactive)
# -------------------------

def select_roi_rel_interactive(window_title: str = "选择ROI：拖拽框选，回车确认，ESC取消") -> Optional[RoiRel]:
    """
    Show a fullscreen screenshot in an OpenCV window, allow mouse drag to select ROI.
    Return RoiRel (x,y,w,h) relative to primary screen size.
    - ENTER: confirm
    - ESC: cancel
    """
    img_bgr, sw, sh = _grab_fullscreen_bgr()
    disp = img_bgr.copy()

    selecting = {"down": False}
    p0 = {"x": 0, "y": 0}
    rect = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

    def _clamp(x: int, y: int) -> Tuple[int, int]:
        return max(0, min(sw - 1, x)), max(0, min(sh - 1, y))

    def on_mouse(event, x, y, flags, param):
        nonlocal disp
        x, y = _clamp(int(x), int(y))

        if event == cv2.EVENT_LBUTTONDOWN:
            selecting["down"] = True
            p0["x"], p0["y"] = x, y
            rect["x1"], rect["y1"], rect["x2"], rect["y2"] = x, y, x, y

        elif event == cv2.EVENT_MOUSEMOVE and selecting["down"]:
            rect["x2"], rect["y2"] = x, y
            disp = img_bgr.copy()
            x1, y1 = min(rect["x1"], rect["x2"]), min(rect["y1"], rect["y2"])
            x2, y2 = max(rect["x1"], rect["x2"]), max(rect["y1"], rect["y2"])
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            selecting["down"] = False
            rect["x2"], rect["y2"] = x, y
            disp = img_bgr.copy()
            x1, y1 = min(rect["x1"], rect["x2"]), min(rect["y1"], rect["y2"])
            x2, y2 = max(rect["x1"], rect["x2"]), max(rect["y1"], rect["y2"])
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_title, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(window_title, on_mouse)

    # Try to fit screen
    try:
        cv2.resizeWindow(window_title, min(1600, sw), min(900, sh))
    except Exception:
        pass

    while True:
        cv2.imshow(window_title, disp)
        key = cv2.waitKey(20) & 0xFF

        # ESC
        if key == 27:
            cv2.destroyWindow(window_title)
            return None

        # ENTER
        if key in (10, 13):
            x1, y1 = min(rect["x1"], rect["x2"]), min(rect["y1"], rect["y2"])
            x2, y2 = max(rect["x1"], rect["x2"]), max(rect["y1"], rect["y2"])
            w = x2 - x1
            h = y2 - y1
            cv2.destroyWindow(window_title)
            if w < 5 or h < 5:
                return None
            return RoiRel(
                x=float(x1) / float(sw),
                y=float(y1) / float(sh),
                w=float(w) / float(sw),
                h=float(h) / float(sh),
            )