# src/roi_selector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import tkinter as tk

from roi import RoiRel


@dataclass
class _Selection:
    x0: int = 0
    y0: int = 0
    x1: int = 0
    y1: int = 0


def _normalize_rect(x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int, int, int]:
    left = min(x0, x1)
    top = min(y0, y1)
    right = max(x0, x1)
    bottom = max(y0, y1)
    return left, top, right, bottom


def select_roi_fullscreen(root: tk.Tk, title: str = "选择检测区域 ROI") -> Optional[RoiRel]:
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    win = tk.Toplevel(root)
    win.title(title)
    win.attributes("-topmost", True)
    win.geometry(f"{screen_w}x{screen_h}+0+0")
    win.overrideredirect(True)

    try:
        win.attributes("-alpha", 0.30)
    except Exception:
        pass

    canvas = tk.Canvas(win, width=screen_w, height=screen_h, highlightthickness=0, cursor="cross")
    canvas.pack(fill="both", expand=True)

    canvas.create_text(
        screen_w // 2,
        30,
        text="拖拽框选 ROI（Enter 确认 / ESC 取消）",
        fill="white",
        font=("Microsoft YaHei", 16),
    )

    sel = _Selection()
    rect_id = None
    result: dict = {"roi": None}

    def on_down(ev):
        nonlocal rect_id
        sel.x0, sel.y0 = int(ev.x), int(ev.y)
        sel.x1, sel.y1 = sel.x0, sel.y0
        if rect_id is not None:
            canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(sel.x0, sel.y0, sel.x1, sel.y1, outline="yellow", width=3)

    def on_move(ev):
        nonlocal rect_id
        if rect_id is None:
            return
        sel.x1, sel.y1 = int(ev.x), int(ev.y)
        canvas.coords(rect_id, sel.x0, sel.y0, sel.x1, sel.y1)

    def finalize(confirm: bool):
        if not confirm or rect_id is None:
            result["roi"] = None
            win.destroy()
            return

        left, top, right, bottom = _normalize_rect(sel.x0, sel.y0, sel.x1, sel.y1)
        width = max(1, right - left)
        height = max(1, bottom - top)

        if width < 20 or height < 20:
            result["roi"] = None
            win.destroy()
            return

        roi = RoiRel(
            x=left / float(screen_w),
            y=top / float(screen_h),
            w=width / float(screen_w),
            h=height / float(screen_h),
        )
        roi = RoiRel(
            x=max(0.0, min(1.0, roi.x)),
            y=max(0.0, min(1.0, roi.y)),
            w=max(0.001, min(1.0, roi.w)),
            h=max(0.001, min(1.0, roi.h)),
        )
        roi = RoiRel(
            x=min(roi.x, 1.0 - roi.w),
            y=min(roi.y, 1.0 - roi.h),
            w=roi.w,
            h=roi.h,
        )
        result["roi"] = roi
        win.destroy()

    def on_escape(_=None):
        finalize(False)

    def on_enter(_=None):
        finalize(True)

    canvas.bind("<Button-1>", on_down)
    canvas.bind("<B1-Motion>", on_move)
    win.bind("<Escape>", on_escape)
    win.bind("<Return>", on_enter)

    win.focus_force()
    canvas.focus_set()
    win.grab_set()
    win.wait_window()
    return result["roi"]