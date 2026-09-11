from __future__ import annotations

from typing import Optional, Tuple

import tkinter as tk

from roi import RoiRel


def _normalize_rect(x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int, int, int]:
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def select_roi_fullscreen(root: tk.Tk, title: str) -> Optional[RoiRel]:
    """Let the user drag a screen-relative ROI on a transparent full-screen overlay."""
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
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
        text="拖拽框选结算标题或结果文字区域，Enter 确认，Esc 取消",
        fill="white",
        font=("Microsoft YaHei", 16),
    )

    start = (0, 0)
    end = (0, 0)
    rect_id: Optional[int] = None
    result: dict[str, Optional[RoiRel]] = {"roi": None}

    def on_down(event: tk.Event) -> None:
        nonlocal start, end, rect_id
        start = (int(event.x), int(event.y))
        end = start
        if rect_id is not None:
            canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(*start, *end, outline="yellow", width=3)

    def on_move(event: tk.Event) -> None:
        nonlocal end
        if rect_id is None:
            return
        end = (int(event.x), int(event.y))
        canvas.coords(rect_id, *start, *end)

    def finish(confirm: bool) -> None:
        if not confirm or rect_id is None:
            win.destroy()
            return
        left, top, right, bottom = _normalize_rect(*start, *end)
        width, height = right - left, bottom - top
        if width < 20 or height < 20:
            win.destroy()
            return
        width = min(width, screen_w - left)
        height = min(height, screen_h - top)
        result["roi"] = RoiRel(
            x=max(0.0, left / screen_w),
            y=max(0.0, top / screen_h),
            w=max(0.001, width / screen_w),
            h=max(0.001, height / screen_h),
        )
        win.destroy()

    canvas.bind("<Button-1>", on_down)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<Escape>", lambda event: finish(False))
    canvas.bind("<Return>", lambda event: finish(True))
    win.bind("<Escape>", lambda event: finish(False))
    win.bind("<Return>", lambda event: finish(True))
    win.focus_force()
    canvas.focus_set()
    win.grab_set()
    win.wait_window()
    return result["roi"]
