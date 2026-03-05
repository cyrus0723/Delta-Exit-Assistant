# src/game_wizard.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from mss import mss
from PIL import Image, ImageTk
import tkinter as tk

from config_store import exe_dir
from roi import RoiRel


def _safe_game_id(raw: str) -> str:
    s = (raw or "").strip().lower()
    # 只允许 a-z 0-9 _ -
    s = re.sub(r"[^a-z0-9_-]+", "", s)
    return s


@dataclass
class NewGameInfo:
    game_id: str          # ow2
    display_name: str     # 守望先锋2
    roi_rel: RoiRel       # 框选得到
    win_label: str = "胜利"
    lose_label: str = "败北"
    draw_label: str = "平局"


def ask_new_game_name(parent_title: str = "新建游戏") -> Optional[Tuple[str, str]]:
    """
    返回 (game_id, display_name)
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    game_id = tk.simpledialog.askstring(parent_title, "请输入游戏ID（英文/数字，如：ow2）", parent=root)
    if not game_id:
        return None
    game_id = _safe_game_id(game_id)
    if not game_id:
        return None

    display_name = tk.simpledialog.askstring(parent_title, "请输入显示名称（中文也可以）", parent=root)
    if not display_name:
        display_name = game_id

    try:
        root.destroy()
    except Exception:
        pass

    return game_id, display_name.strip()


class RoiSelector:
    """
    全屏截图上拖拽框选 ROI，返回 RoiRel
    """
    def __init__(self, title: str = "框选检测区域（拖拽选择，回车确认，Esc取消）"):
        self.title = title
        self._done = False
        self._cancel = False
        self._roi_px = None  # (x1,y1,x2,y2)
        self._start = None

    def _grab_screen(self) -> Tuple[np.ndarray, int, int]:
        with mss() as sct:
            mon = sct.monitors[1]
            img = np.array(sct.grab(mon))  # BGRA
            bgr = img[:, :, :3]
            h, w = bgr.shape[:2]
            return bgr, w, h

    def select(self) -> Optional[RoiRel]:
        bgr, sw, sh = self._grab_screen()
        rgb = bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)

        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.attributes("-topmost", True)
        self.root.state("zoomed")  # 最大化

        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._imgtk = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, image=self._imgtk, anchor=tk.NW)

        self._rect_id = None

        def on_down(event):
            self._start = (event.x, event.y)
            if self._rect_id:
                self.canvas.delete(self._rect_id)
                self._rect_id = None

        def on_move(event):
            if not self._start:
                return
            x1, y1 = self._start
            x2, y2 = event.x, event.y
            if self._rect_id:
                self.canvas.coords(self._rect_id, x1, y1, x2, y2)
            else:
                self._rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

        def on_up(event):
            if not self._start:
                return
            x1, y1 = self._start
            x2, y2 = event.x, event.y
            self._start = None
            self._roi_px = (x1, y1, x2, y2)

        def on_key(event):
            if event.keysym == "Escape":
                self._cancel = True
                self.root.destroy()
            elif event.keysym in ("Return", "KP_Enter"):
                self._done = True
                self.root.destroy()

        self.canvas.bind("<ButtonPress-1>", on_down)
        self.canvas.bind("<B1-Motion>", on_move)
        self.canvas.bind("<ButtonRelease-1>", on_up)
        self.root.bind("<Key>", on_key)

        self.root.mainloop()

        if self._cancel or not self._roi_px:
            return None

        x1, y1, x2, y2 = self._roi_px
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)

        w = max(1, right - left)
        h = max(1, bottom - top)

        return RoiRel(
            x=left / sw,
            y=top / sh,
            w=w / sw,
            h=h / sh,
        )


def create_profile_files(info: NewGameInfo) -> Path:
    """
    在 exe_dir()/assets/profiles/<game_id>.json 写入 profile
    并创建 exe_dir()/assets/templates/<game_id> 目录
    返回 profile 路径
    """
    root = exe_dir()
    prof_dir = root / "assets" / "profiles"
    tpl_dir = root / "assets" / "templates" / info.game_id

    prof_dir.mkdir(parents=True, exist_ok=True)
    tpl_dir.mkdir(parents=True, exist_ok=True)

    profile_path = prof_dir / f"{info.game_id}.json"
    data = {
        "id": info.game_id,
        "display_name": info.display_name,
        "roi_rel": {"x": info.roi_rel.x, "y": info.roi_rel.y, "w": info.roi_rel.w, "h": info.roi_rel.h},
        "templates": [
            {"id": f"{info.game_id}_win", "label": info.win_label, "path": f"assets/templates/{info.game_id}/win.png"},
            {"id": f"{info.game_id}_lose", "label": info.lose_label, "path": f"assets/templates/{info.game_id}/lose.png"},
            {"id": f"{info.game_id}_draw", "label": info.draw_label, "path": f"assets/templates/{info.game_id}/draw.png"},
        ],
    }
    profile_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile_path