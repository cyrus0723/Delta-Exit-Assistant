# src/ui_dialogs.py
from __future__ import annotations

import queue
import threading
import ctypes
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


@dataclass
class _TkJob:
    fn: Callable[[], Any]
    done: threading.Event
    out: Dict[str, Any]


class TkDialogService:
    """
    Run tkinter mainloop in a dedicated thread.
    pystray menu callbacks are not guaranteed to be on main thread,
    so all tkinter dialogs MUST be executed on tkinter thread.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[_TkJob]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._ready = threading.Event()
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        self._root = tk.Tk()
        self._root.withdraw()
        self._root.attributes("-topmost", True)
        self._ready.set()

        def poll():
            try:
                while True:
                    job = self._q.get_nowait()
                    try:
                        job.out["value"] = job.fn()
                    except Exception as e:
                        job.out["error"] = e
                        job.out["value"] = None
                    finally:
                        job.done.set()
            except queue.Empty:
                pass

            self._root.after(50, poll)

        self._root.after(50, poll)
        self._root.mainloop()

    def _call(self, fn: Callable[[], Any], timeout: float = 60.0) -> Any:
        if threading.current_thread() is self._thread:
            return fn()
        job = _TkJob(fn=fn, done=threading.Event(), out={})
        self._q.put(job)
        job.done.wait(timeout=timeout)
        return job.out.get("value", None)

    def run_in_tk(self, fn: Callable[[tk.Tk], Any], timeout: float = 120.0) -> Any:
        return self._call(lambda: fn(self._root), timeout=timeout)

    def show_alert(self, title: str, msg: str) -> None:
        """Queue a nonblocking reminder on the Tk thread."""
        self._q.put(_TkJob(fn=lambda: self._show_alert_window(title, msg),
                           done=threading.Event(), out={}))

    def _show_alert_window(self, title: str, msg: str) -> None:
        class Rect(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        user32 = ctypes.windll.user32
        user32.GetForegroundWindow.restype = ctypes.c_void_p
        user32.GetWindowRect.argtypes = (ctypes.c_void_p, ctypes.POINTER(Rect))
        user32.SetWindowPos.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint)
        foreground = user32.GetForegroundWindow()
        rect = Rect()
        has_foreground = bool(foreground and user32.GetWindowRect(foreground, ctypes.byref(rect)))
        win = tk.Toplevel(self._root)
        win.title("下机提醒")
        win.attributes("-topmost", True)
        win.resizable(False, False)
        win.configure(bg="#f3f5f9")
        win.geometry("540x390")

        header = tk.Frame(win, bg="#f3f5f9")
        header.pack(fill="x", padx=28, pady=(22, 14))
        tk.Label(header, text="◷", font=("Segoe UI", 24), bg="#f3f5f9",
                 fg="#0067c0").pack(side="left", padx=(0, 12))
        tk.Label(header, text="下机提醒", font=("Microsoft YaHei UI", 12, "bold"),
                 bg="#f3f5f9", fg="#0067c0").pack(side="left")
        canvas = tk.Canvas(win, bg="#f3f5f9", highlightthickness=0, height=150)
        canvas.pack(fill="both", expand=True, padx=24)
        card = tk.Frame(canvas, bg="white")
        item = canvas.create_window(20, 20, window=card, anchor="nw")
        def resize_card(event):
            w, h = event.width, event.height
            canvas.delete("card")
            r = 28
            canvas.create_polygon(r, 1, w-r, 1, w-1, 1, w-1, r,
                                  w-1, h-r, w-1, h-1, w-r, h-1,
                                  r, h-1, 1, h-1, 1, h-r, 1, r, 1, 1,
                                  smooth=True, splinesteps=24, fill="white",
                                  outline="#e1e6ed", tags="card")
            canvas.tag_lower("card")
            canvas.itemconfigure(item, width=max(1, w-40), height=max(1, h-40))
        canvas.bind("<Configure>", resize_card)
        tk.Label(card, text=title, font=("Microsoft YaHei UI", 20, "bold"),
                 bg="white", fg="#20252d", wraplength=440, anchor="w",
                 justify="left").pack(fill="x", pady=(0, 12))
        body_frame = tk.Frame(card, bg="white")
        body_frame.pack(fill="both", expand=True)
        scrollbar = tk.Scrollbar(body_frame, width=10)
        body = tk.Text(body_frame, font=("Microsoft YaHei UI", 13), bg="white",
                       fg="#697586", wrap="word", relief="flat", height=3,
                       highlightthickness=0, padx=0, pady=0,
                       yscrollcommand=scrollbar.set)
        scrollbar.configure(command=body.yview)
        scrollbar.pack(side="right", fill="y")
        body.pack(side="left", fill="both", expand=True)
        body.insert("1.0", msg)
        body.configure(state="disabled")
        button = tk.Button(win, text="我已了解", command=win.destroy,
                           font=("Microsoft YaHei UI", 12, "bold"),
                           bg="#0067c0", fg="white", activebackground="#005aab",
                           activeforeground="white", relief="flat", bd=0,
                           cursor="hand2", padx=28, pady=10)
        button.pack(pady=(16, 22), ipadx=18)
        win.bind("<Escape>", lambda _event: win.destroy())
        win.protocol("WM_DELETE_WINDOW", win.destroy)
        win.update_idletasks()
        if has_foreground:
            x = (rect.left + rect.right - win.winfo_width()) // 2
            y = (rect.top + rect.bottom - win.winfo_height()) // 2
        else:
            x = (win.winfo_screenwidth() - win.winfo_width()) // 2
            y = (win.winfo_screenheight() - win.winfo_height()) // 2
        class MonitorInfo(ctypes.Structure):
            _fields_ = [("size", ctypes.c_ulong), ("monitor", Rect),
                        ("work", Rect), ("flags", ctypes.c_ulong)]

        user32.MonitorFromWindow.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        user32.MonitorFromWindow.restype = ctypes.c_void_p
        user32.GetMonitorInfoW.argtypes = (ctypes.c_void_p, ctypes.POINTER(MonitorInfo))
        monitor = user32.MonitorFromWindow(foreground, 2)
        info = MonitorInfo()
        info.size = ctypes.sizeof(info)
        if user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            x = max(info.work.left, min(x, info.work.right - win.winfo_width()))
            y = max(info.work.top, min(y, info.work.bottom - win.winfo_height()))
        win.geometry(f"{x:+d}{y:+d}")
        win.lift()
        user32.SetWindowPos(win.winfo_id(), ctypes.c_void_p(-1), x, y, 0, 0, 0x0001 | 0x0040)
        win.focus_force()
        button.focus_set()

    def info(self, title: str, msg: str) -> None:
        def _f():
            messagebox.showinfo(title, msg, parent=self._root)
            return None

        self._call(_f)

    def confirm(self, title: str, msg: str) -> bool:
        def _f():
            return messagebox.askyesno(title, msg, parent=self._root)

        return bool(self._call(_f))

    def ask_float(self, title: str, prompt: str, initial: float) -> Optional[float]:
        def _f():
            return simpledialog.askfloat(title, prompt, initialvalue=initial, parent=self._root)

        v = self._call(_f)
        return None if v is None else float(v)

    def ask_str(self, title: str, prompt: str, initial: str) -> Optional[str]:
        def _f():
            return simpledialog.askstring(title, prompt, initialvalue=initial, parent=self._root)

        v = self._call(_f)
        return None if v is None else str(v)

    def ask_open_file(self, title: str, filetypes: list[tuple[str, str]]) -> Optional[str]:
        def _f():
            return filedialog.askopenfilename(title=title, filetypes=filetypes, parent=self._root)

        v = self._call(_f)
        return None if not v else str(v)
