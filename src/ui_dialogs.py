# src/ui_dialogs.py
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import tkinter as tk
from tkinter import messagebox, simpledialog


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

    def _call(self, fn: Callable[[], Any], timeout: float = 120.0) -> Any:
        job = _TkJob(fn=fn, done=threading.Event(), out={})
        self._q.put(job)
        job.done.wait(timeout=timeout)
        return job.out.get("value", None)

    def info(self, title: str, msg: str) -> None:
        def _f():
            messagebox.showinfo(title, msg, parent=self._root)
            return None

        self._call(_f)

    def confirm(self, title: str, msg: str) -> bool:
        def _f():
            return bool(messagebox.askyesno(title, msg, parent=self._root))

        v = self._call(_f)
        return bool(v)

    def ask_float(self, title: str, prompt: str, initial: float) -> Optional[float]:
        def _f():
            return simpledialog.askfloat(title, prompt, initialvalue=initial, parent=self._root)

        v = self._call(_f)
        return None if v is None else float(v)

    def ask_str(self, title: str, prompt: str, initial: str = "") -> Optional[str]:
        def _f():
            return simpledialog.askstring(title, prompt, initialvalue=initial, parent=self._root)

        v = self._call(_f)
        return None if v is None else str(v)