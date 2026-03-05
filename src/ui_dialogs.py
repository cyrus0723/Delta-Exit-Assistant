from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    All tkinter dialogs MUST run on tkinter thread.
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

    def run_in_tk(self, fn: Callable[[tk.Tk], Any], timeout: float = 120.0) -> Any:
        def _f():
            return fn(self._root)

        return self._call(_f, timeout=timeout)

    def _call(self, fn: Callable[[], Any], timeout: float = 60.0) -> Any:
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

    # ✅ 新增：列表选择对话框（解决 pystray 子菜单缓存）
    def choose_from_list(self, title: str, prompt: str, options: List[Tuple[str, str]]) -> Optional[str]:
        """
        options: [(value, display_text), ...]
        returns selected value or None
        """
        def _f():
            win = tk.Toplevel(self._root)
            win.title(title)
            win.attributes("-topmost", True)
            win.geometry("520x360")
            win.resizable(False, False)

            lbl = tk.Label(win, text=prompt, anchor="w", justify="left")
            lbl.pack(fill="x", padx=12, pady=10)

            frame = tk.Frame(win)
            frame.pack(fill="both", expand=True, padx=12)

            lb = tk.Listbox(frame, height=12)
            sb = tk.Scrollbar(frame, orient="vertical", command=lb.yview)
            lb.configure(yscrollcommand=sb.set)

            lb.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")

            for _, text in options:
                lb.insert(tk.END, text)

            out = {"value": None}

            def ok():
                sel = lb.curselection()
                if not sel:
                    return
                idx = int(sel[0])
                out["value"] = options[idx][0]
                win.destroy()

            def cancel():
                out["value"] = None
                win.destroy()

            btns = tk.Frame(win)
            btns.pack(fill="x", padx=12, pady=10)
            tk.Button(btns, text="确定", width=12, command=ok).pack(side="right", padx=6)
            tk.Button(btns, text="取消", width=12, command=cancel).pack(side="right", padx=6)

            lb.bind("<Double-Button-1>", lambda e: ok())
            win.protocol("WM_DELETE_WINDOW", cancel)

            win.grab_set()
            win.focus_force()
            win.wait_window()
            return out["value"]

        return self._call(_f, timeout=120.0)