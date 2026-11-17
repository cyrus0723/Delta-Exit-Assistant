"""Small control window sharing the tray application's existing state."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import math

from PIL import Image, ImageDraw, ImageTk

from alert_timer import parse_target_time, parse_countdown_hours
from notify import VALID_NOTIFY_MODES
import windows_startup


class RoundedButton(tk.Canvas):
    """Keyboard-accessible canvas button with the same config surface used by status."""

    def __init__(self, parent, text, command, primary=False, width=120):
        super().__init__(parent, width=width, height=40, highlightthickness=0,
                         bg="white", cursor="hand2", takefocus=True)
        self._text = text
        self._command = command
        self._fill = "#0067c0" if primary else "#f0f4fa"
        self._foreground = "white" if primary else "#275581"
        self._hover = False
        self.bind("<Configure>", self._draw)
        self.bind("<Enter>", lambda _e: self._set_hover(True))
        self.bind("<Leave>", lambda _e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", lambda e: self.invoke() if 0 <= e.x < self.winfo_width() and 0 <= e.y < self.winfo_height() else None)
        self.bind("<Return>", lambda _e: self.invoke())
        self.bind("<space>", lambda _e: self.invoke())
        self.bind("<FocusIn>", self._draw)
        self.bind("<FocusOut>", self._draw)

    def _set_hover(self, value):
        self._hover = value
        self._draw()

    def _draw(self, _event=None):
        self.delete("all")
        w, h, r = self.winfo_width(), self.winfo_height(), 12
        self.create_polygon(r, 1, w-r, 1, w-1, 1, w-1, r, w-1, h-r,
                            w-1, h-1, w-r, h-1, r, h-1, 1, h-1,
                            1, h-r, 1, r, 1, 1, smooth=True, splinesteps=24,
                            fill=self._fill, outline="#8bb7eb" if self._hover or self.focus_get() == self else self._fill)
        self.create_text(w/2, h/2, text=self._text, fill=self._foreground,
                         font=("Microsoft YaHei UI", 10, "bold"))

    def config(self, **kwargs):
        self._text = kwargs.pop("text", self._text)
        self._fill = kwargs.pop("bg", self._fill)
        kwargs.pop("activebackground", None)
        if kwargs:
            super().configure(**kwargs)
        self._draw()

    def cget(self, key):
        return self._text if key == "text" else super().cget(key)

    def invoke(self):
        return self._command()


class MainWindow:
    def __init__(self, dialogs, app):
        self.app = app
        self.dialogs = dialogs
        self.window = None
        self.dialogs.run_in_tk(self._create)

    def _create(self, root):
        win = tk.Toplevel(root)
        self.window = win
        win.title("三角洲下机助手")
        win.geometry(f"1060x{min(780, win.winfo_screenheight() - 80)}")
        win.minsize(940, 600)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        win.configure(bg="#f3f5f9")
        self._setup_styles(win)
        self._pages = {}
        self._nav_buttons = {}
        self._nav_icons = {}

        sidebar = tk.Frame(win, bg="#edf0f5", width=204)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="Δ", font=("Segoe UI", 30, "bold"),
                 bg="#edf0f5", fg="#0067c0").pack(anchor="w", padx=26, pady=(25, 0))
        tk.Label(sidebar, text="三角洲下机助手", font=("Microsoft YaHei UI", 13, "bold"),
                 bg="#edf0f5", fg="#20252d").pack(anchor="w", padx=24, pady=(2, 0))
        tk.Label(sidebar, text="DELTA EXIT ASSISTANT", font=("Segoe UI", 8),
                 bg="#edf0f5", fg="#78828f").pack(anchor="w", padx=25, pady=(5, 30))
        for key, title in (("home", "开始"), ("reminder", "提醒设置"),
                           ("templates", "模板与识别"), ("advanced", "高级设置")):
            self._nav_icons[key] = self._navigation_icon(sidebar, key)
            button = tk.Button(sidebar, text=f"  {title}", anchor="w",
                               image=self._nav_icons[key], compound="left",
                               font=("Microsoft YaHei UI", 11), relief="flat", bd=0,
                               bg="#edf0f5", fg="#4f5966", activebackground="#dfe7f1",
                               activeforeground="#005aab", padx=14, pady=12, cursor="hand2",
                               command=lambda page=key: self._show_page(page))
            button.pack(fill="x", padx=12, pady=3)
            self._nav_buttons[key] = button
        tk.Button(sidebar, text="退出程序", command=lambda: self.app._action_quit(self.app._icon, None),
                  bg="#edf0f5", fg="#637080", activebackground="#e1e6ee", relief="flat", bd=0,
                  font=("Microsoft YaHei UI", 10), cursor="hand2", pady=12).pack(side="bottom", fill="x", padx=16, pady=(0, 18))
        tk.Label(sidebar, text="关闭窗口后继续在托盘运行", bg="#edf0f5", fg="#78828f",
                 font=("Microsoft YaHei UI", 8)).pack(side="bottom", pady=10)

        main = ttk.Frame(win, style="Shell.TFrame", padding=(28, 18, 24, 12))
        main.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(main, style="Shell.TFrame")
        header.pack(fill="x", pady=(0, 12))
        self.page_title = ttk.Label(header, style="Title.TLabel")
        self.page_title.pack(anchor="w")
        self.page_description = ttk.Label(header, style="Subtitle.TLabel")
        self.page_description.pack(anchor="w", pady=(7, 0))
        self._page_host = ttk.Frame(main, style="Shell.TFrame")
        self._page_host.pack(fill="both", expand=True)
        for key in self._nav_buttons:
            content, frame = self._scroll_content(self._page_host)
            self._pages[key] = (content, frame)
        home = self._pages["home"][0]
        reminder = self._pages["reminder"][0]
        templates = self._pages["templates"][0]
        advanced = self._pages["advanced"][0]

        hero = self._card(home, "检测状态")
        status_row = ttk.Frame(hero)
        status_row.pack(fill="x")
        status_text = ttk.Frame(status_row)
        status_text.pack(side="left", fill="x", expand=True)
        self.status_label = ttk.Label(status_text, font=("Microsoft YaHei UI", 20, "bold"),
                                      background="white", foreground="#20252d")
        self.status_label.pack(anchor="w", pady=(0, 5))
        self.status_description = ttk.Label(status_text, style="Muted.TLabel", wraplength=380)
        self.status_description.pack(anchor="w")
        self.toggle = RoundedButton(status_row, "开始检测", self._toggle_detection,
                                    primary=True, width=142)
        self.toggle.pack(side="right", padx=(12, 0))
        self.auto_start = tk.BooleanVar(value=self.app._cfg.get("auto_start_detection") is True)
        ttk.Checkbutton(hero, text="打开助手时自动开始检测", variable=self.auto_start,
                        style="Startup.TCheckbutton", command=self._set_auto_start).pack(anchor="w", pady=(8, 0))
        self.start_with_windows = tk.BooleanVar(value=False)
        try:
            self.start_with_windows.set(windows_startup.is_enabled())
        except OSError:
            pass
        ttk.Checkbutton(hero, text="开机自启动", variable=self.start_with_windows,
                        style="Startup.TCheckbutton", command=self._set_windows_startup).pack(anchor="w", pady=(4, 0))
        row = ttk.Frame(hero)
        row.pack(fill="x", pady=(14, 0))
        ttk.Label(row, text="当前游戏").pack(side="left", padx=(0, 12))
        self.game = ttk.Combobox(row, state="readonly", width=27)
        self.game.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self.game.bind("<<ComboboxSelected>>", self._choose_game)
        RoundedButton(row, "新建游戏", lambda: self._act(self.app._action_new_game), width=108).pack(side="right")
        row = ttk.Frame(hero)
        row.pack(fill="x", pady=(8, 0))
        ttk.Label(row, text="预计单局").pack(side="left", padx=(0, 12))
        self.duration = self._time_stepper(row, "duration")
        ttk.Label(row, text="分钟").pack(side="left")
        RoundedButton(row, "应用时长", self._save_duration, width=108).pack(side="right")

        timing = self._card(home, "提醒时机")
        self.timer = tk.StringVar()
        self._timing_rows = {}
        self.timer.trace_add("write", lambda *_args: self._highlight_timing())
        for label, value in (("每局结束时", "always"), ("玩到指定时间", "clock"), ("再玩", "countdown")):
            row = ttk.Frame(timing, padding=(12, 5), style=f"{value}.TFrame")
            row.pack(fill="x", pady=3)
            self._timing_rows[value] = row
            ttk.Radiobutton(row, text=label, variable=self.timer, value=value,
                            style=f"{value}.TRadiobutton",
                            command=self._set_timer, width=16).pack(side="left")
            if value == "clock":
                self.target = self._time_stepper(row, "clock")
                RoundedButton(row, "应用时间", self._save_timer, width=106).pack(side="right")
            elif value == "countdown":
                self.countdown = self._time_stepper(row, "countdown")
                ttk.Label(row, text="小时").pack(side="left")
                RoundedButton(row, "应用时长", self._save_countdown, width=106).pack(side="right")
            self._bind_timing_row(row, value)
        self.window_label = ttk.Label(timing, style="Plan.TLabel", padding=(12, 9), wraplength=550)
        self.window_label.pack(fill="x", pady=(8, 2))

        modes = self._card(reminder, "提醒方式")
        self.mode = tk.StringVar()
        mode_choices = ttk.Frame(modes)
        mode_choices.pack(fill="x")
        mode_choices.columnconfigure((0, 1), weight=1)
        for index, (label, value) in enumerate((("应用弹窗 + 提示音", "popup"), ("Windows 通知 + 提示音", "native"),
                                              ("仅 Windows 通知", "native_silent"), ("只响铃", "sound"))):
            ttk.Radiobutton(mode_choices, text=label, value=value, variable=self.mode,
                            command=self._set_mode).grid(row=index // 2, column=index % 2, sticky="w", pady=4)
        text_card = self._card(reminder, "提醒文案", help_command=self._show_text_help)
        self._text_editor = ttk.Frame(text_card)
        self._text_editor.pack(fill="x")
        ttk.Label(self._text_editor, text="标题").pack(anchor="w", pady=(0, 6))
        self.title_text = ttk.Entry(self._text_editor)
        self.title_text.pack(fill="x", pady=(0, 8))
        ttk.Label(self._text_editor, text="正文").pack(anchor="w", pady=(0, 6))
        self.body_text = ttk.Entry(self._text_editor)
        self.body_text.pack(fill="x", pady=(0, 8))
        ttk.Button(self._text_editor, text="保存文案", style="Clear.TButton", command=self._save_text).pack(anchor="w")
        self._text_help = None
        self._text_help_button = text_card.help_button
        win.bind("<Button-1>", self._dismiss_text_help, add="+")
        win.bind("<Escape>", lambda _event: self._hide_text_help(), add="+")
        win.bind("<FocusOut>", self._text_help_focus_out, add="+")
        win.bind("<Configure>", self._text_help_window_changed, add="+")
        sound_card = self._card(reminder, "提示音")
        self.sound = ttk.Label(sound_card, style="Muted.TLabel", wraplength=580)
        self.sound.pack(anchor="w", pady=(0, 8))
        actions = ttk.Frame(sound_card)
        actions.pack(fill="x")
        for label, action in (("选择提示音", self.app._action_select_sound),
                              ("试听提示音", self.app._action_test_sound),
                              ("恢复默认音", self.app._action_reset_sound),
                              ("测试提醒", self.app._action_test_notify)):
            ttk.Button(actions, text=label, style="Clear.TButton",
                       command=lambda a=action: self._act(a)).pack(side="left", padx=(0, 8))

        template_card = self._card(templates, "结算画面模板")
        self.template_state = ttk.Label(template_card, style="Muted.TLabel", wraplength=580)
        self.template_state.pack(anchor="w", pady=(0, 16))
        row = ttk.Frame(template_card)
        row.pack(fill="x")
        self.capture_choice = ttk.Combobox(row, state="readonly", width=18)
        self.capture_choice.pack(side="left", padx=(0, 10))
        ttk.Button(row, text="抓取模板", style="Clear.TButton", command=self._capture).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="重载模板", style="Clear.TButton", command=lambda: self._act(self.app._action_reload_templates)).pack(side="left")
        roi_card = self._card(templates, "检测区域")
        row = ttk.Frame(roi_card)
        row.pack(fill="x", pady=(0, 18))
        ttk.Button(row, text="预览检测区域", style="Clear.TButton", command=lambda: self._act(self.app._action_preview_roi)).pack(side="left", padx=(0, 8))
        ttk.Button(row, text="恢复默认区域", style="Clear.TButton", command=lambda: self._act(self.app._action_roi_reset)).pack(side="left")
        ttk.Label(roi_card, text="检测区域微调", style="Muted.TLabel").pack(anchor="w", pady=(0, 8))
        row = ttk.Frame(roi_card)
        row.pack(fill="x", pady=(0, 8))
        for label, direction in (("上移", "up"), ("下移", "down"), ("左移", "left"), ("右移", "right")):
            ttk.Button(row, text=label, style="Clear.TButton", command=lambda d=direction: self._act(self.app._action_roi_move(d))).pack(side="left", padx=(0, 8))
        row = ttk.Frame(roi_card)
        row.pack(fill="x")
        for label, which in (("加宽", "wider"), ("变窄", "narrower"), ("加高", "taller"), ("变矮", "shorter")):
            ttk.Button(row, text=label, style="Clear.TButton", command=lambda w=which: self._act(self.app._action_roi_resize(w))).pack(side="left", padx=(0, 8))
        for label, description, action in (("匹配阈值", "调整结算画面的匹配灵敏度。", self.app._action_set_threshold),
                                          ("回落差值", "控制何时可以再次触发提醒。", self.app._action_set_hysteresis),
                                          ("冷却时间", "设置两次提醒之间的最短间隔。", self.app._action_set_cooldown),
                                          ("扫描间隔", "调整检测频率。", self.app._action_set_interval)):
            card = self._card(advanced, label)
            ttk.Button(card, text="调整", style="Clear.TButton", command=lambda a=action: self._act(a)).pack(side="right")
            ttk.Label(card, text=description, style="Muted.TLabel").pack(side="left", pady=8)
        # Bind only this window; unrelated alert and ROI windows keep their own input handling.
        win.bind("<MouseWheel>", self._on_mousewheel)
        self.refresh()
        self._show_page("home")
        win.deiconify()
        self._poll_status()

    def _highlight_timing(self):
        style = ttk.Style(self.window)
        selected = self.timer.get()
        for value, row in self._timing_rows.items():
            color = "#edf4fe" if value == selected else "white"
            style.configure(f"{value}.TFrame", background=color)
            style.configure(f"{value}.TRadiobutton", background=color)
            style.map(f"{value}.TRadiobutton", background=[("active", color)])
            for child in row.winfo_children():
                if isinstance(child, ttk.Label):
                    child.configure(background=color)
                elif isinstance(child, RoundedButton):
                    tk.Canvas.configure(child, bg=color)

    def _time_stepper(self, parent, kind):
        frame = ttk.Frame(parent)
        frame.pack(side="left", padx=(0, 10))
        entry = ttk.Entry(frame, width=7, justify="center", font=("Segoe UI", 11), style="Time.TEntry")
        entry.pack(side="left", fill="y")
        arrows = tk.Frame(frame, bg="#e9eff8")
        arrows.pack(side="left", padx=(4, 0))

        def step(direction):
            current = entry.get()
            if kind == "clock":
                value = parse_target_time(current)
                if value is None:
                    self.dialogs.info("时间格式错误", "请输入有效的 24 小时时间，例如 23:00。")
                    return
                cursor = entry.index(tk.INSERT)
                minutes = (value.hour * 60 + value.minute + direction * (60 if cursor <= 2 else 1)) % 1440
                text = f"{minutes // 60:02d}:{minutes % 60:02d}"
            elif kind == "countdown":
                value = parse_countdown_hours(current)
                if value is None:
                    self.dialogs.info("时长无效", "请输入 0～24 小时。")
                    return
                text = f"{max(0.25, min(24, value + direction * 0.25)):g}"
            else:
                try:
                    value = int(current)
                except ValueError:
                    self.dialogs.info("时长无效", "请输入 0～1440 分钟的预计单局时长。")
                    return
                text = str(max(0, min(1440, value + direction)))
            entry.delete(0, "end")
            entry.insert(0, text)
            entry.focus_set()
            if kind == "clock":
                entry.icursor(cursor)

        for symbol, direction in (("▴", 1), ("▾", -1)):
            tk.Button(arrows, text=symbol, command=lambda d=direction: step(d),
                      font=("Segoe UI", 8), bg="#e9eff8", fg="#0067c0",
                      activebackground="#d7e5f7", relief="flat", bd=0,
                      padx=8, pady=0, cursor="hand2", takefocus=False).pack()
        def key_step(direction):
            step(direction)
            return "break"
        entry.bind("<Up>", lambda _event: key_step(1))
        entry.bind("<Down>", lambda _event: key_step(-1))
        return entry

    def _setup_styles(self, win):
        style = ttk.Style(win)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), background="white", foreground="#20252d")
        style.configure("TFrame", background="white")
        style.configure("Shell.TFrame", background="#f3f5f9")
        style.configure("TLabel", background="white")
        style.configure("Title.TLabel", background="#f3f5f9", font=("Microsoft YaHei UI", 24, "bold"))
        style.configure("Subtitle.TLabel", background="#f3f5f9", foreground="#697586")
        style.configure("Muted.TLabel", background="white", foreground="#697586")
        style.configure("Plan.TLabel", background="#edf4fe", foreground="#31618f", font=("Microsoft YaHei UI", 9))
        style.configure("Time.TEntry", padding=(8, 6), fieldbackground="white", bordercolor="#cfdaeb")
        style.configure("CardTitle.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Clear.TButton", padding=(12, 6), background="#f8f9fb", bordercolor="#dce1e8", relief="flat")
        style.map("Clear.TButton", background=[("pressed", "#e2e8f0"), ("active", "#eef2f7")])
        for name in ("TEntry", "TCombobox", "TSpinbox"):
            style.configure(name, padding=5, fieldbackground="white", bordercolor="#d5dce6", arrowcolor="#5b6675")
            style.map(name, fieldbackground=[("readonly", "white")], bordercolor=[("focus", "#0067c0")])
        style.configure("TRadiobutton", background="white", padding=(0, 4))
        style.map("TRadiobutton", background=[("active", "white")])
        self._startup_check_images = []
        for selected in (False, True):
            image = Image.new("RGBA", (18, 18), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle((1, 1, 16, 16), radius=3,
                                   fill="#0067c0" if selected else "white",
                                   outline="#0067c0" if selected else "#8996a5", width=1)
            if selected:
                draw.line((4, 9, 7, 12, 13, 5), fill="white", width=2, joint="curve")
            self._startup_check_images.append(ImageTk.PhotoImage(image, master=win))
        style.element_create("Startup.indicator", "image", self._startup_check_images[0],
                             ("selected", self._startup_check_images[1]), width=20, sticky="w")
        style.layout("Startup.TCheckbutton", [("Checkbutton.padding", {"sticky": "nswe", "children": [
            ("Startup.indicator", {"side": "left", "sticky": ""}),
            ("Checkbutton.focus", {"side": "left", "sticky": "w", "children": [
                ("Checkbutton.label", {"sticky": "nswe"})]})]})])
        style.configure("Startup.TCheckbutton", background="white", padding=(0, 2))
        style.map("Startup.TCheckbutton", background=[("active", "white")])
        style.configure("Vertical.TScrollbar", background="#d5dce6", troughcolor="#f3f5f9", borderwidth=0, arrowsize=12)

    def _navigation_icon(self, parent, key):
        # Native buttons paint their own image reliably; avoid child-label overlays.
        image = Image.new("RGBA", (88, 88))
        draw = ImageDraw.Draw(image)
        color = "#52677f"
        if key == "home":
            draw.line([(12, 41), (44, 13), (76, 41)], fill=color, width=6)
            draw.line([(23, 36), (23, 73), (65, 73), (65, 36)], fill=color, width=6)
            draw.line([(39, 73), (39, 51), (51, 51), (51, 73)], fill=color, width=5)
        elif key == "reminder":
            draw.line([(39, 62), (39, 23), (68, 16), (68, 55)], fill=color, width=6)
            draw.ellipse((18, 56, 40, 73), fill=color)
            draw.ellipse((47, 49, 69, 66), fill=color)
        elif key == "templates":
            draw.rounded_rectangle((14, 17, 74, 71), radius=7, outline=color, width=5)
            draw.ellipse((25, 27, 36, 38), fill=color)
            draw.line([(20, 64), (39, 45), (51, 56), (61, 43), (72, 56)], fill=color, width=5)
        else:
            draw.ellipse((23, 23, 65, 65), outline=color, width=6)
            draw.ellipse((36, 36, 52, 52), outline=color, width=5)
            for index in range(8):
                angle = index * math.pi / 4
                draw.line([(44 + math.cos(angle) * 22, 44 + math.sin(angle) * 22),
                           (44 + math.cos(angle) * 32, 44 + math.sin(angle) * 32)],
                          fill=color, width=7)
        return ImageTk.PhotoImage(image.resize((22, 22), Image.Resampling.LANCZOS), master=parent)

    def _hide_text_help(self):
        popup = getattr(self, "_text_help", None)
        if popup is not None:
            popup.destroy()
            self._text_help = None

    def _dismiss_text_help(self, event):
        popup = self._text_help
        widget = event.widget
        if popup is None or widget == self._text_help_button:
            return
        while widget is not None:
            if widget == popup:
                return
            widget = getattr(widget, "master", None)
        self._hide_text_help()

    def _text_help_focus_out(self, _event):
        def check_focus():
            if self.window.winfo_exists() and self.window.focus_displayof() is None:
                self._hide_text_help()
        self.window.after_idle(check_focus)

    def _text_help_window_changed(self, event):
        if event.widget == self.window:
            self._hide_text_help()

    def _show_text_help(self):
        if self._text_help is not None:
            self._hide_text_help()
            return
        popup = tk.Frame(self.window, bg="white", highlightbackground="#b8c9de",
                         highlightthickness=1, padx=12, pady=10)
        self._text_help = popup
        for text, bold in (
                ("文案占位符", True),
                ("{game}  游戏名称   ·   {label}  结算结果", False),
                ("{time}  当前时间   ·   {id}  模板 ID", False),
                ("{score}  匹配分数", False),
                ("{score:.3f}  保留 3 位小数，如 0.923", False),
                ("示例：{game}：{label}，现在是 {time}", False),
                ("使用英文半角大括号；{{ 和 }} 显示大括号。格式写错时按原文显示。", False)):
            tk.Label(popup, text=text, bg="white", fg="#275581" if bold else "#4f5966",
                     font=("Microsoft YaHei UI", 10, "bold" if bold else "normal"),
                     wraplength=320, justify="left", anchor="w").pack(anchor="w", pady=2)
        popup.update_idletasks()
        button = self._text_help_button
        x = button.winfo_rootx() - self.window.winfo_rootx() + button.winfo_width() - popup.winfo_reqwidth()
        y = button.winfo_rooty() - self.window.winfo_rooty() + button.winfo_height() + 5
        x = max(8, min(x, self.window.winfo_width() - popup.winfo_reqwidth() - 8))
        y = max(8, min(y, self.window.winfo_height() - popup.winfo_reqheight() - 8))
        popup.place(x=x, y=y)
        popup.lift()

    def _card(self, parent, title, help_command=None):
        # A canvas border gives cards rounded corners without adding a UI dependency.
        canvas = tk.Canvas(parent, bg="#f3f5f9", highlightthickness=0, height=1)
        canvas.pack(fill="x", pady=(0, 8))
        content = ttk.Frame(canvas, padding=(16, 8, 16, 8))
        item = canvas.create_window(1, 1, window=content, anchor="nw")
        header = ttk.Frame(content)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text=title, style="CardTitle.TLabel").pack(side="left")
        if help_command is not None:
            content.help_button = tk.Button(header, text="?", command=help_command, font=("Segoe UI", 10, "bold"),
                      bg="#edf4fe", fg="#31618f", activebackground="#dce9fa",
                      relief="flat", bd=0, width=2, cursor="hand2",
                      takefocus=True)
            content.help_button.pack(side="right")

        def resize(_event=None):
            width = canvas.winfo_width()
            inset = 10
            canvas.itemconfigure(item, width=max(1, width - inset * 2))
            canvas.coords(item, inset, inset)
            height = content.winfo_reqheight() + inset * 2
            canvas.configure(height=height)
            canvas.delete("border")
            radius = 28
            points = [radius, 1, width-radius, 1, width-1, 1, width-1, radius,
                      width-1, height-radius, width-1, height-1, width-radius, height-1,
                      radius, height-1, 1, height-1, 1, height-radius, 1, radius, 1, 1]
            canvas.create_polygon(points, smooth=True, splinesteps=24, fill="white", outline="#e1e6ed", tags="border")
            canvas.tag_lower("border")
        canvas.bind("<Configure>", resize)
        content.bind("<Configure>", resize)
        return content

    def _show_page(self, key):
        self._hide_text_help()
        titles = {"home": ("开始", "安排好最后一局，剩下的交给提醒。"),
                  "reminder": ("提醒设置", "选择提醒方式，设置你喜欢的文案与提示音。"),
                  "templates": ("模板与识别", "管理结算画面模板，校准检测区域。"),
                  "advanced": ("高级设置", "微调识别参数与提醒间隔。")}
        for page, (_content, frame) in self._pages.items():
            frame.pack_forget()
            self._nav_buttons[page].configure(bg="#dde8f6" if page == key else "#edf0f5",
                                               fg="#005aab" if page == key else "#4f5966")
        self._pages[key][1].pack(fill="both", expand=True)
        self._active_page = key
        self.page_title.configure(text=titles[key][0])
        self.page_description.configure(text=titles[key][1])

    def _scroll_content(self, parent, height=0):
        frame = ttk.Frame(parent, style="Shell.TFrame")
        canvas = tk.Canvas(frame, highlightthickness=0, borderwidth=0, bg="#f3f5f9", height=height)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)

        def update_scrollbar(first, last):
            scrollbar.set(first, last)
            if float(first) <= 0 and float(last) >= 1:
                scrollbar.pack_forget()
            elif not scrollbar.winfo_manager():
                scrollbar.pack(side="right", fill="y", before=canvas)

        canvas.configure(yscrollcommand=update_scrollbar)
        canvas.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(canvas, style="Shell.TFrame")
        item = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(item, width=event.width))
        frame.scroll_canvas = canvas
        return content, frame

    def _on_mousewheel(self, event):
        if isinstance(event.widget, (ttk.Combobox, ttk.Spinbox)):
            return
        canvas = self._pages[self._active_page][1].scroll_canvas
        if canvas.bbox("all") and canvas.bbox("all")[3] > canvas.winfo_height():
            canvas.yview_scroll(-int(event.delta / 120), "units")

    def _act(self, action):
        action(self.app._icon, None)
        self.refresh()

    def _choose_game(self, _event):
        self.app._action_select_profile(self.app._profiles[self.game.current()].id)(self.app._icon, None)
        self.refresh()

    def _bind_timing_row(self, row, value):
        # Bind empty space and labels without intercepting entry/button actions.
        def select(_event):
            self.timer.set(value)
            self._set_timer()
        for child in (row, *row.winfo_children()):
            if isinstance(child, (ttk.Frame, ttk.Label)):
                child.configure(cursor="hand2")
                child.bind("<Button-1>", select, add="+")

    def _set_timer(self):
        mode = self.timer.get()
        if mode == "clock":
            valid = self._save_timer()
        elif mode == "countdown":
            valid = self._save_countdown()
        else:
            self.app._set_timer_mode(mode)
            valid = True
        if not valid:
            self.timer.set(self.app._timer_mode)

    def _save_countdown(self):
        hours = parse_countdown_hours(self.countdown.get())
        if hours is None:
            self.dialogs.info("时长无效", "请输入 0～24 小时。")
            return False
        self.app._countdown_hours = hours
        self.app._set_timer_mode("countdown")
        return True

    def _save_duration(self):
        try:
            duration = int(self.duration.get())
        except ValueError:
            duration = -1
        if not 0 <= duration <= 1440:
            self.dialogs.info("时长无效", "请输入 0～1440 分钟的预计单局时长。")
            return False
        overrides = self.app._cfg.get("timer_game_durations")
        if not isinstance(overrides, dict):
            overrides = {}
            self.app._cfg["timer_game_durations"] = overrides
        overrides[self.app._profile_base.id] = duration
        self.app._persist()
        self.app._rebuild_menu()
        self.refresh()
        return True

    def _save_timer(self):
        target = parse_target_time(self.target.get())
        if target is None:
            self.dialogs.info("时间格式错误", "请输入有效的 24 小时时间，例如 23:00。")
            return False
        self.app._timer_target_time = target
        self.app._set_timer_mode("clock")
        return True

    def _set_mode(self):
        if self.mode.get() in VALID_NOTIFY_MODES:
            self.app._action_set_mode(self.mode.get())(self.app._icon, None)
        self.refresh()

    def _toggle_detection(self):
        if not self.app._detector.is_running():
            if self.timer.get() == "clock" and not self._save_timer():
                return
            if self.timer.get() == "countdown" and not self._save_countdown():
                return
        self._act(self.app._action_stop if self.app._detector.is_running() else self.app._action_start)

    def _set_auto_start(self):
        self.app._cfg["auto_start_detection"] = self.auto_start.get()
        self.app._persist()

    def _set_windows_startup(self):
        enabled = self.start_with_windows.get()
        try:
            windows_startup.set_enabled(enabled)
        except OSError as exc:
            self.start_with_windows.set(not enabled)
            self.dialogs.info("开机自启动设置失败", f"无法更新 Windows 启动项：{exc}")

    def _capture(self):
        if not self.app._profile.templates:
            self.dialogs.info("没有模板", "当前游戏没有可抓取的模板。")
            return
        label = self.capture_choice.get()
        for template in self.app._profile.templates:
            if label == template.label:
                self._act(self.app._action_capture_template(template))
                return

    def _save_text(self):
        title = self.title_text.get().strip()
        body = self.body_text.get().strip()
        if not title or not body:
            self.dialogs.info("提醒文案", "标题和正文都不能为空。")
            return
        self.app._notify_settings.title_tpl = title
        self.app._notify_settings.msg_tpl = body
        self.app._persist()
        self.app._rebuild_menu()

    def _refresh_status(self):
        running = self.app._detector.is_running()
        self.toggle.config(text="停止检测" if running else "开始检测",
                           bg="#c42b3a" if running else "#0067c0",
                           activebackground="#a8202d" if running else "#005aab")
        self.status_label.config(text="正在检测" if running else "准备就绪",
                                 foreground="#0067c0" if running else "#20252d")
        self.status_description.config(text="检测到结算画面后，将按设置提醒你。" if running
                                       else "选好游戏与提醒时机，即可开始检测。")

    def _poll_status(self):
        if not self.window.winfo_exists():
            return
        # Refresh only status, so background updates never replace an unfinished edit.
        self._refresh_status()
        self.window.after(500, self._poll_status)

    def refresh(self):
        if not self.window:
            return
        self.game["values"] = [p.display_name for p in self.app._profiles]
        self.game.current(next((i for i, p in enumerate(self.app._profiles)
                                if p.id == self.app._profile_base.id), 0))
        self._refresh_status()
        self.timer.set(self.app._timer_mode)
        if self.window.focus_get() not in (self.target, self.duration, self.countdown):
            self.target.delete(0, "end")
            self.target.insert(0, self.app._timer_target_time.strftime("%H:%M"))
            self.duration.delete(0, "end")
            self.duration.insert(0, str(self.app._game_duration_min(self.app._profile_base)))
            self.countdown.delete(0, "end")
            self.countdown.insert(0, f"{self.app._countdown_hours:g}")
        if self.app._timer_mode == "countdown":
            timing_text = self.app._countdown_window_label() + "，结算后提醒"
        elif self.app._timer_mode == "clock":
            timing_text = self.app._timer_window_label() + "，下一次结算时提醒"
        else:
            timing_text = "每次检测到新的结算画面时提醒"
        self.window_label.config(text=timing_text)
        self.mode.set(self.app._notify_settings.mode)
        self.title_text.delete(0, "end")
        self.title_text.insert(0, self.app._notify_settings.title_tpl)
        self.body_text.delete(0, "end")
        self.body_text.insert(0, self.app._notify_settings.msg_tpl)
        self.sound.config(text="当前提示音：" + (self.app._notify_settings.sound_path or "该下播了老铁.wav"))
        self.template_state.config(text=f"当前游戏：{self.app._profile_base.display_name}，模板槽位：{len(self.app._profile.templates)}")
        self.capture_choice["values"] = [t.label for t in self.app._profile.templates]
        if self.app._profile.templates:
            self.capture_choice.current(0)

    def show(self):
        def reveal(_root):
            self.refresh()
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()
        self.dialogs.run_in_tk(reveal)
