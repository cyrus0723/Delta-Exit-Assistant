"""Small control window sharing the tray application's existing state."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from alert_timer import parse_target_time, parse_countdown_hours
from notify import VALID_NOTIFY_MODES


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
        screen_height = win.winfo_screenheight()
        win.geometry(f"800x{min(760, screen_height - 80)}")
        win.minsize(700, 570)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        win.configure(bg="#f5f7f6")
        win.bind_all("<MouseWheel>", self._on_mousewheel)
        style = ttk.Style(win)
        style.configure("Clear.TButton", font=("Microsoft YaHei UI", 10), padding=(12, 7))
        style.configure("Clear.TNotebook.Tab", font=("Microsoft YaHei UI", 10), padding=(12, 6))
        outer = ttk.Frame(win, padding=20)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)
        ttk.Label(outer, text="三角洲下机助手", font=("Microsoft YaHei UI", 19, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Button(outer, text="退出程序", style="Clear.TButton",
                   command=lambda: self.app._action_quit(self.app._icon, None)).grid(row=0, column=1, sticky="e")

        controls = ttk.Frame(outer)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 8))

        row = ttk.Frame(controls)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text="当前游戏").pack(side="left")
        self.game = ttk.Combobox(row, state="readonly", width=23)
        self.game.pack(side="left", padx=12)
        self.game.bind("<<ComboboxSelected>>", self._choose_game)
        ttk.Button(row, text="新建游戏", style="Clear.TButton",
                   command=lambda: self._act(self.app._action_new_game)).pack(side="left")

        timing = ttk.LabelFrame(controls, text="提醒时机", padding=12)
        timing.pack(fill="x", pady=(6, 0))
        self.timer = tk.StringVar()
        ttk.Radiobutton(timing, text="每局结束时", variable=self.timer, value="always",
                        command=self._set_timer).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Radiobutton(timing, text="玩到指定时间", variable=self.timer, value="clock",
                        command=self._set_timer).grid(row=1, column=0, sticky="w", pady=5)
        self.target = ttk.Entry(timing, width=8)
        self.target.grid(row=1, column=1, padx=8)
        ttk.Label(timing, text="预计单局").grid(row=1, column=2)
        self.duration = ttk.Spinbox(timing, from_=0, to=1440, width=6)
        self.duration.grid(row=1, column=3, padx=8)
        ttk.Label(timing, text="分钟").grid(row=1, column=4)
        ttk.Button(timing, text="应用时间", style="Clear.TButton", command=self._save_timer).grid(row=1, column=5, padx=8)
        ttk.Radiobutton(timing, text="再玩", variable=self.timer, value="countdown",
                        command=self._set_timer).grid(row=2, column=0, sticky="w", pady=5)
        self.countdown = ttk.Entry(timing, width=8)
        self.countdown.grid(row=2, column=1, padx=8)
        ttk.Label(timing, text="小时").grid(row=2, column=2, sticky="w")
        ttk.Button(timing, text="应用时长", style="Clear.TButton", command=self._save_countdown).grid(row=2, column=5, padx=8)
        self.window_label = ttk.Label(timing)
        self.window_label.grid(row=3, column=0, columnspan=6, sticky="w", pady=(8, 0))

        self.toggle = tk.Button(outer, command=self._toggle_detection,
                                font=("Microsoft YaHei UI", 12, "bold"),
                                fg="white", activeforeground="white", relief="flat",
                                borderwidth=0, padx=44, pady=10, width=14, cursor="hand2")
        self.toggle.grid(row=2, column=0, columnspan=2, pady=(0, 12))
        tabs = ttk.Notebook(outer, style="Clear.TNotebook")
        tabs.grid(row=3, column=0, columnspan=2, sticky="nsew")
        reminder = self._scroll_tab(tabs, "提醒设置")
        templates = self._scroll_tab(tabs, "模板与识别")
        advanced = self._scroll_tab(tabs, "高级设置")
        self.mode = tk.StringVar()
        mode_choices = ttk.Frame(reminder)
        mode_choices.pack(fill="x")
        mode_choices.columnconfigure((0, 1), weight=1, uniform="mode")
        for index, (label, value) in enumerate((("应用弹窗 + 提示音", "popup"),
                                                ("Windows 通知 + 提示音", "native"),
                                                ("仅 Windows 通知", "native_silent"),
                                                ("只响铃", "sound"))):
            ttk.Radiobutton(mode_choices, text=label, value=value, variable=self.mode,
                            command=self._set_mode).grid(row=index // 2, column=index % 2,
                                                         sticky="w", pady=4)
        text_row = ttk.Frame(reminder)
        text_row.pack(fill="x", pady=(8, 0))
        ttk.Label(text_row, text="标题").grid(row=0, column=0, sticky="w", pady=3)
        self.title_text = ttk.Entry(text_row, width=48)
        self.title_text.grid(row=0, column=1, padx=8, pady=3)
        ttk.Label(text_row, text="正文").grid(row=1, column=0, sticky="w", pady=3)
        self.body_text = ttk.Entry(text_row, width=48)
        self.body_text.grid(row=1, column=1, padx=8, pady=3)
        ttk.Button(text_row, text="保存文案", style="Clear.TButton", command=self._save_text).grid(row=0, column=2, rowspan=2)
        actions = ttk.Frame(reminder)
        actions.pack(fill="x", pady=8)
        for label, action in (("选择提示音", self.app._action_select_sound),
                              ("试听提示音", self.app._action_test_sound),
                              ("恢复默认音", self.app._action_reset_sound),
                              ("测试提醒", self.app._action_test_notify)):
            ttk.Button(actions, text=label, style="Clear.TButton",
                       command=lambda a=action: self._act(a)).pack(side="left", padx=(0, 6))
        self.sound = ttk.Label(reminder)
        self.sound.pack(anchor="w")
        self.template_state = ttk.Label(templates)
        self.template_state.pack(anchor="w", pady=(0, 12))
        capture_row = ttk.Frame(templates)
        capture_row.pack(fill="x", pady=4)
        self.capture_choice = ttk.Combobox(capture_row, state="readonly", width=18)
        self.capture_choice.pack(side="left", padx=(0, 8))
        ttk.Button(capture_row, text="抓取模板", style="Clear.TButton", command=self._capture).pack(side="left", padx=(0, 8))
        ttk.Button(capture_row, text="重载模板", style="Clear.TButton", command=lambda: self._act(self.app._action_reload_templates)).pack(side="left")
        roi_row = ttk.Frame(templates)
        roi_row.pack(fill="x", pady=10)
        ttk.Button(roi_row, text="预览检测区域", style="Clear.TButton", command=lambda: self._act(self.app._action_preview_roi)).pack(side="left", padx=(0, 8))
        ttk.Button(roi_row, text="恢复默认区域", style="Clear.TButton", command=lambda: self._act(self.app._action_roi_reset)).pack(side="left")
        move = ttk.LabelFrame(templates, text="检测区域微调", padding=8)
        move.pack(anchor="w", fill="x")
        for label, direction in (("上移", "up"), ("下移", "down"), ("左移", "left"), ("右移", "right")):
            ttk.Button(move, text=label, style="Clear.TButton", command=lambda d=direction: self._act(self.app._action_roi_move(d))).pack(side="left", padx=3)
        size = ttk.Frame(templates)
        size.pack(anchor="w", pady=5)
        for label, which in (("加宽", "wider"), ("变窄", "narrower"), ("加高", "taller"), ("变矮", "shorter")):
            ttk.Button(size, text=label, style="Clear.TButton", command=lambda w=which: self._act(self.app._action_roi_resize(w))).pack(side="left", padx=3)
        for label, action in (("匹配阈值", self.app._action_set_threshold),
                              ("回落差值", self.app._action_set_hysteresis),
                              ("冷却时间", self.app._action_set_cooldown),
                              ("扫描间隔", self.app._action_set_interval)):
            ttk.Button(advanced, text=label, style="Clear.TButton", command=lambda a=action: self._act(a)).pack(anchor="w", pady=5)
        self.refresh()
        win.deiconify()

    def _scroll_tab(self, tabs, title):
        page = ttk.Frame(tabs)
        tabs.add(page, text=title)
        content, frame = self._scroll_content(page)
        frame.pack(fill="both", expand=True)
        return content

    def _scroll_content(self, parent, height=0):
        frame = ttk.Frame(parent)
        canvas = tk.Canvas(frame, highlightthickness=0, borderwidth=0, bg="#f5f7f6", height=height)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        content = ttk.Frame(canvas, padding=14)
        item = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(item, width=event.width))
        return content, frame

    def _on_mousewheel(self, event):
        widget = event.widget
        while widget is not None and not isinstance(widget, tk.Canvas):
            widget = getattr(widget, "master", None)
        if widget is not None:
            widget.yview_scroll(-int(event.delta / 120), "units")

    def _act(self, action):
        action(self.app._icon, None)
        self.refresh()

    def _choose_game(self, _event):
        self.app._action_select_profile(self.app._profiles[self.game.current()].id)(self.app._icon, None)
        self.refresh()

    def _set_timer(self):
        self.app._set_timer_mode(self.timer.get())

    def _save_countdown(self):
        hours = parse_countdown_hours(self.countdown.get())
        try:
            duration = int(self.duration.get())
        except ValueError:
            duration = -1
        if hours is None or not 0 <= duration <= 1440:
            self.dialogs.info("时长无效", "请输入 0～24 小时和 0～1440 分钟的预计单局时长。")
            return False
        self.app._countdown_hours = hours
        overrides = self.app._cfg.get("timer_game_durations")
        if not isinstance(overrides, dict):
            overrides = {}
            self.app._cfg["timer_game_durations"] = overrides
        overrides[self.app._profile_base.id] = duration
        self.app._set_timer_mode("countdown")
        return True

    def _save_timer(self):
        target = parse_target_time(self.target.get())
        try:
            duration = int(self.duration.get())
        except ValueError:
            duration = -1
        if target is None or not 0 <= duration <= 1440:
            self.dialogs.info("时间格式错误", "请输入有效的 24 小时时间和 0～1440 分钟的单局时长。")
            return
        self.app._timer_target_time = target
        overrides = self.app._cfg.get("timer_game_durations")
        if not isinstance(overrides, dict):
            overrides = {}
            self.app._cfg["timer_game_durations"] = overrides
        overrides[self.app._profile_base.id] = duration
        self.app._persist()
        self.app._rebuild_menu()
        self.refresh()

    def _set_mode(self):
        if self.mode.get() in VALID_NOTIFY_MODES:
            self.app._action_set_mode(self.mode.get())(self.app._icon, None)
        self.refresh()

    def _toggle_detection(self):
        if not self.app._detector.is_running() and self.timer.get() == "countdown":
            if not self._save_countdown():
                return
        self._act(self.app._action_stop if self.app._detector.is_running() else self.app._action_start)

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

    def refresh(self):
        if not self.window:
            return
        self.game["values"] = [p.display_name for p in self.app._profiles]
        self.game.current(next((i for i, p in enumerate(self.app._profiles)
                                if p.id == self.app._profile_base.id), 0))
        running = self.app._detector.is_running()
        self.toggle.config(text="停止检测" if running else "开始检测",
                           bg="#bd3540" if running else "#16805a",
                           activebackground="#a82933" if running else "#116747")
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
