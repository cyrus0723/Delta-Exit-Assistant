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
        win.geometry(f"1060x{min(780, win.winfo_screenheight() - 80)}")
        win.minsize(940, 600)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        win.configure(bg="#f3f5f9")
        self._setup_styles(win)
        self._pages = {}
        self._nav_buttons = {}

        sidebar = tk.Frame(win, bg="#edf0f5", width=204)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="Δ", font=("Segoe UI", 30, "bold"),
                 bg="#edf0f5", fg="#0067c0").pack(anchor="w", padx=26, pady=(25, 0))
        tk.Label(sidebar, text="三角洲下机助手", font=("Microsoft YaHei UI", 13, "bold"),
                 bg="#edf0f5", fg="#20252d").pack(anchor="w", padx=24, pady=(2, 0))
        tk.Label(sidebar, text="DELTA EXIT ASSISTANT", font=("Segoe UI", 8),
                 bg="#edf0f5", fg="#78828f").pack(anchor="w", padx=25, pady=(5, 30))
        for key, symbol, title in (("home", "⌂", "开始"), ("reminder", "♪", "提醒设置"),
                                   ("templates", "▧", "模板与识别"), ("advanced", "⚙", "高级设置")):
            button = tk.Button(sidebar, text=f"{symbol}    {title}", anchor="w",
                               font=("Microsoft YaHei UI", 11), relief="flat", bd=0,
                               bg="#edf0f5", fg="#4f5966", activebackground="#dfe7f1",
                               activeforeground="#005aab", padx=18, pady=12, cursor="hand2",
                               command=lambda page=key: self._show_page(page))
            button.pack(fill="x", padx=12, pady=3)
            self._nav_buttons[key] = button
        tk.Button(sidebar, text="退出程序", command=lambda: self.app._action_quit(self.app._icon, None),
                  bg="#edf0f5", fg="#637080", activebackground="#e1e6ee", relief="flat", bd=0,
                  font=("Microsoft YaHei UI", 10), cursor="hand2", pady=12).pack(side="bottom", fill="x", padx=16, pady=(0, 18))
        tk.Label(sidebar, text="关闭窗口后继续在托盘运行", bg="#edf0f5", fg="#78828f",
                 font=("Microsoft YaHei UI", 8)).pack(side="bottom", pady=10)

        main = ttk.Frame(win, style="Shell.TFrame", padding=(28, 26, 24, 14))
        main.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(main, style="Shell.TFrame")
        header.pack(fill="x", pady=(0, 20))
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
        self.status_label = ttk.Label(status_text, font=("Microsoft YaHei UI", 22, "bold"),
                                      background="white", foreground="#20252d")
        self.status_label.pack(anchor="w", pady=(0, 5))
        self.status_description = ttk.Label(status_text, style="Muted.TLabel", wraplength=380)
        self.status_description.pack(anchor="w")
        self.toggle = tk.Button(status_row, command=self._toggle_detection,
                                font=("Microsoft YaHei UI", 11, "bold"),
                                fg="white", activeforeground="white", relief="flat",
                                borderwidth=0, padx=30, pady=10, cursor="hand2")
        self.toggle.pack(side="right", padx=(12, 0))
        row = ttk.Frame(hero)
        row.pack(fill="x", pady=(14, 0))
        ttk.Label(row, text="当前游戏").pack(side="left", padx=(0, 12))
        self.game = ttk.Combobox(row, state="readonly", width=27)
        self.game.pack(side="left", fill="x", expand=True, padx=(0, 12))
        self.game.bind("<<ComboboxSelected>>", self._choose_game)
        ttk.Button(row, text="新建游戏", style="Clear.TButton",
                   command=lambda: self._act(self.app._action_new_game)).pack(side="right")

        timing = self._card(home, "提醒时机")
        self.timer = tk.StringVar()
        for label, value in (("每局结束时", "always"), ("玩到指定时间", "clock"), ("再玩", "countdown")):
            row = ttk.Frame(timing)
            row.pack(fill="x", pady=3)
            ttk.Radiobutton(row, text=label, variable=self.timer, value=value,
                            command=self._set_timer, width=16).pack(side="left")
            if value == "clock":
                self.target = ttk.Entry(row, width=9)
                self.target.pack(side="left", padx=(0, 10))
                ttk.Button(row, text="应用时间", style="Clear.TButton", command=self._save_timer).pack(side="right")
            elif value == "countdown":
                self.countdown = ttk.Entry(row, width=9)
                self.countdown.pack(side="left", padx=(0, 10))
                ttk.Label(row, text="小时").pack(side="left")
                ttk.Button(row, text="应用时长", style="Clear.TButton", command=self._save_countdown).pack(side="right")
        row = ttk.Frame(timing)
        row.pack(fill="x", pady=(6, 4))
        ttk.Label(row, text="预计单局", width=16).pack(side="left")
        self.duration = ttk.Spinbox(row, from_=0, to=1440, width=9)
        self.duration.pack(side="left", padx=(0, 10))
        ttk.Label(row, text="分钟").pack(side="left")
        self.window_label = ttk.Label(timing, style="Muted.TLabel", wraplength=580)
        self.window_label.pack(anchor="w", pady=(8, 2))

        modes = self._card(reminder, "提醒方式")
        self.mode = tk.StringVar()
        mode_choices = ttk.Frame(modes)
        mode_choices.pack(fill="x")
        mode_choices.columnconfigure((0, 1), weight=1)
        for index, (label, value) in enumerate((("应用弹窗 + 提示音", "popup"), ("Windows 通知 + 提示音", "native"),
                                              ("仅 Windows 通知", "native_silent"), ("只响铃", "sound"))):
            ttk.Radiobutton(mode_choices, text=label, value=value, variable=self.mode,
                            command=self._set_mode).grid(row=index // 2, column=index % 2, sticky="w", pady=7)
        text_card = self._card(reminder, "提醒文案")
        ttk.Label(text_card, text="标题").pack(anchor="w", pady=(0, 6))
        self.title_text = ttk.Entry(text_card)
        self.title_text.pack(fill="x", pady=(0, 14))
        ttk.Label(text_card, text="正文").pack(anchor="w", pady=(0, 6))
        self.body_text = ttk.Entry(text_card)
        self.body_text.pack(fill="x", pady=(0, 14))
        ttk.Button(text_card, text="保存文案", style="Clear.TButton", command=self._save_text).pack(anchor="w")
        sound_card = self._card(reminder, "提示音")
        self.sound = ttk.Label(sound_card, style="Muted.TLabel", wraplength=580)
        self.sound.pack(anchor="w", pady=(0, 14))
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
        style.configure("CardTitle.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Clear.TButton", padding=(14, 8), background="#f8f9fb", bordercolor="#dce1e8", relief="flat")
        style.map("Clear.TButton", background=[("pressed", "#e2e8f0"), ("active", "#eef2f7")])
        for name in ("TEntry", "TCombobox", "TSpinbox"):
            style.configure(name, padding=7, fieldbackground="white", bordercolor="#d5dce6", arrowcolor="#5b6675")
            style.map(name, fieldbackground=[("readonly", "white")], bordercolor=[("focus", "#0067c0")])
        style.configure("TRadiobutton", background="white", padding=(0, 4))
        style.map("TRadiobutton", background=[("active", "white")])
        style.configure("Vertical.TScrollbar", background="#d5dce6", troughcolor="#f3f5f9", borderwidth=0, arrowsize=12)

    def _card(self, parent, title):
        # A canvas border gives cards rounded corners without adding a UI dependency.
        canvas = tk.Canvas(parent, bg="#f3f5f9", highlightthickness=0, height=1)
        canvas.pack(fill="x", pady=(0, 12))
        content = ttk.Frame(canvas, padding=(16, 12, 16, 14))
        item = canvas.create_window(1, 1, window=content, anchor="nw")
        ttk.Label(content, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 10))

        def resize(_event=None):
            width = canvas.winfo_width()
            inset = 12
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
        self.app._set_timer_mode("clock")

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
                           bg="#c42b3a" if running else "#0067c0",
                           activebackground="#a8202d" if running else "#005aab")
        self.status_label.config(text="正在检测" if running else "准备就绪",
                                 foreground="#0067c0" if running else "#20252d")
        self.status_description.config(text="检测到结算画面后，将按设置提醒你。" if running
                                       else "选好游戏与提醒时机，即可开始检测。")
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
