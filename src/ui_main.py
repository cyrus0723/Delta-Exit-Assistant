"""Small control window sharing the tray application's existing state."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from alert_timer import parse_target_time
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
        win.geometry("760x640")
        win.minsize(700, 570)
        win.protocol("WM_DELETE_WINDOW", win.withdraw)
        win.configure(bg="#f5f7f6")
        outer = ttk.Frame(win, padding=20)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="三角洲下机助手", font=("Microsoft YaHei UI", 19, "bold")).pack(anchor="w")

        row = ttk.Frame(outer)
        row.pack(fill="x", pady=(22, 8))
        ttk.Label(row, text="当前游戏").pack(side="left")
        self.game = ttk.Combobox(row, state="readonly", width=23)
        self.game.pack(side="left", padx=12)
        self.game.bind("<<ComboboxSelected>>", self._choose_game)
        ttk.Button(row, text="新建游戏", command=lambda: self._act(self.app._action_new_game)).pack(side="left")

        self.status = ttk.Label(outer, font=("Microsoft YaHei UI", 11))
        self.status.pack(anchor="w", pady=8)
        timing = ttk.LabelFrame(outer, text="提醒时机", padding=12)
        timing.pack(fill="x", pady=12)
        self.timer = tk.BooleanVar()
        ttk.Radiobutton(timing, text="每局结束时", variable=self.timer, value=False,
                        command=self._set_timer).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Radiobutton(timing, text="玩到指定时间", variable=self.timer, value=True,
                        command=self._set_timer).grid(row=1, column=0, sticky="w", pady=5)
        self.target = ttk.Entry(timing, width=8)
        self.target.grid(row=1, column=1, padx=8)
        ttk.Label(timing, text="预计单局").grid(row=1, column=2)
        self.duration = ttk.Spinbox(timing, from_=0, to=1440, width=6)
        self.duration.grid(row=1, column=3, padx=8)
        ttk.Label(timing, text="分钟").grid(row=1, column=4)
        ttk.Button(timing, text="应用时间", command=self._save_timer).grid(row=1, column=5, padx=8)
        self.window_label = ttk.Label(timing)
        self.window_label.grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 0))

        self.toggle = ttk.Button(outer, command=self._toggle_detection)
        self.toggle.pack(fill="x", ipady=8, pady=(4, 14))
        tabs = ttk.Notebook(outer)
        tabs.pack(fill="both", expand=True)
        reminder = ttk.Frame(tabs, padding=14)
        templates = ttk.Frame(tabs, padding=14)
        advanced = ttk.Frame(tabs, padding=14)
        tabs.add(reminder, text="提醒设置")
        tabs.add(templates, text="模板与识别")
        tabs.add(advanced, text="高级设置")
        self.mode = tk.StringVar()
        for label, value in (("应用弹窗 + 提示音", "popup"), ("Windows 通知 + 提示音", "native"),
                             ("仅 Windows 通知", "native_silent"), ("只响铃", "sound")):
            ttk.Radiobutton(reminder, text=label, value=value, variable=self.mode,
                            command=self._set_mode).pack(anchor="w", pady=2)
        text_row = ttk.Frame(reminder)
        text_row.pack(fill="x", pady=(8, 0))
        ttk.Label(text_row, text="标题").grid(row=0, column=0, sticky="w", pady=3)
        self.title_text = ttk.Entry(text_row, width=48)
        self.title_text.grid(row=0, column=1, padx=8, pady=3)
        ttk.Label(text_row, text="正文").grid(row=1, column=0, sticky="w", pady=3)
        self.body_text = ttk.Entry(text_row, width=48)
        self.body_text.grid(row=1, column=1, padx=8, pady=3)
        ttk.Button(text_row, text="保存文案", command=self._save_text).grid(row=0, column=2, rowspan=2)
        actions = ttk.Frame(reminder)
        actions.pack(fill="x", pady=8)
        for label, action in (("选择提示音", self.app._action_select_sound),
                              ("试听提示音", self.app._action_test_sound),
                              ("恢复默认音", self.app._action_reset_sound),
                              ("测试提醒", self.app._action_test_notify)):
            ttk.Button(actions, text=label, command=lambda a=action: self._act(a)).pack(side="left", padx=(0, 6))
        self.sound = ttk.Label(reminder)
        self.sound.pack(anchor="w")
        self.template_state = ttk.Label(templates)
        self.template_state.pack(anchor="w", pady=(0, 12))
        capture_row = ttk.Frame(templates)
        capture_row.pack(fill="x", pady=4)
        self.capture_choice = ttk.Combobox(capture_row, state="readonly", width=18)
        self.capture_choice.pack(side="left", padx=(0, 8))
        ttk.Button(capture_row, text="抓取模板", command=self._capture).pack(side="left", padx=(0, 8))
        ttk.Button(capture_row, text="重载模板", command=lambda: self._act(self.app._action_reload_templates)).pack(side="left")
        roi_row = ttk.Frame(templates)
        roi_row.pack(fill="x", pady=10)
        ttk.Button(roi_row, text="预览检测区域", command=lambda: self._act(self.app._action_preview_roi)).pack(side="left", padx=(0, 8))
        ttk.Button(roi_row, text="恢复默认区域", command=lambda: self._act(self.app._action_roi_reset)).pack(side="left")
        move = ttk.LabelFrame(templates, text="检测区域微调", padding=8)
        move.pack(anchor="w", fill="x")
        for label, direction in (("上移", "up"), ("下移", "down"), ("左移", "left"), ("右移", "right")):
            ttk.Button(move, text=label, command=lambda d=direction: self._act(self.app._action_roi_move(d))).pack(side="left", padx=3)
        size = ttk.Frame(templates)
        size.pack(anchor="w", pady=5)
        for label, which in (("加宽", "wider"), ("变窄", "narrower"), ("加高", "taller"), ("变矮", "shorter")):
            ttk.Button(size, text=label, command=lambda w=which: self._act(self.app._action_roi_resize(w))).pack(side="left", padx=3)
        for label, action in (("匹配阈值", self.app._action_set_threshold),
                              ("回落差值", self.app._action_set_hysteresis),
                              ("冷却时间", self.app._action_set_cooldown),
                              ("扫描间隔", self.app._action_set_interval)):
            ttk.Button(advanced, text=label, command=lambda a=action: self._act(a)).pack(anchor="w", pady=5)
        ttk.Button(outer, text="退出程序", command=lambda: self.app._action_quit(self.app._icon, None)).pack(anchor="e", pady=(10, 0))
        self.refresh()
        win.deiconify()

    def _act(self, action):
        action(self.app._icon, None)
        self.refresh()

    def _choose_game(self, _event):
        self.app._action_select_profile(self.app._profiles[self.game.current()].id)(self.app._icon, None)
        self.refresh()

    def _set_timer(self):
        self.app._timer_enabled = self.timer.get()
        self.app._persist()
        self.app._rebuild_menu()

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
        self.status.config(text="检测状态：正在检测" if running else "检测状态：已停止")
        self.toggle.config(text="停止检测" if running else "开始检测")
        self.timer.set(self.app._timer_enabled)
        if self.window.focus_get() not in (self.target, self.duration):
            self.target.delete(0, "end")
            self.target.insert(0, self.app._timer_target_time.strftime("%H:%M"))
            self.duration.delete(0, "end")
            self.duration.insert(0, str(self.app._game_duration_min(self.app._profile_base)))
        self.window_label.config(text=(self.app._timer_window_label() + "，下一次结算时提醒"
                                       if self.app._timer_enabled else "每次检测到新的结算画面时提醒"))
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
