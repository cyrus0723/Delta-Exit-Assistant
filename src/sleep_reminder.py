from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from profiles import GameProfile


DEFAULT_BED_TIME = "22:00"


def normalize_bed_time(value: str) -> str:
    return datetime.strptime(str(value).strip(), "%H:%M").strftime("%H:%M")


@dataclass
class SleepReminderConfig:
    enabled: bool = True
    bed_time: str = DEFAULT_BED_TIME
    stop_after_bed_time: bool = True
    auto_start_detection: bool = True
    title_tpl: str = "{game} 睡觉提醒"
    msg_tpl: str = "已经到休息时间了，这把结束后就下机。"


class SleepReminder:
    def __init__(self, cfg: SleepReminderConfig):
        self.cfg = cfg

    def parse_bed_time(self) -> time:
        try:
            normalized = normalize_bed_time(self.cfg.bed_time)
        except (TypeError, ValueError):
            normalized = DEFAULT_BED_TIME
        return datetime.strptime(normalized, "%H:%M").time()

    def reminder_window(self, profile: GameProfile, now: datetime | None = None) -> tuple[datetime, datetime]:
        now = now or datetime.now()
        bed_dt = datetime.combine(now.date(), self.parse_bed_time())
        lead_minutes = max(0, int(profile.sleep_lead_minutes))
        return bed_dt - timedelta(minutes=lead_minutes), bed_dt

    def is_active_now(self, profile: GameProfile, now: datetime | None = None) -> bool:
        if not self.cfg.enabled:
            return False

        now = now or datetime.now()
        start, end = self.reminder_window(profile, now)
        if self.cfg.stop_after_bed_time:
            return start <= now < end
        return now >= start

    def should_auto_start_detection(self, profile: GameProfile, now: datetime | None = None) -> bool:
        return self.cfg.auto_start_detection and self.is_active_now(profile, now)
