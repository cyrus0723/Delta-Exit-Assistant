from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional


DEFAULT_ESTIMATED_DURATION_MIN = 30
MAX_ESTIMATED_DURATION_MIN = 24 * 60


def parse_target_time(value: str) -> Optional[time]:
    """Parse a user-entered 24-hour clock value such as ``08:00``."""
    parts = value.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = (int(part) for part in parts)
    except ValueError:
        return None
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    return time(hour=hour, minute=minute)


def normalize_duration_min(value: object, default: int = DEFAULT_ESTIMATED_DURATION_MIN) -> int:
    """Return a valid game-duration estimate, falling back for old/bad config."""
    try:
        duration = int(float(value))
    except (TypeError, ValueError):
        duration = default
    return max(0, min(MAX_ESTIMATED_DURATION_MIN, duration))


@dataclass(frozen=True)
class AlertWindow:
    target_at: datetime
    opens_at: datetime


def alert_window(now: datetime, target: time, duration_min: int) -> AlertWindow:
    """Build today's alert window from the configured clock time and duration."""
    target_at = datetime.combine(now.date(), target)
    opens_at = target_at - timedelta(minutes=normalize_duration_min(duration_min))
    return AlertWindow(target_at=target_at, opens_at=opens_at)


def is_alert_window_open(now: datetime, target: time, duration_min: int) -> bool:
    """Allow alerts from the calculated start time onward."""
    return now >= alert_window(now, target, duration_min).opens_at
