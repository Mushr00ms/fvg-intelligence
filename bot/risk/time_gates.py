"""
time_gates.py — Session time management and EOD scheduling.

Controls when entries are allowed and schedules end-of-day actions.
All times are in Eastern Time (America/New_York).
"""

from datetime import datetime, time, timedelta

import pytz

NY_TZ = pytz.timezone("America/New_York")


class TimeGates:
    """
    Session clock: controls entry windows and schedules EOD actions.

    Uses an injected Clock for NTP-corrected time. Falls back to system
    clock if no clock is provided (for testing).
    """

    def __init__(self, config, clock=None):
        self.session_start = _parse_time(config.session_start)      # 09:30
        self.session_end = _parse_time(config.session_end)           # 16:00
        self.last_entry = _parse_time(config.last_entry_time)        # 15:30
        self.cancel_unfilled = _parse_time(config.cancel_unfilled_time)  # 15:50
        self.flatten_time = _parse_time(config.flatten_time)         # 15:55
        self._clock = clock

    def _now(self):
        """Get current ET time from the injected clock or system clock."""
        if self._clock is not None:
            return self._clock.now()
        return datetime.now(NY_TZ)

    def can_enter(self, now=None):
        """
        Check if new entries are allowed at the given time.
        Entries allowed between session_start (09:30) and last_entry (15:30) ET.

        Returns:
            (allowed: bool, reason: str)
        """
        now = now or self._now()
        t = now.time() if hasattr(now, "time") else now

        if t < self.session_start:
            return False, f"Before session start ({self.session_start})"
        if t >= self.last_entry:
            return False, f"Past last entry time ({self.last_entry})"
        return True, ""

    def is_session_active(self, now=None):
        """Check if we're within the trading session."""
        now = now or self._now()
        t = now.time() if hasattr(now, "time") else now
        return self.session_start <= t < self.session_end

    def seconds_until(self, target_time, now=None):
        """
        Calculate seconds from now until a target time today (ET).
        Returns negative if target has already passed.
        """
        now = now or self._now()
        today = now.date()

        target_dt = NY_TZ.localize(datetime.combine(today, target_time))
        delta = (target_dt - now).total_seconds()
        return delta

    def get_eod_schedule(self, now=None):
        """
        Return a list of (seconds_from_now, action_name) for EOD actions.
        Only includes actions that haven't passed yet.

        Actions:
            "cancel_unfilled" at 15:50 ET
            "flatten_all" at 15:55 ET
            "session_end" at 16:00 ET
        """
        now = now or self._now()
        actions = [
            (self.cancel_unfilled, "cancel_unfilled"),
            (self.flatten_time, "flatten_all"),
            (self.session_end, "session_end"),
        ]

        schedule = []
        for action_time, action_name in actions:
            delay = self.seconds_until(action_time, now)
            if delay > 0:
                schedule.append((delay, action_name))

        return schedule

    def now_et(self):
        """Return current time in ET (NTP-corrected if clock available)."""
        return self._now()


def _parse_time(time_str):
    """Parse a time string like '09:30' or '15:45' to a time object."""
    if isinstance(time_str, time):
        return time_str
    parts = time_str.split(":")
    return time(int(parts[0]), int(parts[1]))
