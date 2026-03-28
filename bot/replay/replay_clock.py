"""
replay_clock.py — Precision clock driven by TWS replay bar timestamps.

Drop-in replacement for bot/clock.py:Clock. Instead of NTP-corrected wall
clock, time advances when the engine feeds bar/tick timestamps from TWS
Market Data Replay.

EOD watchdog: if no bar update arrives for 30 seconds of wall-clock time,
force-advance to session_end (16:00 ET) so the normal EOD sequence fires.
This handles the gap after the last RTH bar (~15:55) when TWS stops sending
hasNewBar events.
"""

import time as _time
from datetime import datetime, time, timezone

import pytz

NY_TZ = pytz.timezone("America/New_York")

# Default session end — force-advanced to when watchdog fires
_SESSION_END = time(16, 0)
_WATCHDOG_TIMEOUT_SECS = 30.0


class ReplayClock:
    """
    Clock whose time is driven by TWS replay bar timestamps.

    Usage:
        clock = ReplayClock()
        clock.update(bar_time_et)           # called on each bar callback
        clock.update_from_ib(ib_utc_time)   # startup init from reqCurrentTime()
        now = clock.now()                    # returns replay time in ET
    """

    def __init__(self, logger=None, session_end=None):
        self._logger = logger
        self._current_et = None           # Latest replay time (ET-aware)
        self._last_update_wall = 0.0      # Wall-clock time of last update()
        self._session_end = session_end or _SESSION_END
        self._eod_forced = False
        # Match Clock interface attributes
        self._offset = 0.0
        self._sync_count = 1              # Pretend synced
        self._lock = None

    # ── Primary interface (matches Clock) ────────────────────────────────

    def now(self):
        """
        Current replay time in Eastern Time.

        If no bar has arrived for >30s wall-clock, force-return session_end
        so the engine's EOD checks fire.
        """
        if self._current_et is None:
            # Fallback before first bar — return epoch to prevent crashes
            return NY_TZ.localize(datetime(2000, 1, 1))

        # Watchdog: force session end if no bars arriving
        if not self._eod_forced and self._last_update_wall > 0:
            wall_elapsed = _time.time() - self._last_update_wall
            if wall_elapsed > _WATCHDOG_TIMEOUT_SECS:
                self._force_session_end()

        return self._current_et

    def now_utc(self):
        """Current replay time in UTC."""
        return self.now().astimezone(timezone.utc)

    def today_str(self):
        """Today's date string in replay time: 'YYYY-MM-DD'."""
        return self.now().strftime("%Y-%m-%d")

    def sync(self):
        """No-op — replay clock doesn't use NTP."""
        return True

    def check_resync(self):
        """No-op."""
        pass

    async def validate_with_ib(self, ib_connection):
        """No-op — replay clock is authoritative."""
        pass

    # ── Replay-specific methods ──────────────────────────────────────────

    def update(self, bar_time_et):
        """
        Advance the replay clock to a bar's timestamp.

        Called by the engine on every bar callback (5min, 1min).
        Only advances forward — ignores stale/out-of-order updates.
        """
        if bar_time_et is None:
            return
        if self._current_et is not None and bar_time_et <= self._current_et:
            return  # Don't go backwards
        self._current_et = bar_time_et
        self._last_update_wall = _time.time()
        self._eod_forced = False

    def update_from_ib(self, ib_utc_time):
        """
        Initialize from IB's reqCurrentTime() result (UTC datetime).
        Called once at startup to set the replay date before any bars arrive.
        """
        if ib_utc_time is None:
            return
        if hasattr(ib_utc_time, 'tzinfo') and ib_utc_time.tzinfo is not None:
            et = ib_utc_time.astimezone(NY_TZ)
        else:
            et = pytz.utc.localize(ib_utc_time).astimezone(NY_TZ)
        self._current_et = et
        self._last_update_wall = _time.time()

    # ── Properties (match Clock interface) ───────────────────────────────

    @property
    def offset_ms(self):
        return 0

    @property
    def is_synced(self):
        return True

    @property
    def sync_age_seconds(self):
        return 0.0

    def __repr__(self):
        t = self._current_et.strftime("%H:%M:%S") if self._current_et else "unset"
        return f"ReplayClock(replay_time={t}, eod_forced={self._eod_forced})"

    # ── Internal ─────────────────────────────────────────────────────────

    def _force_session_end(self):
        """Force clock to session_end when watchdog fires."""
        if self._current_et is None:
            return
        replay_date = self._current_et.date()
        forced = NY_TZ.localize(datetime.combine(replay_date, self._session_end))
        self._current_et = forced
        self._eod_forced = True
        if self._logger:
            self._logger.log(
                "replay_clock_eod_forced",
                forced_to=str(self._session_end),
                reason="no bar update for 30s",
            )
