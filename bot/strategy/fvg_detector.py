"""
fvg_detector.py — Real-time FVG detection for the trading bot.

check_fvg_3bars() extracts the core logic from logic/utils/fvg_detection.py
and operates on 3 bar dicts instead of a full DataFrame.

ActiveFVGManager tracks active (un-mitigated) FVGs for the current session.
"""

from collections import deque
from datetime import datetime, time

import pytz

from bot.state.trade_state import FVGRecord, _new_id

NY_TZ = pytz.timezone("America/New_York")

# Default minimum FVG size in points (NQ)
DEFAULT_MIN_SIZE = 0.25


def check_fvg_3bars(bar1, bar2, bar3, min_size=DEFAULT_MIN_SIZE):
    """
    Check if 3 consecutive bars form a Fair Value Gap.

    Each bar is a dict with keys: open, high, low, close, date (ISO string or datetime).

    Logic (from logic/utils/fvg_detection.py lines 32-79):
      Bullish FVG: bar3.low > bar1.high (gap up)
      Bearish FVG: bar3.high < bar1.low (gap down)
      Size must be >= min_size

    Returns:
        FVGRecord if FVG found, None otherwise.
    """
    first_high = bar1["high"]
    first_low = bar1["low"]
    first_open = bar1["open"]
    middle_open = bar2["open"]
    middle_low = bar2["low"]
    middle_high = bar2["high"]
    third_low = bar3["low"]
    third_high = bar3["high"]

    time_candle1 = str(bar1["date"])
    time_candle2 = str(bar2["date"])
    time_candle3 = str(bar3["date"])

    # Bullish FVG: gap between first candle high and third candle low
    if third_low > first_high:
        fvg_size = third_low - first_high
        if fvg_size >= min_size:
            return FVGRecord(
                fvg_id=_new_id(),
                fvg_type="bullish",
                zone_low=first_high,        # y0
                zone_high=third_low,         # y1
                time_candle1=time_candle1,
                time_candle2=time_candle2,
                time_candle3=time_candle3,
                middle_open=middle_open,
                middle_low=middle_low,
                middle_high=middle_high,
                first_open=first_open,
                time_period="",              # Set by ActiveFVGManager
                formation_date="",           # Set by ActiveFVGManager
            )

    # Bearish FVG: gap between third candle high and first candle low
    elif third_high < first_low:
        fvg_size = first_low - third_high
        if fvg_size >= min_size:
            return FVGRecord(
                fvg_id=_new_id(),
                fvg_type="bearish",
                zone_low=third_high,         # y0
                zone_high=first_low,          # y1
                time_candle1=time_candle1,
                time_candle2=time_candle2,
                time_candle3=time_candle3,
                middle_open=middle_open,
                middle_low=middle_low,
                middle_high=middle_high,
                first_open=first_open,
                time_period="",
                formation_date="",
            )

    return None


def _assign_time_period(candle3_time, intervals):
    """
    Map candle3's timestamp to a 30-minute interval string.
    intervals: list of (start_time, end_time) tuples.
    Returns string like "10:30-11:00" or None.
    """
    if isinstance(candle3_time, str):
        candle3_time = datetime.fromisoformat(candle3_time)

    t = candle3_time.time() if hasattr(candle3_time, "time") else candle3_time

    for start, end in intervals:
        if start <= t < end:
            return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
    return None


# Standard 30-minute intervals for US session (09:30 - 16:00)
SESSION_INTERVALS = []
_h, _m = 9, 30
while _h < 16:
    start = time(_h, _m)
    _m += 30
    if _m >= 60:
        _h += 1
        _m -= 60
    end = time(_h, _m)
    SESSION_INTERVALS.append((start, end))


class ActiveFVGManager:
    """
    Manages the set of active (un-mitigated) FVGs for the current session.

    Responsibilities:
    - On each completed 5min bar: check last 3 bars for new FVG
    - Early-skip FVGs whose time_period has no strategy cell
    - Expire all FVGs at session end (16:00 ET)
    - Remove FVG on mitigation
    """

    def __init__(self, strategy_loader, min_fvg_size=DEFAULT_MIN_SIZE, logger=None):
        self._strategy = strategy_loader
        self._min_size = min_fvg_size
        self._logger = logger
        self._active = {}           # fvg_id -> FVGRecord
        self._recent_bars = deque(maxlen=10)

    def on_5min_bar(self, bar):
        """
        Called when a new 5min bar completes.

        Args:
            bar: dict with {open, high, low, close, date}

        Returns:
            FVGRecord if a new FVG was detected and added, None otherwise.
        """
        self._recent_bars.append(bar)

        if len(self._recent_bars) < 3:
            return None

        # Check last 3 bars
        bar1 = self._recent_bars[-3]
        bar2 = self._recent_bars[-2]
        bar3 = self._recent_bars[-1]

        fvg = check_fvg_3bars(bar1, bar2, bar3, self._min_size)
        if fvg is None:
            return None

        # Assign time period
        fvg.time_period = _assign_time_period(bar3["date"], SESSION_INTERVALS)
        if not fvg.time_period:
            return None  # Outside session intervals

        # Set formation date
        if isinstance(bar3["date"], str):
            fvg.formation_date = bar3["date"][:10]
        elif hasattr(bar3["date"], "strftime"):
            fvg.formation_date = bar3["date"].strftime("%Y-%m-%d")

        # Early-skip: check if ANY risk range in this time period has a strategy cell
        # We can't know the exact risk yet (depends on setup), but we can check if the
        # time period has any cells at all
        has_any_cell = False
        bins = [5, 10, 15, 20, 25, 30, 40, 80]
        for i in range(len(bins) - 1):
            risk_range = f"{bins[i]}-{bins[i+1]}"
            if self._strategy._lookup.get((fvg.time_period, risk_range)):
                has_any_cell = True
                break

        if not has_any_cell:
            return None  # No strategy cells for this time period

        # Add to active FVGs
        self._active[fvg.fvg_id] = fvg

        if self._logger:
            self._logger.log(
                "fvg_detected",
                fvg_id=fvg.fvg_id,
                type=fvg.fvg_type,
                zone=[fvg.zone_low, fvg.zone_high],
                size=round(fvg.zone_high - fvg.zone_low, 2),
                time_period=fvg.time_period,
                middle_low=fvg.middle_low,
                middle_high=fvg.middle_high,
            )

        return fvg

    def remove(self, fvg_id):
        """Remove an FVG (after mitigation or expiry)."""
        return self._active.pop(fvg_id, None)

    def expire_all(self):
        """Expire all active FVGs (session end). Returns list of expired IDs."""
        expired = list(self._active.keys())
        self._active.clear()
        return expired

    @property
    def active_fvgs(self):
        """Return list of active FVGRecords."""
        return list(self._active.values())

    @property
    def active_count(self):
        return len(self._active)

    def restore(self, fvg_records):
        """Restore active FVGs from state (crash recovery)."""
        for fvg in fvg_records:
            if not fvg.is_mitigated:
                self._active[fvg.fvg_id] = fvg
