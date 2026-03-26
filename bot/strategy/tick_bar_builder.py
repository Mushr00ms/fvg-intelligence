"""
tick_bar_builder.py — Build 5-min OHLC bars from tick-by-tick data in real time.

Detects bar boundaries from tick timestamps. When a tick from the next
5-min window arrives, emits the tick-OHLC for the just-completed window.
The caller merges this with IB's keepUpToDate bar for authoritative OHLC.

This eliminates the ~5 second delay of keepUpToDate bar subscriptions:
detection fires on the first tick of bar4, not 5 seconds later.
"""

from datetime import datetime, time


RTH_START = time(9, 30)
RTH_END = time(16, 0)


class TickBarBuilder:
    """Accumulates tick-by-tick OHLC within 5-min windows.

    Usage:
        builder = TickBarBuilder()
        for tick in ticks:
            completed = builder.on_tick(tick.price, tick_time_et)
            if completed is not None:
                # A 5-min bar just closed — merge with IB bar and detect FVG
                process(completed)
    """

    def __init__(self, bar_minutes=5):
        self._bar_minutes = bar_minutes
        self._current_window = None  # datetime: start of current 5-min window
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._tick_count = 0

    def on_tick(self, price, tick_time_et):
        """Feed a trade tick. Returns completed tick-OHLC dict if a bar boundary was crossed.

        Args:
            price: Trade price (float).
            tick_time_et: Tick timestamp in ET (timezone-aware datetime).

        Returns:
            dict with {open, high, low, close, date} if a bar just completed, else None.
            Returns None for the first tick of the session (no prior bar to emit).
        """
        # Filter outside RTH
        t = tick_time_et.time() if hasattr(tick_time_et, 'time') else tick_time_et
        if t < RTH_START or t >= RTH_END:
            return None

        window = self.bar_window_start(tick_time_et)

        if self._current_window is None:
            # First tick of session — initialize, no bar to emit
            self._start_new_window(window, price)
            return None

        if window != self._current_window:
            # Bar boundary crossed — emit the completed bar
            completed = self._emit_bar()
            self._start_new_window(window, price)
            return completed

        # Same window — update running OHLC
        self._update(price)
        return None

    def bar_window_start(self, dt):
        """Floor datetime to the nearest bar_minutes boundary.

        Examples (5-min bars):
            10:00:00 -> 10:00:00
            10:03:45 -> 10:00:00
            10:05:00 -> 10:05:00  (belongs to new window)
        """
        minute = (dt.minute // self._bar_minutes) * self._bar_minutes
        return dt.replace(minute=minute, second=0, microsecond=0)

    def reset(self):
        """Clear all state. Call on session start or reconnect."""
        self._current_window = None
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._tick_count = 0

    @property
    def current_window(self):
        """The start time of the window currently being accumulated."""
        return self._current_window

    @property
    def tick_count(self):
        """Number of ticks in the current window."""
        return self._tick_count

    def _start_new_window(self, window, price):
        self._current_window = window
        self._open = price
        self._high = price
        self._low = price
        self._close = price
        self._tick_count = 1

    def _update(self, price):
        if price > self._high:
            self._high = price
        if price < self._low:
            self._low = price
        self._close = price
        self._tick_count += 1

    def _emit_bar(self):
        """Return the completed bar dict for the current window, or None if empty."""
        if self._current_window is None or self._tick_count == 0:
            return None
        return {
            "open": self._open,
            "high": self._high,
            "low": self._low,
            "close": self._close,
            "date": self._current_window,
        }
