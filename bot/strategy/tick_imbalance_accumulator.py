"""
tick_imbalance_accumulator.py — Build 5-min aggressor imbalance bars from live ticks.

Uses the tick rule to classify aggressor side from IB's AllLast tick data:
    price > prev → buyer-initiated  → +size to buy_vol
    price < prev → seller-initiated → +size to sell_vol
    price == prev → carry forward last classification

Validated against Databento native CME side field:
    HFOIV correlation: 0.970 (19 days, 2020-2026)
    Gate p70 agreement: 95.9%

Time range: 08:30-16:00 ET (includes 1h pre-RTH for HFOIV rolling warm-up).
Unlike TickBarBuilder, this does NOT filter to RTH-only.
"""

from datetime import time


ETH_START = time(8, 30)
RTH_END = time(16, 0)


class TickImbalanceAccumulator:
    """Accumulates tick-rule-classified buy/sell volume into 5-min imbalance bars.

    Usage:
        acc = TickImbalanceAccumulator()
        for tick in ticks:
            bar = acc.on_tick(tick.price, tick.size, tick_time_et)
            if bar is not None:
                hfoiv_gate.update(bar["bar_minutes"], bar["imbalance"])
    """

    def __init__(self, bar_minutes=5):
        self._bar_minutes = bar_minutes
        self._current_window = None
        self._buy_vol = 0
        self._sell_vol = 0
        self._last_price = None
        self._last_side = 0  # +1 = buy, -1 = sell, 0 = unknown

    def on_tick(self, price, size, tick_time_et):
        """Feed a trade tick. Returns completed imbalance bar if boundary crossed.

        Args:
            price: Trade price (float). Must be finite.
            size: Trade size in contracts (int). Must be > 0.
            tick_time_et: Tick timestamp in ET (timezone-aware datetime).

        Returns:
            dict {bar_minutes, imbalance, buy_vol, sell_vol} if bar completed,
            else None.
        """
        # Validate inputs — corrupted tick data must never pollute the gate
        if size is None or size <= 0:
            return None
        if price is None or price != price:  # NaN check without math import
            return None

        t = tick_time_et.time() if hasattr(tick_time_et, 'time') else tick_time_et
        if t < ETH_START or t >= RTH_END:
            return None

        # Tick rule classification
        side = self._classify(price)
        self._last_price = price

        window = self._bar_window_start(tick_time_et)

        if self._current_window is None:
            self._current_window = window
            self._accumulate(side, size)
            return None

        if window != self._current_window:
            completed = self._emit()
            self._current_window = window
            self._buy_vol = 0
            self._sell_vol = 0
            self._accumulate(side, size)
            return completed

        self._accumulate(side, size)
        return None

    def reset(self):
        """Clear all state. Call on session start or reconnect."""
        self._current_window = None
        self._buy_vol = 0
        self._sell_vol = 0
        self._last_price = None
        self._last_side = 0

    def _classify(self, price):
        """Tick rule: classify aggressor side from price movement.

        Returns +1 (buyer), -1 (seller), or 0 (unknown/first tick).
        """
        if self._last_price is None:
            return 0

        if price > self._last_price:
            self._last_side = 1
        elif price < self._last_price:
            self._last_side = -1
        # Equal price: carry forward _last_side

        return self._last_side

    def _accumulate(self, side, size):
        if side > 0:
            self._buy_vol += size
        elif side < 0:
            self._sell_vol += size
        # side == 0 (first tick): not attributed to either side

    def _emit(self):
        """Return completed imbalance bar for the current window."""
        if self._current_window is None:
            return None
        bar_minutes = self._current_window.hour * 60 + self._current_window.minute
        return {
            "bar_minutes": bar_minutes,
            "imbalance": self._buy_vol - self._sell_vol,
            "buy_vol": self._buy_vol,
            "sell_vol": self._sell_vol,
        }

    def _bar_window_start(self, dt):
        """Floor datetime to nearest bar_minutes boundary."""
        minute = (dt.minute // self._bar_minutes) * self._bar_minutes
        return dt.replace(minute=minute, second=0, microsecond=0)
