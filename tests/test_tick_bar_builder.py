"""Tests for TickBarBuilder — 5-min OHLC construction from tick-by-tick data."""

import pytest
from datetime import datetime, time

import pytz

from bot.strategy.tick_bar_builder import TickBarBuilder, RTH_START, RTH_END

NY = pytz.timezone("America/New_York")


def _et(h, m, s=0, us=0):
    """Helper: create an ET-aware datetime on 2026-03-22."""
    return NY.localize(datetime(2026, 3, 22, h, m, s, us))


class TestBarWindowStart:
    """Test the 5-min boundary flooring logic."""

    def test_exact_boundary(self):
        b = TickBarBuilder()
        assert b.bar_window_start(_et(10, 0)) == _et(10, 0)

    def test_mid_window(self):
        b = TickBarBuilder()
        assert b.bar_window_start(_et(10, 3, 45)) == _et(10, 0)

    def test_next_boundary(self):
        b = TickBarBuilder()
        assert b.bar_window_start(_et(10, 5)) == _et(10, 5)

    def test_one_second_before_boundary(self):
        b = TickBarBuilder()
        assert b.bar_window_start(_et(10, 4, 59)) == _et(10, 0)

    def test_session_start(self):
        b = TickBarBuilder()
        assert b.bar_window_start(_et(9, 30)) == _et(9, 30)

    def test_session_end_boundary(self):
        b = TickBarBuilder()
        assert b.bar_window_start(_et(15, 55)) == _et(15, 55)


class TestOnTick:
    """Test the core tick processing and bar emission logic."""

    def test_first_tick_returns_none(self):
        """First tick initializes accumulator, no bar to emit."""
        b = TickBarBuilder()
        result = b.on_tick(19500.0, _et(10, 0, 1))
        assert result is None
        assert b.tick_count == 1
        assert b.current_window == _et(10, 0)

    def test_same_window_returns_none(self):
        """Ticks within the same window don't emit a bar."""
        b = TickBarBuilder()
        b.on_tick(19500.0, _et(10, 0, 1))
        result = b.on_tick(19510.0, _et(10, 2, 30))
        assert result is None
        assert b.tick_count == 2

    def test_boundary_crossing_emits_bar(self):
        """First tick in a new window emits the completed previous bar."""
        b = TickBarBuilder()
        b.on_tick(19500.0, _et(10, 0, 1))    # First tick
        b.on_tick(19520.0, _et(10, 1, 0))    # High
        b.on_tick(19490.0, _et(10, 3, 0))    # Low
        b.on_tick(19510.0, _et(10, 4, 59))   # Close

        # Tick crossing into next 5-min window
        completed = b.on_tick(19515.0, _et(10, 5, 0))

        assert completed is not None
        assert completed["open"] == 19500.0
        assert completed["high"] == 19520.0
        assert completed["low"] == 19490.0
        assert completed["close"] == 19510.0
        assert completed["date"] == _et(10, 0)

    def test_new_window_starts_fresh(self):
        """After boundary crossing, new window tracks new ticks."""
        b = TickBarBuilder()
        b.on_tick(19500.0, _et(10, 0, 1))
        b.on_tick(19510.0, _et(10, 4, 59))

        # Cross into next window
        b.on_tick(19515.0, _et(10, 5, 0))

        # New window should track from the crossing tick
        assert b.current_window == _et(10, 5)
        assert b.tick_count == 1

    def test_ohlc_tracking(self):
        """OHLC should track open=first, high=max, low=min, close=last."""
        b = TickBarBuilder()
        prices = [100, 105, 98, 110, 95, 102]
        for i, p in enumerate(prices):
            b.on_tick(float(p), _et(10, 0, i))

        # Trigger emission by crossing into next window
        bar = b.on_tick(101.0, _et(10, 5, 0))

        assert bar["open"] == 100.0
        assert bar["high"] == 110.0
        assert bar["low"] == 95.0
        assert bar["close"] == 102.0

    def test_single_tick_bar(self):
        """A window with only 1 tick produces a valid bar."""
        b = TickBarBuilder()
        b.on_tick(19500.0, _et(10, 0, 1))

        bar = b.on_tick(19510.0, _et(10, 5, 0))

        assert bar["open"] == 19500.0
        assert bar["high"] == 19500.0
        assert bar["low"] == 19500.0
        assert bar["close"] == 19500.0

    def test_gap_skips_empty_windows(self):
        """A gap in ticks emits the last accumulated window, skips empty ones."""
        b = TickBarBuilder()
        b.on_tick(19500.0, _et(10, 0, 1))
        b.on_tick(19510.0, _et(10, 2, 0))

        # Jump forward 15 minutes (3 empty windows)
        bar = b.on_tick(19550.0, _et(10, 15, 0))

        assert bar is not None
        assert bar["date"] == _et(10, 0)  # Emits the 10:00 window
        assert bar["close"] == 19510.0

    def test_exact_boundary_tick_belongs_to_new_window(self):
        """Tick at exactly 10:05:00.000 belongs to 10:05-10:10, not 10:00-10:05."""
        b = TickBarBuilder()
        b.on_tick(19500.0, _et(10, 0, 1))

        bar = b.on_tick(19510.0, _et(10, 5, 0, 0))  # Exactly at boundary

        assert bar is not None
        assert bar["date"] == _et(10, 0)
        assert b.current_window == _et(10, 5)


class TestRTHFiltering:
    """Test that ticks outside RTH are ignored."""

    def test_pre_market_tick_ignored(self):
        b = TickBarBuilder()
        result = b.on_tick(19500.0, _et(9, 29, 59))
        assert result is None
        assert b.tick_count == 0

    def test_post_market_tick_ignored(self):
        b = TickBarBuilder()
        b.on_tick(19500.0, _et(10, 0, 1))  # Valid tick
        result = b.on_tick(19510.0, _et(16, 0, 0))
        assert result is None

    def test_rth_start_tick_accepted(self):
        b = TickBarBuilder()
        result = b.on_tick(19500.0, _et(9, 30, 0))
        assert result is None  # First tick, no bar to emit
        assert b.tick_count == 1  # But tick was accepted

    def test_last_rth_tick_accepted(self):
        b = TickBarBuilder()
        result = b.on_tick(19500.0, _et(15, 59, 59))
        assert result is None
        assert b.tick_count == 1


class TestReset:
    """Test that reset clears all state."""

    def test_reset_clears_state(self):
        b = TickBarBuilder()
        b.on_tick(19500.0, _et(10, 0, 1))
        b.on_tick(19510.0, _et(10, 2, 0))

        b.reset()

        assert b.current_window is None
        assert b.tick_count == 0

    def test_first_tick_after_reset_returns_none(self):
        b = TickBarBuilder()
        b.on_tick(19500.0, _et(10, 0, 1))
        b.on_tick(19510.0, _et(10, 2, 0))

        b.reset()

        # First tick after reset should behave like first-ever tick
        result = b.on_tick(19520.0, _et(10, 5, 0))
        assert result is None
        assert b.tick_count == 1
