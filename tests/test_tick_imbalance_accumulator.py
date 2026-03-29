"""
Tests for TickImbalanceAccumulator (bot/strategy/tick_imbalance_accumulator.py).

Covers: tick rule classification, ETH/RTH time filtering, bar boundary emission,
reset, and manual imbalance calculation verification.
"""

from datetime import datetime, timezone, timedelta
import pytest

from bot.strategy.tick_imbalance_accumulator import TickImbalanceAccumulator

# ET-aware helper
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except ImportError:
    import pytz
    _ET = pytz.timezone("America/New_York")


def _et(h, m, s=0):
    """Build an ET-aware datetime on a fixed date."""
    return datetime(2025, 3, 20, h, m, s, tzinfo=_ET)


class TestTickRuleClassification:

    def test_first_tick_unclassified(self):
        """First tick has no prior price — size not attributed."""
        acc = TickImbalanceAccumulator()
        acc.on_tick(20000.0, 10, _et(9, 30, 1))
        # Feed a second tick in same window to inspect state
        acc.on_tick(20000.0, 5, _et(9, 30, 2))
        # First tick (10 contracts) was unclassified, second (5) carries forward 0
        # Neither buy nor sell should have 10 (the first tick's size)
        # buy_vol and sell_vol together should be < 15
        assert acc._buy_vol + acc._sell_vol < 15

    def test_uptick_classified_as_buy(self):
        acc = TickImbalanceAccumulator()
        acc.on_tick(20000.0, 5, _et(9, 30, 0))   # first tick, unclassified
        acc.on_tick(20001.0, 10, _et(9, 30, 1))   # uptick → buyer
        assert acc._buy_vol == 10
        assert acc._sell_vol == 0

    def test_downtick_classified_as_sell(self):
        acc = TickImbalanceAccumulator()
        acc.on_tick(20000.0, 5, _et(9, 30, 0))
        acc.on_tick(19999.0, 10, _et(9, 30, 1))   # downtick → seller
        assert acc._buy_vol == 0
        assert acc._sell_vol == 10

    def test_equal_price_carries_forward(self):
        acc = TickImbalanceAccumulator()
        acc.on_tick(20000.0, 5, _et(9, 30, 0))    # first, unclassified
        acc.on_tick(20001.0, 10, _et(9, 30, 1))   # uptick → buy
        acc.on_tick(20001.0, 8, _et(9, 30, 2))    # equal → carry buy
        assert acc._buy_vol == 18  # 10 + 8
        assert acc._sell_vol == 0

    def test_direction_reversal(self):
        acc = TickImbalanceAccumulator()
        acc.on_tick(20000.0, 1, _et(9, 30, 0))    # first
        acc.on_tick(20001.0, 10, _et(9, 30, 1))   # up → buy
        acc.on_tick(19999.0, 20, _et(9, 30, 2))   # down → sell
        assert acc._buy_vol == 10
        assert acc._sell_vol == 20


class TestTimeFiltering:

    def test_eth_ticks_accepted(self):
        """Ticks at 08:30 ET should be accepted (not filtered)."""
        acc = TickImbalanceAccumulator()
        result = acc.on_tick(20000.0, 5, _et(8, 30, 0))
        assert acc._current_window is not None  # was accepted

    def test_pre_eth_ticks_ignored(self):
        """Ticks before 08:30 ET should be ignored."""
        acc = TickImbalanceAccumulator()
        result = acc.on_tick(20000.0, 5, _et(8, 29, 59))
        assert result is None
        assert acc._current_window is None

    def test_post_rth_ticks_ignored(self):
        """Ticks at or after 16:00 ET should be ignored."""
        acc = TickImbalanceAccumulator()
        result = acc.on_tick(20000.0, 5, _et(16, 0, 0))
        assert result is None
        assert acc._current_window is None


class TestBarBoundary:

    def test_same_window_returns_none(self):
        acc = TickImbalanceAccumulator()
        assert acc.on_tick(20000.0, 5, _et(9, 30, 0)) is None
        assert acc.on_tick(20001.0, 5, _et(9, 30, 30)) is None
        assert acc.on_tick(20002.0, 5, _et(9, 34, 59)) is None

    def test_boundary_crossing_emits_bar(self):
        acc = TickImbalanceAccumulator()
        acc.on_tick(20000.0, 5, _et(9, 30, 0))
        acc.on_tick(20001.0, 10, _et(9, 31, 0))   # uptick → buy 10
        acc.on_tick(19999.0, 3, _et(9, 32, 0))    # downtick → sell 3

        # Cross into next window
        bar = acc.on_tick(20000.0, 1, _et(9, 35, 0))
        assert bar is not None
        assert bar["bar_minutes"] == 570  # 9*60+30
        assert bar["buy_vol"] == 10
        assert bar["sell_vol"] == 3
        assert bar["imbalance"] == 7  # 10 - 3

    def test_bar_minutes_correct_for_eth(self):
        acc = TickImbalanceAccumulator()
        acc.on_tick(20000.0, 5, _et(8, 30, 0))
        acc.on_tick(20001.0, 5, _et(8, 31, 0))
        bar = acc.on_tick(20002.0, 5, _et(8, 35, 0))
        assert bar is not None
        assert bar["bar_minutes"] == 510  # 8*60+30

    def test_multiple_bars_sequence(self):
        acc = TickImbalanceAccumulator()
        # Bar 1: 09:30-09:35
        acc.on_tick(20000.0, 5, _et(9, 30, 0))
        acc.on_tick(20001.0, 10, _et(9, 33, 0))
        # Bar 2: 09:35-09:40
        bar1 = acc.on_tick(20000.0, 5, _et(9, 35, 0))
        acc.on_tick(19999.0, 20, _et(9, 37, 0))
        # Bar 3: 09:40-09:45
        bar2 = acc.on_tick(20000.0, 5, _et(9, 40, 0))

        assert bar1["bar_minutes"] == 570
        assert bar1["imbalance"] == 10  # one uptick of 10
        assert bar2["bar_minutes"] == 575
        assert bar2["sell_vol"] == 25  # 5 (crossing tick downtick) + 20


class TestReset:

    def test_reset_clears_all_state(self):
        acc = TickImbalanceAccumulator()
        acc.on_tick(20000.0, 5, _et(9, 30, 0))
        acc.on_tick(20001.0, 10, _et(9, 31, 0))

        acc.reset()
        assert acc._current_window is None
        assert acc._buy_vol == 0
        assert acc._sell_vol == 0
        assert acc._last_price is None
        assert acc._last_side == 0

    def test_after_reset_first_tick_unclassified(self):
        acc = TickImbalanceAccumulator()
        acc.on_tick(20000.0, 5, _et(9, 30, 0))
        acc.on_tick(20001.0, 10, _et(9, 31, 0))
        acc.reset()

        # After reset, first tick is unclassified again
        acc.on_tick(19000.0, 100, _et(9, 30, 0))
        assert acc._buy_vol == 0
        assert acc._sell_vol == 0


class TestImbalanceCalc:

    def test_manual_imbalance_calculation(self):
        """Verify imbalance = buy_vol - sell_vol exactly."""
        acc = TickImbalanceAccumulator()
        acc.on_tick(20000.0, 1, _et(10, 0, 0))    # first, unclassified
        acc.on_tick(20001.0, 100, _et(10, 0, 1))  # up → buy 100
        acc.on_tick(20001.0, 50, _et(10, 0, 2))   # equal → buy 50
        acc.on_tick(19999.0, 30, _et(10, 0, 3))   # down → sell 30
        acc.on_tick(19999.0, 20, _et(10, 0, 4))   # equal → sell 20

        bar = acc.on_tick(20000.0, 1, _et(10, 5, 0))
        assert bar["buy_vol"] == 150   # 100 + 50
        assert bar["sell_vol"] == 50   # 30 + 20
        assert bar["imbalance"] == 100  # 150 - 50


class TestTickRulePersistsAcrossBars:

    def test_last_price_persists_across_bar_boundary(self):
        """Tick rule state must carry across bar boundaries."""
        acc = TickImbalanceAccumulator()
        acc.on_tick(20000.0, 5, _et(9, 30, 0))
        acc.on_tick(20001.0, 10, _et(9, 34, 59))  # uptick → buy

        # Cross boundary — last_price should still be 20001.0
        bar = acc.on_tick(20001.0, 8, _et(9, 35, 0))
        assert bar is not None
        # New bar's first tick at same price → carries forward buy
        assert acc._buy_vol == 8
