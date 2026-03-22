"""Tests for trade calculation: entry, stop, target, sizing."""

import pytest
from bot.strategy.trade_calculator import (
    round_to_tick, calculate_setup, calculate_position_size,
    TICK_SIZE, POINT_VALUE,
)
from bot.state.trade_state import FVGRecord


def _make_fvg(fvg_type="bullish", zone_low=19500, zone_high=19515,
              middle_low=19490, middle_high=19530):
    return FVGRecord(
        fvg_id="test-fvg",
        fvg_type=fvg_type,
        zone_low=zone_low,
        zone_high=zone_high,
        time_candle1="2026-03-22T10:30:00",
        time_candle2="2026-03-22T10:35:00",
        time_candle3="2026-03-22T10:40:00",
        middle_open=19505,
        middle_low=middle_low,
        middle_high=middle_high,
        first_open=19480,
        time_period="10:30-11:00",
        formation_date="2026-03-22",
    )


class TestRoundToTick:
    """Tests for NQ tick rounding."""

    def test_already_on_tick(self):
        assert round_to_tick(19500.0) == 19500.0
        assert round_to_tick(19500.25) == 19500.25

    def test_round_down(self):
        assert round_to_tick(19500.10) == 19500.0

    def test_round_up(self):
        assert round_to_tick(19500.15) == 19500.25

    def test_round_midpoint(self):
        # 0.125 should round to 0.25 (Python banker's rounding: 0.125 * 4 = 0.5, rounds to 0)
        # Actually: round(0.125 * 4) / 4 = round(0.5) / 4 = 0 / 4 = 0.0
        # But 19500.125 * 4 = 78000.5, round = 78000 or 78001 (banker's)
        result = round_to_tick(19500.125)
        assert result in (19500.0, 19500.25)  # Either is acceptable

    def test_negative_values(self):
        assert round_to_tick(-0.25) == -0.25

    def test_zero(self):
        assert round_to_tick(0.0) == 0.0

    def test_large_values(self):
        assert round_to_tick(21000.50) == 21000.50
        assert round_to_tick(21000.37) == 21000.25


class TestCalculateSetup:
    """Tests for full trade setup calculation."""

    def test_bullish_mit_extreme(self):
        """MIT+EXTREME bullish: entry at zone_high, stop at middle_low."""
        fvg = _make_fvg(fvg_type="bullish", zone_low=19500, zone_high=19515,
                        middle_low=19490, middle_high=19530)
        cell = {"setup": "mit_extreme", "rr_target": 3.0, "median_risk": 12.25,
                "ev": 0.27, "win_rate": 31.8, "samples": 223}

        order = calculate_setup(fvg, cell, balance=76000)
        assert order is not None
        assert order.side == "BUY"
        assert order.entry_price == 19515.0
        assert order.stop_price == 19490.0
        assert order.risk_pts == 25.0
        assert order.target_price == 19515.0 + (3.0 * 25.0)  # 19590
        assert order.n_value == 3.0

    def test_bearish_mit_extreme(self):
        """MIT+EXTREME bearish: entry at zone_low, stop at middle_high."""
        fvg = _make_fvg(fvg_type="bearish", zone_low=19520, zone_high=19530,
                        middle_low=19500, middle_high=19535)
        cell = {"setup": "mit_extreme", "rr_target": 2.0, "median_risk": 12.0,
                "ev": 0.15, "win_rate": 36.3, "samples": 300}

        order = calculate_setup(fvg, cell, balance=76000)
        assert order is not None
        assert order.side == "SELL"
        assert order.entry_price == 19520.0
        assert order.stop_price == 19535.0
        assert order.risk_pts == 15.0
        assert order.target_price == 19520.0 - (2.0 * 15.0)  # 19490

    def test_bullish_mid_extreme(self):
        """MID+EXTREME bullish: entry at midpoint, stop at middle_low."""
        fvg = _make_fvg(fvg_type="bullish", zone_low=19500, zone_high=19520,
                        middle_low=19490, middle_high=19530)
        cell = {"setup": "mid_extreme", "rr_target": 2.5, "median_risk": 12.5,
                "ev": 0.19, "win_rate": 32.4, "samples": 207}

        order = calculate_setup(fvg, cell, balance=76000)
        assert order is not None
        assert order.side == "BUY"
        assert order.entry_price == 19510.0  # midpoint of 19500-19520
        assert order.stop_price == 19490.0
        assert order.risk_pts == 20.0

    def test_bearish_mid_extreme(self):
        """MID+EXTREME bearish: entry at midpoint, stop at middle_high."""
        fvg = _make_fvg(fvg_type="bearish", zone_low=19520, zone_high=19540,
                        middle_low=19510, middle_high=19545)
        cell = {"setup": "mid_extreme", "rr_target": 2.0, "median_risk": 12.0,
                "ev": 0.16, "win_rate": 38.6, "samples": 303}

        order = calculate_setup(fvg, cell, balance=76000)
        assert order is not None
        assert order.side == "SELL"
        assert order.entry_price == 19530.0  # midpoint of 19520-19540
        assert order.stop_price == 19545.0

    def test_zero_risk_returns_none(self):
        """If entry == stop, risk is 0 → return None."""
        fvg = _make_fvg(fvg_type="bullish", zone_low=19490, zone_high=19490,
                        middle_low=19490, middle_high=19530)
        cell = {"setup": "mit_extreme", "rr_target": 2.0, "median_risk": 0,
                "ev": 0.1, "win_rate": 40, "samples": 200}
        assert calculate_setup(fvg, cell, balance=76000) is None

    def test_prices_are_tick_rounded(self):
        """All prices should be on NQ tick boundaries."""
        fvg = _make_fvg(fvg_type="bullish", zone_low=19500.1, zone_high=19515.3,
                        middle_low=19490.6, middle_high=19530)
        cell = {"setup": "mit_extreme", "rr_target": 2.0, "median_risk": 12,
                "ev": 0.15, "win_rate": 35, "samples": 200}
        order = calculate_setup(fvg, cell, balance=76000)
        assert order is not None
        assert order.entry_price % 0.25 == 0
        assert order.stop_price % 0.25 == 0
        assert order.target_price % 0.25 == 0

    def test_generates_unique_group_id(self):
        """Each setup should get a unique group_id."""
        fvg = _make_fvg()
        cell = {"setup": "mit_extreme", "rr_target": 2.0, "median_risk": 12,
                "ev": 0.15, "win_rate": 35, "samples": 200}
        o1 = calculate_setup(fvg, cell, balance=76000)
        o2 = calculate_setup(fvg, cell, balance=76000)
        assert o1.group_id != o2.group_id


class TestPositionSizing:
    """Tests for calculate_position_size."""

    def test_standard_sizing(self):
        """$76k balance, 1% risk, 7.5pt risk = floor(760/150) = 5 contracts."""
        assert calculate_position_size(76000, 0.01, 7.5) == 5

    def test_larger_risk(self):
        """$76k, 1%, 12.25pt risk = floor(760/245) = 3 contracts."""
        assert calculate_position_size(76000, 0.01, 12.25) == 3

    def test_very_large_risk(self):
        """$76k, 1%, 22pt risk = floor(760/440) = 1 contract."""
        assert calculate_position_size(76000, 0.01, 22.0) == 1

    def test_minimum_one_contract(self):
        """Even with tiny balance, minimum is 1."""
        assert calculate_position_size(1000, 0.01, 50) == 1

    def test_zero_risk_returns_one(self):
        assert calculate_position_size(76000, 0.01, 0) == 1

    def test_zero_balance_returns_one(self):
        assert calculate_position_size(0, 0.01, 10) == 1

    def test_different_point_values(self):
        """ES has $50/point."""
        # $76k, 1%, 5pt risk with $50/pt = floor(760/250) = 3
        assert calculate_position_size(76000, 0.01, 5.0, point_value=50.0) == 3
