"""Tests for mitigation scanning."""

import pytest
from bot.strategy.mitigation_scanner import check_mitigation, scan_active_fvgs
from bot.state.trade_state import FVGRecord


def _make_fvg(fvg_type="bullish", zone_low=19500, zone_high=19515):
    return FVGRecord(
        fvg_id="test-fvg-1",
        fvg_type=fvg_type,
        zone_low=zone_low,
        zone_high=zone_high,
        time_candle1="2026-03-22T10:30:00",
        time_candle2="2026-03-22T10:35:00",
        time_candle3="2026-03-22T10:40:00",
        middle_open=19505,
        middle_low=19490,
        middle_high=19530,
        first_open=19480,
        time_period="10:30-11:00",
        formation_date="2026-03-22",
    )


class TestCheckMitigation:
    """Tests for single-bar mitigation check."""

    def test_bar_touches_zone_low(self):
        """Bar wick reaches up to touch the zone."""
        fvg = _make_fvg(zone_low=19500, zone_high=19515)
        bar = {"open": 19510, "high": 19520, "low": 19505, "close": 19512, "date": "2026-03-22T10:45:00"}
        assert check_mitigation(bar, fvg) is True

    def test_bar_completely_inside_zone(self):
        """Bar entirely within the FVG zone."""
        fvg = _make_fvg(zone_low=19500, zone_high=19520)
        bar = {"open": 19505, "high": 19515, "low": 19502, "close": 19510, "date": "2026-03-22T10:45:00"}
        assert check_mitigation(bar, fvg) is True

    def test_bar_wick_just_touches_zone_high(self):
        """Bar low exactly equals zone_high — still mitigated (wick touch)."""
        fvg = _make_fvg(zone_low=19500, zone_high=19515)
        bar = {"open": 19520, "high": 19530, "low": 19515, "close": 19525, "date": "2026-03-22T10:45:00"}
        assert check_mitigation(bar, fvg) is True

    def test_bar_wick_just_touches_zone_low(self):
        """Bar high exactly equals zone_low — still mitigated."""
        fvg = _make_fvg(zone_low=19500, zone_high=19515)
        bar = {"open": 19495, "high": 19500, "low": 19490, "close": 19498, "date": "2026-03-22T10:45:00"}
        assert check_mitigation(bar, fvg) is True

    def test_bar_completely_above_zone(self):
        """Bar entirely above the zone — no mitigation."""
        fvg = _make_fvg(zone_low=19500, zone_high=19515)
        bar = {"open": 19520, "high": 19540, "low": 19516, "close": 19530, "date": "2026-03-22T10:45:00"}
        assert check_mitigation(bar, fvg) is False

    def test_bar_completely_below_zone(self):
        """Bar entirely below the zone — no mitigation."""
        fvg = _make_fvg(zone_low=19500, zone_high=19515)
        bar = {"open": 19490, "high": 19499, "low": 19480, "close": 19485, "date": "2026-03-22T10:45:00"}
        assert check_mitigation(bar, fvg) is False


class TestScanActiveFVGs:
    """Tests for scanning multiple FVGs."""

    def test_mitigates_one_of_multiple(self):
        fvg1 = _make_fvg(zone_low=19500, zone_high=19515)
        fvg1.fvg_id = "fvg-1"
        fvg2 = _make_fvg(zone_low=19600, zone_high=19615)
        fvg2.fvg_id = "fvg-2"

        # Bar only reaches fvg1's zone
        bar = {"open": 19510, "high": 19520, "low": 19505, "close": 19515, "date": "2026-03-22T10:45:00"}
        result = scan_active_fvgs(bar, [fvg1, fvg2])

        assert len(result) == 1
        assert result[0][0].fvg_id == "fvg-1"
        assert fvg1.is_mitigated is True
        assert fvg2.is_mitigated is False

    def test_mitigates_multiple_same_bar(self):
        fvg1 = _make_fvg(zone_low=19500, zone_high=19510)
        fvg1.fvg_id = "fvg-1"
        fvg2 = _make_fvg(zone_low=19505, zone_high=19515)
        fvg2.fvg_id = "fvg-2"

        # Large bar touches both zones
        bar = {"open": 19495, "high": 19520, "low": 19490, "close": 19510, "date": "2026-03-22T10:45:00"}
        result = scan_active_fvgs(bar, [fvg1, fvg2])

        assert len(result) == 2

    def test_already_mitigated_skipped(self):
        fvg = _make_fvg()
        fvg.is_mitigated = True

        bar = {"open": 19510, "high": 19520, "low": 19505, "close": 19515, "date": "2026-03-22T10:45:00"}
        result = scan_active_fvgs(bar, [fvg])

        assert len(result) == 0

    def test_empty_fvg_list(self):
        bar = {"open": 100, "high": 110, "low": 90, "close": 105, "date": "2026-03-22T10:45:00"}
        result = scan_active_fvgs(bar, [])
        assert len(result) == 0

    def test_mitigation_time_recorded(self):
        fvg = _make_fvg()
        bar = {"open": 19510, "high": 19520, "low": 19505, "close": 19515, "date": "2026-03-22T10:45:00"}
        result = scan_active_fvgs(bar, [fvg])

        assert fvg.mitigation_time == "2026-03-22T10:45:00"
