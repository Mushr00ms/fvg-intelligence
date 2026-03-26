"""Tests for FVG detection (check_fvg_3bars + ActiveFVGManager)."""

import pytest
from bot.strategy.fvg_detector import check_fvg_3bars, _assign_time_period, SESSION_INTERVALS, ActiveFVGManager
from datetime import time


class TestCheckFVG3Bars:
    """Tests for the core 3-bar FVG detection function."""

    def test_bullish_fvg_detected(self, sample_bullish_fvg):
        """Bullish FVG: third candle low (19515) > first candle high (19500)."""
        bar1, bar2, bar3 = sample_bullish_fvg
        fvg = check_fvg_3bars(bar1, bar2, bar3)
        assert fvg is not None
        assert fvg.fvg_type == "bullish"
        assert fvg.zone_low == 19500   # first candle high
        assert fvg.zone_high == 19515  # third candle low
        assert fvg.middle_open == 19505
        assert fvg.middle_low == 19490
        assert fvg.middle_high == 19530

    def test_bearish_fvg_detected(self, sample_bearish_fvg):
        """Bearish FVG: third candle high (19520) < first candle low (19530)."""
        bar1, bar2, bar3 = sample_bearish_fvg
        fvg = check_fvg_3bars(bar1, bar2, bar3)
        assert fvg is not None
        assert fvg.fvg_type == "bearish"
        assert fvg.zone_low == 19520   # third candle high
        assert fvg.zone_high == 19530  # first candle low
        assert fvg.middle_open == 19525
        assert fvg.middle_low == 19500
        assert fvg.middle_high == 19535

    def test_no_fvg_overlapping_bars(self):
        """No FVG when candles overlap."""
        bar1 = {"open": 100, "high": 110, "low": 90, "close": 105, "date": "2026-03-22T10:30:00"}
        bar2 = {"open": 106, "high": 112, "low": 98, "close": 108, "date": "2026-03-22T10:35:00"}
        bar3 = {"open": 107, "high": 115, "low": 100, "close": 113, "date": "2026-03-22T10:40:00"}
        fvg = check_fvg_3bars(bar1, bar2, bar3)
        assert fvg is None

    def test_fvg_below_min_size(self):
        """FVG too small (below 0.25 threshold)."""
        bar1 = {"open": 100, "high": 100.10, "low": 99, "close": 100.05, "date": "2026-03-22T10:30:00"}
        bar2 = {"open": 100.15, "high": 100.30, "low": 100.05, "close": 100.20, "date": "2026-03-22T10:35:00"}
        bar3 = {"open": 100.20, "high": 100.40, "low": 100.20, "close": 100.35, "date": "2026-03-22T10:40:00"}
        # gap = 100.20 - 100.10 = 0.10 < 0.25
        fvg = check_fvg_3bars(bar1, bar2, bar3, min_size=0.25)
        assert fvg is None

    def test_fvg_exact_threshold(self):
        """FVG exactly at min_size threshold should be included."""
        bar1 = {"open": 100, "high": 100.00, "low": 99, "close": 99.5, "date": "2026-03-22T10:30:00"}
        bar2 = {"open": 100.10, "high": 100.50, "low": 99.90, "close": 100.20, "date": "2026-03-22T10:35:00"}
        bar3 = {"open": 100.30, "high": 100.60, "low": 100.25, "close": 100.50, "date": "2026-03-22T10:40:00"}
        # gap = 100.25 - 100.00 = 0.25, exactly threshold
        fvg = check_fvg_3bars(bar1, bar2, bar3, min_size=0.25)
        assert fvg is not None
        assert fvg.fvg_type == "bullish"

    def test_fvg_preserves_first_open(self, sample_bullish_fvg):
        """first_open should be bar1's open."""
        bar1, bar2, bar3 = sample_bullish_fvg
        fvg = check_fvg_3bars(bar1, bar2, bar3)
        assert fvg.first_open == 19480

    def test_fvg_preserves_timestamps(self, sample_bullish_fvg):
        """All three candle timestamps should be stored."""
        bar1, bar2, bar3 = sample_bullish_fvg
        fvg = check_fvg_3bars(bar1, bar2, bar3)
        assert "10:30:00" in fvg.time_candle1
        assert "10:35:00" in fvg.time_candle2
        assert "10:40:00" in fvg.time_candle3

    def test_custom_min_size(self):
        """Custom min_size threshold."""
        bar1 = {"open": 100, "high": 100, "low": 99, "close": 99.5, "date": "2026-03-22T10:30:00"}
        bar2 = {"open": 101, "high": 102, "low": 100.5, "close": 101.5, "date": "2026-03-22T10:35:00"}
        bar3 = {"open": 101.5, "high": 103, "low": 101, "close": 102.5, "date": "2026-03-22T10:40:00"}
        # gap = 101 - 100 = 1.0
        assert check_fvg_3bars(bar1, bar2, bar3, min_size=0.5) is not None
        assert check_fvg_3bars(bar1, bar2, bar3, min_size=1.5) is None

    def test_unique_ids(self, sample_bullish_fvg):
        """Each FVG should get a unique ID."""
        bar1, bar2, bar3 = sample_bullish_fvg
        fvg1 = check_fvg_3bars(bar1, bar2, bar3)
        fvg2 = check_fvg_3bars(bar1, bar2, bar3)
        assert fvg1.fvg_id != fvg2.fvg_id


class TestAssignTimePeriod:
    """Tests for time period assignment."""

    def test_morning_period(self):
        from datetime import datetime
        tp = _assign_time_period("2026-03-22T10:35:00", SESSION_INTERVALS)
        assert tp == "10:30-11:00"

    def test_start_boundary(self):
        tp = _assign_time_period("2026-03-22T09:30:00", SESSION_INTERVALS)
        assert tp == "09:30-10:00"

    def test_end_boundary(self):
        tp = _assign_time_period("2026-03-22T15:55:00", SESSION_INTERVALS)
        assert tp == "15:30-16:00"

    def test_outside_session(self):
        tp = _assign_time_period("2026-03-22T08:00:00", SESSION_INTERVALS)
        assert tp is None

    def test_exact_interval_boundary(self):
        tp = _assign_time_period("2026-03-22T11:00:00", SESSION_INTERVALS)
        assert tp == "11:00-11:30"


class TestActiveFVGManager:
    """Tests for the FVG lifecycle manager."""

    class _CaptureLogger:
        def __init__(self):
            self.records = []

        def log(self, event, **kwargs):
            self.records.append({"event": event, **kwargs})

    def _make_manager(self, sample_strategy, logger=None):
        strategy_dir, _ = sample_strategy
        from bot.strategy.strategy_loader import StrategyLoader
        loader = StrategyLoader(strategy_dir)
        loader.load()
        return ActiveFVGManager(loader, logger=logger)

    def test_detects_fvg_on_bar(self, sample_strategy):
        mgr = self._make_manager(sample_strategy)
        # Feed 3 bars that form a bullish FVG in the 10:30-11:00 window
        bars = [
            {"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T10:30:00"},
            {"open": 19505, "high": 19530, "low": 19490, "close": 19520, "date": "2026-03-22T10:35:00"},
            {"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T10:40:00"},
        ]
        for bar in bars:
            result = mgr.on_5min_bar(bar)

        assert result is not None
        assert mgr.active_count == 1

    def test_skips_fvg_outside_strategy_periods(self, sample_strategy):
        mgr = self._make_manager(sample_strategy)
        # FVG at 09:30 — no strategy cell for this time
        bars = [
            {"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T09:30:00"},
            {"open": 19505, "high": 19530, "low": 19490, "close": 19520, "date": "2026-03-22T09:35:00"},
            {"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T09:40:00"},
        ]
        for bar in bars:
            result = mgr.on_5min_bar(bar)

        assert result is None
        assert mgr.active_count == 0

    def test_detected_log_marks_strategy_membership(self, sample_strategy):
        logger = self._CaptureLogger()
        mgr = self._make_manager(sample_strategy, logger=logger)
        bars = [
            {"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T10:30:00"},
            {"open": 19505, "high": 19530, "low": 19490, "close": 19520, "date": "2026-03-22T10:35:00"},
            {"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T10:40:00"},
        ]
        for bar in bars:
            result = mgr.on_5min_bar(bar)

        assert result is not None
        detected = [r for r in logger.records if r["event"] == "fvg_detected"]
        assert len(detected) == 1
        assert detected[0]["strategy_id"] == "test-strategy"
        assert detected[0]["in_strategy"] is True
        assert detected[0]["in_strategy_time_period"] is True
        assert detected[0]["strategy_cells_for_period"] > 0

    def test_skipped_log_marks_not_in_strategy_period(self, sample_strategy):
        logger = self._CaptureLogger()
        mgr = self._make_manager(sample_strategy, logger=logger)
        bars = [
            {"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T09:30:00"},
            {"open": 19505, "high": 19530, "low": 19490, "close": 19520, "date": "2026-03-22T09:35:00"},
            {"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T09:40:00"},
        ]
        for bar in bars:
            result = mgr.on_5min_bar(bar)

        assert result is None
        skipped = [r for r in logger.records if r["event"] == "fvg_skipped_strategy"]
        assert len(skipped) == 1
        assert skipped[0]["strategy_id"] == "test-strategy"
        assert skipped[0]["in_strategy"] is False
        assert skipped[0]["in_strategy_time_period"] is False
        assert skipped[0]["strategy_cells_for_period"] == 0

    def test_expire_all(self, sample_strategy):
        mgr = self._make_manager(sample_strategy)
        bars = [
            {"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T10:30:00"},
            {"open": 19505, "high": 19530, "low": 19490, "close": 19520, "date": "2026-03-22T10:35:00"},
            {"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T10:40:00"},
        ]
        for bar in bars:
            mgr.on_5min_bar(bar)

        expired = mgr.expire_all()
        assert len(expired) == 1
        assert mgr.active_count == 0

    def test_remove_fvg(self, sample_strategy):
        mgr = self._make_manager(sample_strategy)
        bars = [
            {"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T10:30:00"},
            {"open": 19505, "high": 19530, "low": 19490, "close": 19520, "date": "2026-03-22T10:35:00"},
            {"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T10:40:00"},
        ]
        for bar in bars:
            result = mgr.on_5min_bar(bar)

        mgr.remove(result.fvg_id)
        assert mgr.active_count == 0

    def test_needs_3_bars_minimum(self, sample_strategy):
        mgr = self._make_manager(sample_strategy)
        bar = {"open": 100, "high": 110, "low": 90, "close": 105, "date": "2026-03-22T10:30:00"}
        assert mgr.on_5min_bar(bar) is None
        assert mgr.on_5min_bar(bar) is None

    def test_detect_from_tick_bar(self, sample_strategy):
        """detect_from_tick_bar should detect FVG using cached bar1/bar2 + provided bar3."""
        mgr = self._make_manager(sample_strategy)
        # Feed bar1 and bar2 via normal path
        mgr.on_5min_bar({"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T10:30:00"})
        mgr.on_5min_bar({"open": 19505, "high": 19530, "low": 19490, "close": 19520, "date": "2026-03-22T10:35:00"})

        # bar3 from tick path
        bar3 = {"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T10:40:00"}
        fvg = mgr.detect_from_tick_bar(bar3)

        assert fvg is not None
        assert fvg.fvg_type == "bullish"
        assert mgr.active_count == 1

    def test_detect_from_tick_bar_does_not_append(self, sample_strategy):
        """detect_from_tick_bar should NOT append bar3 to _recent_bars."""
        mgr = self._make_manager(sample_strategy)
        mgr.on_5min_bar({"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T10:30:00"})
        mgr.on_5min_bar({"open": 19505, "high": 19530, "low": 19490, "close": 19520, "date": "2026-03-22T10:35:00"})

        bar3 = {"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T10:40:00"}
        mgr.detect_from_tick_bar(bar3)

        # _recent_bars should still have only 2 bars (bar1 + bar2)
        assert len(mgr._recent_bars) == 2

    def test_detect_from_tick_bar_needs_2_bars(self, sample_strategy):
        """detect_from_tick_bar returns None if fewer than 2 bars cached."""
        mgr = self._make_manager(sample_strategy)
        mgr.on_5min_bar({"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T10:30:00"})

        bar3 = {"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T10:40:00"}
        assert mgr.detect_from_tick_bar(bar3) is None

    def test_append_bar_no_detection(self, sample_strategy):
        """append_bar should add to _recent_bars but not trigger FVG detection."""
        mgr = self._make_manager(sample_strategy)
        mgr.on_5min_bar({"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T10:30:00"})
        mgr.on_5min_bar({"open": 19505, "high": 19530, "low": 19490, "close": 19520, "date": "2026-03-22T10:35:00"})

        # This bar would form a bullish FVG, but append_bar shouldn't detect it
        bar3 = {"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T10:40:00"}
        mgr.append_bar(bar3)

        assert len(mgr._recent_bars) == 3
        assert mgr.active_count == 0  # No detection happened

    def test_append_bar_keeps_deque_in_sync(self, sample_strategy):
        """After append_bar, subsequent on_5min_bar should use the correct bar1/bar2."""
        mgr = self._make_manager(sample_strategy)
        # Feed bars via on_5min_bar
        mgr.on_5min_bar({"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T10:30:00"})
        mgr.on_5min_bar({"open": 19505, "high": 19530, "low": 19490, "close": 19520, "date": "2026-03-22T10:35:00"})

        # Append bar3 without detection (simulating tick-detected + keepUpToDate sync)
        mgr.append_bar({"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T10:40:00"})

        # Now bar4 via on_5min_bar — bar2 and bar3 become the new bar1 and bar2
        # No FVG here (overlapping bars)
        result = mgr.on_5min_bar({"open": 19540, "high": 19555, "low": 19530, "close": 19548, "date": "2026-03-22T10:45:00"})
        assert result is None
        assert len(mgr._recent_bars) == 4
