"""
Tests for the HFOIV gate (bot/risk/hfoiv_gate.py).

Covers:
    - Rolling HFOIV computation
    - Warmup behaviour (insufficient bars / sessions)
    - Cross-session normalization + percentile
    - Graduated threshold sizing
    - reset_day flush mechanics
    - Disabled gate passthrough
"""

import math
import numpy as np
import pytest

from bot.risk.hfoiv_gate import HFOIVGate, HFOIVConfig


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_gate(enabled=True, rolling_bars=4, lookback_sessions=3,
               bucket_minutes=30, thresholds=None):
    """Build a gate with small windows for fast testing."""
    if thresholds is None:
        thresholds = [(90, 0.25), (80, 0.50), (70, 0.75)]
    return HFOIVGate(HFOIVConfig(
        enabled=enabled,
        rolling_bars=rolling_bars,
        lookback_sessions=lookback_sessions,
        bucket_minutes=bucket_minutes,
        thresholds=thresholds,
    ))


def _feed_day(gate, values, start_minutes=510):
    """Feed a list of imbalance values as sequential 5-min bars.

    Default start_minutes=510 = 08:30 ET.
    Returns list of (mult, info) for each bar.
    """
    results = []
    for i, v in enumerate(values):
        bar_min = start_minutes + i * 5
        gate.update(bar_min, v)
        results.append(gate.get_size_multiplier(bar_min))
    return results


# ── Tests ────────────────────────────────────────────────────────────────

class TestRollingHFOIV:
    """Core HFOIV (rolling std of imbalance) computation."""

    def test_insufficient_bars_returns_1(self):
        gate = _make_gate(rolling_bars=4)
        gate.reset_day()
        # Feed only 3 bars — window not full
        for v in [100, 200, 300]:
            gate.update(510, v)
        mult, info = gate.get_size_multiplier(510)
        assert mult == 1.0
        assert info.get("reason") == "insufficient_bars"

    def test_hfoiv_matches_numpy_std(self):
        gate = _make_gate(rolling_bars=4, lookback_sessions=1)
        gate.reset_day()
        values = [100.0, 200.0, 50.0, 300.0]
        for i, v in enumerate(values):
            gate.update(510 + i * 5, v)

        # Manually check the HFOIV equals np.std of the last 4 values
        expected_std = float(np.std(values))
        mult, info = gate.get_size_multiplier(525)  # at last bar
        # Gate returns warmup (0 sessions), but internally computed HFOIV
        assert info.get("hfoiv") == round(expected_std, 2)

    def test_rolling_window_slides(self):
        gate = _make_gate(rolling_bars=4, lookback_sessions=1)
        gate.reset_day()
        # Feed 6 bars — window should only use last 4
        all_values = [10, 20, 30, 40, 50, 60]
        for i, v in enumerate(all_values):
            gate.update(510 + i * 5, v)

        expected_std = float(np.std([30, 40, 50, 60]))
        _, info = gate.get_size_multiplier(535)
        assert info.get("hfoiv") == round(expected_std, 2)


class TestWarmup:
    """Gate returns 1.0 during warmup (insufficient normalization history)."""

    def test_warmup_until_enough_sessions(self):
        gate = _make_gate(rolling_bars=4, lookback_sessions=3)
        # Simulate 2 sessions (need 3)
        for _ in range(2):
            gate.reset_day()
            _feed_day(gate, [100, 200, 50, 300, 150, 250])

        # Session 3 — history only has 2 sessions flushed
        gate.reset_day()
        _feed_day(gate, [100, 200, 50, 300])
        mult, info = gate.get_size_multiplier(530)
        assert mult == 1.0
        assert info.get("reason") == "warmup"
        assert info.get("sessions") == 2

    def test_gate_activates_after_enough_sessions(self):
        gate = _make_gate(rolling_bars=4, lookback_sessions=3)
        # Simulate 3 full sessions
        for _ in range(3):
            gate.reset_day()
            _feed_day(gate, [100, 200, 50, 300, 150, 250])

        # Session 4 — history now has 3 sessions, gate should be active
        gate.reset_day()
        _feed_day(gate, [100, 200, 50, 300])
        mult, info = gate.get_size_multiplier(530)
        # Should NOT return warmup
        assert info.get("reason") is None or info.get("reason") != "warmup"
        assert "percentile" in info


class TestNormalization:
    """Percentile rank against historical HFOIV values."""

    def test_low_hfoiv_gets_high_multiplier(self):
        """Low imbalance volatility → no size reduction."""
        gate = _make_gate(rolling_bars=4, lookback_sessions=3)

        # 3 sessions with moderate imbalance volatility
        for _ in range(3):
            gate.reset_day()
            _feed_day(gate, [100, -100, 100, -100, 100, -100])

        # Session 4: very calm (low imbalance volatility)
        gate.reset_day()
        _feed_day(gate, [10, 11, 12, 13])
        mult, info = gate.get_size_multiplier(530)
        assert mult == 1.0  # low percentile → no cut
        assert info.get("percentile", 100) < 70

    def test_high_hfoiv_gets_low_multiplier(self):
        """High imbalance volatility → size reduction."""
        gate = _make_gate(rolling_bars=4, lookback_sessions=3)

        # 3 sessions with mild imbalance volatility
        for _ in range(3):
            gate.reset_day()
            _feed_day(gate, [10, 11, 12, 13, 14, 15])

        # Session 4: extreme volatility
        gate.reset_day()
        _feed_day(gate, [1000, -1000, 1000, -1000])
        mult, info = gate.get_size_multiplier(530)
        assert mult < 1.0  # high percentile → size cut
        assert info.get("percentile", 0) >= 70


class TestGraduatedThresholds:
    """Threshold tiers produce correct multipliers."""

    def test_threshold_ordering(self):
        """Highest matching threshold wins."""
        gate = _make_gate(
            rolling_bars=4, lookback_sessions=3,
            thresholds=[(70, 0.75), (80, 0.50), (90, 0.25)]
        )

        # Build history with known distribution — 10 identical values per bucket
        for _ in range(3):
            gate.reset_day()
            # Constant imbalance → std ≈ 0 after rolling fills
            _feed_day(gate, [100, 100, 100, 100, 100, 100])

        # Now feed a session with non-zero std → will be above all history (pct=100)
        gate.reset_day()
        _feed_day(gate, [100, -100, 100, -100])
        mult, info = gate.get_size_multiplier(530)
        assert mult == 0.25  # ≥90th percentile
        assert info.get("percentile", 0) >= 90

    def test_custom_single_threshold(self):
        """A/B test with single threshold works."""
        gate = _make_gate(
            rolling_bars=4, lookback_sessions=3,
            thresholds=[(80, 0.50)]
        )

        for _ in range(3):
            gate.reset_day()
            _feed_day(gate, [50, 50, 50, 50, 50, 50])

        gate.reset_day()
        _feed_day(gate, [500, -500, 500, -500])
        mult, _ = gate.get_size_multiplier(530)
        assert mult == 0.50


class TestResetDay:
    """reset_day flushes staged values and clears rolling buffer."""

    def test_rolling_clears_on_reset(self):
        gate = _make_gate(rolling_bars=4)
        gate.reset_day()
        _feed_day(gate, [100, 200, 300, 400])  # fill rolling

        gate.reset_day()
        # After reset, rolling should be empty
        mult, info = gate.get_size_multiplier(510)
        assert mult == 1.0
        assert info.get("reason") == "insufficient_bars"

    def test_session_count_increments(self):
        gate = _make_gate(rolling_bars=4, lookback_sessions=3)
        assert gate._session_count == 0

        gate.reset_day()
        _feed_day(gate, [100, 200, 300, 400])
        assert gate._session_count == 0  # not yet flushed

        gate.reset_day()  # flushes session 1
        assert gate._session_count == 1

        _feed_day(gate, [100, 200, 300, 400])
        gate.reset_day()  # flushes session 2
        assert gate._session_count == 2

    def test_no_flush_if_no_data(self):
        """reset_day with no data doesn't increment session count."""
        gate = _make_gate()
        gate.reset_day()
        gate.reset_day()  # no data fed between resets
        assert gate._session_count == 0


class TestDisabled:
    """Disabled gate always returns 1.0."""

    def test_disabled_passthrough(self):
        gate = _make_gate(enabled=False)
        gate.reset_day()
        _feed_day(gate, [100, -100, 100, -100])
        mult, info = gate.get_size_multiplier(530)
        assert mult == 1.0
        assert info == {}


class TestTimeBuckets:
    """Time bucket mapping."""

    def test_bucket_labels(self):
        gate = _make_gate(bucket_minutes=30)
        assert gate._time_bucket(510) == "08:30-09:00"  # 08:30
        assert gate._time_bucket(570) == "09:30-10:00"  # 09:30
        assert gate._time_bucket(599) == "09:30-10:00"  # 09:59
        assert gate._time_bucket(600) == "10:00-10:30"  # 10:00
        assert gate._time_bucket(930) == "15:30-16:00"  # 15:30

    def test_buckets_are_independent(self):
        """History for one bucket doesn't affect another."""
        gate = _make_gate(rolling_bars=4, lookback_sessions=3, bucket_minutes=30)

        # 3 sessions: feed bars at 09:30 bucket only
        for _ in range(3):
            gate.reset_day()
            _feed_day(gate, [100, 200, 300, 400], start_minutes=570)

        # Session 4: query 10:00 bucket (no history there)
        gate.reset_day()
        _feed_day(gate, [100, 200, 300, 400], start_minutes=600)
        mult, info = gate.get_size_multiplier(615)
        # Should be warmup — the 10:00 bucket has no history
        # (or returns 1.0 because session_count < lookback)
        assert mult == 1.0


class TestInputValidation:
    """Configuration validation — bad configs must raise, not silently break."""

    def test_rolling_bars_zero_raises(self):
        with pytest.raises(ValueError, match="rolling_bars"):
            HFOIVConfig(rolling_bars=0)

    def test_lookback_sessions_zero_raises(self):
        with pytest.raises(ValueError, match="lookback_sessions"):
            HFOIVConfig(lookback_sessions=0)

    def test_empty_thresholds_raises(self):
        with pytest.raises(ValueError, match="thresholds"):
            HFOIVConfig(thresholds=[])

    def test_multiplier_above_1_raises(self):
        with pytest.raises(ValueError, match="<= 1.0"):
            HFOIVConfig(thresholds=[(70, 1.5)])

    def test_negative_multiplier_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            HFOIVConfig(thresholds=[(70, -0.5)])

    def test_percentile_out_of_range_raises(self):
        with pytest.raises(ValueError, match="0-100"):
            HFOIVConfig(thresholds=[(150, 0.5)])


class TestNaNHandling:
    """NaN/Inf must never corrupt the gate or crash."""

    def test_nan_imbalance_ignored(self):
        gate = _make_gate(rolling_bars=4, lookback_sessions=3)
        gate.reset_day()
        gate.update(570, 100)
        gate.update(575, float('nan'))  # should be silently dropped
        gate.update(580, 200)
        assert len(gate._rolling) == 2  # NaN was skipped

    def test_inf_imbalance_ignored(self):
        gate = _make_gate(rolling_bars=4, lookback_sessions=3)
        gate.reset_day()
        gate.update(570, float('inf'))
        assert len(gate._rolling) == 0  # inf was skipped

    def test_gate_returns_1_on_degenerate_data(self):
        """If rolling window somehow gets corrupted, gate is safe."""
        gate = _make_gate(rolling_bars=2, lookback_sessions=1)
        gate.reset_day()
        gate.update(570, 0)
        gate.update(575, 0)  # std of [0, 0] = 0.0 — valid but degenerate
        mult, info = gate.get_size_multiplier(575)
        assert mult == 1.0  # warmup (only 0 sessions)
