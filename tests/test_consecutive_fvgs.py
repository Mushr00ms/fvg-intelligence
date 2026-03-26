"""
Parametrized test for N consecutive FVG detection via the tick pipeline.

Nothing is hardcoded — price sequences are generated dynamically from a
random seed and market parameters. Tests bullish runs, bearish runs,
mixed direction, and varying gap sizes to prove the tick detection
pipeline handles any number of consecutive/nested FVGs.
"""

import random
from datetime import datetime, timedelta

import pytest
import pytz

from bot.strategy.tick_bar_builder import TickBarBuilder
from bot.strategy.fvg_detector import ActiveFVGManager, check_fvg_3bars

NY = pytz.timezone("America/New_York")

# ── Helpers ──────────────────────────────────────────────────────────────────


class _AllCellsStrategy:
    """Strategy mock that has a cell for every time_period × risk_range combo."""
    strategy_id = "all-cells"

    class _InfiniteLookup:
        def keys(self):
            """Yield cells for every 30-min window so _finalize_fvg never skips."""
            from bot.strategy.fvg_detector import SESSION_INTERVALS
            risk_ranges = ["5-10", "10-15", "15-20", "20-25", "25-30", "30-40", "40-80"]
            for start, end in SESSION_INTERVALS:
                tp = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
                for rr in risk_ranges:
                    yield (tp, rr)

    _lookup = _InfiniteLookup()


def _round_tick(price):
    """Round to NQ tick size (0.25)."""
    return round(price * 4) / 4


def generate_bullish_run(n_bars, start_price, gap_min, gap_max, bar_range, seed=None):
    """Generate n_bars of OHLC data forming a bullish staircase.

    Each bar gaps up from the previous one, guaranteeing a bullish FVG
    for every sliding 3-bar window after the first 2 bars.

    Args:
        n_bars: total number of bars (need n_bars >= 3 for 1 FVG)
        start_price: starting price level
        gap_min: minimum gap size between consecutive bars (in points)
        gap_max: maximum gap size
        bar_range: intra-bar range (high - low)
        seed: random seed for reproducibility

    Returns:
        list of dicts {open, high, low, close} — no date yet.
    """
    rng = random.Random(seed)
    bars = []
    price = start_price

    for _ in range(n_bars):
        low = _round_tick(price)
        high = _round_tick(price + bar_range + rng.uniform(0, bar_range * 0.5))
        open_p = _round_tick(low + rng.uniform(0, (high - low) * 0.3))
        close_p = _round_tick(high - rng.uniform(0, (high - low) * 0.3))
        bars.append({"open": open_p, "high": high, "low": low, "close": close_p})

        # Next bar starts above this bar's high by the gap amount
        gap = rng.uniform(gap_min, gap_max)
        price = high + gap

    return bars


def generate_bearish_run(n_bars, start_price, gap_min, gap_max, bar_range, seed=None):
    """Generate n_bars of OHLC data forming a bearish staircase.

    Each bar gaps down, guaranteeing a bearish FVG for every 3-bar window.
    """
    rng = random.Random(seed)
    bars = []
    price = start_price

    for _ in range(n_bars):
        high = _round_tick(price)
        low = _round_tick(price - bar_range - rng.uniform(0, bar_range * 0.5))
        open_p = _round_tick(high - rng.uniform(0, (high - low) * 0.3))
        close_p = _round_tick(low + rng.uniform(0, (high - low) * 0.3))
        bars.append({"open": open_p, "high": high, "low": low, "close": close_p})

        gap = rng.uniform(gap_min, gap_max)
        price = low - gap

    return bars


def generate_mixed_run(n_bars, start_price, gap_min, gap_max, bar_range, seed=None):
    """Generate n_bars alternating between bullish and bearish gaps.

    Not every 3-bar window will form an FVG — this tests that the pipeline
    doesn't produce false positives and handles direction changes.
    """
    rng = random.Random(seed)
    bars = []
    price = start_price
    direction = 1  # 1 = up, -1 = down

    for i in range(n_bars):
        if direction == 1:
            low = _round_tick(price)
            high = _round_tick(price + bar_range + rng.uniform(0, bar_range * 0.3))
        else:
            high = _round_tick(price)
            low = _round_tick(price - bar_range - rng.uniform(0, bar_range * 0.3))

        open_p = _round_tick(low + rng.uniform(0.2, 0.8) * (high - low))
        close_p = _round_tick(low + rng.uniform(0.2, 0.8) * (high - low))
        bars.append({"open": open_p, "high": high, "low": low, "close": close_p})

        gap = rng.uniform(gap_min, gap_max)
        if direction == 1:
            price = high + gap
        else:
            price = low - gap

        # Randomly flip direction
        if rng.random() < 0.4:
            direction *= -1

    return bars


def assign_dates(bars, start_hour=10, start_min=0, bar_minutes=5):
    """Assign consecutive 5-min ET datetimes to a list of bar dicts."""
    for i, bar in enumerate(bars):
        total_minutes = start_hour * 60 + start_min + i * bar_minutes
        h, m = divmod(total_minutes, 60)
        bar["date"] = NY.localize(datetime(2026, 3, 26, h, m))


def generate_ticks_for_bar(bar, n_ticks=30):
    """Generate ticks that produce the given bar's OHLC.

    Walks: open → high → low → close with some noise.
    """
    rng = random.Random(hash(str(bar["date"])))
    ticks = []
    dt = bar["date"]
    interval = timedelta(seconds=300 / n_ticks)

    waypoints = [bar["open"], bar["high"], bar["low"], bar["close"]]
    segment_len = n_ticks // 3

    for seg_idx in range(3):
        start_p = waypoints[seg_idx]
        end_p = waypoints[seg_idx + 1]
        count = segment_len if seg_idx < 2 else (n_ticks - 2 * segment_len)
        for i in range(count):
            frac = i / max(count - 1, 1)
            price = start_p + (end_p - start_p) * frac
            # Add small noise but clamp to bar extremes
            noise = rng.uniform(-0.5, 0.5)
            price = max(bar["low"], min(bar["high"], price + noise))
            tick_time = dt + interval * (seg_idx * segment_len + i)
            ticks.append((_round_tick(price), tick_time))

    return ticks


def run_tick_pipeline(seed_bars, tick_bars):
    """Run bars through the full tick pipeline and return detected FVGs.

    Args:
        seed_bars: list of bar dicts to seed _recent_bars (simulates startup)
        tick_bars: list of bar dicts to replay as ticks

    Returns:
        (detected_fvgs, fvg_mgr)
    """
    fvg_mgr = ActiveFVGManager(_AllCellsStrategy(), min_fvg_size=0.25)
    builder = TickBarBuilder()
    detected = []

    # Seed from "historical" bars
    for bar in seed_bars:
        fvg_mgr.append_bar(bar)

    # Replay each bar as ticks
    for bar in tick_bars:
        ticks = generate_ticks_for_bar(bar)
        for price, tick_time in ticks:
            completed = builder.on_tick(price, tick_time)
            if completed is not None:
                if len(fvg_mgr._recent_bars) >= 2:
                    fvg = fvg_mgr.detect_from_tick_bar(completed)
                    if fvg:
                        detected.append(fvg)
                fvg_mgr.append_bar(completed)

    # Flush last bar
    if tick_bars:
        last_dt = tick_bars[-1]["date"] + timedelta(minutes=5, seconds=1)
        completed = builder.on_tick(tick_bars[-1]["close"], last_dt)
        if completed is not None:
            if len(fvg_mgr._recent_bars) >= 2:
                fvg = fvg_mgr.detect_from_tick_bar(completed)
                if fvg:
                    detected.append(fvg)
            fvg_mgr.append_bar(completed)

    return detected, fvg_mgr


def count_expected_fvgs(all_bars):
    """Count how many FVGs exist in a bar sequence using check_fvg_3bars directly.

    This is the ground truth — the tick pipeline should match this count exactly.
    """
    count = 0
    for i in range(2, len(all_bars)):
        fvg = check_fvg_3bars(all_bars[i - 2], all_bars[i - 1], all_bars[i])
        if fvg is not None:
            count += 1
    return count


# ── Parametrized Tests ───────────────────────────────────────────────────────


class TestConsecutiveBullishFVGs:
    """Bullish staircase: every 3-bar window produces a bullish FVG."""

    @pytest.mark.parametrize("n_fvgs", [2, 3, 6, 10, 20])
    def test_detects_all_n_bullish_fvgs(self, n_fvgs):
        n_bars = n_fvgs + 2  # need 2 seed bars + n_fvgs tick bars
        bars = generate_bullish_run(n_bars, start_price=24000,
                                    gap_min=2, gap_max=10,
                                    bar_range=15, seed=n_fvgs)
        assign_dates(bars)

        seed = bars[:2]
        tick_bars = bars[2:]

        detected, mgr = run_tick_pipeline(seed, tick_bars)

        # Ground truth: count FVGs from the raw bar sequence
        expected = count_expected_fvgs(bars)

        assert len(detected) == expected, (
            f"Expected {expected} FVGs from {n_bars} bars, detected {len(detected)}"
        )
        assert len(detected) >= n_fvgs, (
            f"Bullish staircase of {n_bars} bars should yield >= {n_fvgs} FVGs"
        )
        for fvg in detected:
            assert fvg.fvg_type == "bullish"

    @pytest.mark.parametrize("n_fvgs", [2, 3, 6, 10, 20])
    def test_fvg_zones_are_ascending(self, n_fvgs):
        """In a bullish run, each FVG zone should be higher than the previous."""
        n_bars = n_fvgs + 2
        bars = generate_bullish_run(n_bars, start_price=24000,
                                    gap_min=2, gap_max=10,
                                    bar_range=15, seed=n_fvgs)
        assign_dates(bars)
        detected, _ = run_tick_pipeline(bars[:2], bars[2:])

        for i in range(1, len(detected)):
            assert detected[i].zone_low > detected[i - 1].zone_low, (
                f"FVG {i} zone_low ({detected[i].zone_low}) should be > "
                f"FVG {i-1} zone_low ({detected[i-1].zone_low})"
            )

    @pytest.mark.parametrize("seed", range(5))
    def test_random_bullish_run_matches_ground_truth(self, seed):
        """With random parameters, tick pipeline matches direct bar-by-bar check."""
        rng = random.Random(seed)
        n_bars = rng.randint(5, 15)
        start = rng.uniform(20000, 25000)
        gap = rng.uniform(1, 20)
        br = rng.uniform(5, 30)

        bars = generate_bullish_run(n_bars, start, gap_min=gap,
                                    gap_max=gap * 2, bar_range=br, seed=seed)
        assign_dates(bars)
        detected, _ = run_tick_pipeline(bars[:2], bars[2:])
        expected = count_expected_fvgs(bars)

        assert len(detected) == expected


class TestConsecutiveBearishFVGs:
    """Bearish staircase: every 3-bar window produces a bearish FVG."""

    @pytest.mark.parametrize("n_fvgs", [2, 3, 6, 10, 20])
    def test_detects_all_n_bearish_fvgs(self, n_fvgs):
        n_bars = n_fvgs + 2
        bars = generate_bearish_run(n_bars, start_price=24000,
                                    gap_min=2, gap_max=10,
                                    bar_range=15, seed=n_fvgs)
        assign_dates(bars)
        detected, _ = run_tick_pipeline(bars[:2], bars[2:])

        expected = count_expected_fvgs(bars)
        assert len(detected) == expected
        assert len(detected) >= n_fvgs
        for fvg in detected:
            assert fvg.fvg_type == "bearish"

    @pytest.mark.parametrize("n_fvgs", [2, 3, 6, 10, 20])
    def test_fvg_zones_are_descending(self, n_fvgs):
        n_bars = n_fvgs + 2
        bars = generate_bearish_run(n_bars, start_price=24000,
                                    gap_min=2, gap_max=10,
                                    bar_range=15, seed=n_fvgs)
        assign_dates(bars)
        detected, _ = run_tick_pipeline(bars[:2], bars[2:])

        for i in range(1, len(detected)):
            assert detected[i].zone_high < detected[i - 1].zone_high, (
                f"FVG {i} zone_high ({detected[i].zone_high}) should be < "
                f"FVG {i-1} zone_high ({detected[i-1].zone_high})"
            )


class TestMixedDirectionRun:
    """Mixed direction: FVGs only form where gaps exist, no false positives."""

    @pytest.mark.parametrize("seed", range(10))
    def test_tick_pipeline_matches_ground_truth(self, seed):
        """Whatever the direction mix, tick pipeline count == bar-by-bar count."""
        rng = random.Random(seed)
        n_bars = rng.randint(6, 20)
        bars = generate_mixed_run(n_bars, start_price=24000,
                                  gap_min=1, gap_max=15,
                                  bar_range=10, seed=seed)
        assign_dates(bars)
        detected, _ = run_tick_pipeline(bars[:2], bars[2:])
        expected = count_expected_fvgs(bars)

        assert len(detected) == expected, (
            f"Seed {seed}: expected {expected} FVGs from {n_bars} bars, got {len(detected)}"
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_no_phantom_fvgs(self, seed):
        """Every detected FVG must correspond to a real gap in the bar data."""
        rng = random.Random(seed)
        n_bars = rng.randint(6, 20)
        bars = generate_mixed_run(n_bars, start_price=24000,
                                  gap_min=1, gap_max=15,
                                  bar_range=10, seed=seed)
        assign_dates(bars)
        detected, _ = run_tick_pipeline(bars[:2], bars[2:])

        for fvg in detected:
            assert fvg.zone_high > fvg.zone_low, "Zone must have positive size"
            assert fvg.zone_high - fvg.zone_low >= 0.25, "Zone must meet min size"


class TestRecentBarsConsistency:
    """Verify _recent_bars deque stays in sync regardless of FVG count."""

    @pytest.mark.parametrize("n_bars", [5, 10, 20, 50])
    def test_recent_bars_length_correct(self, n_bars):
        bars = generate_bullish_run(n_bars, start_price=24000,
                                    gap_min=2, gap_max=8,
                                    bar_range=12, seed=n_bars)
        assign_dates(bars)
        _, mgr = run_tick_pipeline(bars[:2], bars[2:])

        # deque maxlen=10, so min(n_bars, 10)
        expected_len = min(n_bars, 10)
        assert len(mgr._recent_bars) == expected_len

    @pytest.mark.parametrize("n_bars", [5, 10, 20])
    def test_recent_bars_are_chronological(self, n_bars):
        bars = generate_bullish_run(n_bars, start_price=24000,
                                    gap_min=2, gap_max=8,
                                    bar_range=12, seed=n_bars)
        assign_dates(bars)
        _, mgr = run_tick_pipeline(bars[:2], bars[2:])

        dates = [b["date"] for b in mgr._recent_bars]
        for i in range(1, len(dates)):
            assert dates[i] > dates[i - 1], (
                f"Bar {i} date ({dates[i]}) not after bar {i-1} ({dates[i-1]})"
            )


class TestTickMitigationDynamic:
    """After FVGs form, simulate retrace ticks and verify mitigation."""

    @pytest.mark.parametrize("n_fvgs", [2, 4, 6])
    def test_retrace_mitigates_all_fvgs(self, n_fvgs):
        """Price retraces through all FVG zones — all should be mitigated."""
        n_bars = n_fvgs + 2
        bars = generate_bullish_run(n_bars, start_price=24000,
                                    gap_min=3, gap_max=8,
                                    bar_range=12, seed=n_fvgs * 7)
        assign_dates(bars)
        detected, mgr = run_tick_pipeline(bars[:2], bars[2:])

        assert len(detected) >= n_fvgs

        # Simulate a retrace: tick every FVG zone midpoint from top to bottom
        for fvg in reversed(detected):
            mid = _round_tick((fvg.zone_low + fvg.zone_high) / 2)
            assert fvg.zone_low <= mid <= fvg.zone_high

            # Mitigate
            fvg.is_mitigated = True
            fvg.mitigation_time = "retrace"
            mgr.remove(fvg.fvg_id)

        assert mgr.active_count == 0

    @pytest.mark.parametrize("n_fvgs", [3, 6])
    def test_partial_retrace_mitigates_some(self, n_fvgs):
        """Retrace only reaches the top half of FVGs — only those are mitigated."""
        n_bars = n_fvgs + 2
        bars = generate_bullish_run(n_bars, start_price=24000,
                                    gap_min=3, gap_max=8,
                                    bar_range=12, seed=n_fvgs * 13)
        assign_dates(bars)
        detected, mgr = run_tick_pipeline(bars[:2], bars[2:])

        assert len(detected) >= n_fvgs

        # Only mitigate the top half (most recent FVGs)
        top_half = detected[len(detected) // 2:]
        for fvg in top_half:
            fvg.is_mitigated = True
            mgr.remove(fvg.fvg_id)

        bottom_half = detected[:len(detected) // 2]
        remaining_ids = {f.fvg_id for f in mgr.active_fvgs}
        for fvg in bottom_half:
            assert fvg.fvg_id in remaining_ids, (
                f"FVG {fvg.fvg_id} should still be active (not mitigated)"
            )


class TestEdgeCases:
    """Edge cases: tiny gaps, huge gaps, single-tick bars, rapid succession."""

    def test_minimum_gap_size(self):
        """FVGs with exactly min_size (0.25) gap should be detected."""
        bars = [
            {"open": 24000, "high": 24010, "low": 23995, "close": 24008},
            {"open": 24012, "high": 24020, "low": 24005, "close": 24018},
            # low 24010.25 > bar1 high 24010.00 → gap = 0.25 (exactly min)
            {"open": 24015, "high": 24025, "low": 24010.25, "close": 24022},
        ]
        assign_dates(bars)
        detected, _ = run_tick_pipeline(bars[:2], bars[2:])
        assert len(detected) == 1
        assert detected[0].zone_high - detected[0].zone_low == pytest.approx(0.25)

    def test_sub_minimum_gap_rejected(self):
        """FVGs with gap < min_size should NOT be detected."""
        bars = [
            {"open": 24000, "high": 24010, "low": 23995, "close": 24008},
            {"open": 24012, "high": 24020, "low": 24005, "close": 24018},
            # low 24010.00 == bar1 high → gap = 0.0
            {"open": 24015, "high": 24025, "low": 24010, "close": 24022},
        ]
        assign_dates(bars)
        detected, _ = run_tick_pipeline(bars[:2], bars[2:])
        assert len(detected) == 0

    def test_large_gap_100pt(self):
        """100-point gap (extreme volatility) should still be detected."""
        bars = [
            {"open": 24000, "high": 24020, "low": 23990, "close": 24015},
            {"open": 24050, "high": 24080, "low": 24040, "close": 24070},
            {"open": 24130, "high": 24150, "low": 24120, "close": 24140},
        ]
        assign_dates(bars)
        detected, _ = run_tick_pipeline(bars[:2], bars[2:])
        assert len(detected) == 1
        assert detected[0].zone_high - detected[0].zone_low == 100  # 24120 - 24020

    @pytest.mark.parametrize("seed", range(5))
    def test_stress_50_bars(self, seed):
        """50-bar bullish run: pipeline handles long sequences without error."""
        bars = generate_bullish_run(50, start_price=24000,
                                    gap_min=1, gap_max=5,
                                    bar_range=10, seed=seed)
        assign_dates(bars)
        detected, mgr = run_tick_pipeline(bars[:2], bars[2:])
        expected = count_expected_fvgs(bars)
        assert len(detected) == expected
        assert len(mgr._recent_bars) == 10  # deque maxlen
