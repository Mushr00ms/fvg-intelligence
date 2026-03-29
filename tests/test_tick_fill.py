"""
Tests for tick data infrastructure: precompute_tick_arrays and enrich_volume parquet loading.

Covers:
- Tick array building from synthetic DataFrames
- Parquet round-trip (write → load)
- Volume at TP computation
- Fallback when parquet missing
- Integrity checks
"""

import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from bot.backtest.precompute_tick_arrays import build_tick_array, check_integrity
from bot.backtest.enrich_volume import (
    load_day_ticks_parquet,
    compute_volume_at_price,
    compute_volume_first_touch,
    compute_volume_through_price,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_raw_tick_df(n=100, symbol="NQH2", base_price=15000.0,
                      hour_start=10, hour_end=11):
    """Build a synthetic Databento-style tick DataFrame.

    Mimics the DataFrame returned by databento.DBNStore.to_df().
    """
    # Timestamps: evenly spaced within the given hour range (ET), stored as UTC
    et_start = pd.Timestamp(f"2022-03-15 {hour_start}:00:00",
                            tz="America/New_York").tz_convert("UTC")
    et_end = pd.Timestamp(f"2022-03-15 {hour_end}:00:00",
                          tz="America/New_York").tz_convert("UTC")
    ts_index = pd.date_range(et_start, et_end, periods=n)

    # Prices on 0.25-point grid
    rng = np.random.RandomState(42)
    price_offsets = rng.randint(-20, 21, size=n) * 0.25
    prices = base_price + price_offsets

    sizes = rng.randint(1, 50, size=n)
    sides = rng.choice(["A", "B"], size=n)

    return pd.DataFrame({
        "ts_event": ts_index,
        "price": prices,
        "size": sizes,
        "side": sides,
        "symbol": symbol,
    })


def _make_enrichment_df(ticks_data):
    """Build a DataFrame matching the format expected by compute_volume_* functions.

    ticks_data: list of (ts_utc_str, price, size, side_str, symbol)
    Returns DataFrame with ts_event as datetime64[ns, UTC] (same as Databento output).
    """
    ts_list, prices, sizes, sides, symbols = [], [], [], [], []
    for ts_str, price, size, side, symbol in ticks_data:
        ts_list.append(pd.Timestamp(ts_str, tz="UTC"))
        prices.append(price)
        sizes.append(size)
        sides.append(side)
        symbols.append(symbol)

    return pd.DataFrame({
        "ts_event": pd.DatetimeIndex(ts_list),
        "price": np.array(prices, dtype=np.float64),
        "size": np.array(sizes, dtype=np.int64),
        "side": sides,
        "symbol": symbols,
    })


# ── precompute_tick_arrays tests ─────────────────────────────────────────────


class TestBuildTickArray:
    def test_basic_conversion(self):
        """build_tick_array converts raw Databento format to compact format."""
        raw = _make_raw_tick_df(n=50, symbol="NQH2")
        result = build_tick_array(raw)

        assert result is not None
        assert set(result.columns) == {"ts_ns", "price", "size", "side", "symbol"}
        assert result["ts_ns"].dtype == np.int64
        assert result["price"].dtype == np.float64
        assert result["size"].dtype == np.int32
        assert result["side"].dtype == np.int8
        assert len(result) == 50

    def test_side_encoding(self):
        """'A' → 1, 'B' → -1."""
        raw = _make_raw_tick_df(n=10)
        result = build_tick_array(raw)

        for _, row in result.iterrows():
            orig = raw.loc[raw["ts_event"].astype("int64") == row["ts_ns"]]
            if not orig.empty:
                side_str = orig["side"].iloc[0]
                expected = 1 if side_str == "A" else -1
                assert row["side"] == expected

    def test_filters_non_nq(self):
        """Spread symbols (containing '-') and non-NQ symbols are excluded."""
        raw = _make_raw_tick_df(n=20, symbol="NQH2")
        spread_rows = _make_raw_tick_df(n=10, symbol="NQH2-NQM2")
        other_rows = _make_raw_tick_df(n=5, symbol="ESH2")
        combined = pd.concat([raw, spread_rows, other_rows], ignore_index=True)

        result = build_tick_array(combined)
        assert result is not None
        assert (result["symbol"] == "NQH2").all()
        assert len(result) == 20

    def test_filters_non_rth(self):
        """Ticks outside 09:30-16:00 ET are excluded."""
        # Pre-market ticks (08:00-09:00 ET)
        pre_mkt = _make_raw_tick_df(n=20, symbol="NQH2", hour_start=8, hour_end=9)
        # RTH ticks (10:00-11:00 ET)
        rth = _make_raw_tick_df(n=30, symbol="NQH2", hour_start=10, hour_end=11)
        combined = pd.concat([pre_mkt, rth], ignore_index=True)

        result = build_tick_array(combined)
        assert result is not None
        assert len(result) == 30  # Only RTH ticks

    def test_sorted_by_timestamp(self):
        """Output is sorted by ts_ns."""
        raw = _make_raw_tick_df(n=50)
        result = build_tick_array(raw)

        ts = result["ts_ns"].to_numpy()
        assert np.all(np.diff(ts) >= 0)

    def test_empty_input(self):
        """Empty DataFrame returns None."""
        empty = pd.DataFrame(columns=["ts_event", "price", "size", "side", "symbol"])
        assert build_tick_array(empty) is None


class TestIntegrity:
    def test_valid_ticks_pass(self):
        """Clean tick array passes integrity checks."""
        raw = _make_raw_tick_df(n=100000)
        result = build_tick_array(raw)
        errors = check_integrity(result, "2022-03-15")
        assert errors == []

    def test_low_tick_count_flagged(self):
        """Too few ticks are flagged."""
        raw = _make_raw_tick_df(n=100)
        result = build_tick_array(raw)
        errors = check_integrity(result, "2022-03-15")
        assert any("low tick count" in e for e in errors)


# ── Parquet round-trip tests ─────────────────────────────────────────────────


class TestParquetRoundTrip:
    def test_write_and_load(self):
        """Parquet written by build_tick_array is loadable by load_day_ticks_parquet."""
        raw = _make_raw_tick_df(n=50, symbol="NQH2")
        tick_df = build_tick_array(raw)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nq_ticks_20220315.parquet")
            tick_df.to_parquet(path, index=False)

            loaded = load_day_ticks_parquet("2022-03-15", tmpdir)

        assert loaded is not None
        assert "ts_event" in loaded.columns
        assert "symbol" in loaded.columns
        assert len(loaded) == 50

        # Side converted back to string
        assert set(loaded["side"].unique()).issubset({"A", "B", ""})

    def test_missing_file_returns_none(self):
        """Missing parquet returns None (graceful fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_day_ticks_parquet("2099-01-01", tmpdir)
        assert result is None


# ── Volume at TP tests ───────────────────────────────────────────────────────


class TestVolumeAtTP:
    def test_basic_volume_at_tp(self):
        """compute_volume_at_price sums contracts at TP price in time window."""
        ticks = _make_enrichment_df([
            ("2022-03-15 15:00:00", 15010.0, 5, "A", "NQH2"),   # at TP
            ("2022-03-15 15:00:01", 15010.0, 10, "B", "NQH2"),  # at TP
            ("2022-03-15 15:00:02", 15005.0, 20, "A", "NQH2"),  # different price
            ("2022-03-15 15:00:03", 15010.0, 7, "A", "NQH2"),   # at TP
        ])

        vol = compute_volume_at_price(
            ticks, "NQH2", 15010.0,
            datetime(2022, 3, 15, 14, 59, 0, tzinfo=timezone.utc),
            datetime(2022, 3, 15, 15, 1, 0, tzinfo=timezone.utc),
        )
        assert vol == 22  # 5 + 10 + 7

    def test_volume_at_tp_excludes_outside_window(self):
        """Ticks outside the time window are excluded."""
        ticks = _make_enrichment_df([
            ("2022-03-15 14:58:00", 15010.0, 100, "A", "NQH2"),  # before window
            ("2022-03-15 15:00:00", 15010.0, 5, "A", "NQH2"),    # in window
            ("2022-03-15 15:02:00", 15010.0, 100, "A", "NQH2"),  # after window
        ])

        vol = compute_volume_at_price(
            ticks, "NQH2", 15010.0,
            datetime(2022, 3, 15, 14, 59, 0, tzinfo=timezone.utc),
            datetime(2022, 3, 15, 15, 1, 0, tzinfo=timezone.utc),
        )
        assert vol == 5

    def test_volume_first_touch_at_tp(self):
        """compute_volume_first_touch works for TP price level."""
        ticks = _make_enrichment_df([
            ("2022-03-15 15:00:00.000", 15010.0, 5, "A", "NQH2"),
            ("2022-03-15 15:00:00.500", 15010.0, 10, "B", "NQH2"),
            ("2022-03-15 15:00:01.500", 15010.0, 100, "A", "NQH2"),  # next second
        ])

        vol = compute_volume_first_touch(
            ticks, "NQH2", 15010.0,
            datetime(2022, 3, 15, 15, 0, 0, tzinfo=timezone.utc),
        )
        assert vol == 15  # 5 + 10 (only first 1-second window)

    def test_no_volume_at_tp_returns_zero(self):
        """No ticks at TP price returns 0."""
        ticks = _make_enrichment_df([
            ("2022-03-15 15:00:00", 15005.0, 50, "A", "NQH2"),
        ])

        vol = compute_volume_at_price(
            ticks, "NQH2", 15010.0,
            datetime(2022, 3, 15, 14, 59, 0, tzinfo=timezone.utc),
            datetime(2022, 3, 15, 15, 1, 0, tzinfo=timezone.utc),
        )
        assert vol == 0

    def test_wrong_symbol_returns_none(self):
        """Ticks for a different symbol return None."""
        ticks = _make_enrichment_df([
            ("2022-03-15 15:00:00", 15010.0, 50, "A", "NQM2"),
        ])

        vol = compute_volume_at_price(
            ticks, "NQH2", 15010.0,
            datetime(2022, 3, 15, 14, 59, 0, tzinfo=timezone.utc),
            datetime(2022, 3, 15, 15, 1, 0, tzinfo=timezone.utc),
        )
        assert vol is None


# ── Volume through TP tests ──────────────────────────────────────────────────


class TestVolumeThroughTP:
    def test_buy_counts_at_or_above(self):
        """BUY position: TP sell limit fills at or above target."""
        ticks = _make_enrichment_df([
            ("2022-03-15 15:00:00", 15010.0, 5, "A", "NQH2"),   # at TP
            ("2022-03-15 15:00:01", 15010.25, 10, "B", "NQH2"), # above TP
            ("2022-03-15 15:00:02", 15015.0, 20, "A", "NQH2"),  # well above TP
            ("2022-03-15 15:00:03", 15005.0, 100, "A", "NQH2"), # below TP — excluded
        ])

        vol = compute_volume_through_price(
            ticks, "NQH2", 15010.0, "BUY",
            datetime(2022, 3, 15, 14, 59, 0, tzinfo=timezone.utc),
            datetime(2022, 3, 15, 15, 1, 0, tzinfo=timezone.utc),
        )
        assert vol == 35  # 5 + 10 + 20

    def test_sell_counts_at_or_below(self):
        """SELL position: TP buy limit fills at or below target."""
        ticks = _make_enrichment_df([
            ("2022-03-15 15:00:00", 15010.0, 5, "A", "NQH2"),   # at TP
            ("2022-03-15 15:00:01", 15009.75, 10, "B", "NQH2"), # below TP
            ("2022-03-15 15:00:02", 15000.0, 20, "A", "NQH2"),  # well below TP
            ("2022-03-15 15:00:03", 15015.0, 100, "A", "NQH2"), # above TP — excluded
        ])

        vol = compute_volume_through_price(
            ticks, "NQH2", 15010.0, "SELL",
            datetime(2022, 3, 15, 14, 59, 0, tzinfo=timezone.utc),
            datetime(2022, 3, 15, 15, 1, 0, tzinfo=timezone.utc),
        )
        assert vol == 35  # 5 + 10 + 20

    def test_none_below_for_buy(self):
        """BUY with all ticks below TP returns 0."""
        ticks = _make_enrichment_df([
            ("2022-03-15 15:00:00", 15005.0, 50, "A", "NQH2"),
            ("2022-03-15 15:00:01", 15008.0, 30, "B", "NQH2"),
        ])

        vol = compute_volume_through_price(
            ticks, "NQH2", 15010.0, "BUY",
            datetime(2022, 3, 15, 14, 59, 0, tzinfo=timezone.utc),
            datetime(2022, 3, 15, 15, 1, 0, tzinfo=timezone.utc),
        )
        assert vol == 0

    def test_respects_time_window(self):
        """Ticks outside the time window are excluded."""
        ticks = _make_enrichment_df([
            ("2022-03-15 14:58:00", 15015.0, 100, "A", "NQH2"),  # before
            ("2022-03-15 15:00:00", 15015.0, 5, "A", "NQH2"),    # in window
            ("2022-03-15 15:02:00", 15015.0, 100, "A", "NQH2"),  # after
        ])

        vol = compute_volume_through_price(
            ticks, "NQH2", 15010.0, "BUY",
            datetime(2022, 3, 15, 14, 59, 0, tzinfo=timezone.utc),
            datetime(2022, 3, 15, 15, 1, 0, tzinfo=timezone.utc),
        )
        assert vol == 5


# ── Parquet-loaded DataFrame works with compute functions ────────────────────


class TestParquetComputeCompat:
    def test_volume_at_price_from_parquet(self):
        """compute_volume_at_price works with DataFrames loaded from parquet."""
        raw = _make_raw_tick_df(n=200, symbol="NQH2", base_price=15000.0,
                                hour_start=10, hour_end=11)
        tick_df = build_tick_array(raw)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nq_ticks_20220315.parquet")
            tick_df.to_parquet(path, index=False)
            loaded = load_day_ticks_parquet("2022-03-15", tmpdir)

        # Pick a price that exists in the data
        test_price = loaded["price"].iloc[0]

        vol = compute_volume_at_price(
            loaded, "NQH2", test_price,
            datetime(2022, 3, 15, 14, 0, 0, tzinfo=timezone.utc),
            datetime(2022, 3, 15, 16, 0, 0, tzinfo=timezone.utc),
        )
        assert vol is not None
        assert vol >= 0
