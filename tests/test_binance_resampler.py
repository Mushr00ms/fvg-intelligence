"""Tests for Binance aggTrades resampler and hardened integrity checker."""

import os
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from logic.utils.binance_resampler import (
    BinanceAggTradesResampler,
    BinanceResamplerIntegrityChecker,
    IntegrityResult,
    IntegrityReport,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def resampler(tmp_path):
    return BinanceAggTradesResampler(
        raw_dir=str(tmp_path / "raw"),
        output_dir=str(tmp_path / "output"),
        backtester_dir=str(tmp_path / "backtester"),
    )


@pytest.fixture
def checker():
    return BinanceResamplerIntegrityChecker()


@pytest.fixture
def sample_csv(tmp_path):
    """Create a small synthetic aggTrades CSV and return its path."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "BTCUSDT-aggTrades-2024-01-15.csv"

    # 10 trades over 5 seconds starting at 2024-01-15 00:00:00 UTC
    base_ts = 1705276800000  # 2024-01-15 00:00:00 UTC in ms
    rows = [
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker",
        f"1,42000.0,0.5,1,1,{base_ts},false",          # sec 0
        f"2,42001.0,0.3,2,2,{base_ts + 200},true",      # sec 0
        f"3,42005.0,1.0,3,3,{base_ts + 1000},false",    # sec 1
        f"4,41990.0,0.2,4,4,{base_ts + 1500},true",     # sec 1
        f"5,42010.0,0.8,5,5,{base_ts + 2000},false",    # sec 2
        f"6,42020.0,0.4,6,6,{base_ts + 3000},false",    # sec 3
        f"7,42015.0,0.6,7,7,{base_ts + 3500},true",     # sec 3
        # sec 4 — no trades (gap)
        f"8,42025.0,0.1,8,8,{base_ts + 5000},false",    # sec 5
        f"9,42030.0,0.9,9,9,{base_ts + 5200},false",    # sec 5
        f"10,42028.0,0.7,10,10,{base_ts + 5800},true",  # sec 5
    ]
    csv_path.write_text("\n".join(rows) + "\n")
    return csv_path


@pytest.fixture
def sample_trades(resampler, sample_csv):
    """Load the sample CSV into a DataFrame."""
    return resampler.load_aggtrades(sample_csv)


# ── Loading ───────────────────────────────────────────────────────────────

class TestLoadAggTrades:
    def test_columns_and_dtypes(self, sample_trades):
        assert list(sample_trades.columns) == ["price", "quantity", "is_buyer_maker", "timestamp"]
        assert sample_trades["price"].dtype == np.float64
        assert sample_trades["quantity"].dtype == np.float64
        assert sample_trades["timestamp"].dt.tz is not None  # UTC-aware

    def test_row_count(self, sample_trades):
        assert len(sample_trades) == 10

    def test_sorted_by_timestamp(self, sample_trades):
        ts = sample_trades["timestamp"]
        assert (ts.diff().dropna() >= pd.Timedelta(0)).all()

    def test_first_trade_values(self, sample_trades):
        first = sample_trades.iloc[0]
        assert first["price"] == 42000.0
        assert first["quantity"] == 0.5


# ── Resampling to 1s ─────────────────────────────────────────────────────

class TestResampleTo1s:
    def test_candle_count(self, resampler, sample_trades):
        """Should produce 5 candles (sec 0,1,2,3,5 — sec 4 has no trades)."""
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        assert len(candles) == 5

    def test_ohlcv_sec0(self, resampler, sample_trades):
        """Sec 0 has trades at 42000.0 and 42001.0, qty 0.5+0.3=0.8."""
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        sec0 = candles.iloc[0]
        assert sec0["open"] == 42000.0
        assert sec0["high"] == 42001.0
        assert sec0["low"] == 42000.0
        assert sec0["close"] == 42001.0
        assert math.isclose(sec0["volume"], 0.8, rel_tol=1e-9)

    def test_ohlcv_sec1(self, resampler, sample_trades):
        """Sec 1 has trades at 42005.0 and 41990.0."""
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        sec1 = candles.iloc[1]
        assert sec1["open"] == 42005.0
        assert sec1["high"] == 42005.0
        assert sec1["low"] == 41990.0
        assert sec1["close"] == 41990.0
        assert math.isclose(sec1["volume"], 1.2, rel_tol=1e-9)

    def test_ohlcv_sec5(self, resampler, sample_trades):
        """Sec 5 has 3 trades: 42025, 42030, 42028."""
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        sec5 = candles.iloc[4]
        assert sec5["open"] == 42025.0
        assert sec5["high"] == 42030.0
        assert sec5["low"] == 42025.0
        assert sec5["close"] == 42028.0
        assert math.isclose(sec5["volume"], 1.7, rel_tol=1e-9)


# ── Resampling to 5s ─────────────────────────────────────────────────────

class TestResampleTo5s:
    def test_candle_count_5s(self, resampler, sample_trades):
        """10 trades span 0–5.8s. With 5s bins: [0-5s) and [5-10s) → 2 candles."""
        candles = resampler.resample_to_ohlcv(sample_trades, "5s")
        assert len(candles) == 2

    def test_first_5s_candle(self, resampler, sample_trades):
        """First 5s bin [0-5s) covers sec 0,1,2,3: open=42000, close=42015."""
        candles = resampler.resample_to_ohlcv(sample_trades, "5s")
        c0 = candles.iloc[0]
        assert c0["open"] == 42000.0
        assert c0["high"] == 42020.0
        assert c0["low"] == 41990.0
        assert c0["close"] == 42015.0
        expected_vol = 0.5 + 0.3 + 1.0 + 0.2 + 0.8 + 0.4 + 0.6
        assert math.isclose(c0["volume"], expected_vol, rel_tol=1e-9)


# ── Empty Intervals ───────────────────────────────────────────────────────

class TestEmptyIntervals:
    def test_gap_produces_no_candle(self, resampler, sample_trades):
        """Second 4 has no trades → no candle at that timestamp."""
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        timestamps = candles.index.tolist()
        base = pd.Timestamp("2024-01-15 00:00:04", tz="UTC")
        assert base not in timestamps

    def test_empty_trades_returns_empty(self, resampler):
        empty = pd.DataFrame(columns=["timestamp", "price", "quantity"])
        candles = resampler.resample_to_ohlcv(empty, "1s")
        assert candles.empty


# ── Integrity: Volume Conservation ────────────────────────────────────────

class TestVolumeConservation:
    def test_exact_match(self, checker, resampler, sample_trades):
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        result = checker.check_volume_conservation(sample_trades, candles)
        assert result.passed
        assert result.details["abs_diff"] < 1e-10

    def test_mismatch_fails(self, checker, sample_trades):
        """Corrupt candle volumes → should fail."""
        candles = pd.DataFrame({
            "open": [42000.0],
            "high": [42001.0],
            "low": [42000.0],
            "close": [42001.0],
            "volume": [999.0],  # wrong
        })
        result = checker.check_volume_conservation(sample_trades, candles)
        assert not result.passed

    def test_5s_volume_conservation(self, checker, resampler, sample_trades):
        """Volume must be conserved at any resample interval."""
        candles_5s = resampler.resample_to_ohlcv(sample_trades, "5s")
        result = checker.check_volume_conservation(sample_trades, candles_5s)
        assert result.passed


# ── Integrity: Price Bounds ───────────────────────────────────────────────

class TestPriceBounds:
    def test_valid_candles(self, checker, resampler, sample_trades):
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        result = checker.check_price_bounds(candles)
        assert result.passed
        assert result.details["violation_count"] == 0

    def test_invalid_high(self, checker):
        """High below open should be caught."""
        candles = pd.DataFrame({
            "open": [100.0],
            "high": [99.0],   # violation: high < open
            "low": [98.0],
            "close": [99.5],
            "volume": [1.0],
        })
        result = checker.check_price_bounds(candles)
        assert not result.passed
        assert result.details["violation_count"] == 1

    def test_invalid_low(self, checker):
        """Low above close should be caught."""
        candles = pd.DataFrame({
            "open": [100.0],
            "high": [105.0],
            "low": [101.0],   # violation: low > open and low > close
            "close": [99.0],
            "volume": [1.0],
        })
        result = checker.check_price_bounds(candles)
        assert not result.passed


# ── Integrity: First/Last Alignment ──────────────────────────────────────

class TestFirstLastAlignment:
    def test_alignment_correct(self, checker, resampler, sample_trades):
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        result = checker.check_first_last_alignment(sample_trades, candles)
        assert result.passed
        assert result.details["first_trade_price"] == 42000.0
        assert result.details["first_candle_open"] == 42000.0
        assert result.details["last_trade_price"] == 42028.0
        assert result.details["last_candle_close"] == 42028.0

    def test_alignment_at_5s(self, checker, resampler, sample_trades):
        """First/last alignment must hold at any interval."""
        candles = resampler.resample_to_ohlcv(sample_trades, "5s")
        result = checker.check_first_last_alignment(sample_trades, candles)
        assert result.passed


# ── Integrity: Record Count ──────────────────────────────────────────────

class TestRecordCount:
    def test_reasonable_count(self, checker, resampler, sample_trades):
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        # Data spans ~6 seconds
        result = checker.check_record_count(candles, "1s", duration_hours=6 / 3600)
        assert result.passed
        assert result.details["actual"] == 5
        assert result.details["theoretical_max"] == 6

    def test_suspiciously_sparse(self, checker):
        """Only 1 candle in a 24h window should fail (< 50% fill)."""
        candles = pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1.0],
        })
        result = checker.check_record_count(candles, "1s", duration_hours=24.0)
        assert not result.passed
        assert result.details["fill_pct"] < 1.0


# ── Integrity: Cross-Resolution Consistency ──────────────────────────────

class TestCrossResolution:
    def test_1s_to_5s_consistency(self, checker, resampler, sample_trades):
        """Resampling 1s candles to 5s must match direct 5s resample from trades."""
        candles_1s = resampler.resample_to_ohlcv(sample_trades, "1s")
        result = checker.check_cross_resolution_consistency(
            sample_trades, candles_1s, "5s"
        )
        assert result.passed, f"Cross-resolution check failed: {result.message}"
        assert result.details["bars_compared"] > 0


# ── Integrity: Timestamp Continuity ──────────────────────────────────────

class TestTimestampContinuity:
    def test_reports_missing_intervals(self, checker, resampler, sample_trades):
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        result = checker.check_timestamp_continuity(candles, "1s")
        # sec 4 is missing → at least 1 gap
        assert result.passed  # gaps are informational, not failures
        assert result.details["missing_count"] >= 1


# ── Integrity: run_all_checks ─────────────────────────────────────────────

class TestRunAllChecks:
    def test_all_pass_on_valid_data(self, checker, resampler, sample_trades):
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        report = checker.run_all_checks(sample_trades, candles, "1s", duration_hours=6 / 3600)
        assert report.all_passed, f"Some checks failed: {report.summary}"
        assert len(report.results) == 5

    def test_summary_format(self, checker, resampler, sample_trades):
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        report = checker.run_all_checks(sample_trades, candles, "1s", duration_hours=6 / 3600)
        assert "5/5 checks passed" in report.summary


# ── Backtester Parquet Schema ─────────────────────────────────────────────

class TestBacktesterSchema:
    def test_parquet_columns_and_dtypes(self, resampler, sample_trades, tmp_path):
        """Output parquet must match the NQ schema: date(str), OHLCV(float64)."""
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        out_path = tmp_path / "btcusdt_1secs_20240115.parquet"
        resampler._write_parquet(candles, out_path, "backtester")

        df = pd.read_parquet(out_path)
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert pd.api.types.is_string_dtype(df["date"])
        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == np.float64, f"{col} should be float64"

    def test_parquet_date_format(self, resampler, sample_trades, tmp_path):
        """Date strings must look like '2024-01-15 00:00:00+00:00'."""
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        out_path = tmp_path / "btcusdt_1secs_20240115.parquet"
        resampler._write_parquet(candles, out_path, "backtester")

        df = pd.read_parquet(out_path)
        first_date = df["date"].iloc[0]
        assert "+00:00" in first_date
        assert "2024-01-15" in first_date

    def test_parquet_roundtrip_values(self, resampler, sample_trades, tmp_path):
        """Values must survive write → read roundtrip."""
        candles = resampler.resample_to_ohlcv(sample_trades, "1s")
        out_path = tmp_path / "btcusdt_1secs_20240115.parquet"
        resampler._write_parquet(candles, out_path, "backtester")

        df = pd.read_parquet(out_path)
        assert df.iloc[0]["open"] == 42000.0
        assert df.iloc[0]["high"] == 42001.0
        assert math.isclose(df.iloc[0]["volume"], 0.8, rel_tol=1e-9)


# ── Monthly CSV Splitting ─────────────────────────────────────────────────

class TestMonthlyCSVSplit:
    def test_splits_by_date(self, resampler, tmp_path):
        """Monthly CSV with trades on 2 different days should produce 2 entries."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        csv_path = raw_dir / "BTCUSDT-aggTrades-2024-01.csv"

        # Day 1: 2024-01-15 00:00:00 UTC = 1705276800000
        # Day 2: 2024-01-16 00:00:00 UTC = 1705363200000
        ts1 = 1705276800000
        ts2 = 1705363200000
        rows = [
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker",
            f"1,42000.0,0.5,1,1,{ts1},false",
            f"2,42001.0,0.3,2,2,{ts1 + 1000},true",
            f"3,43000.0,1.0,3,3,{ts2},false",
            f"4,43010.0,0.2,4,4,{ts2 + 500},true",
        ]
        csv_path.write_text("\n".join(rows) + "\n")

        resampler.raw_dir = raw_dir
        result = resampler.split_monthly_csv(csv_path)

        assert len(result) == 2
        assert date(2024, 1, 15) in result
        assert date(2024, 1, 16) in result
        assert len(result[date(2024, 1, 15)]) == 2
        assert len(result[date(2024, 1, 16)]) == 2
