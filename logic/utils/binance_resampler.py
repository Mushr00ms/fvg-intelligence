"""
binance_resampler.py — Convert Binance aggTrades to OHLCV candles with hardened integrity checks.

Resamples raw aggTrades CSV data into candles at arbitrary intervals (1s, 5s, 15s,
30s, 1min, 5min, 15min) and writes backtester-compatible Parquet files.

Usage:
    from logic.utils.binance_resampler import BinanceAggTradesResampler

    rs = BinanceAggTradesResampler()
    report = rs.process_range(date(2024, 1, 1), date(2024, 1, 31), interval="1s")
    print(report)
"""

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class IntegrityResult:
    """Result of a single integrity check."""
    check_name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class IntegrityReport:
    """Aggregate result of all integrity checks for a single resample."""
    results: list = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        failed = [r.check_name for r in self.results if not r.passed]
        s = f"{passed}/{total} checks passed"
        if failed:
            s += f" — FAILED: {', '.join(failed)}"
        return s


@dataclass
class ProcessingReport:
    """Summary of a process_range() run."""
    days_processed: int = 0
    days_skipped: int = 0
    days_failed: int = 0
    integrity_failures: int = 0
    total_trades: int = 0
    total_candles: int = 0
    errors: list = field(default_factory=list)
    integrity_reports: dict = field(default_factory=dict)  # date_str -> IntegrityReport


# ── Integrity Checker ─────────────────────────────────────────────────────

class BinanceResamplerIntegrityChecker:
    """Comprehensive integrity verification for aggTrades → OHLCV conversion."""

    def check_volume_conservation(
        self, trades: pd.DataFrame, candles: pd.DataFrame
    ) -> IntegrityResult:
        """Verify total trade quantity equals total candle volume.

        Uses rel_tol=1e-9 to account for float64 summation order differences.
        """
        trade_vol = float(trades["quantity"].sum())
        candle_vol = float(candles["volume"].sum())

        if trade_vol == 0 and candle_vol == 0:
            return IntegrityResult(
                "volume_conservation", True, "Both zero (no trades)", {}
            )

        passed = math.isclose(trade_vol, candle_vol, rel_tol=1e-9)
        diff = abs(trade_vol - candle_vol)
        return IntegrityResult(
            check_name="volume_conservation",
            passed=passed,
            message=f"trade_vol={trade_vol:.10f}, candle_vol={candle_vol:.10f}, diff={diff:.2e}",
            details={
                "trade_volume": trade_vol,
                "candle_volume": candle_vol,
                "abs_diff": diff,
            },
        )

    def check_price_bounds(self, candles: pd.DataFrame) -> IntegrityResult:
        """Verify OHLC relationships: high >= max(open,close), low <= min(open,close), high >= low."""
        violations = []

        high_vs_open = candles["high"] < candles["open"]
        high_vs_close = candles["high"] < candles["close"]
        low_vs_open = candles["low"] > candles["open"]
        low_vs_close = candles["low"] > candles["close"]
        high_vs_low = candles["high"] < candles["low"]

        mask = high_vs_open | high_vs_close | low_vs_open | low_vs_close | high_vs_low
        n_violations = int(mask.sum())

        if n_violations > 0:
            bad_indices = candles.index[mask].tolist()[:10]  # first 10
            violations = bad_indices

        return IntegrityResult(
            check_name="price_bounds",
            passed=n_violations == 0,
            message=f"{n_violations} candle(s) with invalid OHLC relationships",
            details={"violation_count": n_violations, "sample_indices": violations},
        )

    def check_timestamp_continuity(
        self, candles: pd.DataFrame, interval: str, expected_start: Optional[pd.Timestamp] = None,
        expected_end: Optional[pd.Timestamp] = None,
    ) -> IntegrityResult:
        """Check for gaps in the candle timestamp grid.

        BTC is 24/7, so gaps indicate seconds/intervals with zero trades (expected)
        or data issues (unexpected). Reports gap count for awareness.
        """
        if candles.empty:
            return IntegrityResult(
                "timestamp_continuity", True, "No candles to check", {}
            )

        idx = candles.index if isinstance(candles.index, pd.DatetimeIndex) else pd.DatetimeIndex(candles["date"])
        start = expected_start or idx.min()
        end = expected_end or idx.max()

        expected = pd.date_range(start=start, end=end, freq=interval)
        actual_set = set(idx)
        missing = sorted(set(expected) - actual_set)

        return IntegrityResult(
            check_name="timestamp_continuity",
            passed=True,  # gaps are informational for BTC (zero-trade intervals)
            message=f"{len(missing)} missing intervals out of {len(expected)} expected "
                    f"({100 * len(missing) / max(len(expected), 1):.1f}% empty)",
            details={
                "expected_count": len(expected),
                "actual_count": len(actual_set),
                "missing_count": len(missing),
                "missing_pct": 100 * len(missing) / max(len(expected), 1),
            },
        )

    def check_cross_resolution_consistency(
        self,
        trades: pd.DataFrame,
        candles_fine: pd.DataFrame,
        coarse_interval: str,
    ) -> IntegrityResult:
        """Verify fine candles re-aggregated to coarser interval match direct resample.

        Resamples candles_fine → coarse_interval, then resamples trades → coarse_interval,
        and compares OHLCV values.
        """
        # Direct resample from trades
        trades_indexed = trades.set_index("timestamp") if "timestamp" in trades.columns else trades
        direct = trades_indexed["price"].resample(coarse_interval, label="left", closed="left").agg(
            open="first", high="max", low="min", close="last"
        )
        direct["volume"] = trades_indexed["quantity"].resample(
            coarse_interval, label="left", closed="left"
        ).sum()
        direct = direct.dropna(subset=["open"])

        # Re-aggregate from fine candles
        fine = candles_fine.set_index(candles_fine.index if isinstance(candles_fine.index, pd.DatetimeIndex) else pd.DatetimeIndex(candles_fine["date"]))
        reagg = fine.resample(coarse_interval, label="left", closed="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna(subset=["open"])

        # Align indices for comparison
        common = direct.index.intersection(reagg.index)
        if len(common) == 0:
            return IntegrityResult(
                "cross_resolution_consistency", False,
                "No overlapping intervals to compare", {}
            )

        d = direct.loc[common]
        r = reagg.loc[common]

        mismatches = {}
        for col in ["open", "high", "low", "close"]:
            diffs = (d[col] - r[col]).abs()
            n_bad = int((diffs > 1e-10).sum())
            if n_bad > 0:
                mismatches[col] = n_bad

        vol_diffs = (d["volume"] - r["volume"]).abs()
        vol_bad = int((vol_diffs > d["volume"].abs() * 1e-9 + 1e-15).sum())
        if vol_bad > 0:
            mismatches["volume"] = vol_bad

        passed = len(mismatches) == 0
        return IntegrityResult(
            check_name="cross_resolution_consistency",
            passed=passed,
            message=f"Compared {len(common)} coarse bars — mismatches: {mismatches or 'none'}",
            details={
                "bars_compared": len(common),
                "mismatches": mismatches,
            },
        )

    def check_first_last_alignment(
        self, trades: pd.DataFrame, candles: pd.DataFrame
    ) -> IntegrityResult:
        """Verify first candle open == first trade price, last candle close == last trade price."""
        if trades.empty or candles.empty:
            return IntegrityResult(
                "first_last_alignment", True, "Empty data", {}
            )

        first_trade_price = float(trades["price"].iloc[0])
        last_trade_price = float(trades["price"].iloc[-1])
        first_candle_open = float(candles["open"].iloc[0])
        last_candle_close = float(candles["close"].iloc[-1])

        open_match = first_trade_price == first_candle_open
        close_match = last_trade_price == last_candle_close

        return IntegrityResult(
            check_name="first_last_alignment",
            passed=open_match and close_match,
            message=(
                f"first_trade={first_trade_price}, first_open={first_candle_open} "
                f"({'OK' if open_match else 'MISMATCH'}); "
                f"last_trade={last_trade_price}, last_close={last_candle_close} "
                f"({'OK' if close_match else 'MISMATCH'})"
            ),
            details={
                "first_trade_price": first_trade_price,
                "first_candle_open": first_candle_open,
                "last_trade_price": last_trade_price,
                "last_candle_close": last_candle_close,
            },
        )

    def check_record_count(
        self, candles: pd.DataFrame, interval: str, duration_hours: float
    ) -> IntegrityResult:
        """Verify candle count is within reasonable bounds for the time span.

        For 24h of 1s bars, theoretical max = 86400. Actual is <= max (empty intervals
        produce no candle). Flags if actual < 50% of max (suspiciously sparse).
        """
        interval_seconds = pd.tseries.frequencies.to_offset(interval).nanos / 1e9
        theoretical_max = int(duration_hours * 3600 / interval_seconds)
        actual = len(candles)

        if theoretical_max == 0:
            return IntegrityResult(
                "record_count", True, "Zero expected bars", {}
            )

        fill_pct = 100 * actual / theoretical_max
        passed = actual <= theoretical_max and fill_pct >= 50

        return IntegrityResult(
            check_name="record_count",
            passed=passed,
            message=f"{actual} candles / {theoretical_max} theoretical max ({fill_pct:.1f}% fill)",
            details={
                "actual": actual,
                "theoretical_max": theoretical_max,
                "fill_pct": fill_pct,
            },
        )

    def run_all_checks(
        self,
        trades: pd.DataFrame,
        candles: pd.DataFrame,
        interval: str,
        duration_hours: float = 24.0,
    ) -> IntegrityReport:
        """Run all single-resolution integrity checks."""
        report = IntegrityReport()
        report.results.append(self.check_volume_conservation(trades, candles))
        report.results.append(self.check_price_bounds(candles))
        report.results.append(self.check_timestamp_continuity(candles, interval))
        report.results.append(self.check_first_last_alignment(trades, candles))
        report.results.append(self.check_record_count(candles, interval, duration_hours))
        return report


# ── Resampler ─────────────────────────────────────────────────────────────

class BinanceAggTradesResampler:
    """Convert Binance aggTrades CSV files to OHLCV candles with integrity verification."""

    AGGTRADES_COLUMNS = [
        "agg_trade_id", "price", "quantity", "first_trade_id",
        "last_trade_id", "transact_time", "is_buyer_maker",
    ]
    AGGTRADES_DTYPES = {
        "agg_trade_id": "int64",
        "price": "float64",
        "quantity": "float64",
        "first_trade_id": "int64",
        "last_trade_id": "int64",
        "transact_time": "int64",
        "is_buyer_maker": "bool",
    }

    def __init__(
        self,
        raw_dir: str = "logic/binance_data/raw",
        output_dir: str = "logic/binance_data",
        backtester_dir: str = "bot/data",
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.backtester_dir = Path(backtester_dir)
        self.checker = BinanceResamplerIntegrityChecker()

    # ── Loading ───────────────────────────────────────────────────────────

    @staticmethod
    def _has_header(csv_path: Path) -> bool:
        """Detect whether the CSV has a header row (vs raw numeric data)."""
        with open(csv_path, "r") as f:
            first_line = f.readline().strip()
        return first_line.startswith("agg_trade_id")

    def _csv_read_kwargs(self, csv_path: Path, extra: dict = None) -> dict:
        """Build pd.read_csv kwargs handling both header and headerless CSVs."""
        kwargs = {"dtype": self.AGGTRADES_DTYPES}
        if self._has_header(csv_path):
            kwargs["usecols"] = ["price", "quantity", "transact_time", "is_buyer_maker"]
        else:
            kwargs["header"] = None
            kwargs["names"] = self.AGGTRADES_COLUMNS
            kwargs["usecols"] = ["price", "quantity", "transact_time", "is_buyer_maker"]
        if extra:
            kwargs.update(extra)
        return kwargs

    def load_aggtrades(self, csv_path: Path) -> pd.DataFrame:
        """Load a Binance aggTrades CSV into a DataFrame.

        Handles both header and headerless CSVs (Binance changed format over time).
        Returns DataFrame with columns: timestamp (UTC datetime), price, quantity, is_buyer_maker.
        Sorted by timestamp.
        """
        df = pd.read_csv(csv_path, **self._csv_read_kwargs(csv_path))
        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
        df = df.drop(columns=["transact_time"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def load_aggtrades_chunked(self, csv_path: Path, chunksize: int = 5_000_000):
        """Yield chunks of aggTrades from a large CSV (memory-safe for monthly files).

        Each chunk is a DataFrame with columns: timestamp, price, quantity, is_buyer_maker.
        """
        kwargs = self._csv_read_kwargs(csv_path, extra={"chunksize": chunksize})
        reader = pd.read_csv(csv_path, **kwargs)
        for chunk in reader:
            chunk["timestamp"] = pd.to_datetime(chunk["transact_time"], unit="ms", utc=True)
            chunk = chunk.drop(columns=["transact_time"])
            yield chunk

    # ── Resampling ────────────────────────────────────────────────────────

    def resample_to_ohlcv(self, trades: pd.DataFrame, interval: str) -> pd.DataFrame:
        """Resample aggTrades to OHLCV candles at the given interval.

        Args:
            trades: DataFrame with columns [timestamp, price, quantity].
            interval: Pandas frequency string (e.g., '1s', '5s', '1min', '5min').

        Returns:
            DataFrame with DatetimeIndex (UTC) and columns [open, high, low, close, volume].
            Intervals with no trades are excluded.
        """
        if trades.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        indexed = trades.set_index("timestamp")

        candles = indexed["price"].resample(interval, label="left", closed="left").agg(
            open="first", high="max", low="min", close="last"
        )
        candles["volume"] = indexed["quantity"].resample(
            interval, label="left", closed="left"
        ).sum()

        candles = candles.dropna(subset=["open"])
        return candles

    # ── Batch Processing ──────────────────────────────────────────────────

    def process_range(
        self,
        start_date: date,
        end_date: date,
        interval: str = "1s",
        output_format: str = "backtester",
        symbol: str = "btcusdt",
        verify: bool = True,
    ) -> ProcessingReport:
        """Process a date range: load CSVs → resample → verify → write Parquet.

        Args:
            start_date, end_date: Date range (inclusive).
            interval: Candle interval (e.g., '1s', '5s', '1min').
            output_format: 'backtester' writes {symbol}_1secs_YYYYMMDD.parquet to backtester_dir.
                           'analysis' writes {symbol}_{interval}_YYYYMMDD.parquet to output_dir.
            symbol: Symbol prefix for output filenames.
            verify: Run integrity checks on each day.
        """
        report = ProcessingReport()
        current = start_date

        while current <= end_date:
            date_str = current.strftime("%Y%m%d")
            date_iso = current.strftime("%Y-%m-%d")

            # Determine output path and check if already exists
            out_path = self._output_path(current, interval, output_format, symbol)
            if out_path.exists():
                logger.info("[%s] Skipped (output exists)", date_iso)
                report.days_skipped += 1
                current += timedelta(days=1)
                continue

            # Find source CSV (daily or monthly)
            trades_df = self._load_day_trades(current, symbol)
            if trades_df is None:
                logger.debug("[%s] No source data", date_iso)
                report.days_skipped += 1
                current += timedelta(days=1)
                continue

            if trades_df.empty:
                logger.debug("[%s] Empty trades", date_iso)
                report.days_skipped += 1
                current += timedelta(days=1)
                continue

            logger.info("[%s] %d trades → resampling to %s", date_iso, len(trades_df), interval)
            report.total_trades += len(trades_df)

            # Resample
            candles = self.resample_to_ohlcv(trades_df, interval)
            report.total_candles += len(candles)

            # Integrity checks
            if verify:
                duration_hours = 24.0
                ir = self.checker.run_all_checks(trades_df, candles, interval, duration_hours)
                report.integrity_reports[date_iso] = ir
                if not ir.all_passed:
                    report.integrity_failures += 1
                    logger.warning("[%s] INTEGRITY CHECK FAILED: %s", date_iso, ir.summary)
                    report.errors.append(f"{date_iso}: {ir.summary}")
                else:
                    logger.info("[%s] Integrity OK: %s", date_iso, ir.summary)

            # Write output
            self._write_parquet(candles, out_path, output_format)
            report.days_processed += 1
            logger.info("[%s] Wrote %d candles → %s", date_iso, len(candles), out_path.name)

            current += timedelta(days=1)

        logger.info(
            "Done: %d processed, %d skipped, %d failed, %d integrity failures, "
            "%d total trades → %d total candles",
            report.days_processed, report.days_skipped, report.days_failed,
            report.integrity_failures, report.total_trades, report.total_candles,
        )
        return report

    # ── Day Extraction from Monthly CSVs ──────────────────────────────────

    def split_monthly_csv(self, csv_path: Path) -> dict:
        """Split a monthly aggTrades CSV into per-day DataFrames (memory-safe).

        Returns {date: DataFrame} where each DataFrame has columns
        [timestamp, price, quantity, is_buyer_maker].
        """
        day_frames = {}
        for chunk in self.load_aggtrades_chunked(csv_path):
            chunk["trade_date"] = chunk["timestamp"].dt.date
            for day, group in chunk.groupby("trade_date"):
                clean = group.drop(columns=["trade_date"])
                if day in day_frames:
                    day_frames[day] = pd.concat([day_frames[day], clean], ignore_index=True)
                else:
                    day_frames[day] = clean.reset_index(drop=True)
        return day_frames

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _load_day_trades(self, target_date: date, symbol: str) -> Optional[pd.DataFrame]:
        """Load trades for a specific day from daily CSV or monthly CSV."""
        sym = symbol.upper()

        # Try daily CSV first
        daily_name = f"{sym}-aggTrades-{target_date.strftime('%Y-%m-%d')}.csv"
        daily_path = self.raw_dir / daily_name
        if daily_path.exists():
            df = self.load_aggtrades(daily_path)
            return df[df["timestamp"].dt.date == target_date].reset_index(drop=True)

        # Try monthly CSV
        monthly_name = f"{sym}-aggTrades-{target_date.strftime('%Y-%m')}.csv"
        monthly_path = self.raw_dir / monthly_name
        if monthly_path.exists():
            for chunk in self.load_aggtrades_chunked(monthly_path):
                day_mask = chunk["timestamp"].dt.date == target_date
                day_chunk = chunk[day_mask]
                if not day_chunk.empty:
                    return day_chunk.reset_index(drop=True)
            return pd.DataFrame(columns=["timestamp", "price", "quantity", "is_buyer_maker"])

        return None

    def _output_path(self, target_date: date, interval: str, output_format: str, symbol: str) -> Path:
        """Build the output Parquet path."""
        date_str = target_date.strftime("%Y%m%d")
        if output_format == "backtester":
            self.backtester_dir.mkdir(parents=True, exist_ok=True)
            return self.backtester_dir / f"{symbol}_1secs_{date_str}.parquet"
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            interval_tag = interval.replace("min", "m")
            return self.output_dir / f"{symbol}_{interval_tag}_{date_str}.parquet"

    def _write_parquet(self, candles: pd.DataFrame, path: Path, output_format: str):
        """Write candles to Parquet in the expected schema.

        Backtester format: columns [date(str UTC), open, high, low, close, volume] as float64.
        Analysis format: DatetimeIndex preserved, columns [open, high, low, close, volume].
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "backtester":
            out = candles.reset_index()
            # Rename the index column to 'date', matching NQ parquet schema
            idx_col = out.columns[0]
            out = out.rename(columns={idx_col: "date"})
            out["date"] = out["date"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
            for col in ["open", "high", "low", "close", "volume"]:
                out[col] = out[col].astype("float64")
            out.to_parquet(path, index=False)
        else:
            candles.to_parquet(path)
