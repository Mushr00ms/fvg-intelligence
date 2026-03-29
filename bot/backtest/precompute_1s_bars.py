#!/usr/bin/env python3
"""
precompute_1s_bars.py — Build 1-second OHLCV bars from Databento tick files.

Produces parquet files identical to IB-fetched 1s bars:
    Columns: date (str, UTC), open, high, low, close, volume (all float64)
    RTH only: 09:30-16:00 ET

Three integrity layers:
    1. Per-bar:  OHLC consistency, NQ tick grid (0.25), volume > 0
    2. Per-day:  no duplicate timestamps, reasonable bar count, RTH boundaries
    3. Cross-validation:  --validate compares output against existing IB bars
                          for overlapping dates (catches contract/tz/agg bugs)

Usage:
    # Process all tick files in extract dir:
    python3 bot/backtest/precompute_1s_bars.py --extract-dir /tmp/databento_trades_nq

    # Skip already-processed days:
    python3 bot/backtest/precompute_1s_bars.py --extract-dir /tmp/databento_trades_nq --skip-existing

    # Cross-validate against IB bars (run after processing):
    python3 bot/backtest/precompute_1s_bars.py --extract-dir /tmp/databento_trades_nq --validate
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from logic.utils.contract_utils import generate_nq_expirations, get_contract_for_date
from bot.backtest.enrich_volume import expiry_to_symbol

NQ_TICK = 0.25
RTH_START_MINUTES = 570   # 09:30 ET
RTH_END_MINUTES = 960     # 16:00 ET

DEFAULT_OUT_DIR = os.path.join(_ROOT, "bot", "data")


def load_day_ticks(date_compact: str, extract_dir: str):
    """Load a single day's .trades.dbn.zst into a DataFrame."""
    import databento as db

    path = os.path.join(extract_dir, f"glbx-mdp3-{date_compact}.trades.dbn.zst")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    try:
        store = db.DBNStore.from_file(path)
        df = store.to_df()
    except (ValueError, Exception):
        return None

    return df if not df.empty else None


def build_1s_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """Aggregate ticks into 1-second OHLCV bars for the given NQ symbol.

    Filters to RTH (09:30-16:00 ET), groups by 1-second floor.
    Returns DataFrame matching IB 1s bar format or None.
    """
    df_sym = df[df["symbol"] == symbol]
    if df_sym.empty:
        return None

    # Convert to ET for RTH filtering
    ts_et = df_sym["ts_event"].dt.tz_convert("America/New_York")
    minutes = ts_et.dt.hour * 60 + ts_et.dt.minute
    rth_mask = (minutes >= RTH_START_MINUTES) & (minutes < RTH_END_MINUTES)
    df_rth = df_sym.loc[rth_mask].copy()

    if df_rth.empty:
        return None

    # Floor to 1-second in UTC (matches IB bar format)
    df_rth["bar_time"] = df_rth["ts_event"].dt.floor("1s")

    # Aggregate OHLCV per second
    grouped = df_rth.groupby("bar_time")
    prices = grouped["price"]
    sizes = grouped["size"]

    bars = pd.DataFrame({
        "date": [str(t) for t in prices.first().index],
        "open": prices.first().values,
        "high": prices.max().values,
        "low": prices.min().values,
        "close": prices.last().values,
        "volume": sizes.sum().values.astype(np.float64),
    })

    return bars


# ── Integrity checks ─────────────────────────────────────────────────────

def check_bar_integrity(bars: pd.DataFrame, date_str: str) -> list:
    """Layer 1: Per-bar OHLC consistency and tick grid checks.

    Returns list of error strings (empty = all good).
    """
    errors = []

    # OHLC consistency
    bad_high = bars[(bars["high"] < bars["open"]) | (bars["high"] < bars["close"])]
    if len(bad_high):
        errors.append(f"{date_str}: {len(bad_high)} bars with high < open or close")

    bad_low = bars[(bars["low"] > bars["open"]) | (bars["low"] > bars["close"])]
    if len(bad_low):
        errors.append(f"{date_str}: {len(bad_low)} bars with low > open or close")

    # NQ tick grid: all prices must be multiples of 0.25
    for col in ["open", "high", "low", "close"]:
        remainders = (bars[col] / NQ_TICK) % 1
        off_grid = bars[remainders > 1e-9]
        if len(off_grid):
            errors.append(f"{date_str}: {len(off_grid)} bars with {col} off 0.25 tick grid")

    # Volume must be positive (every bar has at least 1 trade)
    zero_vol = bars[bars["volume"] <= 0]
    if len(zero_vol):
        errors.append(f"{date_str}: {len(zero_vol)} bars with zero volume")

    return errors


def check_day_integrity(bars: pd.DataFrame, date_str: str) -> list:
    """Layer 2: Per-day structural checks.

    Returns list of error strings.
    """
    errors = []

    # Reasonable bar count (RTH = 23,400 seconds, expect 15k-23.4k bars)
    if len(bars) < 10000:
        errors.append(f"{date_str}: suspiciously low bar count ({len(bars)}) — "
                      f"expected 15000-23400 for full RTH")
    if len(bars) > 23400:
        errors.append(f"{date_str}: too many bars ({len(bars)}) — max 23400 for RTH")

    # No duplicate timestamps
    dupes = bars["date"].duplicated().sum()
    if dupes > 0:
        errors.append(f"{date_str}: {dupes} duplicate timestamps")

    # Verify RTH boundaries (first bar ≥ 09:30, last bar < 16:00 in ET)
    first_ts = pd.Timestamp(bars["date"].iloc[0])
    last_ts = pd.Timestamp(bars["date"].iloc[-1])
    first_et = first_ts.tz_convert("America/New_York")
    last_et = last_ts.tz_convert("America/New_York")

    first_min = first_et.hour * 60 + first_et.minute
    last_min = last_et.hour * 60 + last_et.minute

    if first_min < RTH_START_MINUTES:
        errors.append(f"{date_str}: first bar {first_et} is before RTH (09:30 ET)")
    if last_min >= RTH_END_MINUTES:
        errors.append(f"{date_str}: last bar {last_et} is at or after RTH end (16:00 ET)")

    # Price sanity: NQ should be > 1000 (has been >5000 since ~2017)
    for col in ["open", "high", "low", "close"]:
        if bars[col].min() < 1000:
            errors.append(f"{date_str}: {col} min={bars[col].min():.2f} — too low for NQ")
        if bars[col].max() > 50000:
            errors.append(f"{date_str}: {col} max={bars[col].max():.2f} — too high for NQ")

    return errors


def cross_validate(out_dir: str, ib_dir: str, max_days: int = 50):
    """Layer 3: Compare Databento-derived bars against IB bars for overlapping dates.

    Checks: bar count, OHLC correlation, price difference statistics.
    """
    import glob

    # Find overlapping dates
    db_files = {os.path.basename(f).replace("nq_1secs_", "").replace(".parquet", ""): f
                for f in glob.glob(os.path.join(out_dir, "nq_1secs_*.parquet"))}
    ib_files = {os.path.basename(f).replace("nq_1secs_", "").replace(".parquet", ""): f
                for f in glob.glob(os.path.join(ib_dir, "nq_1secs_*.parquet"))}

    # Only compare files from the Databento output dir vs IB dir
    # (if they're the same dir, find dates that have BOTH sources)
    overlap = sorted(set(db_files.keys()) & set(ib_files.keys()))

    # Filter to only dates where the files are different (different sources)
    if out_dir == ib_dir:
        print("  Cross-validation skipped: output dir is same as IB dir")
        print("  Use --out-dir to write to a separate directory for validation")
        return

    if not overlap:
        print("  No overlapping dates for cross-validation")
        return

    overlap = overlap[:max_days]
    print(f"\n  Cross-validating {len(overlap)} overlapping dates...")

    close_diffs = []
    bar_count_diffs = []
    failures = []

    for date_str in overlap:
        db_df = pd.read_parquet(db_files[date_str])
        ib_df = pd.read_parquet(ib_files[date_str])

        # Bar count comparison
        count_diff = abs(len(db_df) - len(ib_df))
        bar_count_diffs.append(count_diff)

        # Merge on timestamp to compare matching bars
        db_df["date_ts"] = pd.to_datetime(db_df["date"], utc=True)
        ib_df["date_ts"] = pd.to_datetime(ib_df["date"], utc=True)
        merged = db_df.merge(ib_df, on="date_ts", suffixes=("_db", "_ib"))

        if len(merged) == 0:
            failures.append(f"{date_str}: zero matching timestamps")
            continue

        # Close price difference
        diff = (merged["close_db"] - merged["close_ib"]).abs()
        close_diffs.extend(diff.tolist())

        max_diff = diff.max()
        if max_diff > 1.0:  # More than 1 point difference
            failures.append(f"{date_str}: max close diff = {max_diff:.2f} pts")

        # OHLC correlation
        for col in ["open", "high", "low", "close"]:
            corr = merged[f"{col}_db"].corr(merged[f"{col}_ib"])
            if corr < 0.999:
                failures.append(f"{date_str}: {col} correlation = {corr:.6f} (expected > 0.999)")

    # Summary
    close_arr = np.array(close_diffs) if close_diffs else np.array([0])
    bar_arr = np.array(bar_count_diffs)
    print(f"\n  Results across {len(overlap)} days:")
    print(f"    Close price diff:  mean={close_arr.mean():.4f}  "
          f"median={np.median(close_arr):.4f}  max={close_arr.max():.4f}  "
          f"p99={np.percentile(close_arr, 99):.4f} pts")
    print(f"    Bar count diff:    mean={bar_arr.mean():.1f}  max={bar_arr.max()}")
    print(f"    Match rate:        {len(overlap) - len(failures)}/{len(overlap)} "
          f"({(len(overlap) - len(failures)) / len(overlap) * 100:.0f}%)")

    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in failures[:20]:
            print(f"    {f}")
    else:
        print("    All days PASSED cross-validation")


def main():
    parser = argparse.ArgumentParser(
        description="Build 1-second OHLCV bars from Databento tick files"
    )
    parser.add_argument("--extract-dir", required=True,
                        help="Directory with extracted .trades.dbn.zst files")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Output directory for 1s bar parquets (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip days that already have output parquets")
    parser.add_argument("--validate", action="store_true",
                        help="Cross-validate output against IB bars (requires --ib-dir)")
    parser.add_argument("--ib-dir", default=os.path.join(_ROOT, "bot", "data"),
                        help="Directory with IB 1s bar parquets (for cross-validation)")
    args = parser.parse_args()

    if not os.path.isdir(args.extract_dir):
        print(f"ERROR: Extract dir {args.extract_dir} does not exist.")
        sys.exit(1)

    # If only validating, skip processing
    if args.validate:
        cross_validate(args.out_dir, args.ib_dir)
        return

    # ── Discover tick files ──────────────────────────────────────────
    tick_files = sorted([
        f for f in os.listdir(args.extract_dir)
        if f.endswith(".trades.dbn.zst")
    ])
    if not tick_files:
        print(f"No tick files found in {args.extract_dir}")
        sys.exit(1)

    dates = []
    for f in tick_files:
        parts = f.replace(".trades.dbn.zst", "").split("-")
        if len(parts) >= 3:
            dates.append(parts[2])

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Extract dir: {args.extract_dir}")
    print(f"Output dir:  {args.out_dir}")
    print(f"Tick files:  {len(dates)} ({dates[0]} → {dates[-1]})")

    # ── NQ expiration calendar ───────────────────────────────────────
    from datetime import datetime
    start_year = int(dates[0][:4])
    end_year = int(dates[-1][:4]) + 1
    exp_dates = generate_nq_expirations(start_year, end_year)

    # ── Process each day ─────────────────────────────────────────────
    t0 = time.time()
    processed = 0
    skipped = 0
    no_data = 0
    integrity_errors = []

    for i, date_compact in enumerate(dates):
        date_str = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
        out_file = os.path.join(args.out_dir, f"nq_1secs_{date_compact}.parquet")

        if args.skip_existing and os.path.exists(out_file):
            skipped += 1
            continue

        # Resolve front-month contract
        trade_dt = datetime.strptime(date_compact, "%Y%m%d")
        exp = get_contract_for_date(trade_dt, exp_dates, roll_days=8)
        symbol = expiry_to_symbol(exp)

        # Load ticks
        df = load_day_ticks(date_compact, args.extract_dir)
        if df is None:
            no_data += 1
            continue

        # Build bars from the NQ outright with the most RTH volume.
        # On roll days the contract resolver picks the new front-month,
        # but most volume is still on the old contract.
        ts_et = df["ts_event"].dt.tz_convert("America/New_York")
        minutes_col = ts_et.dt.hour * 60 + ts_et.dt.minute
        rth_mask = (minutes_col >= RTH_START_MINUTES) & (minutes_col < RTH_END_MINUTES)
        df_rth_all = df.loc[rth_mask]

        day_symbols = [
            s for s in df_rth_all["symbol"].unique()
            if len(s) >= 3 and len(s) <= 4 and s.startswith("NQ") and "-" not in s
        ]
        # Rank by RTH volume, pick highest
        sym_vol = {s: int(df_rth_all.loc[df_rth_all["symbol"] == s, "size"].sum())
                   for s in day_symbols}
        ranked = sorted(sym_vol.items(), key=lambda x: x[1], reverse=True)

        bars = None
        used_sym = symbol
        for sym_candidate, vol in ranked:
            bars = build_1s_bars(df, sym_candidate)
            if bars is not None and not bars.empty:
                used_sym = sym_candidate
                break

        if bars is None or bars.empty:
            no_data += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(dates)}] {date_str} — no data")
            continue

        # ── Integrity checks ─────────────────────────────────────────
        errs = check_bar_integrity(bars, date_str)
        errs += check_day_integrity(bars, date_str)

        if errs:
            integrity_errors.extend(errs)
            for e in errs:
                print(f"  INTEGRITY: {e}")
            # Still save — but flag it. Fatal errors (OHLC broken) are
            # rare and worth investigating manually.

        bars.to_parquet(out_file, index=False)
        processed += 1

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (processed + no_data + skipped) / elapsed if elapsed > 0 else 0
            suffix = f" ({used_sym})" if used_sym != symbol else ""
            print(f"  [{i+1}/{len(dates)}] {date_str}{suffix} — "
                  f"{len(bars)} bars, {rate:.1f} days/s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s — "
          f"processed: {processed}, skipped: {skipped}, no_data: {no_data}")

    if integrity_errors:
        print(f"\nIntegrity issues ({len(integrity_errors)}):")
        for e in integrity_errors[:30]:
            print(f"  {e}")
        if len(integrity_errors) > 30:
            print(f"  ... and {len(integrity_errors) - 30} more")
    else:
        print("\nAll bars passed integrity checks")

    # ── Auto cross-validate if IB bars exist in the same dir ─────────
    import glob
    ib_count = len(glob.glob(os.path.join(args.ib_dir, "nq_1secs_*.parquet")))
    if ib_count > 0 and args.out_dir != args.ib_dir:
        print(f"\nAuto cross-validation against {ib_count} IB bar files...")
        cross_validate(args.out_dir, args.ib_dir)


if __name__ == "__main__":
    main()
