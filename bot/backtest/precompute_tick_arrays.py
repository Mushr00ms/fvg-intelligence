#!/usr/bin/env python3
"""
precompute_tick_arrays.py — Build per-day tick parquets from Databento .dbn.zst files.

Produces compact parquet files for tick-precise fill simulation and enrichment:
    Columns: ts_ns (int64), price (float64), size (int32), side (int8), symbol (str)
    RTH only: 09:30-16:00 ET
    All NQ outrights included (for roll-day fallback)

Usage:
    # Process all tick files (auto-detects zip, 4 workers):
    python3 bot/backtest/precompute_tick_arrays.py

    # Custom workers and skip existing:
    python3 bot/backtest/precompute_tick_arrays.py --workers 8 --skip-existing

    # Filter to a year:
    python3 bot/backtest/precompute_tick_arrays.py --year 2022 --workers 6
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from bot.backtest.enrich_volume import find_trades_zip, find_all_trades_zips, extract_zip

# Persistent extract dir on Linux FS (not /tmp/)
DEFAULT_EXTRACT_DIR = os.path.expanduser("~/databento_trades_nq")
DEFAULT_OUT_DIR = os.path.join(_ROOT, "bot", "data", "ticks")

RTH_START_MINUTES = 570   # 09:30 ET
RTH_END_MINUTES = 960     # 16:00 ET
NQ_TICK = 0.25


def load_day_ticks_raw(date_compact: str, extract_dir: str):
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


def build_tick_array(df: pd.DataFrame) -> pd.DataFrame | None:
    """Convert raw Databento tick DataFrame to compact tick array format.

    Filters to RTH (09:30-16:00 ET), NQ outrights only.
    Returns DataFrame with columns: ts_ns, price, size, side, symbol.
    """
    # Filter to NQ outrights only (4-char symbols like NQH2, NQM2, no spreads)
    nq_mask = (
        df["symbol"].str.startswith("NQ")
        & (df["symbol"].str.len() <= 4)
        & ~df["symbol"].str.contains("-", na=False)
    )
    df_nq = df.loc[nq_mask]
    if df_nq.empty:
        return None

    # Filter to RTH
    ts_et = df_nq["ts_event"].dt.tz_convert("America/New_York")
    minutes = ts_et.dt.hour * 60 + ts_et.dt.minute
    rth_mask = (minutes >= RTH_START_MINUTES) & (minutes < RTH_END_MINUTES)
    df_rth = df_nq.loc[rth_mask]

    if df_rth.empty:
        return None

    # Build compact array — normalize to ns for consistent int64 across pandas versions
    ts_ns = df_rth["ts_event"].to_numpy("datetime64[ns]").view("int64")
    prices = df_rth["price"].to_numpy(dtype=np.float64)
    sizes = df_rth["size"].to_numpy(dtype=np.int32)

    # Encode side: 'A' (buyer aggressor) = 1, 'B' (seller aggressor) = -1
    side_raw = df_rth["side"].values
    side_encoded = np.where(
        side_raw == "A", np.int8(1),
        np.where(side_raw == "B", np.int8(-1), np.int8(0))
    )

    symbols = df_rth["symbol"].values

    result = pd.DataFrame({
        "ts_ns": ts_ns,
        "price": prices,
        "size": sizes,
        "side": side_encoded,
        "symbol": symbols,
    })

    # Sort by timestamp (should already be sorted, but enforce)
    result.sort_values("ts_ns", inplace=True, ignore_index=True)

    return result


def check_integrity(df: pd.DataFrame, date_str: str) -> list:
    """Integrity checks for tick array."""
    errors = []

    # Tick count sanity
    n = len(df)
    if n < 50000:
        errors.append(f"{date_str}: low tick count ({n}) — expected 100k-600k for NQ RTH")
    if n > 1000000:
        errors.append(f"{date_str}: high tick count ({n}) — expected 100k-600k for NQ RTH")

    # Price on 0.25 grid
    remainders = (df["price"] / NQ_TICK) % 1
    off_grid = (remainders > 1e-9).sum()
    if off_grid > 0:
        errors.append(f"{date_str}: {off_grid} ticks off 0.25 tick grid")

    # Timestamp monotonic
    if not np.all(np.diff(df["ts_ns"].to_numpy()) >= 0):
        errors.append(f"{date_str}: timestamps not monotonically non-decreasing")

    # Size positive
    bad_size = (df["size"] <= 0).sum()
    if bad_size > 0:
        errors.append(f"{date_str}: {bad_size} ticks with non-positive size")

    return errors


# ── Single-day worker (runs in subprocess) ───────────────────────────────────

def _process_one_day(date_compact, extract_dir, out_dir, skip_existing):
    """Process a single day: load .dbn.zst → build tick array → save parquet.

    Returns (date_compact, status, n_ticks, symbols, errors).
    status: 'ok' | 'skipped' | 'no_data'
    """
    date_str = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
    out_file = os.path.join(out_dir, f"nq_ticks_{date_compact}.parquet")

    if skip_existing and os.path.exists(out_file):
        return (date_compact, "skipped", 0, [], [])

    df = load_day_ticks_raw(date_compact, extract_dir)
    if df is None:
        return (date_compact, "no_data", 0, [], [])

    tick_df = build_tick_array(df)
    if tick_df is None or tick_df.empty:
        return (date_compact, "no_data", 0, [], [])

    errs = check_integrity(tick_df, date_str)
    tick_df.to_parquet(out_file, index=False)

    symbols = list(tick_df["symbol"].unique())
    return (date_compact, "ok", len(tick_df), symbols, errs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build per-day tick parquets from Databento .dbn.zst files"
    )
    parser.add_argument("--extract-dir", default=DEFAULT_EXTRACT_DIR,
                        help=f"Directory with extracted .dbn.zst files (default: {DEFAULT_EXTRACT_DIR})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Output directory for tick parquets (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip days that already have output parquets")
    parser.add_argument("--zip", default=None,
                        help="Path to Databento trades zip (auto-detected if omitted)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip zip extraction if files are already extracted")
    parser.add_argument("--workers", type=int, default=min(4, cpu_count()),
                        help=f"Number of parallel workers (default: {min(4, cpu_count())})")
    parser.add_argument("--year", type=int, default=None,
                        help="Filter to a specific year (e.g. --year 2022)")
    args = parser.parse_args()

    # ── Handle zip extraction (all GLBX zips) ──────────────────────────
    if not args.skip_extract:
        if args.zip:
            zips = [args.zip]
        else:
            zips = find_all_trades_zips(_ROOT)
        if zips:
            print(f"Found {len(zips)} trades zip(s)")
            print(f"Extracting to {args.extract_dir} ...")
            for zp in zips:
                print(f"  {os.path.basename(zp)}")
                extract_zip(zp, args.extract_dir, verbose=False)
            n = len([f for f in os.listdir(args.extract_dir) if f.endswith(".dbn.zst")])
            print(f"  {n} tick files ready")
        elif not os.path.isdir(args.extract_dir):
            print(f"ERROR: No zip found and extract dir {args.extract_dir} does not exist.")
            sys.exit(1)

    if not os.path.isdir(args.extract_dir):
        print(f"ERROR: Extract dir {args.extract_dir} does not exist.")
        sys.exit(1)

    # ── Discover tick files ───────────────────────────────────────────
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

    # Filter to year if specified
    if args.year:
        year_str = str(args.year)
        dates = [d for d in dates if d.startswith(year_str)]
        if not dates:
            print(f"No tick files found for year {args.year}")
            sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Extract dir: {args.extract_dir}")
    print(f"Output dir:  {args.out_dir}")
    print(f"Tick files:  {len(dates)} ({dates[0]} → {dates[-1]})")
    print(f"Workers:     {args.workers}")

    # ── Process days in parallel ──────────────────────────────────────
    t0 = time.time()
    processed = 0
    skipped = 0
    no_data = 0
    integrity_errors = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_one_day, dc, args.extract_dir, args.out_dir, args.skip_existing
            ): dc
            for dc in dates
        }

        for future in as_completed(futures):
            done += 1
            date_compact, status, n_ticks, symbols, errs = future.result()

            if status == "ok":
                processed += 1
                if errs:
                    integrity_errors.extend(errs)
                    for e in errs:
                        print(f"  INTEGRITY: {e}")
            elif status == "skipped":
                skipped += 1
            else:
                no_data += 1

            if done % 50 == 0 or done == 1:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                date_str = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
                sym_str = ", ".join(symbols) if symbols else "—"
                print(f"  [{done}/{len(dates)}] {date_str} — "
                      f"{n_ticks:,} ticks, {sym_str}, {rate:.1f} days/s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s — "
          f"processed: {processed}, skipped: {skipped}, no_data: {no_data}")
    if processed > 0:
        print(f"Throughput: {processed / elapsed:.1f} days/s ({args.workers} workers)")

    if integrity_errors:
        print(f"\nIntegrity issues ({len(integrity_errors)}):")
        for e in integrity_errors[:30]:
            print(f"  {e}")
    else:
        print("\nAll tick arrays passed integrity checks")


if __name__ == "__main__":
    main()
