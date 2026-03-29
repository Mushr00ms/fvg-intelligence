#!/usr/bin/env python3
"""
precompute_imbalance.py — Build 5-min aggressor delta bars from Databento tick files.

Processes .trades.dbn.zst tick files into lightweight parquet files containing
5-min aggressor imbalance bars.  Uses the native CME side field (A=buyer, B=seller)
for ground-truth classification.

Output per day: nq_imbalance_5min_YYYYMMDD.parquet
    Columns: date (datetime ET), buy_vol, sell_vol, imbalance, total_vol

Time range: 08:30-16:00 ET (includes 1h pre-RTH for HFOIV rolling warm-up).
5-min grid aligned to backtester: 08:30, 08:35, ..., 15:55.

Usage:
    # Auto-detect zip, precompute all days:
    python3 bot/backtest/precompute_imbalance.py

    # Explicit paths:
    python3 bot/backtest/precompute_imbalance.py \\
        --extract-dir /tmp/databento_trades_nq \\
        --out-dir bot/data/imbalance

    # Only process missing days (idempotent):
    python3 bot/backtest/precompute_imbalance.py --skip-existing
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

# Reuse symbol resolution from enrich_volume
from bot.backtest.enrich_volume import (
    expiry_to_symbol,
    find_trades_zip,
    extract_zip,
    DEFAULT_EXTRACT_DIR,
)

# 08:30 ET = 510 minutes since midnight.  16:00 ET = 960 minutes.
_ETH_START_MINUTES = 510   # 08:30
_RTH_END_MINUTES = 960     # 16:00

DEFAULT_OUT_DIR = os.path.join(_ROOT, "bot", "data", "imbalance")


def load_day_ticks(date_str: str, extract_dir: str) -> pd.DataFrame | None:
    """Load a single day's .trades.dbn.zst into a DataFrame.

    Returns DataFrame with ts_event, price, size, side, symbol or None.
    """
    import databento as db

    compact = date_str.replace("-", "")
    path = os.path.join(extract_dir, f"glbx-mdp3-{compact}.trades.dbn.zst")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    try:
        store = db.DBNStore.from_file(path)
        df = store.to_df()
    except (ValueError, Exception):
        return None

    if df.empty:
        return None

    return df


def compute_imbalance_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """Aggregate tick data into 5-min aggressor imbalance bars.

    Filters to the given NQ outright symbol, 08:30-16:00 ET window.
    Groups by 5-min floor, sums buy_vol (side='A') and sell_vol (side='B').

    Returns DataFrame with columns: date, buy_vol, sell_vol, imbalance, total_vol
    or None if no data.
    """
    # Filter to target symbol
    df_sym = df[df["symbol"] == symbol]
    if df_sym.empty:
        return None

    # Convert to ET
    ts_et = df_sym["ts_event"].dt.tz_convert("America/New_York")
    minutes = ts_et.dt.hour * 60 + ts_et.dt.minute

    # Filter to 08:30-16:00 ET
    mask = (minutes >= _ETH_START_MINUTES) & (minutes < _RTH_END_MINUTES)
    df_window = df_sym.loc[mask].copy()
    if df_window.empty:
        return None

    # Floor timestamps to 5-min grid
    df_window["bar_time"] = ts_et.loc[mask].dt.floor("5min")

    # Aggregate per 5-min bar
    buy_mask = df_window["side"] == "A"
    sell_mask = df_window["side"] == "B"

    buy_vol = df_window.loc[buy_mask].groupby("bar_time")["size"].sum()
    sell_vol = df_window.loc[sell_mask].groupby("bar_time")["size"].sum()

    # Build result aligned to all bars — cast to int64 to avoid uint32 overflow
    all_bars = df_window.groupby("bar_time")["size"].sum()
    buy_vals = buy_vol.reindex(all_bars.index, fill_value=0).values.astype(np.int64)
    sell_vals = sell_vol.reindex(all_bars.index, fill_value=0).values.astype(np.int64)
    result = pd.DataFrame({
        "date": all_bars.index,
        "buy_vol": buy_vals,
        "sell_vol": sell_vals,
        "total_vol": all_bars.values.astype(np.int64),
    })
    result["imbalance"] = result["buy_vol"] - result["sell_vol"]

    return result.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute 5-min aggressor imbalance bars from Databento tick files"
    )
    parser.add_argument(
        "--zip", default=None,
        help="Path to Databento trades zip (auto-detected if omitted)"
    )
    parser.add_argument(
        "--extract-dir", default=DEFAULT_EXTRACT_DIR,
        help=f"Directory with extracted .dbn.zst files (default: {DEFAULT_EXTRACT_DIR})"
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"Output directory for imbalance parquets (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip zip extraction (files already extracted)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip days that already have output parquets"
    )
    args = parser.parse_args()

    # ── Extract tick files if needed ─────────────────────────────────
    if not args.skip_extract:
        zip_path = args.zip or find_trades_zip(_ROOT)
        if zip_path is None:
            print("ERROR: No Databento trades zip found. Use --zip or --skip-extract.")
            sys.exit(1)
        print(f"Trades zip: {zip_path}")
        print(f"Extracting to {args.extract_dir} ...")
        extract_zip(zip_path, args.extract_dir)
    else:
        if not os.path.isdir(args.extract_dir):
            print(f"ERROR: Extract dir {args.extract_dir} does not exist.")
            sys.exit(1)

    # ── Discover available tick files ────────────────────────────────
    tick_files = sorted([
        f for f in os.listdir(args.extract_dir)
        if f.endswith(".trades.dbn.zst")
    ])
    if not tick_files:
        print(f"No tick files found in {args.extract_dir}")
        sys.exit(1)

    # Extract dates from filenames: glbx-mdp3-YYYYMMDD.trades.dbn.zst
    dates = []
    for f in tick_files:
        parts = f.replace(".trades.dbn.zst", "").split("-")
        if len(parts) >= 3:
            date_str = parts[2]  # YYYYMMDD
            dates.append(date_str)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Output: {args.out_dir}")
    print(f"Tick files: {len(dates)} ({dates[0]} → {dates[-1]})")

    # ── Build NQ expiration calendar ─────────────────────────────────
    from datetime import datetime
    start_year = int(dates[0][:4])
    end_year = int(dates[-1][:4]) + 1
    exp_dates = generate_nq_expirations(start_year, end_year)

    # ── Process each day ─────────────────────────────────────────────
    t0 = time.time()
    processed = 0
    skipped = 0
    no_data = 0

    for i, date_compact in enumerate(dates):
        date_str = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
        out_file = os.path.join(args.out_dir, f"nq_imbalance_5min_{date_compact}.parquet")

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

        # Pick the NQ outright with the most volume in the 08:30-16:00 window.
        # On roll days the contract resolver picks the new front-month,
        # but most volume is still on the old contract.
        ts_et = df["ts_event"].dt.tz_convert("America/New_York")
        minutes_col = ts_et.dt.hour * 60 + ts_et.dt.minute
        window_mask = (minutes_col >= _ETH_START_MINUTES) & (minutes_col < _RTH_END_MINUTES)
        df_window = df.loc[window_mask]

        day_symbols = [
            s for s in df_window["symbol"].unique()
            if len(s) >= 3 and len(s) <= 4 and s.startswith("NQ") and "-" not in s
        ]
        sym_vol = {s: int(df_window.loc[df_window["symbol"] == s, "size"].sum())
                   for s in day_symbols}
        ranked = sorted(sym_vol.items(), key=lambda x: x[1], reverse=True)

        result = None
        used_sym = symbol
        for sym_candidate, vol in ranked:
            result = compute_imbalance_bars(df, sym_candidate)
            if result is not None and not result.empty:
                used_sym = sym_candidate
                break

        if result is None or result.empty:
            no_data += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(dates)}] {date_str} — no data")
            continue

        result.to_parquet(out_file, index=False)
        processed += 1

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (processed + no_data + skipped) / elapsed if elapsed > 0 else 0
            suffix = f" ({used_sym})" if used_sym != symbol else ""
            print(f"  [{i+1}/{len(dates)}] {date_str}{suffix} — "
                  f"{len(result)} bars, {rate:.1f} days/s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s — "
          f"processed: {processed}, skipped: {skipped}, no_data: {no_data}")


if __name__ == "__main__":
    main()
