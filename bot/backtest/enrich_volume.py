#!/usr/bin/env python3
"""
enrich_volume.py — Enrich backtest trade results with Databento tick volume at entry price.

For each trade: sums all tick volume that traded at exactly trade.entry_price between
formation_time and exit_time, using the correct front-month NQ contract per the
project's roll logic (quarterly expiry, 8-day roll).

This lets you answer: "Across all touches before TP, how many contracts traded at my
exact limit level? Is it consistently >30 — making confirmed_limit unnecessary?"

Usage:
    # Auto-detect zip, enrich 2025 results:
    python3 bot/backtest/enrich_volume.py --results bot/backtest/results/2025.json

    # Explicit zip path, skip re-extraction if already extracted:
    python3 bot/backtest/enrich_volume.py \\
        --results bot/backtest/results/2025.json \\
        --zip /path/to/GLBX-trades.zip \\
        --skip-extract

    # Force re-enrich even if volume_at_entry already present:
    python3 bot/backtest/enrich_volume.py --results ... --force
"""

import argparse
import json
import os
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from logic.utils.contract_utils import generate_nq_expirations, get_contract_for_date

# NQ futures month letter codes (quarterly contracts only)
_MONTH_LETTERS = {3: "H", 6: "M", 9: "U", 12: "Z"}

# Extract to Linux FS for 10x faster I/O vs /mnt/c/
DEFAULT_EXTRACT_DIR = "/tmp/databento_trades_nq"


def expiry_to_symbol(exp_date) -> str:
    """Convert expiration date to Databento front-month symbol.

    e.g. datetime(2025, 3, 21) -> 'NQH5'
         datetime(2025, 6, 20) -> 'NQM5'
    """
    if hasattr(exp_date, "date") and callable(exp_date.date):
        d = exp_date.date()
    else:
        d = exp_date
    letter = _MONTH_LETTERS[d.month]
    year_digit = d.year % 10
    return f"NQ{letter}{year_digit}"


def find_trades_zip(root: str) -> str | None:
    """Auto-detect the Databento trades zip in the project root."""
    for fname in os.listdir(root):
        if not (fname.startswith("GLBX") and fname.endswith(".zip")):
            continue
        path = os.path.join(root, fname)
        try:
            with zipfile.ZipFile(path) as zf:
                names = [i.filename for i in zf.infolist()]
                if any(".trades.dbn.zst" in n for n in names):
                    return path
        except Exception:
            continue
    return None


def extract_zip(zip_path: str, out_dir: str, verbose: bool = True):
    """Extract all .trades.dbn.zst files from zip to out_dir.

    Skips files that are already present (idempotent re-runs).
    """
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        entries = [i for i in zf.infolist() if ".trades.dbn.zst" in i.filename]
        for info in entries:
            dest = os.path.join(out_dir, os.path.basename(info.filename))
            if os.path.exists(dest):
                continue  # Already extracted
            with zf.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            if verbose:
                print(f"  extracted {os.path.basename(info.filename)}")
    if verbose:
        n = len([f for f in os.listdir(out_dir) if f.endswith(".dbn.zst")])
        print(f"  {n} tick files ready in {out_dir}")


def parse_utc(ts_str: str) -> datetime:
    """Parse a timestamp string to UTC-aware datetime.

    Handles:
      - "2025-01-02 12:25:00-05:00"  (ET-aware — formation_time format)
      - "2025-01-02 20:32:05"         (naive UTC — entry_time / exit_time format)
    """
    ts_str = ts_str.strip().replace(" ", "T", 1)
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_day_ticks(date_str: str, extract_dir: str):
    """Load a single day's .trades.dbn.zst into a DataFrame.

    Returns (prices_array, sizes_array, ts_ns_array, symbol_series) or None if missing.
    """
    import databento as db

    compact = date_str.replace("-", "")
    path = os.path.join(extract_dir, f"glbx-mdp3-{compact}.trades.dbn.zst")
    if not os.path.exists(path):
        return None

    store = db.DBNStore.from_file(path)
    df = store.to_df()

    if df.empty:
        return None

    return df


def compute_volume_at_price(df: pd.DataFrame, symbol: str,
                             entry_price: float,
                             formation_utc: datetime,
                             exit_utc: datetime) -> int | None:
    """Sum all tick sizes at exact entry_price for the given symbol within time window.

    Uses int64 nanosecond timestamps throughout for vectorized numpy speed.

    Returns:
        Total contracts traded at entry_price in [formation_time, exit_time].
        None if the symbol had no ticks on this day (data issue).
    """
    df_sym = df[df["symbol"] == symbol]
    if df_sym.empty:
        return None

    prices = df_sym["price"].to_numpy(dtype=np.float64)
    sizes = df_sym["size"].to_numpy(dtype=np.int64)
    # ts_event is datetime64[ns, UTC] — astype int64 gives nanoseconds since epoch
    ts_ns = df_sym["ts_event"].astype("int64").to_numpy()

    t_start_ns = pd.Timestamp(formation_utc).value
    t_end_ns = pd.Timestamp(exit_utc).value

    # NQ prices are on 0.25-point grid → exact float64 comparison is safe,
    # but use a tiny epsilon (1e-6 pts) to guard against any IEEE754 edge case
    mask = (
        (np.abs(prices - entry_price) < 1e-6)
        & (ts_ns >= t_start_ns)
        & (ts_ns <= t_end_ns)
    )

    return int(sizes[mask].sum())


def compute_volume_first_touch(df: pd.DataFrame, symbol: str,
                                entry_price: float,
                                entry_utc: datetime) -> int | None:
    """Sum tick volume at exact entry_price during the 1-second bar of first touch.

    The backtester's entry_time is the timestamp of the 1s bar where price first
    reaches the entry level. We look at all ticks at exact entry_price within
    [entry_time, entry_time + 1 second) — this is the liquidity available on
    first contact.

    Returns:
        Contracts traded at entry_price during that 1-second window.
        None if the symbol had no ticks on this day.
    """
    df_sym = df[df["symbol"] == symbol]
    if df_sym.empty:
        return None

    prices = df_sym["price"].to_numpy(dtype=np.float64)
    sizes = df_sym["size"].to_numpy(dtype=np.int64)
    ts_ns = df_sym["ts_event"].astype("int64").to_numpy()

    t_start_ns = pd.Timestamp(entry_utc).value
    t_end_ns = t_start_ns + 1_000_000_000  # +1 second in nanoseconds

    mask = (
        (np.abs(prices - entry_price) < 1e-6)
        & (ts_ns >= t_start_ns)
        & (ts_ns < t_end_ns)
    )

    return int(sizes[mask].sum())


def print_summary(trades: list):
    """Print distribution of volume metrics across enriched trades."""
    enriched = [t for t in trades if t.get("volume_at_entry") is not None]
    if not enriched:
        print("No trades enriched.")
        return

    def stats(vlist, label):
        if not vlist:
            return
        a = np.array(vlist)
        pct_30 = (a >= 30).sum() / len(a) * 100
        print(f"  {label:10s}  n={len(a):4d}  "
              f"median={np.median(a):5.0f}  "
              f"mean={np.mean(a):5.0f}  "
              f"p10={np.percentile(a,10):4.0f}  "
              f"p90={np.percentile(a,90):5.0f}  "
              f"≥30: {pct_30:.0f}%")

    def breakdown(a, label):
        print(f"\n  {label} — threshold breakdown ({len(a)} trades):")
        buckets = [(0, 0, "=0 (no ticks)"), (1, 9, "1-9"), (10, 29, "10-29"),
                   (30, 99, "30-99"), (100, None, "≥100")]
        for lo, hi, lbl in buckets:
            if hi is None:
                n = (a >= lo).sum()
            elif lo == 0:
                n = (a == 0).sum()
            else:
                n = ((a >= lo) & (a <= hi)).sum()
            bar = "█" * int(n / max(len(a), 1) * 40)
            print(f"    {lbl:15s}: {n:4d} ({n/max(len(a),1)*100:4.0f}%)  {bar}")

    # ── Total volume (formation → exit) ──
    vols = [t["volume_at_entry"] for t in enriched]
    tp_vols = [t["volume_at_entry"] for t in enriched if t.get("exit_reason") == "TP"]
    sl_vols = [t["volume_at_entry"] for t in enriched if t.get("exit_reason") == "SL"]

    print("\n── Volume at Entry (formation → exit) ─────────────────────────────")
    stats(vols,    "ALL")
    stats(tp_vols, "TP wins")
    stats(sl_vols, "SL loss")
    breakdown(np.array(vols), "Total volume")

    # ── First-touch volume (1-second bar) ──
    ft_enriched = [t for t in trades if t.get("volume_first_touch") is not None]
    if ft_enriched:
        ft_vols = [t["volume_first_touch"] for t in ft_enriched]
        ft_tp = [t["volume_first_touch"] for t in ft_enriched if t.get("exit_reason") == "TP"]
        ft_sl = [t["volume_first_touch"] for t in ft_enriched if t.get("exit_reason") == "SL"]

        print("\n── First Touch Volume (1s bar at entry) ───────────────────────────")
        stats(ft_vols, "ALL")
        stats(ft_tp,   "TP wins")
        stats(ft_sl,   "SL loss")
        breakdown(np.array(ft_vols), "First touch")

    print("───────────────────────────────────────────────────────────────────")

    # ── Adjusted P&L (fill probability discount) ──
    adj_trades = [t for t in trades if t.get("adjusted_pnl_dollars") is not None]
    if adj_trades:
        raw_pnl = sum(t["pnl_dollars"] for t in adj_trades)
        adj_pnl = sum(t["adjusted_pnl_dollars"] for t in adj_trades)
        hi_conf = [t for t in adj_trades if t.get("fill_probability", 1) >= 1.0]
        lo_conf = [t for t in adj_trades if t.get("fill_probability", 1) < 1.0]
        lo_raw = sum(t["pnl_dollars"] for t in lo_conf)
        lo_adj = sum(t["adjusted_pnl_dollars"] for t in lo_conf)
        print(f"\n── Fill Probability Adjusted P&L ──────────────────────────────────")
        print(f"  Raw P&L (touch=fill):     ${raw_pnl:>10,.0f}  ({len(adj_trades)} trades)")
        print(f"  Adjusted P&L:             ${adj_pnl:>10,.0f}  (discount on {len(lo_conf)} low-vol trades)")
        print(f"  Discount amount:          ${raw_pnl - adj_pnl:>10,.0f}  ({abs(raw_pnl - adj_pnl) / abs(raw_pnl) * 100:.1f}% of raw)")
        print(f"  High-confidence ({len(hi_conf)}):     ${sum(t['pnl_dollars'] for t in hi_conf):>10,.0f}")
        print(f"  Low-vol raw ({len(lo_conf)}):          ${lo_raw:>10,.0f}")
        print(f"  Low-vol adjusted ({len(lo_conf)}):     ${lo_adj:>10,.0f}")
        print(f"───────────────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich backtest results with Databento tick volume at entry price"
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to backtest results JSON (e.g. bot/backtest/results/2025.json)"
    )
    parser.add_argument(
        "--zip", default=None,
        help="Path to Databento trades zip (auto-detected from project root if omitted)"
    )
    parser.add_argument(
        "--extract-dir", default=DEFAULT_EXTRACT_DIR,
        help=f"Directory to extract .dbn.zst files (default: {DEFAULT_EXTRACT_DIR})"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip zip extraction if files are already extracted"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-enrich even if volume_at_entry is already present"
    )
    args = parser.parse_args()

    # ── Locate zip ────────────────────────────────────────────────────────────
    zip_path = args.zip
    if zip_path is None:
        zip_path = find_trades_zip(_ROOT)
        if zip_path is None:
            print("ERROR: No Databento trades zip found in project root. Use --zip to specify.")
            sys.exit(1)
    print(f"Trades zip : {zip_path}")

    # ── Extract ───────────────────────────────────────────────────────────────
    if not args.skip_extract:
        print(f"Extracting to {args.extract_dir} ...")
        extract_zip(zip_path, args.extract_dir)
    else:
        n = len([f for f in os.listdir(args.extract_dir) if f.endswith(".dbn.zst")])
        print(f"Skipping extraction — {n} files in {args.extract_dir}")

    # ── Load results ──────────────────────────────────────────────────────────
    results_path = os.path.join(_ROOT, args.results) if not os.path.isabs(args.results) else args.results
    with open(results_path) as f:
        results = json.load(f)
    trades = results.get("trades", [])

    already_done = sum(1 for t in trades if t.get("volume_first_touch") is not None)
    if already_done and not args.force:
        print(f"\n{already_done}/{len(trades)} trades already enriched.")
        print("Use --force to re-enrich. Running summary on existing data.")
        print_summary(trades)
        return

    print(f"\nEnriching {len(trades)} trades from {results_path}")

    # ── Build NQ expiration calendar ──────────────────────────────────────────
    exp_dates = generate_nq_expirations(2024, 2026)

    # ── Group trades by date (load each day's ticks once) ────────────────────
    by_date: dict[str, list] = defaultdict(list)
    for t in trades:
        by_date[t["date"]].append(t)

    total_dates = len(by_date)
    enriched_count = 0
    missing_days = []

    for i, date_str in enumerate(sorted(by_date.keys())):
        day_trades = by_date[date_str]

        # Resolve front-month contract for this date
        trade_dt = datetime.strptime(date_str, "%Y-%m-%d")
        exp = get_contract_for_date(trade_dt, exp_dates, roll_days=8)
        symbol = expiry_to_symbol(exp)

        print(f"[{i+1:3d}/{total_dates}] {date_str}  {symbol}  ({len(day_trades)} trades)", end="", flush=True)

        # Load tick data for this day
        df = load_day_ticks(date_str, args.extract_dir)
        if df is None:
            print(f"  ✗ no tick file")
            for t in day_trades:
                t["volume_at_entry"] = None
                t["volume_first_touch"] = None
            missing_days.append(date_str)
            continue

        # Build list of NQ outright symbols available on this day (for roll fallback)
        # Outrights = symbols matching NQ[A-Z]\d, excluding spreads (contain '-')
        day_symbols = [
            s for s in df["symbol"].unique()
            if len(s) == 4 and s.startswith("NQ") and "-" not in s
        ]

        # Per-trade lookup: exact price + time window + front-month symbol
        # On roll days the primary symbol may have 0 volume — fall back to other outrights
        day_results = []
        for trade in day_trades:
            formation_utc = parse_utc(trade["formation_time"])
            entry_utc = parse_utc(trade["entry_time"])
            exit_utc = parse_utc(trade["exit_time"])

            vol_total = compute_volume_at_price(
                df, symbol,
                trade["entry_price"],
                formation_utc,
                exit_utc,
            )
            vol_ft = compute_volume_first_touch(
                df, symbol,
                trade["entry_price"],
                entry_utc,
            )

            # Roll fallback: if primary symbol gave 0/None, try other outrights
            used_sym = symbol
            if (vol_total is None or vol_total == 0) and len(day_symbols) > 1:
                for alt_sym in day_symbols:
                    if alt_sym == symbol:
                        continue
                    alt_vol = compute_volume_at_price(
                        df, alt_sym,
                        trade["entry_price"],
                        formation_utc,
                        exit_utc,
                    )
                    if alt_vol is not None and alt_vol > (vol_total or 0):
                        alt_ft = compute_volume_first_touch(
                            df, alt_sym,
                            trade["entry_price"],
                            entry_utc,
                        )
                        vol_total = alt_vol
                        vol_ft = alt_ft
                        used_sym = alt_sym

            trade["volume_at_entry"] = vol_total
            trade["volume_first_touch"] = vol_ft

            # Fill probability: ≥30 vol = 100%, <30 vol = vol/30 (linear discount)
            FILL_THRESHOLD = 30
            if vol_total is not None and vol_total >= FILL_THRESHOLD:
                fill_prob = 1.0
            elif vol_total is not None and vol_total > 0:
                fill_prob = round(vol_total / FILL_THRESHOLD, 4)
            else:
                fill_prob = 0.0

            trade["fill_probability"] = fill_prob
            trade["adjusted_pnl_dollars"] = round(trade["pnl_dollars"] * fill_prob, 2)

            suffix = f"→{used_sym}" if used_sym != symbol else ""
            day_results.append(f"{vol_total}({vol_ft}ft){suffix}")
            enriched_count += 1

        print(f"  [{', '.join(day_results)}]")

    # ── Save enriched JSON ────────────────────────────────────────────────────
    results["trades"] = trades
    with open(results_path, "w") as f:
        json.dump(results, f, separators=(",", ":"))

    print(f"\nSaved → {results_path}")
    if missing_days:
        print(f"Missing tick files for {len(missing_days)} days: {missing_days}")

    print_summary(trades)


if __name__ == "__main__":
    main()
