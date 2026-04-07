#!/usr/bin/env python3
"""
sweep_btc_fvg.py — Parallel BTC FVG parameter sweep.

Loads resampled BTC candles once, then fans out configs across all CPU cores
using multiprocessing (fork — COW sharing of read-only DataFrames).

Sweep axes:
    Timeframe (TF):        5min, 15min, 1h, 4h
    Min FVG size (bps):    5, 10, 15, 20, 30, 50, 75, 100
    Time period (min):     60, 120, 240
    Mitigation window:     per-TF bar counts

Fixed expansion windows per TF:
    5min  → 48 bars  (4h)
    15min → 16 bars  (4h)
    1h    → 6 bars   (6h)
    4h    → 6 bars   (24h)

Outputs per-trade JSON for offline re-bucketing of risk_bps.

Usage:
    python3 scripts/sweep_btc_fvg.py
    python3 scripts/sweep_btc_fvg.py --tf 5min --workers 8
    python3 scripts/sweep_btc_fvg.py --tf 15min 1h --data-dir /home/cr0wn/binance_data/resampled
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

# ── Constants ────────────────────────────────────────────────────────────

TF_CONFIG = {
    "5min": {
        "detect_tag": "5m",
        "walk_tag": "1m",
        "expansion_bars": 48,       # 48 × 5min = 4h
        "walk_expansion_bars": 240, # 240 × 1min = 4h
        "mitigation_sweep": [15, 30, 45, 60, 90],
    },
    "15min": {
        "detect_tag": "15m",
        "walk_tag": "5m",
        "expansion_bars": 16,       # 16 × 15min = 4h
        "walk_expansion_bars": 48,  # 48 × 5min = 4h
        "mitigation_sweep": [10, 20, 30, 40, 60],
    },
    "1h": {
        "detect_tag": "1h",
        "walk_tag": "15m",
        "expansion_bars": 6,        # 6 × 1h = 6h
        "walk_expansion_bars": 24,  # 24 × 15min = 6h
        "mitigation_sweep": [6, 12, 18, 24, 36],
    },
    "4h": {
        "detect_tag": "4h",
        "walk_tag": "1h",
        "expansion_bars": 6,        # 6 × 4h = 24h
        "walk_expansion_bars": 24,  # 24 × 1h = 24h
        "mitigation_sweep": [4, 8, 12, 16, 24],
    },
}

MIN_FVG_BPS_SWEEP = [5, 10, 15, 20, 30, 50, 75, 100]
TIME_PERIOD_SWEEP = [60, 120, 240]

# ── Globals shared via fork COW ──────────────────────────────────────────

_G_DETECT_DATA = {}  # tf_key -> DataFrame
_G_WALK_DATA = {}    # tf_key -> DataFrame


def _init_worker(detect_data, walk_data):
    """Set module-level globals in forked workers."""
    global _G_DETECT_DATA, _G_WALK_DATA
    _G_DETECT_DATA = detect_data
    _G_WALK_DATA = walk_data


def _run_one(args):
    """Worker: run a single config. Returns (label, summary, trade_count)."""
    label, tf_key, config = args

    from logic.utils.btc_fvg_analyzer import analyze_btc_fvgs, summarize_trades, trades_to_dicts

    df_detect = _G_DETECT_DATA[tf_key]
    walk_key = TF_CONFIG[tf_key]["walk_tag"]
    df_walk = _G_WALK_DATA[walk_key]

    t0 = time.time()

    # Suppress stdout
    _real = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        trades = analyze_btc_fvgs(df_detect, df_walk, config)
    finally:
        sys.stdout.close()
        sys.stdout = _real

    elapsed = time.time() - t0
    summary = summarize_trades(trades)
    summary["label"] = label
    summary["elapsed"] = round(elapsed, 1)
    summary["config"] = config
    summary["tf"] = tf_key

    # Convert trades to dicts for JSON
    summary["trades_data"] = trades_to_dicts(trades)

    return summary


# ── Data loading ─────────────────────────────────────────────────────────

def load_resampled_data(data_dir, tf_tag, start_date=None, end_date=None):
    """Load all resampled parquets for a timeframe into a single DataFrame.

    Args:
        data_dir: Base dir (e.g. /home/cr0wn/binance_data/resampled)
        tf_tag: Subdirectory tag (e.g. "5m", "1m", "15m", "1h", "4h")
        start_date: Optional YYYYMMDD filter
        end_date: Optional YYYYMMDD filter

    Returns:
        Sorted DataFrame with columns [date, open, high, low, close, volume].
    """
    import pandas as pd

    pattern = os.path.join(data_dir, tf_tag, f"solusdt_{tf_tag}_*.parquet")
    files = sorted(glob.glob(pattern))

    if start_date:
        files = [f for f in files if _extract_date(f) >= start_date]
    if end_date:
        files = [f for f in files if _extract_date(f) <= end_date]

    if not files:
        raise FileNotFoundError(
            f"No parquet files found for {tf_tag} in {data_dir}/{tf_tag}/ "
            f"(pattern: solusdt_{tf_tag}_*.parquet)"
        )

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if isinstance(df.index, pd.DatetimeIndex) and "date" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        elif "date" in df.columns and df["date"].dtype == object:
            df["date"] = pd.to_datetime(df["date"], utc=True)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(combined["date"]):
        combined["date"] = pd.to_datetime(combined["date"], utc=True)

    print(f"  Loaded {tf_tag}: {len(combined):,} bars from {len(files)} files "
          f"({combined['date'].iloc[0]} → {combined['date'].iloc[-1]})")

    return combined


def _extract_date(filepath):
    """Extract YYYYMMDD from filename like btcusdt_5m_20240101.parquet."""
    base = os.path.basename(filepath)
    parts = base.replace(".parquet", "").split("_")
    return parts[-1]  # YYYYMMDD


# ── Print results ────────────────────────────────────────────────────────

def print_table(results, sort_key="best_ev", limit=30):
    """Print ranked sweep results."""
    # Filter to results that have trades
    results = [r for r in results if r["total_trades"] > 0]

    # Extract best EV for sorting
    for r in results:
        best_ev = -999
        best_setup = ""
        for setup, data in r.get("setups", {}).items():
            if data["best_ev"] > best_ev:
                best_ev = data["best_ev"]
                best_setup = setup
        r["_best_ev"] = best_ev
        r["_best_setup"] = best_setup
        r["_best_n"] = r["setups"].get(best_setup, {}).get("best_n", 0)

    results = sorted(results, key=lambda r: r.get("_best_ev", -999), reverse=True)

    header = (
        f"{'#':>3}  {'Label':<50} {'TF':>4} {'Trades':>6} "
        f"{'Setup':<13} {'Best N':>6} {'EV':>7} "
        f"{'WR@1.5':>6} {'WR@2.0':>6} {'Time Pds':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results[:limit]):
        bs = r["_best_setup"]
        setup_data = r["setups"].get(bs, {})
        rr_stats = setup_data.get("rr_stats", {})

        wr_15 = rr_stats.get(1.5, {}).get("win_rate", 0) * 100
        wr_20 = rr_stats.get(2.0, {}).get("win_rate", 0) * 100

        print(
            f"{i+1:>3}  {r['label']:<50} {r.get('tf', ''):>4} "
            f"{r['total_trades']:>6} {bs:<13} {r['_best_n']:>6.2f} "
            f"{r['_best_ev']:>7.4f} {wr_15:>5.1f}% {wr_20:>5.1f}% "
            f"{r.get('time_periods_with_trades', 0):>8}"
        )

    if len(results) > limit:
        print(f"  ... ({len(results) - limit} more)")
    print("=" * len(header))


def print_per_tf_summary(results):
    """Print best config per timeframe."""
    tf_groups = {}
    for r in results:
        tf = r.get("tf", "?")
        if tf not in tf_groups:
            tf_groups[tf] = []
        tf_groups[tf].append(r)

    print("\n" + "=" * 80)
    print("  PER-TIMEFRAME BEST CONFIGS")
    print("=" * 80)

    for tf in ["5min", "15min", "1h", "4h"]:
        group = tf_groups.get(tf, [])
        if not group:
            continue

        # Filter to those with trades
        group = [r for r in group if r["total_trades"] > 0]
        if not group:
            print(f"\n  {tf}: no configs produced trades")
            continue

        # Best by EV
        for r in group:
            best_ev = -999
            for setup, data in r.get("setups", {}).items():
                if data["best_ev"] > best_ev:
                    best_ev = data["best_ev"]
            r["_sort_ev"] = best_ev

        group.sort(key=lambda r: r["_sort_ev"], reverse=True)
        best = group[0]

        print(f"\n  {tf}:")
        print(f"    Best config:  {best['label']}")
        print(f"    Total trades: {best['total_trades']}")
        for setup, data in best.get("setups", {}).items():
            rr = data.get("rr_stats", {})
            print(f"    {setup}: count={data['count']}, best_n={data['best_n']}, "
                  f"EV={data['best_ev']:.4f}, avg_risk={data['avg_risk_bps']:.0f}bps")
            for nv in [1.0, 1.5, 2.0, 2.5, 3.0]:
                s = rr.get(nv, {})
                if s:
                    print(f"      {nv:.1f}R: WR={s['win_rate']*100:.1f}% EV={s['ev']:.4f} "
                          f"({s['wins']}/{s['total']})")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BTC FVG parameter sweep (parallel)")
    parser.add_argument("--data-dir", default="/home/cr0wn/binance_data/sol_resampled",
                        help="Base directory with resampled parquets")
    parser.add_argument("--tf", nargs="+", default=None,
                        help="Timeframes to sweep (default: all)")
    parser.add_argument("--start", default=None, help="Start date YYYYMMDD")
    parser.add_argument("--end", default=None, help="End date YYYYMMDD")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = auto)")
    parser.add_argument("--output-dir", default=os.path.join(_ROOT, "scripts", "sol_sweep_results"),
                        help="Directory for results JSON")
    parser.add_argument("--save-trades", action="store_true",
                        help="Save per-trade data (large files, needed for re-bucketing)")
    args = parser.parse_args()

    import pandas as pd

    timeframes = args.tf or list(TF_CONFIG.keys())

    # Validate timeframes
    for tf in timeframes:
        if tf not in TF_CONFIG:
            print(f"ERROR: Unknown timeframe '{tf}'. Valid: {list(TF_CONFIG.keys())}")
            sys.exit(1)

    # Auto-size workers
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    max_by_ram = max(2, int((mem_gb - 4) / 1.5))
    max_by_cpu = mp.cpu_count()
    n_workers = args.workers or min(max_by_ram, max_by_cpu)
    print(f"Workers: {n_workers} (RAM: {mem_gb:.0f}GB, {max_by_cpu} cores)")

    # ── Load all required data ──────────────────────────────────────────
    print("\nLoading data...")
    detect_data = {}
    walk_data = {}

    # Collect all needed tags
    needed_detect = set()
    needed_walk = set()
    for tf in timeframes:
        cfg = TF_CONFIG[tf]
        needed_detect.add((tf, cfg["detect_tag"]))
        needed_walk.add(cfg["walk_tag"])

    for tf, tag in needed_detect:
        detect_data[tf] = load_resampled_data(args.data_dir, tag, args.start, args.end)

    for tag in needed_walk:
        if tag not in walk_data:
            walk_data[tag] = load_resampled_data(args.data_dir, tag, args.start, args.end)

    # ── Build all configs ───────────────────────────────────────────────
    all_jobs = []

    for tf in timeframes:
        tf_cfg = TF_CONFIG[tf]
        walk_exp_bars = tf_cfg["walk_expansion_bars"]

        for min_bps in MIN_FVG_BPS_SWEEP:
            for period_min in TIME_PERIOD_SWEEP:
                for mit_bars in tf_cfg["mitigation_sweep"]:
                    label = f"{tf}_bps{min_bps}_p{period_min}m_mit{mit_bars}"
                    config = {
                        "min_fvg_bps": min_bps,
                        "mitigation_window_bars": mit_bars,
                        "expansion_window_bars": walk_exp_bars,
                        "time_period_minutes": period_min,
                    }
                    all_jobs.append((label, tf, config))

    print(f"\nTotal configs: {len(all_jobs)}")
    for tf in timeframes:
        n = sum(1 for j in all_jobs if j[1] == tf)
        print(f"  {tf}: {n} configs")
    print()

    # ── Run sweep ───────────────────────────────────────────────────────
    ctx = mp.get_context("fork")
    t0 = time.time()

    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(detect_data, walk_data),
    ) as pool:
        all_results = []
        total = len(all_jobs)
        done = 0

        for result in pool.imap_unordered(_run_one, all_jobs, chunksize=4):
            done += 1
            trades_n = result["total_trades"]
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done}/{total}] {result['label']} → {trades_n} trades "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")
            all_results.append(result)

    total_elapsed = time.time() - t0
    print(f"\nAll {len(all_results)} configs done in {total_elapsed:.0f}s "
          f"({total_elapsed / len(all_results):.1f}s avg)")

    # ── Results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SWEEP RESULTS — RANKED BY BEST EV")
    print("=" * 80)
    print_table(all_results, limit=40)
    print_per_tf_summary(all_results)

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # Summary (without per-trade data, small file)
    summary_results = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != "trades_data"}
        summary_results.append(r_copy)

    summary_path = os.path.join(args.output_dir, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_results, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    # Per-trade data (large, optional)
    if args.save_trades:
        trades_path = os.path.join(args.output_dir, "sweep_trades.json")
        trades_output = {}
        for r in all_results:
            if r.get("trades_data"):
                trades_output[r["label"]] = r["trades_data"]
        with open(trades_path, "w") as f:
            json.dump(trades_output, f, separators=(",", ":"), default=str)
        size_mb = os.path.getsize(trades_path) / (1024 * 1024)
        print(f"Trade data saved to {trades_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
