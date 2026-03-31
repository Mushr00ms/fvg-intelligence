#!/usr/bin/env python3
"""
sweep_risk_buckets_nq.py — Sweep the 40-80 NQ risk bucket split point.

NQ volatility has grown over time, making the 40-80 pt bucket heterogeneous.
This script tests alternative splits (e.g. 40-60 + 60-X) to find configurations
that yield more EV from the current strategy's qualifying cells.

Requires the fvg_cache parquet to exist (run main.py once first):
    cd logic && FVG_TICKER=NQ FVG_TIMEFRAME=5min FVG_PERIOD="5 years" python main.py

Usage:
    python3 scripts/sweep_risk_buckets_nq.py
    python3 scripts/sweep_risk_buckets_nq.py --parquet logic/fvg_cache/fvg_results_5min_XXX.parquet
    python3 scripts/sweep_risk_buckets_nq.py --min-ev 0.10 --min-samples 30
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "logic"))

from datetime import time as dtime
from utils.rr_analysis import aggregate_rr_cells, N_VALUES, SETUPS

# ── Config ────────────────────────────────────────────────────────────────

# Matching the current active strategy's source dataset
SESSION_START = dtime(9, 30)
SESSION_END = dtime(16, 0)
INTERVAL_MINUTES = 30

BASE_BINS = [5, 10, 15, 20, 25, 30, 40, 80]

# The 40-80 split variants to try:
#   split_at: where to divide 40-80 (e.g. 60 → 40-60, 60-upper)
#   upper:    upper edge of the last bucket (80 = keep current, 100/120/150/200 = extend)
SPLIT_POINTS = list(range(45, 80, 5))   # 45, 50, 55, 60, 65, 70, 75
UPPER_BOUNDS  = [80, 100, 120, 150, 200] # extend beyond current 80


def build_bin_configs():
    """Return list of (label, bins) for all sweep variants plus the baseline."""
    configs = [("baseline", BASE_BINS)]

    for split in SPLIT_POINTS:
        for upper in UPPER_BOUNDS:
            if split >= upper:
                continue
            bins = BASE_BINS[:-1] + [split, upper]   # replace 80 → split, upper
            label = f"split{split}_upper{upper}"
            configs.append((label, bins))

    return configs


# ── Per-cell EV quality ───────────────────────────────────────────────────

def best_ev_for_setup(setup_data):
    """Return (best_ev, best_n_idx, samples) for a setup dict."""
    evs = setup_data.get("evs", [])
    n_valid = setup_data.get("valid", 0)
    if not evs or n_valid == 0:
        return None, None, 0
    best_idx = int(np.argmax(evs))
    return evs[best_idx], best_idx, n_valid


def score_cells(cells, min_ev, min_samples):
    """
    Score cells that are in the 40+ risk range.

    Returns dict with aggregate stats.
    """
    qualifying = []
    for cell in cells:
        lo = int(cell["risk_range"].split("-")[0])
        if lo < 40:
            continue

        for setup in SETUPS:
            sd = cell["setups"].get(setup, {})
            best_ev, best_idx, samples = best_ev_for_setup(sd)
            if best_ev is None or best_ev < min_ev or samples < min_samples:
                continue

            qualifying.append({
                "time_period": cell["time_period"],
                "risk_range": cell["risk_range"],
                "setup": setup,
                "best_ev": best_ev,
                "best_n": N_VALUES[best_idx],
                "samples": samples,
                "all_evs": sd.get("evs", []),
            })

    if not qualifying:
        return {
            "n_cells": 0,
            "total_samples": 0,
            "avg_ev": 0.0,
            "max_ev": 0.0,
            "ev_weighted_samples": 0.0,
            "cells": [],
        }

    total_samples = sum(q["samples"] for q in qualifying)
    avg_ev = np.mean([q["best_ev"] for q in qualifying])
    max_ev = max(q["best_ev"] for q in qualifying)
    ev_weighted = sum(q["best_ev"] * q["samples"] for q in qualifying)

    return {
        "n_cells": len(qualifying),
        "total_samples": total_samples,
        "avg_ev": round(avg_ev, 4),
        "max_ev": round(max_ev, 4),
        "ev_weighted_samples": round(ev_weighted, 1),
        "cells": qualifying,
    }


# ── Main sweep ────────────────────────────────────────────────────────────

def run_sweep(df_fvgs, min_ev, min_samples):
    configs = build_bin_configs()
    results = []

    print(f"\nSweeping {len(configs)} bin configurations...\n")

    for label, bins in configs:
        cells = aggregate_rr_cells(
            df_fvgs,
            fvg_filter_start_time=SESSION_START,
            fvg_filter_end_time=SESSION_END,
            interval_minutes=INTERVAL_MINUTES,
            risk_bins=bins,
            min_samples=1,  # keep all, filter in score_cells
        )
        score = score_cells(cells, min_ev=min_ev, min_samples=min_samples)
        results.append({
            "label": label,
            "bins": bins,
            "bins_40plus": [b for b in bins if b >= 40],
            **score,
        })

    return results


def print_results(results, baseline_label="baseline"):
    baseline = next((r for r in results if r["label"] == baseline_label), None)

    def delta(val, base_val, fmt=".1f"):
        d = val - base_val
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:{fmt}}"

    print(f"\n{'='*110}")
    print(f"  40+ BUCKET SWEEP  (min_ev threshold applied, sorted by ev_weighted_samples)")
    print(f"{'='*110}")
    print(
        f"{'#':>3} {'Label':<28} {'Buckets (40+)':<22} "
        f"{'Cells':>5} {'Samples':>7} {'AvgEV':>7} {'MaxEV':>7} {'EVxS':>9} "
        f"{'ΔCells':>7} {'ΔAvgEV':>8} {'ΔEVxS':>9}"
    )
    print("-" * 110)

    sorted_r = sorted(results, key=lambda r: r["ev_weighted_samples"], reverse=True)

    for i, r in enumerate(sorted_r):
        buckets_str = "→".join(
            f"{r['bins_40plus'][j]}-{r['bins_40plus'][j+1]}"
            for j in range(len(r["bins_40plus"]) - 1)
        )
        if baseline and r["label"] != baseline_label:
            dc = delta(r["n_cells"], baseline["n_cells"], fmt="d")
            da = delta(r["avg_ev"], baseline["avg_ev"], fmt=".4f")
            de = delta(r["ev_weighted_samples"], baseline["ev_weighted_samples"], fmt=".0f")
        else:
            dc = da = de = "—"

        marker = " ◄ baseline" if r["label"] == baseline_label else ""
        print(
            f"{i+1:>3} {r['label']:<28} {buckets_str:<22} "
            f"{r['n_cells']:>5} {r['total_samples']:>7} {r['avg_ev']:>7.4f} "
            f"{r['max_ev']:>7.4f} {r['ev_weighted_samples']:>9.1f} "
            f"{dc:>7} {da:>8} {de:>9}{marker}"
        )

    print("=" * 110)

    # Best config
    non_baseline = [r for r in sorted_r if r["label"] != baseline_label]
    if non_baseline:
        best = non_baseline[0]
        print(f"\n  Best config: {best['label']}")
        bp = best['bins_40plus']
        bucket_str = " → ".join(f"{bp[j]}-{bp[j+1]}" for j in range(len(bp)-1))
        print(f"  Buckets (40+): {bucket_str}")
        print(f"  Cells: {best['n_cells']}  AvgEV: {best['avg_ev']:.4f}  EVxSamples: {best['ev_weighted_samples']:.0f}")
        if baseline:
            print(f"  vs baseline: cells {delta(best['n_cells'], baseline['n_cells'], fmt='d')}  "
                  f"AvgEV {delta(best['avg_ev'], baseline['avg_ev'], fmt='.4f')}  "
                  f"EVxSamples {delta(best['ev_weighted_samples'], baseline['ev_weighted_samples'], fmt='.0f')}")

        print(f"\n  Qualifying cells in best config:")
        print(f"  {'Time Period':<16} {'Risk Range':<12} {'Setup':<14} {'BestN':>6} {'EV':>8} {'Samples':>8}")
        print(f"  {'-'*68}")
        for c in sorted(best["cells"], key=lambda x: (-x["best_ev"], x["time_period"])):
            print(f"  {c['time_period']:<16} {c['risk_range']:<12} {c['setup']:<14} "
                  f"{c['best_n']:>6.2f} {c['best_ev']:>8.4f} {c['samples']:>8}")


def find_parquet(parquet_arg):
    """Find the 5min fvg_cache parquet with RR columns."""
    if parquet_arg:
        return parquet_arg

    pattern = os.path.join(_ROOT, "logic", "fvg_cache", "fvg_results_5min_*.parquet")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default=None,
                        help="Path to fvg_results_5min_*.parquet (auto-detected if omitted)")
    parser.add_argument("--min-ev", type=float, default=0.10,
                        help="Minimum best-EV to count a cell as qualifying (default: 0.10)")
    parser.add_argument("--min-samples", type=int, default=30,
                        help="Minimum samples to count a cell as qualifying (default: 30)")
    args = parser.parse_args()

    parquet = find_parquet(args.parquet)
    if not parquet:
        print(
            "\nERROR: No fvg_cache parquet found.\n"
            "Run the pipeline first to generate it:\n\n"
            "    cd logic\n"
            "    FVG_TICKER=NQ FVG_TIMEFRAME=5min FVG_PERIOD='5 years' python main.py\n\n"
            "Then re-run this script.\n"
        )
        sys.exit(1)

    print(f"Loading: {os.path.relpath(parquet, _ROOT)}")
    df = pd.read_parquet(parquet)
    print(f"  {len(df)} FVGs loaded")

    # Verify RR columns are present
    rr_cols = [c for c in df.columns if c.startswith("rr_")]
    if not rr_cols:
        print(
            "\nERROR: Parquet has no RR simulation columns.\n"
            "Re-run the pipeline so RR columns are computed and cached.\n"
        )
        sys.exit(1)

    print(f"  RR columns: {len(rr_cols)} found")
    print(f"  min_ev threshold: {args.min_ev}  min_samples: {args.min_samples}")

    results = run_sweep(df, min_ev=args.min_ev, min_samples=args.min_samples)
    print_results(results)


if __name__ == "__main__":
    main()
