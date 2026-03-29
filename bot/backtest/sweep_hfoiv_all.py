#!/usr/bin/env python3
"""
sweep_hfoiv_all.py — Run HFOIV sweeps for all years + continuous, in parallel.

Launches 7 sweep processes simultaneously:
    6 standalone years (2020-2025), each with 3 workers
    + 1 continuous (2020-2025) with 6 workers
    = 24 workers total, fits in 32GB RAM

Results are merged and ranked at the end.

Usage:
    python3 bot/backtest/sweep_hfoiv_all.py --strategy mixed-best-ev-v3-touch-moderate
    python3 bot/backtest/sweep_hfoiv_all.py --balance 100000
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
SWEEP_SCRIPT = os.path.join(_ROOT, "bot", "backtest", "sweep_hfoiv.py")
RESULTS_DIR = os.path.join(_ROOT, "bot", "backtest", "results")


def run_sweep(label, start, end, workers, strategy, balance, out_file):
    """Run a single sweep as a subprocess. Returns (label, out_file, elapsed, returncode)."""
    cmd = [
        sys.executable, SWEEP_SCRIPT,
        "--strategy", strategy,
        "--start", start,
        "--end", end,
        "--balance", str(balance),
        "--workers", str(workers),
        "--json-output", out_file,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    return label, out_file, elapsed, proc.returncode, proc.stderr[-500:] if proc.returncode != 0 else ""


def print_ranked(all_results, sort_key="profit_factor", limit=25):
    """Print a ranked table."""
    results = sorted(all_results, key=lambda r: r.get(sort_key, 0), reverse=True)
    header = (
        f"{'#':>3}  {'Label':<40} {'Trades':>6} {'WR%':>5} {'PF':>5} "
        f"{'Net P&L':>12} {'MaxDD':>9} {'DD%':>5} {'Sharpe':>6} "
        f"{'Scaled':>6}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results[:limit]):
        pnl_str = f"${r['net_pnl']:>10,.0f}"
        dd_str = f"${r['max_dd']:>7,.0f}"
        scaled_str = f"{r['hfoiv_scaled']:>4}" if r["hfoiv_scaled"] else "   -"
        print(
            f"{i+1:>3}  {r['label']:<40} {r['trades']:>6} "
            f"{r['win_rate']:>5.1f} {r['profit_factor']:>5.3f} "
            f"{pnl_str} {dd_str} {r['max_dd_pct']:>5.1f} "
            f"{r['sharpe_approx']:>6.3f} {scaled_str}"
        )
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Run all HFOIV sweeps in parallel")
    parser.add_argument("--strategy", default="mixed-best-ev-v3-touch-moderate")
    parser.add_argument("--balance", type=float, default=100000)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Define all sweep jobs ────────────────────────────────────────
    # 6 standalone × 3 workers + 1 continuous × 6 workers = 24 total
    jobs = [
        ("2020", "20200101", "20201231", 3),
        ("2021", "20210101", "20211231", 3),
        ("2022", "20220101", "20221231", 3),
        ("2023", "20230101", "20231231", 3),
        ("2024", "20240101", "20241231", 3),
        ("2025", "20250101", "20251231", 3),
        ("2020-2025", "20200101", "20251231", 6),
    ]

    # Run sequentially — each sweep gets workers, no CPU contention
    workers = 12
    print(f"Running {len(jobs)} sweeps sequentially ({workers} workers each)")
    print(f"Strategy: {args.strategy}  Balance: ${args.balance:,.0f}")
    print()

    t0 = time.time()

    for label, start, end, _ in jobs:
        out_file = os.path.join(RESULTS_DIR, f"hfoiv_sweep_{label}.json")
        print(f"  [{label}] {start}-{end}...", end="", flush=True)
        label_r, out_f, elapsed, rc, err = run_sweep(
            label, start, end, workers,
            args.strategy, args.balance, out_file
        )
        if rc == 0:
            with open(out_file) as f:
                n = len(json.load(f))
            print(f"  {n} configs in {elapsed:.0f}s")
        else:
            print(f"  FAILED (rc={rc}) {err}")

    total_elapsed = time.time() - t0
    print(f"\nAll sweeps done in {total_elapsed:.0f}s")

    # ── Load and analyze all results ─────────────────────────────────
    print("\n" + "=" * 90)

    # Per-year winners
    yearly_winners = {}
    for label, _, _, _ in jobs:
        out_file = os.path.join(RESULTS_DIR, f"hfoiv_sweep_{label}.json")
        if not os.path.exists(out_file):
            continue
        with open(out_file) as f:
            results = json.load(f)

        baseline = next((r for r in results if r["label"] == "baseline"), None)
        hfoiv_results = [r for r in results if r["label"] != "baseline"]

        best_pf = max(hfoiv_results, key=lambda r: r["profit_factor"])
        best_pnl = max(hfoiv_results, key=lambda r: r["net_pnl"])
        best_sharpe = max(hfoiv_results, key=lambda r: r["sharpe_approx"])

        yearly_winners[label] = {
            "baseline": baseline,
            "best_pf": best_pf,
            "best_pnl": best_pnl,
            "best_sharpe": best_sharpe,
        }

    # Summary table
    print(f"\n{'Year':<12} {'Baseline PF':>12} {'Best PF':>8} {'Config (PF)':>30} "
          f"{'Best P&L':>12} {'Config (P&L)':>30}")
    print("-" * 108)

    for label in ["2020", "2021", "2022", "2023", "2024", "2025", "2020-2025"]:
        w = yearly_winners.get(label)
        if not w:
            continue
        bl = w["baseline"]
        bp = w["best_pf"]
        bn = w["best_pnl"]
        print(f"{label:<12} {bl['profit_factor']:>12.3f} {bp['profit_factor']:>8.3f} "
              f"{bp['label']:>30} ${bn['net_pnl']:>10,.0f} {bn['label']:>30}")

    # Cross-year stability analysis
    print("\n\n=== CROSS-YEAR STABILITY ===")
    print("Configs that appear in the top-10 PF for 4+ years:\n")

    # Collect top-10 PF per standalone year
    from collections import Counter
    top10_counts = Counter()
    for label in ["2020", "2021", "2022", "2023", "2024", "2025"]:
        out_file = os.path.join(RESULTS_DIR, f"hfoiv_sweep_{label}.json")
        if not os.path.exists(out_file):
            continue
        with open(out_file) as f:
            results = json.load(f)
        hfoiv = [r for r in results if r["label"] != "baseline"]
        top10 = sorted(hfoiv, key=lambda r: r["profit_factor"], reverse=True)[:10]
        for r in top10:
            top10_counts[r["label"]] += 1

    stable = [(cfg, count) for cfg, count in top10_counts.most_common() if count >= 3]
    if stable:
        print(f"{'Config':<40} {'Years in Top-10':>15}")
        print("-" * 58)
        for cfg, count in stable:
            years_in = []
            for label in ["2020", "2021", "2022", "2023", "2024", "2025"]:
                out_file = os.path.join(RESULTS_DIR, f"hfoiv_sweep_{label}.json")
                if not os.path.exists(out_file):
                    continue
                with open(out_file) as f:
                    results = json.load(f)
                hfoiv = [r for r in results if r["label"] != "baseline"]
                top10 = sorted(hfoiv, key=lambda r: r["profit_factor"], reverse=True)[:10]
                if any(r["label"] == cfg for r in top10):
                    years_in.append(label)
            print(f"{cfg:<40} {count}/6   ({', '.join(years_in)})")
    else:
        print("No config appeared in top-10 PF for 3+ years")

    # Continuous winner detail
    print("\n\n=== CONTINUOUS 2020-2025 ===")
    out_file = os.path.join(RESULTS_DIR, "hfoiv_sweep_2020-2025.json")
    if os.path.exists(out_file):
        with open(out_file) as f:
            results = json.load(f)
        bl = next(r for r in results if r["label"] == "baseline")
        print(f"\nBaseline: P&L=${bl['net_pnl']:,.0f}  PF={bl['profit_factor']:.3f}  DD={bl['max_dd_pct']:.1f}%")

        print("\nTop 15 by P&L:")
        print_ranked([r for r in results if r["label"] != "baseline"],
                     sort_key="net_pnl", limit=15)

        print("\nTop 15 by Profit Factor:")
        print_ranked([r for r in results if r["label"] != "baseline"],
                     sort_key="profit_factor", limit=15)

        # Best P&L with DD <= baseline
        better = [r for r in results
                  if r["label"] != "baseline"
                  and r["net_pnl"] > bl["net_pnl"]
                  and r["max_dd_pct"] <= bl["max_dd_pct"]]
        if better:
            better.sort(key=lambda r: r["net_pnl"], reverse=True)
            print(f"\nBest P&L with DD ≤ baseline ({bl['max_dd_pct']:.1f}%):")
            print_ranked(better, sort_key="net_pnl", limit=10)

    # Save combined summary
    summary = {
        "yearly_winners": {},
        "cross_year_stable": stable if stable else [],
    }
    for label, w in yearly_winners.items():
        summary["yearly_winners"][label] = {
            "baseline_pf": w["baseline"]["profit_factor"],
            "baseline_pnl": w["baseline"]["net_pnl"],
            "best_pf_config": w["best_pf"]["label"],
            "best_pf_value": w["best_pf"]["profit_factor"],
            "best_pnl_config": w["best_pnl"]["label"],
            "best_pnl_value": w["best_pnl"]["net_pnl"],
        }
    with open(os.path.join(RESULTS_DIR, "hfoiv_sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {RESULTS_DIR}/hfoiv_sweep_summary.json")


if __name__ == "__main__":
    main()
