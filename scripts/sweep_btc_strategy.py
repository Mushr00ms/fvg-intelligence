#!/usr/bin/env python3
"""
sweep_btc_strategy.py — Find optimal cell selection for BTC FVG strategy.

Loads per-trade data, groups into cells, sweeps selection thresholds,
replays equity curves chronologically. Finds Pareto frontier of PnL vs DD.

Sweep axes:
  - min_ev: minimum best-EV to include a cell (0.01 .. 0.30)
  - min_samples: minimum samples per cell (50 .. 300)
  - setup: mit_extreme only, or both
  - max_trades_per_day: cap daily trades (3, 5, 8, 15, unlimited)

Usage:
    python3 scripts/sweep_btc_strategy.py
    python3 scripts/sweep_btc_strategy.py --year 2024
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

N_VALUES = [round(1.0 + i * 0.25, 2) for i in range(9)]
RISK_BINS = [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 994]


# ── Cell building ────────────────────────────────────────────────────────

def build_cells(trades):
    """Group trades into (time_period, risk_bucket, setup) cells.
    Returns dict: cell_key -> {best_n, best_ev, samples, trades: [list]}
    """
    groups = defaultdict(list)
    for t in trades:
        rb = None
        for i in range(len(RISK_BINS) - 1):
            if RISK_BINS[i] <= t["risk_bps"] < RISK_BINS[i + 1]:
                rb = f"{RISK_BINS[i]}-{RISK_BINS[i+1]}"
                break
        if rb is None:
            continue
        key = (t["time_period"], rb, t["setup"])
        groups[key].append(t)

    cells = {}
    for key, group in groups.items():
        n = len(group)
        if n < 10:
            continue

        # Find best N for this cell
        best_ev = -999
        best_n = 1.0
        for nv in N_VALUES:
            nv_str = str(nv)
            wins = sum(1 for t in group if t["outcomes"].get(nv_str) is True)
            wr = wins / n
            ev = wr * (nv + 1) - 1
            if ev > best_ev:
                best_ev = ev
                best_n = nv

        cells[key] = {
            "best_n": best_n,
            "best_ev": round(best_ev, 4),
            "samples": n,
            "trades": group,
        }

    return cells


# ── Equity replay ────────────────────────────────────────────────────────

@dataclass
class ReplayResult:
    trades: int
    wins: int
    win_rate: float
    net_pnl: float
    max_dd_pct: float
    sharpe: float
    final_balance: float
    trades_per_day: float
    cells_used: int
    avg_ev: float


def replay_equity(
    cells,
    min_ev,
    min_samples,
    setups,
    max_trades_day,
    balance=100000,
    risk_pct=0.01,
    year_filter=None,
):
    """Replay trades chronologically through qualifying cells.

    Each qualifying trade risks a fixed risk_pct of starting balance.
    Win = +best_n * risk, Loss = -1 * risk.
    """
    # Select qualifying cells
    qualifying = {}
    for key, cell in cells.items():
        tp, rb, setup = key
        if setup not in setups:
            continue
        if cell["best_ev"] < min_ev:
            continue
        if cell["samples"] < min_samples:
            continue
        qualifying[key] = cell

    if not qualifying:
        return None

    # Collect all trades from qualifying cells, tag with cell's best_n
    all_trades = []
    for key, cell in qualifying.items():
        best_n = cell["best_n"]
        nv_str = str(best_n)
        for t in cell["trades"]:
            ft = t["formation_time"]
            if year_filter and not ft.startswith(year_filter):
                continue
            won = t["outcomes"].get(nv_str) is True
            all_trades.append((ft, won, best_n, t["risk_bps"]))

    if not all_trades:
        return None

    # Sort chronologically
    all_trades.sort(key=lambda x: x[0])

    # Replay
    bal = balance
    peak = bal
    max_dd_pct = 0
    wins = 0
    daily_pnl = defaultdict(float)
    daily_count = defaultdict(int)
    executed = 0

    for ft, won, best_n, risk_bps in all_trades:
        day = ft[:10]

        # Daily trade cap
        if max_trades_day and daily_count[day] >= max_trades_day:
            continue

        risk_dollar = balance * risk_pct  # fixed risk, no compounding
        if won:
            pnl = risk_dollar * best_n
            wins += 1
        else:
            pnl = -risk_dollar

        bal += pnl
        daily_pnl[day] += pnl
        daily_count[day] += 1
        executed += 1

        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak if peak > 0 else 0
        if dd > max_dd_pct:
            max_dd_pct = dd

    if executed == 0:
        return None

    # Sharpe from daily P&L
    import numpy as np
    dpnl = np.array(list(daily_pnl.values()))
    sharpe = 0
    if len(dpnl) > 1 and dpnl.std() > 0:
        sharpe = (dpnl.mean() / dpnl.std()) * math.sqrt(365)

    trading_days = len(daily_count)
    avg_ev = sum(c["best_ev"] for c in qualifying.values()) / len(qualifying)

    return ReplayResult(
        trades=executed,
        wins=wins,
        win_rate=round(wins / executed * 100, 1),
        net_pnl=round(bal - balance, 0),
        max_dd_pct=round(max_dd_pct * 100, 1),
        sharpe=round(sharpe, 3),
        final_balance=round(bal, 0),
        trades_per_day=round(executed / max(trading_days, 1), 1),
        cells_used=len(qualifying),
        avg_ev=round(avg_ev, 4),
    )


# ── Sweep ────────────────────────────────────────────────────────────────

def run_sweep(cells, year_filter=None):
    """Sweep all parameter combos, return list of results."""
    results = []

    ev_sweep = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    samples_sweep = [30, 50, 80, 100, 150, 200, 300]
    setup_sweep = [
        ("mit_only", {"mit_extreme"}),
        ("both", {"mit_extreme", "mid_extreme"}),
    ]
    max_trades_sweep = [3, 5, 8, 15, 999]

    total = len(ev_sweep) * len(samples_sweep) * len(setup_sweep) * len(max_trades_sweep)
    done = 0

    for min_ev in ev_sweep:
        for min_samp in samples_sweep:
            for setup_label, setups in setup_sweep:
                for max_td in max_trades_sweep:
                    r = replay_equity(
                        cells, min_ev, min_samp, setups, max_td,
                        year_filter=year_filter,
                    )
                    done += 1
                    if r is None:
                        continue

                    label = f"ev{min_ev:.2f}_s{min_samp}_{setup_label}_td{max_td}"
                    results.append({
                        "label": label,
                        "min_ev": min_ev,
                        "min_samples": min_samp,
                        "setups": setup_label,
                        "max_trades_day": max_td,
                        **vars(r),
                    })

    print(f"  {done} configs swept, {len(results)} with trades")
    return results


# ── Display ──────────────────────────────────────────────────────────────

def print_table(results, sort_key="net_pnl", limit=30, title=""):
    results = sorted(results, key=lambda r: r.get(sort_key, 0), reverse=True)

    if title:
        print(f"\n{'='*130}")
        print(f"  {title}")
        print(f"{'='*130}")

    header = (
        f"{'#':>3} {'Label':<38} {'Cells':>5} {'Trades':>6} {'T/Day':>5} "
        f"{'WR%':>5} {'Net PnL':>12} {'DD%':>5} {'Sharpe':>6} {'AvgEV':>7}"
    )
    print(header)
    print("-" * 130)

    for i, r in enumerate(results[:limit]):
        pnl_str = f"${r['net_pnl']:>10,.0f}"
        print(
            f"{i+1:>3} {r['label']:<38} {r['cells_used']:>5} {r['trades']:>6} "
            f"{r['trades_per_day']:>5.1f} {r['win_rate']:>5.1f} {pnl_str} "
            f"{r['max_dd_pct']:>5.1f} {r['sharpe']:>6.3f} {r['avg_ev']:>7.4f}"
        )

    print("=" * 130)


def find_pareto(results):
    """Find Pareto-optimal configs (max PnL for given DD)."""
    # Sort by DD ascending
    by_dd = sorted(results, key=lambda r: r["max_dd_pct"])
    pareto = []
    best_pnl = -float("inf")
    for r in by_dd:
        if r["net_pnl"] > best_pnl:
            best_pnl = r["net_pnl"]
            pareto.append(r)
    # Also include configs that dominate on Sharpe
    return pareto


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-file",
                        default=os.path.join(_ROOT, "scripts", "btc_sweep_results",
                                             "5min_results", "sweep_trades.json"))
    parser.add_argument("--config", default="5min_bps5_p60m_mit90")
    parser.add_argument("--year", default=None, help="Filter to year (e.g. 2024)")
    parser.add_argument("--output", default=os.path.join(_ROOT, "scripts",
                                                          "btc_sweep_results", "strategy_sweep.json"))
    args = parser.parse_args()

    print(f"Loading trades from {args.config}...")
    with open(args.trades_file) as f:
        all_trades = json.load(f)
    trades = all_trades[args.config]
    print(f"  {len(trades)} trades")

    if args.year:
        trades = [t for t in trades if t["formation_time"].startswith(args.year)]
        print(f"  Filtered to {args.year}: {len(trades)} trades")

    print("Building cells...")
    cells = build_cells(trades)
    print(f"  {len(cells)} cells")

    # Show cell distribution
    ev_dist = defaultdict(int)
    for c in cells.values():
        bucket = round(c["best_ev"] * 10) / 10  # round to 0.1
        ev_dist[bucket] += 1
    print("  EV distribution:")
    for ev in sorted(ev_dist.keys()):
        print(f"    EV >= {ev:.1f}: {sum(v for k, v in ev_dist.items() if k >= ev)} cells")

    print("\nRunning sweep...")
    results = run_sweep(cells, year_filter=args.year)

    # Rankings
    print_table(results, sort_key="net_pnl", limit=20, title="RANKED BY NET P&L")
    print_table(results, sort_key="sharpe", limit=20, title="RANKED BY SHARPE")

    # Pareto frontier
    pareto = find_pareto(results)
    print_table(pareto, sort_key="max_dd_pct", limit=20, title="PARETO FRONTIER (max PnL per DD level)")

    # Sweet spot analysis
    print("\n  SWEET SPOT ANALYSIS:")
    for target_dd in [5, 8, 10, 15, 20]:
        candidates = [r for r in results if r["max_dd_pct"] <= target_dd]
        if candidates:
            best = max(candidates, key=lambda r: r["net_pnl"])
            print(f"    DD <= {target_dd:>2}%: {best['label']:<40} PnL=${best['net_pnl']:>10,.0f}  "
                  f"DD={best['max_dd_pct']}%  Sharpe={best['sharpe']:.3f}  "
                  f"Trades={best['trades']}  T/Day={best['trades_per_day']}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
