#!/usr/bin/env python3
"""
sweep_hfoiv.py — Parallel HFOIV gate parameter sweep.

Loads data once, then fans out backtest runs across all CPU cores using
multiprocessing (fork — COW sharing of the read-only DataFrame).

Sweep axes:
    Phase 1: threshold × multiplier (fix rolling=12, lookback=60, bucket=30)
    Phase 2: rolling_bars × lookback_sessions (top configs from Phase 1)
    Phase 3: graduated threshold schemes + bucket width

Usage:
    python3 bot/backtest/sweep_hfoiv.py --strategy mixed-best-ev-v3-touch-moderate
    python3 bot/backtest/sweep_hfoiv.py --strategy-file path/to/strategy.json --start 20250101
    python3 bot/backtest/sweep_hfoiv.py --workers 16  # limit parallelism
"""

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

# ── Globals shared via fork COW ──────────────────────────────────────────
_G_DF = None
_G_STRATEGY = None
_G_BASE_CONFIG = None


def _init_worker(df, strategy, base_config):
    """Called once per forked worker to set module-level globals."""
    global _G_DF, _G_STRATEGY, _G_BASE_CONFIG
    _G_DF = df
    _G_STRATEGY = strategy
    _G_BASE_CONFIG = base_config


def _run_one(args):
    """Worker function: run a single backtest config. Uses fork-inherited globals."""
    label, hfoiv_overrides = args
    # Import here to avoid top-level import cost in workers
    from bot.backtest.backtester import run_backtest

    cfg = copy.deepcopy(_G_BASE_CONFIG)
    cfg["hfoiv"] = hfoiv_overrides

    t0 = time.time()
    _real = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        trades, final_bal = run_backtest(_G_DF, _G_STRATEGY, cfg)
    finally:
        sys.stdout.close()
        sys.stdout = _real
    elapsed = time.time() - t0

    metrics = _summarise(trades, cfg["balance"], final_bal)
    metrics["label"] = label
    metrics["elapsed"] = round(elapsed, 1)
    metrics["config"] = hfoiv_overrides
    return metrics


def _summarise(trades, start_balance, final_balance):
    """Extract key metrics from a backtest run."""
    import numpy as np
    from collections import defaultdict

    if not trades:
        return {
            "trades": 0, "wins": 0, "win_rate": 0, "net_pnl": 0,
            "profit_factor": 0, "max_dd": 0, "max_dd_pct": 0,
            "avg_win": 0, "avg_loss": 0, "final_balance": start_balance,
            "hfoiv_scaled": 0, "sharpe_approx": 0,
        }

    total = len(trades)
    wins = sum(1 for t in trades if t.is_win)
    losses_n = sum(1 for t in trades if t.exit_reason == "SL")
    gross_profit = sum(t.pnl_dollars for t in trades if t.pnl_dollars > 0)
    gross_loss = sum(t.pnl_dollars for t in trades if t.pnl_dollars < 0)
    pf = abs(gross_profit / gross_loss) if gross_loss else float("inf")
    net_pnl = gross_profit + gross_loss

    avg_win = gross_profit / wins if wins else 0
    avg_loss = gross_loss / losses_n if losses_n else 0

    equity = [start_balance]
    for t in trades:
        equity.append(equity[-1] + t.pnl_dollars)
    peak = equity[0]
    max_dd = 0
    max_dd_pct = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        dd_pct = dd / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    # Daily P&L + Sharpe
    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t.date] += t.pnl_dollars
    if len(daily_pnl) > 1:
        dv = np.array(list(daily_pnl.values()))
        sharpe = (dv.mean() / dv.std()) * math.sqrt(252) if dv.std() > 0 else 0
    else:
        sharpe = 0

    # Drawdown percentiles (end-of-day equity)
    _daily_eq = defaultdict(float)
    _run = start_balance
    for t in trades:
        _run += t.pnl_dollars
        _daily_eq[t.date] = _run
    _eod = [_daily_eq[d] for d in sorted(_daily_eq.keys())]
    _eod_dd = []
    _pk = start_balance
    for v in _eod:
        if v > _pk:
            _pk = v
        _eod_dd.append((_pk - v) / _pk * 100 if _pk > 0 else 0)
    _dd_a = np.array(_eod_dd)
    dd_avg = round(float(_dd_a.mean()), 1) if len(_dd_a) else 0
    dd_p50 = round(float(np.percentile(_dd_a, 50)), 1) if len(_dd_a) else 0
    dd_p75 = round(float(np.percentile(_dd_a, 75)), 1) if len(_dd_a) else 0
    dd_p95 = round(float(np.percentile(_dd_a, 95)), 1) if len(_dd_a) else 0

    hfoiv_scaled = sum(1 for t in trades if "HFOIV" in (t.dd_note or ""))

    return {
        "trades": total, "wins": wins,
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "net_pnl": round(net_pnl, 0),
        "profit_factor": round(pf, 3),
        "max_dd": round(max_dd, 0),
        "max_dd_pct": round(max_dd_pct * 100, 1),
        "dd_avg_pct": dd_avg, "dd_p50_pct": dd_p50,
        "dd_p75_pct": dd_p75, "dd_p95_pct": dd_p95,
        "avg_win": round(avg_win, 0), "avg_loss": round(avg_loss, 0),
        "final_balance": round(final_balance, 0),
        "hfoiv_scaled": hfoiv_scaled,
        "sharpe_approx": round(sharpe, 3),
    }


def _run_batch(pool, jobs, all_results, tag=""):
    """Submit a batch of (label, cfg) jobs to the pool and collect results."""
    if not jobs:
        return []
    t0 = time.time()
    print(f"  Submitting {len(jobs)} configs across workers...", flush=True)
    results = pool.map(_run_one, jobs)
    elapsed = time.time() - t0
    all_results.extend(results)
    print(f"  {tag} done — {len(jobs)} configs in {elapsed:.0f}s "
          f"({elapsed/len(jobs):.1f}s avg, {len(jobs)/elapsed:.1f} configs/s)")
    return results


def print_table(results, sort_key="profit_factor", limit=30):
    """Print a formatted comparison table."""
    results = sorted(results, key=lambda r: r.get(sort_key, 0), reverse=True)

    header = (
        f"{'#':>3}  {'Label':<40} {'Trades':>6} {'WR%':>5} {'PF':>5} "
        f"{'Net P&L':>12} {'MaxDD':>9} {'DD%':>5} {'Sharpe':>6} "
        f"{'Scaled':>6}"
    )
    print("\n" + "=" * len(header))
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

    if len(results) > limit:
        print(f"  ... ({len(results) - limit} more)")
    print("=" * len(header))

    best_pf = max(results, key=lambda r: r["profit_factor"])
    best_dd = min(results, key=lambda r: r["max_dd_pct"])
    best_sharpe = max(results, key=lambda r: r["sharpe_approx"])
    best_pnl = max(results, key=lambda r: r["net_pnl"])
    print(f"\n  Best PF:     {best_pf['label']}  (PF={best_pf['profit_factor']:.3f})")
    print(f"  Best DD:     {best_dd['label']}  (DD={best_dd['max_dd_pct']:.1f}%)")
    print(f"  Best Sharpe: {best_sharpe['label']}  (Sharpe={best_sharpe['sharpe_approx']:.3f})")
    print(f"  Best P&L:    {best_pnl['label']}  (${best_pnl['net_pnl']:,.0f})")


def main():
    parser = argparse.ArgumentParser(description="HFOIV gate parameter sweep (parallel)")
    parser.add_argument("--strategy", help="Strategy ID")
    parser.add_argument("--strategy-file", help="Strategy JSON path")
    parser.add_argument("--data-dir", default=os.path.join(_ROOT, "bot", "data"))
    parser.add_argument("--start", default="20240101")
    parser.add_argument("--end", default="20251230")
    parser.add_argument("--balance", type=float, default=76000)
    parser.add_argument("--imbalance-dir",
                        default=os.path.join(_ROOT, "bot", "data", "imbalance"))
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = all cores)")
    parser.add_argument("--json-output", default=None,
                        help="Save all results to JSON")
    args = parser.parse_args()

    # Auto-size workers to fit in RAM.
    # Each forked worker needs ~3GB (groupby copy + arrays).
    # Reserve 4GB for parent + OS.
    _mem_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    _max_by_ram = max(2, int((_mem_gb - 4) / 2))
    _max_by_cpu = mp.cpu_count()
    n_workers = args.workers or min(_max_by_ram, _max_by_cpu)
    print(f"Workers: {n_workers} (RAM: {_mem_gb:.0f}GB → max {_max_by_ram} workers, {_max_by_cpu} cores)")

    # ── Load data once (shared via fork COW) ─────────────────────────
    from bot.backtest.backtester import load_1s_bars, load_strategy
    print("Loading data...")
    strategy = load_strategy(args.strategy, args.strategy_file)
    df_1s = load_1s_bars(args.data_dir, args.start, args.end)

    # Pre-add trade_date column so run_backtest's mutation doesn't
    # trigger a COW page copy in every forked worker
    df_1s['trade_date'] = df_1s['date'].dt.date

    base_config = {
        "balance": args.balance,
        "risk_pct": 0.01,  # fallback; overridden by risk_tiers when active
        "max_concurrent": 3,
        "max_daily_trades": 15,
        "min_fvg_size": 0.25,
        "slip": False,
        "risk_tiers": True,
        "mit_entry_ticks": 0,
        "tp_mode": "fixed",
        "margin_per_contract": 33000.0,
        "tier_reset": False,
        "streak": False,
        "streak_base": 0.005,
        "streak_mult": 1.5,
        "streak_max": 0.04,
        "imbalance_dir": args.imbalance_dir,
    }

    # ── Build ALL configs upfront ────────────────────────────────────
    all_jobs = []

    # Baseline
    all_jobs.append(("baseline", {"enabled": False}))

    # Phase 1: threshold × multiplier
    for pct in [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
        for mult in [0.25, 0.50, 0.75]:
            all_jobs.append((
                f"p{pct}_x{mult}",
                {"enabled": True, "rolling_bars": 12, "lookback_sessions": 60,
                 "bucket_minutes": 30, "thresholds": [(pct, mult)]},
            ))

    # Graduated schemes
    for name, thresholds in [
        ("grad_50_65_80", [(50, 0.75), (65, 0.50), (80, 0.25)]),
        ("grad_55_70_85", [(55, 0.75), (70, 0.50), (85, 0.25)]),
        ("grad_60_75_90", [(60, 0.75), (75, 0.50), (90, 0.25)]),
        ("grad_65_80_95", [(65, 0.75), (80, 0.50), (95, 0.25)]),
        ("grad_70_80_90", [(70, 0.75), (80, 0.50), (90, 0.25)]),
        ("grad_75_85_95", [(75, 0.75), (85, 0.50), (95, 0.25)]),
        ("grad_60_70_80", [(60, 0.75), (70, 0.50), (80, 0.25)]),
        ("grad_65_75_85", [(65, 0.75), (75, 0.50), (85, 0.25)]),
        ("grad_80_90_95", [(80, 0.75), (90, 0.50), (95, 0.25)]),
        # Two-tier
        ("grad2_60_80",  [(60, 0.75), (80, 0.25)]),
        ("grad2_65_85",  [(65, 0.75), (85, 0.25)]),
        ("grad2_70_90",  [(70, 0.75), (90, 0.25)]),
        ("grad2_75_95",  [(75, 0.75), (95, 0.25)]),
        ("grad2_50_75",  [(50, 0.75), (75, 0.25)]),
        ("grad2_55_80",  [(55, 0.75), (80, 0.25)]),
    ]:
        all_jobs.append((
            name,
            {"enabled": True, "rolling_bars": 12, "lookback_sessions": 60,
             "bucket_minutes": 30, "thresholds": thresholds},
        ))

    # Phase 2: rolling × lookback sweep on representative configs
    for pct in [50, 60, 70, 80, 90]:
        for mult in [0.25, 0.50]:
            for rolling in [6, 9, 15, 18, 24]:
                for lookback in [30, 60, 90]:
                    all_jobs.append((
                        f"p{pct}_x{mult}_r{rolling}_lb{lookback}",
                        {"enabled": True, "rolling_bars": rolling,
                         "lookback_sessions": lookback,
                         "bucket_minutes": 30, "thresholds": [(pct, mult)]},
                    ))

    # Phase 3: bucket width on representative configs
    for pct in [60, 70, 80]:
        for mult in [0.25, 0.50]:
            for rolling in [12, 18]:
                for bkt in [15, 60]:
                    all_jobs.append((
                        f"p{pct}_x{mult}_r{rolling}_bkt{bkt}",
                        {"enabled": True, "rolling_bars": rolling,
                         "lookback_sessions": 60,
                         "bucket_minutes": bkt, "thresholds": [(pct, mult)]},
                    ))

    print(f"\nTotal configs: {len(all_jobs)} (baseline + {len(all_jobs)-1} HFOIV)")
    print()

    # ── Run all in parallel ──────────────────────────────────────────
    # Use fork context so workers inherit df_1s via COW
    ctx = mp.get_context("fork")
    all_results = []

    t0 = time.time()
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(df_1s, strategy, base_config),
    ) as pool:
        all_results = pool.map(_run_one, all_jobs)

    total_elapsed = time.time() - t0
    print(f"\nAll {len(all_results)} configs done in {total_elapsed:.0f}s "
          f"({total_elapsed/len(all_results):.1f}s avg)")

    # ── Final rankings ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL RANKINGS — BY PROFIT FACTOR")
    print("=" * 80)
    print_table(all_results, sort_key="profit_factor", limit=40)

    print("\n  RANKED BY SHARPE:")
    print_table(all_results, sort_key="sharpe_approx", limit=20)

    # ── Save ─────────────────────────────────────────────────────────
    out_path = args.json_output or os.path.join(
        _ROOT, "bot", "backtest", "results", "hfoiv_sweep.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for r in all_results:
        if "config" in r:
            cfg = r["config"]
            if "thresholds" in cfg:
                cfg["thresholds"] = [list(t) for t in cfg["thresholds"]]
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
