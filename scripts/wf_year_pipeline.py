#!/usr/bin/env python3
"""
wf_year_pipeline.py — End-to-end walk-forward pipeline for one (train_window, test_year) pair.

For a given train window and test year, this script:
  1. Builds an unbounded candidate strategy from a fvg parquet via the selector
  2. Runs a train backtest over the train window
  3. Drops cells with negative train P&L (rule A)
  4. Runs the OOS backtest on the test year
  5. Saves the OOS result to the dashboard manifest

Assumes:
  - The fvg parquet for the train window already exists in logic/fvg_cache/
    (run logic/main.py with FVG_START/FVG_END first)
  - 1-second bar data for the test year is in bot/data/

Usage:
    python3 scripts/wf_year_pipeline.py \\
        --parquet logic/fvg_cache/fvg_results_5min_<hash>.parquet \\
        --train-start 20200101 --train-end 20221231 \\
        --test-start 20230101 --test-end 20231231 \\
        --label wf-2020-2022-test-2023
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STRATEGIES = os.path.join(_ROOT, "logic", "strategies")
_TMP = "/tmp"

sys.path.insert(0, _ROOT)
from bot.backtest.backtest_store import save_results  # noqa: E402


def _run(cmd: list[str], log_path: str) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    print(f"[LOG] {log_path}")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        with open(log_path) as f:
            print(f.read()[-2000:])
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="FVG parquet for the train window")
    p.add_argument("--train-start", required=True, help="YYYYMMDD")
    p.add_argument("--train-end",   required=True, help="YYYYMMDD")
    p.add_argument("--test-start",  required=True, help="YYYYMMDD")
    p.add_argument("--test-end",    required=True, help="YYYYMMDD")
    p.add_argument("--label",       required=True, help="Short label for filenames, e.g. wf-2020-2022-test-2023")
    p.add_argument("--balance",     type=float, default=80000)
    p.add_argument("--risk-pct",    type=float, default=0.01)
    p.add_argument("--source-dataset", default="rr_nq_5min_custom_30min")
    args = p.parse_args()

    candidate_path = os.path.join(_STRATEGIES, f"{args.label}-candidate.json")
    pruned_path    = os.path.join(_STRATEGIES, f"{args.label}-pruned.json")
    train_log      = os.path.join(_TMP, f"{args.label}_train.log")
    train_json     = os.path.join(_TMP, f"{args.label}_train.json")
    test_log       = os.path.join(_TMP, f"{args.label}_test.log")
    test_json      = os.path.join(_TMP, f"{args.label}_test.json")

    # 1. Build candidate strategy
    print(f"\n=== {args.label}: build candidate ===")
    _run([
        "python3", "scripts/build_nq_strategy_from_window.py",
        "--parquet", args.parquet,
        "--out", candidate_path,
        "--strategy-id", f"{args.label}-candidate",
        "--strategy-name", f"{args.label} candidate (unbounded)",
        "--source-dataset", args.source_dataset,
        "--description", f"WF train {args.train_start}->{args.train_end}, test {args.test_start}->{args.test_end}",
    ], os.path.join(_TMP, f"{args.label}_build.log"))

    # 2. Train backtest
    print(f"\n=== {args.label}: train backtest ===")
    _run([
        "python3", "bot/backtest/backtester.py",
        "--strategy-file", candidate_path,
        "--start", args.train_start, "--end", args.train_end,
        "--risk-tiers",
        "--hfoiv", "--hfoiv-rolling", "6", "--hfoiv-lookback", "90",
        "--balance", str(args.balance), "--risk-pct", str(args.risk_pct),
        "--json-output", train_json,
    ], train_log)

    # 3. Prune cells with negative train P&L
    train = json.load(open(train_json))
    losers = {(c['time_period'], c['risk_range'], c['setup'], round(float(c['n_value']), 2))
              for c in train['cell_performance'] if c['total_pnl'] < 0}
    print(f"[PRUNE] {len(losers)} cells with negative train P&L")

    src = json.load(open(candidate_path))
    kept = [c for c in src['cells']
            if (c['time_period'], c['risk_range'], c['setup'], round(float(c['rr_target']), 2)) not in losers]
    print(f"[PRUNE] candidate {len(src['cells'])} -> pruned {len(kept)}")

    ts = sum(c['samples'] for c in kept) or 1
    wev = round(sum(c['ev'] * c['samples'] for c in kept) / ts, 4)
    tpd = round(sum((c.get('trades_per_day') or 0) for c in kept), 4)
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    pruned = dict(src)
    pruned['meta'] = dict(src['meta'])
    pruned['meta']['id'] = f"{args.label}-pruned"
    pruned['meta']['name'] = f"{args.label} (pruned)"
    pruned['meta']['updated_at'] = now
    pruned['meta']['description'] = src['meta']['description'] + f" Pruned {len(losers)} negative-train-PnL cells."
    pruned['cells'] = kept
    pruned['stats'] = {
        'total_cells': len(kept), 'enabled_cells': len(kept),
        'weighted_ev': wev, 'expected_trades_per_day': tpd,
        'time_coverage': sorted({c['time_period'] for c in kept}),
    }
    json.dump(pruned, open(pruned_path, 'w'), indent=2)
    print(f"[WRITE] {pruned_path}  cells={len(kept)} wEV={wev} tpd={tpd}")

    # 4. OOS test backtest
    print(f"\n=== {args.label}: OOS test backtest ===")
    _run([
        "python3", "bot/backtest/backtester.py",
        "--strategy-file", pruned_path,
        "--start", args.test_start, "--end", args.test_end,
        "--risk-tiers",
        "--hfoiv", "--hfoiv-rolling", "6", "--hfoiv-lookback", "90",
        "--balance", str(args.balance), "--risk-pct", str(args.risk_pct),
        "--json-output", test_json,
    ], test_log)

    # 5. Save OOS result to dashboard
    results = json.load(open(test_json))
    results.setdefault('meta', {})
    results['meta']['strategy_id'] = f"{args.label}-pruned"
    results['meta']['strategy_name'] = f"WF train {args.train_start[:4]}-{args.train_end[:4]} (pruned, {len(kept)} cells) — OOS {args.test_start[:4]}"
    run_id = save_results(results)

    s = results['summary']
    print()
    print(f"=== {args.label}: DONE ===")
    print(f"  cells: {len(kept)} (pruned from {len(src['cells'])})")
    print(f"  OOS  : {s['total_trades']} trades  P&L=${s['net_pnl']:,.0f} ({s['pnl_pct']:+.1f}%)  PF={s['profit_factor']:.2f}  maxDD={s['max_dd_pct']:.1f}%")
    print(f"  saved to dashboard as run_id={run_id}")


if __name__ == '__main__':
    main()
