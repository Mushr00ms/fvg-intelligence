#!/usr/bin/env python3
"""Find SOL configs that are profitable EVERY year individually @ 1x leverage.

Loads sweep_trades.json once, then for each (config × cell-selection) computes
per-year PnL/DD/Sharpe from a single chronological replay. Ranks by worst-year
PnL (the robustness floor).
"""
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

# Reuse helpers from sweep_sol_final
sys.path.insert(0, os.path.dirname(__file__))
from sweep_sol_final import (
    _bucket, build_cells, prepare_trades, TF_MAX_HOLD_MIN,
    EV_SWEEP, SAMPLE_SWEEP, SETUP_SWEEP,
    CAPITAL, RISK_PCT, ENTRY_FEE, TP_FEE, SL_FEE,
)

YEARS = ["2021", "2022", "2023", "2024", "2025"]
LEVERAGE = 1.0


def replay_yearly(prepared, qual_keys, cell_best_n, max_hold_min):
    """Single chronological replay; bucket per-year stats."""
    cap_notional = CAPITAL * LEVERAGE
    risk_dollar = CAPITAL * RISK_PCT
    max_hold_s = max_hold_min * 60

    open_releases = []
    open_notionals = []

    # Per-year tracking. Equity curve continues across years (compounding off, fixed risk).
    yr_pnl = defaultdict(float)
    yr_wins = defaultdict(int)
    yr_losses = defaultdict(int)
    yr_daily = defaultdict(lambda: defaultdict(float))

    # For DD calc: per-year peak relative to running balance
    bal = CAPITAL
    yr_peak = {}
    yr_max_dd = defaultdict(float)
    yr_start_bal = {}

    last_year = None

    for entry_epoch, day_str, key, rbps, outcomes in prepared:
        if key not in qual_keys:
            continue
        bn = cell_best_n[key]
        won = outcomes.get(str(bn)) is True

        if open_releases:
            keep_r, keep_n = [], []
            for r, n in zip(open_releases, open_notionals):
                if r > entry_epoch:
                    keep_r.append(r); keep_n.append(n)
            open_releases, open_notionals = keep_r, keep_n

        notional = risk_dollar * 10000 / rbps
        if sum(open_notionals) + notional > cap_notional:
            continue

        if won:
            fee = notional * (ENTRY_FEE + TP_FEE)
            pnl = risk_dollar * bn - fee
        else:
            fee = notional * (ENTRY_FEE + SL_FEE)
            pnl = -risk_dollar - fee

        bal += pnl
        year = day_str[:4]
        if year != last_year:
            yr_start_bal[year] = bal - pnl  # bal before this trade
            yr_peak[year] = yr_start_bal[year]
            last_year = year

        yr_pnl[year] += pnl
        if won:
            yr_wins[year] += 1
        else:
            yr_losses[year] += 1
        yr_daily[year][day_str] += pnl

        if bal > yr_peak[year]:
            yr_peak[year] = bal
        dd = (yr_peak[year] - bal) / yr_peak[year] * 100 if yr_peak[year] > 0 else 0
        if dd > yr_max_dd[year]:
            yr_max_dd[year] = dd

        open_releases.append(entry_epoch + max_hold_s)
        open_notionals.append(notional)

    # Compute per-year stats
    out = {}
    for y in YEARS:
        n = yr_wins[y] + yr_losses[y]
        if n == 0:
            out[y] = {"trades": 0, "pnl": 0, "dd": 0, "sharpe": 0, "wr": 0}
            continue
        dpnl = np.array(list(yr_daily[y].values()))
        sharpe = (dpnl.mean() / dpnl.std()) * math.sqrt(365) if dpnl.std() > 0 else 0
        out[y] = {
            "trades": n,
            "pnl": round(yr_pnl[y], 0),
            "dd": round(yr_max_dd[y], 1),
            "sharpe": round(sharpe, 2),
            "wr": round(yr_wins[y] / n * 100, 1),
        }
    return out


def sweep_config(label, tf, trades):
    cells = build_cells(trades)
    if not cells:
        return []
    cell_best_n = {k: c["best_n_fee"] for k, c in cells.items()}
    prepared = prepare_trades(trades)
    max_hold = TF_MAX_HOLD_MIN.get(tf, 240)

    out = []
    for min_ev in EV_SWEEP:
        for min_samp in SAMPLE_SWEEP:
            for setup_label, setups in SETUP_SWEEP:
                qual = {
                    k for k, c in cells.items()
                    if k[2] in setups and c["best_ev_fee"] >= min_ev and c["samples"] >= min_samp
                }
                if not qual:
                    continue
                yr_stats = replay_yearly(prepared, qual, cell_best_n, max_hold)
                # Skip if any year has 0 trades
                if any(yr_stats[y]["trades"] == 0 for y in YEARS):
                    continue
                pnls = [yr_stats[y]["pnl"] for y in YEARS]
                worst_pnl = min(pnls)
                total_pnl = sum(pnls)
                worst_dd = max(yr_stats[y]["dd"] for y in YEARS)
                min_sharpe = min(yr_stats[y]["sharpe"] for y in YEARS)
                out.append({
                    "config_label": label,
                    "tf": tf,
                    "select_label": f"ev{min_ev:.2f}_s{min_samp}_{setup_label}",
                    "cells": len(qual),
                    "worst_pnl": worst_pnl,
                    "total_pnl": total_pnl,
                    "worst_dd": worst_dd,
                    "min_sharpe": min_sharpe,
                    "yearly": yr_stats,
                })
    return out


def main():
    import time
    trades_path = "scripts/sol_sweep_results/sweep_trades.json"
    summary_path = "scripts/sol_sweep_results/sweep_summary.json"

    print(f"Loading {trades_path}...")
    with open(trades_path) as f:
        all_trades = json.load(f)
    print(f"  {len(all_trades)} configs")

    with open(summary_path) as f:
        summary = json.load(f)
    label_to_tf = {r["label"]: r["tf"] for r in summary}

    jobs = [(l, label_to_tf.get(l, "?")) for l in all_trades
            if label_to_tf.get(l, "?") in TF_MAX_HOLD_MIN and all_trades[l]]
    print(f"  {len(jobs)} jobs")

    all_results = []
    t0 = time.time()
    for done, (label, tf) in enumerate(jobs, 1):
        all_results.extend(sweep_config(label, tf, all_trades[label]))
        all_trades[label] = None
        if done % 10 == 0 or done == len(jobs):
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            print(f"  [{done}/{len(jobs)}] {len(all_results)} robust rows  el={el:.0f}s eta={eta:.0f}s", flush=True)

    # Filter to "profitable every year"
    robust = [r for r in all_results if r["worst_pnl"] > 0]
    print(f"\n{len(robust)}/{len(all_results)} configs profitable in EVERY year")

    # Rank by worst-year PnL (the floor)
    robust.sort(key=lambda r: r["worst_pnl"], reverse=True)

    print(f"\n{'='*150}")
    print("  TOP 20 — ranked by WORST-year PnL (robustness floor) @ 1x")
    print(f"{'='*150}")
    print(f"{'config':<33} {'select':<32} {'cells':>5} {'worst$':>9} {'total$':>10} {'worstDD%':>8} {'minShrp':>8}  per-year PnL")
    print("-" * 200)
    for r in robust[:20]:
        per_yr = " ".join(f"{y}:${r['yearly'][y]['pnl']:>6,.0f}/DD{r['yearly'][y]['dd']:>4.1f}" for y in YEARS)
        print(f"{r['config_label']:<33} {r['select_label']:<32} {r['cells']:>5} ${r['worst_pnl']:>7,.0f} ${r['total_pnl']:>8,.0f} {r['worst_dd']:>8.1f} {r['min_sharpe']:>8.2f}  {per_yr}")

    # Also rank by total PnL among robust ones
    print(f"\n{'='*150}")
    print("  TOP 20 — ranked by TOTAL 5Y PnL among configs profitable every year @ 1x")
    print(f"{'='*150}")
    for r in sorted(robust, key=lambda r: r["total_pnl"], reverse=True)[:20]:
        per_yr = " ".join(f"{y}:${r['yearly'][y]['pnl']:>6,.0f}/DD{r['yearly'][y]['dd']:>4.1f}" for y in YEARS)
        print(f"{r['config_label']:<33} {r['select_label']:<32} {r['cells']:>5} ${r['worst_pnl']:>7,.0f} ${r['total_pnl']:>8,.0f} {r['worst_dd']:>8.1f} {r['min_sharpe']:>8.2f}  {per_yr}")

    # Save
    out_path = "scripts/sol_sweep_results/yearly_sweep.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} rows ({len(robust)} robust) to {out_path}")


if __name__ == "__main__":
    main()
