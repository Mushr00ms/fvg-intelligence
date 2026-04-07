#!/usr/bin/env python3
"""
sweep_sol_final.py — SOL FVG strategy sweep with COIN-M fees, capital cap, leverage axis.

Loads sweep_trades.json from sweep_sol_fvg.py output. For every (TF × bps × period × mit)
config and every (min_ev × min_samples × setup × leverage) cell-selection, replays trades
chronologically with:
  - Fixed risk_pct = 1% of $80,000 starting balance ($800/trade)
  - COIN-M Regular fees: 0.02% maker entry & TP, 0.05% taker SL
  - Capital cap: open notional ≤ $80k × leverage
  - Concurrent positions allowed; each trade locks margin from entry until
    entry + max_hold_minutes (per-TF worst case from expansion window)

Reports per-TF and per-leverage winners ranked by net PnL with DD/Sharpe filters.

Usage:
    python3 scripts/sweep_sol_final.py
    python3 scripts/sweep_sol_final.py --year 2025
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# ── Constants ────────────────────────────────────────────────────────────

N_VALUES = [round(1.0 + i * 0.25, 2) for i in range(9)]
RISK_BINS = [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 60, 80, 110, 150, 200, 300, 500, 994]

# COIN-M Regular fees
ENTRY_FEE = 0.0002   # 0.02% maker
TP_FEE    = 0.0002   # 0.02% maker
SL_FEE    = 0.0005   # 0.05% taker

CAPITAL = 80_000
RISK_PCT = 0.01
LEVERAGE_SWEEP = [1.0, 1.25, 1.5, 1.75, 2.0]

# Per-TF max-hold approximation (minutes), used for capital lock duration.
# Matches walk-expansion-window from sweep_sol_fvg.py TF_CONFIG.
TF_MAX_HOLD_MIN = {
    "5min": 240,
    "15min": 240,
    "1h": 360,
}


# ── Fee math ─────────────────────────────────────────────────────────────

def _bucket(risk_bps):
    for i in range(len(RISK_BINS) - 1):
        if RISK_BINS[i] <= risk_bps < RISK_BINS[i + 1]:
            return f"{RISK_BINS[i]}-{RISK_BINS[i+1]}"
    return None


def fee_loss_ratio(risk_bps):
    return (ENTRY_FEE + SL_FEE) * 10000 / risk_bps


def fee_win_ratio(risk_bps):
    return (ENTRY_FEE + TP_FEE) * 10000 / risk_bps


def exact_group_ev_after_fees(group, n_value):
    nv_str = str(n_value)
    pnl_r = []
    for trade in group:
        rbps = trade["risk_bps"]
        if trade["outcomes"].get(nv_str) is True:
            pnl_r.append(n_value - fee_win_ratio(rbps))
        else:
            pnl_r.append(-1 - fee_loss_ratio(rbps))
    return float(np.mean(pnl_r))


# ── Cell building ────────────────────────────────────────────────────────

def build_cells(trades):
    """Group trades into (time_period, risk_bucket, setup) cells with fee-adjusted best_n."""
    groups = defaultdict(list)
    for t in trades:
        rb = _bucket(t["risk_bps"])
        if rb is None:
            continue
        groups[(t["time_period"], rb, t["setup"])].append(t)

    cells = {}
    for key, group in groups.items():
        n = len(group)
        if n < 30:
            continue
        best_ev_fee = -999
        best_n_fee = 1.0
        for nv in N_VALUES:
            ev_f = exact_group_ev_after_fees(group, nv)
            if ev_f > best_ev_fee:
                best_ev_fee = ev_f
                best_n_fee = nv
        cells[key] = {
            "samples": n,
            "best_n_fee": best_n_fee,
            "best_ev_fee": round(best_ev_fee, 4),
        }
    return cells


# ── Replay with capital cap + leverage ───────────────────────────────────

def replay(prepared, qual_keys, cell_best_n, leverage, max_hold_min):
    """Chronological replay with concurrency-aware margin gate.

    `prepared` is a list of tuples sorted by entry epoch:
        (entry_epoch, day_str, key, rbps, outcomes_dict)
    """
    cap_notional = CAPITAL * leverage
    risk_dollar = CAPITAL * RISK_PCT
    max_hold_s = max_hold_min * 60

    timeline = []
    for entry_epoch, day_str, key, rbps, outcomes in prepared:
        if key not in qual_keys:
            continue
        bn = cell_best_n[key]
        won = outcomes.get(str(bn)) is True
        timeline.append((entry_epoch, day_str, won, bn, rbps))

    open_releases = []  # release_epoch
    open_notionals = []
    bal = CAPITAL
    peak = bal
    max_dd_pct = 0
    wins = losses = 0
    total_fees = 0
    rejected_capital = 0
    daily_pnl = defaultdict(float)

    for entry_epoch, day_str, won, bn, rbps in timeline:
        # Release expired
        if open_releases:
            keep_r = []
            keep_n = []
            for r, n in zip(open_releases, open_notionals):
                if r > entry_epoch:
                    keep_r.append(r); keep_n.append(n)
            open_releases = keep_r
            open_notionals = keep_n

        notional = risk_dollar * 10000 / rbps
        if sum(open_notionals) + notional > cap_notional:
            rejected_capital += 1
            continue

        if won:
            fee = notional * (ENTRY_FEE + TP_FEE)
            pnl = risk_dollar * bn - fee
            wins += 1
        else:
            fee = notional * (ENTRY_FEE + SL_FEE)
            pnl = -risk_dollar - fee
            losses += 1

        total_fees += fee
        bal += pnl
        daily_pnl[day_str] += pnl

        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100 if peak > 0 else 0
        if dd > max_dd_pct:
            max_dd_pct = dd

        open_releases.append(entry_epoch + max_hold_s)
        open_notionals.append(notional)

    total = wins + losses
    if total == 0:
        return None
    trading_days = len(daily_pnl)
    dpnl = np.array(list(daily_pnl.values()))
    sharpe = (dpnl.mean() / dpnl.std()) * math.sqrt(365) if dpnl.std() > 0 else 0

    return {
        "trades": total,
        "wins": wins,
        "win_rate": round(wins / total * 100, 1),
        "net_pnl": round(bal - CAPITAL, 0),
        "return_pct": round((bal - CAPITAL) / CAPITAL * 100, 1),
        "total_fees": round(total_fees, 0),
        "max_dd_pct": round(max_dd_pct, 1),
        "sharpe": round(sharpe, 3),
        "trades_per_day": round(total / max(trading_days, 1), 2),
        "rejected_capital": rejected_capital,
    }


# ── Sweep over cell-selection × leverage ─────────────────────────────────

EV_SWEEP = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
SAMPLE_SWEEP = [30, 50, 100, 200]
SETUP_SWEEP = [
    ("mit_only", {"mit_extreme"}),
    ("both", {"mit_extreme", "mid_extreme"}),
]


def _parse_epoch(s):
    # 'YYYY-MM-DD HH:MM:SS+00:00' or with 'T'
    s2 = s.replace("T", " ")
    if "+" in s2[10:]:
        s2 = s2.split("+")[0]
    elif s2.endswith("Z"):
        s2 = s2[:-1]
    if "." in s2:
        s2 = s2.split(".")[0]
    return datetime.strptime(s2, "%Y-%m-%d %H:%M:%S").timestamp()


def prepare_trades(trades):
    """Pre-parse trades once. Returns sorted list of tuples."""
    out = []
    for t in trades:
        rb = _bucket(t["risk_bps"])
        if rb is None:
            continue
        ts = t.get("mitigation_time") or t["formation_time"]
        try:
            ep = _parse_epoch(ts)
        except Exception:
            try:
                ep = _parse_epoch(t["formation_time"])
            except Exception:
                continue
        day_str = ts[:10]
        key = (t["time_period"], rb, t["setup"])
        out.append((ep, day_str, key, t["risk_bps"], t["outcomes"]))
    out.sort(key=lambda x: x[0])
    return out


def sweep_one_config(label, tf, trades, year_filter=None):
    if year_filter:
        replay_trades = [t for t in trades if t["formation_time"].startswith(year_filter)]
    else:
        replay_trades = trades

    cells = build_cells(trades)  # full-sample cell selection
    if not cells:
        return []

    cell_best_n = {k: c["best_n_fee"] for k, c in cells.items()}
    prepared = prepare_trades(replay_trades)
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
                for lev in LEVERAGE_SWEEP:
                    r = replay(prepared, qual, cell_best_n, lev, max_hold)
                    if r is None:
                        continue
                    r.update({
                        "config_label": label,
                        "tf": tf,
                        "min_ev": min_ev,
                        "min_samples": min_samp,
                        "setups": setup_label,
                        "leverage": lev,
                        "cells": len(qual),
                        "select_label": f"ev{min_ev:.2f}_s{min_samp}_{setup_label}_lev{lev:.2f}",
                    })
                    out.append(r)
    return out


# Worker for multiprocessing
_G_TRADES = None
def _init_worker(trades_dict):
    global _G_TRADES
    _G_TRADES = trades_dict

def _worker(args):
    label, tf, year = args
    trades = _G_TRADES.get(label, [])
    if not trades:
        return []
    return sweep_one_config(label, tf, trades, year_filter=year)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-file",
                        default=os.path.join(_ROOT, "scripts", "sol_sweep_results", "sweep_trades.json"))
    parser.add_argument("--summary-file",
                        default=os.path.join(_ROOT, "scripts", "sol_sweep_results", "sweep_summary.json"))
    parser.add_argument("--year", default=None)
    parser.add_argument("--output", default=os.path.join(_ROOT, "scripts", "sol_sweep_results", "final_sweep.json"))
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    print(f"Loading {args.trades_file} ...")
    with open(args.trades_file) as f:
        all_trades = json.load(f)
    print(f"  {len(all_trades)} configs")

    # Need to know which TF each label belongs to
    with open(args.summary_file) as f:
        summary = json.load(f)
    label_to_tf = {r["label"]: r["tf"] for r in summary}

    # Serial execution — multiprocessing OOMs because COW on 12GB dict explodes RAM.
    jobs = [(label, label_to_tf.get(label, "?")) for label in all_trades
            if label_to_tf.get(label, "?") in TF_MAX_HOLD_MIN and all_trades[label]]
    print(f"  {len(jobs)} jobs (serial)")
    all_results = []
    import time as _time
    t0 = _time.time()
    for done, (label, tf) in enumerate(jobs, 1):
        trades = all_trades[label]
        all_results.extend(sweep_one_config(label, tf, trades, year_filter=args.year))
        # Free reference so GC can reclaim if needed
        all_trades[label] = None
        if done % 10 == 0 or done == len(jobs):
            elapsed = _time.time() - t0
            eta = elapsed / done * (len(jobs) - done)
            print(f"  [{done}/{len(jobs)}] {len(all_results)} rows  "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"\nTotal result rows: {len(all_results)}")

    # ── Reports ─────────────────────────────────────────────────────────
    def fmt(r):
        return (f"  {r['config_label']:<32} {r['tf']:>5} {r['select_label']:<35} "
                f"cells={r['cells']:>3} trades={r['trades']:>5} "
                f"WR={r['win_rate']:>5.1f}% PnL=${r['net_pnl']:>10,.0f} "
                f"({r['return_pct']:>6.1f}%) DD={r['max_dd_pct']:>5.1f}% "
                f"Sharpe={r['sharpe']:>6.2f} T/D={r['trades_per_day']:>5.2f}")

    print(f"\n{'='*140}")
    print(f"  TOP {args.top} BY NET PnL (all configs, all leverage)")
    print(f"{'='*140}")
    for r in sorted(all_results, key=lambda r: r["net_pnl"], reverse=True)[:args.top]:
        print(fmt(r))

    # Per-TF best
    for tf in ["5min", "15min", "1h"]:
        tf_results = [r for r in all_results if r["tf"] == tf]
        if not tf_results:
            continue
        print(f"\n{'='*140}")
        print(f"  TOP 10 — {tf}")
        print(f"{'='*140}")
        for r in sorted(tf_results, key=lambda r: r["net_pnl"], reverse=True)[:10]:
            print(fmt(r))

    # Per-leverage best
    for lev in LEVERAGE_SWEEP:
        lev_results = [r for r in all_results if r["leverage"] == lev]
        if not lev_results:
            continue
        print(f"\n{'='*140}")
        print(f"  TOP 5 — leverage {lev}x")
        print(f"{'='*140}")
        for r in sorted(lev_results, key=lambda r: r["net_pnl"], reverse=True)[:5]:
            print(fmt(r))

    # Sweet spot per DD
    print(f"\n{'='*140}")
    print(f"  SWEET SPOT — best PnL within max DD threshold")
    print(f"{'='*140}")
    for target_dd in [5, 8, 10, 15, 20, 30]:
        candidates = [r for r in all_results if r["max_dd_pct"] <= target_dd and r["net_pnl"] > 0]
        if candidates:
            best = max(candidates, key=lambda r: r["net_pnl"])
            print(f"  DD <= {target_dd:>2}%: ", fmt(best).strip())

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} rows to {args.output}")


if __name__ == "__main__":
    main()
