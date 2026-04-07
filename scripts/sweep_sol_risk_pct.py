#!/usr/bin/env python3
"""Sweep risk_pct for a given SOL config @ 1x leverage.

Tests how PnL/DD scale as we bump risk per trade from 1% to 4%.
Accounts for capital-cap rejection (higher risk → larger notional → more rejects).
"""
import json
import math
import os
import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sweep_sol_final import (
    build_cells, prepare_trades, TF_MAX_HOLD_MIN,
    CAPITAL, ENTRY_FEE, TP_FEE, SL_FEE,
)

LEVERAGE = 1.0
YEARS = ["2021", "2022", "2023", "2024", "2025"]
RISK_PCTS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.0125, 0.015, 0.02]


def replay(prepared, qual_keys, cell_best_n, max_hold_min, risk_pct):
    cap_notional = CAPITAL * LEVERAGE
    risk_dollar = CAPITAL * risk_pct  # fixed risk on starting capital
    max_hold_s = max_hold_min * 60

    open_releases, open_notionals = [], []
    bal = CAPITAL
    yr_pnl = defaultdict(float)
    yr_wins = defaultdict(int)
    yr_losses = defaultdict(int)
    yr_daily = defaultdict(lambda: defaultdict(float))
    yr_peak = {}
    yr_max_dd = defaultdict(float)
    yr_rejected = defaultdict(int)
    last_year = None

    overall_peak = bal
    overall_max_dd = 0

    for entry_epoch, day_str, key, rbps, outcomes in prepared:
        year = day_str[:4]
        if year not in YEARS:
            continue
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
            yr_rejected[year] += 1
            continue

        if won:
            fee = notional * (ENTRY_FEE + TP_FEE)
            pnl = risk_dollar * bn - fee
        else:
            fee = notional * (ENTRY_FEE + SL_FEE)
            pnl = -risk_dollar - fee

        if year != last_year:
            yr_peak[year] = bal
            last_year = year

        bal += pnl
        yr_pnl[year] += pnl
        if won: yr_wins[year] += 1
        else: yr_losses[year] += 1
        yr_daily[year][day_str] += pnl

        if bal > yr_peak[year]:
            yr_peak[year] = bal
        ydd = (yr_peak[year] - bal) / yr_peak[year] * 100 if yr_peak[year] > 0 else 0
        if ydd > yr_max_dd[year]:
            yr_max_dd[year] = ydd

        if bal > overall_peak:
            overall_peak = bal
        odd = (overall_peak - bal) / overall_peak * 100 if overall_peak > 0 else 0
        if odd > overall_max_dd:
            overall_max_dd = odd

        open_releases.append(entry_epoch + max_hold_s)
        open_notionals.append(notional)

    out = {"yearly": {}, "total_pnl": bal - CAPITAL, "max_dd_5y": round(overall_max_dd, 1)}
    for y in YEARS:
        n = yr_wins[y] + yr_losses[y]
        if n == 0:
            out["yearly"][y] = None
            continue
        dpnl = np.array(list(yr_daily[y].values()))
        sharpe = (dpnl.mean() / dpnl.std()) * math.sqrt(365) if dpnl.std() > 0 else 0
        out["yearly"][y] = {
            "pnl": round(yr_pnl[y], 0),
            "dd": round(yr_max_dd[y], 1),
            "sharpe": round(sharpe, 2),
            "wr": round(yr_wins[y]/n*100, 1),
            "trades": n,
            "rejected": yr_rejected[y],
        }
    return out


CONFIGS = [
    ("1h_bps5_p60m_mit36",   "ev0.07_s30_mit_only", "Best PnL+robust"),
    ("1h_bps5_p60m_mit36",   "ev0.03_s30_both",     "LOCKED original"),
    ("1h_bps10_p60m_mit36",  "ev0.20_s30_both",     "Robust #3 (low DD)"),
]


def parse_select(s):
    parts = s.split("_")
    ev = float(parts[0][2:])
    samp = int(parts[1][1:])
    setup_label = "_".join(parts[2:])
    setups = {"mit_extreme"} if setup_label == "mit_only" else {"mit_extreme", "mid_extreme"}
    return ev, samp, setups


def main():
    print("Loading sweep_trades.json...")
    with open("scripts/sol_sweep_results/sweep_trades.json") as f:
        all_trades = json.load(f)

    for cfg_label, sel, name in CONFIGS:
        if cfg_label not in all_trades:
            continue
        trades = all_trades[cfg_label]
        cells = build_cells(trades)
        cell_best_n = {k: c["best_n_fee"] for k, c in cells.items()}
        prepared = prepare_trades(trades)
        ev, samp, setups = parse_select(sel)
        qual = {k for k, c in cells.items()
                if k[2] in setups and c["best_ev_fee"] >= ev and c["samples"] >= samp}
        max_hold = TF_MAX_HOLD_MIN["1h"]

        print(f"\n{'='*120}")
        print(f"  {name}: {cfg_label} | {sel}  ({len(qual)} cells)")
        print(f"{'='*120}")
        print(f"  {'risk%':>6} {'5Y PnL':>11} {'5Y ret%':>8} {'5Y DD%':>7} {'worst yr$':>10} {'worst yr DD%':>13} {'tot reject':>11}")
        print(f"  {'-'*78}")
        for rp in RISK_PCTS:
            r = replay(prepared, qual, cell_best_n, max_hold, rp)
            yrly = [r["yearly"][y] for y in YEARS if r["yearly"][y]]
            if not yrly:
                print(f"  {rp*100:>5.2f}% — no trades"); continue
            worst_pnl = min(y["pnl"] for y in yrly)
            worst_dd = max(y["dd"] for y in yrly)
            tot_rej = sum(y["rejected"] for y in yrly)
            print(f"  {rp*100:>5.2f}% ${r['total_pnl']:>9,.0f} {r['total_pnl']/CAPITAL*100:>7.1f}% "
                  f"{r['max_dd_5y']:>6.1f}% ${worst_pnl:>8,.0f} {worst_dd:>12.1f}% {tot_rej:>11}")


if __name__ == "__main__":
    main()
