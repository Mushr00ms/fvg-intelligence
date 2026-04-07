#!/usr/bin/env python3
"""SOL @ 1x with COMPOUNDING (1% of current balance) — 2024 + 2025 only.

Cells still selected from the full 5Y data (out-of-sample for 2024-25).
Risk per trade = 1% of *current* equity, not starting balance.
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
    EV_SWEEP, SAMPLE_SWEEP, SETUP_SWEEP,
    CAPITAL, RISK_PCT, ENTRY_FEE, TP_FEE, SL_FEE,
)

LEVERAGE = 1.0
YEARS = ["2024", "2025"]


def replay_compound(prepared, qual_keys, cell_best_n, max_hold_min):
    cap_notional_factor = LEVERAGE  # cap = bal * factor (compounds with equity)
    max_hold_s = max_hold_min * 60

    open_releases, open_notionals = [], []
    bal = CAPITAL
    yr_start_bal = {}
    yr_pnl = defaultdict(float)
    yr_wins = defaultdict(int)
    yr_losses = defaultdict(int)
    yr_daily = defaultdict(lambda: defaultdict(float))
    yr_peak = {}
    yr_max_dd = defaultdict(float)
    rejected = defaultdict(int)
    last_year = None

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

        cap_notional = bal * cap_notional_factor   # compounds with equity
        risk_dollar = bal * RISK_PCT                # compounds with equity
        notional = risk_dollar * 10000 / rbps

        if sum(open_notionals) + notional > cap_notional:
            rejected[year] += 1
            continue

        if won:
            fee = notional * (ENTRY_FEE + TP_FEE)
            pnl = risk_dollar * bn - fee
        else:
            fee = notional * (ENTRY_FEE + SL_FEE)
            pnl = -risk_dollar - fee

        if year != last_year:
            yr_start_bal[year] = bal
            yr_peak[year] = bal
            last_year = year

        bal += pnl
        yr_pnl[year] += pnl
        if won: yr_wins[year] += 1
        else: yr_losses[year] += 1
        yr_daily[year][day_str] += pnl

        if bal > yr_peak[year]:
            yr_peak[year] = bal
        dd = (yr_peak[year] - bal) / yr_peak[year] * 100 if yr_peak[year] > 0 else 0
        if dd > yr_max_dd[year]:
            yr_max_dd[year] = dd

        open_releases.append(entry_epoch + max_hold_s)
        open_notionals.append(notional)

    out = {}
    for y in YEARS:
        n = yr_wins[y] + yr_losses[y]
        if n == 0:
            out[y] = None
            continue
        dpnl = np.array(list(yr_daily[y].values()))
        sharpe = (dpnl.mean() / dpnl.std()) * math.sqrt(365) if dpnl.std() > 0 else 0
        ret_pct = yr_pnl[y] / yr_start_bal[y] * 100
        out[y] = {
            "start_bal": round(yr_start_bal[y], 0),
            "pnl": round(yr_pnl[y], 0),
            "ret_pct": round(ret_pct, 1),
            "dd": round(yr_max_dd[y], 1),
            "sharpe": round(sharpe, 2),
            "wr": round(yr_wins[y]/n*100, 1),
            "trades": n,
            "rejected": rejected[y],
        }
    out["final_bal"] = round(bal, 0)
    out["total_ret_pct"] = round((bal - CAPITAL) / CAPITAL * 100, 1)
    return out


# Configs to test (the locked + the robust winners)
CONFIGS_TO_TEST = [
    ("1h_bps5_p60m_mit36",   "ev0.03_s30_both",    "LOCKED original"),
    ("1h_bps30_p120m_mit36", "ev0.05_s30_mit_only", "Robust #1"),
    ("1h_bps5_p240m_mit12",  "ev0.15_s30_both",    "Robust #2"),
    ("1h_bps10_p60m_mit36",  "ev0.20_s30_both",    "Robust #3 (lowest DD)"),
    ("1h_bps5_p60m_mit18",   "ev0.20_s30_both",    "Robust #4"),
]


def parse_select(s):
    # ev0.05_s30_mit_only or ev0.20_s30_both
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
    print(f"  {len(all_trades)} configs loaded")

    print(f"\n{'='*120}")
    print(f"  COMPOUNDED 1x — 2024 + 2025 (cells from full 5Y, out-of-sample)")
    print(f"  Risk per trade = 1% of CURRENT equity. Capital cap = 1.0x current equity.")
    print(f"{'='*120}\n")

    for cfg_label, sel, name in CONFIGS_TO_TEST:
        if cfg_label not in all_trades:
            print(f"  SKIP {cfg_label}: not in trades")
            continue
        trades = all_trades[cfg_label]
        cells = build_cells(trades)
        cell_best_n = {k: c["best_n_fee"] for k, c in cells.items()}
        prepared = prepare_trades(trades)

        ev, samp, setups = parse_select(sel)
        qual = {
            k for k, c in cells.items()
            if k[2] in setups and c["best_ev_fee"] >= ev and c["samples"] >= samp
        }
        tf = "1h" if cfg_label.startswith("1h") else ("15min" if cfg_label.startswith("15min") else "5min")
        max_hold = TF_MAX_HOLD_MIN[tf]

        r = replay_compound(prepared, qual, cell_best_n, max_hold)

        print(f"\n  {name}: {cfg_label} | {sel}  ({len(qual)} cells)")
        print(f"  {'-'*100}")
        for y in YEARS:
            yr = r[y]
            if yr is None:
                print(f"    {y}: no trades")
                continue
            print(f"    {y}:  start=${yr['start_bal']:>9,.0f}  PnL=${yr['pnl']:>9,.0f} ({yr['ret_pct']:>+6.1f}%)  "
                  f"DD={yr['dd']:>5.1f}%  Sharpe={yr['sharpe']:>5.2f}  WR={yr['wr']:>5.1f}%  "
                  f"trades={yr['trades']:>4}  rejected={yr['rejected']}")
        print(f"    Final balance: ${r['final_bal']:,.0f}  →  total return {r['total_ret_pct']:+.1f}%")


if __name__ == "__main__":
    main()
