#!/usr/bin/env python3
"""Full 5Y compounded replay at 0.7% risk for locked SOL config."""
import json, math, sys, os
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sweep_sol_final import (
    build_cells, prepare_trades, TF_MAX_HOLD_MIN,
    CAPITAL, ENTRY_FEE, TP_FEE, SL_FEE,
)

LEVERAGE = 1.0
RISK_PCT = 0.007
YEARS = ["2021", "2022", "2023", "2024", "2025"]

CONFIGS = [
    ("1h_bps5_p60m_mit36", "ev0.03_s30_both",     "LOCKED"),
    ("1h_bps5_p60m_mit36", "ev0.07_s30_mit_only", "BestPnL+robust"),
    ("1h_bps10_p60m_mit36","ev0.20_s30_both",     "Robust #3"),
]


def parse_select(s):
    parts = s.split("_")
    ev = float(parts[0][2:]); samp = int(parts[1][1:])
    setup_label = "_".join(parts[2:])
    setups = {"mit_extreme"} if setup_label == "mit_only" else {"mit_extreme", "mid_extreme"}
    return ev, samp, setups


def replay_compound(prepared, qual_keys, cell_best_n, max_hold_min):
    max_hold_s = max_hold_min * 60
    open_releases, open_notionals = [], []
    bal = CAPITAL
    yr_start = {}
    yr_pnl = defaultdict(float)
    yr_wins = defaultdict(int)
    yr_losses = defaultdict(int)
    yr_daily = defaultdict(lambda: defaultdict(float))
    yr_peak = {}
    yr_max_dd = defaultdict(float)
    yr_rejected = defaultdict(int)
    overall_peak = bal
    overall_max_dd = 0
    last_year = None

    for entry_epoch, day_str, key, rbps, outcomes in prepared:
        year = day_str[:4]
        if key not in qual_keys:
            continue
        bn = cell_best_n[key]
        won = outcomes.get(str(bn)) is True

        if open_releases:
            kr, kn = [], []
            for r, n in zip(open_releases, open_notionals):
                if r > entry_epoch:
                    kr.append(r); kn.append(n)
            open_releases, open_notionals = kr, kn

        cap_notional = bal * LEVERAGE
        risk_dollar = bal * RISK_PCT
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
            yr_start[year] = bal
            yr_peak[year] = bal
            last_year = year

        bal += pnl
        yr_pnl[year] += pnl
        if won: yr_wins[year] += 1
        else: yr_losses[year] += 1
        yr_daily[year][day_str] += pnl

        if bal > yr_peak[year]:
            yr_peak[year] = bal
        ydd = (yr_peak[year] - bal) / yr_peak[year] * 100
        if ydd > yr_max_dd[year]:
            yr_max_dd[year] = ydd

        if bal > overall_peak:
            overall_peak = bal
        odd = (overall_peak - bal) / overall_peak * 100
        if odd > overall_max_dd:
            overall_max_dd = odd

        open_releases.append(entry_epoch + max_hold_s)
        open_notionals.append(notional)

    return {
        "final_bal": bal,
        "max_dd_5y": overall_max_dd,
        "yr_start": dict(yr_start),
        "yr_pnl": dict(yr_pnl),
        "yr_wins": dict(yr_wins),
        "yr_losses": dict(yr_losses),
        "yr_dd": dict(yr_max_dd),
        "yr_daily": {k: dict(v) for k, v in yr_daily.items()},
        "yr_rejected": dict(yr_rejected),
    }


def main():
    print("Loading sweep_trades.json...")
    with open("scripts/sol_sweep_results/sweep_trades.json") as f:
        all_trades = json.load(f)

    print(f"\nCOMPOUNDED 5Y @ 1x leverage, 0.7% of current equity per trade")
    print(f"Starting balance: ${CAPITAL:,}\n")

    for cfg, sel, name in CONFIGS:
        if cfg not in all_trades:
            continue
        trades = all_trades[cfg]
        cells = build_cells(trades)
        cell_best_n = {k: c["best_n_fee"] for k, c in cells.items()}
        prepared = prepare_trades(trades)
        ev, samp, setups = parse_select(sel)
        qual = {k for k, c in cells.items()
                if k[2] in setups and c["best_ev_fee"] >= ev and c["samples"] >= samp}
        max_hold = TF_MAX_HOLD_MIN["1h"]

        r = replay_compound(prepared, qual, cell_best_n, max_hold)

        print(f"{'='*110}")
        print(f"  {name}: {cfg} | {sel}  ({len(qual)} cells)")
        print(f"{'='*110}")
        print(f"  {'year':>5}  {'start bal':>11}  {'end bal':>11}  {'PnL':>11}  {'ret%':>7}  {'DD%':>6}  {'Sharpe':>7}  {'WR%':>5}  {'trades':>6}  {'rej':>5}")
        print(f"  {'-'*100}")
        for y in YEARS:
            if y not in r["yr_start"]:
                continue
            sb = r["yr_start"][y]
            pnl = r["yr_pnl"][y]
            eb = sb + pnl
            n = r["yr_wins"][y] + r["yr_losses"][y]
            wr = r["yr_wins"][y] / n * 100 if n else 0
            dpnl = np.array(list(r["yr_daily"][y].values()))
            shp = (dpnl.mean() / dpnl.std()) * math.sqrt(365) if dpnl.std() > 0 else 0
            print(f"  {y:>5}  ${sb:>9,.0f}  ${eb:>9,.0f}  ${pnl:>9,.0f}  {pnl/sb*100:>+6.1f}%  {r['yr_dd'][y]:>5.1f}%  {shp:>7.2f}  {wr:>5.1f}  {n:>6}  {r['yr_rejected'][y]:>5}")
        tot_ret = (r['final_bal'] - CAPITAL) / CAPITAL * 100
        print(f"  {'-'*100}")
        print(f"  TOTAL: final=${r['final_bal']:,.0f}  return={tot_ret:+.1f}%  max5Y_DD={r['max_dd_5y']:.1f}%")
        print()


if __name__ == "__main__":
    main()
