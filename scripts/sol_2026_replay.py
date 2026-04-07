#!/usr/bin/env python3
"""Replay locked SOL config on 2026 YTD data.

Cells built from full 5Y (2021-2025) sweep_trades.
2026 trades from sweep_sol_fvg.py output (--start 20251201).
Compounded 0.7% risk, 1x leverage, fee-aware.
"""
import json, math, sys, os
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from sweep_sol_final import (
    build_cells, prepare_trades, TF_MAX_HOLD_MIN,
    CAPITAL, ENTRY_FEE, TP_FEE, SL_FEE,
)

CFG_LABEL = "1h_bps5_p60m_mit36"
SELECT = "ev0.03_s30_both"
RISK_PCT = 0.007
LEVERAGE = 1.0


def parse_select(s):
    parts = s.split("_")
    ev = float(parts[0][2:]); samp = int(parts[1][1:])
    setup_label = "_".join(parts[2:])
    setups = {"mit_extreme"} if setup_label == "mit_only" else {"mit_extreme", "mid_extreme"}
    return ev, samp, setups


def main():
    print("Loading FULL 5Y trades for cell selection...")
    with open("scripts/sol_sweep_results/sweep_trades.json") as f:
        all5y = json.load(f)
    trades_5y = all5y[CFG_LABEL]
    print(f"  {len(trades_5y):,} trades 2021-2025")

    print("Building cells from 5Y data (fee-aware)...")
    cells = build_cells(trades_5y)
    cell_best_n = {k: c["best_n_fee"] for k, c in cells.items()}
    ev, samp, setups = parse_select(SELECT)
    qual = {k for k, c in cells.items()
            if k[2] in setups and c["best_ev_fee"] >= ev and c["samples"] >= samp}
    print(f"  {len(cells)} total cells, {len(qual)} qualifying ({SELECT})")

    print("\nLoading 2026 YTD trades...")
    with open("scripts/sol_sweep_results/2026/sweep_trades.json") as f:
        trades_2026 = json.load(f)[CFG_LABEL]
    # Filter to 2026 only
    trades_2026 = [t for t in trades_2026 if t["formation_time"].startswith("2026")]
    print(f"  {len(trades_2026)} trades in 2026 YTD")

    prepared = prepare_trades(trades_2026)
    max_hold_s = TF_MAX_HOLD_MIN["1h"] * 60

    # Compounded replay starting from $80k
    open_releases, open_notionals = [], []
    bal = CAPITAL
    peak = bal
    max_dd = 0
    wins = losses = 0
    monthly_pnl = defaultdict(float)
    daily_pnl = defaultdict(float)
    rejected = 0
    qualified = 0
    fees_paid = 0

    for entry_epoch, day_str, key, rbps, outcomes in prepared:
        if key not in qual:
            continue
        qualified += 1
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
            rejected += 1
            continue

        if won:
            fee = notional * (ENTRY_FEE + TP_FEE)
            pnl = risk_dollar * bn - fee
            wins += 1
        else:
            fee = notional * (ENTRY_FEE + SL_FEE)
            pnl = -risk_dollar - fee
            losses += 1

        fees_paid += fee
        bal += pnl
        monthly_pnl[day_str[:7]] += pnl
        daily_pnl[day_str] += pnl

        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd:
            max_dd = dd

        open_releases.append(entry_epoch + max_hold_s)
        open_notionals.append(notional)

    n = wins + losses
    print(f"\n{'='*80}")
    print(f"  SOL 2026 YTD — LOCKED config compounded from $80,000")
    print(f"  {CFG_LABEL} | {SELECT} | 0.7% risk | 1x leverage | COIN-M fees")
    print(f"{'='*80}")
    print(f"  Final balance:  ${bal:,.0f}")
    print(f"  Net PnL:        ${bal - CAPITAL:+,.0f}")
    print(f"  Return:         {(bal - CAPITAL) / CAPITAL * 100:+.1f}%")
    print(f"  Max DD:         {max_dd:.1f}%")
    print(f"  Trades:         {n}  ({wins}W / {losses}L)")
    print(f"  Win rate:       {wins/n*100 if n else 0:.1f}%")
    print(f"  Qualified hits: {qualified}  (rejected by capital cap: {rejected})")
    print(f"  Fees paid:      ${fees_paid:,.0f}")

    if daily_pnl:
        dp = np.array(list(daily_pnl.values()))
        sharpe = (dp.mean() / dp.std()) * math.sqrt(365) if dp.std() > 0 else 0
        print(f"  Sharpe:         {sharpe:.2f}")

    print(f"\n  Monthly PnL:")
    for m in sorted(monthly_pnl):
        print(f"    {m}: ${monthly_pnl[m]:+,.0f}")


if __name__ == "__main__":
    main()
