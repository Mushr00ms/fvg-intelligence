#!/usr/bin/env python3
"""BTC at $50k starting capital, 1x leverage, capital-cap aware.

Sweep risk_pct to find optimal under capital constraint.
Both fixed-risk and compounded modes. Reports monthly stats for steady-income view.

BTC fee model: USDC Regular (0% maker / 0.04% taker)
Starting balance: $50,000
Leverage: 1x (max notional = current equity)
"""
import json, math, sys, os
from collections import defaultdict
from datetime import datetime
import numpy as np

CAPITAL = 50_000
LEVERAGE = 1.0
ENTRY_FEE = 0.0       # USDC maker
TP_FEE = 0.0          # USDC maker
SL_FEE = 0.0004       # USDC taker

# 5min FVG → max hold approximation (walk window 240 min)
MAX_HOLD_MIN = 240
RISK_BINS = [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 994]
N_VALUES = [round(1.0 + i * 0.25, 2) for i in range(9)]

LOCKED_CFG = "5min_bps5_p60m_mit90"
SETUPS_BOTH = {"mit_extreme", "mid_extreme"}

# ── helpers ──────────────────────────────────────────────────────────────

def _bucket(rbps):
    for i in range(len(RISK_BINS) - 1):
        if RISK_BINS[i] <= rbps < RISK_BINS[i + 1]:
            return f"{RISK_BINS[i]}-{RISK_BINS[i+1]}"
    return None


def fee_loss_ratio(rbps): return (ENTRY_FEE + SL_FEE) * 10000 / rbps
def fee_win_ratio(rbps): return (ENTRY_FEE + TP_FEE) * 10000 / rbps


def exact_group_ev_after_fees(group, n):
    nv = str(n)
    pnl_r = []
    for t in group:
        rbps = t["risk_bps"]
        if t["outcomes"].get(nv) is True:
            pnl_r.append(n - fee_win_ratio(rbps))
        else:
            pnl_r.append(-1 - fee_loss_ratio(rbps))
    return float(np.mean(pnl_r))


def build_cells(trades):
    groups = defaultdict(list)
    for t in trades:
        rb = _bucket(t["risk_bps"])
        if rb is None: continue
        groups[(t["time_period"], rb, t["setup"])].append(t)
    cells = {}
    for k, g in groups.items():
        n = len(g)
        if n < 30: continue
        best_ev, best_n = -999, 1.0
        for nv in N_VALUES:
            ev = exact_group_ev_after_fees(g, nv)
            if ev > best_ev:
                best_ev, best_n = ev, nv
        cells[k] = {"samples": n, "best_n_fee": best_n, "best_ev_fee": round(best_ev, 4)}
    return cells


def _parse_epoch(s):
    s2 = s.replace("T", " ")
    if "+" in s2[10:]: s2 = s2.split("+")[0]
    elif s2.endswith("Z"): s2 = s2[:-1]
    if "." in s2: s2 = s2.split(".")[0]
    return datetime.strptime(s2, "%Y-%m-%d %H:%M:%S").timestamp()


def prepare(trades):
    out = []
    for t in trades:
        rb = _bucket(t["risk_bps"])
        if rb is None: continue
        ts = t.get("mitigation_time") or t["formation_time"]
        try: ep = _parse_epoch(ts)
        except: continue
        out.append((ep, ts[:10], (t["time_period"], rb, t["setup"]), t["risk_bps"], t["outcomes"]))
    out.sort(key=lambda x: x[0])
    return out


def replay(prepared, qual_keys, cell_best_n, risk_pct, compound):
    """Returns dict with stats. compound=True scales risk to current bal."""
    max_hold_s = MAX_HOLD_MIN * 60
    open_releases, open_notionals = [], []
    bal = CAPITAL
    peak = bal
    max_dd = 0
    wins = losses = 0
    rejected = 0
    fees = 0
    monthly_pnl = defaultdict(float)
    daily_pnl = defaultdict(float)
    yr_pnl = defaultdict(float)
    yr_dd = defaultdict(float)
    yr_peak = {}
    yr_wins = defaultdict(int)
    yr_losses = defaultdict(int)
    last_yr = None

    for entry_epoch, day_str, key, rbps, outcomes in prepared:
        if key not in qual_keys:
            continue
        bn = cell_best_n[key]
        won = outcomes.get(str(bn)) is True
        year = day_str[:4]

        if open_releases:
            kr, kn = [], []
            for r, n in zip(open_releases, open_notionals):
                if r > entry_epoch:
                    kr.append(r); kn.append(n)
            open_releases, open_notionals = kr, kn

        if compound:
            cap_notional = bal * LEVERAGE
            risk_dollar = bal * risk_pct
        else:
            cap_notional = CAPITAL * LEVERAGE
            risk_dollar = CAPITAL * risk_pct

        notional = risk_dollar * 10000 / rbps

        if sum(open_notionals) + notional > cap_notional:
            rejected += 1
            continue

        if won:
            fee = notional * (ENTRY_FEE + TP_FEE)
            pnl = risk_dollar * bn - fee
            wins += 1; yr_wins[year] += 1
        else:
            fee = notional * (ENTRY_FEE + SL_FEE)
            pnl = -risk_dollar - fee
            losses += 1; yr_losses[year] += 1

        fees += fee
        bal += pnl
        monthly_pnl[day_str[:7]] += pnl
        daily_pnl[day_str] += pnl
        yr_pnl[year] += pnl

        if year != last_yr:
            yr_peak[year] = bal - pnl
            last_yr = year
        if bal > yr_peak[year]:
            yr_peak[year] = bal
        ydd = (yr_peak[year] - bal) / yr_peak[year] * 100
        if ydd > yr_dd[year]:
            yr_dd[year] = ydd

        if bal > peak:
            peak = bal
        dd = (peak - bal) / peak * 100
        if dd > max_dd:
            max_dd = dd

        open_releases.append(entry_epoch + max_hold_s)
        open_notionals.append(notional)

    n = wins + losses
    months = sorted(monthly_pnl.keys())
    monthly_arr = np.array([monthly_pnl[m] for m in months])
    pos_months = sum(1 for v in monthly_arr if v > 0)
    daily_arr = np.array(list(daily_pnl.values())) if daily_pnl else np.array([0])
    sharpe = (daily_arr.mean() / daily_arr.std()) * math.sqrt(365) if daily_arr.std() > 0 else 0

    return {
        "final_bal": bal,
        "net_pnl": bal - CAPITAL,
        "ret_pct": (bal - CAPITAL) / CAPITAL * 100,
        "max_dd_pct": max_dd,
        "trades": n,
        "wr": wins / n * 100 if n else 0,
        "rejected": rejected,
        "fees": fees,
        "sharpe": sharpe,
        "months": len(months),
        "pos_months": pos_months,
        "pos_pct": pos_months / len(months) * 100 if months else 0,
        "monthly_mean": float(monthly_arr.mean()) if len(monthly_arr) else 0,
        "monthly_std": float(monthly_arr.std()) if len(monthly_arr) else 0,
        "monthly_min": float(monthly_arr.min()) if len(monthly_arr) else 0,
        "monthly_max": float(monthly_arr.max()) if len(monthly_arr) else 0,
        "yearly_pnl": dict(yr_pnl),
        "yearly_dd": dict(yr_dd),
        "monthly_pnl": dict(monthly_pnl),
    }


def main():
    print("Loading BTC sweep_trades.json...")
    with open("scripts/btc_sweep_results/5min_results/sweep_trades.json") as f:
        all_trades = json.load(f)
    trades = all_trades[LOCKED_CFG]
    print(f"  {len(trades):,} trades for {LOCKED_CFG}")

    print("Building fee-aware cells...")
    cells = build_cells(trades)
    cell_best_n = {k: c["best_n_fee"] for k, c in cells.items()}
    qual = {k for k, c in cells.items()
            if k[2] in SETUPS_BOTH and c["best_ev_fee"] >= 0.07 and c["samples"] >= 30}
    print(f"  {len(cells)} total cells, {len(qual)} qualifying (ev0.07_s30_both)")

    prepared = prepare(trades)
    print(f"  {len(prepared):,} trades prepared")

    print(f"\n{'='*120}")
    print(f"  BTC | $50,000 start | 1x leverage | USDC fees | locked cells (ev0.07_s30_both)")
    print(f"  Max notional cap = current equity (compounded) or starting capital (fixed)")
    print(f"{'='*120}")

    risk_levels = [0.0002, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.0075, 0.01]

    for mode_label, comp in [("COMPOUNDED", True)]:
        print(f"\n  --- {mode_label} ---")
        print(f"  {'risk%':>6}  {'final$':>12}  {'ret%':>10}  {'DD%':>6}  {'trades':>6}  {'WR%':>5}  {'rej':>5}  "
              f"{'mo+%':>5}  {'mo mean':>9}  {'mo std':>8}  {'Sharpe':>7}")
        print("  " + "-"*116)
        for rp in risk_levels:
            r = replay(prepared, qual, cell_best_n, rp, comp)
            print(f"  {rp*100:>5.2f}%  ${r['final_bal']:>10,.0f}  {r['ret_pct']:>+9.1f}%  "
                  f"{r['max_dd_pct']:>5.1f}%  {r['trades']:>6}  {r['wr']:>5.1f}  {r['rejected']:>5}  "
                  f"{r['pos_pct']:>4.0f}%  ${r['monthly_mean']:>7,.0f}  ${r['monthly_std']:>6,.0f}  {r['sharpe']:>7.2f}")

    # Highlight best fixed-risk and best compounded
    print(f"\n{'='*120}")
    print("  YEARLY BREAKDOWN — best fixed @ 0.7% and best compounded @ 0.7% ")
    print(f"{'='*120}")
    for mode_label, comp in [("FIXED 0.7%", False), ("COMPOUNDED 0.7%", True)]:
        r = replay(prepared, qual, cell_best_n, 0.007, comp)
        print(f"\n  {mode_label} → final ${r['final_bal']:,.0f}  return {r['ret_pct']:+.1f}%  DD {r['max_dd_pct']:.1f}%")
        for y in sorted(r['yearly_pnl']):
            print(f"    {y}: ${r['yearly_pnl'][y]:>+10,.0f}  DD {r['yearly_dd'][y]:>5.1f}%")
        print(f"    Monthly: mean ${r['monthly_mean']:,.0f}  std ${r['monthly_std']:,.0f}  "
              f"min ${r['monthly_min']:,.0f}  max ${r['monthly_max']:,.0f}  positive {r['pos_pct']:.0f}% of {r['months']} months")


if __name__ == "__main__":
    main()
