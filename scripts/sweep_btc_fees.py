#!/usr/bin/env python3
"""
sweep_btc_fees.py — Re-sweep BTC FVG strategy with USDC fees baked in.

USDC Regular: maker=0%, taker=0.04%
Entry = limit (maker = free), TP = limit (maker = free), SL = stop-market (taker = 0.04%)

Fee per trade:
  WIN:  fee = notional * (0 + 0)     = 0  (both sides maker)
  LOSS: fee = notional * (0 + 0.04%) = notional * 0.0004

  notional = risk_dollar / (risk_bps / 10000) = risk_dollar * 10000 / risk_bps

  So loss_fee_as_fraction_of_risk = 0.0004 * 10000 / risk_bps = 4 / risk_bps

  At 10 bps risk: loss costs 1.0R + 0.4R in fees = 1.4R total loss
  At 20 bps risk: loss costs 1.0R + 0.2R = 1.2R total loss
  At 50 bps risk: loss costs 1.0R + 0.08R = 1.08R total loss

EV formula after fees:
  EV = mean(per-trade R after fees)
     = mean(n_value - win_fee_ratio(risk_bps_i)) on wins
       and (-1 - loss_fee_ratio(risk_bps_i)) on losses

This uses each trade's own risk_bps so cell selection matches replay math.

This script recomputes best_n per cell with fees, finds surviving cells,
and runs the strategy sweep with fee-adjusted P&L.
"""

import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_STORE_PATH = os.path.join(_ROOT, "logic", "utils", "strategy_store.py")
_store_spec = importlib.util.spec_from_file_location("strategy_store", _STORE_PATH)
_strategy_store = importlib.util.module_from_spec(_store_spec)
_store_spec.loader.exec_module(_strategy_store)

save_strategy = _strategy_store.save_strategy
set_active_strategy = _strategy_store.set_active_strategy

N_VALUES = [round(1.0 + i * 0.25, 2) for i in range(9)]
RISK_BINS = [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 994]

# USDC Regular fees — 0% maker rebate, 0.04% taker
ENTRY_FEE = 0.0        # maker limit = 0%
TP_FEE = 0.0           # maker limit = 0%  (limit TP order)
SL_FEE = 0.0004        # taker stop = 0.04% (stop-market SL)


def fee_loss_ratio(risk_bps):
    """Extra cost of a loss as fraction of risk, from SL taker fee.
    Entry is free (maker), SL is taker on full notional.
    notional = risk_dollar * 10000 / risk_bps
    fee = notional * SL_FEE = risk_dollar * 10000 * SL_FEE / risk_bps
    fee / risk_dollar = 10000 * SL_FEE / risk_bps = 4 / risk_bps
    """
    return (ENTRY_FEE + SL_FEE) * 10000 / risk_bps


def fee_win_ratio(risk_bps):
    """Fee cost on a win as fraction of risk (both sides maker = 0 for USDC)."""
    return (ENTRY_FEE + TP_FEE) * 10000 / risk_bps


def ev_after_fees(win_rate, n_value, avg_risk_bps):
    """Compute EV after fees."""
    fw = fee_win_ratio(avg_risk_bps)
    fl = fee_loss_ratio(avg_risk_bps)
    return win_rate * (n_value - fw) - (1 - win_rate) * (1 + fl)


def exact_group_ev_after_fees(group, n_value):
    """Compute fee-aware EV exactly using each trade's own risk in bps."""
    nv_str = str(n_value)
    pnl_r = []
    for trade in group:
        rbps = trade["risk_bps"]
        if trade["outcomes"].get(nv_str) is True:
            pnl_r.append(n_value - fee_win_ratio(rbps))
        else:
            pnl_r.append(-1 - fee_loss_ratio(rbps))
    return float(np.mean(pnl_r))


def build_strategy_export(export_id, export_name, export_description, export_spec):
    """Build a bot-ready strategy JSON from qualifying fee-adjusted cells."""
    cells = []
    for key in sorted(export_spec["qual"].keys()):
        tp, rb, setup = key
        cell = export_spec["qual"][key]
        rr_stats = cell["rr_fee"][cell["best_n_fee"]]
        cells.append({
            "time_period": tp,
            "risk_range": rb,
            "setup": setup,
            "rr_target": cell["best_n_fee"],
            "best_n": cell["best_n_fee"],
            "ev": round(cell["best_ev_fee"], 4),
            "win_rate": rr_stats["wr"],
            "samples": cell["samples"],
            "median_risk": round(cell["median_risk_bps"], 2),
            "median_risk_bps": round(cell["median_risk_bps"], 2),
            "avg_risk_bps": cell["avg_risk_bps"],
            "trades_per_day": round(cell["trades_per_day"], 2),
            "enabled": True,
            "notes": "",
        })

    return {
        "schema_version": "1.0",
        "meta": {
            "id": export_id,
            "name": export_name,
            "description": export_description,
            "source_dataset": "rr_btcusdt_5min_5y_60min",
            "ticker": "BTCUSDT",
            "timeframe": "5min",
            "selection_label": export_spec["label"],
            "risk_rules": {
                "risk_bins": RISK_BINS,
            },
            "fee_model": {
                "entry_fee": ENTRY_FEE,
                "tp_fee": TP_FEE,
                "sl_fee": SL_FEE,
                "model": "exact_per_trade_risk_bps",
            },
        },
        "filters": {
            "min_samples": export_spec["min_samples"],
            "min_ev": export_spec["min_ev"],
            "setups": sorted(export_spec["setups"]),
        },
        "cells": cells,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default=None, help="Filter trades to year (e.g. 2025)")
    parser.add_argument("--export-label", default=None, help="Export strategy for a sweep label, e.g. ev0.07_s30_both")
    parser.add_argument("--export-strategy-id", default=None, help="Override exported strategy id")
    parser.add_argument("--export-name", default=None, help="Override exported strategy name")
    parser.add_argument("--set-active", action="store_true", help="Set the exported strategy as the active strategy")
    args = parser.parse_args()

    print("Loading trades...")
    with open(os.path.join(_ROOT, "scripts", "btc_sweep_results",
                           "5min_results", "sweep_trades.json")) as f:
        all_trades = json.load(f)
    all_5y = all_trades["5min_bps5_p60m_mit90"]
    print(f"  {len(all_5y):,} total trades (5Y)")

    # Always build cells from FULL 5Y data (out-of-sample selection)
    # Then filter trades to --year for replay
    if args.year:
        replay_trades = [t for t in all_5y if t["formation_time"].startswith(args.year)]
        print(f"  {len(replay_trades):,} trades for replay ({args.year})")
    else:
        replay_trades = all_5y

    # Build cells from full 5Y (never in-sample)
    trades = all_5y
    dataset_days = len({t["formation_time"][:10] for t in trades})
    groups = defaultdict(list)
    for t in trades:
        rb = None
        for i in range(len(RISK_BINS) - 1):
            if RISK_BINS[i] <= t["risk_bps"] < RISK_BINS[i + 1]:
                rb = f"{RISK_BINS[i]}-{RISK_BINS[i+1]}"
                break
        if rb is None:
            continue
        groups[(t["time_period"], rb, t["setup"])].append(t)

    print(f"  {len(groups)} raw cells")

    # Compute fee-adjusted EV per cell, find best_n after fees
    cells = {}
    for key, group in groups.items():
        n = len(group)
        if n < 30:
            continue

        tp, rb, setup = key
        avg_risk_bps = np.mean([t["risk_bps"] for t in group])
        median_risk_bps = np.median([t["risk_bps"] for t in group])
        avg_fee_loss_ratio = np.mean([fee_loss_ratio(t["risk_bps"]) for t in group])
        trades_per_day = len(group) / dataset_days if dataset_days else 0

        best_ev_raw = -999
        best_n_raw = 1.0
        best_ev_fee = -999
        best_n_fee = 1.0

        rr_raw = {}
        rr_fee = {}

        for nv in N_VALUES:
            nv_str = str(nv)
            wins = sum(1 for t in group if t["outcomes"].get(nv_str) is True)
            wr = wins / n

            ev_raw = wr * (nv + 1) - 1
            ev_f = exact_group_ev_after_fees(group, nv)

            rr_raw[nv] = {"wr": round(wr, 4), "ev": round(ev_raw, 4), "wins": wins}
            rr_fee[nv] = {"wr": round(wr, 4), "ev": round(ev_f, 4), "wins": wins}

            if ev_raw > best_ev_raw:
                best_ev_raw = ev_raw
                best_n_raw = nv
            if ev_f > best_ev_fee:
                best_ev_fee = ev_f
                best_n_fee = nv

        cells[key] = {
            "samples": n,
            "avg_risk_bps": round(avg_risk_bps, 1),
            "median_risk_bps": round(median_risk_bps, 2),
            "trades_per_day": round(trades_per_day, 2),
            "best_n_raw": best_n_raw,
            "best_ev_raw": round(best_ev_raw, 4),
            "best_n_fee": best_n_fee,
            "best_ev_fee": round(best_ev_fee, 4),
            "fee_loss_ratio": round(avg_fee_loss_ratio, 4),
            "rr_raw": rr_raw,
            "rr_fee": rr_fee,
            "trades": group,
        }

    # Summary: how many cells survive
    all_cells = list(cells.values())
    pos_raw = [c for c in all_cells if c["best_ev_raw"] > 0]
    pos_fee = [c for c in all_cells if c["best_ev_fee"] > 0]

    print(f"\n{'='*80}")
    print(f"  CELL SURVIVAL AFTER USDC FEES")
    print(f"  Entry: {ENTRY_FEE*100:.3f}% (maker)  TP: {TP_FEE*100:.3f}% (maker)  SL: {SL_FEE*100:.3f}% (taker)")
    print(f"{'='*80}")
    print(f"  Total cells (30+ samples): {len(all_cells)}")
    print(f"  Positive EV (raw):         {len(pos_raw)}")
    print(f"  Positive EV (after fees):  {len(pos_fee)}")
    print(f"  Killed by fees:            {len(pos_raw) - len(pos_fee)}")

    # Breakdown by risk bucket
    print(f"\n  {'Bucket':>10} {'Cells':>6} {'+EV raw':>8} {'+EV fee':>8} {'Killed':>7} {'Avg fee/loss':>13}")
    print(f"  {'-'*58}")
    for bi in range(len(RISK_BINS) - 1):
        lo, hi = RISK_BINS[bi], RISK_BINS[bi + 1]
        bk = f"{lo}-{hi}"
        bucket_cells = [c for c in all_cells if any(
            k[1] == bk for k in cells if cells[k] is c
        )]
        # Simpler approach
        bucket_cells = []
        for key, c in cells.items():
            if key[1] == bk:
                bucket_cells.append(c)

        if not bucket_cells:
            continue
        n_raw = sum(1 for c in bucket_cells if c["best_ev_raw"] > 0)
        n_fee = sum(1 for c in bucket_cells if c["best_ev_fee"] > 0)
        avg_fl = np.mean([c["fee_loss_ratio"] for c in bucket_cells])
        print(f"  {bk:>10} {len(bucket_cells):>6} {n_raw:>8} {n_fee:>8} {n_raw-n_fee:>7} {avg_fl:>12.1%}")

    # Breakdown by setup
    print(f"\n  {'Setup':>14} {'Cells':>6} {'+EV raw':>8} {'+EV fee':>8}")
    print(f"  {'-'*40}")
    for setup in ["mit_extreme", "mid_extreme"]:
        sc = [c for k, c in cells.items() if k[2] == setup]
        n_raw = sum(1 for c in sc if c["best_ev_raw"] > 0)
        n_fee = sum(1 for c in sc if c["best_ev_fee"] > 0)
        print(f"  {setup:>14} {len(sc):>6} {n_raw:>8} {n_fee:>8}")

    # Show best_n shift: how many cells changed their optimal R:R after fees
    shifted = sum(1 for c in pos_fee if c["best_n_fee"] != c["best_n_raw"])
    print(f"\n  Cells where best N shifted after fees: {shifted}/{len(pos_fee)}")

    # N distribution after fees
    n_dist = defaultdict(int)
    for c in pos_fee:
        n_dist[c["best_n_fee"]] += 1
    print(f"  Best N distribution (fee-adjusted):")
    for nv in sorted(n_dist.keys()):
        print(f"    {nv:.2f}R: {n_dist[nv]} cells")

    # === Strategy replay with fees ===
    print(f"\n{'='*80}")
    print(f"  STRATEGY REPLAY: $100k fixed risk, 1%, USDC fees")
    print(f"{'='*80}")

    # Sweep min_ev thresholds on fee-adjusted EV
    ev_thresholds = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    min_samples_list = [30, 50, 100, 200]
    setup_configs = [
        ("mit_only", {"mit_extreme"}),
        ("both", {"mit_extreme", "mid_extreme"}),
    ]

    BALANCE = 100_000
    RISK_PCT = 0.01

    results = []
    export_specs = {}

    for min_ev in ev_thresholds:
        for min_samp in min_samples_list:
            for setup_label, setups in setup_configs:
                # Select qualifying cells (by fee-adjusted EV)
                qual = {}
                for key, cell in cells.items():
                    tp, rb, setup = key
                    if setup not in setups:
                        continue
                    if cell["best_ev_fee"] < min_ev:
                        continue
                    if cell["samples"] < min_samp:
                        continue
                    qual[key] = cell

                if not qual:
                    continue

                # Collect trades from replay set (not cell trades — avoids in-sample bias)
                trade_list = []
                for t in replay_trades:
                    rb = None
                    for bi in range(len(RISK_BINS) - 1):
                        if RISK_BINS[bi] <= t["risk_bps"] < RISK_BINS[bi + 1]:
                            rb = f"{RISK_BINS[bi]}-{RISK_BINS[bi+1]}"
                            break
                    if rb is None:
                        continue
                    key = (t["time_period"], rb, t["setup"])
                    if key not in qual:
                        continue
                    bn = qual[key]["best_n_fee"]
                    nv_str = str(bn)
                    won = t["outcomes"].get(nv_str) is True
                    trade_list.append((t["formation_time"], won, bn, t["risk_bps"]))

                trade_list.sort()

                # Replay with fixed risk + fees
                risk_dollar = BALANCE * RISK_PCT
                bal = BALANCE
                peak = bal
                max_dd_pct = 0
                wins = losses = 0
                total_fees = 0
                daily_pnl = defaultdict(float)

                for ft, won, bn, rbps in trade_list:
                    notional = risk_dollar * 10000 / rbps

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
                    daily_pnl[ft[:10]] += pnl

                    if bal > peak:
                        peak = bal
                    dd = (peak - bal) / peak * 100 if peak > 0 else 0
                    if dd > max_dd_pct:
                        max_dd_pct = dd

                total = wins + losses
                if total == 0:
                    continue
                trading_days = len(daily_pnl)
                dpnl = np.array(list(daily_pnl.values()))
                sharpe = (dpnl.mean() / dpnl.std()) * math.sqrt(365) if dpnl.std() > 0 else 0
                avg_ev = np.mean([c["best_ev_fee"] for c in qual.values()])

                results.append({
                    "label": f"ev{min_ev:.2f}_s{min_samp}_{setup_label}",
                    "min_ev": min_ev,
                    "min_samples": min_samp,
                    "setups": setup_label,
                    "cells": len(qual),
                    "trades": total,
                    "wins": wins,
                    "win_rate": round(wins / total * 100, 1),
                    "net_pnl": round(bal - BALANCE, 0),
                    "total_fees": round(total_fees, 0),
                    "fee_pct_of_pnl_gross": round(total_fees / (bal - BALANCE + total_fees) * 100, 1) if (bal - BALANCE + total_fees) > 0 else 999,
                    "max_dd_pct": round(max_dd_pct, 1),
                    "sharpe": round(sharpe, 3),
                    "trades_per_day": round(total / trading_days, 1),
                    "avg_ev_fee": round(avg_ev, 4),
                })
                export_specs[f"ev{min_ev:.2f}_s{min_samp}_{setup_label}"] = {
                    "label": f"ev{min_ev:.2f}_s{min_samp}_{setup_label}",
                    "min_ev": min_ev,
                    "min_samples": min_samp,
                    "setups": setups,
                    "qual": dict(qual),
                }

    # Print results
    results.sort(key=lambda r: r["net_pnl"], reverse=True)
    print(f"\n{'#':>3} {'Label':<30} {'Cells':>5} {'Trades':>7} {'T/D':>5} {'WR':>5} "
          f"{'Net PnL':>12} {'Fees':>10} {'Fee%':>5} {'DD%':>5} {'Sharpe':>6} {'AvgEV':>7}")
    print("-" * 120)
    for i, r in enumerate(results[:30]):
        print(f"{i+1:>3} {r['label']:<30} {r['cells']:>5} {r['trades']:>7} "
              f"{r['trades_per_day']:>5.1f} {r['win_rate']:>5.1f} "
              f"${r['net_pnl']:>10,.0f} ${r['total_fees']:>8,.0f} {r['fee_pct_of_pnl_gross']:>5.1f} "
              f"{r['max_dd_pct']:>5.1f} {r['sharpe']:>6.3f} {r['avg_ev_fee']:>7.4f}")

    # Sweet spot
    print(f"\n  SWEET SPOT (fee-adjusted):")
    for target_dd in [5, 8, 10, 15, 20]:
        candidates = [r for r in results if r["max_dd_pct"] <= target_dd and r["net_pnl"] > 0]
        if candidates:
            best = max(candidates, key=lambda r: r["net_pnl"])
            print(f"    DD <= {target_dd:>2}%: {best['label']:<30} PnL=${best['net_pnl']:>10,.0f}  "
                  f"Fees=${best['total_fees']:>8,.0f}  DD={best['max_dd_pct']}%  "
                  f"Sharpe={best['sharpe']:.3f}  Trades={best['trades']}  T/D={best['trades_per_day']}")

    # Save
    out_path = os.path.join(_ROOT, "scripts", "btc_sweep_results", "fee_adjusted_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    if args.export_label:
        export_spec = export_specs.get(args.export_label)
        if export_spec is None:
            raise ValueError(f"Unknown export label: {args.export_label}")

        export_id = args.export_strategy_id or f"btc-5min-locked-{args.export_label.replace('.', '').replace('_', '-')}"
        export_name = args.export_name or export_id
        result = next(r for r in results if r["label"] == args.export_label)
        export_description = (
            f"Locked fee-aware exact EV strategy: EV>={result['min_ev']:.2f}, "
            f"{result['min_samples']}+ samples, {result['setups']}, "
            f"{result['cells']} cells, ~{result['trades_per_day']} trades/day, "
            f"2025 replay DD {result['max_dd_pct']:.1f}%, Sharpe {result['sharpe']:.3f}."
        )
        strategy = build_strategy_export(export_id, export_name, export_description, export_spec)
        strategy_dir = os.path.join(_ROOT, "logic", "strategies")
        strategy_id = save_strategy(strategy, strategy_dir)
        print(f"Exported strategy to {os.path.join(strategy_dir, f'{strategy_id}.json')}")
        if args.set_active:
            set_active_strategy(strategy_id, strategy_dir)
            print(f"Set active_strategy={strategy_id}")


if __name__ == "__main__":
    main()
