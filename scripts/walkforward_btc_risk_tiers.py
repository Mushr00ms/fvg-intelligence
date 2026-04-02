#!/usr/bin/env python3
"""
walkforward_btc_risk_tiers.py - Walk-forward BTC cell/overlay study.

Workflow:
1. Train cells on 2020-2023 only using exact fee-aware EV.
2. Design prune/tier overlays on 2024 only.
3. Evaluate untouched on 2025.

The study stays aligned with the live leverage-aware BTC family by:
- dropping sub-20bps buckets
- using exact USDC fee-aware R outcomes
- replaying 1% risk on a configurable reference account

Outputs:
- full result grid JSON
- concise console summary highlighting:
  - best validation config
  - best validation config under DD cap
  - top stable configs that stayed positive in both validation and test
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRADES_PATH = ROOT / "scripts" / "btc_sweep_results" / "official_audit_trades.json"
OUT_DIR = ROOT / "scripts" / "btc_sweep_results"

RISK_BINS = [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 994]
N_VALUES = [round(1.0 + i * 0.25, 2) for i in range(9)]
SL_FEE = 0.0004
DEFAULT_START_BALANCE = 100_000.0
BASE_RISK_PCT = 0.01
LOW_RISK_BUCKETS = {"1-7", "7-10", "10-12", "12-14", "14-17", "17-20"}


def risk_to_range(risk_bps: float) -> str | None:
    for i in range(len(RISK_BINS) - 1):
        if RISK_BINS[i] <= risk_bps < RISK_BINS[i + 1]:
            return f"{RISK_BINS[i]}-{RISK_BINS[i + 1]}"
    return None


def loss_fee_ratio(risk_bps: float) -> float:
    return 10_000 * SL_FEE / risk_bps


def exact_group_ev(group: list[dict], n_value: float) -> float:
    pnl_r = []
    nv_str = str(n_value)
    for trade in group:
        won = trade["outcomes"].get(nv_str) is True
        if won:
            pnl_r.append(n_value)
        else:
            pnl_r.append(-(1.0 + loss_fee_ratio(trade["risk_bps"])))
    return float(np.mean(pnl_r))


def load_trades(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def split_years(trades: list[dict], train_end: str, val_year: str, test_year: str):
    train = [t for t in trades if "2020" <= t["formation_time"][:4] <= train_end]
    val = [t for t in trades if t["formation_time"][:4] == val_year]
    test = [t for t in trades if t["formation_time"][:4] == test_year]
    return train, val, test


def build_train_cells(trades: list[dict]) -> dict:
    groups = defaultdict(list)
    for trade in trades:
        risk_range = risk_to_range(trade["risk_bps"])
        if risk_range is None or risk_range in LOW_RISK_BUCKETS:
            continue
        key = (trade["time_period"], risk_range, trade["setup"])
        groups[key].append(trade)

    cells = {}
    for key, group in groups.items():
        best_n = 1.0
        best_ev = -1e9
        best_win_rate = 0.0
        for n_value in N_VALUES:
            wins = sum(1 for trade in group if trade["outcomes"].get(str(n_value)) is True)
            win_rate = wins / len(group)
            ev = exact_group_ev(group, n_value)
            if ev > best_ev:
                best_ev = ev
                best_n = n_value
                best_win_rate = win_rate
        cells[key] = {
            "best_n": best_n,
            "best_ev": best_ev,
            "samples": len(group),
            "win_rate": best_win_rate,
            "avg_risk_bps": float(np.mean([trade["risk_bps"] for trade in group])),
        }
    return cells


def build_year_events(trades: list[dict], train_cells: dict) -> dict[str, list[dict]]:
    events_by_year = defaultdict(list)
    for trade in trades:
        risk_range = risk_to_range(trade["risk_bps"])
        if risk_range is None:
            continue
        key = (trade["time_period"], risk_range, trade["setup"])
        cell = train_cells.get(key)
        if cell is None:
            continue
        won = trade["outcomes"].get(str(cell["best_n"])) is True
        pnl_r = cell["best_n"] if won else -(1.0 + loss_fee_ratio(trade["risk_bps"]))
        events_by_year[trade["formation_time"][:4]].append(
            {
                "formation_time": trade["formation_time"],
                "cell_key": key,
                "setup": trade["setup"],
                "risk_range": risk_range,
                "pnl_r": pnl_r,
            }
        )

    for year in events_by_year:
        events_by_year[year].sort(key=lambda e: e["formation_time"])
    return events_by_year


def select_cells(train_cells: dict, min_ev: float, min_samples: int, setups: set[str]) -> dict:
    return {
        key: cell
        for key, cell in train_cells.items()
        if cell["best_ev"] >= min_ev and cell["samples"] >= min_samples and key[2] in setups
    }


def build_validation_stats(val_events: list[dict], selected_cells: dict) -> dict:
    stats = defaultdict(lambda: {"trades": 0, "r_sum": 0.0})
    for event in val_events:
        if event["cell_key"] not in selected_cells:
            continue
        stats[event["cell_key"]]["trades"] += 1
        stats[event["cell_key"]]["r_sum"] += event["pnl_r"]
    return stats


def replay_events(
    events: list[dict],
    selected_cells: dict,
    val_stats: dict,
    prune_min_trades: int | None,
    mid_weight: float,
    bucket20_weight: float,
    start_balance: float,
    compound: bool,
) -> dict:
    pruned = set()
    if prune_min_trades is not None:
        for cell_key, stat in val_stats.items():
            if stat["trades"] >= prune_min_trades and stat["r_sum"] < 0:
                pruned.add(cell_key)

    balance = start_balance
    peak = balance
    max_dd_pct = 0.0
    wins = 0
    total = 0
    daily_pnl = defaultdict(float)
    active_cells = set()

    for event in events:
        cell_key = event["cell_key"]
        if cell_key not in selected_cells or cell_key in pruned:
            continue

        weight = 1.0
        if event["setup"] == "mid_extreme":
            weight *= mid_weight
        if event["risk_range"] == "20-24":
            weight *= bucket20_weight
        if weight <= 0:
            continue

        if balance <= 0:
            break

        risk_base = balance if compound else start_balance
        pnl = risk_base * BASE_RISK_PCT * weight * event["pnl_r"]
        balance += pnl
        peak = max(peak, balance)
        if peak > 0:
            dd_pct = (peak - balance) / peak * 100
            max_dd_pct = max(max_dd_pct, dd_pct)

        total += 1
        wins += 1 if event["pnl_r"] > 0 else 0
        daily_pnl[event["formation_time"][:10]] += pnl
        active_cells.add(cell_key)

    daily_series = np.array(list(daily_pnl.values()))
    sharpe = 0.0
    if len(daily_series) > 1 and daily_series.std() > 0:
        sharpe = float((daily_series.mean() / daily_series.std()) * math.sqrt(365))

    return {
        "cells": len(active_cells),
        "trades": total,
        "win_rate": round(wins / total * 100, 1) if total else 0.0,
        "net_pnl": round(balance - start_balance, 1),
        "max_dd_pct": round(max_dd_pct, 2),
        "sharpe": round(sharpe, 3),
        "pruned_cells": len(pruned),
        "final_balance": round(balance, 1),
    }


def summarize_config(row: dict) -> str:
    prune = "none" if row["prune_min_trades"] is None else str(row["prune_min_trades"])
    return (
        f"ev>={row['min_ev']:.2f} s>={row['min_samples']} {row['setups']} "
        f"prune={prune} mid={row['mid_weight']:.2f} b20={row['bucket20_weight']:.2f}"
    )


def print_result_block(title: str, rows: list[dict], limit: int = 10):
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")
    header = (
        f"{'#':>3} {'Config':<58} {'Val PnL':>10} {'Val DD':>7} {'Val Sh':>7} "
        f"{'Test PnL':>10} {'Test DD':>8} {'Test Sh':>7} {'Cells':>6}"
    )
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(rows[:limit], start=1):
        print(
            f"{idx:>3} {summarize_config(row):<58} "
            f"{row['validation']['net_pnl']:>10,.0f} {row['validation']['max_dd_pct']:>7.2f} {row['validation']['sharpe']:>7.3f} "
            f"{row['test']['net_pnl']:>10,.0f} {row['test']['max_dd_pct']:>8.2f} {row['test']['sharpe']:>7.3f} "
            f"{row['test']['cells']:>6}"
        )


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for row in rows:
        key = (
            row["validation"]["net_pnl"],
            row["validation"]["max_dd_pct"],
            row["validation"]["sharpe"],
            row["validation"]["trades"],
            row["test"]["net_pnl"],
            row["test"]["max_dd_pct"],
            row["test"]["sharpe"],
            row["test"]["trades"],
            row["test"]["cells"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", default="2023")
    parser.add_argument("--validation-year", default="2024")
    parser.add_argument("--test-year", default="2025")
    parser.add_argument("--trades-path", default=str(DEFAULT_TRADES_PATH))
    parser.add_argument("--start-balance", type=float, default=DEFAULT_START_BALANCE)
    parser.add_argument("--compound", action="store_true", help="Compound 1%% risk from current balance")
    parser.add_argument("--output", default=str(OUT_DIR / "walkforward_btc_risk_tiers.json"))
    args = parser.parse_args()

    print("Loading BTC audit trades...")
    trades = load_trades(Path(args.trades_path))
    train, val, test = split_years(trades, args.train_end, args.validation_year, args.test_year)
    print(f"  train: {len(train):,} trades")
    print(f"  validation: {len(val):,} trades")
    print(f"  test: {len(test):,} trades")

    print("Building train-only cells...")
    train_cells = build_train_cells(train)
    year_events = build_year_events(trades, train_cells)
    val_events = year_events[args.validation_year]
    test_events = year_events[args.test_year]
    print(f"  train cells: {len(train_cells)}")

    base_grid = []
    for min_ev in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        for min_samples in [30, 50, 100, 150, 200]:
            base_grid.append((min_ev, min_samples, "mit_only", {"mit_extreme"}))
            base_grid.append((min_ev, min_samples, "both", {"mit_extreme", "mid_extreme"}))

    overlay_grid = []
    for prune_min_trades in [None, 5, 10, 15, 20]:
        for mid_weight in [1.0, 0.75, 0.5, 0.25]:
            for bucket20_weight in [1.0, 0.75, 0.5, 0.25]:
                overlay_grid.append((prune_min_trades, mid_weight, bucket20_weight))

    rows = []
    for min_ev, min_samples, setup_label, setups in base_grid:
        selected = select_cells(train_cells, min_ev, min_samples, setups)
        if not selected:
            continue
        val_stats = build_validation_stats(val_events, selected)
        for prune_min_trades, mid_weight, bucket20_weight in overlay_grid:
            validation = replay_events(
                val_events,
                selected,
                val_stats,
                prune_min_trades,
                mid_weight,
                bucket20_weight,
                args.start_balance,
                args.compound,
            )
            test_result = replay_events(
                test_events,
                selected,
                val_stats,
                prune_min_trades,
                mid_weight,
                bucket20_weight,
                args.start_balance,
                args.compound,
            )
            rows.append(
                {
                    "min_ev": min_ev,
                    "min_samples": min_samples,
                    "setups": setup_label,
                    "prune_min_trades": prune_min_trades,
                    "mid_weight": mid_weight,
                    "bucket20_weight": bucket20_weight,
                    "validation": validation,
                    "test": test_result,
                }
            )

    print(f"  configs evaluated: {len(rows)}")

    best_validation = max(rows, key=lambda row: row["validation"]["sharpe"])
    dd30_rows = [row for row in rows if row["validation"]["max_dd_pct"] <= 30 and row["validation"]["trades"] > 0]
    best_validation_dd30 = max(dd30_rows, key=lambda row: row["validation"]["net_pnl"])
    stable_rows = [
        row for row in rows
        if row["validation"]["net_pnl"] > 0 and row["test"]["net_pnl"] > 0
    ]
    stable_rows.sort(key=lambda row: row["test"]["sharpe"], reverse=True)
    stable_no_overlay = [
        row for row in stable_rows
        if row["prune_min_trades"] is None and row["mid_weight"] == 1.0 and row["bucket20_weight"] == 1.0
    ]
    stable_rows = dedupe_rows(stable_rows)
    stable_no_overlay = dedupe_rows(stable_no_overlay)

    print_result_block(f"Best {args.validation_year} Validation Sharpe", [best_validation], limit=1)
    print_result_block(f"Best {args.validation_year} Validation PnL With DD<=30%", [best_validation_dd30], limit=1)
    print_result_block(
        f"Top Stable Configs (Positive In {args.validation_year} And {args.test_year})",
        stable_rows,
        limit=12,
    )
    print_result_block(f"Top Stable Base Configs (No Overlay)", stable_no_overlay, limit=12)

    payload = {
        "meta": {
            "train_end": args.train_end,
            "validation_year": args.validation_year,
            "test_year": args.test_year,
            "start_balance": args.start_balance,
            "base_risk_pct": BASE_RISK_PCT,
            "compound": args.compound,
            "sl_fee": SL_FEE,
            "risk_bins": RISK_BINS,
            "overlay_grid_size": len(overlay_grid),
            "base_grid_size": len(base_grid),
        },
        "summary": {
            "train_trades": len(train),
            "validation_trades": len(val),
            "test_trades": len(test),
            "train_cells": len(train_cells),
            "configs_evaluated": len(rows),
            "best_validation": best_validation,
            "best_validation_dd30": best_validation_dd30,
            "stable_positive_count": len(stable_rows),
        },
        "results": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
