#!/usr/bin/env python3
"""
export_btc_walkforward_strategy.py - Export a walk-forward BTC strategy JSON.

Builds train-only cells, applies a validation-year prune rule, and writes a
strategy JSON compatible with both the leverage sweep and crypto bot runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.walkforward_btc_risk_tiers import (
    DEFAULT_TRADES_PATH,
    BASE_RISK_PCT,
    build_train_cells,
    build_validation_stats,
    build_year_events,
    load_trades,
    select_cells,
    split_years,
)

OUT_DIR = ROOT / "bot" / "strategies"


def _pruned_cells(val_stats: dict, prune_min_trades: int | None) -> set:
    pruned = set()
    if prune_min_trades is None:
        return pruned
    for cell_key, stat in val_stats.items():
        if stat["trades"] >= prune_min_trades and stat["r_sum"] < 0:
            pruned.add(cell_key)
    return pruned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-path", default=str(DEFAULT_TRADES_PATH))
    parser.add_argument("--train-end", default="2024")
    parser.add_argument("--validation-year", default="2025")
    parser.add_argument("--min-ev", type=float, required=True)
    parser.add_argument("--min-samples", type=int, required=True)
    parser.add_argument("--setups", choices=["mit_only", "both"], required=True)
    parser.add_argument("--prune-min-trades", type=int, default=None)
    parser.add_argument("--id", required=True, help="Strategy id / filename stem")
    parser.add_argument("--ticker", default="BTCUSDT")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    trades = load_trades(Path(args.trades_path))
    train, val, _ = split_years(trades, args.train_end, args.validation_year, args.validation_year)
    train_cells = build_train_cells(train)
    year_events = build_year_events(trades, train_cells)

    setup_set = {"mit_extreme"} if args.setups == "mit_only" else {"mit_extreme", "mid_extreme"}
    selected = select_cells(train_cells, args.min_ev, args.min_samples, setup_set)
    val_stats = build_validation_stats(year_events[args.validation_year], selected)
    pruned = _pruned_cells(val_stats, args.prune_min_trades)

    train_days = (
        (datetime.fromisoformat(f"{args.train_end}-12-31") - datetime.fromisoformat("2020-01-01")).days + 1
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    cells = []
    for cell_key, cell in sorted(selected.items()):
        if cell_key in pruned:
            continue
        tp, rb, setup = cell_key
        cells.append(
            {
                "time_period": tp,
                "risk_range": rb,
                "setup": setup,
                "rr_target": cell["best_n"],
                "best_n": cell["best_n"],
                "ev": round(cell["best_ev"], 4),
                "win_rate": round(cell["win_rate"], 4),
                "samples": cell["samples"],
                "median_risk": round(cell["avg_risk_bps"], 2),
                "median_risk_bps": round(cell["avg_risk_bps"], 2),
                "avg_risk_bps": round(cell["avg_risk_bps"], 2),
                "trades_per_day": round(cell["samples"] / max(train_days, 1), 3),
                "enabled": True,
                "notes": (
                    f"train<= {args.train_end}; validation prune on {args.validation_year}; "
                    f"prune_min_trades={args.prune_min_trades}"
                ),
            }
        )

    output = Path(args.output) if args.output else (OUT_DIR / f"{args.id}.json")
    payload = {
        "schema_version": "1.0",
        "meta": {
            "id": args.id,
            "name": args.id,
            "description": (
                f"Walk-forward BTC strategy: train<= {args.train_end}, prune on {args.validation_year}, "
                f"EV>={args.min_ev:.2f}, samples>={args.min_samples}, setups={args.setups}, "
                f"prune>={args.prune_min_trades or 'none'}."
            ),
            "source_dataset": Path(args.trades_path).name,
            "ticker": args.ticker,
            "timeframe": "5min",
            "selection_label": (
                f"wf_train{args.train_end}_prune{args.validation_year}_"
                f"ev{args.min_ev:.2f}_s{args.min_samples}_{args.setups}"
            ),
            "risk_rules": {
                "risk_bins": [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 994],
            },
            "fee_model": {
                "entry_fee": 0.0,
                "tp_fee": 0.0,
                "sl_fee": 0.0004,
                "model": "exact_per_trade_risk_bps",
            },
            "created_at": now_iso,
            "updated_at": now_iso,
        },
        "filters": {
            "train_end": args.train_end,
            "validation_year": args.validation_year,
            "min_samples": args.min_samples,
            "min_ev": args.min_ev,
            "setups": sorted(setup_set),
            "prune_min_trades": args.prune_min_trades,
            "risk_per_trade": BASE_RISK_PCT,
            "min_risk_bps": 20.0,
        },
        "cells": cells,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"saved {len(cells)} cells to {output}")


if __name__ == "__main__":
    main()
