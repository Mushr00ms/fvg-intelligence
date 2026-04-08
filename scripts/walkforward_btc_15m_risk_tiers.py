#!/usr/bin/env python3
"""
walkforward_btc_15m_risk_tiers.py - Walk-forward BTC 15m config/cell study.

Mirrors the current 5m BTC walk-forward flow, but starts from the 15m sweep
trade archive where each top-level key is a 15m FVG config label.

Workflow per 15m config label:
1. Train cells on years <= train_end
2. Design prune/tier overlays on validation_year
3. Evaluate untouched on test_year

Outputs:
- full result grid JSON across all evaluated 15m config labels
- concise console summary highlighting:
  - best validation config
  - best validation config under DD cap
  - top stable configs that stayed positive in both validation and test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.walkforward_btc_risk_tiers import (
    BASE_RISK_PCT,
    RISK_BINS,
    SL_FEE,
    build_train_cells,
    build_validation_stats,
    build_year_events,
    dedupe_rows,
    print_result_block,
    replay_events,
    select_cells,
    split_years,
    summarize_config,
)


DEFAULT_TRADES_PATH = ROOT / "scripts" / "btc_sweep_results" / "15min_results" / "sweep_trades.json"
OUT_DIR = ROOT / "scripts" / "btc_sweep_results" / "15min_results"


def load_trade_sets(path: Path) -> dict[str, list[dict]]:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict[label -> trades] in {path}, got {type(payload).__name__}")
    return payload


def choose_labels(
    trade_sets: dict[str, list[dict]],
    requested: list[str] | None,
    *,
    limit: int | None,
) -> list[str]:
    labels = sorted(trade_sets)
    if requested:
        missing = [label for label in requested if label not in trade_sets]
        if missing:
            raise KeyError(f"Unknown config_label(s): {', '.join(missing)}")
        labels = requested
    if limit is not None:
        labels = labels[:limit]
    return labels


def stable_row_key(row: dict) -> tuple:
    return (
        row["validation"]["net_pnl"],
        row["validation"]["max_dd_pct"],
        row["validation"]["sharpe"],
        row["validation"]["trades"],
        row["test"]["net_pnl"],
        row["test"]["max_dd_pct"],
        row["test"]["sharpe"],
        row["test"]["trades"],
        row["test"]["cells"],
        row["config_label"],
    )


def dedupe_stable_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for row in rows:
        key = stable_row_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def print_labeled_block(title: str, rows: list[dict], limit: int = 10):
    print(f"\n{'=' * 150}")
    print(f"  {title}")
    print(f"{'=' * 150}")
    header = (
        f"{'#':>3} {'Config Label':<28} {'Selector':<58} {'Val PnL':>10} {'Val DD':>7} "
        f"{'Val Sh':>7} {'Test PnL':>10} {'Test DD':>8} {'Test Sh':>7} {'Cells':>6}"
    )
    print(header)
    print("-" * len(header))
    for idx, row in enumerate(rows[:limit], start=1):
        print(
            f"{idx:>3} {row['config_label']:<28} {summarize_config(row):<58} "
            f"{row['validation']['net_pnl']:>10,.0f} {row['validation']['max_dd_pct']:>7.2f} "
            f"{row['validation']['sharpe']:>7.3f} {row['test']['net_pnl']:>10,.0f} "
            f"{row['test']['max_dd_pct']:>8.2f} {row['test']['sharpe']:>7.3f} "
            f"{row['test']['cells']:>6}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", default="2023")
    parser.add_argument("--validation-year", default="2024")
    parser.add_argument("--test-year", default="2025")
    parser.add_argument("--trades-path", default=str(DEFAULT_TRADES_PATH))
    parser.add_argument(
        "--config-label",
        action="append",
        dest="config_labels",
        help="Exact 15m config label to evaluate. Repeat to run multiple labels.",
    )
    parser.add_argument("--limit-labels", type=int, default=None, help="Evaluate only the first N labels after filtering")
    parser.add_argument("--start-balance", type=float, default=100_000.0)
    parser.add_argument("--compound", action="store_true", help="Compound 1%% risk from current balance")
    parser.add_argument("--dd-cap", type=float, default=30.0, help="DD cap used for validation ranking summary")
    parser.add_argument("--output", default=str(OUT_DIR / "walkforward_btc_15m_risk_tiers.json"))
    args = parser.parse_args()

    trade_sets = load_trade_sets(Path(args.trades_path))
    labels = choose_labels(trade_sets, args.config_labels, limit=args.limit_labels)

    print(f"Loading BTC 15m trade sets from {Path(args.trades_path).name}...")
    print(f"  config labels available: {len(trade_sets):,}")
    print(f"  config labels selected : {len(labels):,}")

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
    label_summaries = []

    for idx, label in enumerate(labels, start=1):
        trades = trade_sets[label]
        train, val, test = split_years(trades, args.train_end, args.validation_year, args.test_year)
        train_cells = build_train_cells(train)
        year_events = build_year_events(trades, train_cells)
        val_events = year_events[args.validation_year]
        test_events = year_events[args.test_year]

        if idx == 1 or idx % 10 == 0 or idx == len(labels):
            print(
                f"  [{idx}/{len(labels)}] {label}: "
                f"train={len(train):,} val={len(val):,} test={len(test):,} train_cells={len(train_cells):,}"
            )

        label_row_count_before = len(rows)

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
                        "config_label": label,
                        "train_trades": len(train),
                        "validation_trades": len(val),
                        "test_trades": len(test),
                        "train_cells": len(train_cells),
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

        label_rows = rows[label_row_count_before:]
        stable_count = sum(
            1
            for row in label_rows
            if row["validation"]["net_pnl"] > 0 and row["test"]["net_pnl"] > 0
        )
        label_summaries.append(
            {
                "config_label": label,
                "train_trades": len(train),
                "validation_trades": len(val),
                "test_trades": len(test),
                "train_cells": len(train_cells),
                "configs_evaluated": len(label_rows),
                "stable_positive_count": stable_count,
            }
        )

    if not rows:
        raise RuntimeError("No walk-forward rows were produced. Check the selected labels and year split.")

    print(f"  configs evaluated: {len(rows):,}")

    best_validation = max(rows, key=lambda row: row["validation"]["sharpe"])
    dd_rows = [
        row
        for row in rows
        if row["validation"]["max_dd_pct"] <= args.dd_cap and row["validation"]["trades"] > 0
    ]
    best_validation_dd = max(dd_rows, key=lambda row: row["validation"]["net_pnl"]) if dd_rows else None

    stable_rows = [
        row
        for row in rows
        if row["validation"]["net_pnl"] > 0 and row["test"]["net_pnl"] > 0
    ]
    stable_rows.sort(key=lambda row: row["test"]["sharpe"], reverse=True)
    stable_rows = dedupe_stable_rows(stable_rows)

    stable_no_overlay = [
        row
        for row in stable_rows
        if row["prune_min_trades"] is None and row["mid_weight"] == 1.0 and row["bucket20_weight"] == 1.0
    ]
    stable_no_overlay = dedupe_rows(stable_no_overlay)

    print_labeled_block(f"Best {args.validation_year} Validation Sharpe", [best_validation], limit=1)
    if best_validation_dd is not None:
        print_labeled_block(
            f"Best {args.validation_year} Validation PnL With DD<={args.dd_cap:.0f}%",
            [best_validation_dd],
            limit=1,
        )
    print_labeled_block(
        f"Top Stable Configs (Positive In {args.validation_year} And {args.test_year})",
        stable_rows,
        limit=12,
    )
    print_labeled_block("Top Stable Base Configs (No Overlay)", stable_no_overlay, limit=12)

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
            "trades_path": str(Path(args.trades_path)),
            "labels_evaluated": len(labels),
            "overlay_grid_size": len(overlay_grid),
            "base_grid_size": len(base_grid),
            "dd_cap": args.dd_cap,
        },
        "summary": {
            "config_labels": label_summaries,
            "configs_evaluated": len(rows),
            "best_validation": best_validation,
            "best_validation_dd_cap": best_validation_dd,
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
