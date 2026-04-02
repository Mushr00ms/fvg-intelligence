#!/usr/bin/env python3
"""
export_btc_strategy_heatmap.py - Save a BTC strategy as a dashboard heatmap dataset.

This maps a strategy's cell grid into the existing dashboard heatmap store so the
frontend can render it using the same API/storage path as the other heatmaps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def _load_heatmap_store():
    store_path = ROOT / "logic" / "utils" / "heatmap_store.py"
    spec = importlib.util.spec_from_file_location("heatmap_store", store_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _strategy_to_frame(strategy: dict) -> pd.DataFrame:
    rows = []
    for cell in strategy.get("cells", []):
        if not cell.get("enabled", True):
            continue
        rows.append(
            {
                "time_period": cell["time_period"],
                "size_range": cell["risk_range"],
                "total_fvgs": cell.get("samples"),
                "optimal_target": cell.get("best_n", cell.get("rr_target")),
                "optimal_ev": cell.get("ev"),
                "avg_risk_points": cell.get("avg_risk_bps"),
                "avg_rr": cell.get("best_n", cell.get("rr_target")),
                "rr_1_0_hit_rate": round(float(cell.get("win_rate", 0.0)) * 100, 2)
                if cell.get("win_rate") is not None
                else None,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-path", required=True)
    parser.add_argument("--store-dir", default=None)
    args = parser.parse_args()

    strategy_path = ROOT / args.strategy_path if not os.path.isabs(args.strategy_path) else Path(args.strategy_path)
    with open(strategy_path) as f:
        strategy = json.load(f)

    df = _strategy_to_frame(strategy)
    if df.empty:
        raise ValueError("Strategy has no enabled cells to export")

    heatmap_store = _load_heatmap_store()
    meta = strategy.get("meta", {})
    filters = strategy.get("filters", {})
    size_values = sorted(
        {
            float(str(cell["risk_range"]).split("-")[0])
            for cell in strategy.get("cells", [])
            if cell.get("enabled", True)
        }
    )
    size_start = min(size_values) if size_values else 20.0
    size_end = max(
        float(str(cell["risk_range"]).split("-")[1])
        for cell in strategy.get("cells", [])
        if cell.get("enabled", True)
    ) if strategy.get("cells") else 43.0
    train_end = filters.get("train_end", "na")
    validation_year = filters.get("validation_year", "na")
    dataset_id = heatmap_store.save_dataset(
        df=df,
        ticker=meta.get("ticker", "BTCUSDT"),
        timeframe_label="5mcell",
        data_period=f"wf_t{train_end}_p{validation_year}",
        session_period_minutes=60,
        size_filtering_method="risk_range",
        size_range_start=size_start,
        size_range_end=size_end,
        size_range_step=1.0,
        min_expansion_size=float(filters.get("min_ev", 0.0)),
        source_csv_path=str(strategy_path),
        store_dir=args.store_dir,
    )
    print(dataset_id)


if __name__ == "__main__":
    main()
