#!/usr/bin/env python3
"""
export_btc_strategy_rr_dataset.py - Export a BTC strategy JSON into rr_data.

This makes a saved strategy visible in the RR dashboard at `/`, which reads
`/api/rr/manifest` and expects datasets in `logic/rr_data/`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RR_STORE_DIR = ROOT / "logic" / "rr_data"
N_VALUES = [round(1.0 + i * 0.25, 2) for i in range(9)]
SETUPS = ["mit_open", "mit_extreme", "mid_open", "mid_extreme"]
SETUP_LABELS = {
    "mit_open": "Mitigation + Open Stop",
    "mit_extreme": "Mitigation + Extreme Stop",
    "mid_open": "Midpoint + Open Stop",
    "mid_extreme": "Midpoint + Extreme Stop",
}
DEFAULT_RISK_BINS = [5, 10, 15, 20, 25, 30, 40, 50, 200]


def _empty_setup():
    return {
        "activated": 0,
        "valid": 0,
        "median_risk": None,
        "wins": [0] * 9,
        "win_rates": [None] * 9,
        "evs": [None] * 9,
    }


def build_rr_cells(strategy: dict) -> list[dict]:
    cells = []

    for cell in strategy.get("cells", []):
        if not cell.get("enabled", True):
            continue

        entry = {
            "time_period": cell["time_period"],
            "risk_range": cell["risk_range"],
            "sample_count": int(cell.get("samples", 0)),
            "setups": {setup: _empty_setup() for setup in SETUPS},
        }

        setup_name = cell["setup"]
        target = float(cell.get("best_n", cell.get("rr_target")))
        try:
            idx = N_VALUES.index(target)
        except ValueError:
            continue

        samples = int(cell.get("samples", 0))
        win_rate = cell.get("win_rate")
        ev = cell.get("ev")
        wins = 0
        if win_rate is not None and samples > 0:
            wr_decimal = float(win_rate)
            if wr_decimal > 1:
                wr_decimal = wr_decimal / 100.0
            wins = int(round(wr_decimal * samples))
            win_rate_pct = round(wr_decimal * 100.0, 2)
        else:
            win_rate_pct = None

        s = entry["setups"][setup_name]
        s["activated"] = samples
        s["valid"] = samples
        s["median_risk"] = round(float(cell.get("avg_risk_bps", cell.get("median_risk_bps", 0.0))), 2) if samples else None
        s["wins"][idx] = wins
        s["win_rates"][idx] = win_rate_pct
        s["evs"][idx] = round(float(ev), 4) if ev is not None else None

        cells.append(entry)

    cells.sort(key=lambda c: (c["time_period"], c["risk_range"]))
    return cells


def get_rr_dataset_id(ticker: str, timeframe_label: str, data_period: str, session_period_minutes: int) -> str:
    return f"rr_{ticker.lower()}_{timeframe_label}_{data_period}_{session_period_minutes}min"


def save_rr_dataset(cells: list[dict], ticker: str, timeframe_label: str, data_period: str, session_period_minutes: int) -> str:
    RR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    dataset_id = get_rr_dataset_id(ticker, timeframe_label, data_period, session_period_minutes)
    now = datetime.now().isoformat()

    payload = {
        "meta": {
            "id": dataset_id,
            "ticker": ticker,
            "timeframe_label": timeframe_label,
            "data_period": data_period,
            "session_period_minutes": session_period_minutes,
            "risk_bins": DEFAULT_RISK_BINS,
            "n_values": N_VALUES,
            "setups": SETUPS,
            "setup_labels": SETUP_LABELS,
            "created_at": now,
        },
        "cells": cells,
    }

    data_path = RR_STORE_DIR / f"{dataset_id}.json"
    tmp_path = RR_STORE_DIR / f"{dataset_id}.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    tmp_path.replace(data_path)

    manifest_path = RR_STORE_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"datasets": []}

    manifest["datasets"] = [d for d in manifest.get("datasets", []) if d["id"] != dataset_id]
    manifest["datasets"].append(
        {
            "id": dataset_id,
            "ticker": ticker,
            "timeframe_label": timeframe_label,
            "data_period": data_period,
            "session_period_minutes": session_period_minutes,
            "created_at": now,
        }
    )
    manifest["last_updated"] = now

    manifest_tmp = RR_STORE_DIR / "manifest.json.tmp"
    with open(manifest_tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    manifest_tmp.replace(manifest_path)

    return dataset_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-path", required=True)
    parser.add_argument("--data-period", required=True, help="Label shown in RR dashboard, e.g. wf24p25")
    parser.add_argument("--timeframe-label", default="5min")
    parser.add_argument("--session-period-minutes", type=int, default=60)
    parser.add_argument("--ticker", default=None)
    args = parser.parse_args()

    strategy_path = ROOT / args.strategy_path if not Path(args.strategy_path).is_absolute() else Path(args.strategy_path)
    with open(strategy_path) as f:
        strategy = json.load(f)

    cells = build_rr_cells(strategy)
    ticker = args.ticker or strategy.get("meta", {}).get("ticker", "BTCUSDT")

    dataset_id = save_rr_dataset(
        cells=cells,
        ticker=ticker,
        timeframe_label=args.timeframe_label,
        data_period=args.data_period,
        session_period_minutes=args.session_period_minutes,
    )
    print(dataset_id)


if __name__ == "__main__":
    main()
