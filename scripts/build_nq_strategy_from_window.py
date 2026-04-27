#!/usr/bin/env python3
"""
build_nq_strategy_from_window.py — Build a NQ 5min strategy JSON from a
train-window FVG parquet, mirroring the v4 selection rules used by the live
``mixed-best-ev-v3-touch-moderate`` strategy.

Selection rules (reverse-engineered from the live strategy file):
  - extreme-stop setups only: mit_extreme, mid_extreme
  - per (time_period, risk_range) cell, pick the (setup, rr_target) with
    the highest EV
  - keep cells with EV >= --min-ev (default 0.07) and samples >= --min-samples
    (default 30)
  - risk bins: DEFAULT_RISK_BINS (v4 split: 5,10,15,20,25,30,40,50,200)
  - session 09:30-16:00 ET, 30-minute time periods

The output JSON uses the same schema as the live strategy and copies
``meta.slippage``, ``meta.risk_rules``, ``meta.hard_gates`` and
``meta.hfoiv_gate`` verbatim from the source strategy file. Only the
identity fields, source dataset, cells and stats change.

Usage:
    python3 scripts/build_nq_strategy_from_window.py \
        --parquet logic/fvg_cache/fvg_results_5min_<hash>.parquet \
        --source-strategy bot/strategies/mixed-best-ev-v3-touch-moderate.json \
        --out bot/strategies/mixed-best-ev-wf-2020-2024.json \
        --strategy-id mixed-best-ev-wf-2020-2024 \
        --strategy-name "Mixed Best EV WF 2020-2024" \
        --source-dataset rr_nq_5min_2020_2024_30min
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from datetime import time as dtime

import numpy as np
import pandas as pd

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "logic"))

from utils.rr_analysis import (  # noqa: E402
    DEFAULT_RISK_BINS,
    N_VALUES,
    aggregate_rr_cells,
)

SESSION_START = dtime(9, 30)
SESSION_END = dtime(16, 0)
INTERVAL_MINUTES = 30
EXTREME_SETUPS = ("mit_extreme", "mid_extreme")


def newest_parquet() -> str:
    files = glob.glob(os.path.join(_ROOT, "logic", "fvg_cache", "fvg_results_5min_*.parquet"))
    if not files:
        raise SystemExit("No fvg_cache parquet found — run logic/main.py first")
    return max(files, key=os.path.getmtime)


def select_best_per_cell(cells, min_ev: float, min_samples: int) -> list[dict]:
    """For each (time, risk) cell, pick the best (setup, rr_target) by EV."""
    selected = []
    for cell in cells:
        best = None  # (ev, setup, rr_idx, win_rate, samples)
        for setup in EXTREME_SETUPS:
            sd = cell["setups"].get(setup) or {}
            evs = sd.get("evs") or []
            wrs = sd.get("win_rates") or []
            n_valid = sd.get("valid", 0) or 0
            if n_valid < min_samples or not evs:
                continue
            for i, ev in enumerate(evs):
                if ev is None:
                    continue
                if ev < min_ev:
                    continue
                if best is None or ev > best[0]:
                    best = (ev, setup, i, wrs[i], n_valid)
        if best is None:
            continue
        ev, setup, idx, wr, n = best
        selected.append(
            {
                "time_period": cell["time_period"],
                "risk_range": cell["risk_range"],
                "setup": setup,
                "rr_target": float(N_VALUES[idx]),
                "ev": round(float(ev), 4),
                "win_rate": round(float(wr), 2) if wr is not None else None,
                "samples": int(n),
                "trades_per_day": None,  # filled in later once we know n_days
                "enabled": True,
            }
        )
    return selected


def count_trading_days(parquet_path: str) -> int:
    """Count unique RTH session dates (09:30-16:00 ET) — drops overnight bars."""
    df = pd.read_parquet(parquet_path, columns=["time_candle1"])
    s = pd.to_datetime(df["time_candle1"], errors="coerce").dropna()
    if s.empty:
        return 0
    t = s.dt.time
    rth = s[(t >= SESSION_START) & (t < SESSION_END)]
    return int(rth.dt.normalize().nunique())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default=None, help="FVG parquet path (default: newest in fvg_cache)")
    p.add_argument("--source-strategy", default=os.path.join(_ROOT, "bot", "strategies", "mixed-best-ev-v3-touch-moderate.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--strategy-id", required=True)
    p.add_argument("--strategy-name", required=True)
    p.add_argument("--source-dataset", required=True)
    p.add_argument("--description", default="")
    p.add_argument("--min-ev", type=float, default=0.07)
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--max-trades-per-day", type=float, default=None,
                   help="If set, greedy-fill cells sorted by EV*sqrt(samples) until this budget is reached")
    args = p.parse_args()

    parquet = args.parquet or newest_parquet()
    print(f"[INFO] Using parquet: {parquet}")
    df = pd.read_parquet(parquet)
    print(f"[INFO] Loaded {len(df):,} FVG rows")

    n_days = count_trading_days(parquet)
    print(f"[INFO] Trading days in window: {n_days}")

    cells = aggregate_rr_cells(
        df_fvgs=df,
        fvg_filter_start_time=SESSION_START,
        fvg_filter_end_time=SESSION_END,
        interval_minutes=INTERVAL_MINUTES,
        risk_bins=DEFAULT_RISK_BINS,
        min_samples=5,
    )
    print(f"[INFO] Aggregated {len(cells)} raw cells")

    picked = select_best_per_cell(cells, min_ev=args.min_ev, min_samples=args.min_samples)
    print(f"[INFO] {len(picked)} cells passed (min_ev={args.min_ev}, min_samples={args.min_samples})")

    if n_days > 0:
        for c in picked:
            c["trades_per_day"] = round(c["samples"] / n_days, 4)

    # Optional greedy budget cap: sort by EV*sqrt(samples) desc and pick until
    # cumulative trades/day <= budget. Mirrors the live strategy's apparent
    # ≤5 trades/day cap (sum of live cells' trades_per_day = 4.9031).
    if args.max_trades_per_day is not None:
        budget = float(args.max_trades_per_day)
        ranked = sorted(picked, key=lambda c: c["ev"] * (c["samples"] ** 0.5), reverse=True)
        chosen = []
        used = 0.0
        for c in ranked:
            tpd = c["trades_per_day"] or 0
            if used + tpd > budget:
                continue
            chosen.append(c)
            used += tpd
        print(f"[INFO] Budget cap {budget}/day applied: {len(chosen)} cells, {used:.4f} trades/day")
        picked = chosen

    # Sort cells by (time_period, risk_range) for stable diffs
    def risk_key(rr: str) -> tuple[int, int]:
        a, b = rr.split("-")
        return int(a), int(b)
    picked.sort(key=lambda c: (c["time_period"], risk_key(c["risk_range"])))

    with open(args.source_strategy) as f:
        src = json.load(f)
    src_meta = src["meta"]

    total_samples = sum(c["samples"] for c in picked) or 1
    weighted_ev = round(sum(c["ev"] * c["samples"] for c in picked) / total_samples, 4)
    expected_tpd = round(sum(c["trades_per_day"] or 0 for c in picked), 4)
    time_cov = sorted({c["time_period"] for c in picked})

    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    out = {
        "schema_version": src.get("schema_version", "1.0"),
        "meta": {
            "name": args.strategy_name,
            "description": args.description or f"Walk-forward NQ strategy. Train window built from {os.path.basename(parquet)}. Selection: extreme-stop setups, EV>={args.min_ev}, samples>={args.min_samples}, best (setup,rr) per (time,risk).",
            "ticker": src_meta.get("ticker", "NQ"),
            "timeframe": src_meta.get("timeframe", "5min"),
            "source_dataset": args.source_dataset,
            "session_minutes": src_meta.get("session_minutes", 30),
            "slippage": src_meta["slippage"],
            "risk_rules": src_meta["risk_rules"],
            "id": args.strategy_id,
            "created_at": now_iso,
            "updated_at": now_iso,
            "hard_gates": src_meta["hard_gates"],
            "hfoiv_gate": src_meta["hfoiv_gate"],
        },
        "cells": picked,
        "stats": {
            "total_cells": len(picked),
            "enabled_cells": len(picked),
            "weighted_ev": weighted_ev,
            "expected_trades_per_day": expected_tpd,
            "time_coverage": time_cov,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] Wrote {args.out}")
    print(f"[OK] cells={len(picked)} weighted_ev={weighted_ev} trades/day={expected_tpd}")


if __name__ == "__main__":
    main()
