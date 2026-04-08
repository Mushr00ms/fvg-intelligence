#!/usr/bin/env python3
"""
simulate_btc_strategy_capital_cap.py - Capital-cap replay for BTC strategy JSONs.

Supports both:
- flat trade files (list of trade dicts)
- labeled sweep archives (dict[label -> list[trade dict]])

Sizing mirrors the runtime more closely than the older capital-cap sweep:
- quantity = risk_dollar / per_unit_loss
- per_unit_loss = stop distance + entry maker fee + stop taker fee
- notional cap = current balance * leverage
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ENTRY_FEE = 0.0
TP_FEE = 0.0
SL_FEE = 0.0004
MAX_HOLD_MIN = 240
RISK_BINS = [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 994]


def parse_epoch(ts: str) -> float:
    value = ts.replace("T", " ")
    if "+" in value[10:]:
        value = value.split("+")[0]
    elif value.endswith("Z"):
        value = value[:-1]
    if "." in value:
        value = value.split(".")[0]
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timestamp()


def load_strategy(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def risk_to_range(risk_bps: float) -> str | None:
    for idx in range(len(RISK_BINS) - 1):
        if RISK_BINS[idx] <= risk_bps < RISK_BINS[idx + 1]:
            return f"{RISK_BINS[idx]}-{RISK_BINS[idx + 1]}"
    return None


def load_trade_source(path: Path):
    with open(path) as f:
        return json.load(f)


def pick_trades(payload, config_label: str | None) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported trade payload type: {type(payload).__name__}")
    if config_label is None:
        raise ValueError("Trade payload is label-keyed; --config-label is required")
    if config_label not in payload:
        raise KeyError(f"Unknown config label: {config_label}")
    return payload[config_label]


def build_lookup(strategy: dict) -> dict:
    lookup = {}
    for cell in strategy.get("cells", []):
        if not cell.get("enabled", True):
            continue
        lookup[(cell["time_period"], cell["risk_range"], cell["setup"])] = cell["best_n"]
    return lookup


def prepare_events(trades: list[dict], lookup: dict, year_filter: str | None) -> list[dict]:
    events = []
    for trade in trades:
        ts = trade.get("mitigation_time") or trade["formation_time"]
        year = ts[:4]
        if year_filter and year != year_filter:
            continue

        risk_range = trade.get("risk_range")
        if risk_range is None:
            risk_range = risk_to_range(float(trade["risk_bps"]))
        key = (trade["time_period"], risk_range, trade["setup"]) if risk_range else None
        if key is None:
            continue
        best_n = lookup.get(key)
        if best_n is None:
            continue

        won = trade["outcomes"].get(str(best_n))
        if won is None:
            continue

        entry_price = float(trade["entry_price"])
        stop_price = float(trade["stop_price"])
        per_unit_loss = abs(entry_price - stop_price) + (entry_price * ENTRY_FEE) + (stop_price * SL_FEE)
        if per_unit_loss <= 0:
            continue

        events.append(
            {
                "epoch": parse_epoch(ts),
                "day": ts[:10],
                "month": ts[:7],
                "year": year,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "per_unit_loss": per_unit_loss,
                "best_n": float(best_n),
                "won": bool(won),
            }
        )
    events.sort(key=lambda item: item["epoch"])
    return events


def replay(
    events: list[dict],
    *,
    start_balance: float,
    risk_pct: float,
    leverage: float,
    compound: bool,
) -> dict:
    max_hold_seconds = MAX_HOLD_MIN * 60
    balance = start_balance
    peak = balance
    max_dd_pct = 0.0
    wins = 0
    losses = 0
    rejected = 0
    monthly_pnl = defaultdict(float)
    daily_pnl = defaultdict(float)
    yearly_pnl = defaultdict(float)
    yearly_dd = defaultdict(float)
    yearly_peak = {}
    last_year = None
    open_positions = []  # (release_epoch, notional)

    for event in events:
        if open_positions:
            open_positions = [pos for pos in open_positions if pos[0] > event["epoch"]]

        if balance <= 0:
            break

        risk_base = balance if compound else start_balance
        risk_dollar = risk_base * risk_pct
        quantity = risk_dollar / event["per_unit_loss"]
        notional = quantity * event["entry_price"]

        open_notional = sum(pos[1] for pos in open_positions)
        if open_notional + notional > balance * leverage:
            rejected += 1
            continue

        if event["won"]:
            pnl = risk_dollar * event["best_n"] - (notional * (ENTRY_FEE + TP_FEE))
            wins += 1
        else:
            pnl = -risk_dollar - (quantity * event["entry_price"] * ENTRY_FEE) - (quantity * event["stop_price"] * SL_FEE)
            losses += 1

        balance += pnl
        monthly_pnl[event["month"]] += pnl
        daily_pnl[event["day"]] += pnl
        yearly_pnl[event["year"]] += pnl

        if event["year"] != last_year:
            yearly_peak[event["year"]] = balance - pnl
            last_year = event["year"]
        if balance > yearly_peak[event["year"]]:
            yearly_peak[event["year"]] = balance

        current_year_dd = (yearly_peak[event["year"]] - balance) / yearly_peak[event["year"]] * 100
        yearly_dd[event["year"]] = max(yearly_dd[event["year"]], current_year_dd)

        if balance > peak:
            peak = balance
        max_dd_pct = max(max_dd_pct, (peak - balance) / peak * 100 if peak > 0 else 0.0)

        open_positions.append((event["epoch"] + max_hold_seconds, notional))

    total_trades = wins + losses
    monthly_arr = np.array([monthly_pnl[m] for m in sorted(monthly_pnl)]) if monthly_pnl else np.array([])
    daily_arr = np.array(list(daily_pnl.values())) if daily_pnl else np.array([])
    sharpe = 0.0
    if len(daily_arr) > 1 and daily_arr.std() > 0:
        sharpe = float((daily_arr.mean() / daily_arr.std()) * math.sqrt(365))

    return {
        "start_balance": start_balance,
        "risk_pct": risk_pct,
        "leverage": leverage,
        "compound": compound,
        "final_balance": round(balance, 2),
        "net_pnl": round(balance - start_balance, 2),
        "ret_pct": round((balance - start_balance) / start_balance * 100, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total_trades * 100, 1) if total_trades else 0.0,
        "rejected": rejected,
        "months": len(monthly_arr),
        "monthly_mean": round(float(monthly_arr.mean()), 2) if len(monthly_arr) else 0.0,
        "monthly_std": round(float(monthly_arr.std()), 2) if len(monthly_arr) else 0.0,
        "monthly_min": round(float(monthly_arr.min()), 2) if len(monthly_arr) else 0.0,
        "monthly_max": round(float(monthly_arr.max()), 2) if len(monthly_arr) else 0.0,
        "positive_months": int(sum(1 for value in monthly_arr if value > 0)),
        "sharpe": round(sharpe, 3),
        "yearly_pnl": {year: round(value, 2) for year, value in yearly_pnl.items()},
        "yearly_dd": {year: round(value, 2) for year, value in yearly_dd.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy-path", required=True)
    parser.add_argument("--trades-path", required=True)
    parser.add_argument("--config-label", default=None, help="Required when trades file is label-keyed")
    parser.add_argument("--year", default=None, help="Optional year filter, e.g. 2025")
    parser.add_argument("--start-balance", type=float, default=50_000.0)
    parser.add_argument("--risk-pct", type=float, default=0.001)
    parser.add_argument("--risk-pcts", default=None, help="Comma-separated risk levels to sweep, e.g. 0.0007,0.001,0.0015")
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--compound", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    strategy = load_strategy(Path(args.strategy_path))
    trade_payload = load_trade_source(Path(args.trades_path))
    config_label = args.config_label or strategy.get("meta", {}).get("config_label")
    trades = pick_trades(trade_payload, config_label)
    lookup = build_lookup(strategy)
    events = prepare_events(trades, lookup, args.year)

    if not events:
        raise RuntimeError("No qualifying events found for the selected strategy / trades / year")

    risk_levels = [args.risk_pct]
    if args.risk_pcts:
        risk_levels = [float(item.strip()) for item in args.risk_pcts.split(",") if item.strip()]

    results = [
        replay(
            events,
            start_balance=args.start_balance,
            risk_pct=risk_pct,
            leverage=args.leverage,
            compound=args.compound,
        )
        for risk_pct in risk_levels
    ]

    payload = {
        "strategy_path": str(Path(args.strategy_path)),
        "trades_path": str(Path(args.trades_path)),
        "config_label": config_label,
        "year": args.year,
        "event_count": len(events),
        "results": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"saved replay results to {out_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
