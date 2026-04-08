#!/usr/bin/env python3
"""
analyze_btc_marginal_cells.py - Rank additive BTC cells under a capital cap.

Compares a base strategy against a broader candidate strategy on the same trade
archive, then measures each extra cell by its incremental PnL after the
leverage cap and overlap effects are applied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.simulate_btc_strategy_capital_cap import (  # noqa: E402
    ENTRY_FEE,
    MAX_HOLD_MIN,
    SL_FEE,
    build_lookup,
    load_strategy,
    load_trade_source,
    parse_epoch,
    pick_trades,
    risk_to_range,
)


def prepare_events(trades: list[dict], lookup: dict, year_filter: str | None) -> list[dict]:
    events = []
    for trade in trades:
        ts = trade.get("mitigation_time") or trade["formation_time"]
        if year_filter and ts[:4] != year_filter:
            continue

        risk_range = trade.get("risk_range")
        if risk_range is None:
            risk_range = risk_to_range(float(trade["risk_bps"]))
        if risk_range is None:
            continue

        cell_key = (trade["time_period"], risk_range, trade["setup"])
        best_n = lookup.get(cell_key)
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
                "cell_key": cell_key,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "per_unit_loss": per_unit_loss,
                "best_n": float(best_n),
                "won": bool(won),
            }
        )
    events.sort(key=lambda event: event["epoch"])
    return events


def replay_subset(
    events: list[dict],
    allowed_cells: set[tuple[str, str, str]],
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
    open_positions = []

    for event in events:
        if event["cell_key"] not in allowed_cells:
            continue

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
            pnl = risk_dollar * event["best_n"] - (notional * ENTRY_FEE)
            wins += 1
        else:
            pnl = -risk_dollar - (quantity * event["entry_price"] * ENTRY_FEE) - (quantity * event["stop_price"] * SL_FEE)
            losses += 1

        balance += pnl
        if balance > peak:
            peak = balance
        if peak > 0:
            max_dd_pct = max(max_dd_pct, (peak - balance) / peak * 100)

        open_positions.append((event["epoch"] + max_hold_seconds, notional))

    trades = wins + losses
    return {
        "net_pnl": round(balance - start_balance, 2),
        "final_balance": round(balance, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / trades * 100, 1) if trades else 0.0,
        "rejected": rejected,
    }


def format_cell(cell_key: tuple[str, str, str]) -> str:
    time_period, risk_range, setup = cell_key
    return f"{time_period} | {risk_range} | {setup}"


def analyze(
    events: list[dict],
    base_cells: set[tuple[str, str, str]],
    candidate_cells: set[tuple[str, str, str]],
    *,
    start_balance: float,
    risk_pct: float,
    leverage: float,
    compound: bool,
    greedy_steps: int,
) -> dict:
    baseline = replay_subset(
        events,
        base_cells,
        start_balance=start_balance,
        risk_pct=risk_pct,
        leverage=leverage,
        compound=compound,
    )

    extras = sorted(candidate_cells - base_cells)
    marginal_rows = []
    for cell_key in extras:
        result = replay_subset(
            events,
            base_cells | {cell_key},
            start_balance=start_balance,
            risk_pct=risk_pct,
            leverage=leverage,
            compound=compound,
        )
        marginal_rows.append(
            {
                "cell": format_cell(cell_key),
                "cell_key": list(cell_key),
                "net_pnl": result["net_pnl"],
                "max_dd_pct": result["max_dd_pct"],
                "trades": result["trades"],
                "rejected": result["rejected"],
                "delta_net_pnl": round(result["net_pnl"] - baseline["net_pnl"], 2),
                "delta_dd_pct": round(result["max_dd_pct"] - baseline["max_dd_pct"], 2),
                "delta_trades": result["trades"] - baseline["trades"],
                "delta_rejected": result["rejected"] - baseline["rejected"],
            }
        )

    marginal_rows.sort(key=lambda row: (row["delta_net_pnl"], -row["delta_dd_pct"]), reverse=True)

    greedy_rows = []
    chosen = set(base_cells)
    remaining = set(extras)
    current = baseline
    while remaining and len(greedy_rows) < greedy_steps:
        best_cell = None
        best_result = None
        best_delta = None
        for cell_key in sorted(remaining):
            result = replay_subset(
                events,
                chosen | {cell_key},
                start_balance=start_balance,
                risk_pct=risk_pct,
                leverage=leverage,
                compound=compound,
            )
            delta = result["net_pnl"] - current["net_pnl"]
            if best_delta is None or delta > best_delta:
                best_delta = delta
                best_cell = cell_key
                best_result = result

        if best_cell is None or best_result is None or best_delta is None or best_delta <= 0:
            break

        greedy_rows.append(
            {
                "step": len(greedy_rows) + 1,
                "cell": format_cell(best_cell),
                "cell_key": list(best_cell),
                "net_pnl": best_result["net_pnl"],
                "max_dd_pct": best_result["max_dd_pct"],
                "trades": best_result["trades"],
                "rejected": best_result["rejected"],
                "delta_net_pnl": round(best_result["net_pnl"] - current["net_pnl"], 2),
                "delta_dd_pct": round(best_result["max_dd_pct"] - current["max_dd_pct"], 2),
                "delta_trades": best_result["trades"] - current["trades"],
                "delta_rejected": best_result["rejected"] - current["rejected"],
            }
        )
        chosen.add(best_cell)
        remaining.remove(best_cell)
        current = best_result

    return {
        "baseline": baseline,
        "extra_cell_count": len(extras),
        "marginal": marginal_rows,
        "greedy_path": greedy_rows,
        "greedy_final": current,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-strategy", required=True)
    parser.add_argument("--candidate-strategy", required=True)
    parser.add_argument("--trades-path", required=True)
    parser.add_argument("--year", default=None)
    parser.add_argument("--start-balance", type=float, default=50_000.0)
    parser.add_argument("--risk-pct", type=float, required=True)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--compound", action="store_true")
    parser.add_argument("--base-config-label", default=None)
    parser.add_argument("--candidate-config-label", default=None)
    parser.add_argument("--greedy-steps", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_strategy = load_strategy(Path(args.base_strategy))
    candidate_strategy = load_strategy(Path(args.candidate_strategy))
    trade_payload = load_trade_source(Path(args.trades_path))

    base_trades = pick_trades(
        trade_payload,
        args.base_config_label or base_strategy.get("meta", {}).get("config_label"),
    )
    candidate_trades = pick_trades(
        trade_payload,
        args.candidate_config_label or candidate_strategy.get("meta", {}).get("config_label"),
    )

    if base_trades is not candidate_trades and base_trades != candidate_trades:
        raise ValueError("Base and candidate strategies resolved to different trade sets")

    base_lookup = build_lookup(base_strategy)
    candidate_lookup = build_lookup(candidate_strategy)
    events = prepare_events(base_trades, candidate_lookup, args.year)

    analysis = analyze(
        events,
        set(base_lookup.keys()),
        set(candidate_lookup.keys()),
        start_balance=args.start_balance,
        risk_pct=args.risk_pct,
        leverage=args.leverage,
        compound=args.compound,
        greedy_steps=args.greedy_steps,
    )

    payload = {
        "meta": {
            "base_strategy": str(Path(args.base_strategy).resolve()),
            "candidate_strategy": str(Path(args.candidate_strategy).resolve()),
            "trades_path": str(Path(args.trades_path).resolve()),
            "year": args.year,
            "start_balance": args.start_balance,
            "risk_pct": args.risk_pct,
            "leverage": args.leverage,
            "compound": args.compound,
        },
        **analysis,
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)

    print("baseline", analysis["baseline"])
    print("extra_cell_count", analysis["extra_cell_count"])
    print("top_marginal")
    for row in analysis["marginal"][:10]:
        print(row)
    print("greedy_path")
    for row in analysis["greedy_path"]:
        print(row)
    print("greedy_final", analysis["greedy_final"])


if __name__ == "__main__":
    main()
