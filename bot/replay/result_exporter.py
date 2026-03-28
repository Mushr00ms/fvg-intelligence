"""
result_exporter.py — Export replay session results in backtester-compatible format.

Converts DailyState.closed_trades (list of OrderGroup) into the same JSON
structure produced by backtester.py:build_results_json(), enabling direct
trade-by-trade comparison via the comparator.
"""

import json
import os
from datetime import datetime


def export_session(daily_state, config, output_dir=None, displacement_log=None):
    """
    Export one replay session's trades in backtester-compatible JSON format.

    Args:
        daily_state: DailyState with closed_trades populated
        config: BotConfig (for strategy/balance metadata)
        output_dir: Where to save (default: bot/replay/results/)
        displacement_log: Optional margin displacement event log

    Returns:
        Path to the saved JSON file.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    trades = []
    for i, og in enumerate(daily_state.closed_trades, 1):
        entry = og.actual_entry_price or og.entry_price
        exit_price = og.actual_exit_price

        if og.side == "BUY":
            pnl_pts = round(exit_price - entry, 2) if exit_price else 0
        else:
            pnl_pts = round(entry - exit_price, 2) if exit_price else 0

        trades.append({
            "trade_id": i,
            "date": daily_state.date,
            "fvg_type": og.fvg_id.split("_")[0] if "_" in og.fvg_id else "",
            "time_period": "",  # Not stored on OrderGroup; comparator matches by entry price
            "risk_range": "",
            "setup": og.setup,
            "side": og.side,
            "entry_price": entry,
            "stop_price": og.stop_price,
            "target_price": og.target_price,
            "risk_pts": og.risk_pts,
            "n_value": og.n_value,
            "contracts": og.filled_qty or og.target_qty,
            "entry_time": og.filled_at or og.submitted_at or "",
            "exit_time": og.closed_at or "",
            "exit_price": exit_price,
            "exit_reason": og.close_reason,
            "pnl_pts": pnl_pts,
            "pnl_dollars": round(og.realized_pnl, 2),
            "is_win": og.close_reason == "TP",
            "group_id": og.group_id,
            "fvg_id": og.fvg_id,
        })

    result = {
        "meta": {
            "source": "replay",
            "replay_date": daily_state.date,
            "balance": daily_state.start_balance,
            "risk_pct": config.risk_per_trade,
            "replay_mode": True,
        },
        "summary": {
            "total_trades": len(trades),
            "wins": sum(1 for t in trades if t["is_win"]),
            "losses": sum(1 for t in trades if t["exit_reason"] == "SL"),
            "eod_exits": sum(1 for t in trades
                             if t["exit_reason"] in ("EOD", "FLATTEN")),
            "net_pnl": round(daily_state.realized_pnl, 2),
            "start_balance": daily_state.start_balance,
            "end_balance": round(
                daily_state.start_balance + daily_state.realized_pnl, 2),
            "kill_switch_hit": daily_state.kill_switch_active,
        },
        "trades": trades,
    }

    if displacement_log:
        result["displacement_log"] = displacement_log

    filename = f"replay_{daily_state.date.replace('-', '')}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Replay results exported: {filepath} ({len(trades)} trades)")
    return filepath
