"""
comparator.py — Diff replay results vs backtest results, trade by trade.

Categories:
  MATCH          — same trade, same outcome, P&L within tolerance
  DIFF           — same trade, different outcome (exit reason/price/contracts)
  ONLY_IN_REPLAY — engine took a trade the backtester didn't
  ONLY_IN_BT     — backtester took a trade the engine didn't

Margin displacement events (scenario 8) are reported separately as
standalone validation — not compared against the backtester.
"""

import json
import os
from collections import defaultdict


PNL_TOLERANCE = 1.50  # $ tolerance for commission rounding differences


def _trade_key(trade):
    """Composite key to match the same trade across replay and backtest."""
    return (
        trade.get("date", ""),
        trade.get("side", ""),
        round(trade.get("entry_price", 0), 2),
        trade.get("setup", ""),
    )


def compare_day(replay_trades, backtest_trades):
    """
    Compare replay trades vs backtest trades for one day.

    Returns:
        dict with keys: matches, diffs, only_replay, only_bt, details
    """
    replay_by_key = {}
    for t in replay_trades:
        key = _trade_key(t)
        replay_by_key[key] = t

    bt_by_key = {}
    for t in backtest_trades:
        key = _trade_key(t)
        bt_by_key[key] = t

    matches = []
    diffs = []
    only_replay = []
    only_bt = []

    all_keys = set(replay_by_key.keys()) | set(bt_by_key.keys())

    for key in sorted(all_keys):
        rt = replay_by_key.get(key)
        bt = bt_by_key.get(key)

        if rt and bt:
            # Both have this trade — compare outcome
            same_reason = rt.get("exit_reason") == bt.get("exit_reason")
            same_exit = (round(rt.get("exit_price", 0), 2) ==
                         round(bt.get("exit_price", 0), 2))
            same_pnl = abs((rt.get("pnl_dollars", 0) or 0) -
                           (bt.get("pnl_dollars", 0) or 0)) <= PNL_TOLERANCE
            same_qty = (rt.get("contracts", 0) == bt.get("contracts", 0))

            if same_reason and same_pnl:
                matches.append({"key": key, "replay": rt, "backtest": bt})
            else:
                detail = {
                    "key": key,
                    "replay": rt,
                    "backtest": bt,
                    "diff_fields": [],
                }
                if not same_reason:
                    detail["diff_fields"].append(
                        f"exit_reason: {rt.get('exit_reason')} vs {bt.get('exit_reason')}")
                if not same_exit:
                    detail["diff_fields"].append(
                        f"exit_price: {rt.get('exit_price')} vs {bt.get('exit_price')}")
                if not same_pnl:
                    detail["diff_fields"].append(
                        f"pnl: ${rt.get('pnl_dollars', 0):.2f} vs ${bt.get('pnl_dollars', 0):.2f}")
                if not same_qty:
                    detail["diff_fields"].append(
                        f"contracts: {rt.get('contracts')} vs {bt.get('contracts')}")
                diffs.append(detail)
        elif rt:
            only_replay.append({"key": key, "trade": rt})
        else:
            only_bt.append({"key": key, "trade": bt})

    return {
        "matches": matches,
        "diffs": diffs,
        "only_replay": only_replay,
        "only_bt": only_bt,
    }


def compare_results(replay_data, backtest_data, target_date=None):
    """
    Compare replay results with backtest results.

    Args:
        replay_data: dict from result_exporter (single day)
        backtest_data: dict from backtester's build_results_json() (may be multi-day)
        target_date: optional date filter for multi-day backtest

    Returns:
        dict with comparison results + displacement validation
    """
    replay_trades = replay_data.get("trades", [])
    replay_date = replay_data.get("meta", {}).get("replay_date", "")

    # Filter backtest trades to target date
    bt_trades = backtest_data.get("trades", [])
    if target_date:
        bt_trades = [t for t in bt_trades if t.get("date") == target_date]
    elif replay_date:
        bt_trades = [t for t in bt_trades if t.get("date") == replay_date]

    comparison = compare_day(replay_trades, bt_trades)
    comparison["date"] = replay_date or target_date or ""
    comparison["replay_count"] = len(replay_trades)
    comparison["backtest_count"] = len(bt_trades)

    # Displacement validation (standalone, not compared with backtester)
    displacement = replay_data.get("displacement_log", [])
    comparison["displacement_log"] = displacement

    return comparison


def print_comparison(result):
    """Print human-readable comparison report."""
    date = result.get("date", "?")
    print(f"\n{'=' * 60}")
    print(f"  Replay vs Backtest: {date}")
    print(f"  Replay: {result['replay_count']} trades | "
          f"Backtest: {result['backtest_count']} trades")
    print(f"{'=' * 60}\n")

    # Matches
    for m in result["matches"]:
        rt = m["replay"]
        pnl = rt.get("pnl_dollars", 0)
        sign = "+" if pnl >= 0 else ""
        print(f"  MATCH  {rt['side']:<4} {rt['entry_price']:<10.2f} "
              f"{rt.get('setup', ''):<14} "
              f"{rt.get('exit_reason', ''):<3} {sign}${pnl:.0f}")

    # Diffs
    for d in result["diffs"]:
        rt = d["replay"]
        bt = d["backtest"]
        print(f"  DIFF   {rt['side']:<4} {rt['entry_price']:<10.2f} "
              f"{rt.get('setup', ''):<14} "
              f"replay:{rt.get('exit_reason', '')} bt:{bt.get('exit_reason', '')}")
        for field in d["diff_fields"]:
            print(f"         ↳ {field}")

    # Only in replay
    for o in result["only_replay"]:
        t = o["trade"]
        print(f"  ONLY_REPLAY  {t['side']:<4} {t['entry_price']:<10.2f} "
              f"{t.get('setup', ''):<14} {t.get('exit_reason', '')}")

    # Only in backtest
    for o in result["only_bt"]:
        t = o["trade"]
        print(f"  ONLY_BT      {t['side']:<4} {t['entry_price']:<10.2f} "
              f"{t.get('setup', ''):<14} {t.get('exit_reason', '')}")

    # Displacement validation
    disp = result.get("displacement_log", [])
    if disp:
        print(f"\n  MARGIN DISPLACEMENT ⚡ ({len(disp)} events):")
        for event in disp:
            action = event.get("action", "")
            if action == "suspend":
                print(f"    {event['time']}  SUSPEND {event['suspended_id']} "
                      f"(entry {event['suspended_entry']}) "
                      f"for {event['for_id']} (entry {event['for_entry']})")
                print(f"      ↳ {event['reason']}")
            elif action == "suspend_new":
                print(f"    {event['time']}  SUSPEND_NEW {event['suspended_id']} "
                      f"(entry {event['suspended_entry']})")
                print(f"      ↳ {event['reason']}")
            elif action == "reactivate":
                print(f"    {event['time']}  REACTIVATE {event['group_id']} "
                      f"(entry {event['entry']})")
            elif action == "reactivate_blocked":
                print(f"    {event['time']}  REACTIVATE_BLOCKED {event['group_id']}")
                print(f"      ↳ {event['reason']}")

    # Summary
    total = (len(result["matches"]) + len(result["diffs"]) +
             len(result["only_replay"]) + len(result["only_bt"]))
    match_rate = (len(result["matches"]) / total * 100) if total else 0

    print(f"\n  Summary: {len(result['matches'])} match, "
          f"{len(result['diffs'])} diff, "
          f"{len(result['only_replay'])} replay-only, "
          f"{len(result['only_bt'])} bt-only"
          f" ({match_rate:.0f}% match rate)")
    if disp:
        print(f"  Displacement: {len(disp)} events logged")
    print()


def compare_directory(replay_dir, backtest_path):
    """Compare all replay results in a directory against a backtest file."""
    with open(backtest_path) as f:
        bt_data = json.load(f)

    results = []
    for fname in sorted(os.listdir(replay_dir)):
        if not fname.startswith("replay_") or not fname.endswith(".json"):
            continue
        filepath = os.path.join(replay_dir, fname)
        with open(filepath) as f:
            replay_data = json.load(f)
        result = compare_results(replay_data, bt_data)
        results.append(result)
        print_comparison(result)

    if results:
        total_match = sum(len(r["matches"]) for r in results)
        total_diff = sum(len(r["diffs"]) for r in results)
        total_only_r = sum(len(r["only_replay"]) for r in results)
        total_only_bt = sum(len(r["only_bt"]) for r in results)
        total_disp = sum(len(r.get("displacement_log", [])) for r in results)
        total = total_match + total_diff + total_only_r + total_only_bt

        print(f"{'=' * 60}")
        print(f"  OVERALL: {len(results)} days replayed")
        print(f"  {total_match}/{total} trades match "
              f"({total_match / total * 100:.1f}%)" if total else "  No trades")
        if total_diff:
            print(f"  {total_diff} diffs")
        if total_only_r:
            print(f"  {total_only_r} replay-only")
        if total_only_bt:
            print(f"  {total_only_bt} backtest-only")
        if total_disp:
            print(f"  {total_disp} displacement events validated")
        print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare replay vs backtest results")
    parser.add_argument("--replay", required=True,
                        help="Path to replay results file or directory")
    parser.add_argument("--backtest", required=True,
                        help="Path to backtest results JSON")
    parser.add_argument("--date", default=None,
                        help="Filter backtest to specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    if os.path.isdir(args.replay):
        compare_directory(args.replay, args.backtest)
    else:
        with open(args.replay) as f:
            replay = json.load(f)
        with open(args.backtest) as f:
            backtest = json.load(f)
        result = compare_results(replay, backtest, target_date=args.date)
        print_comparison(result)
