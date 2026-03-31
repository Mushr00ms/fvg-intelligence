"""
scenario_selector.py — Analyze backtest results to pick high-value replay dates.

Instead of brute-forcing every day through TWS replay, this module identifies
dates that stress specific bot subsystems. Each scenario targets a code path
where backtester and engine could diverge.

Scenario 8 (margin displacement) is standalone validation — not compared
against the backtester, which has no equivalent logic.
"""

import json
from collections import defaultdict
from datetime import datetime, time


# ── Scenario detection functions ─────────────────────────────────────────

def _detect_kill_switch(trades_by_day, daily_pnl, start_balance):
    """Days where cumulative intraday loss crossed -10% of day-start balance."""
    dates = []
    running = start_balance
    for day in sorted(trades_by_day.keys()):
        day_trades = trades_by_day[day]
        day_start = running
        cumulative = 0.0
        halt_trade = 0
        for t in day_trades:
            cumulative += t["pnl_dollars"]
            halt_trade += 1
            if day_start > 0 and cumulative <= -(day_start * 0.10):
                dates.append((day, f"halt after {halt_trade} trades, "
                              f"daily P&L ${cumulative:,.0f}"))
                break
        running += sum(t["pnl_dollars"] for t in day_trades)
    return dates


def _detect_dd_scaling(trades_by_day):
    """Days with drawdown-scaled trades."""
    dates = []
    for day, trades in trades_by_day.items():
        dd_trades = [t for t in trades if t.get("dd_note") and "DD scale" in t["dd_note"]]
        if dd_trades:
            notes = dd_trades[0]["dd_note"]
            dates.append((day, f"{len(dd_trades)} scaled trades ({notes})"))
    return dates


def _detect_eod_flatten(trades_by_day):
    """Days with positions flattened at session end."""
    dates = []
    for day, trades in trades_by_day.items():
        eod = [t for t in trades if t.get("exit_reason") == "EOD"]
        if eod:
            dates.append((day, f"{len(eod)} EOD exit(s)"))
    return dates


def _detect_near_cutoff(trades_by_day):
    """Days with entries accepted between 15:00-15:44 ET (near the gate)."""
    dates = []
    for day, trades in trades_by_day.items():
        late = []
        for t in trades:
            entry_time = t.get("entry_time", "")
            if not entry_time:
                continue
            try:
                et = datetime.fromisoformat(str(entry_time).replace("Z", "+00:00"))
                if et.hour == 15 and et.minute < 45:
                    late.append(t)
            except (ValueError, TypeError):
                # Try parsing HH:MM from string
                if "15:" in str(entry_time):
                    parts = str(entry_time).split("15:")
                    if len(parts) > 1:
                        try:
                            minute = int(parts[1][:2])
                            if minute < 45:
                                late.append(t)
                        except ValueError:
                            pass
        if late:
            dates.append((day, f"{len(late)} entries after 15:00"))
    return dates


def _detect_max_concurrent(trades_by_day):
    """Days where 3+ trades had overlapping entry→exit windows."""
    dates = []
    for day, trades in trades_by_day.items():
        # Build intervals from entry_time → exit_time
        intervals = []
        for t in trades:
            entry = t.get("entry_time", "")
            exit_ = t.get("exit_time", "")
            if entry and exit_:
                intervals.append((str(entry), str(exit_)))
        if len(intervals) < 3:
            continue
        # Sweep line: count max overlap
        events = []
        for start, end in intervals:
            events.append((start, 1))
            events.append((end, -1))
        events.sort()
        concurrent = 0
        max_concurrent = 0
        for _, delta in events:
            concurrent += delta
            max_concurrent = max(max_concurrent, concurrent)
        if max_concurrent >= 3:
            dates.append((day, f"max {max_concurrent} concurrent positions"))
    return dates


def _detect_high_trade(trades_by_day):
    """Days with 8+ trades."""
    dates = []
    for day, trades in trades_by_day.items():
        if len(trades) >= 8:
            dates.append((day, f"{len(trades)} trades"))
    return dates


def _detect_margin_constrained(trades_by_day):
    """Days with margin-capped trades."""
    dates = []
    for day, trades in trades_by_day.items():
        capped = [t for t in trades if t.get("dd_note") and "margin cap" in t["dd_note"]]
        if capped:
            dates.append((day, f"{len(capped)} margin-capped"))
    return dates


def _detect_margin_displacement(trades_by_day):
    """Days with margin pressure AND near-simultaneous entries (displacement candidates).

    These are days where the engine's MarginPriorityManager would potentially
    suspend a far order to take a nearer one. The backtester has no equivalent.
    """
    dates = []
    for day, trades in trades_by_day.items():
        has_margin = any(t.get("dd_note") and "margin cap" in t["dd_note"]
                         for t in trades)
        if not has_margin or len(trades) < 2:
            continue
        # Check for entries within 30 minutes of each other
        entry_times = []
        for t in trades:
            entry = t.get("entry_time", "")
            if entry:
                try:
                    et = datetime.fromisoformat(str(entry).replace("Z", "+00:00"))
                    entry_times.append(et)
                except (ValueError, TypeError):
                    pass
        entry_times.sort()
        has_close_entries = False
        for i in range(len(entry_times) - 1):
            delta = (entry_times[i + 1] - entry_times[i]).total_seconds()
            if delta <= 1800:  # 30 minutes
                has_close_entries = True
                break
        if has_close_entries:
            dates.append((day, f"{len(trades)} trades with margin pressure, "
                          f"close entry timing"))
    return dates


def _detect_large_fvg(trades_by_day):
    """Days with large FVG / wide stop trades."""
    dates = []
    for day, trades in trades_by_day.items():
        large = [t for t in trades
                 if t.get("risk_range") in ("40-50", "50-200") or (t.get("risk_pts") or 0) > 30]
        if large:
            max_risk = max(t.get("risk_pts", 0) for t in large)
            dates.append((day, f"{len(large)} large FVG trades "
                          f"(max risk {max_risk:.1f} pts)"))
    return dates


def _detect_quiet_days(trades_by_day, all_trading_days):
    """Days with 0 trades (validates correct filtering)."""
    trade_days = set(trades_by_day.keys())
    quiet = [d for d in all_trading_days if d not in trade_days]
    return [(d, "no trades") for d in quiet]


def _detect_mixed_exits(trades_by_day):
    """Days with TP + SL + EOD exits on the same day."""
    dates = []
    for day, trades in trades_by_day.items():
        reasons = set(t.get("exit_reason", "") for t in trades)
        if "TP" in reasons and "SL" in reasons and "EOD" in reasons:
            dates.append((day, f"TP + SL + EOD ({len(trades)} trades)"))
    return dates


# ── Scenario registry ────────────────────────────────────────────────────

SCENARIOS = [
    ("Kill switch", _detect_kill_switch, True),       # needs extra args
    ("DD scaling", _detect_dd_scaling, False),
    ("EOD flatten", _detect_eod_flatten, False),
    ("Near-cutoff entries", _detect_near_cutoff, False),
    ("Max concurrent", _detect_max_concurrent, False),
    ("High-trade day", _detect_high_trade, False),
    ("Margin-constrained", _detect_margin_constrained, False),
    ("Margin displacement ⚡", _detect_margin_displacement, False),
    ("Large FVG", _detect_large_fvg, False),
    ("Quiet day", None, True),                        # needs all_trading_days
    ("Mixed exits", _detect_mixed_exits, False),
]


# ── Main selector ────────────────────────────────────────────────────────

def select_replay_dates(backtest_json, max_per_scenario=3, max_total=30):
    """
    Analyze backtest results and pick high-value dates for replay.

    Args:
        backtest_json: dict from backtester's build_results_json()
        max_per_scenario: max dates to pick per scenario
        max_total: overall cap on dates

    Returns:
        list of (date, scenario_tags, reason) tuples
    """
    trades = backtest_json.get("trades", [])
    if not trades:
        return []

    # Group trades by date
    trades_by_day = defaultdict(list)
    for t in trades:
        trades_by_day[t["date"]].append(t)

    all_days = sorted(trades_by_day.keys())
    start_balance = backtest_json.get("meta", {}).get("balance", 76000)

    # Run each detector
    scenario_dates = {}
    for name, detector, needs_extra in SCENARIOS:
        if name == "Kill switch":
            daily_pnl = backtest_json.get("daily_pnl", [])
            hits = _detect_kill_switch(trades_by_day, daily_pnl, start_balance)
        elif name == "Quiet day":
            hits = _detect_quiet_days(trades_by_day, all_days)
        else:
            hits = detector(trades_by_day)
        scenario_dates[name] = hits

    # Select dates: prioritize scenarios, deduplicate
    selected = []  # (date, [tags], reason)
    seen_dates = set()

    # Priority order (most valuable first)
    priority = [
        "Kill switch",
        "Margin displacement ⚡",
        "DD scaling",
        "EOD flatten",
        "Max concurrent",
        "Near-cutoff entries",
        "High-trade day",
        "Margin-constrained",
        "Large FVG",
        "Mixed exits",
        "Quiet day",
    ]

    for scenario_name in priority:
        hits = scenario_dates.get(scenario_name, [])
        added = 0
        for date, reason in hits:
            if len(selected) >= max_total:
                break
            if date in seen_dates:
                # Add tag to existing entry
                for s in selected:
                    if s[0] == date:
                        s[1].append(scenario_name)
                        break
                continue
            if added >= max_per_scenario:
                continue
            selected.append((date, [scenario_name], reason))
            seen_dates.add(date)
            added += 1

    selected.sort(key=lambda x: x[0])
    return selected


def print_replay_plan(selected_dates):
    """Print human-readable replay schedule grouped by scenario."""
    if not selected_dates:
        print("No replay dates selected (backtest had no trades?).")
        return

    print(f"\n{'=' * 60}")
    print(f"  REPLAY PLAN: {len(selected_dates)} dates")
    print(f"{'=' * 60}\n")

    # Group by primary scenario
    by_scenario = defaultdict(list)
    for date, tags, reason in selected_dates:
        by_scenario[tags[0]].append((date, tags, reason))

    for scenario, entries in by_scenario.items():
        marker = " ⚡ standalone validation" if "⚡" in scenario else ""
        print(f"  {scenario} ({len(entries)} dates){marker}:")
        for date, tags, reason in entries:
            extra_tags = f" [+{', '.join(tags[1:])}]" if len(tags) > 1 else ""
            print(f"    {date}  {reason}{extra_tags}")
        print()

    print(f"  Set TWS to replay each date and run:")
    print(f"    python -m bot.replay --strategy <name> --balance <N>")
    print(f"{'=' * 60}\n")


def load_and_select(backtest_path, max_per_scenario=3, max_total=30):
    """Load backtest JSON and return selected replay dates."""
    with open(backtest_path) as f:
        data = json.load(f)
    return select_replay_dates(data, max_per_scenario, max_total)
