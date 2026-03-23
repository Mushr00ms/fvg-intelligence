"""
backtester.py — Strategy backtester using 1-second IB historical data.

Replays historical data bar-by-bar:
    1. Constructs 5-min bars on the fly for FVG detection
    2. Uses 1-second bars for precise mitigation detection
    3. Uses 1-second bars for exact fill simulation (stop vs target ordering)
    4. Matches detected FVGs to strategy cells
    5. Tracks position sizing, P&L, and risk gates

Usage:
    python3 bot/backtest/backtester.py --strategy morning-momentum-v3
    python3 bot/backtest/backtester.py --strategy-file path/to/strategy.json
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional

import pandas as pd

# Add project root
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from bot.strategy.fvg_detector import check_fvg_3bars, SESSION_INTERVALS, _assign_time_period
from bot.strategy.trade_calculator import round_to_tick, TICK_SIZE, POINT_VALUE


# ── Data Loading ──────────────────────────────────────────────────────────

def load_1s_bars(data_dir, start_date=None, end_date=None):
    """
    Load cached 1-second bar parquet files into a single DataFrame.
    Files are named: nq_1secs_YYYYMMDD.parquet
    """
    pattern = os.path.join(data_dir, "nq_1secs_*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No 1-second bar files found in {data_dir}")

    dfs = []
    for f in files:
        # Extract date from filename
        fname = os.path.basename(f)
        date_str = fname.replace("nq_1secs_", "").replace(".parquet", "")
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
        dfs.append(pd.read_parquet(f))

    if not dfs:
        raise FileNotFoundError(f"No data files in range {start_date}-{end_date}")

    df = pd.concat(dfs, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert('America/New_York')
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Loaded {len(df):,} 1-second bars from {len(dfs)} files")
    return df


def resample_to_5min(df_1s):
    """Resample 1-second bars to 5-minute OHLCV bars."""
    df = df_1s.set_index('date')
    bars = df.resample('5min', label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    return bars


# ── Strategy Loading ─────────────────────────────────────────────────────

def load_strategy(strategy_id=None, strategy_file=None):
    """Load strategy from file or strategy store."""
    if strategy_file:
        with open(strategy_file) as f:
            return json.load(f)

    if strategy_id:
        import importlib.util
        store_path = os.path.join(_ROOT, "logic", "utils", "strategy_store.py")
        spec = importlib.util.spec_from_file_location("strategy_store", store_path)
        store = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(store)
        return store.load_strategy(strategy_id)

    raise ValueError("Provide --strategy or --strategy-file")


def build_strategy_lookup(strategy):
    """Build {(time_period, risk_range): cell_config} from strategy cells."""
    lookup = {}
    for cell in strategy.get("cells", []):
        if not cell.get("enabled", True):
            continue
        key = (cell["time_period"], cell["risk_range"])
        config = {
            "setup": cell["setup"],
            "rr_target": cell["rr_target"],
            "ev": cell.get("ev", 0),
            "win_rate": cell.get("win_rate", 0),
            "samples": cell.get("samples", 0),
        }
        # Keep highest EV if duplicate
        if key not in lookup or config["ev"] > lookup[key]["ev"]:
            lookup[key] = config
    return lookup


def risk_to_range(risk_pts):
    """Map risk in points to risk range bucket string."""
    bins = [5, 10, 15, 20, 25, 30, 40, 80]
    for i in range(len(bins) - 1):
        if bins[i] <= risk_pts < bins[i + 1]:
            return f"{bins[i]}-{bins[i + 1]}"
    return None


# ── Trade Simulation ─────────────────────────────────────────────────────

@dataclass
class Trade:
    """A single backtested trade."""
    trade_id: int
    date: str
    fvg_type: str           # bullish / bearish
    time_period: str
    risk_range: str
    setup: str
    side: str               # BUY / SELL
    entry_price: float
    stop_price: float
    target_price: float
    risk_pts: float
    n_value: float
    contracts: int
    entry_time: str = ""
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""   # TP / SL / EOD
    pnl_pts: float = 0.0
    pnl_dollars: float = 0.0
    is_win: bool = False


def walk_trade_1s(df_1s, entry_time, entry_price, stop_price, target_price,
                  side, session_end_time=time(16, 0)):
    """
    Walk a trade forward on 1-second bars to determine outcome.

    Uses 1-second precision to correctly determine whether stop or target
    was hit first within each second.

    Args:
        df_1s: 1-second DataFrame with columns [date, open, high, low, close]
        entry_time: datetime of entry
        entry_price: limit entry price
        stop_price: stop loss price
        target_price: take profit price
        side: "BUY" or "SELL"
        session_end_time: time to force exit (default 16:00)

    Returns:
        (exit_time, exit_price, exit_reason) or None if not filled
    """
    # Filter to bars after entry time
    mask = df_1s['date'] > entry_time
    bars_after = df_1s[mask]

    if bars_after.empty:
        return None

    for _, bar in bars_after.iterrows():
        bar_time = bar['date']

        # Check session end
        if hasattr(bar_time, 'time') and bar_time.time() >= session_end_time:
            return (str(bar_time), bar['close'], "EOD")

        if side == "BUY":
            # Check stop first (conservative): did low touch stop?
            if bar['low'] <= stop_price:
                # Check if target was also hit — did high reach target?
                if bar['high'] >= target_price:
                    # Both hit in same second — use open to determine order
                    # If open is closer to stop, stop hit first
                    if bar['open'] <= stop_price:
                        return (str(bar_time), stop_price, "SL")
                    elif bar['open'] >= target_price:
                        return (str(bar_time), target_price, "TP")
                    else:
                        # Ambiguous — conservative: stop hit first
                        return (str(bar_time), stop_price, "SL")
                return (str(bar_time), stop_price, "SL")
            # Check target
            if bar['high'] >= target_price:
                return (str(bar_time), target_price, "TP")

        else:  # SELL
            # Check stop first: did high touch stop?
            if bar['high'] >= stop_price:
                if bar['low'] <= target_price:
                    if bar['open'] >= stop_price:
                        return (str(bar_time), stop_price, "SL")
                    elif bar['open'] <= target_price:
                        return (str(bar_time), target_price, "TP")
                    else:
                        return (str(bar_time), stop_price, "SL")
                return (str(bar_time), stop_price, "SL")
            if bar['low'] <= target_price:
                return (str(bar_time), target_price, "TP")

    # Never hit stop or target — EOD
    last_bar = bars_after.iloc[-1]
    return (str(last_bar['date']), last_bar['close'], "EOD")


def check_entry_fill(df_1s, entry_time, entry_price, side):
    """
    Check if the limit entry order would have filled on 1-second bars.

    For BUY limit: fills when price trades at or below entry_price.
    For SELL limit: fills when price trades at or above entry_price.

    Returns fill_time (datetime str) or None.
    """
    mask = df_1s['date'] >= entry_time
    bars = df_1s[mask]

    for _, bar in bars.iterrows():
        if side == "BUY" and bar['low'] <= entry_price:
            return str(bar['date'])
        elif side == "SELL" and bar['high'] >= entry_price:
            return str(bar['date'])

    return None


# ── Backtest Engine ──────────────────────────────────────────────────────

def run_backtest(df_1s, strategy, config=None):
    """
    Run a full backtest of a strategy against 1-second historical data.

    Process per day:
    1. Construct 5-min bars from 1s data
    2. Detect FVGs on each completed 5-min bar
    3. Check 1s bars for mitigation of active FVGs
    4. On mitigation: match to strategy cell, calculate setup
    5. Simulate entry fill, then walk trade on 1s bars

    Returns:
        list of Trade objects
    """
    config = config or {}
    balance = config.get("balance", 76000)
    risk_pct = config.get("risk_pct", 0.01)
    max_concurrent = config.get("max_concurrent", 3)
    max_daily_trades = config.get("max_daily_trades", 15)
    min_fvg_size = config.get("min_fvg_size", 0.25)
    use_slip = config.get("slip", False)
    use_risk_tiers = config.get("risk_tiers", False)

    # Load risk tier config from strategy meta if enabled
    _risk_rules = strategy.get("meta", {}).get("risk_rules", {}) if use_risk_tiers else {}
    _small_buckets = set(_risk_rules.get("small_buckets", []))
    _small_risk = _risk_rules.get("small_risk_pct", risk_pct)
    _medium_risk = _risk_rules.get("medium_risk_pct", risk_pct)
    _large_buckets = set(_risk_rules.get("large_buckets", []))
    _large_risk = _risk_rules.get("large_risk_pct", risk_pct)

    strategy_lookup = build_strategy_lookup(strategy)
    print(f"Strategy: {strategy.get('meta', {}).get('name', '?')} ({len(strategy_lookup)} cells)")

    # Group 1s bars by trading day
    df_1s['trade_date'] = df_1s['date'].dt.date
    trading_days = sorted(df_1s['trade_date'].unique())
    print(f"Trading days: {len(trading_days)} ({trading_days[0]} → {trading_days[-1]})")

    all_trades = []
    trade_counter = 0
    running_balance = balance

    # Hard gate: no trading Dec 20 onwards (holiday low-liquidity)
    no_trade_after_dec = strategy.get("meta", {}).get("hard_gates", {}).get("no_trade_after_dec")

    for day in trading_days:
        if no_trade_after_dec and day.month == 12 and day.day >= (no_trade_after_dec + 1):
            continue

        day_df = df_1s[df_1s['trade_date'] == day].copy()
        if day_df.empty:
            continue

        # Build 5-min bars for this day
        bars_5min = resample_to_5min(day_df)
        if len(bars_5min) < 3:
            continue

        # Track state for this day
        active_fvgs = []
        open_positions = 0
        daily_trades = 0
        daily_pnl = 0.0

        # Process each 5-min bar for FVG detection
        for i in range(2, len(bars_5min)):
            bar1 = bars_5min.iloc[i - 2].to_dict()
            bar2 = bars_5min.iloc[i - 1].to_dict()
            bar3 = bars_5min.iloc[i].to_dict()

            fvg = check_fvg_3bars(bar1, bar2, bar3, min_fvg_size)
            if fvg is None:
                continue

            # Assign time period
            fvg.time_period = _assign_time_period(bar3['date'], SESSION_INTERVALS)
            if not fvg.time_period:
                continue
            fvg.formation_date = str(day)

            # Check if any strategy cell could match this time period
            has_cell = any(
                tp == fvg.time_period for tp, _ in strategy_lookup.keys()
            )
            if not has_cell:
                continue

            active_fvgs.append((fvg, bar3['date']))

        # Process mitigations on 1s bars
        for fvg, formation_time in active_fvgs:
            if daily_trades >= max_daily_trades:
                break

            # Find first 1s bar that mitigates this FVG (after formation)
            post_formation = day_df[day_df['date'] > formation_time]
            mit_time = None
            for _, bar in post_formation.iterrows():
                if bar['low'] <= fvg.zone_high and bar['high'] >= fvg.zone_low:
                    mit_time = bar['date']
                    break

            if mit_time is None:
                continue

            # Check time gate (no entries after 15:45)
            if hasattr(mit_time, 'time') and mit_time.time() >= time(15, 45):
                continue

            # Calculate setup and match to strategy
            for setup_type in ["mit_extreme", "mid_extreme"]:
                if daily_trades >= max_daily_trades:
                    break
                if open_positions >= max_concurrent:
                    break

                # Entry price
                if setup_type == "mit_extreme":
                    entry = round_to_tick(
                        fvg.zone_high if fvg.fvg_type == "bullish" else fvg.zone_low
                    )
                else:
                    entry = round_to_tick((fvg.zone_high + fvg.zone_low) / 2)

                side = "BUY" if fvg.fvg_type == "bullish" else "SELL"

                # Slippage: entry 1 tick deeper into zone
                if use_slip:
                    if side == "BUY":
                        entry = round_to_tick(entry - TICK_SIZE)
                    else:
                        entry = round_to_tick(entry + TICK_SIZE)

                # Stop price
                stop = round_to_tick(
                    fvg.middle_low if fvg.fvg_type == "bullish" else fvg.middle_high
                )

                risk_pts = round_to_tick(abs(entry - stop))
                if risk_pts <= 0:
                    continue

                risk_range = risk_to_range(risk_pts)
                if not risk_range:
                    continue

                cell = strategy_lookup.get((fvg.time_period, risk_range))
                if cell is None or cell["setup"] != setup_type:
                    continue

                # Target
                n = cell["rr_target"]
                target_dist = round_to_tick(n * risk_pts)
                if side == "BUY":
                    target = round_to_tick(entry + target_dist)
                else:
                    target = round_to_tick(entry - target_dist)

                # Slippage: TP 1 tick early
                if use_slip:
                    if side == "BUY":
                        target = round_to_tick(target - TICK_SIZE)
                    else:
                        target = round_to_tick(target + TICK_SIZE)

                # Position size — use risk tiers if enabled
                if use_risk_tiers:
                    if risk_range in _large_buckets:
                        _rpct = _large_risk
                    elif risk_range in _small_buckets:
                        _rpct = _small_risk
                    else:
                        _rpct = _medium_risk
                    risk_budget = running_balance * _rpct
                else:
                    risk_budget = running_balance * risk_pct
                contracts = max(1, math.floor(risk_budget / (risk_pts * POINT_VALUE)))

                # Per-trade risk gate
                if risk_pts * POINT_VALUE * contracts > running_balance * risk_pct * 1.01:
                    contracts = max(1, math.floor(running_balance * risk_pct / (risk_pts * POINT_VALUE)))

                # Check entry fill on 1s bars
                fill_time = check_entry_fill(day_df, mit_time, entry, side)
                if fill_time is None:
                    continue

                # Walk trade on 1s bars
                result = walk_trade_1s(day_df, pd.Timestamp(fill_time), entry, stop, target, side)
                if result is None:
                    continue

                exit_time, exit_price, exit_reason = result

                # Calculate P&L
                if side == "BUY":
                    pnl_pts = exit_price - entry
                else:
                    pnl_pts = entry - exit_price
                pnl_dollars = round(pnl_pts * contracts * POINT_VALUE, 2)

                trade_counter += 1
                trade = Trade(
                    trade_id=trade_counter,
                    date=str(day),
                    fvg_type=fvg.fvg_type,
                    time_period=fvg.time_period,
                    risk_range=risk_range,
                    setup=setup_type,
                    side=side,
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    risk_pts=risk_pts,
                    n_value=n,
                    contracts=contracts,
                    entry_time=fill_time,
                    exit_time=exit_time,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_pts=round(pnl_pts, 2),
                    pnl_dollars=pnl_dollars,
                    is_win=exit_reason == "TP",
                )
                all_trades.append(trade)

                running_balance += pnl_dollars
                daily_trades += 1
                daily_pnl += pnl_dollars

                # Only one setup per FVG
                break

        # Daily kill switch check
        if daily_pnl <= -(balance * 0.03):
            print(f"  {day}: KILL SWITCH at ${daily_pnl:.0f} ({daily_trades} trades)")

    return all_trades, running_balance


# ── Report ────────────────────────────────────────────────────────────────

def print_report(trades, start_balance, final_balance):
    """Print a comprehensive backtest report."""
    print("\n" + "=" * 70)
    print("  BACKTEST REPORT")
    print("=" * 70)

    if not trades:
        print("  No trades executed.")
        return

    total = len(trades)
    wins = sum(1 for t in trades if t.is_win)
    losses = sum(1 for t in trades if t.exit_reason == "SL")
    eod = sum(1 for t in trades if t.exit_reason == "EOD")
    win_rate = wins / total * 100 if total else 0

    total_pnl = sum(t.pnl_dollars for t in trades)
    gross_profit = sum(t.pnl_dollars for t in trades if t.pnl_dollars > 0)
    gross_loss = sum(t.pnl_dollars for t in trades if t.pnl_dollars < 0)
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

    avg_win = gross_profit / wins if wins else 0
    avg_loss = gross_loss / losses if losses else 0

    # Drawdown
    equity = [start_balance]
    for t in trades:
        equity.append(equity[-1] + t.pnl_dollars)
    peak = equity[0]
    max_dd = 0
    max_dd_pct = 0
    for e in equity:
        if e > peak:
            peak = e
        dd = peak - e
        dd_pct = dd / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

    # Per-day stats
    days = set(t.date for t in trades)
    trades_per_day = total / len(days) if days else 0

    # Per-cell breakdown
    cell_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in trades:
        key = f"{t.time_period} | {t.risk_range} | {t.setup}"
        cell_stats[key]["trades"] += 1
        cell_stats[key]["wins"] += 1 if t.is_win else 0
        cell_stats[key]["pnl"] += t.pnl_dollars

    print(f"\n  Period:          {trades[0].date} → {trades[-1].date} ({len(days)} days)")
    print(f"  Start balance:   ${start_balance:,.0f}")
    print(f"  Final balance:   ${final_balance:,.0f}")
    print(f"  Total P&L:       ${total_pnl:,.0f} ({total_pnl/start_balance*100:+.1f}%)")
    print(f"\n  Total trades:    {total}")
    print(f"  Wins:            {wins} ({win_rate:.1f}%)")
    print(f"  Losses:          {losses}")
    print(f"  EOD exits:       {eod}")
    print(f"  Trades/day:      {trades_per_day:.1f}")
    print(f"\n  Gross profit:    ${gross_profit:,.0f}")
    print(f"  Gross loss:      ${gross_loss:,.0f}")
    print(f"  Profit factor:   {profit_factor:.2f}")
    print(f"  Avg win:         ${avg_win:,.0f}")
    print(f"  Avg loss:        ${avg_loss:,.0f}")
    print(f"\n  Max drawdown:    ${max_dd:,.0f} ({max_dd_pct*100:.1f}%)")

    print(f"\n  {'CELL':<42} {'TRADES':>6} {'WR%':>6} {'P&L':>10}")
    print(f"  {'-'*42} {'-'*6} {'-'*6} {'-'*10}")
    for cell, stats in sorted(cell_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] else 0
        print(f"  {cell:<42} {stats['trades']:>6} {wr:>5.1f}% ${stats['pnl']:>9,.0f}")

    print("\n" + "=" * 70)


def export_trades_csv(trades, output_path):
    """Export trade list to CSV."""
    rows = []
    for t in trades:
        rows.append({
            "trade_id": t.trade_id, "date": t.date, "fvg_type": t.fvg_type,
            "time_period": t.time_period, "risk_range": t.risk_range,
            "setup": t.setup, "side": t.side, "contracts": t.contracts,
            "entry_price": t.entry_price, "stop_price": t.stop_price,
            "target_price": t.target_price, "risk_pts": t.risk_pts,
            "n_value": t.n_value, "entry_time": t.entry_time,
            "exit_time": t.exit_time, "exit_price": t.exit_price,
            "exit_reason": t.exit_reason, "pnl_pts": t.pnl_pts,
            "pnl_dollars": t.pnl_dollars, "is_win": t.is_win,
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nTrade log exported to {output_path}")


# ── JSON Results ──────────────────────────────────────────────────────────

def build_results_json(trades, start_balance, final_balance, strategy, config):
    """Build the complete results dict for the API/dashboard."""
    if not trades:
        return {
            "meta": {
                "strategy_id": strategy.get("meta", {}).get("id", ""),
                "strategy_name": strategy.get("meta", {}).get("name", ""),
                "balance": start_balance,
                "risk_pct": config.get("risk_pct", 0.01),
            },
            "summary": {},
            "equity_curve": [],
            "drawdown_curve": [],
            "daily_pnl": [],
            "cell_performance": [],
            "hourly_pnl": [],
            "trades": [],
        }

    total = len(trades)
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if t.exit_reason == "SL"]
    eod = [t for t in trades if t.exit_reason == "EOD"]
    win_rate = len(wins) / total * 100 if total else 0

    total_pnl = sum(t.pnl_dollars for t in trades)
    gross_profit = sum(t.pnl_dollars for t in trades if t.pnl_dollars > 0)
    gross_loss = sum(t.pnl_dollars for t in trades if t.pnl_dollars < 0)
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else None

    avg_win = gross_profit / len(wins) if wins else 0
    avg_loss = gross_loss / len(losses) if losses else 0

    days = sorted(set(t.date for t in trades))
    trades_per_day = total / len(days) if days else 0

    # Equity curve + drawdown
    equity_curve = []
    drawdown_curve = []
    balance = start_balance
    peak = start_balance
    max_dd = 0
    max_dd_pct = 0
    for t in trades:
        balance += t.pnl_dollars
        if balance > peak:
            peak = balance
        dd = peak - balance
        dd_pct = dd / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
        equity_curve.append({
            "trade_id": t.trade_id, "date": t.date,
            "exit_time": t.exit_time, "balance": round(balance, 2),
            "pnl": round(t.pnl_dollars, 2), "exit_reason": t.exit_reason,
        })
        drawdown_curve.append({
            "trade_id": t.trade_id, "date": t.date,
            "drawdown": round(dd, 2), "drawdown_pct": round(dd_pct * 100, 2),
        })

    # Daily P&L
    daily_map = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
    for t in trades:
        daily_map[t.date]["pnl"] += t.pnl_dollars
        daily_map[t.date]["trades"] += 1
        if t.is_win:
            daily_map[t.date]["wins"] += 1
    daily_pnl = [
        {"date": d, "pnl": round(v["pnl"], 2), "trades": v["trades"], "wins": v["wins"]}
        for d, v in sorted(daily_map.items())
    ]

    # Cell performance
    cell_map = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0})
    for t in trades:
        key = (t.time_period, t.risk_range, t.setup, t.n_value)
        cell_map[key]["trades"] += 1
        cell_map[key]["pnl"] += t.pnl_dollars
        if t.is_win:
            cell_map[key]["wins"] += 1
        elif t.exit_reason == "SL":
            cell_map[key]["losses"] += 1
    cell_performance = sorted([
        {
            "time_period": k[0], "risk_range": k[1], "setup": k[2], "n_value": k[3],
            "trades": v["trades"], "wins": v["wins"], "losses": v["losses"],
            "win_rate": round(v["wins"] / v["trades"] * 100, 1) if v["trades"] else 0,
            "total_pnl": round(v["pnl"], 2),
            "avg_pnl": round(v["pnl"] / v["trades"], 2) if v["trades"] else 0,
        }
        for k, v in cell_map.items()
    ], key=lambda x: -x["total_pnl"])

    # Hourly P&L
    hourly_map = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
    for t in trades:
        hourly_map[t.time_period]["trades"] += 1
        hourly_map[t.time_period]["pnl"] += t.pnl_dollars
        if t.is_win:
            hourly_map[t.time_period]["wins"] += 1
    hourly_pnl = sorted([
        {
            "time_period": tp,
            "trades": v["trades"], "wins": v["wins"],
            "win_rate": round(v["wins"] / v["trades"] * 100, 1) if v["trades"] else 0,
            "pnl": round(v["pnl"], 2),
        }
        for tp, v in hourly_map.items()
    ], key=lambda x: x["time_period"])

    # Trade list
    trade_list = [
        {
            "id": t.trade_id, "date": t.date, "fvg_type": t.fvg_type,
            "time_period": t.time_period, "risk_range": t.risk_range,
            "setup": t.setup, "side": t.side, "contracts": t.contracts,
            "entry_price": t.entry_price, "stop_price": t.stop_price,
            "target_price": t.target_price, "risk_pts": t.risk_pts,
            "n_value": t.n_value, "entry_time": t.entry_time,
            "exit_time": t.exit_time, "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "pnl_pts": t.pnl_pts, "pnl_dollars": round(t.pnl_dollars, 2),
        }
        for t in trades
    ]

    return {
        "meta": {
            "strategy_id": strategy.get("meta", {}).get("id", ""),
            "strategy_name": strategy.get("meta", {}).get("name", ""),
            "start_date": trades[0].date if trades else "",
            "end_date": trades[-1].date if trades else "",
            "balance": start_balance,
            "risk_pct": config.get("risk_pct", 0.01),
            "trading_days": len(days),
        },
        "summary": {
            "total_trades": total,
            "wins": len(wins),
            "losses": len(losses),
            "eod_exits": len(eod),
            "win_rate": round(win_rate, 1),
            "net_pnl": round(total_pnl, 2),
            "pnl_pct": round(total_pnl / start_balance * 100, 1) if start_balance else 0,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor else None,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_drawdown": round(max_dd, 2),
            "max_dd_pct": round(max_dd_pct * 100, 1),
            "trades_per_day": round(trades_per_day, 1),
            "total_contracts": sum(t.contracts for t in trades),
            "final_balance": round(final_balance, 2),
        },
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "daily_pnl": daily_pnl,
        "cell_performance": cell_performance,
        "hourly_pnl": hourly_pnl,
        "trades": trade_list,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def load_databento_bars(symbol, start_date, end_date):
    """
    Load 1-minute bars from Databento data for backtesting.
    Returns DataFrame in the same format as load_1s_bars (columns: date, open, high, low, close, volume).
    """
    import sys as _sys
    _sys.path.insert(0, _ROOT)
    from logic.utils.data_cache_utils import fetch_market_data

    start_str = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}" if start_date else "2020-01-02"
    end_str = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}" if end_date else "2025-12-31"

    df = fetch_market_data(
        symbol=symbol, timeframe='1min', expiration_dates=None,
        period='1 year', custom_start=start_str, custom_end=end_str,
        roll_days=8, debug=False
    )
    df = df.reset_index()[['date', 'open', 'high', 'low', 'close', 'volume']]
    # Filter to RTH only (09:30-16:00 ET)
    df = df[(df['date'].dt.hour * 60 + df['date'].dt.minute >= 570) &
            (df['date'].dt.hour * 60 + df['date'].dt.minute < 960)]
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Loaded {len(df):,} Databento 1-min RTH bars")
    return df


def main():
    parser = argparse.ArgumentParser(description="FVG Strategy Backtester")
    parser.add_argument("--strategy", help="Strategy ID from logic/strategies/")
    parser.add_argument("--strategy-file", help="Path to strategy JSON file")
    parser.add_argument("--data-source", default="ib", choices=["ib", "databento"],
                        help="Data source: 'ib' (1-sec parquet) or 'databento' (1-min)")
    parser.add_argument("--data-dir", default=os.path.join(_ROOT, "bot", "data"),
                        help="Directory with cached 1s bar parquet files (ib only)")
    parser.add_argument("--start", help="Start date YYYYMMDD")
    parser.add_argument("--end", help="End date YYYYMMDD")
    parser.add_argument("--balance", type=float, default=76000, help="Starting balance")
    parser.add_argument("--risk-pct", type=float, default=0.01, help="Risk per trade (0.01=1%%)")
    parser.add_argument("--slip", action="store_true",
                        help="Apply realistic slippage: entry 1 tick deeper, TP 1 tick early")
    parser.add_argument("--risk-tiers", action="store_true",
                        help="Use 3-tier risk from strategy meta (0.5%%/1%%/1.5%%)")
    parser.add_argument("--output", help="Export trades to CSV")
    parser.add_argument("--json-output", help="Export full results as JSON (for dashboard)")
    args = parser.parse_args()

    # Load strategy
    strategy = load_strategy(args.strategy, args.strategy_file)
    print(f"Strategy: {strategy.get('meta', {}).get('name', 'unknown')}")

    # Load data
    if args.data_source == "databento":
        df = load_databento_bars("NQ", args.start, args.end)
    else:
        df = load_1s_bars(args.data_dir, args.start, args.end)

    # Build config
    config = {
        "balance": args.balance,
        "risk_pct": args.risk_pct,
        "max_concurrent": 3,
        "max_daily_trades": 15,
        "min_fvg_size": 0.25,
        "slip": args.slip,
        "risk_tiers": args.risk_tiers,
    }
    trades, final_balance = run_backtest(df, strategy, config)

    # Report
    print_report(trades, args.balance, final_balance)

    # JSON output (for dashboard API)
    if args.json_output:
        results = build_results_json(trades, args.balance, final_balance, strategy, config)
        results["meta"]["data_source"] = args.data_source
        results["meta"]["sizing"] = "compounding"
        if args.slip:
            results["meta"]["slippage"] = "entry 1tick deeper, TP 1tick early"
        if args.risk_tiers:
            results["meta"]["risk_tiers"] = strategy.get("meta", {}).get("risk_rules", {})
        os.makedirs(os.path.dirname(args.json_output) or '.', exist_ok=True)
        with open(args.json_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults JSON exported to {args.json_output}")

    # CSV export
    if args.output:
        export_trades_csv(trades, args.output)


if __name__ == "__main__":
    main()
