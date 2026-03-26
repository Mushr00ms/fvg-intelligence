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

from bot.backtest.us_holidays import is_trading_day
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
        if not is_trading_day(date_str):
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
    zone_high: float = 0.0
    zone_low: float = 0.0
    formation_time: str = ""
    entry_time: str = ""
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""   # TP / SL / EOD
    pnl_pts: float = 0.0
    pnl_dollars: float = 0.0
    is_win: bool = False
    # Runner mode fields (populated when tp_mode != "fixed")
    tp_touched: bool = False          # Did price reach TP level?
    runner_exit_reason: str = ""      # How the runner phase ended: SL/BE/EOD
    runner_exit_price: float = 0.0
    tp_exit_contracts: int = 0
    runner_contracts: int = 0
    excursion_pts: float = 0.0        # Max favorable move past TP
    excursion_r: float = 0.0          # Excursion in R-multiples


import numpy as np
try:
    import numba as nb
except ModuleNotFoundError:
    class _NumbaFallback:
        @staticmethod
        def njit(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    nb = _NumbaFallback()


# Exit reason codes for numba (can't return strings from njit)
_EXIT_TP = 1
_EXIT_SL = 2
_EXIT_EOD = 3
_EXIT_NONE = 0


@nb.njit(cache=True)
def _walk_trade_numba(timestamps, opens, highs, lows, closes, start_idx,
                      entry_price, stop_price, tp_trigger, is_buy, eod_minutes):
    """Numba-accelerated trade walk on numpy arrays.

    Returns (exit_idx, exit_price, exit_reason_code).
    exit_idx=-1 means no exit found.
    """
    n = len(timestamps)
    for i in range(start_idx, n):
        # Session end check: timestamps are minutes-since-midnight
        if timestamps[i] >= eod_minutes:
            return i, closes[i], _EXIT_EOD

        if is_buy:
            if lows[i] <= stop_price:
                if highs[i] >= tp_trigger:
                    if opens[i] <= stop_price:
                        return i, stop_price, _EXIT_SL
                    elif opens[i] >= tp_trigger:
                        return i, entry_price, _EXIT_TP  # placeholder, caller uses target_price
                    else:
                        return i, stop_price, _EXIT_SL
                return i, stop_price, _EXIT_SL
            if highs[i] >= tp_trigger:
                return i, entry_price, _EXIT_TP
        else:
            if highs[i] >= stop_price:
                if lows[i] <= tp_trigger:
                    if opens[i] >= stop_price:
                        return i, stop_price, _EXIT_SL
                    elif opens[i] <= tp_trigger:
                        return i, entry_price, _EXIT_TP
                    else:
                        return i, stop_price, _EXIT_SL
                return i, stop_price, _EXIT_SL
            if lows[i] <= tp_trigger:
                return i, entry_price, _EXIT_TP

    # EOD: use last bar
    return n - 1, closes[n - 1], _EXIT_EOD


@nb.njit(cache=True)
def _check_entry_numba(timestamps, highs, lows, start_idx, entry_price, is_buy):
    """Numba-accelerated entry fill check. Returns bar index or -1."""
    n = len(timestamps)
    for i in range(start_idx, n):
        if is_buy and lows[i] <= entry_price:
            return i
        elif not is_buy and highs[i] >= entry_price:
            return i
    return -1


_TP_MODE_FIXED = 0
_TP_MODE_RUNNER = 1      # SL stays at original
_TP_MODE_RUNNER_BE = 2   # SL moves to BE+1 tick
_TP_MODE_RUNNER_TRAIL = 3  # SL moves to BE, then trails by 1R


@nb.njit(cache=True)
def _walk_trade_runner_numba(timestamps, opens, highs, lows, closes, start_idx,
                             entry_price, stop_price, tp_trigger, is_buy,
                             eod_minutes, tp_mode, tick_size):
    """Numba-accelerated trade walk with runner modes.

    Phase 1: Walk until SL or TP touch (same as fixed mode for SL).
    Phase 2 (if TP touched): Continue with modified stop.
      - runner (mode 1): SL stays at original stop_price
      - runner-be (mode 2): SL moves to entry +/- 1 tick (BE+1 tick profit)
      - runner-trail (mode 3): SL moves to entry, then trails best price by 1R

    Returns (exit_idx, exit_price, exit_reason, tp_touch_idx, max_excursion_pts).
    tp_touch_idx=-1 if TP never touched.
    """
    n = len(timestamps)
    tp_touch_idx = -1
    max_favorable = 0.0
    risk_pts = entry_price - stop_price if is_buy else stop_price - entry_price

    # Phase 1: find TP touch or SL
    for i in range(start_idx, n):
        if timestamps[i] >= eod_minutes:
            return i, closes[i], _EXIT_EOD, tp_touch_idx, max_favorable

        if is_buy:
            if lows[i] <= stop_price:
                if highs[i] >= tp_trigger:
                    if opens[i] <= stop_price:
                        return i, stop_price, _EXIT_SL, -1, 0.0
                    elif opens[i] >= tp_trigger:
                        tp_touch_idx = i
                        break  # → phase 2
                    else:
                        return i, stop_price, _EXIT_SL, -1, 0.0
                return i, stop_price, _EXIT_SL, -1, 0.0
            if highs[i] >= tp_trigger:
                tp_touch_idx = i
                break  # → phase 2
        else:
            if highs[i] >= stop_price:
                if lows[i] <= tp_trigger:
                    if opens[i] >= stop_price:
                        return i, stop_price, _EXIT_SL, -1, 0.0
                    elif opens[i] <= tp_trigger:
                        tp_touch_idx = i
                        break
                    else:
                        return i, stop_price, _EXIT_SL, -1, 0.0
                return i, stop_price, _EXIT_SL, -1, 0.0
            if lows[i] <= tp_trigger:
                tp_touch_idx = i
                break

    # If we never touched TP, EOD
    if tp_touch_idx < 0:
        return n - 1, closes[n - 1], _EXIT_EOD, -1, 0.0

    # Phase 2: runner — TP touched, continue with modified stop
    if tp_mode == _TP_MODE_RUNNER:
        runner_stop = stop_price  # SL stays at original
        trailing = False
    elif tp_mode == _TP_MODE_RUNNER_BE:
        trailing = False
        # BE + 1 tick profit
        if is_buy:
            runner_stop = entry_price + tick_size
        else:
            runner_stop = entry_price - tick_size
    else:
        trailing = True
        if is_buy:
            runner_stop = entry_price
        else:
            runner_stop = entry_price

    for i in range(tp_touch_idx + 1, n):
        if timestamps[i] >= eod_minutes:
            # Track excursion on final bar before EOD
            if is_buy:
                exc = highs[i] - tp_trigger
                if exc > max_favorable:
                    max_favorable = exc
                eod_exit = closes[i]
                if eod_exit < runner_stop:
                    eod_exit = runner_stop
            else:
                exc = tp_trigger - lows[i]
                if exc > max_favorable:
                    max_favorable = exc
                eod_exit = closes[i]
                if eod_exit > runner_stop:
                    eod_exit = runner_stop
            return i, eod_exit, _EXIT_EOD, tp_touch_idx, max_favorable

        # Track max favorable excursion past TP
        if is_buy:
            exc = highs[i] - tp_trigger
            if exc > max_favorable:
                max_favorable = exc
            if trailing:
                candidate_stop = highs[i] - risk_pts
                if candidate_stop < entry_price:
                    candidate_stop = entry_price
                if candidate_stop > runner_stop:
                    runner_stop = candidate_stop
            # Check runner stop
            if lows[i] <= runner_stop:
                return i, runner_stop, _EXIT_SL, tp_touch_idx, max_favorable
        else:
            exc = tp_trigger - lows[i]
            if exc > max_favorable:
                max_favorable = exc
            if trailing:
                candidate_stop = lows[i] + risk_pts
                if candidate_stop > entry_price:
                    candidate_stop = entry_price
                if candidate_stop < runner_stop:
                    runner_stop = candidate_stop
            if highs[i] >= runner_stop:
                return i, runner_stop, _EXIT_SL, tp_touch_idx, max_favorable

    # End of data
    final_exit = closes[n - 1]
    if is_buy and final_exit < runner_stop:
        final_exit = runner_stop
    elif (not is_buy) and final_exit > runner_stop:
        final_exit = runner_stop
    return n - 1, final_exit, _EXIT_EOD, tp_touch_idx, max_favorable


@nb.njit(cache=True)
def _excursion_past_tp_numba(highs, lows, start_idx, tp_price, is_buy, eod_minutes, minutes):
    """Max favorable excursion past TP from TP bar to EOD."""
    n = len(highs)
    max_exc = 0.0
    for i in range(start_idx, n):
        if minutes[i] >= eod_minutes:
            break
        if is_buy:
            exc = highs[i] - tp_price
        else:
            exc = tp_price - lows[i]
        if exc > max_exc:
            max_exc = exc
    return max_exc


@nb.njit(cache=True)
def _find_mitigation_numba(highs, lows, start_idx, zone_low, zone_high):
    """Numba-accelerated mitigation scan. Returns bar index or -1."""
    n = len(highs)
    for i in range(start_idx, n):
        if lows[i] <= zone_high and highs[i] >= zone_low:
            return i
    return -1


def _prepare_day_arrays(day_df):
    """Extract numpy arrays from day DataFrame for numba functions.

    Converts timestamps to minutes-since-midnight for fast EOD comparison.
    Returns dict of arrays + the original dates for result lookup.
    """
    dates = day_df['date'].values  # numpy datetime64 array
    # Convert to minutes since midnight for EOD check
    # pandas Timestamp .hour/.minute via numpy
    dt_index = pd.DatetimeIndex(day_df['date'])
    minutes = (dt_index.hour * 60 + dt_index.minute).values.astype(np.float64)

    return {
        'dates': dates,
        'minutes': minutes,
        'opens': day_df['open'].values.astype(np.float64),
        'highs': day_df['high'].values.astype(np.float64),
        'lows': day_df['low'].values.astype(np.float64),
        'closes': day_df['close'].values.astype(np.float64),
    }


def walk_trade_1s(df_1s, entry_time, entry_price, stop_price, target_price,
                  side, session_end_time=time(16, 0), tp_trigger=None,
                  _arrays=None, _start_hint=0, tp_mode="fixed"):
    """
    Walk a trade forward on 1-second bars to determine outcome.

    Uses numba-accelerated inner loop for ~30x speedup over iterrows.

    tp_mode controls what happens when TP is reached:
        "fixed":     Exit at TP (default, classic behavior)
        "runner":    TP touched → SL stays, let trade run until SL or EOD
        "runner-be": TP touched → SL moves to BE+1 tick, run until stop or EOD
        "runner-trail": TP touched → SL moves to BE, then trails best price by 1R

    Returns:
        Fixed mode:  (exit_time, exit_price, exit_reason) or None
        Runner modes: (exit_time, exit_price, exit_reason, tp_touched, excursion_pts) or None
    """
    if tp_trigger is None:
        tp_trigger = target_price

    if _arrays is not None:
        arr = _arrays
    else:
        arr = _prepare_day_arrays(df_1s)

    # Find start index: first bar after entry_time
    entry_ts = np.datetime64(entry_time)
    if _start_hint > 0 and _start_hint < len(arr['dates']) and arr['dates'][_start_hint] > entry_ts:
        start_idx = _start_hint
    else:
        start_idx = np.searchsorted(arr['dates'], entry_ts, side='right')

    if start_idx >= len(arr['dates']):
        return None

    eod_minutes = session_end_time.hour * 60 + session_end_time.minute
    is_buy = side == "BUY"

    if tp_mode == "fixed":
        idx, exit_price, reason = _walk_trade_numba(
            arr['minutes'], arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            start_idx, entry_price, stop_price, tp_trigger, is_buy, eod_minutes
        )

        if reason == _EXIT_NONE:
            return None

        exit_time = str(pd.Timestamp(arr['dates'][idx]))

        if reason == _EXIT_TP:
            return (exit_time, target_price, "TP")
        elif reason == _EXIT_SL:
            return (exit_time, stop_price, "SL")
        else:
            return (exit_time, arr['closes'][idx], "EOD")

    else:
        # Runner modes
        if tp_mode == "runner":
            mode_code = _TP_MODE_RUNNER
        elif tp_mode == "runner-be":
            mode_code = _TP_MODE_RUNNER_BE
        else:
            mode_code = _TP_MODE_RUNNER_TRAIL

        idx, exit_price, reason, tp_idx, exc_pts = _walk_trade_runner_numba(
            arr['minutes'], arr['opens'], arr['highs'], arr['lows'], arr['closes'],
            start_idx, entry_price, stop_price, tp_trigger, is_buy,
            eod_minutes, mode_code, TICK_SIZE
        )

        if reason == _EXIT_NONE:
            return None

        exit_time = str(pd.Timestamp(arr['dates'][idx]))
        tp_touched = tp_idx >= 0

        if reason == _EXIT_SL:
            actual_exit_price = exit_price  # runner_stop or original stop
        elif reason == _EXIT_EOD:
            actual_exit_price = arr['closes'][idx]
        else:
            actual_exit_price = exit_price

        # Determine exit reason string
        if tp_touched:
            if reason == _EXIT_SL:
                if tp_mode == "runner-be":
                    reason_str = "BE"
                elif tp_mode == "runner-trail":
                    reason_str = "TRAIL"
                else:
                    reason_str = "SL"
            else:
                reason_str = "EOD"
        else:
            reason_str = "SL" if reason == _EXIT_SL else "EOD"

        return (exit_time, actual_exit_price, reason_str, tp_touched,
                round_to_tick(exc_pts))


def _resolve_tp_mode(tp_mode, contracts):
    """Map public TP modes to the engine behavior for this trade."""
    if tp_mode == "split":
        return "fixed" if contracts <= 1 else "runner-trail"
    return tp_mode


def _summarize_trade_exit(side, entry_price, target_price, contracts, tp_mode, walk_result):
    """Convert raw walk output into trade-level exit and P&L fields."""
    resolved_mode = _resolve_tp_mode(tp_mode, contracts)
    summary = {
        "exit_time": "",
        "exit_price": 0.0,
        "exit_reason": "",
        "pnl_pts": 0.0,
        "tp_touched": False,
        "runner_exit_reason": "",
        "runner_exit_price": 0.0,
        "tp_exit_contracts": 0,
        "runner_contracts": 0,
        "excursion_pts": 0.0,
    }

    if resolved_mode == "fixed":
        exit_time, exit_price, exit_reason = walk_result
        tp_touched = exit_reason == "TP"
        excursion_pts = 0.0
        if side == "BUY":
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price

        summary.update({
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "pnl_pts": pnl_pts,
            "tp_touched": tp_touched,
            "excursion_pts": excursion_pts,
        })
        return summary

    exit_time, runner_exit_price, runner_exit_reason, tp_touched, excursion_pts = walk_result
    if side == "BUY":
        runner_pnl_pts = runner_exit_price - entry_price
        target_pnl_pts = target_price - entry_price
    else:
        runner_pnl_pts = entry_price - runner_exit_price
        target_pnl_pts = entry_price - target_price

    if tp_mode == "split" and contracts > 1 and tp_touched:
        tp_exit_contracts = contracts - 1
        gross_pnl_pts = target_pnl_pts * tp_exit_contracts + runner_pnl_pts
        summary.update({
            "exit_time": exit_time,
            "exit_price": runner_exit_price,
            "exit_reason": "SPLIT",
            "pnl_pts": gross_pnl_pts / contracts,
            "tp_touched": True,
            "runner_exit_reason": runner_exit_reason,
            "runner_exit_price": runner_exit_price,
            "tp_exit_contracts": tp_exit_contracts,
            "runner_contracts": 1,
            "excursion_pts": excursion_pts,
        })
        return summary

    summary.update({
        "exit_time": exit_time,
        "exit_price": runner_exit_price,
        "exit_reason": runner_exit_reason,
        "pnl_pts": runner_pnl_pts,
        "tp_touched": tp_touched,
        "runner_exit_reason": runner_exit_reason,
        "runner_exit_price": runner_exit_price if tp_touched else 0.0,
        "excursion_pts": excursion_pts,
    })
    return summary


def check_entry_fill(df_1s, entry_time, entry_price, side,
                     _arrays=None, _start_hint=0):
    """
    Check if the limit entry order would have filled on 1-second bars.

    Numba-accelerated. Returns fill_time (datetime str) or None.
    """
    if _arrays is not None:
        arr = _arrays
    else:
        arr = _prepare_day_arrays(df_1s)

    entry_ts = np.datetime64(pd.Timestamp(entry_time))
    if _start_hint > 0 and _start_hint < len(arr['dates']) and arr['dates'][_start_hint] >= entry_ts:
        start_idx = _start_hint
    else:
        start_idx = np.searchsorted(arr['dates'], entry_ts, side='left')

    if start_idx >= len(arr['dates']):
        return None

    is_buy = side == "BUY"
    idx = _check_entry_numba(arr['dates'], arr['highs'], arr['lows'],
                             start_idx, entry_price, is_buy)

    if idx < 0:
        return None
    return str(pd.Timestamp(arr['dates'][idx]))


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
    mit_entry_ticks = config.get("mit_entry_ticks", 0)  # MIT entry slippage (0=limit, 1-4=MIT)
    confirmed_limit = config.get("confirmed_limit", False)  # fill confirmation: require 1 tick past, fill at limit
    tp_mode = config.get("tp_mode", "fixed")  # fixed, runner, runner-be, split

    # IB tiered commission: per-side IB rate (marginal) + $1.40 exchange/reg
    # Round-trip = 2 sides. Monthly volume determines IB rate tier.
    _EXCHANGE_FEE_PER_SIDE = 1.40
    _IB_TIERS = [  # (cumulative_threshold, per_side_ib_rate)
        (1000,  0.85),
        (10000, 0.65),
        (20000, 0.45),
        (float('inf'), 0.25),
    ]
    _monthly_contracts = 0  # reset each month
    _current_month = None

    def _calc_commission(num_contracts, trade_date):
        """Calculate round-trip commission with IB tiered pricing."""
        nonlocal _monthly_contracts, _current_month
        # Reset on new month
        trade_month = (trade_date.year, trade_date.month) if hasattr(trade_date, 'year') else None
        if trade_month != _current_month:
            _monthly_contracts = 0
            _current_month = trade_month

        total_comm = 0.0
        remaining = num_contracts
        vol = _monthly_contracts
        for threshold, rate in _IB_TIERS:
            if remaining <= 0:
                break
            available = threshold - vol
            if available <= 0:
                continue
            batch = min(remaining, available)
            per_side = rate + _EXCHANGE_FEE_PER_SIDE
            total_comm += batch * per_side * 2  # round-trip = 2 sides
            remaining -= batch
            vol += batch

        _monthly_contracts += num_contracts
        return round(total_comm, 2)
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

    # Group 1s bars by trading day (pre-group once, avoids O(N) scan per day)
    df_1s['trade_date'] = df_1s['date'].dt.date
    _day_groups = {day: grp for day, grp in df_1s.groupby('trade_date')}
    trading_days = sorted(_day_groups.keys())
    print(f"Trading days: {len(trading_days)} ({trading_days[0]} → {trading_days[-1]})")

    all_trades = []
    trade_counter = 0
    running_balance = balance

    # Hard gate: no trading Dec 20 onwards (holiday low-liquidity)
    no_trade_after_dec = strategy.get("meta", {}).get("hard_gates", {}).get("no_trade_after_dec")

    for day in trading_days:
        if no_trade_after_dec and day.month == 12 and day.day >= (no_trade_after_dec + 1):
            continue

        day_df = _day_groups[day]
        if day_df.empty:
            continue

        # Build 5-min bars for this day
        bars_5min = resample_to_5min(day_df)
        if len(bars_5min) < 3:
            continue

        # Pre-extract numpy arrays for numba (once per day, reused for all trades)
        day_arrays = _prepare_day_arrays(day_df)

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

            # FVG is only confirmed when bar3 CLOSES (5 min after bar3 open).
            # Mitigation scan must start after bar3 close, not bar3 open.
            bar3_close_time = bar3['date'] + pd.Timedelta(minutes=5)
            active_fvgs.append((fvg, bar3_close_time))

        # Process mitigations on 1s bars
        for fvg, formation_time in active_fvgs:
            if daily_trades >= max_daily_trades:
                break

            # Find first 1s bar that mitigates this FVG (after formation)
            form_ts = np.datetime64(formation_time)
            mit_start = np.searchsorted(day_arrays['dates'], form_ts, side='right')
            mit_idx = _find_mitigation_numba(
                day_arrays['highs'], day_arrays['lows'], mit_start,
                fvg.zone_low, fvg.zone_high,
            )
            if mit_idx < 0:
                continue
            mit_time = pd.Timestamp(day_arrays['dates'][mit_idx])

            # Check time gate (no entries after 15:45 ET)
            # Use pre-computed ET minutes array (not UTC timestamp)
            mit_minutes_et = day_arrays['minutes'][mit_idx]
            if mit_minutes_et >= 15 * 60 + 45:
                continue

            # Calculate setup and match to strategy
            for setup_type in ["mit_extreme", "mid_extreme"]:
                if daily_trades >= max_daily_trades:
                    break
                if open_positions >= max_concurrent:
                    break

                # Entry price (exact — used for cell lookup, matching live engine)
                if setup_type == "mit_extreme":
                    entry_exact = round_to_tick(
                        fvg.zone_high if fvg.fvg_type == "bullish" else fvg.zone_low
                    )
                else:
                    entry_exact = round_to_tick((fvg.zone_high + fvg.zone_low) / 2)

                side = "BUY" if fvg.fvg_type == "bullish" else "SELL"

                # Stop price
                stop = round_to_tick(
                    fvg.middle_low if fvg.fvg_type == "bullish" else fvg.middle_high
                )

                # Cell lookup uses exact entry (no slippage) — same as live engine
                risk_pts_exact = round_to_tick(abs(entry_exact - stop))
                if risk_pts_exact <= 0:
                    continue

                risk_range = risk_to_range(risk_pts_exact)
                if not risk_range:
                    continue

                cell = strategy_lookup.get((fvg.time_period, risk_range))
                if cell is None or cell["setup"] != setup_type:
                    continue

                # Apply entry slippage for fill simulation:
                # --slip: legacy 1-tick model (entry + TP slippage)
                # --confirmed-limit: require 1 tick past to confirm fill, but fill at limit price
                # --mit-entry N: MIT order slippage (N ticks on entry only, TP = limit touch)
                if confirmed_limit:
                    # Fill at exact limit price; confirmation handled at entry check below
                    entry_fill = entry_exact
                elif use_slip:
                    # Legacy: entry 1 tick deeper + TP 1 tick past
                    if side == "BUY":
                        entry_fill = round_to_tick(entry_exact - TICK_SIZE)
                    else:
                        entry_fill = round_to_tick(entry_exact + TICK_SIZE)
                elif mit_entry_ticks > 0:
                    # MIT: entry N ticks worse (market fill penalty)
                    # BUY: fill higher (pay more), SELL: fill lower (receive less)
                    slip_amount = TICK_SIZE * mit_entry_ticks
                    if side == "BUY":
                        entry_fill = round_to_tick(entry_exact + slip_amount)
                    else:
                        entry_fill = round_to_tick(entry_exact - slip_amount)
                else:
                    entry_fill = entry_exact

                # Risk from actual fill price (for position sizing and target calc)
                risk_pts = round_to_tick(abs(entry_fill - stop))
                if risk_pts <= 0:
                    continue

                # Target based on fill price
                n = cell["rr_target"]
                target_dist = round_to_tick(n * risk_pts)
                if side == "BUY":
                    target = round_to_tick(entry_fill + target_dist)
                else:
                    target = round_to_tick(entry_fill - target_dist)

                # TP trigger: legacy slip requires 1 tick past; all other modes = touch
                # (validated: 99.7% of TP-first winners go past target)
                if use_slip:
                    if side == "BUY":
                        tp_trigger = round_to_tick(target + TICK_SIZE)
                    else:
                        tp_trigger = round_to_tick(target - TICK_SIZE)
                else:
                    tp_trigger = target

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

                # Check entry fill:
                # --slip: price must reach slipped level (1 tick into zone) to confirm fill
                # --confirmed-limit: price must go 1 tick past to confirm limit fill
                # --mit-entry: MIT triggers on touch at exact price, fills worse (market)
                # default: limit at exact price, touch = fill
                if confirmed_limit:
                    # BUY: price must dip 1 tick below entry to confirm fill
                    # SELL: price must rise 1 tick above entry to confirm fill
                    if side == "BUY":
                        entry_check_price = round_to_tick(entry_exact - TICK_SIZE)
                    else:
                        entry_check_price = round_to_tick(entry_exact + TICK_SIZE)
                elif use_slip:
                    entry_check_price = entry_fill
                else:
                    entry_check_price = entry_exact
                fill_time = check_entry_fill(day_df, mit_time, entry_check_price, side,
                                             _arrays=day_arrays, _start_hint=mit_idx)
                if fill_time is None:
                    continue

                # Walk trade with fill price (slipped if MIT), limit TP (touch)
                resolved_tp_mode = _resolve_tp_mode(tp_mode, contracts)
                result = walk_trade_1s(day_df, pd.Timestamp(fill_time), entry_fill, stop, target, side,
                                       tp_trigger=tp_trigger, _arrays=day_arrays,
                                       tp_mode=resolved_tp_mode)
                if result is None:
                    continue

                exit_summary = _summarize_trade_exit(
                    side, entry_fill, target, contracts, tp_mode, result
                )
                exit_time = exit_summary["exit_time"]
                exit_price = exit_summary["exit_price"]
                exit_reason = exit_summary["exit_reason"]
                pnl_pts = exit_summary["pnl_pts"]
                tp_touched = exit_summary["tp_touched"]
                runner_exit_reason = exit_summary["runner_exit_reason"]
                runner_exit_price = exit_summary["runner_exit_price"]
                tp_exit_contracts = exit_summary["tp_exit_contracts"]
                runner_contracts = exit_summary["runner_contracts"]
                excursion_pts = exit_summary["excursion_pts"]

                # For fixed TP wins, measure how far price went past target
                if resolved_tp_mode == "fixed" and exit_reason == "TP":
                    exit_ts = np.datetime64(pd.Timestamp(exit_time))
                    tp_bar_idx = np.searchsorted(day_arrays['dates'], exit_ts, side='left')
                    excursion_pts = round_to_tick(_excursion_past_tp_numba(
                        day_arrays['highs'], day_arrays['lows'],
                        tp_bar_idx, target, side == "BUY",
                        16 * 60, day_arrays['minutes'],
                    ))

                commission = _calc_commission(contracts, day)
                pnl_dollars = round(pnl_pts * contracts * POINT_VALUE - commission, 2)

                # Determine is_win
                if resolved_tp_mode == "fixed":
                    is_win = exit_reason == "TP"
                else:
                    is_win = pnl_dollars > 0

                trade_counter += 1
                excursion_r = round(excursion_pts / risk_pts, 2) if risk_pts > 0 and excursion_pts > 0 else 0.0
                trade = Trade(
                    trade_id=trade_counter,
                    date=str(day),
                    fvg_type=fvg.fvg_type,
                    time_period=fvg.time_period,
                    risk_range=risk_range,
                    setup=setup_type,
                    side=side,
                    entry_price=entry_fill,
                    stop_price=stop,
                    target_price=target,
                    risk_pts=risk_pts,
                    n_value=n,
                    contracts=contracts,
                    zone_high=fvg.zone_high,
                    zone_low=fvg.zone_low,
                    formation_time=fvg.time_candle2,
                    entry_time=fill_time,
                    exit_time=exit_time,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_pts=round(pnl_pts, 2),
                    pnl_dollars=pnl_dollars,
                    is_win=is_win,
                    tp_touched=tp_touched,
                    runner_exit_reason=runner_exit_reason,
                    runner_exit_price=runner_exit_price,
                    tp_exit_contracts=tp_exit_contracts,
                    runner_contracts=runner_contracts,
                    excursion_pts=excursion_pts,
                    excursion_r=excursion_r,
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
        if dd_pct > max_dd_pct:
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
            "zone_high": t.zone_high, "zone_low": t.zone_low,
            "formation_time": t.formation_time,
            "n_value": t.n_value, "entry_time": t.entry_time,
            "exit_time": t.exit_time, "exit_price": t.exit_price,
            "exit_reason": t.exit_reason, "pnl_pts": t.pnl_pts,
            "pnl_dollars": t.pnl_dollars, "is_win": t.is_win,
            "tp_touched": t.tp_touched, "excursion_pts": t.excursion_pts,
            "excursion_r": t.excursion_r, "runner_exit_reason": t.runner_exit_reason,
            "runner_exit_price": t.runner_exit_price,
            "tp_exit_contracts": t.tp_exit_contracts,
            "runner_contracts": t.runner_contracts,
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
        if dd_pct > max_dd_pct:
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
            "zone_high": t.zone_high, "zone_low": t.zone_low,
            "formation_time": t.formation_time,
            "n_value": t.n_value, "entry_time": t.entry_time,
            "exit_time": t.exit_time, "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "pnl_pts": t.pnl_pts, "pnl_dollars": round(t.pnl_dollars, 2),
            "tp_touched": t.tp_touched, "excursion_pts": t.excursion_pts,
            "excursion_r": t.excursion_r, "runner_exit_reason": t.runner_exit_reason,
            "runner_exit_price": t.runner_exit_price,
            "tp_exit_contracts": t.tp_exit_contracts,
            "runner_contracts": t.runner_contracts,
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
    parser.add_argument("--mit-entry", type=int, default=0, metavar="TICKS",
                        help="MIT entry: N ticks slippage on entry fill (0=limit, 1-4=realistic)")
    parser.add_argument("--confirmed-limit", action="store_true",
                        help="Confirmed limit fills: require 1 tick past entry+TP, fill at limit price")
    parser.add_argument("--tp-mode", default="fixed", choices=["fixed", "runner", "runner-be", "split"],
                        help="TP exit mode: fixed, runner, runner-be, or split ((n-1) at TP + 1 trailing runner)")
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
        "confirmed_limit": args.confirmed_limit,
        "risk_tiers": args.risk_tiers,
        "mit_entry_ticks": args.mit_entry,
        "tp_mode": args.tp_mode,
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
        elif args.mit_entry > 0:
            results["meta"]["slippage"] = f"MIT entry {args.mit_entry}tick, limit TP (touch)"
        if args.tp_mode != "fixed":
            results["meta"]["tp_mode"] = args.tp_mode
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
