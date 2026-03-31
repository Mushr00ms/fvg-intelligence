"""
btc_fvg_analyzer.py — Self-contained BTC FVG analysis for parameter sweeps.

Pure-function design: no config.py imports, no global state.
Takes OHLCV DataFrames + config dict, returns per-trade records.

Handles:
  - BPS normalization (FVG size & risk relative to price at formation)
  - Continuous 24h sessions (no session-end cutoff)
  - Bar-based mitigation + expansion windows
  - Only mit_extreme + mid_extreme setups

Usage:
    from logic.utils.btc_fvg_analyzer import analyze_btc_fvgs

    trades = analyze_btc_fvgs(
        df_detect,           # OHLCV at detection TF (e.g. 5min)
        df_walk,             # OHLCV at walk TF (e.g. 1min) for precise stop/target
        config={
            "min_fvg_bps": 20,
            "mitigation_window_bars": 60,
            "expansion_window_bars": 48,
            "time_period_minutes": 60,
        },
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── R:R levels to evaluate ──────────────────────────────────────────────
N_VALUES = [round(1.0 + i * 0.25, 2) for i in range(9)]  # 1.0 .. 3.0

# Setups: only extreme-stop variants
SETUPS = ("mit_extreme", "mid_extreme")


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """A single simulated trade from an FVG."""
    # FVG identity
    fvg_type: str              # "bullish" / "bearish"
    formation_time: object     # timestamp of candle 3
    formation_price: float     # close of candle 2 (reference price for bps)
    fvg_size: float            # absolute USD size of gap
    fvg_size_bps: float        # gap size as bps of formation_price

    # Setup
    setup: str                 # "mit_extreme" / "mid_extreme"
    entry_price: float
    stop_price: float
    risk: float                # |entry - stop| in USD
    risk_bps: float            # risk as bps of formation_price

    # Outcome
    max_exp: float             # max favorable excursion before stop (USD)
    max_exp_bps: float         # max favorable excursion in bps

    # Context
    time_period: str           # e.g. "14:00-15:00"
    mitigation_time: object    # timestamp of mitigation
    bars_to_mitigation: int    # detection-TF bars until mitigation

    # Per-RR outcomes (precomputed for speed)
    outcomes: dict = field(default_factory=dict)  # {n_value: bool (win/loss)}


# ── Core FVG detection (inlined, no config imports) ─────────────────────

def _detect_fvgs(data: pd.DataFrame, min_size: float) -> list:
    """Detect FVGs. Returns list of tuples (same format as fvg_detection.py).

    Each non-None entry:
        (fvg_type, y0, y1, time_c1, time_c2, time_c3, idx_c3,
         middle_open, middle_low, middle_high, first_open)
    """
    highs = data["high"].values
    lows = data["low"].values
    opens = data["open"].values
    dates = data["date"].values
    n = len(data)
    results = []

    for i in range(2, n):
        first_high = highs[i - 2]
        first_low = lows[i - 2]
        first_open = opens[i - 2]
        middle_open = opens[i - 1]
        middle_low = lows[i - 1]
        middle_high = highs[i - 1]
        third_low = lows[i]
        third_high = highs[i]

        # Bullish FVG: gap up
        if third_low > first_high:
            fvg_size = third_low - first_high
            if fvg_size >= min_size:
                results.append((
                    "bullish", first_high, third_low,
                    dates[i - 2], dates[i - 1], dates[i], i,
                    middle_open, middle_low, middle_high, first_open,
                ))
                continue

        # Bearish FVG: gap down
        if third_high < first_low:
            fvg_size = first_low - third_high
            if fvg_size >= min_size:
                results.append((
                    "bearish", third_high, first_low,
                    dates[i - 2], dates[i - 1], dates[i], i,
                    middle_open, middle_low, middle_high, first_open,
                ))

    return results


# ── Mitigation search (bar-based window, no session cutoff) ─────────────

def _find_mitigation(
    df_detect: pd.DataFrame,
    fvg: tuple,
    max_bars: int,
) -> Optional[tuple]:
    """Search for mitigation within max_bars of the detection timeframe.

    Returns (mitigation_idx, bars_to_mitigation) or None.
    Mitigation = any bar whose range overlaps the FVG zone.
    """
    _, y0, y1, _, _, _, idx_c3, *_ = fvg
    zone_low, zone_high = min(y0, y1), max(y0, y1)

    start = idx_c3 + 1
    end = min(start + max_bars, len(df_detect))

    lows = df_detect["low"].values
    highs = df_detect["high"].values

    for j in range(start, end):
        if lows[j] <= zone_high and highs[j] >= zone_low:
            return j, j - idx_c3

    return None


# ── Trade walk (max expansion before stop touch) ────────────────────────

def _walk_trade_bars(
    df: pd.DataFrame,
    start_idx: int,
    max_bars: int,
    fvg_type: str,
    entry_price: float,
    stop_price: float,
) -> float:
    """Walk up to max_bars from start_idx. Return max favorable expansion (USD).

    Stop is touch-based (conservative: expansion on stop bar excluded).
    """
    end_idx = min(start_idx + max_bars, len(df))
    max_exp = 0.0

    lows = df["low"].values
    highs = df["high"].values

    for j in range(start_idx, end_idx):
        low = lows[j]
        high = highs[j]

        # Check stop first
        if fvg_type == "bullish" and low <= stop_price:
            break
        if fvg_type == "bearish" and high >= stop_price:
            break

        # Track favorable expansion
        if fvg_type == "bullish":
            exp = high - entry_price
        else:
            exp = entry_price - low

        if exp > max_exp:
            max_exp = exp

    return max_exp


# ── Time period assignment ──────────────────────────────────────────────

def _build_time_periods(interval_minutes: int) -> list:
    """Generate 24h time period labels at the given interval.

    Returns list of (start_str, end_str, start_minutes, end_minutes).
    E.g. for 60min: [("00:00", "01:00", 0, 60), ("01:00", "02:00", 60, 120), ...]
    """
    periods = []
    for m in range(0, 1440, interval_minutes):
        end_m = m + interval_minutes
        if end_m > 1440:
            end_m = 1440
        start_str = f"{m // 60:02d}:{m % 60:02d}"
        end_str = f"{end_m // 60:02d}:{end_m % 60:02d}"
        periods.append((start_str, end_str, m, end_m))
    return periods


def _assign_time_period(ts, periods: list) -> Optional[str]:
    """Assign a timestamp to a time period. Returns label like '14:00-15:00'."""
    if hasattr(ts, 'hour'):
        minutes = ts.hour * 60 + ts.minute
    else:
        # numpy datetime64
        ts_pd = pd.Timestamp(ts)
        minutes = ts_pd.hour * 60 + ts_pd.minute

    for start_str, end_str, start_m, end_m in periods:
        if start_m <= minutes < end_m:
            return f"{start_str}-{end_str}"
    return None


# ── Main analysis function ──────────────────────────────────────────────

def analyze_btc_fvgs(
    df_detect: pd.DataFrame,
    df_walk: pd.DataFrame,
    config: dict,
) -> list[TradeRecord]:
    """Run full BTC FVG analysis pipeline.

    Args:
        df_detect: OHLCV at detection timeframe (e.g. 5min, 15min, 1h, 4h).
                   Must have columns: date, open, high, low, close.
                   DatetimeIndex or 'date' column in UTC.
        df_walk:   OHLCV at walk timeframe (finer resolution for stop/target).
                   Same schema. For 5min detection, use 1min walk.
                   For 1h/4h detection, use 5min walk.
        config:    Dict with keys:
            min_fvg_bps           : int   — minimum FVG size in basis points
            mitigation_window_bars: int   — max detection-TF bars to search for mitigation
            expansion_window_bars : int   — max walk-TF bars for trade resolution
            time_period_minutes   : int   — width of time buckets (60, 120, 240)

    Returns:
        List of TradeRecord — one per (FVG × setup) combination.
        Only includes mitigated FVGs with valid risk > 0.
    """
    min_fvg_bps = config["min_fvg_bps"]
    mit_window = config["mitigation_window_bars"]
    exp_window = config["expansion_window_bars"]
    time_period_min = config["time_period_minutes"]

    # Ensure date column exists (not just index)
    for df in (df_detect, df_walk):
        if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)

    # Pre-build time periods (24h UTC grid)
    periods = _build_time_periods(time_period_min)

    # Pre-index walk data for fast searchsorted
    walk_dates = df_walk["date"].values

    # We need reference prices to convert min_fvg_bps → absolute threshold.
    # Use a rolling approach: for each candle, the "reference price" is the
    # close of candle 2 (middle candle). But for detection, we need a
    # threshold per-candle. Approach: detect ALL FVGs with a very low
    # absolute threshold, then filter by bps post-hoc. This avoids
    # needing per-bar adaptive thresholds.

    # Detect with near-zero threshold, then filter by bps
    raw_fvgs = _detect_fvgs(df_detect, min_size=0.01)

    trades: list[TradeRecord] = []

    for fvg in raw_fvgs:
        (fvg_type, y0, y1, time_c1, time_c2, time_c3, idx_c3,
         middle_open, middle_low, middle_high, first_open) = fvg

        zone_low, zone_high = min(y0, y1), max(y0, y1)
        fvg_size = zone_high - zone_low

        # Reference price: close of candle 2 (middle candle)
        ref_price = df_detect["close"].values[idx_c3 - 1]
        if ref_price <= 0:
            continue

        fvg_size_bps = (fvg_size / ref_price) * 10000

        # Filter by min bps
        if fvg_size_bps < min_fvg_bps:
            continue

        # Search for mitigation
        mit_result = _find_mitigation(df_detect, fvg, mit_window)
        if mit_result is None:
            continue

        mit_idx, bars_to_mit = mit_result
        mitigation_time = df_detect["date"].values[mit_idx]

        # Find mitigation point in walk data
        walk_start = np.searchsorted(walk_dates, mitigation_time, side="right")
        if walk_start >= len(df_walk):
            continue

        # Assign time period (based on formation time of candle 3)
        time_period = _assign_time_period(time_c3, periods)
        if time_period is None:
            continue

        # Define entry/stop for each setup
        fvg_midpoint = (zone_high + zone_low) / 2

        setup_configs = {}

        # mit_extreme: entry @ zone edge, stop @ middle candle extreme
        if fvg_type == "bullish":
            setup_configs["mit_extreme"] = (zone_high, middle_low)
            setup_configs["mid_extreme"] = (fvg_midpoint, middle_low)
        else:
            setup_configs["mit_extreme"] = (zone_low, middle_high)
            setup_configs["mid_extreme"] = (fvg_midpoint, middle_high)

        # Simulate each setup
        for setup_name, (entry_price, stop_price) in setup_configs.items():
            risk = abs(entry_price - stop_price)
            if risk <= 0:
                continue

            risk_bps = (risk / ref_price) * 10000

            if setup_name == "mit_extreme":
                # Mitigation entry: walk starts right after mitigation
                max_exp = _walk_trade_bars(
                    df_walk, walk_start, exp_window,
                    fvg_type, entry_price, stop_price,
                )
            else:
                # mid_extreme: need to check if midpoint is reached
                mid_exp = _resolve_mid_extreme(
                    df_walk, walk_start, exp_window,
                    fvg_type, entry_price, stop_price, fvg_midpoint,
                    # Also check the mitigation bar itself
                    mit_bar_idx=walk_start - 1 if walk_start > 0 else None,
                )
                if mid_exp is None:
                    continue  # midpoint never reached
                max_exp = mid_exp

            max_exp_bps = (max_exp / ref_price) * 10000

            # Precompute win/loss at each R:R level
            outcomes = {}
            for n in N_VALUES:
                outcomes[n] = bool(max_exp >= n * risk)

            trades.append(TradeRecord(
                fvg_type=fvg_type,
                formation_time=time_c3,
                formation_price=ref_price,
                fvg_size=fvg_size,
                fvg_size_bps=fvg_size_bps,
                setup=setup_name,
                entry_price=entry_price,
                stop_price=stop_price,
                risk=risk,
                risk_bps=risk_bps,
                max_exp=max_exp,
                max_exp_bps=max_exp_bps,
                time_period=time_period,
                mitigation_time=mitigation_time,
                bars_to_mitigation=bars_to_mit,
                outcomes=outcomes,
            ))

    return trades


def _resolve_mid_extreme(
    df_walk: pd.DataFrame,
    walk_start: int,
    max_bars: int,
    fvg_type: str,
    entry_price: float,
    stop_price: float,
    midpoint: float,
    mit_bar_idx: Optional[int] = None,
) -> Optional[float]:
    """Check if midpoint is reached, then walk trade from activation.

    Returns max_exp (float) or None if midpoint never reached.
    """
    lows = df_walk["low"].values
    highs = df_walk["high"].values
    end_idx = min(walk_start + max_bars, len(df_walk))

    # Check mitigation bar itself for midpoint reach
    if mit_bar_idx is not None and 0 <= mit_bar_idx < len(df_walk):
        if fvg_type == "bullish" and lows[mit_bar_idx] <= midpoint:
            return _walk_trade_bars(
                df_walk, walk_start, max_bars,
                fvg_type, entry_price, stop_price,
            )
        if fvg_type == "bearish" and highs[mit_bar_idx] >= midpoint:
            return _walk_trade_bars(
                df_walk, walk_start, max_bars,
                fvg_type, entry_price, stop_price,
            )

    # Search post-mitigation bars for midpoint touch
    for j in range(walk_start, end_idx):
        if fvg_type == "bullish" and lows[j] <= midpoint:
            # Activation on bar j, walk starts from j+1
            remaining = max_bars - (j - walk_start + 1)
            if remaining <= 0:
                return 0.0
            return _walk_trade_bars(
                df_walk, j + 1, remaining,
                fvg_type, entry_price, stop_price,
            )
        if fvg_type == "bearish" and highs[j] >= midpoint:
            remaining = max_bars - (j - walk_start + 1)
            if remaining <= 0:
                return 0.0
            return _walk_trade_bars(
                df_walk, j + 1, remaining,
                fvg_type, entry_price, stop_price,
            )

    return None  # midpoint never reached


# ── Utility: summarize trades for sweep ranking ─────────────────────────

def summarize_trades(trades: list[TradeRecord]) -> dict:
    """Compute aggregate metrics from a list of TradeRecords.

    Returns dict with:
        total_fvgs, total_trades, per_setup breakdown,
        aggregate win rates / EV at each R:R level,
        sample distribution across time periods.
    """
    if not trades:
        return {
            "total_trades": 0,
            "setups": {},
            "time_periods_with_trades": 0,
        }

    result = {"total_trades": len(trades)}
    setup_groups = {}
    time_periods = set()

    for t in trades:
        time_periods.add(t.time_period)
        if t.setup not in setup_groups:
            setup_groups[t.setup] = []
        setup_groups[t.setup].append(t)

    result["time_periods_with_trades"] = len(time_periods)
    result["setups"] = {}

    for setup, group in setup_groups.items():
        n = len(group)
        risk_bps_vals = [t.risk_bps for t in group]
        avg_risk_bps = sum(risk_bps_vals) / n

        rr_stats = {}
        for nv in N_VALUES:
            wins = sum(1 for t in group if t.outcomes.get(nv, False))
            wr = wins / n
            ev = wr * (nv + 1) - 1
            rr_stats[nv] = {"wins": wins, "total": n, "win_rate": round(wr, 4), "ev": round(ev, 4)}

        # Best EV across R:R levels
        best_n = max(rr_stats, key=lambda k: rr_stats[k]["ev"])
        best_ev = rr_stats[best_n]["ev"]

        result["setups"][setup] = {
            "count": n,
            "avg_risk_bps": round(avg_risk_bps, 1),
            "rr_stats": rr_stats,
            "best_n": best_n,
            "best_ev": best_ev,
        }

    return result


# ── Utility: convert trades to dicts for JSON serialization ─────────────

def trades_to_dicts(trades: list[TradeRecord]) -> list[dict]:
    """Convert TradeRecords to plain dicts for JSON output."""
    out = []
    for t in trades:
        d = {
            "fvg_type": t.fvg_type,
            "formation_time": str(t.formation_time),
            "formation_price": round(t.formation_price, 2),
            "fvg_size": round(t.fvg_size, 2),
            "fvg_size_bps": round(t.fvg_size_bps, 2),
            "setup": t.setup,
            "entry_price": round(t.entry_price, 2),
            "stop_price": round(t.stop_price, 2),
            "risk": round(t.risk, 2),
            "risk_bps": round(t.risk_bps, 2),
            "max_exp": round(t.max_exp, 2),
            "max_exp_bps": round(t.max_exp_bps, 2),
            "time_period": t.time_period,
            "mitigation_time": str(t.mitigation_time),
            "bars_to_mitigation": t.bars_to_mitigation,
            "outcomes": {str(k): v for k, v in t.outcomes.items()},
        }
        out.append(d)
    return out
