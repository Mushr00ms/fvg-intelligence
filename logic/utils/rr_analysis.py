"""
rr_analysis.py — Risk-Reward trade simulation for FVG setups.

Simulates 4 trade setups per mitigated FVG:
  - mit_open:    Entry @ mitigation level, Stop @ middle candle open
  - mit_extreme: Entry @ mitigation level, Stop @ middle candle extreme
  - mid_open:    Entry @ FVG midpoint,     Stop @ middle candle open
  - mid_extreme: Entry @ FVG midpoint,     Stop @ middle candle extreme

For each setup, sweeps RR targets n=1.0..3.0 (step 0.25) to compute
win rates and expectancy per time x size cell.
"""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

from .time_utils import create_time_intervals, assign_time_period_to_fvgs
from .fvg_analysis import calculate_fvg_size

ny_tz = pytz.timezone("America/New_York")

SETUPS = ["mit_open", "mit_extreme", "mid_open", "mid_extreme"]
SETUP_LABELS = {
    "mit_open": "Mitigation + Open Stop",
    "mit_extreme": "Mitigation + Extreme Stop",
    "mid_open": "Midpoint + Open Stop",
    "mid_extreme": "Midpoint + Extreme Stop",
}
N_VALUES = [round(1.0 + i * 0.25, 2) for i in range(9)]  # [1.0, 1.25, ..., 3.0]


# ---------------------------------------------------------------------------
# Per-FVG trade simulation
# ---------------------------------------------------------------------------

def _walk_trade(candles_df, fvg_type, entry_price, stop_price):
    """
    Walk candles and return max favorable expansion before stop is touched.

    Stop is touch-based:
      bullish → stop hit when candle low  <= stop_price
      bearish → stop hit when candle high >= stop_price
    """
    max_exp = 0.0
    for _, row in candles_df.iterrows():
        low = row["low"]
        high = row["high"]

        # Check stop first (touch-based, conservative: stop candle expansion excluded)
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

    return round(max_exp * 4) / 4


def simulate_fvg_trades(df_walk, fvg_data, fvg_filter_end_time):
    """
    Simulate all 4 trade setups for a single mitigated FVG.

    Args:
        df_walk: DataFrame to walk for stop/target checks (1min for precision)
        fvg_data: Series with FVG data (zone_high, zone_low, middle_*, etc.)
        fvg_filter_end_time: Session end time

    Returns dict keyed by setup name, each containing:
        activated : bool
        risk      : float (points, rounded to 0.25)
        max_exp   : float (max expansion before stop touch, rounded to 0.25)

    Returns None if FVG cannot be processed.
    """
    if not fvg_data["is_mitigated"]:
        return None

    fvg_type = fvg_data["fvg_type"]
    zone_low = fvg_data["zone_low"]
    zone_high = fvg_data["zone_high"]
    middle_open = fvg_data["middle_open"]
    middle_low = fvg_data["middle_low"]
    middle_high = fvg_data["middle_high"]
    mitigation_time = fvg_data["mitigation_time"]
    fvg_midpoint = (zone_high + zone_low) / 2

    # ----- Build post-mitigation window -----
    start_idx = df_walk["date"].searchsorted(mitigation_time, side="right")
    if start_idx >= len(df_walk):
        return None

    mitigation_date = mitigation_time.date()
    if hasattr(fvg_filter_end_time, "time"):
        end_time_obj = fvg_filter_end_time.time()
    else:
        end_time_obj = fvg_filter_end_time
    end_time = datetime.combine(mitigation_date, end_time_obj)
    end_time = ny_tz.localize(end_time)
    if mitigation_time >= end_time:
        return None

    end_idx = df_walk["date"].searchsorted(end_time, side="right")
    post_mit_df = df_walk.iloc[start_idx:end_idx]
    if post_mit_df.empty:
        return None

    # ----- Define entry and stop prices -----
    if fvg_type == "bullish":
        entry_mit = zone_high
        entry_mid = fvg_midpoint
        stop_open = middle_open
        stop_extreme = middle_low
    else:
        entry_mit = zone_low
        entry_mid = fvg_midpoint
        stop_open = middle_open
        stop_extreme = middle_high

    results = {}

    # ----- Mitigation entry setups (always activated on mitigated FVGs) -----
    for stop_key, stop_price in [("open", stop_open), ("extreme", stop_extreme)]:
        setup_key = f"mit_{stop_key}"
        risk = round(abs(entry_mit - stop_price) * 4) / 4

        if risk <= 0:
            results[setup_key] = {"activated": True, "risk": 0.0, "max_exp": 0.0}
            continue

        max_exp = _walk_trade(post_mit_df, fvg_type, entry_mit, stop_price)
        results[setup_key] = {"activated": True, "risk": risk, "max_exp": max_exp}

    # ----- Midpoint entry setups (conditional — price must reach midpoint) -----
    # Check mitigation candle itself for midpoint reach
    mid_reached_at_mitigation = False
    if start_idx > 0:
        mit_candle = df_walk.iloc[start_idx - 1]
        if fvg_type == "bullish" and mit_candle["low"] <= fvg_midpoint:
            mid_reached_at_mitigation = True
        elif fvg_type == "bearish" and mit_candle["high"] >= fvg_midpoint:
            mid_reached_at_mitigation = True

    if mid_reached_at_mitigation:
        # Midpoint reached on mitigation candle — walk starts from first post-mitigation candle
        post_activation_df = post_mit_df
    else:
        # Search subsequent candles for midpoint touch
        activation_offset = None
        for i, (_, row) in enumerate(post_mit_df.iterrows()):
            if fvg_type == "bullish" and row["low"] <= fvg_midpoint:
                activation_offset = i
                break
            elif fvg_type == "bearish" and row["high"] >= fvg_midpoint:
                activation_offset = i
                break

        if activation_offset is None:
            # Midpoint never reached — mid_* setups not activated
            for stop_key in ["open", "extreme"]:
                setup_key = f"mid_{stop_key}"
                stop_price = stop_open if stop_key == "open" else stop_extreme
                risk = round(abs(entry_mid - stop_price) * 4) / 4
                results[setup_key] = {"activated": False, "risk": risk, "max_exp": 0.0}
            return results

        # Walk starts from candle AFTER the activation candle
        post_activation_df = post_mit_df.iloc[activation_offset + 1:]

    for stop_key, stop_price in [("open", stop_open), ("extreme", stop_extreme)]:
        setup_key = f"mid_{stop_key}"
        risk = round(abs(entry_mid - stop_price) * 4) / 4

        if risk <= 0:
            results[setup_key] = {"activated": True, "risk": 0.0, "max_exp": 0.0}
            continue

        if post_activation_df.empty:
            results[setup_key] = {"activated": True, "risk": risk, "max_exp": 0.0}
            continue

        max_exp = _walk_trade(post_activation_df, fvg_type, entry_mid, stop_price)
        results[setup_key] = {"activated": True, "risk": risk, "max_exp": max_exp}

    return results


# ---------------------------------------------------------------------------
# Batch simulation: add per-FVG RR columns to DataFrame
# ---------------------------------------------------------------------------

RR_COLUMNS = []
for _s in SETUPS:
    RR_COLUMNS.append(f"rr_{_s}_activated")
    RR_COLUMNS.append(f"rr_{_s}_risk")
    RR_COLUMNS.append(f"rr_{_s}_max_exp")


def compute_rr_for_fvgs(df_fvgs, df_walk, filter_end_time, cache_file=None):
    """
    Run RR simulation for all mitigated FVGs.

    Args:
        df_fvgs: DataFrame with FVG data
        df_walk: DataFrame to walk for stop/target (1min candles for precision)
        filter_end_time: Session end time

    Adds columns: rr_{setup}_activated, rr_{setup}_risk, rr_{setup}_max_exp
    for each of the 4 setups.
    """
    missing = [c for c in RR_COLUMNS if c not in df_fvgs.columns]
    if not missing:
        return df_fvgs

    print(f"[INFO] Computing RR trade simulations...")

    # Initialise columns
    for col in RR_COLUMNS:
        if col not in df_fvgs.columns:
            if col.endswith("_activated"):
                df_fvgs[col] = False
            else:
                df_fvgs[col] = np.nan

    mitigated = df_fvgs[df_fvgs["is_mitigated"]]
    if mitigated.empty:
        return df_fvgs

    count = len(mitigated)
    print(f"[INFO] Simulating 4 trade setups for {count:,} mitigated FVGs...")

    for idx, row in mitigated.iterrows():
        result = simulate_fvg_trades(df_walk, row, filter_end_time)
        if result is None:
            continue

        for setup in SETUPS:
            if setup not in result:
                continue
            sr = result[setup]
            df_fvgs.loc[idx, f"rr_{setup}_activated"] = sr["activated"]
            df_fvgs.loc[idx, f"rr_{setup}_risk"] = sr["risk"]
            df_fvgs.loc[idx, f"rr_{setup}_max_exp"] = sr["max_exp"]

    if cache_file:
        df_fvgs.to_parquet(cache_file, index=False)
        print(f"[INFO] Updated cache with RR simulation data")

    return df_fvgs


# ---------------------------------------------------------------------------
# Aggregation: time x risk cells -> win rates / EV per setup x n
# ---------------------------------------------------------------------------

DEFAULT_RISK_BINS = [5, 10, 15, 20, 25, 30, 40, 80]


def aggregate_rr_cells(
    df_fvgs,
    fvg_filter_start_time,
    fvg_filter_end_time,
    interval_minutes=30,
    risk_bins=None,
    min_samples=5,
):
    """
    Aggregate per-FVG RR simulation results into time x risk cells.

    Args:
        df_fvgs:                DataFrame with RR simulation columns
        fvg_filter_start_time:  Session start time
        fvg_filter_end_time:    Session end time
        interval_minutes:       Time period width (e.g. 30 min)
        risk_bins:              List of bin edges (default: DEFAULT_RISK_BINS)
        min_samples:            Drop cells where no setup has >= this many samples

    Returns list of cell dicts:
        time_period    : str
        risk_range     : str   (e.g. "10.00-15.00")
        setups         : dict per setup -> {activated, valid, avg_risk, wins[], win_rates[], evs[]}
    """
    time_intervals = create_time_intervals(fvg_filter_start_time, fvg_filter_end_time, interval_minutes)
    df = df_fvgs.copy()
    df = assign_time_period_to_fvgs(df, time_intervals)

    if risk_bins is None:
        risk_bins = DEFAULT_RISK_BINS

    if len(risk_bins) < 2:
        return []

    # Only mitigated FVGs matter for RR
    mitigated = df[df["is_mitigated"] == True].copy()
    if mitigated.empty:
        return []

    cells = []

    for time_period in mitigated["time_period"].unique():
        if pd.isna(time_period):
            continue

        period_fvgs = mitigated[mitigated["time_period"] == time_period]

        for bi in range(len(risk_bins) - 1):
            bin_lo = float(risk_bins[bi])
            bin_hi = float(risk_bins[bi + 1])
            risk_range = f"{bin_lo:.0f}-{bin_hi:.0f}"

            setups_data = {}
            cell_total = 0

            for setup in SETUPS:
                col_act = f"rr_{setup}_activated"
                col_risk = f"rr_{setup}_risk"
                col_exp = f"rr_{setup}_max_exp"

                if col_act not in period_fvgs.columns:
                    setups_data[setup] = _empty_setup_result()
                    continue

                # Filter: activated, risk in this bin, valid data
                activated = period_fvgs[period_fvgs[col_act] == True]
                in_bin = activated[
                    activated[col_risk].notna() &
                    (activated[col_risk] >= bin_lo) &
                    (activated[col_risk] < bin_hi) &
                    activated[col_exp].notna()
                ]
                n_activated = len(activated[
                    activated[col_risk].notna() &
                    (activated[col_risk] >= bin_lo) &
                    (activated[col_risk] < bin_hi)
                ])
                n_valid = len(in_bin)
                cell_total = max(cell_total, n_valid)

                if n_valid == 0:
                    setups_data[setup] = _empty_setup_result(activated=n_activated)
                    continue

                median_risk = round(in_bin[col_risk].median(), 2)
                risks = in_bin[col_risk].values
                exps = in_bin[col_exp].values

                wins = []
                win_rates = []
                evs = []
                for n in N_VALUES:
                    w = int((exps >= n * risks).sum())
                    wr = round(w / n_valid * 100, 2)
                    ev = round(wr / 100 * (n + 1) - 1, 4)
                    wins.append(w)
                    win_rates.append(wr)
                    evs.append(ev)

                setups_data[setup] = {
                    "activated": n_activated,
                    "valid": n_valid,
                    "median_risk": median_risk,
                    "wins": wins,
                    "win_rates": win_rates,
                    "evs": evs,
                }

            # Skip cells where no setup has enough samples
            if cell_total >= min_samples:
                cells.append({
                    "time_period": time_period,
                    "risk_range": risk_range,
                    "sample_count": cell_total,
                    "setups": setups_data,
                })

    return cells


def _empty_setup_result(activated=0):
    """Return an empty result dict for a setup with no valid data."""
    return {
        "activated": activated,
        "valid": 0,
        "median_risk": None,
        "wins": [0] * len(N_VALUES),
        "win_rates": [None] * len(N_VALUES),
        "evs": [None] * len(N_VALUES),
    }


# ---------------------------------------------------------------------------
# Trade sample lookup: find matching FVGs for a cell and return trade details
# ---------------------------------------------------------------------------

def find_sample_trades(parquet_path, time_period, risk_range, setup, n_value, max_samples=20):
    """
    Find FVGs matching a heatmap cell and return trade details.

    Returns list of dicts with FVG zone, entry, stop, target, outcome, timestamps.
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["is_mitigated"] == True].copy()

    col_risk = f"rr_{setup}_risk"
    col_exp = f"rr_{setup}_max_exp"
    col_act = f"rr_{setup}_activated"

    if col_risk not in df.columns:
        return []

    # Parse risk range
    parts = risk_range.split("-")
    risk_lo, risk_hi = float(parts[0]), float(parts[1])

    # Filter by risk bin
    df = df[
        (df[col_act] == True) &
        df[col_risk].notna() & (df[col_risk] >= risk_lo) & (df[col_risk] < risk_hi) &
        df[col_exp].notna()
    ]

    if df.empty:
        return []

    # Filter by time period
    tp_start, tp_end = time_period.split("-")
    from datetime import time as dt_time
    tp_start_h, tp_start_m = map(int, tp_start.strip().split(":"))
    tp_end_h, tp_end_m = map(int, tp_end.strip().split(":"))
    tp_start_t = dt_time(tp_start_h, tp_start_m)
    tp_end_t = dt_time(tp_end_h, tp_end_m)

    mask = df["time_candle3"].apply(
        lambda dt: tp_start_t <= dt.time() < tp_end_t if pd.notna(dt) else False
    )
    df = df[mask]

    if df.empty:
        return []

    # Determine entry/stop/target per FVG
    trades = []
    sample = df.sample(n=min(max_samples, len(df)), random_state=42)

    for _, row in sample.iterrows():
        fvg_type = row["fvg_type"]
        zone_high = float(row["zone_high"])
        zone_low = float(row["zone_low"])
        midpoint = (zone_high + zone_low) / 2

        if "mit" in setup:
            entry = zone_high if fvg_type == "bullish" else zone_low
        else:
            entry = midpoint

        if "extreme" in setup:
            stop = float(row["middle_low"]) if fvg_type == "bullish" else float(row["middle_high"])
        else:
            stop = float(row["middle_open"])

        risk = float(row[col_risk])
        max_exp = float(row[col_exp])
        target = entry + n_value * risk if fvg_type == "bullish" else entry - n_value * risk
        won = max_exp >= n_value * risk

        trades.append({
            "fvg_type": fvg_type,
            "zone_high": zone_high,
            "zone_low": zone_low,
            "middle_open": float(row["middle_open"]),
            "middle_low": float(row["middle_low"]),
            "middle_high": float(row["middle_high"]),
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "risk": round(risk, 2),
            "max_exp": round(max_exp, 2),
            "won": won,
            "time_candle1": str(row["time_candle1"]),
            "time_candle2": str(row["time_candle2"]),
            "time_candle3": str(row["time_candle3"]),
            "mitigation_time": str(row["mitigation_time"]),
        })

    return trades


def load_candles_around_trade(cache_dir, trade, timeframe="5min", bars_before=10, bars_after=40):
    """
    Load OHLC candle data around a trade from databento cache.

    Returns list of {date, open, high, low, close} dicts.
    """
    import json as _json
    meta_path = os.path.join(cache_dir, "cache_metadata.json")
    if not os.path.exists(meta_path):
        return []

    with open(meta_path) as f:
        meta = _json.load(f)

    # Find a cache entry matching NQ + timeframe that covers the trade date
    from dateutil import parser as dtparser
    trade_date = dtparser.parse(trade["time_candle1"])

    target_file = None
    for key, entry in meta.get("entries", {}).items():
        if entry.get("symbol") == "NQ" and entry.get("timeframe") == timeframe:
            # Check if this cache covers the trade date
            try:
                cache_start = dtparser.parse(entry["start_date"])
                cache_end = dtparser.parse(entry["end_date"])
                if not (cache_start <= trade_date <= cache_end):
                    continue
            except (KeyError, ValueError):
                continue
            for tier in ["tier1", "tier2", "tier3_10yr"]:
                path = os.path.join(cache_dir, tier, f"{key}.parquet")
                if os.path.exists(path):
                    target_file = path
                    break
            if target_file:
                break

    if not target_file:
        return []

    df = pd.read_parquet(target_file)
    if "date" not in df.columns and df.index.name == "date":
        df = df.reset_index()

    # Find window around trade
    t3 = dtparser.parse(trade["time_candle1"])

    # Find nearest index
    idx = df["date"].searchsorted(t3)
    start = max(0, idx - bars_before)
    end = min(len(df), idx + bars_after)
    window = df.iloc[start:end]

    candles = []
    for _, row in window.iterrows():
        candles.append({
            "date": str(row["date"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        })

    return candles


# ---------------------------------------------------------------------------
# Storage: save / load RR datasets
# ---------------------------------------------------------------------------

_DEFAULT_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rr_data")


def _ensure_store(store_dir=None):
    d = store_dir or _DEFAULT_STORE_DIR
    os.makedirs(d, exist_ok=True)
    return d


def get_rr_dataset_id(ticker, timeframe_label, data_period, session_period_minutes):
    """Deterministic dataset ID for RR data."""
    return (
        f"rr_{ticker.lower()}_{timeframe_label}_{data_period}"
        f"_{session_period_minutes}min"
    )


def save_rr_dataset(
    cells,
    ticker,
    timeframe_label,
    data_period,
    session_period_minutes,
    store_dir=None,
):
    """
    Persist RR aggregation results to JSON.

    Returns the dataset ID.
    """
    store = _ensure_store(store_dir)

    # Normalise period string
    for long, short in [("years", "y"), ("year", "y"), ("months", "m"), ("month", "m"),
                        ("weeks", "w"), ("week", "w"), ("days", "d"), ("day", "d")]:
        data_period = data_period.replace(f" {long}", short).replace(long, short)
    data_period = data_period.replace(" ", "")

    dataset_id = get_rr_dataset_id(
        ticker, timeframe_label, data_period,
        session_period_minutes,
    )

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
            "created_at": datetime.now().isoformat(),
        },
        "cells": _clean_cells(cells),
    }

    data_file = os.path.join(store, f"{dataset_id}.json")
    tmp = data_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, data_file)

    # Update manifest
    manifest_path = os.path.join(store, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"datasets": []}

    manifest["datasets"] = [d for d in manifest["datasets"] if d["id"] != dataset_id]
    manifest["datasets"].append({
        "id": dataset_id,
        "ticker": ticker,
        "timeframe_label": timeframe_label,
        "data_period": data_period,
        "session_period_minutes": session_period_minutes,
        "created_at": payload["meta"]["created_at"],
    })
    manifest["last_updated"] = payload["meta"]["created_at"]

    tmp_m = manifest_path + ".tmp"
    with open(tmp_m, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_m, manifest_path)

    print(f"[INFO] RR dataset saved: {dataset_id}")
    return dataset_id


def load_rr_dataset(dataset_id, store_dir=None):
    """Load an RR dataset from JSON. Returns the full payload dict."""
    store = _ensure_store(store_dir)
    path = os.path.join(store, f"{dataset_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"RR dataset '{dataset_id}' not found at {path}")
    with open(path) as f:
        return json.load(f)


def get_rr_manifest(store_dir=None):
    """Return the RR manifest dict."""
    store = _ensure_store(store_dir)
    path = os.path.join(store, "manifest.json")
    if not os.path.exists(path):
        return {"datasets": []}
    with open(path) as f:
        return json.load(f)


def _clean_cells(cells):
    """Replace NaN / inf with None for JSON serialisation."""
    cleaned = []
    for cell in cells:
        c = {
            "time_period": cell["time_period"],
            "risk_range": cell["risk_range"],
            "sample_count": cell["sample_count"],
            "setups": {},
        }
        for setup, data in cell["setups"].items():
            c["setups"][setup] = {
                "activated": data["activated"],
                "valid": data["valid"],
                "median_risk": _safe(data["median_risk"]),
                "wins": data["wins"],
                "win_rates": [_safe(v) for v in data["win_rates"]],
                "evs": [_safe(v) for v in data["evs"]],
            }
        cleaned.append(c)
    return cleaned


def _safe(v):
    """Convert NaN/inf/None to None for JSON."""
    if v is None:
        return None
    try:
        import math
        if math.isnan(v) or math.isinf(v):
            return None
    except (TypeError, ValueError):
        pass
    return v
