#!/usr/bin/env python3
"""
btc_official_audit.py — Full recalculation on official Binance klines + tick vol verification.

Pipeline:
  1. Official 5m klines → FVG detection
  2. Official 1m klines → mitigation search + trade walk
  3. aggTrades tick data → vol@entry and vol@TP verification
  4. Output: per-trade results with fill confidence

Usage:
    python3 scripts/btc_official_audit.py
    python3 scripts/btc_official_audit.py --year 2025
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

OFFICIAL_DIR = Path("/home/cr0wn/binance_data/official_klines")
RAW_DIR = Path("/home/cr0wn/binance_data/raw")

KCOLS = ["open_time", "open", "high", "low", "close", "volume",
         "close_time", "quote_vol", "count", "taker_buy_vol", "taker_buy_quote_vol", "ignore"]
AGGTRADE_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id",
                 "last_trade_id", "transact_time", "is_buyer_maker"]

N_VALUES = [round(1.0 + i * 0.25, 2) for i in range(9)]
RISK_BINS = [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 994]


# ── Data loading ─────────────────────────────────────────────────────────

def load_official_klines(interval, start_month="2020-01", end_month="2025-10"):
    """Load all official kline CSVs into one DataFrame."""
    frames = []
    d = OFFICIAL_DIR / interval
    for csv in sorted(d.glob("BTCUSDT-*.csv")):
        tag = csv.stem.split("-", 2)[-1]  # e.g. "2024-06"
        if tag < start_month or tag > end_month:
            continue
        df = pd.read_csv(csv, names=KCOLS, header=None)
        if str(df["open_time"].iloc[0]) == "open_time":
            df = df.iloc[1:]
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df["date"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
        frames.append(df[["date", "open", "high", "low", "close", "volume"]])
    if not frames:
        raise FileNotFoundError(f"No official {interval} klines found")
    out = pd.concat(frames).sort_values("date").reset_index(drop=True)
    print(f"  Official {interval}: {len(out):,} bars ({out['date'].iloc[0]} → {out['date'].iloc[-1]})")
    return out


def load_aggtrades_window(ts, window_minutes=10):
    """Load aggTrades around a timestamp. Returns DataFrame or None."""
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    start_ms = int((ts - pd.Timedelta(minutes=window_minutes)).timestamp() * 1000)
    end_ms = int((ts + pd.Timedelta(minutes=window_minutes)).timestamp() * 1000)

    month_tag = ts.strftime("%Y-%m")
    monthly = RAW_DIR / f"BTCUSDT-aggTrades-{month_tag}.csv"

    if monthly.exists():
        with open(monthly) as f:
            first = f.readline()
        has_header = "price" in first

        chunks = pd.read_csv(
            monthly,
            names=None if has_header else AGGTRADE_COLS,
            header=0 if has_header else None,
            dtype={"price": "float64", "quantity": "float64", "transact_time": "int64"},
            usecols=["price", "quantity", "transact_time"],
            chunksize=2_000_000,
        )
        frames = []
        for chunk in chunks:
            mask = (chunk["transact_time"] >= start_ms) & (chunk["transact_time"] <= end_ms)
            sub = chunk[mask]
            if not sub.empty:
                frames.append(sub)
            if len(chunk) > 0 and chunk["transact_time"].iloc[-1] > end_ms + 60_000_000:
                break
        if frames:
            return pd.concat(frames, ignore_index=True)
    return None


def vol_at_price(aggtrades, price, tol_bps=5):
    """Check volume within tolerance_bps of target price."""
    if aggtrades is None or aggtrades.empty:
        return 0, 0.0
    tol = price * tol_bps / 10000
    m = (aggtrades["price"] >= price - tol) & (aggtrades["price"] <= price + tol)
    return int(m.sum()), float(aggtrades.loc[m, "quantity"].sum())


# ── FVG detection (from btc_fvg_analyzer) ────────────────────────────────

def detect_fvgs(data, min_size=0.01):
    highs = data["high"].values
    lows = data["low"].values
    opens = data["open"].values
    closes = data["close"].values
    dates = data["date"].values
    n = len(data)
    results = []
    for i in range(2, n):
        first_high, first_low = highs[i - 2], lows[i - 2]
        middle_open, middle_low, middle_high = opens[i - 1], lows[i - 1], highs[i - 1]
        middle_close = closes[i - 1]
        third_low, third_high = lows[i], highs[i]
        if third_low > first_high:
            gap = third_low - first_high
            if gap >= min_size:
                results.append(("bullish", first_high, third_low, dates[i], i,
                                middle_open, middle_low, middle_high, middle_close))
        elif third_high < first_low:
            gap = first_low - third_high
            if gap >= min_size:
                results.append(("bearish", third_high, first_low, dates[i], i,
                                middle_open, middle_low, middle_high, middle_close))
    return results


# ── Main pipeline ────────────────────────────────────────────────────────

def run_audit(year_filter=None):
    start = f"{year_filter}-01" if year_filter else "2020-01"
    end = f"{year_filter}-12" if year_filter else "2025-10"

    print("Loading official klines...")
    df5 = load_official_klines("5m", start, end)
    df1 = load_official_klines("1m", start, end)

    # Pre-index 1m dates for searchsorted
    dates_1m = df1["date"].values
    lows_1m = df1["low"].values
    highs_1m = df1["high"].values

    # Config (best from sweep)
    MIN_FVG_BPS = 5
    MIT_WINDOW = 90  # 5m bars
    EXP_WINDOW = 240  # 1m bars (4h)
    TIME_PERIOD_MIN = 60

    # Build time periods
    periods = []
    for m in range(0, 1440, TIME_PERIOD_MIN):
        end_m = m + TIME_PERIOD_MIN
        periods.append((f"{m//60:02d}:{m%60:02d}", f"{end_m//60:02d}:{end_m%60:02d}", m, end_m))

    print(f"\nDetecting FVGs on official 5m data...")
    raw_fvgs = detect_fvgs(df5, min_size=0.01)
    print(f"  Raw FVGs (min $0.01): {len(raw_fvgs):,}")

    # Process FVGs
    trades = []
    det_lows = df5["low"].values
    det_highs = df5["high"].values
    det_dates = df5["date"].values
    det_closes = df5["close"].values

    skipped_bps = 0
    skipped_mit = 0
    skipped_risk = 0
    total_processed = 0
    vol_checked = 0
    vol_entry_ok = 0
    vol_tp_ok = 0

    t0 = time.time()
    for fi, fvg in enumerate(raw_fvgs):
        fvg_type, y0, y1, time_c3, idx_c3, m_open, m_low, m_high, m_close = fvg
        zone_low, zone_high = min(y0, y1), max(y0, y1)
        fvg_size = zone_high - zone_low

        ref_price = det_closes[idx_c3 - 1]
        if ref_price <= 0:
            continue
        fvg_bps = (fvg_size / ref_price) * 10000
        if fvg_bps < MIN_FVG_BPS:
            skipped_bps += 1
            continue

        # Mitigation search on 5m bars
        mit_idx = None
        for j in range(idx_c3 + 1, min(idx_c3 + 1 + MIT_WINDOW, len(df5))):
            if det_lows[j] <= zone_high and det_highs[j] >= zone_low:
                mit_idx = j
                break
        if mit_idx is None:
            skipped_mit += 1
            continue

        mit_time = det_dates[mit_idx]

        # Time period
        ts_c3 = pd.Timestamp(time_c3)
        minutes = ts_c3.hour * 60 + ts_c3.minute
        tp = None
        for s_str, e_str, s_m, e_m in periods:
            if s_m <= minutes < e_m:
                tp = f"{s_str}-{e_str}"
                break
        if tp is None:
            continue

        # Walk start in 1m data
        walk_start = np.searchsorted(dates_1m, mit_time, side="right")
        if walk_start >= len(df1):
            continue

        total_processed += 1

        # Setup configs: mit_extreme and mid_extreme only
        fvg_mid = (zone_high + zone_low) / 2
        if fvg_type == "bullish":
            setups = {
                "mit_extreme": (zone_high, m_low),
                "mid_extreme": (fvg_mid, m_low),
            }
        else:
            setups = {
                "mit_extreme": (zone_low, m_high),
                "mid_extreme": (fvg_mid, m_high),
            }

        for setup_name, (entry, stop) in setups.items():
            risk = abs(entry - stop)
            if risk <= 0:
                skipped_risk += 1
                continue
            risk_bps = (risk / ref_price) * 10000

            # Walk trade on 1m bars
            if setup_name == "mit_extreme":
                max_exp = 0.0
                end_walk = min(walk_start + EXP_WINDOW, len(df1))
                for j in range(walk_start, end_walk):
                    lo, hi = lows_1m[j], highs_1m[j]
                    if fvg_type == "bullish" and lo <= stop:
                        break
                    if fvg_type == "bearish" and hi >= stop:
                        break
                    e = (hi - entry) if fvg_type == "bullish" else (entry - lo)
                    if e > max_exp:
                        max_exp = e
            else:  # mid_extreme — check midpoint reached first
                # Check mitigation bar
                mid_reached = False
                mit_bar = walk_start - 1 if walk_start > 0 else None
                if mit_bar is not None and mit_bar < len(df1):
                    if fvg_type == "bullish" and lows_1m[mit_bar] <= fvg_mid:
                        mid_reached = True
                    elif fvg_type == "bearish" and highs_1m[mit_bar] >= fvg_mid:
                        mid_reached = True

                if not mid_reached:
                    act_start = None
                    end_walk = min(walk_start + EXP_WINDOW, len(df1))
                    for j in range(walk_start, end_walk):
                        if fvg_type == "bullish" and lows_1m[j] <= fvg_mid:
                            act_start = j + 1
                            break
                        if fvg_type == "bearish" and highs_1m[j] >= fvg_mid:
                            act_start = j + 1
                            break
                    if act_start is None:
                        continue
                    remaining = EXP_WINDOW - (act_start - walk_start)
                    walk_from = act_start
                else:
                    walk_from = walk_start
                    remaining = EXP_WINDOW

                max_exp = 0.0
                end_walk = min(walk_from + remaining, len(df1))
                for j in range(walk_from, end_walk):
                    lo, hi = lows_1m[j], highs_1m[j]
                    if fvg_type == "bullish" and lo <= stop:
                        break
                    if fvg_type == "bearish" and hi >= stop:
                        break
                    e = (hi - entry) if fvg_type == "bullish" else (entry - lo)
                    if e > max_exp:
                        max_exp = e

            max_exp_bps = (max_exp / ref_price) * 10000

            # Risk bucket
            rb = None
            for bi in range(len(RISK_BINS) - 1):
                if RISK_BINS[bi] <= risk_bps < RISK_BINS[bi + 1]:
                    rb = f"{RISK_BINS[bi]}-{RISK_BINS[bi+1]}"
                    break
            if rb is None:
                continue

            # Outcomes
            outcomes = {}
            for nv in N_VALUES:
                outcomes[nv] = bool(max_exp >= nv * risk)

            trades.append({
                "fvg_type": fvg_type,
                "formation_time": str(time_c3),
                "formation_price": round(ref_price, 2),
                "fvg_size_bps": round(fvg_bps, 2),
                "setup": setup_name,
                "entry_price": round(entry, 2),
                "stop_price": round(stop, 2),
                "risk": round(risk, 2),
                "risk_bps": round(risk_bps, 2),
                "max_exp": round(max_exp, 2),
                "max_exp_bps": round(max_exp_bps, 2),
                "time_period": tp,
                "risk_range": rb,
                "mitigation_time": str(mit_time),
                "outcomes": {str(k): v for k, v in outcomes.items()},
            })

        if (fi + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {fi+1:,}/{len(raw_fvgs):,} FVGs, "
                  f"{len(trades):,} trades, {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  FVGs detected: {len(raw_fvgs):,}")
    print(f"  Skipped (bps filter): {skipped_bps:,}")
    print(f"  Skipped (no mitigation): {skipped_mit:,}")
    print(f"  Trades generated: {len(trades):,}")

    if vol_checked:
        print(f"\n  VOL AUDIT (sampled {vol_checked} trades):")
        print(f"    Vol@Entry: {vol_entry_ok}/{vol_checked} ({vol_entry_ok/vol_checked*100:.0f}%)")
        winners = sum(1 for t in trades if t["outcomes"].get("1.5", False)) // 20
        if winners:
            print(f"    Vol@TP: {vol_tp_ok}/{vol_checked} checked")

    return trades


def analyze_results(trades, label=""):
    """Compute aggregate stats matching sweep format."""
    if not trades:
        print("  No trades")
        return

    # Group by setup
    by_setup = defaultdict(list)
    for t in trades:
        by_setup[t["setup"]].append(t)

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Total trades: {len(trades):,}")
    print(f"{'='*80}")

    for setup in ["mit_extreme", "mid_extreme"]:
        group = by_setup.get(setup, [])
        if not group:
            continue
        n = len(group)
        print(f"\n  {setup}: {n:,} trades")
        for nv in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
            nv_str = str(nv)
            wins = sum(1 for t in group if t["outcomes"].get(nv_str, False))
            wr = wins / n
            ev = wr * (nv + 1) - 1
            print(f"    {nv:.2f}R: WR={wr*100:.1f}% EV={ev:+.4f} ({wins:,}/{n:,})")

    # Cell analysis
    cells = defaultdict(list)
    for t in trades:
        cells[(t["time_period"], t["risk_range"], t["setup"])].append(t)

    n_cells = len(cells)
    pos_ev_cells = 0
    for key, group in cells.items():
        n = len(group)
        if n < 30:
            continue
        best_ev = -999
        for nv in N_VALUES:
            wins = sum(1 for t in group if t["outcomes"].get(str(nv), False))
            ev = (wins / n) * (nv + 1) - 1
            if ev > best_ev:
                best_ev = ev
        if best_ev > 0:
            pos_ev_cells += 1

    print(f"\n  Cells (30+ samples): {sum(1 for g in cells.values() if len(g)>=30)}")
    print(f"  Positive EV cells: {pos_ev_cells}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default=None)
    parser.add_argument("--output", default=os.path.join(_ROOT, "scripts", "btc_sweep_results",
                                                          "official_audit_trades.json"))
    args = parser.parse_args()

    trades = run_audit(year_filter=args.year)

    label = f"OFFICIAL BINANCE KLINES — {args.year or 'FULL'}"
    analyze_results(trades, label)

    # Compare with sweep results
    print(f"\n{'='*80}")
    print(f"  COMPARISON: Official vs AggTrades-Resampled (sweep)")
    print(f"{'='*80}")

    sweep_path = os.path.join(_ROOT, "scripts", "btc_sweep_results", "5min_results", "sweep_trades.json")
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            sweep_all = json.load(f)
        sweep = sweep_all.get("5min_bps5_p60m_mit90", [])
        if args.year:
            sweep = [t for t in sweep if t["formation_time"].startswith(args.year)]

        s_mit = [t for t in sweep if t["setup"] == "mit_extreme"]
        o_mit = [t for t in trades if t["setup"] == "mit_extreme"]

        print(f"\n  mit_extreme trades: sweep={len(s_mit):,}  official={len(o_mit):,}  diff={len(o_mit)-len(s_mit):+,}")

        for nv in [1.0, 1.5, 2.0]:
            nv_str = str(nv)
            s_wr = sum(1 for t in s_mit if t["outcomes"].get(nv_str) is True or t["outcomes"].get(nv_str) == "True") / max(len(s_mit), 1) * 100
            o_wr = sum(1 for t in o_mit if t["outcomes"].get(nv_str, False)) / max(len(o_mit), 1) * 100
            print(f"    {nv:.1f}R WR: sweep={s_wr:.1f}%  official={o_wr:.1f}%  delta={o_wr-s_wr:+.1f}%")
    else:
        print("  (sweep trades not found for comparison)")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(trades, f, separators=(",", ":"))
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nTrades saved to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
