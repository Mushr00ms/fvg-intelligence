#!/usr/bin/env python3
"""
btc_official_audit.py - Rebuild BTC trade candidates from official Binance futures klines.

Pipeline:
1. Official 5m klines -> FVG detection
2. Official 1m klines -> mitigation search + trade walk
3. Output: per-trade results compatible with the BTC strategy tooling
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DATA_DIR = ROOT / "data" / "binance_official"
DEFAULT_OUTPUT_DIR = ROOT / "scripts" / "btc_sweep_results"

KCOLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_vol",
    "count",
    "taker_buy_vol",
    "taker_buy_quote_vol",
    "ignore",
]

N_VALUES = [round(1.0 + i * 0.25, 2) for i in range(9)]
RISK_BINS = [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 994]


def load_official_klines(
    symbol: str,
    interval: str,
    data_dir: Path,
    *,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    frames = []
    src = data_dir / symbol / interval
    for csv_path in sorted(src.glob(f"{symbol}-{interval}-*.csv")):
        tag = csv_path.stem.split("-", 2)[-1]
        if tag < start_month or tag > end_month:
            continue
        df = pd.read_csv(csv_path, names=KCOLS, header=None)
        if str(df["open_time"].iloc[0]) == "open_time":
            df = df.iloc[1:]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["date"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
        frames.append(df[["date", "open", "high", "low", "close", "volume"]])

    if not frames:
        raise FileNotFoundError(f"No official {symbol} {interval} klines found under {src}")

    out = pd.concat(frames).sort_values("date").reset_index(drop=True)
    print(f"  Official {symbol} {interval}: {len(out):,} bars ({out['date'].iloc[0]} -> {out['date'].iloc[-1]})")
    return out


def detect_fvgs(data: pd.DataFrame, min_size: float = 0.01) -> list[tuple]:
    highs = data["high"].values
    lows = data["low"].values
    opens = data["open"].values
    closes = data["close"].values
    dates = data["date"].values
    n = len(data)
    results = []
    for idx in range(2, n):
        first_high, first_low = highs[idx - 2], lows[idx - 2]
        middle_open, middle_low, middle_high = opens[idx - 1], lows[idx - 1], highs[idx - 1]
        middle_close = closes[idx - 1]
        third_low, third_high = lows[idx], highs[idx]

        if third_low > first_high:
            gap = third_low - first_high
            if gap >= min_size:
                results.append(
                    ("bullish", first_high, third_low, dates[idx], idx, middle_open, middle_low, middle_high, middle_close)
                )
        elif third_high < first_low:
            gap = first_low - third_high
            if gap >= min_size:
                results.append(
                    ("bearish", third_high, first_low, dates[idx], idx, middle_open, middle_low, middle_high, middle_close)
                )
    return results


def run_audit(
    *,
    symbol: str,
    data_dir: Path,
    year_filter: str | None,
    start_month: str | None,
    end_month: str | None,
) -> list[dict]:
    start = start_month or (f"{year_filter}-01" if year_filter else "2020-01")
    end = end_month or (f"{year_filter}-12" if year_filter else "2026-12")

    print("Loading official klines...")
    df5 = load_official_klines(symbol, "5m", data_dir, start_month=start, end_month=end)
    df1 = load_official_klines(symbol, "1m", data_dir, start_month=start, end_month=end)

    dates_1m = df1["date"].values
    lows_1m = df1["low"].values
    highs_1m = df1["high"].values

    min_fvg_bps = 5
    mit_window = 90
    exp_window = 240
    time_period_minutes = 60

    periods = []
    for minute in range(0, 1440, time_period_minutes):
        end_minute = minute + time_period_minutes
        periods.append((f"{minute // 60:02d}:{minute % 60:02d}", f"{end_minute // 60:02d}:{end_minute % 60:02d}", minute, end_minute))

    print("\nDetecting FVGs on official 5m data...")
    raw_fvgs = detect_fvgs(df5, min_size=0.01)
    print(f"  Raw FVGs (min $0.01): {len(raw_fvgs):,}")

    det_lows = df5["low"].values
    det_highs = df5["high"].values
    det_dates = df5["date"].values
    det_closes = df5["close"].values

    trades = []
    skipped_bps = 0
    skipped_mit = 0
    skipped_risk = 0
    start_ts = time.time()

    for fvg_index, fvg in enumerate(raw_fvgs):
        fvg_type, y0, y1, time_c3, idx_c3, middle_open, middle_low, middle_high, _middle_close = fvg
        zone_low, zone_high = min(y0, y1), max(y0, y1)
        fvg_size = zone_high - zone_low

        ref_price = det_closes[idx_c3 - 1]
        if ref_price <= 0:
            continue
        fvg_bps = (fvg_size / ref_price) * 10_000
        if fvg_bps < min_fvg_bps:
            skipped_bps += 1
            continue

        mitigation_idx = None
        for walk_idx in range(idx_c3 + 1, min(idx_c3 + 1 + mit_window, len(df5))):
            if det_lows[walk_idx] <= zone_high and det_highs[walk_idx] >= zone_low:
                mitigation_idx = walk_idx
                break
        if mitigation_idx is None:
            skipped_mit += 1
            continue

        mitigation_time = det_dates[mitigation_idx]
        ts_c3 = pd.Timestamp(time_c3)
        minutes = ts_c3.hour * 60 + ts_c3.minute
        time_period = None
        for start_label, end_label, start_minute, end_minute in periods:
            if start_minute <= minutes < end_minute:
                time_period = f"{start_label}-{end_label}"
                break
        if time_period is None:
            continue

        walk_start = np.searchsorted(dates_1m, mitigation_time, side="right")
        if walk_start >= len(df1):
            continue

        fvg_mid = (zone_high + zone_low) / 2
        if fvg_type == "bullish":
            setups = {
                "mit_extreme": (zone_high, middle_low),
                "mid_extreme": (fvg_mid, middle_low),
            }
        else:
            setups = {
                "mit_extreme": (zone_low, middle_high),
                "mid_extreme": (fvg_mid, middle_high),
            }

        for setup_name, (entry, stop) in setups.items():
            risk = abs(entry - stop)
            if risk <= 0:
                skipped_risk += 1
                continue
            risk_bps = (risk / ref_price) * 10_000

            if setup_name == "mit_extreme":
                walk_from = walk_start
                remaining = exp_window
            else:
                mid_reached = False
                mit_bar = walk_start - 1 if walk_start > 0 else None
                if mit_bar is not None and mit_bar < len(df1):
                    if fvg_type == "bullish" and lows_1m[mit_bar] <= fvg_mid:
                        mid_reached = True
                    elif fvg_type == "bearish" and highs_1m[mit_bar] >= fvg_mid:
                        mid_reached = True

                if mid_reached:
                    walk_from = walk_start
                    remaining = exp_window
                else:
                    activation_idx = None
                    for minute_idx in range(walk_start, min(walk_start + exp_window, len(df1))):
                        if fvg_type == "bullish" and lows_1m[minute_idx] <= fvg_mid:
                            activation_idx = minute_idx + 1
                            break
                        if fvg_type == "bearish" and highs_1m[minute_idx] >= fvg_mid:
                            activation_idx = minute_idx + 1
                            break
                    if activation_idx is None:
                        continue
                    walk_from = activation_idx
                    remaining = exp_window - (activation_idx - walk_start)

            max_exp = 0.0
            for minute_idx in range(walk_from, min(walk_from + remaining, len(df1))):
                low_1m = lows_1m[minute_idx]
                high_1m = highs_1m[minute_idx]
                if fvg_type == "bullish" and low_1m <= stop:
                    break
                if fvg_type == "bearish" and high_1m >= stop:
                    break
                excursion = (high_1m - entry) if fvg_type == "bullish" else (entry - low_1m)
                if excursion > max_exp:
                    max_exp = excursion

            max_exp_bps = (max_exp / ref_price) * 10_000
            risk_range = None
            for idx in range(len(RISK_BINS) - 1):
                if RISK_BINS[idx] <= risk_bps < RISK_BINS[idx + 1]:
                    risk_range = f"{RISK_BINS[idx]}-{RISK_BINS[idx + 1]}"
                    break
            if risk_range is None:
                continue

            outcomes = {str(n_value): bool(max_exp >= n_value * risk) for n_value in N_VALUES}
            trades.append(
                {
                    "symbol": symbol,
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
                    "time_period": time_period,
                    "risk_range": risk_range,
                    "mitigation_time": str(mitigation_time),
                    "outcomes": outcomes,
                }
            )

        if (fvg_index + 1) % 10_000 == 0:
            elapsed = time.time() - start_ts
            print(f"  Processed {fvg_index + 1:,}/{len(raw_fvgs):,} FVGs, {len(trades):,} trades, {elapsed:.0f}s")

    elapsed = time.time() - start_ts
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  FVGs detected: {len(raw_fvgs):,}")
    print(f"  Skipped (bps filter): {skipped_bps:,}")
    print(f"  Skipped (no mitigation): {skipped_mit:,}")
    print(f"  Skipped (invalid risk): {skipped_risk:,}")
    print(f"  Trades generated: {len(trades):,}")
    return trades


def analyze_results(trades: list[dict], label: str = "") -> None:
    if not trades:
        print("  No trades")
        return

    by_setup = defaultdict(list)
    for trade in trades:
        by_setup[trade["setup"]].append(trade)

    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"  Total trades: {len(trades):,}")
    print(f"{'=' * 80}")

    for setup in ["mit_extreme", "mid_extreme"]:
        group = by_setup.get(setup, [])
        if not group:
            continue
        print(f"\n  {setup}: {len(group):,} trades")
        for n_value in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
            n_key = str(n_value)
            wins = sum(1 for trade in group if trade["outcomes"].get(n_key, False))
            win_rate = wins / len(group)
            ev = win_rate * (n_value + 1) - 1
            print(f"    {n_value:.2f}R: WR={win_rate * 100:.1f}% EV={ev:+.4f} ({wins:,}/{len(group):,})")

    cells = defaultdict(list)
    for trade in trades:
        cells[(trade["time_period"], trade["risk_range"], trade["setup"])].append(trade)

    pos_ev_cells = 0
    eligible_cells = 0
    for group in cells.values():
        if len(group) < 30:
            continue
        eligible_cells += 1
        best_ev = -999.0
        for n_value in N_VALUES:
            wins = sum(1 for trade in group if trade["outcomes"].get(str(n_value), False))
            ev = (wins / len(group)) * (n_value + 1) - 1
            if ev > best_ev:
                best_ev = ev
        if best_ev > 0:
            pos_ev_cells += 1

    print(f"\n  Cells (30+ samples): {eligible_cells}")
    print(f"  Positive EV cells: {pos_ev_cells}")


def default_output_path(symbol: str, year: str | None, start_month: str | None, end_month: str | None) -> Path:
    if year:
        return DEFAULT_OUTPUT_DIR / f"official_audit_trades_{symbol}_{year}.json"
    if start_month or end_month:
        return DEFAULT_OUTPUT_DIR / f"official_audit_trades_{symbol}_{start_month or 'start'}_{end_month or 'end'}.json"
    return DEFAULT_OUTPUT_DIR / f"official_audit_trades_{symbol}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--year", default=None)
    parser.add_argument("--start-month", default=None, help="YYYY-MM")
    parser.add_argument("--end-month", default=None, help="YYYY-MM")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else default_output_path(args.symbol, args.year, args.start_month, args.end_month)
    trades = run_audit(
        symbol=args.symbol,
        data_dir=Path(args.data_dir),
        year_filter=args.year,
        start_month=args.start_month,
        end_month=args.end_month,
    )

    label_window = args.year or f"{args.start_month or 'FULL'}->{args.end_month or 'LATEST'}"
    analyze_results(trades, f"OFFICIAL BINANCE KLINES - {args.symbol} - {label_window}")

    if args.symbol == "BTCUSDT":
        print(f"\n{'=' * 80}")
        print("  COMPARISON: Official vs AggTrades-Resampled (sweep)")
        print(f"{'=' * 80}")
        sweep_path = ROOT / "scripts" / "btc_sweep_results" / "5min_results" / "sweep_trades.json"
        if sweep_path.exists():
            with open(sweep_path) as f:
                sweep_all = json.load(f)
            sweep = sweep_all.get("5min_bps5_p60m_mit90", [])
            if args.year:
                sweep = [trade for trade in sweep if trade["formation_time"].startswith(args.year)]
            sweep_mit = [trade for trade in sweep if trade["setup"] == "mit_extreme"]
            official_mit = [trade for trade in trades if trade["setup"] == "mit_extreme"]
            print(
                f"\n  mit_extreme trades: sweep={len(sweep_mit):,}  "
                f"official={len(official_mit):,}  diff={len(official_mit) - len(sweep_mit):+,}"
            )
            for n_value in [1.0, 1.5, 2.0]:
                n_key = str(n_value)
                sweep_wr = sum(
                    1 for trade in sweep_mit if trade["outcomes"].get(n_key) is True or trade["outcomes"].get(n_key) == "True"
                ) / max(len(sweep_mit), 1) * 100
                official_wr = sum(1 for trade in official_mit if trade["outcomes"].get(n_key, False)) / max(len(official_mit), 1) * 100
                print(f"    {n_value:.1f}R WR: sweep={sweep_wr:.1f}%  official={official_wr:.1f}%  delta={official_wr - sweep_wr:+.1f}%")
        else:
            print("  (sweep trades not found for comparison)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trades, f, separators=(",", ":"))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nTrades saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
