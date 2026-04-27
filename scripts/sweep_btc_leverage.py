#!/usr/bin/env python3
"""
sweep_btc_leverage.py - Leverage-aware simulation for BTC FVG strategy.

Computes precise trade exit times from official 1m klines, then sweeps
leverage levels to find the minimum required and the liquidation-safe ceiling.

Model (Binance cross-margin perpetuals):
  - Starting equity : $10,000 (configurable)
  - Risk per trade  : 1% of starting equity by default
                      1% of current equity when --compound is enabled
  - Notional        : risk_dollar * 10_000 / risk_bps
  - Initial margin  : notional / leverage
  - Free margin     : equity - sum(initial_margins of open positions)
  - Missed trade    : initial_margin > free_margin (can't open)
  - Win             : equity += risk_dollar * best_n          (TP maker = 0 fee)
  - Loss            : equity -= risk_dollar + notional*0.0004 (SL taker fee)
  - Liquidation     : equity < total_open_notional * 0.004    (maint. margin 0.4%)

Usage:
    python3 scripts/sweep_btc_leverage.py --year 2025
    python3 scripts/sweep_btc_leverage.py             # full data range
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

KLINES_DIR = Path("/home/cr0wn/binance_data/official_klines")
DEFAULT_STRATEGY_PATH = _ROOT / "bot/strategies/btc-5min-5xlev-ev007-s30-both.json"
OUT_DIR = _ROOT / "scripts/btc_sweep_results"

KCOLS = ["open_time", "open", "high", "low", "close", "volume",
         "close_time", "quote_vol", "count", "taker_buy_vol",
         "taker_buy_quote_vol", "ignore"]

RISK_BINS    = [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 994]
N_VALUES     = [round(1.0 + i * 0.25, 2) for i in range(9)]
SL_FEE       = 0.0004   # taker stop-market
MAINT_MARGIN = 0.004    # Binance BTC maintenance margin rate
START_EQUITY = 10_000.0
RISK_PCT     = 0.01
EXP_WINDOW   = 240      # 1m bars (4h)
MIT_WINDOW   = 90       # 5m bars


# ── Data loading ──────────────────────────────────────────────────────────

def load_klines(interval: str, start_month: str, end_month: str) -> pd.DataFrame:
    d = KLINES_DIR / interval
    frames = []
    for csv in sorted(d.glob("BTCUSDT-*.csv")):
        tag = csv.stem.rsplit("-", 2)[-2] + "-" + csv.stem.rsplit("-", 1)[-1]
        if tag < start_month or tag > end_month:
            continue
        df = pd.read_csv(csv, names=KCOLS, header=None)
        if str(df["open_time"].iloc[0]) == "open_time":
            df = df.iloc[1:]
        for c in ("open", "high", "low", "close"):
            df[c] = df[c].astype(float)
        df["date"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms", utc=True)
        frames.append(df[["date", "open", "high", "low", "close"]])
    if not frames:
        raise FileNotFoundError(f"No {interval} klines for {start_month}->{end_month}")
    out = pd.concat(frames).sort_values("date").reset_index(drop=True)
    print(f"  {interval}: {len(out):,} bars  "
          f"({out['date'].iloc[0].strftime('%Y-%m-%d')} -> "
          f"{out['date'].iloc[-1].strftime('%Y-%m-%d')})")
    return out


# ── Strategy cell lookup ──────────────────────────────────────────────────

def load_cell_lookup(strategy_path: Path) -> dict:
    with open(strategy_path) as f:
        s = json.load(f)
    lookup = {}
    for cell in s["cells"]:
        key = (cell["time_period"], cell["risk_range"], cell["setup"])
        lookup[key] = {
            "best_n": cell["best_n"],
            "avg_risk_bps": cell["avg_risk_bps"],
        }
    print(f"  Strategy cells loaded: {len(lookup)}")
    return lookup


# ── FVG detection (inlined, minimal) ─────────────────────────────────────

def detect_fvgs(data: pd.DataFrame) -> list:
    highs  = data["high"].values
    lows   = data["low"].values
    opens  = data["open"].values
    closes = data["close"].values
    dates  = data["date"].values
    results = []
    for i in range(2, len(data)):
        fh, fl = highs[i - 2], lows[i - 2]
        mo, ml, mh = opens[i - 1], lows[i - 1], highs[i - 1]
        mc = closes[i - 1]
        tl, th = lows[i], highs[i]
        if tl > fh:
            results.append(("bullish", fh, tl, dates[i], i, mo, ml, mh, mc))
        elif th < fl:
            results.append(("bearish", th, fl, dates[i], i, mo, ml, mh, mc))
    return results


# ── Trade detection + exit time walk ─────────────────────────────────────

def run_trade_walk(
    df5: pd.DataFrame,
    df1: pd.DataFrame,
    cell_lookup: dict,
    year_filter: str | None,
) -> list[dict]:
    """
    Detect qualifying FVGs on 5m, walk on 1m, record precise exit_time.
    Returns list of trade dicts with entry_time, exit_time, won, etc.
    """
    lows_5  = df5["low"].values
    highs_5 = df5["high"].values
    dates_5 = df5["date"].values
    closes_5 = df5["close"].values

    lows_1  = df1["low"].values
    highs_1 = df1["high"].values
    dates_1 = df1["date"].values

    periods = []
    for m in range(0, 1440, 60):
        s = f"{m//60:02d}:{m%60:02d}"
        e = f"{(m+60)//60:02d}:{(m+60)%60:02d}"
        periods.append((s, e, m, m + 60))

    raw_fvgs = detect_fvgs(df5)
    print(f"  Raw FVGs detected: {len(raw_fvgs):,}")

    trades = []
    skipped_bps = skipped_mit = skipped_cell = skipped_risk = 0
    t0 = time.time()

    for fi, fvg in enumerate(raw_fvgs):
        fvg_type, y0, y1, time_c3, idx_c3, m_open, m_low, m_high, m_close = fvg

        # Optional year filter on formation time
        if year_filter:
            ts = pd.Timestamp(time_c3)
            if str(ts.year) != year_filter:
                continue

        zone_low  = min(y0, y1)
        zone_high = max(y0, y1)
        fvg_size  = zone_high - zone_low
        ref_price = closes_5[idx_c3 - 1]
        if ref_price <= 0:
            continue

        fvg_bps = fvg_size / ref_price * 10_000
        if fvg_bps < 5:
            skipped_bps += 1
            continue

        # Time period
        ts_c3   = pd.Timestamp(time_c3)
        minutes = ts_c3.hour * 60 + ts_c3.minute
        tp_label = None
        for s_str, e_str, s_m, e_m in periods:
            if s_m <= minutes < e_m:
                tp_label = f"{s_str}-{e_str}"
                break
        if tp_label is None:
            continue

        # Mitigation on 5m
        mit_idx = None
        for j in range(idx_c3 + 1, min(idx_c3 + 1 + MIT_WINDOW, len(df5))):
            if lows_5[j] <= zone_high and highs_5[j] >= zone_low:
                mit_idx = j
                break
        if mit_idx is None:
            skipped_mit += 1
            continue

        mit_time   = dates_5[mit_idx]
        walk_start = int(np.searchsorted(dates_1, mit_time, side="right"))
        if walk_start >= len(df1):
            continue

        fvg_mid = (zone_high + zone_low) / 2

        if fvg_type == "bullish":
            setups_cfg = {
                "mit_extreme": (zone_high, m_low),
                "mid_extreme": (fvg_mid,   m_low),
            }
        else:
            setups_cfg = {
                "mit_extreme": (zone_low,  m_high),
                "mid_extreme": (fvg_mid,   m_high),
            }

        for setup_name, (entry, stop) in setups_cfg.items():
            risk = abs(entry - stop)
            if risk <= 0:
                skipped_risk += 1
                continue

            risk_bps = risk / ref_price * 10_000

            # Risk bucket
            rb = None
            for bi in range(len(RISK_BINS) - 1):
                if RISK_BINS[bi] <= risk_bps < RISK_BINS[bi + 1]:
                    rb = f"{RISK_BINS[bi]}-{RISK_BINS[bi+1]}"
                    break
            if rb is None:
                continue

            cell_key = (tp_label, rb, setup_name)
            if cell_key not in cell_lookup:
                skipped_cell += 1
                continue

            cell   = cell_lookup[cell_key]
            best_n = cell["best_n"]
            target = entry + best_n * risk if fvg_type == "bullish" else entry - best_n * risk

            # Walk on 1m bars, record exit_time precisely
            # Convention: check SL first on same bar (conservative)
            won       = False
            exit_time = None
            end_walk  = min(walk_start + EXP_WINDOW, len(df1))

            # mid_extreme: find midpoint activation first
            actual_walk_start = walk_start
            if setup_name == "mid_extreme":
                activated = False
                # Check mitigation bar itself
                mit_bar = walk_start - 1 if walk_start > 0 else None
                if mit_bar is not None:
                    if fvg_type == "bullish" and lows_1[mit_bar] <= fvg_mid:
                        activated = True
                    elif fvg_type == "bearish" and highs_1[mit_bar] >= fvg_mid:
                        activated = True

                if not activated:
                    for j in range(walk_start, end_walk):
                        if fvg_type == "bullish" and lows_1[j] <= fvg_mid:
                            actual_walk_start = j + 1
                            activated = True
                            break
                        if fvg_type == "bearish" and highs_1[j] >= fvg_mid:
                            actual_walk_start = j + 1
                            activated = True
                            break
                if not activated:
                    continue  # midpoint never reached

            for j in range(actual_walk_start, end_walk):
                lo, hi = lows_1[j], highs_1[j]
                # Check SL first (conservative)
                if fvg_type == "bullish":
                    if lo <= stop:
                        exit_time = dates_1[j]
                        won = False
                        break
                    if hi >= target:
                        exit_time = dates_1[j]
                        won = True
                        break
                else:
                    if hi >= stop:
                        exit_time = dates_1[j]
                        won = False
                        break
                    if lo <= target:
                        exit_time = dates_1[j]
                        won = True
                        break

            # Timeout: treat as loss, exit at last walk bar
            if exit_time is None:
                last_bar = min(actual_walk_start + EXP_WINDOW - 1, len(df1) - 1)
                exit_time = dates_1[last_bar]
                won = False

            trades.append({
                "entry_time": pd.Timestamp(mit_time),
                "exit_time":  pd.Timestamp(exit_time),
                "setup":      setup_name,
                "risk_bps":   round(risk_bps, 3),
                "best_n":     best_n,
                "won":        won,
            })

        if (fi + 1) % 20_000 == 0:
            print(f"    FVGs processed: {fi+1:,}/{len(raw_fvgs):,}  "
                  f"trades so far: {len(trades):,}  "
                  f"elapsed: {time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    print(f"  Walk complete in {elapsed:.0f}s")
    print(f"  Skipped - bps:{skipped_bps}  mit:{skipped_mit}  "
          f"cell:{skipped_cell}  risk:{skipped_risk}")
    print(f"  Qualifying trades: {len(trades):,}")
    return trades


# ── Leverage simulation ───────────────────────────────────────────────────

def simulate_leverage(trades: list[dict], leverage: float,
                      start_equity: float = START_EQUITY,
                      compound: bool = False) -> dict:
    """
    Simulate the strategy at a given leverage level.

    Uses fixed risk dollar by default (1% of starting equity), or compounds
    risk_dollar from current equity when compound=True.

    Trades are sorted by entry_time. Open positions are tracked by their
    exit_time. At each new trade entry, we check if margin is available.
    P&L updates equity when a position closes. Drawdown is the standard
    realized peak-to-trough drawdown on closed equity.

    Returns summary dict.
    """
    trades_sorted = sorted(trades, key=lambda t: t["entry_time"])

    equity    = start_equity
    peak      = equity
    min_equity = equity
    max_dd    = 0.0
    wins = losses = missed = liq_events = 0
    total_fees = 0.0
    daily_pnl  = defaultdict(float)
    peak_open_loss_pct = 0.0
    peak_margin_usage_pct = 0.0
    max_trade_notional = 0.0

    # Open positions: list of (exit_time, notional, risk_dollar, best_n, won)
    open_positions: list[tuple] = []

    def _update_open_exposure_stats():
        nonlocal peak_open_loss_pct, peak_margin_usage_pct
        if equity <= 0 or not open_positions:
            return
        open_notional = sum(p[1] for p in open_positions)
        open_margin = open_notional / leverage
        # Worst-case realized loss if every open position stops out here.
        open_loss = sum(p[2] + p[1] * SL_FEE for p in open_positions)
        peak_open_loss_pct = max(peak_open_loss_pct, open_loss / equity * 100)
        peak_margin_usage_pct = max(peak_margin_usage_pct, open_margin / equity * 100)

    def close_expired(current_time):
        nonlocal equity, wins, losses, total_fees, peak, min_equity, max_dd
        still_open = []
        for pos in open_positions:
            exit_t, notional, rd, best_n, won = pos
            if exit_t <= current_time:
                if won:
                    pnl = rd * best_n
                    fee = 0.0
                    wins += 1
                else:
                    fee = notional * SL_FEE
                    pnl = -rd - fee
                    losses += 1
                equity += pnl
                total_fees += fee
                daily_pnl[exit_t.strftime("%Y-%m-%d")] += pnl
                if equity > peak:
                    peak = equity
                if equity < min_equity:
                    min_equity = equity
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            else:
                still_open.append(pos)
        open_positions[:] = still_open

    for t in trades_sorted:
        close_expired(t["entry_time"])

        # Current open notional and margin
        open_notional = sum(p[1] for p in open_positions)
        open_margin   = open_notional / leverage

        # Liquidation check: equity < open_notional * maint_margin
        if open_notional > 0 and equity < open_notional * MAINT_MARGIN:
            liq_events += 1
            equity = max(0.0, open_notional * MAINT_MARGIN)
            min_equity = min(min_equity, equity)
            open_positions.clear()
            open_notional = 0.0
            open_margin   = 0.0

        if equity <= 0:
            missed += 1
            continue

        risk_dollar    = equity * RISK_PCT if compound else start_equity * RISK_PCT
        notional       = risk_dollar * 10_000 / t["risk_bps"]
        initial_margin = notional / leverage
        free_margin    = equity - open_margin

        if initial_margin > free_margin:
            missed += 1
            continue

        if notional > max_trade_notional:
            max_trade_notional = notional
        open_positions.append((
            t["exit_time"],
            notional,
            risk_dollar,
            t["best_n"],
            t["won"],
        ))
        _update_open_exposure_stats()

    # Close all remaining positions at end of period
    if open_positions:
        last_exit = max(p[0] for p in open_positions)
        close_expired(last_exit + pd.Timedelta(seconds=1))

    total   = wins + losses
    taken   = total + 0  # missed trades not counted in total
    all_    = total + missed

    dpnl = np.array(list(daily_pnl.values()))
    sharpe = (dpnl.mean() / dpnl.std() * math.sqrt(365)
              if len(dpnl) > 1 and dpnl.std() > 0 else 0.0)

    # Peak concurrent open positions
    # (approximate from trade duration overlap — recompute quickly)
    peak_concurrent = _peak_concurrent(trades_sorted)

    return {
        "leverage":         leverage,
        "trades_total":     all_,
        "trades_taken":     total,
        "missed":           missed,
        "missed_pct":       round(missed / all_ * 100, 1) if all_ else 0,
        "wins":             wins,
        "losses":           losses,
        "win_rate":         round(wins / total * 100, 1) if total else 0,
        "liq_events":       liq_events,
        "final_equity":     round(equity, 2),
        "peak_equity":      round(peak, 2),
        "min_equity":       round(min_equity, 2),
        "net_pnl":          round(equity - start_equity, 2),
        "total_fees":       round(total_fees, 2),
        "max_dd_pct":       round(max_dd, 2),
        "dd_from_start_pct": round(max(0.0, (start_equity - min_equity) / start_equity * 100), 2),
        "sharpe":           round(sharpe, 3),
        "peak_concurrent":  peak_concurrent,
        "peak_open_loss_pct": round(peak_open_loss_pct, 2),
        "peak_margin_usage_pct": round(peak_margin_usage_pct, 2),
        "max_trade_notional": round(max_trade_notional, 2),
    }


def _peak_concurrent(trades_sorted: list[dict]) -> int:
    """Find maximum number of simultaneously open positions."""
    events = []
    for t in trades_sorted:
        events.append((t["entry_time"], +1))
        events.append((t["exit_time"],  -1))
    events.sort(key=lambda x: x[0])
    peak = cur = 0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default=None,
                        help="Filter to a single year, e.g. 2025")
    parser.add_argument("--equity", type=float, default=START_EQUITY,
                        help="Starting account equity (default 10000)")
    parser.add_argument("--compound", action="store_true",
                        help="Compound risk dollar with equity (default: fixed risk dollar)")
    parser.add_argument("--strategy-path", default=str(DEFAULT_STRATEGY_PATH))
    args = parser.parse_args()

    if args.year:
        start_month = f"{args.year}-01"
        end_month   = f"{args.year}-12"
        # Need a bit of 5m history before year start for FVG formation context
        start_5m = f"{int(args.year)-1}-12"
    else:
        start_month = "2020-01"
        end_month   = "2025-12"
        start_5m    = "2020-01"

    compound     = args.compound
    start_equity = args.equity

    print("=" * 70)
    print(f"  BTC LEVERAGE SWEEP  |  period: {start_month} -> {end_month}")
    print(f"  Equity: ${start_equity:,.0f}  Risk: {RISK_PCT*100:.0f}%/trade  "
          f"Mode: {'COMPOUNDING' if compound else 'FIXED RISK'}")
    print(f"  Strategy: {Path(args.strategy_path).name}")
    print("=" * 70)

    print("\nLoading strategy cells...")
    cell_lookup = load_cell_lookup(Path(args.strategy_path))

    print("\nLoading official klines...")
    df5 = load_klines("5m", start_5m,    end_month)
    df1 = load_klines("1m", start_month, end_month)

    print("\nDetecting FVGs and walking trades (precise exit times)...")
    trades = run_trade_walk(df5, df1, cell_lookup, year_filter=args.year)

    if not trades:
        print("No qualifying trades found.")
        return

    # Quick sanity: win rate cross-check
    n_won = sum(1 for t in trades if t["won"])
    print(f"\n  Overall win rate (1m walk): {n_won/len(trades)*100:.1f}%  "
          f"({n_won}/{len(trades)})")

    # Trade duration stats
    durations = [(t["exit_time"] - t["entry_time"]).total_seconds() / 60
                 for t in trades]
    print(f"  Trade duration (min): "
          f"median={np.median(durations):.0f}  "
          f"p95={np.percentile(durations, 95):.0f}  "
          f"max={max(durations):.0f}")

    peak_conc = _peak_concurrent(sorted(trades, key=lambda t: t["entry_time"]))
    print(f"  Peak concurrent open positions: {peak_conc}")

    # Minimum leverage needed to ever open any trade
    ref_risk = start_equity * RISK_PCT
    max_single_notional = max(ref_risk * 10_000 / t["risk_bps"] for t in trades)
    min_lev_single = max_single_notional / start_equity
    print(f"  Largest single-trade notional at starting equity: ${max_single_notional:,.0f}  "
          f"-> needs >{min_lev_single:.1f}x on ${start_equity:,.0f}")

    # Sweep leverage levels
    leverage_levels = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]

    print(f"\n{'=' * 90}")
    print(f"  LEVERAGE SWEEP")
    print(f"{'=' * 90}")
    print(f"  {'Lev':>4}  {'Taken':>7}  {'Missed':>7}  {'Miss%':>6}  "
          f"{'WR%':>5}  {'LiqEv':>6}  {'PeakCon':>8}  "
          f"{'FinalEq':>10}  {'NetPnL':>10}  {'MaxDD%':>7}  {'Sharpe':>7}")
    print(f"  {'-'*86}")

    results = []
    for lev in leverage_levels:
        r = simulate_leverage(trades, lev, start_equity=start_equity, compound=compound)
        results.append(r)
        liq_flag = " !!!" if r["liq_events"] > 0 else ""
        print(f"  {lev:>4}x  "
              f"{r['trades_taken']:>7,}  "
              f"{r['missed']:>7,}  "
              f"{r['missed_pct']:>5.1f}%  "
              f"{r['win_rate']:>5.1f}  "
              f"{r['liq_events']:>6}{liq_flag:4}  "
              f"{r['peak_concurrent']:>8}  "
              f"${r['final_equity']:>9,.0f}  "
              f"${r['net_pnl']:>9,.0f}  "
              f"{r['max_dd_pct']:>6.1f}%  "
              f"{r['sharpe']:>7.3f}")

    # Analysis
    zero_miss   = [r for r in results if r["missed"] == 0]
    zero_liq    = [r for r in results if r["liq_events"] == 0]
    sweet_spot  = [r for r in results if r["missed"] == 0 and r["liq_events"] == 0]

    print(f"\n  --- Key thresholds ---")
    if zero_miss:
        lev0 = zero_miss[0]
        print(f"  Minimum leverage to take 100% of trades : {lev0['leverage']}x")
    else:
        print(f"  No tested leverage takes 100% of trades")

    if zero_liq:
        print(f"  Liquidation-free up to                  : {zero_liq[-1]['leverage']}x")

    if sweet_spot:
        best = max(sweet_spot, key=lambda r: r["net_pnl"])
        print(f"  Optimal (0 missed, 0 liq, max PnL)      : {best['leverage']}x  "
              f"-> ${best['net_pnl']:,.0f} net  DD={best['max_dd_pct']}%  "
              f"Sharpe={best['sharpe']}")

    # Save results
    suffix = ""
    if args.year:
        suffix += f"_{args.year}"
    if compound:
        suffix += "_compound"
    out_path = OUT_DIR / f"leverage_sweep{suffix}.json"
    with open(out_path, "w") as f:
        json.dump({
            "year": args.year,
            "equity": start_equity,
            "compound": compound,
            "trades_total": len(trades),
            "peak_concurrent": peak_conc,
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
