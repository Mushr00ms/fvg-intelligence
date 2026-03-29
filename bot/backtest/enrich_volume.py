#!/usr/bin/env python3
"""
enrich_volume.py — Enrich backtest trade results with Databento tick volume at entry price.

For each trade: sums all tick volume that traded at exactly trade.entry_price between
formation_time and exit_time, using the correct front-month NQ contract per the
project's roll logic (quarterly expiry, 8-day roll).

This lets you answer: "Across all touches before TP, how many contracts traded at my
exact limit level? Is it consistently >30 — making confirmed_limit unnecessary?"

Usage:
    # Auto-detect zip, enrich 2025 results:
    python3 bot/backtest/enrich_volume.py --results bot/backtest/results/2025.json

    # Explicit zip path, skip re-extraction if already extracted:
    python3 bot/backtest/enrich_volume.py \\
        --results bot/backtest/results/2025.json \\
        --zip /path/to/GLBX-trades.zip \\
        --skip-extract

    # Force re-enrich even if volume_at_entry already present:
    python3 bot/backtest/enrich_volume.py --results ... --force
"""

import argparse
import json
import os
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from logic.utils.contract_utils import generate_nq_expirations, get_contract_for_date

# NQ futures month letter codes (quarterly contracts only)
_MONTH_LETTERS = {3: "H", 6: "M", 9: "U", 12: "Z"}

# Extract to Linux FS for 10x faster I/O vs /mnt/c/
DEFAULT_EXTRACT_DIR = os.path.expanduser("~/databento_trades_nq")
DEFAULT_TICK_DIR = os.path.join(_ROOT, "bot", "data", "ticks")


def expiry_to_symbol(exp_date) -> str:
    """Convert expiration date to Databento front-month symbol.

    e.g. datetime(2025, 3, 21) -> 'NQH5'
         datetime(2025, 6, 20) -> 'NQM5'
    """
    if hasattr(exp_date, "date") and callable(exp_date.date):
        d = exp_date.date()
    else:
        d = exp_date
    letter = _MONTH_LETTERS[d.month]
    year_digit = d.year % 10
    return f"NQ{letter}{year_digit}"


def find_trades_zip(root: str) -> str | None:
    """Auto-detect the first Databento trades zip in the project root."""
    for fname in sorted(os.listdir(root)):
        if not (fname.startswith("GLBX") and fname.endswith(".zip")):
            continue
        path = os.path.join(root, fname)
        try:
            with zipfile.ZipFile(path) as zf:
                names = [i.filename for i in zf.infolist()]
                if any(".trades.dbn.zst" in n for n in names):
                    return path
        except Exception:
            continue
    return None


def find_all_trades_zips(root: str) -> list[str]:
    """Find ALL Databento trades zips in the project root."""
    zips = []
    for fname in sorted(os.listdir(root)):
        if not (fname.startswith("GLBX") and fname.endswith(".zip")):
            continue
        path = os.path.join(root, fname)
        try:
            with zipfile.ZipFile(path) as zf:
                names = [i.filename for i in zf.infolist()]
                if any(".trades.dbn.zst" in n for n in names):
                    zips.append(path)
        except Exception:
            continue
    return zips


def extract_zip(zip_path: str, out_dir: str, verbose: bool = True):
    """Extract all .trades.dbn.zst files from zip to out_dir.

    Skips files that are already present (idempotent re-runs).
    """
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        entries = [i for i in zf.infolist() if ".trades.dbn.zst" in i.filename]
        for info in entries:
            dest = os.path.join(out_dir, os.path.basename(info.filename))
            if os.path.exists(dest):
                continue  # Already extracted
            with zf.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            if verbose:
                print(f"  extracted {os.path.basename(info.filename)}")
    if verbose:
        n = len([f for f in os.listdir(out_dir) if f.endswith(".dbn.zst")])
        print(f"  {n} tick files ready in {out_dir}")


def parse_utc(ts_str: str) -> datetime:
    """Parse a timestamp string to UTC-aware datetime.

    Handles:
      - "2025-01-02 12:25:00-05:00"  (ET-aware — formation_time format)
      - "2025-01-02 20:32:05"         (naive UTC — entry_time / exit_time format)
    """
    ts_str = ts_str.strip().replace(" ", "T", 1)
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_day_ticks(date_str: str, extract_dir: str):
    """Load a single day's .trades.dbn.zst into a DataFrame.

    Returns (prices_array, sizes_array, ts_ns_array, symbol_series) or None if missing.
    """
    import databento as db

    compact = date_str.replace("-", "")
    path = os.path.join(extract_dir, f"glbx-mdp3-{compact}.trades.dbn.zst")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    try:
        store = db.DBNStore.from_file(path)
        df = store.to_df()
    except (ValueError, Exception):
        return None

    if df.empty:
        return None

    return df


def load_day_ticks_parquet(date_str: str, tick_dir: str):
    """Load precomputed tick parquet for a day.

    Converts parquet format (ts_ns/int8 side) back to the DataFrame format
    expected by compute_volume_* functions (ts_event/str side), so all
    existing enrichment code works unchanged.

    Returns DataFrame or None if file missing.
    """
    compact = date_str.replace("-", "")
    path = os.path.join(tick_dir, f"nq_ticks_{compact}.parquet")
    if not os.path.exists(path):
        return None

    df = pd.read_parquet(path)
    if df.empty:
        return None

    # Convert ts_ns (int64) → ts_event (datetime64[ns, UTC])
    df["ts_event"] = pd.to_datetime(df["ts_ns"], unit="ns", utc=True)

    # Convert side (int8: 1/-1/0) → str ('A'/'B'/'')
    side_int = df["side"].to_numpy()
    df["side"] = np.where(side_int == 1, "A", np.where(side_int == -1, "B", ""))

    return df


def compute_volume_at_price(df: pd.DataFrame, symbol: str,
                             entry_price: float,
                             formation_utc: datetime,
                             exit_utc: datetime) -> int | None:
    """Sum all tick sizes at exact entry_price for the given symbol within time window.

    Uses int64 nanosecond timestamps throughout for vectorized numpy speed.

    Returns:
        Total contracts traded at entry_price in [formation_time, exit_time].
        None if the symbol had no ticks on this day (data issue).
    """
    df_sym = df[df["symbol"] == symbol]
    if df_sym.empty:
        return None

    prices = df_sym["price"].to_numpy(dtype=np.float64)
    sizes = df_sym["size"].to_numpy(dtype=np.int64)
    # ts_event may be datetime64[ns] or datetime64[us] — normalize to ns int64
    ts_ns = df_sym["ts_event"].to_numpy("datetime64[ns]").view("int64")

    t_start_ns = pd.Timestamp(formation_utc).value
    t_end_ns = pd.Timestamp(exit_utc).value

    # NQ prices are on 0.25-point grid → exact float64 comparison is safe,
    # but use a tiny epsilon (1e-6 pts) to guard against any IEEE754 edge case
    mask = (
        (np.abs(prices - entry_price) < 1e-6)
        & (ts_ns >= t_start_ns)
        & (ts_ns <= t_end_ns)
    )

    return int(sizes[mask].sum())


def compute_volume_first_touch(df: pd.DataFrame, symbol: str,
                                entry_price: float,
                                entry_utc: datetime) -> int | None:
    """Sum tick volume at exact entry_price during the 1-second bar of first touch.

    The backtester's entry_time is the timestamp of the 1s bar where price first
    reaches the entry level. We look at all ticks at exact entry_price within
    [entry_time, entry_time + 1 second) — this is the liquidity available on
    first contact.

    Returns:
        Contracts traded at entry_price during that 1-second window.
        None if the symbol had no ticks on this day.
    """
    df_sym = df[df["symbol"] == symbol]
    if df_sym.empty:
        return None

    prices = df_sym["price"].to_numpy(dtype=np.float64)
    sizes = df_sym["size"].to_numpy(dtype=np.int64)
    ts_ns = df_sym["ts_event"].to_numpy("datetime64[ns]").view("int64")

    t_start_ns = pd.Timestamp(entry_utc).value
    t_end_ns = t_start_ns + 1_000_000_000  # +1 second in nanoseconds

    mask = (
        (np.abs(prices - entry_price) < 1e-6)
        & (ts_ns >= t_start_ns)
        & (ts_ns < t_end_ns)
    )

    return int(sizes[mask].sum())


def compute_volume_through_price(df: pd.DataFrame, symbol: str,
                                  target_price: float, side: str,
                                  start_utc: datetime,
                                  end_utc: datetime) -> int | None:
    """Sum tick volume at or through a price level within a time window.

    For a BUY position's TP (sell limit): counts ticks >= target_price.
    For a SELL position's TP (buy limit): counts ticks <= target_price.

    This captures all fills that would execute a resting limit order at
    the target, including ticks that gap through the level.

    Returns:
        Total contracts at-or-better than target_price.
        None if the symbol had no ticks on this day.
    """
    df_sym = df[df["symbol"] == symbol]
    if df_sym.empty:
        return None

    prices = df_sym["price"].to_numpy(dtype=np.float64)
    sizes = df_sym["size"].to_numpy(dtype=np.int64)
    ts_ns = df_sym["ts_event"].to_numpy("datetime64[ns]").view("int64")

    t_start_ns = pd.Timestamp(start_utc).value
    t_end_ns = pd.Timestamp(end_utc).value

    time_mask = (ts_ns >= t_start_ns) & (ts_ns <= t_end_ns)

    # BUY position → TP is a sell limit → fills at or above target
    # SELL position → TP is a buy limit → fills at or below target
    if side == "BUY":
        price_mask = prices >= target_price - 1e-6
    else:
        price_mask = prices <= target_price + 1e-6

    mask = time_mask & price_mask
    return int(sizes[mask].sum())


def find_bracket_end(df: pd.DataFrame, symbol: str,
                     stop_price: float, side: str,
                     after_utc: datetime) -> datetime:
    """Find when the SL would trigger or EOD, scanning ticks after a given time.

    For TP exits: tells us how long the bracket would have stayed open,
    giving the full liquidity window for the TP limit order.

    Returns the timestamp of the first SL-triggering tick, or 16:00 ET EOD.
    """
    df_sym = df[df["symbol"] == symbol]
    if df_sym.empty:
        # Default to EOD
        d = after_utc.date() if hasattr(after_utc, 'date') else after_utc
        return datetime(after_utc.year, after_utc.month, after_utc.day,
                        21, 0, 0, tzinfo=timezone.utc)  # 16:00 ET = 21:00 UTC

    prices = df_sym["price"].to_numpy(dtype=np.float64)
    ts_ns = df_sym["ts_event"].to_numpy("datetime64[ns]").view("int64")

    t_start_ns = pd.Timestamp(after_utc).value

    # Scan forward from after_utc
    start_idx = np.searchsorted(ts_ns, t_start_ns, side='left')

    # BUY position → SL triggers when price <= stop
    # SELL position → SL triggers when price >= stop
    for i in range(start_idx, len(prices)):
        if side == "BUY" and prices[i] <= stop_price:
            return pd.Timestamp(ts_ns[i], unit='ns', tz='UTC').to_pydatetime()
        elif side != "BUY" and prices[i] >= stop_price:
            return pd.Timestamp(ts_ns[i], unit='ns', tz='UTC').to_pydatetime()

    # No SL trigger found → EOD
    return datetime(after_utc.year, after_utc.month, after_utc.day,
                    21, 0, 0, tzinfo=timezone.utc)


def compute_stop_slippage(df: pd.DataFrame, symbol: str,
                          stop_price: float, side: str,
                          exit_utc: datetime,
                          qty: int = 1) -> dict | None:
    """Simulate stop-loss market order fill using tick data.

    A stop order becomes a market order when triggered. For large positions,
    this eats through multiple price levels. We walk ticks from the trigger
    point, accumulating volume at each price until `qty` contracts are filled.
    The VWAP of those fills is the actual exit price.

    This is conservative — real fills may be better due to hidden/iceberg
    liquidity not visible in trade prints.

    Args:
        df: day's tick DataFrame (all symbols)
        symbol: NQ contract symbol
        stop_price: the stop order price
        side: "BUY" or "SELL" (the position side, not the stop side)
        exit_utc: backtester's exit timestamp (UTC-aware)
        qty: number of contracts to fill

    Returns:
        dict with actual_stop_price (VWAP), slippage_pts, fills breakdown
        None if no matching ticks found (data gap)
    """
    df_sym = df[df["symbol"] == symbol]
    if df_sym.empty:
        return None

    prices = df_sym["price"].to_numpy(dtype=np.float64)
    sizes = df_sym["size"].to_numpy(dtype=np.int64)
    ts_ns = df_sym["ts_event"].to_numpy("datetime64[ns]").view("int64")

    # Find the trigger point: first tick at-or-through stop near exit time
    t_center_ns = pd.Timestamp(exit_utc).value
    t_start_ns = t_center_ns - 2_000_000_000   # look back 2s to find trigger

    start_idx = int(np.searchsorted(ts_ns, t_start_ns, side='left'))

    is_buy = side == "BUY"
    trigger_idx = -1
    for i in range(start_idx, len(prices)):
        if is_buy and prices[i] <= stop_price:
            trigger_idx = i
            break
        elif not is_buy and prices[i] >= stop_price:
            trigger_idx = i
            break

    if trigger_idx < 0:
        return {"actual_stop_price": stop_price, "slippage_pts": 0.0,
                "slippage_per_contract": 0.0}

    # Walk forward from trigger until all qty contracts are filled.
    # A stop-market order fills at the BID side once triggered:
    # - Ticks at-or-worse than stop: real adverse fills (use actual price)
    # - Ticks better than stop: represent ask-side trades (buyer aggressor),
    #   not available bid liquidity. Cap these at stop price — the best
    #   a stop-market can fill is at the stop level itself.
    filled = 0
    cost = 0.0

    for i in range(trigger_idx, len(prices)):
        tick_price = float(prices[i])
        # Cap at stop price — can't fill better than trigger level
        if is_buy:
            fill_price = min(tick_price, stop_price)   # sell stop: fills at or below
        else:
            fill_price = max(tick_price, stop_price)   # buy stop: fills at or above
        tick_size = int(sizes[i])
        take = min(tick_size, qty - filled)
        cost += fill_price * take
        filled += take
        if filled >= qty:
            break

    if filled == 0:
        return {"actual_stop_price": stop_price, "slippage_pts": 0.0,
                "slippage_per_contract": 0.0}

    vwap = cost / filled

    if is_buy:
        slippage = round(stop_price - vwap, 2)
    else:
        slippage = round(vwap - stop_price, 2)

    # Dollar cost is exact: each fill is grid_price × int_qty × $20
    slippage_dollars = round((stop_price * filled - cost) * 20.0, 2) if is_buy else \
                       round((cost - stop_price * filled) * 20.0, 2)

    return {
        "actual_stop_price": round(vwap, 2),
        "slippage_pts": slippage,
        "slippage_per_contract": round(slippage_dollars / filled, 2) if filled else 0,
        "slippage_dollars": slippage_dollars,
        "filled_of_qty": filled,
    }


def print_summary(trades: list):
    """Print distribution of volume metrics across enriched trades."""
    enriched = [t for t in trades if t.get("volume_at_entry") is not None]
    if not enriched:
        print("No trades enriched.")
        return

    def stats(vlist, label):
        if not vlist:
            return
        a = np.array(vlist)
        pct_30 = (a >= 30).sum() / len(a) * 100
        print(f"  {label:10s}  n={len(a):4d}  "
              f"median={np.median(a):5.0f}  "
              f"mean={np.mean(a):5.0f}  "
              f"p10={np.percentile(a,10):4.0f}  "
              f"p90={np.percentile(a,90):5.0f}  "
              f"≥30: {pct_30:.0f}%")

    def breakdown(a, label):
        print(f"\n  {label} — threshold breakdown ({len(a)} trades):")
        buckets = [(0, 0, "=0 (no ticks)"), (1, 9, "1-9"), (10, 29, "10-29"),
                   (30, 99, "30-99"), (100, None, "≥100")]
        for lo, hi, lbl in buckets:
            if hi is None:
                n = (a >= lo).sum()
            elif lo == 0:
                n = (a == 0).sum()
            else:
                n = ((a >= lo) & (a <= hi)).sum()
            bar = "█" * int(n / max(len(a), 1) * 40)
            print(f"    {lbl:15s}: {n:4d} ({n/max(len(a),1)*100:4.0f}%)  {bar}")

    # ── Total volume (formation → exit) ──
    vols = [t["volume_at_entry"] for t in enriched]
    tp_vols = [t["volume_at_entry"] for t in enriched if t.get("exit_reason") == "TP"]
    sl_vols = [t["volume_at_entry"] for t in enriched if t.get("exit_reason") == "SL"]

    print("\n── Volume at Entry (formation → exit) ─────────────────────────────")
    stats(vols,    "ALL")
    stats(tp_vols, "TP wins")
    stats(sl_vols, "SL loss")
    breakdown(np.array(vols), "Total volume")

    # ── First-touch volume (1-second bar) ──
    ft_enriched = [t for t in trades if t.get("volume_first_touch") is not None]
    if ft_enriched:
        ft_vols = [t["volume_first_touch"] for t in ft_enriched]
        ft_tp = [t["volume_first_touch"] for t in ft_enriched if t.get("exit_reason") == "TP"]
        ft_sl = [t["volume_first_touch"] for t in ft_enriched if t.get("exit_reason") == "SL"]

        print("\n── First Touch Volume (1s bar at entry) ───────────────────────────")
        stats(ft_vols, "ALL")
        stats(ft_tp,   "TP wins")
        stats(ft_sl,   "SL loss")
        breakdown(np.array(ft_vols), "First touch")

    print("───────────────────────────────────────────────────────────────────")

    # ── Volume at TP (entry → SL/EOD) ──
    tp_enriched = [t for t in trades if t.get("volume_at_tp") is not None]
    if tp_enriched:
        tp_vols = [t["volume_at_tp"] for t in tp_enriched]
        tp_tp_vols = [t["volume_at_tp"] for t in tp_enriched if t.get("exit_reason") == "TP"]

        print("\n── Volume at Exact TP (entry → exit) ──────────────────────────────")
        stats(tp_vols,    "ALL")
        stats(tp_tp_vols, "TP wins")
        breakdown(np.array(tp_vols), "Exact TP")

    print("───────────────────────────────────────────────────────────────────")

    # ── Adjusted P&L (fill probability discount) ──
    adj_trades = [t for t in trades if t.get("adjusted_pnl_dollars") is not None]
    if adj_trades:
        raw_pnl = sum(t.get("raw_pnl_dollars", t["pnl_dollars"]) for t in adj_trades)
        adj_pnl = sum(t["adjusted_pnl_dollars"] for t in adj_trades)
        hi_conf = [t for t in adj_trades if t.get("fill_probability", 1) >= 1.0]
        lo_conf = [t for t in adj_trades if t.get("fill_probability", 1) < 1.0]
        lo_raw = sum(t.get("raw_pnl_dollars", t["pnl_dollars"]) for t in lo_conf)
        lo_adj = sum(t["adjusted_pnl_dollars"] for t in lo_conf)
        print(f"\n── Fill Probability Adjusted P&L ──────────────────────────────────")
        print(f"  Raw P&L (touch=fill):     ${raw_pnl:>10,.0f}  ({len(adj_trades)} trades)")
        print(f"  Adjusted P&L:             ${adj_pnl:>10,.0f}  (discount on {len(lo_conf)} low-vol trades)")
        print(f"  Discount amount:          ${raw_pnl - adj_pnl:>10,.0f}  ({abs(raw_pnl - adj_pnl) / abs(raw_pnl) * 100:.1f}% of raw)")
        print(f"  High-confidence ({len(hi_conf)}):     ${sum(t['pnl_dollars'] for t in hi_conf):>10,.0f}")
        print(f"  Low-vol raw ({len(lo_conf)}):          ${lo_raw:>10,.0f}")
        print(f"  Low-vol adjusted ({len(lo_conf)}):     ${lo_adj:>10,.0f}")
        print(f"───────────────────────────────────────────────────────────────────")

    # ── Stop Slippage ──
    sl_enriched = [t for t in trades if t.get("stop_slippage_pts") is not None]
    if sl_enriched:
        slipped = [t for t in sl_enriched if t["stop_slippage_pts"] > 0]
        zero_slip = [t for t in sl_enriched if t["stop_slippage_pts"] == 0]
        total_sl = len(sl_enriched)
        raw_sl_pnl = sum(t["pnl_dollars"] for t in sl_enriched)
        adj_sl_pnl = sum(t.get("slippage_adjusted_pnl", t["pnl_dollars"]) for t in sl_enriched)
        total_slip_cost = sum(
            t["stop_slippage_pts"] * t["contracts"] * 20.0
            for t in slipped
        )

        print(f"\n── Stop Slippage (Databento tick-accurate) ────────────────────────")
        print(f"  SL trades analyzed:       {total_sl}")
        print(f"  Zero slippage:            {len(zero_slip)} ({len(zero_slip)/total_sl*100:.0f}%)")
        print(f"  Slipped:                  {len(slipped)} ({len(slipped)/total_sl*100:.0f}%)")
        if slipped:
            slips = np.array([t["stop_slippage_pts"] for t in slipped])
            print(f"  Slip (pts):  min={slips.min():.2f}  med={np.median(slips):.2f}  "
                  f"avg={slips.mean():.2f}  p90={np.percentile(slips,90):.2f}  max={slips.max():.2f}")

            # By contract count
            from collections import defaultdict as _dd
            by_qty = _dd(list)
            for t in slipped:
                by_qty[t["contracts"]].append(t["stop_slippage_pts"])
            print(f"\n  {'Qty':>4s}  {'Slipped':>7s}  {'Total':>5s}  {'Rate':>5s}  {'AvgSlip':>7s}  {'MaxSlip':>7s}  {'Cost$':>8s}")
            all_by_qty = _dd(int)
            for t in sl_enriched:
                all_by_qty[t["contracts"]] += 1
            for qty in sorted(by_qty.keys()):
                s = np.array(by_qty[qty])
                total_for_qty = all_by_qty[qty]
                cost = sum(v * qty * 20 for v in s)
                print(f"  {qty:>4d}  {len(s):>5d}    {total_for_qty:>5d}  {len(s)/total_for_qty*100:>4.0f}%  "
                      f"{s.mean():>6.2f}  {s.max():>6.2f}  ${cost:>7,.0f}")

        print(f"\n  Backtest SL P&L:          ${raw_sl_pnl:>10,.0f}")
        print(f"  After slippage:           ${adj_sl_pnl:>10,.0f}")
        print(f"  Total slippage cost:      ${total_slip_cost:>10,.0f}")

        # Full P&L impact
        raw_total = sum(t.get("raw_pnl_dollars", t["pnl_dollars"]) for t in trades)
        adj_total = sum(t["pnl_dollars"] for t in trades)
        print(f"\n  Raw P&L (touch=fill):     ${raw_total:>10,.0f}")
        print(f"  Slippage-adjusted P&L:    ${adj_total:>10,.0f}")
        print(f"  Slippage impact:          ${adj_total - raw_total:>10,.0f} ({total_slip_cost/abs(raw_total)*100:.1f}% of raw P&L)")
        print(f"───────────────────────────────────────────────────────────────────")


def main():
    parser = argparse.ArgumentParser(
        description="Enrich backtest results with Databento tick volume at entry price"
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to backtest results JSON (e.g. bot/backtest/results/2025.json)"
    )
    parser.add_argument(
        "--zip", default=None,
        help="Path to Databento trades zip (auto-detected from project root if omitted)"
    )
    parser.add_argument(
        "--extract-dir", default=DEFAULT_EXTRACT_DIR,
        help=f"Directory to extract .dbn.zst files (default: {DEFAULT_EXTRACT_DIR})"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip zip extraction if files are already extracted"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-enrich even if volume_at_entry is already present"
    )
    parser.add_argument(
        "--tick-dir", default=DEFAULT_TICK_DIR,
        help=f"Directory with precomputed tick parquets (default: {DEFAULT_TICK_DIR}). "
             "Falls back to raw .dbn.zst if parquet missing for a day."
    )
    args = parser.parse_args()

    # ── Check for precomputed tick parquets ─────────────────────────────────
    tick_dir = args.tick_dir
    has_tick_parquets = os.path.isdir(tick_dir) and any(
        f.endswith(".parquet") for f in os.listdir(tick_dir)
    ) if os.path.isdir(tick_dir) else False

    if has_tick_parquets:
        n_parquets = len([f for f in os.listdir(tick_dir) if f.endswith(".parquet")])
        print(f"Tick parquets: {n_parquets} files in {tick_dir}")

    # ── Locate zip (needed as fallback if parquets don't cover all days) ─────
    zip_path = args.zip
    has_raw = False
    if zip_path is None:
        zip_path = find_trades_zip(_ROOT)
    if zip_path:
        print(f"Trades zip : {zip_path}")
        if not args.skip_extract:
            print(f"Extracting to {args.extract_dir} ...")
            extract_zip(zip_path, args.extract_dir)
        else:
            n = len([f for f in os.listdir(args.extract_dir) if f.endswith(".dbn.zst")])
            print(f"Skipping extraction — {n} files in {args.extract_dir}")
        has_raw = True
    elif not has_tick_parquets:
        print("ERROR: No tick parquets and no Databento trades zip found. "
              "Use --tick-dir or --zip to specify.")
        sys.exit(1)

    # ── Load results ──────────────────────────────────────────────────────────
    results_path = os.path.join(_ROOT, args.results) if not os.path.isabs(args.results) else args.results
    with open(results_path) as f:
        results = json.load(f)
    trades = results.get("trades", [])

    already_done = sum(1 for t in trades if t.get("volume_first_touch") is not None)
    if already_done and not args.force:
        print(f"\n{already_done}/{len(trades)} trades already enriched.")
        print("Use --force to re-enrich. Running summary on existing data.")
        print_summary(trades)
        return

    print(f"\nEnriching {len(trades)} trades from {results_path}")

    # ── Build NQ expiration calendar ──────────────────────────────────────────
    # Derive year range from trade dates
    all_years = set()
    for t in trades:
        y = int(t["date"][:4])
        all_years.add(y)
    min_year = min(all_years) if all_years else 2020
    max_year = max(all_years) + 1 if all_years else 2027
    exp_dates = generate_nq_expirations(min_year, max_year)

    # ── Group trades by date (load each day's ticks once) ────────────────────
    by_date: dict[str, list] = defaultdict(list)
    for t in trades:
        by_date[t["date"]].append(t)

    total_dates = len(by_date)
    enriched_count = 0
    missing_days = []

    for i, date_str in enumerate(sorted(by_date.keys())):
        day_trades = by_date[date_str]

        # Resolve front-month contract for this date
        trade_dt = datetime.strptime(date_str, "%Y-%m-%d")
        exp = get_contract_for_date(trade_dt, exp_dates, roll_days=8)
        symbol = expiry_to_symbol(exp)

        print(f"[{i+1:3d}/{total_dates}] {date_str}  {symbol}  ({len(day_trades)} trades)", end="", flush=True)

        # Load tick data — prefer precomputed parquets, fall back to raw .dbn.zst
        df = None
        tick_source = ""
        if has_tick_parquets:
            df = load_day_ticks_parquet(date_str, tick_dir)
            if df is not None:
                tick_source = "pq"
        if df is None and has_raw:
            df = load_day_ticks(date_str, args.extract_dir)
            if df is not None:
                tick_source = "dbn"
        if df is None:
            print(f"  ✗ no tick file")
            for t in day_trades:
                t["volume_at_entry"] = None
                t["volume_first_touch"] = None
                t["volume_at_tp"] = None
            missing_days.append(date_str)
            continue

        # Build list of NQ outright symbols available on this day (for roll fallback)
        # Outrights = symbols matching NQ[A-Z]\d, excluding spreads (contain '-')
        day_symbols = [
            s for s in df["symbol"].unique()
            if len(s) == 4 and s.startswith("NQ") and "-" not in s
        ]

        # Per-trade lookup: exact price + time window + front-month symbol
        # On roll days the primary symbol may have 0 volume — fall back to other outrights
        day_results = []
        for trade in day_trades:
            formation_utc = parse_utc(trade["formation_time"])
            entry_utc = parse_utc(trade["entry_time"])
            exit_utc = parse_utc(trade["exit_time"])
            # exit_time is the START of the exit bar. For volume lookups,
            # extend to cover the full bar where the fill actually occurs.
            # For stop slippage: use raw exit_utc (has its own ±2s search).
            if exit_utc.second == 0 and exit_utc.microsecond == 0:
                exit_utc_vol = exit_utc + timedelta(seconds=60)   # minute bar
            else:
                exit_utc_vol = exit_utc + timedelta(seconds=1)    # 1s bar

            vol_total = compute_volume_at_price(
                df, symbol,
                trade["entry_price"],
                formation_utc,
                exit_utc_vol,
            )
            vol_ft = compute_volume_first_touch(
                df, symbol,
                trade["entry_price"],
                entry_utc,
            )

            # Roll fallback: if primary symbol gave 0/None, try other outrights
            used_sym = symbol
            if (vol_total is None or vol_total == 0) and len(day_symbols) > 1:
                for alt_sym in day_symbols:
                    if alt_sym == symbol:
                        continue
                    alt_vol = compute_volume_at_price(
                        df, alt_sym,
                        trade["entry_price"],
                        formation_utc,
                        exit_utc_vol,
                    )
                    if alt_vol is not None and alt_vol > (vol_total or 0):
                        alt_ft = compute_volume_first_touch(
                            df, alt_sym,
                            trade["entry_price"],
                            entry_utc,
                        )
                        vol_total = alt_vol
                        vol_ft = alt_ft
                        used_sym = alt_sym

            trade["volume_at_entry"] = vol_total
            trade["volume_first_touch"] = vol_ft

            # Volume at TP: measure over full bracket lifetime [entry, SL-or-EOD]
            # For TP exits: extend past TP to when SL would trigger or EOD,
            # giving the full liquidity window the limit order had to fill.
            # For SL/EOD exits: window is already [entry, exit].
            tp_price = trade.get("target_price")
            trade_side = trade.get("side", "BUY")
            stop_price = trade.get("stop_price")
            if tp_price is not None:
                if trade.get("exit_reason") == "TP" and stop_price:
                    bracket_end = find_bracket_end(
                        df, used_sym, stop_price, trade_side, exit_utc_vol,
                    )
                else:
                    bracket_end = exit_utc_vol

                vol_tp = compute_volume_at_price(
                    df, used_sym, tp_price, entry_utc, bracket_end,
                )
                trade["volume_at_tp"] = vol_tp

                # First-touch at TP: volume at TP during the exit 1s bar
                if trade.get("exit_reason") == "TP":
                    vol_tp_ft = compute_volume_first_touch(
                        df, used_sym, tp_price, exit_utc_vol,
                    )
                    trade["volume_tp_first_touch"] = vol_tp_ft
                else:
                    trade["volume_tp_first_touch"] = None
            else:
                trade["volume_at_tp"] = None
                trade["volume_tp_first_touch"] = None

            # Fill probability: ≥30 vol = 100%, <30 vol = vol/30 (linear discount)
            FILL_THRESHOLD = 30
            if vol_total is not None and vol_total >= FILL_THRESHOLD:
                fill_prob = 1.0
            elif vol_total is not None and vol_total > 0:
                fill_prob = round(vol_total / FILL_THRESHOLD, 4)
            else:
                fill_prob = 0.0

            trade["fill_probability"] = fill_prob
            trade["adjusted_pnl_dollars"] = round(trade["pnl_dollars"] * fill_prob, 2)

            # Stop slippage: for SL exits, simulate market order fill
            if trade.get("exit_reason") == "SL":
                sl_result = compute_stop_slippage(
                    df, used_sym,
                    trade["stop_price"], trade["side"],
                    exit_utc,
                    qty=trade.get("contracts", 1),
                )
                if sl_result:
                    trade["actual_stop_price"] = sl_result["actual_stop_price"]
                    trade["stop_slippage_pts"] = sl_result["slippage_pts"]
                    trade["stop_slippage_per_ct"] = sl_result["slippage_per_contract"]

            suffix = f"→{used_sym}" if used_sym != symbol else ""
            vtt = trade.get('volume_through_tp', 0)
            tp_tag = f" thru:{vtt}" if tp_price else ""
            day_results.append(f"{vol_total}({vol_ft}ft{tp_tag}){suffix}")
            enriched_count += 1

        print(f"  [{tick_source}] [{', '.join(day_results[:5])}{'...' if len(day_results) > 5 else ''}]")

    # ── Restore raw values before adjusting (idempotent re-enrichment) ───────
    for t in trades:
        if "raw_pnl_dollars" in t:
            t["pnl_dollars"] = t["raw_pnl_dollars"]
            t["pnl_pts"] = t["raw_pnl_pts"]
            t["exit_price"] = t["raw_exit_price"]

    # ── Apply per-trade slippage adjustments ──────────────────────────────────
    for t in trades:
        if t.get("stop_slippage_pts", 0) > 0 and t.get("exit_reason") == "SL":
            slip = t["stop_slippage_pts"]
            slip_cost = slip * t["contracts"] * 20.0
            if "raw_pnl_dollars" not in t:
                t["raw_pnl_dollars"] = t["pnl_dollars"]
                t["raw_pnl_pts"] = t["pnl_pts"]
                t["raw_exit_price"] = t.get("exit_price", t["stop_price"])
            t["pnl_dollars"] = round(t["raw_pnl_dollars"] - slip_cost, 2)
            t["pnl_pts"] = round(t["raw_pnl_pts"] - slip, 2)
            t["exit_price"] = t["actual_stop_price"]
            t["slippage_adjusted_pnl"] = t["pnl_dollars"]
        elif "raw_pnl_dollars" not in t:
            t["raw_pnl_dollars"] = t["pnl_dollars"]
            t["raw_pnl_pts"] = t["pnl_pts"]
            t["raw_exit_price"] = t.get("exit_price", 0)

    # ── Recompute equity curve with adjusted PNL ──────────────────────────────
    start_balance = (results.get("meta") or {}).get("balance", 100000)
    running_bal = start_balance
    for t in trades:
        running_bal += t["pnl_dollars"]
        t["balance"] = round(running_bal, 2)

    # ── Compute enrichment summary stats ────────────────────────────────────
    sl_trades = [t for t in trades if t.get("exit_reason") == "SL"]
    slipped = [t for t in sl_trades if t.get("stop_slippage_pts", 0) > 0]
    total_slippage_cost = sum(
        t["stop_slippage_pts"] * t["contracts"] * 20.0
        for t in slipped
    )
    # pnl_dollars is already adjusted; raw totals come from raw_pnl_dollars
    raw_pnl = sum(t.get("raw_pnl_dollars", t["pnl_dollars"]) for t in trades)
    slippage_adjusted_pnl = round(raw_pnl - total_slippage_cost, 2)

    enrichment_summary = {
        "enriched_trades": enriched_count,
        "missing_days": len(missing_days),
        "sl_trades": len(sl_trades),
        "sl_slipped": len(slipped),
        "sl_slippage_rate": round(len(slipped) / len(sl_trades) * 100, 1) if sl_trades else 0,
        "avg_slippage_pts": round(
            sum(t["stop_slippage_pts"] for t in slipped) / len(slipped), 2
        ) if slipped else 0,
        "max_slippage_pts": max(
            (t["stop_slippage_pts"] for t in slipped), default=0
        ),
        "total_slippage_cost": round(total_slippage_cost, 2),
        "raw_pnl": round(raw_pnl, 2),
        "slippage_adjusted_pnl": slippage_adjusted_pnl,
        "slippage_pnl_impact_pct": round(
            total_slippage_cost / abs(raw_pnl) * 100, 1
        ) if raw_pnl != 0 else 0,
        "tp_exits_with_volume_through": sum(
            1 for t in trades
            if t.get("exit_reason") == "TP" and t.get("volume_through_tp", 0) > 0
        ),
        "tp_exits_total": sum(1 for t in trades if t.get("exit_reason") == "TP"),
    }

    # Update summary — overwrite primary PNL fields, preserve originals as raw_*
    if "summary" in results:
        s = results["summary"]
        # Preserve raw values (only on first enrichment)
        if "raw_net_pnl" not in s:
            s["raw_net_pnl"] = s.get("net_pnl", 0)
            s["raw_pnl_pct"] = s.get("pnl_pct", 0)
            s["raw_final_balance"] = s.get("final_balance", 0)
            s["raw_gross_loss"] = s.get("gross_loss", 0)
            s["raw_profit_factor"] = s.get("profit_factor", 0)

        # Overwrite with slippage-adjusted values
        s["net_pnl"] = slippage_adjusted_pnl
        start_bal = (results.get("meta") or {}).get("balance", 100000)
        s["pnl_pct"] = round(slippage_adjusted_pnl / start_bal * 100, 1)
        s["final_balance"] = round(start_bal + slippage_adjusted_pnl, 2)
        s["gross_loss"] = round(s.get("raw_gross_loss", 0) - total_slippage_cost, 2)
        raw_gross_profit = s.get("gross_profit", 0)
        adj_gross_loss = abs(s["gross_loss"])
        s["profit_factor"] = round(raw_gross_profit / adj_gross_loss, 2) if adj_gross_loss > 0 else 0
        s["avg_loss"] = round(s["gross_loss"] / s.get("losses", 1), 2)

        # Recompute max drawdown from adjusted equity curve
        peak = start_bal
        max_dd = 0
        for t in trades:
            bal = t.get("balance", start_bal)
            if bal > peak:
                peak = bal
            dd = peak - bal
            if dd > max_dd:
                max_dd = dd
        s["max_drawdown"] = round(max_dd, 2)
        s["max_dd_pct"] = round(max_dd / peak * 100, 1) if peak > 0 else 0
        s["total_slippage_cost"] = round(total_slippage_cost, 2)
        s["slippage_pnl_impact_pct"] = enrichment_summary["slippage_pnl_impact_pct"]
        s["sl_slippage_rate"] = enrichment_summary["sl_slippage_rate"]
        s["avg_slippage_pts"] = enrichment_summary["avg_slippage_pts"]

    results["enrichment"] = enrichment_summary

    # ── Save enriched JSON ────────────────────────────────────────────────────
    results["trades"] = trades
    with open(results_path, "w") as f:
        json.dump(results, f, separators=(",", ":"))

    # ── Update manifest ──────────────────────────────────────────────────────
    manifest_path = os.path.join(os.path.dirname(results_path), "manifest.json")
    if os.path.exists(manifest_path):
        run_id = os.path.basename(results_path).replace(".json", "")
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            for run in manifest.get("runs", []):
                if run.get("run_id") == run_id:
                    if "raw_net_pnl" not in run:
                        run["raw_net_pnl"] = run.get("net_pnl", 0)
                        run["raw_pnl_pct"] = run.get("pnl_pct", 0)
                        run["raw_final_balance"] = run.get("final_balance", 0)
                    run["net_pnl"] = slippage_adjusted_pnl
                    run["pnl_pct"] = results["summary"]["pnl_pct"]
                    run["final_balance"] = results["summary"]["final_balance"]
                    run["profit_factor"] = results["summary"]["profit_factor"]
                    run["total_slippage_cost"] = round(total_slippage_cost, 2)
                    run["enriched"] = True
                    break
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=4)
            print(f"Manifest updated → {manifest_path}")
        except Exception as e:
            print(f"Warning: could not update manifest: {e}")

    print(f"\nSaved → {results_path}")
    if missing_days:
        print(f"Missing tick files for {len(missing_days)} days: {missing_days}")

    print_summary(trades)


if __name__ == "__main__":
    main()
