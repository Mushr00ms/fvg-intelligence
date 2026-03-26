"""
trade_viewer.py — Interactive trade visualizer backend.

Serves the trade_viewer.html frontend and provides API endpoints:
  GET /api/trades              — list all trades from the results JSON
  GET /api/trade/{id}/bars     — OHLC bars for a trade's day (filtered window)
  GET /api/meta                — backtest metadata / summary

Usage:
    python3 bot/backtest/trade_viewer.py
    python3 bot/backtest/trade_viewer.py --results path/to/results.json --port 8766
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA_DIR = os.path.join(_ROOT, "bot", "data")
_DEFAULT_RESULTS = os.path.join(
    _ROOT, "bot", "backtest", "results", "v3_moderate_2026ytd_confirmed.json"
)
_HTML_FILE = os.path.join(_ROOT, "bot", "backtest", "trade_viewer.html")

app = FastAPI(title="Trade Viewer", docs_url=None, redoc_url=None)

# ── In-memory state loaded at startup ────────────────────────────────────────

_results: dict = {}
_trades: list = []
_day_cache: dict = {}   # date_str -> DataFrame (1s bars, ET timezone)


def _load_results(path: str):
    global _results, _trades
    with open(path) as f:
        _results = json.load(f)
    _trades = _results.get("trades", [])
    print(f"Loaded {len(_trades)} trades from {path}")


def _load_day(date_str: str) -> pd.DataFrame:
    """Load 1-second bars for a calendar date (YYYY-MM-DD) into ET timezone."""
    if date_str in _day_cache:
        return _day_cache[date_str]

    compact = date_str.replace("-", "")
    path = os.path.join(_DATA_DIR, f"nq_1secs_{compact}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No bar file for {date_str}: {path}")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values("date").reset_index(drop=True)
    _day_cache[date_str] = df
    return df


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample 1s bar DataFrame to target timeframe."""
    if tf == "1s":
        return df
    rule = {"1min": "1min", "5min": "5min"}[tf]
    out = (
        df.set_index("date")
        .resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open"])
        .reset_index()
    )
    return out


def _parse_dt(ts_str: str) -> datetime:
    """Parse a datetime string into an aware ET datetime.

    Handles both:
      - "2026-01-02 10:50:25-05:00"  (tz-aware, already in ET)
      - "2026-01-02 15:10:06"        (naive UTC from backtester — convert to ET)
    """
    import zoneinfo
    et = zoneinfo.ZoneInfo("America/New_York")

    ts_str = ts_str.strip()
    if " " in ts_str and "T" not in ts_str:
        ts_str = ts_str.replace(" ", "T", 1)

    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        # Backtester stores naive UTC timestamps — attach UTC then convert to ET
        dt = dt.replace(tzinfo=timezone.utc).astimezone(et)
    else:
        # Already has tz info (e.g., -05:00 from old format)
        dt = dt.astimezone(et)
    return dt


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    with open(_HTML_FILE) as f:
        content = f.read()
    return HTMLResponse(content=content)


@app.get("/api/meta")
def get_meta():
    return JSONResponse({
        "meta": _results.get("meta", {}),
        "summary": _results.get("summary", {}),
        "total": len(_trades),
    })


@app.get("/api/trades")
def get_trades():
    """Return all trades (lightweight list without bar data)."""
    return JSONResponse({"trades": _trades, "total": len(_trades)})


@app.get("/api/trade/{trade_id}/bars")
def get_trade_bars(
    trade_id: int,
    tf: str = Query("5min", pattern="^(1s|1min|5min)$"),
    window_before: int = Query(30, description="minutes before entry to include"),
    window_after: int = Query(30, description="minutes after exit to include"),
):
    """Return OHLC bars for the given trade's day, windowed around entry/exit."""
    # Find trade
    trade = next((t for t in _trades if t["id"] == trade_id), None)
    if trade is None:
        raise HTTPException(404, f"Trade {trade_id} not found")

    date_str = trade["date"]  # YYYY-MM-DD

    try:
        df_1s = _load_day(date_str)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    # Parse entry/exit times
    entry_dt = _parse_dt(trade["entry_time"])
    exit_dt = _parse_dt(trade["exit_time"])

    # Window: before_entry and after_exit, but clamp to full session
    window_start = entry_dt - timedelta(minutes=window_before)
    window_end = exit_dt + timedelta(minutes=window_after)

    # Filter 1s bars to window
    df_window = df_1s[
        (df_1s["date"] >= window_start) & (df_1s["date"] <= window_end)
    ].copy()

    if df_window.empty:
        raise HTTPException(404, f"No bars found in window for trade {trade_id} on {date_str}")

    # Resample
    df_tf = _resample(df_window, tf)

    # Serialize — return timestamps as ISO strings
    bars = []
    for _, row in df_tf.iterrows():
        bars.append({
            "t": row["date"].isoformat(),
            "o": row["open"],
            "h": row["high"],
            "l": row["low"],
            "c": row["close"],
            "v": row["volume"],
        })

    # Convert trade times to ET ISO for frontend chart alignment
    trade_et = dict(trade)
    trade_et["entry_time_et"] = _parse_dt(trade["entry_time"]).isoformat()
    trade_et["exit_time_et"] = _parse_dt(trade["exit_time"]).isoformat()
    if trade.get("formation_time"):
        trade_et["formation_time"] = _parse_dt(trade["formation_time"]).isoformat()

    return JSONResponse({
        "trade_id": trade_id,
        "date": date_str,
        "tf": tf,
        "bars": bars,
        "trade": trade_et,
    })


@app.get("/api/trade/{trade_id}/fullday")
def get_trade_fullday(
    trade_id: int,
    tf: str = Query("5min", pattern="^(1s|1min|5min)$"),
):
    """Return full RTH session bars for the trade's day (09:30-16:00 ET)."""
    trade = next((t for t in _trades if t["id"] == trade_id), None)
    if trade is None:
        raise HTTPException(404, f"Trade {trade_id} not found")

    date_str = trade["date"]
    try:
        df_1s = _load_day(date_str)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    # RTH filter
    df_rth = df_1s[
        (df_1s["date"].dt.hour * 60 + df_1s["date"].dt.minute >= 9 * 60 + 30) &
        (df_1s["date"].dt.hour * 60 + df_1s["date"].dt.minute < 16 * 60)
    ].copy()

    df_tf = _resample(df_rth, tf)

    bars = [
        {"t": r["date"].isoformat(), "o": r["open"], "h": r["high"],
         "l": r["low"], "c": r["close"], "v": r["volume"]}
        for _, r in df_tf.iterrows()
    ]

    trade_et = dict(trade)
    trade_et["entry_time_et"] = _parse_dt(trade["entry_time"]).isoformat()
    trade_et["exit_time_et"] = _parse_dt(trade["exit_time"]).isoformat()
    if trade.get("formation_time"):
        trade_et["formation_time"] = _parse_dt(trade["formation_time"]).isoformat()

    return JSONResponse({
        "trade_id": trade_id,
        "date": date_str,
        "tf": tf,
        "bars": bars,
        "trade": trade_et,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FVG Trade Viewer")
    parser.add_argument(
        "--results", default=_DEFAULT_RESULTS,
        help="Path to backtest results JSON"
    )
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _load_results(args.results)
    print(f"Trade viewer running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
