#!/usr/bin/env python3
"""
heatmap_api.py — FastAPI server for the FVG Heatmap Dashboard.

Endpoints:
    GET  /                          → index.html
    GET  /api/manifest              → manifest JSON
    GET  /api/dataset/{id}          → data JSON
    GET  /api/configs               → NQ + ES config JSONs
    POST /api/compute               → spawn analysis subprocess, return {job_id}
    GET  /api/jobs/{job_id}/stream  → SSE stream of subprocess stdout

Run:
    python api/heatmap_api.py
"""

import json
import os
import subprocess
import sys
import threading
import uuid

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGIC_DIR = os.path.join(_REPO_ROOT, "logic")
if _LOGIC_DIR not in sys.path:
    sys.path.insert(0, _LOGIC_DIR)

import asyncio

import fastapi
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from utils.heatmap_store import get_manifest, load_dataset
from utils.rr_analysis import get_rr_manifest, load_rr_dataset, find_sample_trades, load_candles_around_trade
from utils.strategy_store import (
    save_strategy, load_strategy, delete_strategy,
    get_strategy_manifest, set_active_strategy, get_active_strategy,
    validate_strategy,
)

app = FastAPI(title="FVG Heatmap API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_DASHBOARD_DIR = os.path.join(_REPO_ROOT, "heatmap_dashboard")
_CONFIGS_DIR   = os.path.join(_LOGIC_DIR, "configs")
_MAIN_PY       = os.path.join(_LOGIC_DIR, "main.py")

# ── In-memory job store ─────────────────────────────────────────────────────
_jobs: dict = {}
_jobs_lock = threading.Lock()


def _reader(job_id: str, proc: subprocess.Popen):
    """Background thread: drain subprocess stdout into job store."""
    job = _jobs[job_id]
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        with _jobs_lock:
            job["lines"].append(line)
    proc.wait()
    with _jobs_lock:
        job["done"] = True
        job["exit_code"] = proc.returncode


# ── Period conversion ───────────────────────────────────────────────────────
_SHORT_TO_LONG = {"y": "years", "m": "months", "w": "weeks", "d": "days", "q": "quarters"}


def _period_long(short: str) -> str:
    """'5y' → '5 years', '15y' → '15 years', etc."""
    for k, v in _SHORT_TO_LONG.items():
        if short.endswith(k) and short[:-1].isdigit():
            return f"{short[:-1]} {v}"
    return short  # already long or unrecognised


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    """Main page — RR Explorer (strategy builder, backtest, etc.)."""
    rr = os.path.join(_DASHBOARD_DIR, "rr_dashboard.html")
    if not os.path.exists(rr):
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(rr, media_type="text/html")


@app.get("/heatmap")
def serve_heatmap_dashboard():
    """Legacy heatmap workspace dashboard."""
    index = os.path.join(_DASHBOARD_DIR, "index.html")
    if not os.path.exists(index):
        raise HTTPException(status_code=404, detail="Heatmap dashboard not found")
    return FileResponse(index, media_type="text/html")


@app.get("/api/manifest")
def api_manifest():
    return JSONResponse(get_manifest())


@app.get("/api/dataset/{dataset_id}")
def api_dataset(dataset_id: str):
    try:
        df = load_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    rows = df.astype(object).where(df.notna(), None).values.tolist()
    return JSONResponse({"columns": list(df.columns), "rows": rows})


@app.get("/api/overview-stats")
def api_overview_stats():
    """Aggregate midpoint-crossing stats across ALL datasets, grouped by timeframe."""
    manifest = get_manifest()
    datasets = manifest.get("datasets", [])

    # Per-timeframe accumulators
    acc = {}  # tf → {total, crossed, mid_wsum, mid_wn}
    for entry in datasets:
        tf = entry.get("timeframe_label", "?")
        ds_id = entry["id"]
        method = entry.get("size_filtering_method", "bins")

        try:
            df = load_dataset(ds_id)
        except Exception:
            continue

        if "total_fvgs" not in df.columns:
            continue

        # For cumulative, only use the smallest size threshold to avoid double-counting
        if method == "cumulative" and "size_range" in df.columns:
            try:
                smallest = df["size_range"].apply(lambda v: float(str(v).split("-")[0])).min()
                df = df[df["size_range"].apply(lambda v: float(str(v).split("-")[0])) == smallest]
            except Exception:
                pass

        if tf not in acc:
            acc[tf] = {"total": 0, "crossed": 0, "mid_wsum": 0.0, "mid_wn": 0}

        a = acc[tf]
        for _, row in df.iterrows():
            n = row.get("total_fvgs")
            if n is None or (hasattr(n, '__class__') and n != n):  # NaN check
                continue
            n = int(n)
            a["total"] += n

            pct = row.get("midpoint_crossed_pct")
            if pct is not None and pct == pct:  # not NaN
                a["crossed"] += round(float(pct) / 100 * n)

            avg = row.get("avg_midpoint_crossing_count")
            if avg is not None and avg == avg and n > 0:
                a["mid_wsum"] += float(avg) * n
                a["mid_wn"] += n

    result = {}
    for tf, a in acc.items():
        result[tf] = {
            "total":  a["total"],
            "midPct": round(a["crossed"] / a["total"] * 100, 1) if a["total"] > 0 and a["crossed"] > 0 else None,
            "avgMid": round(a["mid_wsum"] / a["mid_wn"], 1) if a["mid_wn"] > 0 else None,
        }

    return JSONResponse(result)


@app.get("/api/configs")
def api_configs():
    configs = {}
    for name in ("NQ_config.json", "ES_config.json"):
        path = os.path.join(_CONFIGS_DIR, name)
        if os.path.exists(path):
            with open(path) as f:
                configs[name.split("_")[0]] = json.load(f)
    return JSONResponse(configs)


# ── Compute ─────────────────────────────────────────────────────────────────

class ComputeRequest(BaseModel):
    ticker: str          # "NQ" / "ES"
    timeframe: str       # "5min" / "15min"
    period: str          # short form: "5y", "15y", ...
    session_minutes: int # 30 / 60
    method: str          # "bins" / "cumulative"
    size_start: float
    size_end: float
    size_step: float
    min_exp: float


@app.post("/api/compute")
def api_compute(req: ComputeRequest):
    # Basic validation
    if req.ticker not in ("NQ", "ES"):
        raise HTTPException(status_code=400, detail="ticker must be NQ or ES")
    if req.timeframe not in ("3min", "5min", "15min"):
        raise HTTPException(status_code=400, detail="timeframe must be 3min, 5min or 15min")
    if req.method not in ("bins", "cumulative"):
        raise HTTPException(status_code=400, detail="method must be bins or cumulative")
    if req.size_step <= 0 or req.size_start >= req.size_end:
        raise HTTPException(status_code=400, detail="invalid size range")

    env = {
        **os.environ,
        "FVG_TICKER":          req.ticker,
        "FVG_PERIOD":          _period_long(req.period),
        "FVG_TIMEFRAME":       req.timeframe,
        "FVG_SESSION_MINUTES": str(req.session_minutes),
        "FVG_METHOD":          req.method,
        "FVG_SIZE_START":      str(req.size_start),
        "FVG_SIZE_END":        str(req.size_end),
        "FVG_SIZE_STEP":       str(req.size_step),
        "FVG_MIN_EXP":         str(req.min_exp),
    }

    proc = subprocess.Popen(
        [sys.executable, _MAIN_PY],
        cwd=_LOGIC_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"lines": [], "done": False, "exit_code": None, "proc": proc}

    threading.Thread(target=_reader, args=(job_id, proc), daemon=True).start()

    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/api/jobs/{job_id}/stream")
async def api_job_stream(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_gen():
        idx = 0
        while True:
            with _jobs_lock:
                job   = _jobs.get(job_id, {})
                lines = job.get("lines", [])[idx:]
                done  = job.get("done", False)
                code  = job.get("exit_code")

            for line in lines:
                # SSE format: "data: ...\n\n"
                yield f"data: {line}\n\n"
            idx += len(lines)

            if done:
                if code == 0:
                    yield "event: done\ndata: success\n\n"
                else:
                    yield f"event: error\ndata: process exited with code {code}\n\n"
                return

            await asyncio.sleep(0.15)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/jobs/{job_id}")
def api_job_cancel(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    proc = job.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
    return JSONResponse({"cancelled": True})


_RR_DASHBOARD = os.path.join(_DASHBOARD_DIR, "rr_dashboard.html")


@app.get("/rr")
def serve_rr_dashboard():
    if not os.path.exists(_RR_DASHBOARD):
        raise HTTPException(status_code=404, detail="RR Dashboard not found")
    return FileResponse(_RR_DASHBOARD, media_type="text/html")


@app.get("/api/rr/manifest")
def api_rr_manifest():
    return JSONResponse(get_rr_manifest())


_DATABENTO_CACHE = os.path.join(_LOGIC_DIR, "databento_cache")
_FVG_CACHE       = os.path.join(_LOGIC_DIR, "fvg_cache")


@app.get("/api/rr/trade-sample")
def api_rr_trade_sample(
    time_period: str,
    risk_range: str,
    setup: str = "mit_extreme",
    n: float = 2.0,
    sample_idx: int = 0,
    tf: str = "5min",
):
    """Return candle data + trade levels for a specific FVG from a heatmap cell."""
    import glob

    # Find the parquet with RR columns
    parquets = sorted(glob.glob(os.path.join(_FVG_CACHE, "fvg_results_5min_*.parquet")),
                      key=os.path.getmtime, reverse=True)

    rr_parquet = None
    for p in parquets:
        import pandas as pd
        import pyarrow.parquet as pq
        schema = pq.read_schema(p)
        if "rr_mit_extreme_risk" in schema.names:
            rr_parquet = p
            break

    if not rr_parquet:
        raise HTTPException(status_code=404, detail="No FVG parquet with RR data found")

    trades = find_sample_trades(rr_parquet, time_period, risk_range, setup, n)
    if not trades:
        raise HTTPException(status_code=404, detail="No matching trades found for this cell")

    idx = sample_idx % len(trades)
    trade = trades[idx]

    bars_before = 50 if tf == "1min" else 10
    bars_after = 200 if tf == "1min" else 40
    candles = load_candles_around_trade(_DATABENTO_CACHE, trade, timeframe=tf,
                                       bars_before=bars_before, bars_after=bars_after)

    return JSONResponse({
        "trade": trade,
        "candles": candles,
        "total_samples": len(trades),
        "sample_idx": idx,
    })


@app.get("/api/rr/{dataset_id}")
def api_rr_dataset(dataset_id: str):
    try:
        data = load_rr_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"RR dataset '{dataset_id}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(data)


# ── Strategy endpoints ─────────────────────────────────────────────────────

@app.get("/api/strategies")
def api_strategies_list():
    """List all strategies (manifest)."""
    return JSONResponse(get_strategy_manifest())


@app.get("/api/strategies/active")
def api_strategies_get_active():
    """Get the currently active strategy."""
    strategy = get_active_strategy()
    if strategy is None:
        return JSONResponse({"active": None})
    return JSONResponse(strategy)


@app.get("/api/strategies/{strategy_id}")
def api_strategies_get(strategy_id: str):
    """Load a specific strategy."""
    try:
        data = load_strategy(strategy_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")
    return JSONResponse(data)


@app.post("/api/strategies")
async def api_strategies_save(request: Request):
    """Create or update a strategy."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    errors = validate_strategy(body)
    if errors:
        return JSONResponse({"saved": False, "errors": errors}, status_code=422)

    try:
        strategy_id = save_strategy(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"saved": True, "id": strategy_id})


@app.delete("/api/strategies/{strategy_id}")
def api_strategies_delete(strategy_id: str):
    """Delete a strategy."""
    try:
        delete_strategy(strategy_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"deleted": True, "id": strategy_id})


@app.put("/api/strategies/active/{strategy_id}")
def api_strategies_set_active(strategy_id: str):
    """Set the active strategy for the bot."""
    try:
        set_active_strategy(strategy_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found")
    return JSONResponse({"active": True, "id": strategy_id})


# ── Backtest endpoints ─────────────────────────────────────────────────────

_BACKTESTER_PY = os.path.join(_REPO_ROOT, "bot", "backtest", "backtester.py")
_BACKTEST_RESULTS_DIR = os.path.join(_REPO_ROOT, "bot", "backtest", "results")


class BacktestRequest(BaseModel):
    strategy_id: str
    start_date: str = ""      # YYYYMMDD, empty = all available
    end_date: str = ""
    balance: float = 76000
    risk_pct: float = 0.01
    data_source: str = "ib"   # "ib" or "databento"
    slip: bool = True         # realistic slippage (entry 1 tick deeper, TP 1 tick early)
    risk_tiers: bool = True   # use 3-tier risk from strategy meta


@app.post("/api/backtest")
def api_backtest_run(req: BacktestRequest):
    """Run a backtest as a subprocess. Returns job_id for streaming progress."""
    os.makedirs(_BACKTEST_RESULTS_DIR, exist_ok=True)
    job_id = str(uuid.uuid4())
    json_out = os.path.join(_BACKTEST_RESULTS_DIR, f"{job_id}.json")

    cmd = [
        sys.executable, _BACKTESTER_PY,
        "--strategy", req.strategy_id,
        "--balance", str(req.balance),
        "--risk-pct", str(req.risk_pct),
        "--data-source", req.data_source,
        "--json-output", json_out,
    ]
    if req.start_date:
        cmd += ["--start", req.start_date]
    if req.end_date:
        cmd += ["--end", req.end_date]
    if req.slip:
        cmd += ["--slip"]
    if req.risk_tiers:
        cmd += ["--risk-tiers"]

    proc = subprocess.Popen(
        cmd, cwd=_REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    with _jobs_lock:
        _jobs[job_id] = {
            "lines": [], "done": False, "exit_code": None, "proc": proc,
            "type": "backtest", "json_out": json_out,
        }

    threading.Thread(target=_reader, args=(job_id, proc), daemon=True).start()
    return JSONResponse({"job_id": job_id}, status_code=202)


@app.get("/api/backtest/{job_id}/stream")
async def api_backtest_stream(job_id: str):
    """SSE stream of backtest stdout."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_gen():
        cursor = 0
        while True:
            with _jobs_lock:
                new_lines = job["lines"][cursor:]
                cursor = len(job["lines"])
                done = job["done"]
            for line in new_lines:
                yield f"data: {json.dumps({'line': line})}\n\n"
            if done:
                yield f"data: {json.dumps({'done': True, 'exit_code': job['exit_code']})}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/api/backtest/{job_id}/results")
def api_backtest_results(job_id: str):
    """Get backtest results JSON after completion."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    # Check if job completed and JSON output exists
    json_path = None
    if job and job.get("json_out"):
        json_path = job["json_out"]
    else:
        # Try direct file lookup
        json_path = os.path.join(_BACKTEST_RESULTS_DIR, f"{job_id}.json")

    if not json_path or not os.path.exists(json_path):
        if job and not job.get("done"):
            raise HTTPException(status_code=202, detail="Backtest still running")
        raise HTTPException(status_code=404, detail="Results not found")

    with open(json_path) as f:
        return JSONResponse(json.load(f))


@app.get("/api/backtest/history")
def api_backtest_history():
    """List all completed backtest runs."""
    os.makedirs(_BACKTEST_RESULTS_DIR, exist_ok=True)
    manifest_path = os.path.join(_BACKTEST_RESULTS_DIR, "manifest.json")

    # Load existing manifest or start fresh
    manifest = {"runs": [], "last_updated": None}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            pass

    # Check for result files missing from manifest and append them
    known_ids = {r["run_id"] for r in manifest.get("runs", [])}
    added = False
    for f in sorted(os.listdir(_BACKTEST_RESULTS_DIR)):
        if f.endswith(".json") and f != "manifest.json":
            run_id = f.replace(".json", "")
            if run_id in known_ids:
                continue
            try:
                with open(os.path.join(_BACKTEST_RESULTS_DIR, f)) as fh:
                    data = json.load(fh)
                meta = data.get("meta", {})
                summary = data.get("summary", {})
                manifest["runs"].append({
                    "run_id": run_id,
                    "strategy_id": meta.get("strategy_id", ""),
                    "strategy_name": meta.get("strategy_name", ""),
                    "start_date": meta.get("start_date", ""),
                    "end_date": meta.get("end_date", ""),
                    "balance": meta.get("balance", 0),
                    "total_trades": summary.get("total_trades", 0),
                    "net_pnl": summary.get("net_pnl", 0),
                    "pnl_pct": summary.get("pnl_pct", 0),
                    "win_rate": summary.get("win_rate", 0),
                    "profit_factor": summary.get("profit_factor"),
                    "max_dd_pct": summary.get("max_dd_pct", 0),
                    "saved_at": data.get("saved_at", ""),
                })
                added = True
            except Exception:
                continue

    # Persist updated manifest if new runs were found
    if added:
        from datetime import datetime, timezone
        manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
        tmp = manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp, manifest_path)

    return JSONResponse(manifest)


# ── Trade Bar endpoints (for inline trade viewer) ──────────────────────────

_BAR_DATA_DIR = os.path.join(_REPO_ROOT, "bot", "data")
_bar_day_cache: dict = {}   # date_str -> DataFrame


def _load_day_bars(date_str: str):
    """Load 1-second bars for a calendar date (YYYY-MM-DD) into ET timezone."""
    if date_str in _bar_day_cache:
        return _bar_day_cache[date_str]
    compact = date_str.replace("-", "")
    path = os.path.join(_BAR_DATA_DIR, f"nq_1secs_{compact}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No bar file for {date_str}")
    import pandas as _pd
    df = _pd.read_parquet(path)
    df["date"] = _pd.to_datetime(df["date"], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values("date").reset_index(drop=True)
    _bar_day_cache[date_str] = df
    return df


def _resample_bars(df, tf: str):
    """Resample 1s bars to target timeframe."""
    if tf == "1s":
        return df
    import pandas as _pd
    return (
        df.set_index("date")
        .resample(tf, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open"])
        .reset_index()
    )


def _parse_trade_dt(ts_str: str):
    """Parse trade timestamp to ET-aware datetime."""
    from datetime import datetime as _dt, timezone as _tz
    import zoneinfo
    et = zoneinfo.ZoneInfo("America/New_York")
    ts_str = ts_str.strip().replace(" ", "T", 1) if " " in ts_str and "T" not in ts_str else ts_str
    dt = _dt.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_tz.utc).astimezone(et)
    else:
        dt = dt.astimezone(et)
    return dt


def _get_trade_from_run(run_id: str, trade_id: int):
    """Look up a trade from a backtest results file."""
    path = os.path.join(_BACKTEST_RESULTS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    with open(path) as f:
        data = json.load(f)
    trade = next((t for t in data.get("trades", []) if t["id"] == trade_id), None)
    if trade is None:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found in run {run_id}")
    return trade


def _bars_to_json(df, trade):
    """Convert DataFrame bars + trade times to JSON response."""
    bars = [
        {"t": r["date"].isoformat(), "o": r["open"], "h": r["high"],
         "l": r["low"], "c": r["close"], "v": r["volume"]}
        for _, r in df.iterrows()
    ]
    trade_et = dict(trade)
    trade_et["entry_time_et"] = _parse_trade_dt(trade["entry_time"]).isoformat()
    trade_et["exit_time_et"] = _parse_trade_dt(trade["exit_time"]).isoformat()
    if trade.get("formation_time"):
        trade_et["formation_time_et"] = _parse_trade_dt(trade["formation_time"]).isoformat()
    return {"trade_id": trade["id"], "date": trade["date"], "bars": bars, "trade": trade_et}


@app.get("/api/backtest/{run_id}/trade/{trade_id}/bars")
def api_trade_bars(
    run_id: str, trade_id: int,
    tf: str = Query("5min", pattern="^(1s|1min|5min)$"),
    window: int = Query(30, description="minutes before/after entry/exit"),
):
    """Return OHLC bars for a trade, windowed around entry/exit."""
    trade = _get_trade_from_run(run_id, trade_id)
    try:
        df_1s = _load_day_bars(trade["date"])
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    entry_dt = _parse_trade_dt(trade["entry_time"])
    exit_dt = _parse_trade_dt(trade["exit_time"])
    from datetime import timedelta
    df_win = df_1s[
        (df_1s["date"] >= entry_dt - timedelta(minutes=window)) &
        (df_1s["date"] <= exit_dt + timedelta(minutes=window))
    ].copy()
    if df_win.empty:
        raise HTTPException(status_code=404, detail="No bars in window")

    df_tf = _resample_bars(df_win, tf)
    resp = _bars_to_json(df_tf, trade)
    resp["tf"] = tf
    return JSONResponse(resp)


@app.get("/api/backtest/{run_id}/trade/{trade_id}/fullday")
def api_trade_fullday(
    run_id: str, trade_id: int,
    tf: str = Query("5min", pattern="^(1s|1min|5min)$"),
):
    """Return full RTH session bars (09:30-16:00 ET) for a trade's day."""
    trade = _get_trade_from_run(run_id, trade_id)
    try:
        df_1s = _load_day_bars(trade["date"])
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    df_rth = df_1s[
        (df_1s["date"].dt.hour * 60 + df_1s["date"].dt.minute >= 570) &
        (df_1s["date"].dt.hour * 60 + df_1s["date"].dt.minute < 960)
    ].copy()

    df_tf = _resample_bars(df_rth, tf)
    resp = _bars_to_json(df_tf, trade)
    resp["tf"] = tf
    return JSONResponse(resp)


# ── Live Bot endpoints ─────────────────────────────────────────────────────

_BOT_STATE_DIR = os.path.join(_REPO_ROOT, "bot", "bot_state")
_BOT_LOG_DIR = os.path.join(_REPO_ROOT, "bot", "logs")
_BOT_DB_PATH = os.path.join(_REPO_ROOT, "bot", "bot_state", "bot_trades.db")
_BOT_DEPOSITS_PATH = os.path.join(_REPO_ROOT, "bot", "bot_state", "deposits.json")


def _load_deposits():
    """Load deposit/withdrawal ledger. Each entry: {ts, amount_usd, note}.
    ts is ISO-8601. amount_usd positive for deposits, negative for withdrawals."""
    try:
        with open(_BOT_DEPOSITS_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _today_str():
    """Current date in ET as YYYY-MM-DD."""
    import pytz
    from datetime import datetime
    return datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")


@app.get("/live")
async def live_page():
    """Live bot monitoring dashboard."""
    return FileResponse(os.path.join(_DASHBOARD_DIR, "live_dashboard.html"))


@app.get("/api/bot/state")
def api_bot_state():
    """Return the current bot DailyState JSON."""
    # Find the most recent state file
    os.makedirs(_BOT_STATE_DIR, exist_ok=True)
    state_file = os.path.join(_BOT_STATE_DIR, "daily_state.json")
    if not os.path.exists(state_file):
        # Try date-stamped file
        state_file = os.path.join(_BOT_STATE_DIR, f"{_today_str()}.json")
    if not os.path.exists(state_file):
        # Try any .json in the dir
        for f in sorted(os.listdir(_BOT_STATE_DIR), reverse=True):
            if f.endswith(".json") and f != "deposits.json":
                state_file = os.path.join(_BOT_STATE_DIR, f)
                break
    if not os.path.exists(state_file):
        return JSONResponse({"error": "No bot state found", "running": False})
    try:
        with open(state_file) as f:
            state = json.load(f)
        state["running"] = True
        # Check if stale (last_updated > 60s ago → probably not running)
        from datetime import datetime
        try:
            last = datetime.fromisoformat(state.get("last_updated", ""))
            import pytz
            now = datetime.now(pytz.timezone("America/New_York"))
            if hasattr(last, 'tzinfo') and last.tzinfo:
                age = (now - last).total_seconds()
            else:
                age = 999
            state["running"] = age < 120  # Consider stale after 2 minutes
            state["state_age_seconds"] = round(age, 1)
        except Exception:
            pass
        # Enrich closed trades with DB commission data (state file has gross PnL only)
        try:
            if os.path.exists(_BOT_DB_PATH) and state.get("closed_trades"):
                import sqlite3
                conn = sqlite3.connect(_BOT_DB_PATH)
                conn.row_factory = sqlite3.Row
                db_trades = {
                    r["group_id"]: dict(r) for r in conn.execute(
                        "SELECT group_id, commission, net_pnl FROM trades "
                        "WHERE trade_date = ? AND exit_reason IS NOT NULL",
                        [state.get("date", "")],
                    ).fetchall()
                }
                conn.close()
                total_comm = 0.0
                for t in state["closed_trades"]:
                    db_row = db_trades.get(t.get("group_id"))
                    if db_row and db_row["commission"]:
                        t["realized_pnl"] = db_row["net_pnl"]
                        t["commission"] = db_row["commission"]
                        total_comm += db_row["commission"]
                if total_comm > 0:
                    state["realized_pnl"] = round(
                        sum(t.get("realized_pnl", 0) for t in state["closed_trades"]), 2
                    )
        except Exception:
            pass  # Non-critical — fall back to state file values
        return JSONResponse(state)
    except Exception as e:
        return JSONResponse({"error": str(e), "running": False})


@app.get("/api/bot/events")
async def api_bot_events():
    """SSE stream of live bot events. Tails today's JSONL log file."""
    log_file = os.path.join(_BOT_LOG_DIR, f"{_today_str()}.jsonl")

    async def event_stream():
        """Tail the JSONL log file, yielding new lines as SSE events."""
        # Start from the end of existing file (don't replay history on connect)
        file_pos = 0
        if os.path.exists(log_file):
            file_pos = os.path.getsize(log_file)

        # Send initial connection event
        yield f"data: {json.dumps({'event': 'connected', 'log_file': log_file, 'position': file_pos})}\n\n"

        while True:
            try:
                if os.path.exists(log_file):
                    size = os.path.getsize(log_file)
                    if size > file_pos:
                        with open(log_file, 'r') as f:
                            f.seek(file_pos)
                            new_data = f.read()
                            file_pos = f.tell()
                        for line in new_data.strip().split('\n'):
                            if line.strip():
                                try:
                                    evt = json.loads(line)
                                    yield f"data: {json.dumps(evt)}\n\n"
                                except json.JSONDecodeError:
                                    yield f"data: {json.dumps({'event': 'raw', 'line': line})}\n\n"
                    elif size < file_pos:
                        # File was truncated/rotated
                        file_pos = 0
                else:
                    file_pos = 0
            except Exception as e:
                yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/bot/events/history")
def api_bot_events_history():
    """Return all events from today's log (for initial page load)."""
    log_file = os.path.join(_BOT_LOG_DIR, f"{_today_str()}.jsonl")
    events = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return JSONResponse({"events": events, "count": len(events)})


@app.get("/api/bot/stats")
def api_bot_stats():
    """Return aggregate stats from the bot's SQLite database."""
    if not os.path.exists(_BOT_DB_PATH):
        return JSONResponse({"error": "No bot database found"})
    try:
        sys.path.insert(0, os.path.join(_REPO_ROOT, "bot"))
        from db import TradeDB
        db = TradeDB(_BOT_DB_PATH)
        return JSONResponse({
            "daily": db.get_daily_summary(days=30),
            "cells": db.get_cell_performance(),
            "equity": db.get_equity_curve(limit=500),
            "hourly": db.get_hourly_performance(),
            "funnel": db.get_funnel(_today_str()),
            "deposits": _load_deposits(),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ── Bot Analytics Endpoints ───────────────────────────────────────────────

def _get_bot_db():
    """Get a TradeDB instance for bot analytics queries."""
    if not os.path.exists(_BOT_DB_PATH):
        return None
    sys.path.insert(0, os.path.join(_REPO_ROOT, "bot"))
    from db import TradeDB
    return TradeDB(_BOT_DB_PATH)


@app.get("/api/bot/equity-curve")
def api_bot_equity_curve():
    """Equity curve: trade-by-trade balance progression."""
    db = _get_bot_db()
    if not db:
        return JSONResponse({"error": "No bot database found"})
    try:
        return JSONResponse({"equity": db.get_equity_curve(limit=200), "deposits": _load_deposits()})
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/api/bot/cell-performance")
def api_bot_cell_performance():
    """Per-cell P&L: setup × time_period × risk_range breakdown."""
    db = _get_bot_db()
    if not db:
        return JSONResponse({"error": "No bot database found"})
    try:
        cells = db.get_cell_performance(days=30)
        # Enrich with strategy baseline win rate
        try:
            strat_path = os.path.join(
                _REPO_ROOT, "bot", "strategies", "manifest.json")
            with open(strat_path) as f:
                manifest = json.load(f)
                active_id = manifest.get("active_strategy") or manifest.get("active")
            if active_id:
                with open(os.path.join(
                        _REPO_ROOT, "bot", "strategies", f"{active_id}.json")) as f:
                    strat = json.load(f)
                baseline = {
                    (c["time_period"], c["risk_range"], c["setup"]): c["win_rate"]
                    for c in strat.get("cells", [])
                }
                for c in cells:
                    key = (c.get("time_period"), c.get("risk_range"), c.get("setup"))
                    c["baseline_wr"] = baseline.get(key)
                # Drop cells from old strategies that don't map to current strategy
                cells = [c for c in cells if c.get("baseline_wr") is not None]
        except Exception:
            pass
        return JSONResponse({"cells": cells})
    except Exception as e:
        return JSONResponse({"error": str(e)})


def _compute_backtest_percentiles(active_id: str) -> dict:
    """Compute weekly/monthly return percentiles from WF backtest daily P&L.

    Reads all backtest runs whose strategy_id shares the same WF family as
    the active strategy (matched by the base name pattern). Collects daily
    P&L, groups into weeks/months as % of balance, returns percentile dict.

    Returns empty dict if insufficient data.
    """
    import glob
    from collections import defaultdict
    from datetime import datetime

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "bot", "backtest", "results")
    manifest_path = os.path.join(results_dir, "manifest.json")

    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path) as f:
        manifest = json.load(f)

    all_runs = manifest.get("runs", [])
    if not all_runs:
        return {}

    # Filter runs whose strategy_id matches active_id; fall back to all runs
    # if none match (backward compat for manifests without strategy_id).
    matched = [r["run_id"] for r in all_runs if r.get("strategy_id") == active_id]
    run_ids = matched if matched else [r["run_id"] for r in all_runs]

    all_daily = []
    for run_id in run_ids:
        result_path = os.path.join(results_dir, f"{run_id}.json")
        if not os.path.exists(result_path):
            continue
        with open(result_path) as f:
            data = json.load(f)
        start_bal = data.get("meta", {}).get("balance", 80000)
        # Normalize daily dollar P&L as % of the run's starting balance.
        # This gives "what fraction of my capital did I make/lose today"
        # and sums correctly to the known annual return (72100/80000 = 90.1%).
        for dp in data.get("daily_pnl", []):
            all_daily.append({
                "date": dp["date"],
                "pnl_pct": dp["pnl"] / start_bal * 100,
            })

    if len(all_daily) < 50:
        return {}

    all_daily.sort(key=lambda x: x["date"])

    # Group into ISO weeks and calendar months
    weekly_pct = defaultdict(float)
    monthly_pct = defaultdict(float)
    for dp in all_daily:
        dt = datetime.strptime(dp["date"], "%Y-%m-%d")
        weekly_pct[dt.strftime("%Y-W%W")] += dp["pnl_pct"]
        monthly_pct[dp["date"][:7]] += dp["pnl_pct"]

    weeks = sorted(weekly_pct.values())
    months = sorted(monthly_pct.values())

    def _pctl(data, p):
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = min(f + 1, len(data) - 1)
        return data[f] + (k - f) * (data[c] - data[f])

    result = {}
    if len(weeks) >= 20:
        result["week"] = {
            f"p{p}": round(_pctl(weeks, p), 4)
            for p in (5, 10, 25, 50, 75, 90, 95)
        }
    if len(months) >= 6:
        result["month"] = {
            f"p{p}": round(_pctl(months, p), 4)
            for p in (5, 10, 25, 50, 75, 90, 95)
        }
    return result


# Cache: recompute at most once per hour
_pctl_cache = {"data": {}, "time": 0, "strategy": ""}


def _get_backtest_percentiles(active_id: str) -> dict:
    """Cached wrapper for _compute_backtest_percentiles."""
    import time as _time
    now = _time.time()
    if (_pctl_cache["strategy"] == active_id
            and _pctl_cache["data"]
            and now - _pctl_cache["time"] < 3600):
        return _pctl_cache["data"]
    result = _compute_backtest_percentiles(active_id)
    _pctl_cache["data"] = result
    _pctl_cache["time"] = now
    _pctl_cache["strategy"] = active_id
    return result


def _strategy_daily_ev_pct(strat: dict) -> float:
    """Compute daily expected return as a fraction of balance from strategy cells.

    For each cell: expected_pnl_per_trade = ev × risk_pct × balance
    (because contracts = risk_pct × balance / (risk_pts × point_value),
     and pnl = ev × risk_pts × contracts × point_value = ev × risk_pct × balance)

    daily_ev_pct = Σ(trades_per_day × ev × risk_pct_for_bucket)
    """
    risk_rules = strat.get("meta", {}).get("risk_rules", {})
    small_set = set(risk_rules.get("small_buckets", ["5-10", "10-15"]))
    large_set = set(risk_rules.get("large_buckets", ["40-50", "50-200"]))
    small_pct = risk_rules.get("small_risk_pct", 0.005)
    medium_pct = risk_rules.get("medium_risk_pct", 0.015)
    large_pct = risk_rules.get("large_risk_pct", 0.03)

    daily_ev = 0.0
    for c in strat.get("cells", []):
        if not c.get("enabled", True):
            continue
        rr = c.get("risk_range", "")
        if rr in small_set:
            rpct = small_pct
        elif rr in large_set:
            rpct = large_pct
        else:
            rpct = medium_pct
        daily_ev += c.get("trades_per_day", 0) * c.get("ev", 0) * rpct

    return daily_ev


@app.get("/api/bot/period-pnl")
def api_bot_period_pnl():
    """Current day/week/month PNL + WR with cell-weighted baseline WR."""
    db = _get_bot_db()
    if not db:
        return JSONResponse({"week_pnl": 0, "week_trades": 0, "month_pnl": 0, "month_trades": 0})
    try:
        result = db.get_period_pnl()
        # Compute cell-weighted baseline WR for each period
        try:
            strat_path = os.path.join(
                _REPO_ROOT, "bot", "strategies", "manifest.json")
            with open(strat_path) as f:
                manifest = json.load(f)
                active_id = manifest.get("active_strategy") or manifest.get("active")
            if active_id:
                with open(os.path.join(
                        _REPO_ROOT, "bot", "strategies", f"{active_id}.json")) as f:
                    strat = json.load(f)
                baseline = {
                    (c["time_period"], c["risk_range"], c["setup"]): c["win_rate"]
                    for c in strat.get("cells", [])
                }
                for period in ("today", "week", "month"):
                    cells = result.pop(f"{period}_cells", [])
                    if not cells:
                        result[f"{period}_baseline_wr"] = None
                        continue
                    # Weighted avg: sum(cell_baseline_wr × cell_trades) / total_trades
                    weighted_sum = 0.0
                    total_n = 0
                    for c in cells:
                        key = (c["time_period"], c["risk_range"], c["setup"])
                        bwr = baseline.get(key)
                        if bwr is not None:
                            weighted_sum += bwr * c["n"]
                            total_n += c["n"]
                    result[f"{period}_baseline_wr"] = round(weighted_sum / total_n, 1) if total_n > 0 else None
        except Exception:
            # Clean up cell data even on error
            for period in ("today", "week", "month"):
                result.pop(f"{period}_cells", None)
        # Strategy-derived expected PnL: compute from active strategy cells
        # daily_ev_pct = Σ(trades_per_day × cell_ev × risk_pct_for_bucket)
        # Then: week = daily × 5, month = daily × 21
        # Percentiles estimated from per-trade σ and √N scaling
        try:
            state_file = os.path.join(_BOT_STATE_DIR, "bot_state.json")
            with open(state_file) as f:
                state = json.load(f)
            bal = (state.get("start_balance") or 80000) + (state.get("realized_pnl") or 0)

            # Compute percentiles from WF backtest daily P&L (dynamic).
            # Reads all runs in backtest manifest, groups daily PnL into
            # weeks/months as % of balance, returns empirical percentiles.
            # Cached for 1 hour. Auto-updates when strategy or backtests change.
            pctls = _get_backtest_percentiles(active_id) if active_id else {}

            if "week" in pctls:
                wp = pctls["week"]
                result["week_expected_pnl"] = round(bal * wp["p50"] / 100, 0)
                result["week_pctls"] = {
                    k: round(bal * v / 100) for k, v in wp.items()
                }
            if "month" in pctls:
                mp = pctls["month"]
                result["month_expected_pnl"] = round(bal * mp["p50"] / 100, 0)
                result["month_pctls"] = {
                    k: round(bal * v / 100) for k, v in mp.items()
                }
        except Exception:
            pass
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/api/bot/slippage")
def api_bot_slippage():
    """Cumulative slippage stats since live start."""
    db = _get_bot_db()
    if not db:
        return JSONResponse({"error": "No bot database found"})
    try:
        rows = db.query("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN exit_reason = 'SL' THEN 1 ELSE 0 END) as total_sl,
                SUM(CASE WHEN exit_reason = 'SL' AND stop_slippage_pts > 0 THEN 1 ELSE 0 END) as slipped_sl,
                ROUND(AVG(CASE WHEN exit_reason = 'SL' AND stop_slippage_pts > 0 THEN stop_slippage_pts END), 2) as avg_slip_pts,
                ROUND(MAX(stop_slippage_pts), 2) as max_slip_pts,
                ROUND(SUM(stop_slippage_pts * contracts * 20), 2) as total_slip_cost,
                ROUND(SUM(CASE WHEN net_pnl < 0 THEN ABS(net_pnl) ELSE 0 END), 2) as gross_loss,
                ROUND(SUM(commission), 2) as total_commission
            FROM trades WHERE exit_reason IS NOT NULL
        """)
        r = rows[0] if rows else {}
        slip_cost = r.get("total_slip_cost") or 0
        gross_loss = r.get("gross_loss") or 0
        r["slip_pct_of_gross_loss"] = round(slip_cost / gross_loss * 100, 1) if gross_loss > 0 else 0
        # Backtest baseline: 0.61pt avg slippage (2025 enrichment)
        r["baseline_avg_slip_pts"] = 0.61
        return JSONResponse(r)
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/api/bot/daily-history")
def api_bot_daily_history(days: int = 30):
    """Daily P&L history for the last N trading days (default 30)."""
    db = _get_bot_db()
    if not db:
        return JSONResponse({"error": "No bot database found"})
    try:
        return JSONResponse({"daily": db.get_daily_summary(days=days)})
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/api/bot/funnel")
def api_bot_funnel():
    """FVG → mitigation → trade → win conversion funnel for today."""
    db = _get_bot_db()
    if not db:
        return JSONResponse({"error": "No bot database found"})
    try:
        return JSONResponse(db.get_funnel(_today_str()))
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/api/bot/slippage")
def api_bot_slippage():
    """Entry and stop slippage analysis by setup type."""
    db = _get_bot_db()
    if not db:
        return JSONResponse({"error": "No bot database found"})
    try:
        return JSONResponse({"slippage": db.get_slippage_report(days=30)})
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/api/bot/drift")
def api_bot_drift():
    """Compare live trade outcomes to strategy cell expectations (win rate drift)."""
    db = _get_bot_db()
    if not db:
        return JSONResponse({"error": "No bot database found"})
    try:
        # Per-cell realized vs expected win rate
        drift = db.query('''
            SELECT
                setup, time_period, risk_range,
                COUNT(*) as live_trades,
                ROUND(SUM(CASE WHEN exit_reason = 'TP' THEN 1.0 ELSE 0 END) / COUNT(*) * 100, 1) as realized_wr,
                ROUND(AVG(cell_win_rate), 1) as expected_wr,
                ROUND(SUM(net_pnl), 2) as realized_pnl,
                ROUND(AVG(cell_ev), 4) as expected_ev
            FROM trades
            WHERE exit_reason IS NOT NULL
              AND trade_date >= date("now", "-30 days")
            GROUP BY setup, time_period, risk_range
            HAVING COUNT(*) >= 3
            ORDER BY realized_pnl DESC
        ''')
        # Aggregate drift
        overall = db.query('''
            SELECT
                COUNT(*) as total_trades,
                ROUND(SUM(CASE WHEN exit_reason = 'TP' THEN 1.0 ELSE 0 END) / NULLIF(COUNT(*), 0) * 100, 1) as realized_wr,
                ROUND(AVG(cell_win_rate), 1) as expected_wr,
                ROUND(SUM(net_pnl), 2) as realized_pnl,
                ROUND(AVG(entry_slippage_pts), 3) as avg_entry_slip,
                ROUND(AVG(CASE WHEN exit_reason = 'SL' THEN stop_slippage_pts END), 3) as avg_stop_slip
            FROM trades
            WHERE exit_reason IS NOT NULL
              AND trade_date >= date("now", "-30 days")
        ''')
        return JSONResponse({
            "per_cell": drift,
            "overall": overall[0] if overall else {},
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
