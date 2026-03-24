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
from fastapi import FastAPI, HTTPException, Request
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
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            return JSONResponse(json.load(f))

    # Build manifest from existing result files
    runs = []
    for f in sorted(os.listdir(_BACKTEST_RESULTS_DIR)):
        if f.endswith(".json") and f != "manifest.json":
            try:
                with open(os.path.join(_BACKTEST_RESULTS_DIR, f)) as fh:
                    data = json.load(fh)
                meta = data.get("meta", {})
                summary = data.get("summary", {})
                runs.append({
                    "run_id": f.replace(".json", ""),
                    "strategy_name": meta.get("strategy_name", ""),
                    "start_date": meta.get("start_date", ""),
                    "end_date": meta.get("end_date", ""),
                    "total_trades": summary.get("total_trades", 0),
                    "net_pnl": summary.get("net_pnl", 0),
                    "win_rate": summary.get("win_rate", 0),
                })
            except Exception:
                continue
    return JSONResponse({"runs": runs})


# ── Live Bot endpoints ─────────────────────────────────────────────────────

_BOT_STATE_DIR = os.path.join(_REPO_ROOT, "bot", "bot_state")
_BOT_LOG_DIR = os.path.join(_REPO_ROOT, "bot", "logs")
_BOT_DB_PATH = os.path.join(_REPO_ROOT, "bot", "fvg_bot.db")


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
            if f.endswith(".json"):
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
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
