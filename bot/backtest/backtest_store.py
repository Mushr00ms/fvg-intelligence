"""
backtest_store.py — Store/load backtest results for the dashboard.

Layout:
    bot/backtest/results/
        manifest.json         # list of all runs with summary
        {run_id}.json         # full results per run
"""

import json
import os
import uuid
from datetime import datetime, timezone


_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _store_dir(d=None):
    d = d or _DEFAULT_DIR
    os.makedirs(d, exist_ok=True)
    return d


def save_results(results, store_dir=None):
    """Save backtest results and update manifest. Returns run_id."""
    d = _store_dir(store_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    now_iso = datetime.now(timezone.utc).isoformat()

    results["run_id"] = run_id
    results["saved_at"] = now_iso

    # Write results
    path = os.path.join(d, f"{run_id}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)

    # Update manifest
    manifest = load_manifest(store_dir)
    meta = results.get("meta", {})
    summary = results.get("summary", {})
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
        "saved_at": now_iso,
    })
    manifest["last_updated"] = now_iso

    mpath = os.path.join(d, "manifest.json")
    tmp = mpath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, mpath)

    return run_id


def load_results(run_id, store_dir=None):
    """Load a backtest result by run_id."""
    d = _store_dir(store_dir)
    path = os.path.join(d, f"{run_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Backtest run '{run_id}' not found")
    with open(path) as f:
        return json.load(f)


def load_manifest(store_dir=None):
    """Load the manifest."""
    d = _store_dir(store_dir)
    path = os.path.join(d, "manifest.json")
    if not os.path.exists(path):
        return {"runs": [], "last_updated": None}
    with open(path) as f:
        return json.load(f)


def delete_results(run_id, store_dir=None):
    """Delete a backtest run."""
    d = _store_dir(store_dir)
    path = os.path.join(d, f"{run_id}.json")
    if os.path.exists(path):
        os.remove(path)
    manifest = load_manifest(store_dir)
    manifest["runs"] = [r for r in manifest["runs"] if r["run_id"] != run_id]
    mpath = os.path.join(d, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
