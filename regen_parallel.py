#!/usr/bin/env python3
"""Regenerate ALL heatmap datasets in parallel."""
import json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logic", "heatmap_data", "manifest.json")
LOGIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logic")
MAX_WORKERS = 4

_SHORT_TO_LONG = {"y": " years", "m": " months", "w": " weeks", "d": " days"}

def period_to_env(p):
    for k, v in _SHORT_TO_LONG.items():
        if p.endswith(k) and p[:-1].isdigit():
            return p[:-1] + v
    return p

def run_one(d):
    ds_id = d["id"]
    env = {
        **os.environ,
        "FVG_TICKER":          "NQ",
        "FVG_TIMEFRAME":       d["timeframe_label"],
        "FVG_PERIOD":          period_to_env(d["data_period"]),
        "FVG_SESSION_MINUTES": str(d["session_period_minutes"]),
        "FVG_METHOD":          d["size_filtering_method"],
        "FVG_SIZE_START":      str(d["size_range_start"]),
        "FVG_SIZE_END":        str(d["size_range_end"]),
        "FVG_SIZE_STEP":       str(d["size_range_step"]),
        "FVG_MIN_EXP":         str(d["min_expansion_size"]),
    }
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=LOGIC_DIR, env=env,
        capture_output=True, text=True, timeout=900,
    )
    elapsed = time.time() - t0
    ok = result.returncode == 0
    err = ""
    if not ok:
        lines = (result.stderr or result.stdout or "").strip().split("\n")
        err = lines[-1] if lines else "unknown"
    return ds_id, ok, elapsed, err

def main():
    with open(MANIFEST) as f:
        datasets = json.load(f)["datasets"]

    total = len(datasets)
    print(f"=== Regenerating {total} datasets ({MAX_WORKERS} parallel workers) ===\n", flush=True)

    done = 0
    failed = []
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, d): d["id"] for d in datasets}
        for future in as_completed(futures):
            ds_id, ok, elapsed, err = future.result()
            done += 1
            sym = "✓" if ok else "✗"
            status = f"{elapsed:.0f}s" if ok else f"FAILED {elapsed:.0f}s — {err}"
            print(f"  [{done}/{total}] {sym} {ds_id}  ({status})", flush=True)
            if not ok:
                failed.append(ds_id)

    wall = time.time() - t_start
    print(f"\n=== Complete: {total - len(failed)}/{total} succeeded in {wall:.0f}s wall time ===")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f_id in failed:
            print(f"  - {f_id}")

if __name__ == "__main__":
    main()
