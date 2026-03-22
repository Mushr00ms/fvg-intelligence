#!/usr/bin/env python3
"""Regenerate all heatmap datasets from the manifest with current analysis code."""
import json, os, subprocess, sys, time

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logic", "heatmap_data", "manifest.json")

def period_to_env(p):
    """Convert '5y' → '5 years', '15y' → '15 years', '2y' → '2 years'."""
    if p.endswith("y"):
        return p[:-1] + " years"
    return p

with open(MANIFEST) as f:
    datasets = json.load(f)["datasets"]

total = len(datasets)
print(f"=== Regenerating {total} datasets ===\n")

failed = []
for i, d in enumerate(datasets, 1):
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

    print(f"\n[{i}/{total}] {ds_id}")
    print(f"  tf={d['timeframe_label']} period={d['data_period']} session={d['session_period_minutes']}min "
          f"method={d['size_filtering_method']} range={d['size_range_start']}-{d['size_range_end']} "
          f"step={d['size_range_step']} exp={d['min_expansion_size']}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logic"),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ✗ FAILED in {elapsed:.0f}s")
        # Print last 5 lines of stderr for context
        err_lines = (result.stderr or result.stdout or "").strip().split("\n")
        for line in err_lines[-5:]:
            print(f"    {line}")
        failed.append(ds_id)
    else:
        print(f"  ✓ done in {elapsed:.0f}s")

print(f"\n=== Complete: {total - len(failed)}/{total} succeeded ===")
if failed:
    print(f"Failed ({len(failed)}):")
    for f_id in failed:
        print(f"  - {f_id}")
