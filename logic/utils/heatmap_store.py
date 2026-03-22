"""
heatmap_store.py — Save/load heatmap analysis results as JSON for the dashboard.

Store layout:
    logic/heatmap_data/
        manifest.json          # index of all datasets
        {id}.json              # column-oriented data file per dataset
"""

import json
import math
import os
import re
from datetime import datetime, timezone

import pandas as pd

def _normalize_period(p: str) -> str:
    """Normalize period to short form: '5 years' → '5y', '15 months' → '15m', etc."""
    result = str(p).strip().lower()
    for long, short in [
        ("years", "y"), ("year", "y"),
        ("months", "m"), ("month", "m"),
        ("weeks", "w"), ("week", "w"),
        ("days", "d"), ("day", "d"),
        ("quarters", "q"), ("quarter", "q"),
    ]:
        result = result.replace(long, short)
    return result.replace(" ", "")


_DEFAULT_STORE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "heatmap_data")
)

# Columns written to data files (in order)
DATA_COLUMNS = [
    "time_period",
    "size_range",
    "total_fvgs",
    "mitigated_fvgs",
    "mitigation_rate",
    "invalidation_rate",
    "avg_expansion_size",
    "p75_expansion_size",
    "expansion_efficiency",
    "p75_expansion_efficiency",
    "p75_mitigation_time",
    "p75_expansion_time",
    "p75_time_to_target",
    "p75_time_to_invalidation",
    "optimal_target",
    "optimal_ev",
    "avg_penetration_depth",
    "p75_penetration_depth",
    "avg_penetration_candle_count",
    "avg_penetration_depth_ratio",
    "avg_midpoint_crossing_count",
    "midpoint_crossed_pct",
    "avg_risk_points",
    "avg_rr",
    "rr_1_0_hit_rate",
    "rr_1_5_hit_rate",
    "rr_2_0_hit_rate",
]


def _store(store_dir):
    d = store_dir or _DEFAULT_STORE_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _manifest_path(store_dir):
    return os.path.join(_store(store_dir), "manifest.json")


def _data_path(dataset_id, store_dir):
    return os.path.join(_store(store_dir), f"{dataset_id}.json")


def _clean_float(v):
    """Convert float to JSON-safe value (round to 4dp, NaN → None)."""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def get_dataset_id(
    ticker,
    timeframe_label,
    data_period,
    session_period_minutes,
    size_range_start,
    size_range_end,
    min_expansion_size,
):
    """Return a deterministic, filesystem-safe dataset ID."""
    ticker = str(ticker).lower()
    tf = str(timeframe_label).lower()
    dp = _normalize_period(data_period)
    spm = int(session_period_minutes)
    start = f"{float(size_range_start):.2f}"
    end = f"{float(size_range_end):.2f}"
    exp_f = float(min_expansion_size)
    exp = str(int(exp_f)) if exp_f == int(exp_f) else str(exp_f).replace(".", "_")
    return f"{ticker}_{tf}_{dp}_{spm}min_{start}to{end}_exp{exp}"


def save_dataset(
    df,
    ticker,
    timeframe_label,
    data_period,
    session_period_minutes,
    size_filtering_method,
    size_range_start,
    size_range_end,
    size_range_step,
    min_expansion_size,
    source_csv_path=None,
    store_dir=None,
):
    """
    Persist a size-time analysis DataFrame to the heatmap store.

    Returns the dataset ID string.
    """
    data_period = _normalize_period(data_period)
    dataset_id = get_dataset_id(
        ticker, timeframe_label, data_period,
        session_period_minutes, size_range_start, size_range_end, min_expansion_size,
    )

    # --- Build column-oriented data payload ---
    # Keep only columns that exist in df
    available = [c for c in DATA_COLUMNS if c in df.columns]
    # Ensure integer columns
    int_cols = {"total_fvgs", "mitigated_fvgs"}

    rows = []
    for _, row in df[available].iterrows():
        rec = []
        for col in available:
            v = row[col]
            if col in int_cols:
                rec.append(int(v) if pd.notna(v) else None)
            elif col in ("time_period", "size_range"):
                rec.append(str(v) if pd.notna(v) else None)
            else:
                rec.append(_clean_float(v))
        rows.append(rec)

    # Collect unique labels for manifest
    time_periods = sorted(df["time_period"].dropna().unique().tolist(), key=_time_sort_key) if "time_period" in df.columns else []
    size_ranges = sorted(df["size_range"].dropna().unique().tolist(), key=_size_sort_key) if "size_range" in df.columns else []

    now_iso = datetime.now(timezone.utc).isoformat()

    data_payload = {
        "meta": {
            "id": dataset_id,
            "ticker": ticker,
            "timeframe_label": timeframe_label,
            "data_period": data_period,
            "session_period_minutes": session_period_minutes,
            "size_filtering_method": size_filtering_method,
            "size_range_start": float(size_range_start),
            "size_range_end": float(size_range_end),
            "size_range_step": float(size_range_step),
            "min_expansion_size": float(min_expansion_size),
            "source_csv_path": source_csv_path,
            "created_at": now_iso,
        },
        "columns": available,
        "rows": rows,
    }

    # Write data file atomically
    store = _store(store_dir)
    data_file = _data_path(dataset_id, store_dir)
    tmp_data = data_file + ".tmp"
    with open(tmp_data, "w") as f:
        json.dump(data_payload, f, separators=(",", ":"))
    os.replace(tmp_data, data_file)

    # Update manifest
    manifest = _load_manifest_raw(store_dir)
    entry = {
        "id": dataset_id,
        "ticker": ticker,
        "timeframe_label": timeframe_label,
        "data_period": data_period,
        "session_period_minutes": session_period_minutes,
        "size_filtering_method": size_filtering_method,
        "size_range_start": float(size_range_start),
        "size_range_end": float(size_range_end),
        "size_range_step": float(size_range_step),
        "min_expansion_size": float(min_expansion_size),
        "created_at": now_iso,
        "data_file": os.path.relpath(data_file),
        "time_periods": [str(x) for x in time_periods],
        "size_ranges": [str(x) for x in size_ranges],
    }

    # Replace existing entry with same id or append
    datasets = manifest.get("datasets", [])
    datasets = [d for d in datasets if d["id"] != dataset_id]
    datasets.append(entry)
    manifest["datasets"] = datasets
    manifest["last_updated"] = now_iso

    manifest_path = _manifest_path(store_dir)
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_manifest, manifest_path)

    return dataset_id


def load_dataset(dataset_id, store_dir=None):
    """Load a dataset from the store and return as a DataFrame."""
    data_file = _data_path(dataset_id, store_dir)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Dataset '{dataset_id}' not found at {data_file}")
    with open(data_file) as f:
        payload = json.load(f)
    return pd.DataFrame(payload["rows"], columns=payload["columns"])


def get_manifest(store_dir=None):
    """Return the manifest dict."""
    return _load_manifest_raw(store_dir)


def _load_manifest_raw(store_dir):
    path = _manifest_path(store_dir)
    if not os.path.exists(path):
        return {"schema_version": "1.0", "last_updated": None, "datasets": []}
    with open(path) as f:
        return json.load(f)


def _time_sort_key(tp):
    """Sort key for time period strings like '09:30-10:30'."""
    try:
        start = str(tp).split("-")[0].strip()
        h, m = map(int, start.split(":"))
        return h * 60 + m
    except (ValueError, IndexError):
        return 9999


def _size_sort_key(sr):
    """Sort key for size range strings like '10.00' or '10.00-20.00'."""
    try:
        return float(str(sr).split("-")[0])
    except (ValueError, IndexError):
        return 9999.0


# ---------------------------------------------------------------------------
# CSV filename regex for backfill
# Pattern: {ticker}_fvg_{label}_{period}_{interval}min_size_time_analysis_{start}to{end}_exp{exp}_results.csv
# ---------------------------------------------------------------------------
_CSV_RE = re.compile(
    r"^(?P<ticker>[a-z0-9]+)_fvg_"
    r"(?P<label>[a-z0-9]+)_"
    r"(?P<period>[0-9]+[ymdwq])_"
    r"(?P<interval>[0-9]+)min_"
    r"size_time_analysis_"
    r"(?P<start>[0-9]+\.[0-9]+)to(?P<end>[0-9]+\.[0-9]+)_"
    r"exp(?P<exp>[0-9]+(?:\.[0-9]+)?)_results\.csv$",
    re.IGNORECASE,
)


def backfill_from_csv_dir(csv_dir, store_dir=None):
    """
    Parse existing CSVs in csv_dir by filename and populate the store.

    Returns the number of datasets successfully imported.
    """
    count = 0
    csv_dir = os.path.abspath(csv_dir)
    if not os.path.isdir(csv_dir):
        print(f"[backfill] Directory not found: {csv_dir}")
        return count

    for fname in sorted(os.listdir(csv_dir)):
        m = _CSV_RE.match(fname)
        if not m:
            continue
        fpath = os.path.join(csv_dir, fname)
        try:
            df = pd.read_csv(fpath)
            if df.empty:
                continue

            ticker = m.group("ticker").upper()
            label = m.group("label")
            period = m.group("period")
            interval = int(m.group("interval"))
            start = float(m.group("start"))
            end = float(m.group("end"))
            exp = float(m.group("exp"))

            # Infer step from size_range column if possible
            step = 1.0
            if "size_range" in df.columns:
                vals = df["size_range"].dropna().unique()
                numeric = []
                for v in vals:
                    try:
                        numeric.append(float(str(v).split("-")[0]))
                    except ValueError:
                        pass
                numeric = sorted(set(numeric))
                if len(numeric) >= 2:
                    diffs = [numeric[i+1] - numeric[i] for i in range(len(numeric)-1)]
                    step = round(min(diffs), 4)

            # Infer filtering method from size_range format
            method = "bins"
            if "size_range" in df.columns:
                sample = str(df["size_range"].iloc[0])
                if "-" not in sample:
                    method = "cumulative"

            # Skip if the already-stored dataset has more columns
            # (prevents a less-complete CSV from overwriting a richer one)
            ds_id = get_dataset_id(ticker, label, period, interval, start, end, exp)
            existing_path = _data_path(ds_id, store_dir)
            if os.path.exists(existing_path):
                try:
                    with open(existing_path) as ef:
                        existing = json.load(ef)
                    existing_cols = set(existing.get("columns", []))
                    new_cols = set(c for c in DATA_COLUMNS if c in df.columns)
                    if existing_cols >= new_cols:
                        print(f"[backfill] Skipped {fname} (existing has richer columns)")
                        continue
                except Exception:
                    pass

            save_dataset(
                df=df,
                ticker=ticker,
                timeframe_label=label,
                data_period=period,
                session_period_minutes=interval,
                size_filtering_method=method,
                size_range_start=start,
                size_range_end=end,
                size_range_step=step,
                min_expansion_size=exp,
                source_csv_path=fpath,
                store_dir=store_dir,
            )
            print(f"[backfill] Imported {fname}")
            count += 1
        except Exception as e:
            print(f"[backfill] Skipped {fname}: {e}")

    print(f"[backfill] Done — {count} datasets imported")
    return count
