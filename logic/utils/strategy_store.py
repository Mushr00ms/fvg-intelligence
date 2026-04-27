"""
strategy_store.py — Save/load trading strategies as JSON for the bot and dashboard.

Store layout:
    bot/strategies/
        manifest.json          # index of all strategies + active_strategy pointer
        {id}.json              # individual strategy file
"""

import json
import os
import re
from datetime import datetime, timezone


_DEFAULT_STORE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "bot", "strategies")
)

# Valid R:R target values — any positive float (optimizer may produce values outside 1.0-3.0)
VALID_RR_TARGETS = None  # No fixed list; validated as positive number below

# Valid setup types
VALID_SETUPS = ["mit_extreme", "mid_extreme"]

SCHEMA_VERSION = "1.0"


def _store_dir(store_dir=None):
    d = store_dir or _DEFAULT_STORE_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _manifest_path(store_dir=None):
    return os.path.join(_store_dir(store_dir), "manifest.json")


def _strategy_path(strategy_id, store_dir=None):
    return os.path.join(_store_dir(store_dir), f"{strategy_id}.json")


def _slugify(name):
    """Convert a strategy name to a filesystem-safe ID."""
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "unnamed"


def _load_manifest_raw(store_dir=None):
    path = _manifest_path(store_dir)
    if not os.path.exists(path):
        return {
            "strategies": [],
            "active_strategy": None,
            "last_updated": None,
        }
    with open(path) as f:
        return json.load(f)


def _save_manifest(manifest, store_dir=None):
    """Atomic manifest write."""
    path = _manifest_path(store_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)


def _compute_stats(cells):
    """Recompute the stats block from enabled cells."""
    enabled = [c for c in cells if c.get("enabled", True)]
    if not enabled:
        return {
            "total_cells": len(cells),
            "enabled_cells": 0,
            "weighted_ev": 0.0,
            "expected_trades_per_day": 0.0,
            "time_coverage": [],
        }

    total_samples = sum(c.get("samples", 0) for c in enabled)
    if total_samples > 0:
        weighted_ev = sum(
            c.get("ev", 0) * c.get("samples", 0) for c in enabled
        ) / total_samples
    else:
        weighted_ev = 0.0

    trades_per_day = sum(c.get("trades_per_day", 0) for c in enabled)
    time_periods = sorted(set(c["time_period"] for c in enabled), key=_time_sort_key)

    return {
        "total_cells": len(cells),
        "enabled_cells": len(enabled),
        "weighted_ev": round(weighted_ev, 4),
        "expected_trades_per_day": round(trades_per_day, 2),
        "time_coverage": time_periods,
    }


def _time_sort_key(tp):
    """Sort key for time period strings like '09:30-10:00'."""
    try:
        start = str(tp).split("-")[0].strip()
        h, m = map(int, start.split(":"))
        return h * 60 + m
    except (ValueError, IndexError):
        return 9999


def _ensure_id(strategy):
    """Ensure strategy has an id derived from name."""
    meta = strategy.get("meta", {})
    if not meta.get("id"):
        name = meta.get("name", "Unnamed Strategy")
        meta["id"] = _slugify(name)
        strategy["meta"] = meta
    return meta["id"]


def validate_strategy(strategy):
    """
    Validate a strategy dict. Returns list of error strings (empty = valid).
    """
    errors = []

    if strategy.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"Unsupported schema version: {strategy.get('schema_version')} "
            f"(expected {SCHEMA_VERSION})"
        )

    meta = strategy.get("meta", {})
    if not meta.get("name"):
        errors.append("Missing strategy name in meta")

    cells = strategy.get("cells", [])
    if not cells:
        errors.append("No cells defined")

    enabled = [c for c in cells if c.get("enabled", True)]
    if not enabled:
        errors.append("No enabled cells")

    required_fields = [
        "time_period", "risk_range", "setup", "rr_target",
        "ev", "win_rate", "samples",
    ]
    for i, cell in enumerate(cells):
        for field in required_fields:
            if field not in cell:
                label = f"{cell.get('time_period', '?')}/{cell.get('risk_range', '?')}"
                errors.append(f"Cell {i} ({label}) missing field: {field}")

        if cell.get("setup") and cell["setup"] not in VALID_SETUPS:
            errors.append(
                f"Cell {i} has invalid setup: {cell['setup']} "
                f"(expected one of {VALID_SETUPS})"
            )

        rr = cell.get("rr_target")
        if rr is not None and (not isinstance(rr, (int, float)) or rr <= 0):
            errors.append(
                f"Cell {i} has invalid rr_target: {rr} (must be positive number)"
            )

    return errors


def save_strategy(strategy, store_dir=None):
    """
    Save a strategy to the store. Recomputes stats, writes atomically,
    and updates manifest. Returns the strategy ID.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Ensure schema version
    strategy["schema_version"] = SCHEMA_VERSION

    # Ensure ID
    strategy_id = _ensure_id(strategy)

    # Set timestamps
    meta = strategy["meta"]
    if not meta.get("created_at"):
        meta["created_at"] = now_iso
    meta["updated_at"] = now_iso

    # Recompute stats
    strategy["stats"] = _compute_stats(strategy.get("cells", []))

    # Write strategy file atomically
    store = _store_dir(store_dir)
    filepath = _strategy_path(strategy_id, store_dir)
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(strategy, f, indent=2)
    os.replace(tmp, filepath)

    # Update manifest
    manifest = _load_manifest_raw(store_dir)
    strategies = manifest.get("strategies", [])

    # Build manifest entry
    entry = {
        "id": strategy_id,
        "name": meta.get("name", ""),
        "description": meta.get("description", ""),
        "source_dataset": meta.get("source_dataset", ""),
        "ticker": meta.get("ticker", ""),
        "timeframe": meta.get("timeframe", ""),
        "cell_count": strategy["stats"]["total_cells"],
        "enabled_count": strategy["stats"]["enabled_cells"],
        "weighted_ev": strategy["stats"]["weighted_ev"],
        "expected_trades_per_day": strategy["stats"]["expected_trades_per_day"],
        "created_at": meta.get("created_at", now_iso),
        "updated_at": now_iso,
    }

    # Replace existing or append
    strategies = [s for s in strategies if s["id"] != strategy_id]
    strategies.append(entry)
    manifest["strategies"] = strategies
    manifest["last_updated"] = now_iso

    _save_manifest(manifest, store_dir)

    return strategy_id


def load_strategy(strategy_id, store_dir=None):
    """Load a strategy by ID. Returns dict or raises FileNotFoundError."""
    filepath = _strategy_path(strategy_id, store_dir)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Strategy '{strategy_id}' not found at {filepath}")
    with open(filepath) as f:
        return json.load(f)


def delete_strategy(strategy_id, store_dir=None):
    """Delete a strategy file and remove from manifest."""
    filepath = _strategy_path(strategy_id, store_dir)
    if os.path.exists(filepath):
        os.remove(filepath)

    manifest = _load_manifest_raw(store_dir)
    manifest["strategies"] = [
        s for s in manifest.get("strategies", []) if s["id"] != strategy_id
    ]

    # Clear active if it was the deleted one
    if manifest.get("active_strategy") == strategy_id:
        manifest["active_strategy"] = None

    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    _save_manifest(manifest, store_dir)


def get_strategy_manifest(store_dir=None):
    """Return the manifest dict."""
    return _load_manifest_raw(store_dir)


def set_active_strategy(strategy_id, store_dir=None):
    """Set the active strategy in the manifest. Validates the strategy exists."""
    filepath = _strategy_path(strategy_id, store_dir)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Strategy '{strategy_id}' not found")

    manifest = _load_manifest_raw(store_dir)
    manifest["active_strategy"] = strategy_id
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()
    _save_manifest(manifest, store_dir)


def get_active_strategy(store_dir=None):
    """Load and return the currently active strategy, or None if none set."""
    manifest = _load_manifest_raw(store_dir)
    active_id = manifest.get("active_strategy")
    if not active_id:
        return None
    try:
        return load_strategy(active_id, store_dir)
    except FileNotFoundError:
        return None


def get_strategy_mtime(strategy_id, store_dir=None):
    """Return the modification time of a strategy file (for hot-reload)."""
    filepath = _strategy_path(strategy_id, store_dir)
    if not os.path.exists(filepath):
        return None
    return os.path.getmtime(filepath)
