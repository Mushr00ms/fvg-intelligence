"""Tests for strategy store: save, load, manifest, active strategy."""

import json
import os
import pytest
import importlib
import sys

# Import strategy_store directly to avoid logic.utils.__init__ pulling in heavy deps
_mod_path = os.path.join(os.path.dirname(__file__), "..", "logic", "utils", "strategy_store.py")
spec = importlib.util.spec_from_file_location("strategy_store", _mod_path)
strategy_store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy_store)

save_strategy = strategy_store.save_strategy
load_strategy = strategy_store.load_strategy
delete_strategy = strategy_store.delete_strategy
get_strategy_manifest = strategy_store.get_strategy_manifest
set_active_strategy = strategy_store.set_active_strategy
get_active_strategy = strategy_store.get_active_strategy
validate_strategy = strategy_store.validate_strategy
get_strategy_mtime = strategy_store.get_strategy_mtime
SCHEMA_VERSION = strategy_store.SCHEMA_VERSION


def _make_strategy(name="Test Strategy", cells=None):
    if cells is None:
        cells = [
            {
                "time_period": "10:30-11:00",
                "risk_range": "10-15",
                "setup": "mit_extreme",
                "rr_target": 3.0,
                "enabled": True,
                "win_rate": 31.8,
                "ev": 0.2736,
                "median_risk": 12.25,
                "samples": 223,
                "trades_per_day": 0.18,
                "notes": "",
            }
        ]
    return {
        "schema_version": "1.0",
        "meta": {
            "name": name,
            "source_dataset": "rr_nq_5min_5y_30min",
            "ticker": "NQ",
            "timeframe": "5min",
        },
        "filters": {"min_samples": 200, "require_all_evs_positive": True},
        "cells": cells,
    }


class TestSaveAndLoad:
    """Tests for save/load round-trip."""

    def test_save_creates_file(self, tmp_dir):
        strategy = _make_strategy()
        sid = save_strategy(strategy, store_dir=tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, f"{sid}.json"))

    def test_save_creates_manifest(self, tmp_dir):
        strategy = _make_strategy()
        save_strategy(strategy, store_dir=tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "manifest.json"))

    def test_load_returns_saved_data(self, tmp_dir):
        strategy = _make_strategy()
        sid = save_strategy(strategy, store_dir=tmp_dir)
        loaded = load_strategy(sid, store_dir=tmp_dir)
        assert loaded["meta"]["name"] == "Test Strategy"
        assert len(loaded["cells"]) == 1
        assert loaded["cells"][0]["setup"] == "mit_extreme"

    def test_save_generates_id_from_name(self, tmp_dir):
        strategy = _make_strategy(name="Morning Momentum v3")
        sid = save_strategy(strategy, store_dir=tmp_dir)
        assert sid == "morning-momentum-v3"

    def test_save_recomputes_stats(self, tmp_dir):
        strategy = _make_strategy()
        sid = save_strategy(strategy, store_dir=tmp_dir)
        loaded = load_strategy(sid, store_dir=tmp_dir)
        assert loaded["stats"]["total_cells"] == 1
        assert loaded["stats"]["enabled_cells"] == 1
        assert loaded["stats"]["weighted_ev"] == 0.2736

    def test_save_updates_timestamps(self, tmp_dir):
        strategy = _make_strategy()
        sid = save_strategy(strategy, store_dir=tmp_dir)
        loaded = load_strategy(sid, store_dir=tmp_dir)
        assert loaded["meta"]["created_at"] is not None
        assert loaded["meta"]["updated_at"] is not None

    def test_overwrite_existing(self, tmp_dir):
        strategy = _make_strategy()
        sid1 = save_strategy(strategy, store_dir=tmp_dir)
        strategy["cells"][0]["ev"] = 0.5
        sid2 = save_strategy(strategy, store_dir=tmp_dir)
        assert sid1 == sid2
        loaded = load_strategy(sid1, store_dir=tmp_dir)
        assert loaded["cells"][0]["ev"] == 0.5

    def test_load_nonexistent_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            load_strategy("nonexistent", store_dir=tmp_dir)


class TestManifest:
    """Tests for manifest operations."""

    def test_manifest_empty_initially(self, tmp_dir):
        manifest = get_strategy_manifest(store_dir=tmp_dir)
        assert manifest["strategies"] == []
        assert manifest["active_strategy"] is None

    def test_manifest_updated_on_save(self, tmp_dir):
        strategy = _make_strategy()
        save_strategy(strategy, store_dir=tmp_dir)
        manifest = get_strategy_manifest(store_dir=tmp_dir)
        assert len(manifest["strategies"]) == 1
        assert manifest["strategies"][0]["name"] == "Test Strategy"

    def test_multiple_strategies_in_manifest(self, tmp_dir):
        save_strategy(_make_strategy("Strategy A"), store_dir=tmp_dir)
        save_strategy(_make_strategy("Strategy B"), store_dir=tmp_dir)
        manifest = get_strategy_manifest(store_dir=tmp_dir)
        assert len(manifest["strategies"]) == 2


class TestActiveStrategy:
    """Tests for active strategy management."""

    def test_set_active(self, tmp_dir):
        strategy = _make_strategy()
        sid = save_strategy(strategy, store_dir=tmp_dir)
        set_active_strategy(sid, store_dir=tmp_dir)
        manifest = get_strategy_manifest(store_dir=tmp_dir)
        assert manifest["active_strategy"] == sid

    def test_get_active(self, tmp_dir):
        strategy = _make_strategy()
        sid = save_strategy(strategy, store_dir=tmp_dir)
        set_active_strategy(sid, store_dir=tmp_dir)
        active = get_active_strategy(store_dir=tmp_dir)
        assert active is not None
        assert active["meta"]["name"] == "Test Strategy"

    def test_get_active_none(self, tmp_dir):
        assert get_active_strategy(store_dir=tmp_dir) is None

    def test_set_active_nonexistent_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            set_active_strategy("nonexistent", store_dir=tmp_dir)


class TestDelete:
    """Tests for strategy deletion."""

    def test_delete_removes_file(self, tmp_dir):
        strategy = _make_strategy()
        sid = save_strategy(strategy, store_dir=tmp_dir)
        delete_strategy(sid, store_dir=tmp_dir)
        assert not os.path.exists(os.path.join(tmp_dir, f"{sid}.json"))

    def test_delete_removes_from_manifest(self, tmp_dir):
        strategy = _make_strategy()
        sid = save_strategy(strategy, store_dir=tmp_dir)
        delete_strategy(sid, store_dir=tmp_dir)
        manifest = get_strategy_manifest(store_dir=tmp_dir)
        assert len(manifest["strategies"]) == 0

    def test_delete_clears_active_if_match(self, tmp_dir):
        strategy = _make_strategy()
        sid = save_strategy(strategy, store_dir=tmp_dir)
        set_active_strategy(sid, store_dir=tmp_dir)
        delete_strategy(sid, store_dir=tmp_dir)
        manifest = get_strategy_manifest(store_dir=tmp_dir)
        assert manifest["active_strategy"] is None


class TestValidation:
    """Tests for strategy validation."""

    def test_valid_strategy(self):
        strategy = _make_strategy()
        strategy["schema_version"] = SCHEMA_VERSION
        errors = validate_strategy(strategy)
        assert errors == []

    def test_wrong_schema_version(self):
        strategy = _make_strategy()
        strategy["schema_version"] = "99.0"
        errors = validate_strategy(strategy)
        assert any("schema" in e.lower() for e in errors)

    def test_missing_name(self):
        strategy = _make_strategy()
        strategy["schema_version"] = SCHEMA_VERSION
        strategy["meta"]["name"] = ""
        errors = validate_strategy(strategy)
        assert any("name" in e.lower() for e in errors)

    def test_no_cells(self):
        strategy = _make_strategy()
        strategy["schema_version"] = SCHEMA_VERSION
        strategy["cells"] = []
        errors = validate_strategy(strategy)
        assert any("no cells" in e.lower() for e in errors)

    def test_invalid_setup(self):
        strategy = _make_strategy()
        strategy["schema_version"] = SCHEMA_VERSION
        strategy["cells"][0]["setup"] = "invalid_setup"
        errors = validate_strategy(strategy)
        assert any("invalid setup" in e.lower() for e in errors)

    def test_invalid_rr_target(self):
        strategy = _make_strategy()
        strategy["schema_version"] = SCHEMA_VERSION
        strategy["cells"][0]["rr_target"] = -1.0  # Negative is invalid
        errors = validate_strategy(strategy)
        assert any("rr_target" in e.lower() for e in errors)

    def test_valid_rr_target_outside_legacy_range(self):
        """Optimizer may produce targets like 0.5R or 4.5R — these are valid."""
        strategy = _make_strategy()
        strategy["schema_version"] = SCHEMA_VERSION
        for rr in [0.25, 0.5, 4.5, 10.0]:
            strategy["cells"][0]["rr_target"] = rr
            errors = validate_strategy(strategy)
            assert not any("rr_target" in e.lower() for e in errors), f"rr_target={rr} should be valid"

    def test_missing_required_field(self):
        strategy = _make_strategy()
        strategy["schema_version"] = SCHEMA_VERSION
        del strategy["cells"][0]["ev"]
        errors = validate_strategy(strategy)
        assert any("ev" in e for e in errors)


class TestMtime:
    """Tests for file modification time tracking."""

    def test_mtime_after_save(self, tmp_dir):
        strategy = _make_strategy()
        sid = save_strategy(strategy, store_dir=tmp_dir)
        mtime = get_strategy_mtime(sid, store_dir=tmp_dir)
        assert mtime is not None
        assert isinstance(mtime, float)

    def test_mtime_nonexistent(self, tmp_dir):
        assert get_strategy_mtime("nonexistent", store_dir=tmp_dir) is None
