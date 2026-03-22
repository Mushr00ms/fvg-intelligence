"""Tests for strategy loader: load, validate, lookup, hot-reload."""

import pytest
from bot.strategy.strategy_loader import StrategyLoader


class TestStrategyLoader:
    """Tests for loading and querying the active strategy."""

    def test_load_active_strategy(self, sample_strategy):
        strategy_dir, _ = sample_strategy
        loader = StrategyLoader(strategy_dir)
        loader.load()
        assert loader.strategy_id == "test-strategy"
        assert loader.cell_count == 2  # 3 cells but 1 disabled

    def test_find_cell_matching(self, sample_strategy):
        strategy_dir, _ = sample_strategy
        loader = StrategyLoader(strategy_dir)
        loader.load()
        # Cell: 10:30-11:00, 10-15, mit_extreme
        cell = loader.find_cell("10:30-11:00", 12.5)  # 12.5 falls in 10-15 range
        assert cell is not None
        assert cell["setup"] == "mit_extreme"
        assert cell["rr_target"] == 3.0
        assert cell["ev"] == 0.2736

    def test_find_cell_no_match(self, sample_strategy):
        strategy_dir, _ = sample_strategy
        loader = StrategyLoader(strategy_dir)
        loader.load()
        # No cell for 09:30-10:00
        cell = loader.find_cell("09:30-10:00", 12.5)
        assert cell is None

    def test_find_cell_wrong_risk_range(self, sample_strategy):
        strategy_dir, _ = sample_strategy
        loader = StrategyLoader(strategy_dir)
        loader.load()
        # 10:30-11:00 has cells for 10-15, but risk=7.5 is in 5-10
        cell = loader.find_cell("10:30-11:00", 7.5)
        assert cell is None

    def test_disabled_cells_excluded(self, sample_strategy):
        strategy_dir, _ = sample_strategy
        loader = StrategyLoader(strategy_dir)
        loader.load()
        # Cell at 14:30-15:00 / 5-10 exists but is disabled
        cell = loader.find_cell("14:30-15:00", 7.5)
        assert cell is None

    def test_risk_to_range_mapping(self):
        loader = StrategyLoader("/tmp/nonexistent")
        assert loader._risk_to_range(7.5) == "5-10"
        assert loader._risk_to_range(12.0) == "10-15"
        assert loader._risk_to_range(15.0) == "15-20"
        assert loader._risk_to_range(22.0) == "20-25"
        assert loader._risk_to_range(5.0) == "5-10"
        assert loader._risk_to_range(9.99) == "5-10"
        assert loader._risk_to_range(10.0) == "10-15"

    def test_risk_to_range_out_of_bounds(self):
        loader = StrategyLoader("/tmp/nonexistent")
        assert loader._risk_to_range(3.0) is None   # Below 5
        assert loader._risk_to_range(80.0) is None   # At upper boundary (80 is edge of last bin)
        assert loader._risk_to_range(100.0) is None  # Above 80

    def test_load_no_strategy_raises(self, tmp_dir):
        import os
        strategy_dir = os.path.join(tmp_dir, "empty_strategies")
        os.makedirs(strategy_dir, exist_ok=True)
        loader = StrategyLoader(strategy_dir)
        with pytest.raises(RuntimeError, match="No active strategy"):
            loader.load()


class TestDuplicateKeyResolution:
    """Test that duplicate (time_period, risk_range) keeps highest EV."""

    def test_keeps_highest_ev(self, tmp_dir):
        import json, os
        strategy_dir = os.path.join(tmp_dir, "strategies")
        os.makedirs(strategy_dir, exist_ok=True)

        strategy = {
            "schema_version": "1.0",
            "meta": {"id": "dup-test", "name": "Dup Test", "ticker": "NQ", "timeframe": "5min"},
            "filters": {},
            "cells": [
                {
                    "time_period": "10:30-11:00", "risk_range": "10-15",
                    "setup": "mit_extreme", "rr_target": 3.0,
                    "enabled": True, "ev": 0.27, "win_rate": 31.8,
                    "samples": 223, "median_risk": 12.25, "trades_per_day": 0.18,
                },
                {
                    "time_period": "10:30-11:00", "risk_range": "10-15",
                    "setup": "mid_extreme", "rr_target": 2.75,
                    "enabled": True, "ev": 0.19, "win_rate": 31.7,
                    "samples": 309, "median_risk": 12.5, "trades_per_day": 0.25,
                },
            ],
            "stats": {},
        }

        with open(os.path.join(strategy_dir, "dup-test.json"), "w") as f:
            json.dump(strategy, f)
        manifest = {
            "strategies": [{"id": "dup-test"}],
            "active_strategy": "dup-test",
        }
        with open(os.path.join(strategy_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        loader = StrategyLoader(strategy_dir)
        loader.load()

        cell = loader.find_cell("10:30-11:00", 12.5)
        assert cell is not None
        assert cell["setup"] == "mit_extreme"  # Higher EV (0.27 > 0.19)
        assert cell["ev"] == 0.27
