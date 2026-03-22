"""Shared fixtures for bot tests."""

import os
import sys
import json
import tempfile
import pytest

# Add project root to path
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory for state/strategy files."""
    return str(tmp_path)


@pytest.fixture
def sample_strategy(tmp_dir):
    """Create a sample strategy file and return its path and data."""
    strategy = {
        "schema_version": "1.0",
        "meta": {
            "id": "test-strategy",
            "name": "Test Strategy",
            "description": "Unit test strategy",
            "created_at": "2026-03-22T10:00:00",
            "updated_at": "2026-03-22T10:00:00",
            "source_dataset": "rr_nq_5min_5y_30min",
            "ticker": "NQ",
            "timeframe": "5min",
        },
        "filters": {
            "min_samples": 150,
            "require_all_evs_positive": True,
        },
        "cells": [
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
            },
            {
                "time_period": "13:00-13:30",
                "risk_range": "5-10",
                "setup": "mid_extreme",
                "rr_target": 2.5,
                "enabled": True,
                "win_rate": 35.2,
                "ev": 0.2306,
                "median_risk": 7.5,
                "samples": 384,
                "trades_per_day": 0.30,
                "notes": "",
            },
            {
                "time_period": "14:30-15:00",
                "risk_range": "5-10",
                "setup": "mit_extreme",
                "rr_target": 1.5,
                "enabled": False,
                "win_rate": 47.7,
                "ev": 0.193,
                "median_risk": 7.5,
                "samples": 329,
                "trades_per_day": 0.26,
                "notes": "disabled for testing",
            },
        ],
        "stats": {},
    }

    strategy_dir = os.path.join(tmp_dir, "strategies")
    os.makedirs(strategy_dir, exist_ok=True)

    # Write strategy file
    with open(os.path.join(strategy_dir, "test-strategy.json"), "w") as f:
        json.dump(strategy, f)

    # Write manifest
    manifest = {
        "strategies": [{"id": "test-strategy", "name": "Test Strategy"}],
        "active_strategy": "test-strategy",
        "last_updated": "2026-03-22T10:00:00",
    }
    with open(os.path.join(strategy_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    return strategy_dir, strategy


@pytest.fixture
def sample_bullish_fvg():
    """A bullish FVG: third candle low > first candle high."""
    return (
        {"open": 19480, "high": 19500, "low": 19470, "close": 19495, "date": "2026-03-22T10:30:00"},
        {"open": 19505, "high": 19530, "low": 19490, "close": 19520, "date": "2026-03-22T10:35:00"},
        {"open": 19525, "high": 19550, "low": 19515, "close": 19545, "date": "2026-03-22T10:40:00"},
    )


@pytest.fixture
def sample_bearish_fvg():
    """A bearish FVG: third candle high < first candle low."""
    return (
        {"open": 19550, "high": 19560, "low": 19530, "close": 19535, "date": "2026-03-22T11:00:00"},
        {"open": 19525, "high": 19535, "low": 19500, "close": 19505, "date": "2026-03-22T11:05:00"},
        {"open": 19510, "high": 19520, "low": 19490, "close": 19495, "date": "2026-03-22T11:10:00"},
    )


@pytest.fixture
def bot_config():
    """Create a BotConfig for testing."""
    from bot.bot_config import BotConfig
    return BotConfig(
        ib_port=7497,
        paper_mode=True,
        dry_run=True,
        state_dir=tempfile.mkdtemp(),
        log_dir=tempfile.mkdtemp(),
        strategy_dir=tempfile.mkdtemp(),
    )
