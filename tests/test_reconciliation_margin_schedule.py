import json

from bot.backtest.backtester import _margin_per_contract_for_minute, _normalize_margin_schedule
from bot.backtest.eod_reconciler import build_backtest_config, load_margin_schedule
from bot.bot_config import BotConfig


def test_load_margin_schedule_filters_live_rth_snapshots(tmp_path):
    log_path = tmp_path / "2026-04-09.jsonl"
    rows = [
        {"ts": "2026-04-09T08:31:41.405540-04:00", "event": "margin_fetched", "mode": "live", "per_contract": 55049.39},
        {"ts": "2026-04-09T09:31:43.878953-04:00", "event": "margin_fetched", "mode": "live", "per_contract": 38631.67},
        {"ts": "2026-04-09T10:01:44.065105-04:00", "event": "margin_fetched", "mode": "live", "per_contract": 38520.40},
        {"ts": "2026-04-09T10:31:44.065105-04:00", "event": "margin_fetched", "mode": "paper", "per_contract": 12345.0},
    ]
    log_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    schedule = load_margin_schedule(str(tmp_path), "2026-04-09")

    assert schedule == [
        {"minute": 9 * 60 + 31, "per_contract": 38631.67},
        {"minute": 10 * 60 + 1, "per_contract": 38520.4},
    ]


def test_margin_schedule_lookup_uses_latest_known_snapshot():
    schedule = _normalize_margin_schedule([
        {"minute": 9 * 60 + 31, "per_contract": 38631.67},
        {"minute": 10 * 60 + 1, "per_contract": 38520.40},
    ])

    assert _margin_per_contract_for_minute(schedule, 10 * 60, 36750.0) == 38631.67
    assert _margin_per_contract_for_minute(schedule, 10 * 60 + 5, 36750.0) == 38520.40
    assert _margin_per_contract_for_minute([], 10 * 60 + 5, 36750.0) == 36750.0


def test_build_backtest_config_carries_margin_schedule():
    cfg = build_backtest_config(
        BotConfig(),
        {"meta": {}, "cells": []},
        100000.0,
        margin_schedule=[{"minute": 600, "per_contract": 38520.4}],
    )

    assert cfg["margin_schedule"] == [{"minute": 600, "per_contract": 38520.4}]
