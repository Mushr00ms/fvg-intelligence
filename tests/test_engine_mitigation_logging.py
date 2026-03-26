"""Tests for detection-phase strategy skip logging in BotEngine."""

import asyncio
from types import SimpleNamespace

from bot.core.engine import BotEngine
from bot.state.trade_state import DailyState, FVGRecord


class _CaptureLogger:
    def __init__(self):
        self.records = []

    def log(self, event, **kwargs):
        self.records.append({"event": event, **kwargs})


class _NoCellStrategy:
    strategy_id = "test-strategy"

    def find_cell(self, time_period, risk_pts):
        return None


def _make_bearish_fvg():
    return FVGRecord(
        fvg_id="fvg-no-cell",
        fvg_type="bearish",
        zone_low=24410.25,
        zone_high=24414.75,
        time_candle1="2026-03-25T09:50:00",
        time_candle2="2026-03-25T09:55:00",
        time_candle3="2026-03-25T10:00:00",
        middle_open=24390.0,
        middle_low=24363.0,
        middle_high=24420.25,
        first_open=24400.0,
        time_period="09:30-10:00",
        formation_date="2026-03-25",
    )


def test_process_detection_logs_skips_when_no_strategy_cell():
    engine = BotEngine.__new__(BotEngine)
    engine.logger = _CaptureLogger()
    engine.strategy = _NoCellStrategy()
    engine.daily_state = DailyState(date="2026-03-25", start_balance=76000.0)
    engine.config = SimpleNamespace(risk_per_trade=0.01)
    engine.fvg_mgr = SimpleNamespace(
        remove=lambda fvg_id: None,
        active_fvgs=[],
    )

    asyncio.run(engine._process_detection(_make_bearish_fvg()))

    skipped = [r for r in engine.logger.records if r["event"] == "setup_skipped_strategy"]
    assert len(skipped) == 2
    assert {r["setup"] for r in skipped} == {"mit_extreme", "mid_extreme"}
    assert all(r["reason"] == "no_strategy_cell" for r in skipped)
    assert {r["risk_range"] for r in skipped} == {"10-15", "5-10"}
