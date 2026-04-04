"""Tests: engine skips non-trading days and witching days at startup."""

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytz

from bot.core.engine import BotEngine

NY_TZ = pytz.timezone("America/New_York")


class _CaptureLogger:
    def __init__(self):
        self.records = []

    def log(self, event, **kwargs):
        self.records.append({"event": event, **kwargs})


class _MockClock:
    def __init__(self, dt):
        self._dt = dt

    def now(self):
        return self._dt


class _MockStrategyLoader:
    """Strategy with no_trade_witching_day enabled."""
    strategy_id = "mock-witching"
    strategy = {"meta": {"hard_gates": {"no_trade_witching_day": True}}}

    def load(self):
        pass


class _MockStrategyLoaderNoWitching:
    """Strategy with no witching gate flags."""
    strategy_id = "mock-normal"
    strategy = {"meta": {"hard_gates": {}}}

    def load(self):
        pass


def _make_engine(dt, strategy=None):
    engine = BotEngine.__new__(BotEngine)
    engine._shutdown = False
    engine.clock = _MockClock(NY_TZ.localize(dt))
    engine.logger = _CaptureLogger()
    engine.config = SimpleNamespace(
        execution_backend="ib",
        paper_mode=True,
        dry_run=True,
        min_fvg_size=4.0,
        test_connection=False,
    )
    if strategy is not None:
        engine.strategy = strategy
    return engine


class TestHolidaySkip:
    def test_good_friday_2026(self):
        engine = _make_engine(datetime(2026, 4, 3, 9, 0))
        asyncio.run(engine._startup())
        assert engine._shutdown is True
        closed = [r for r in engine.logger.records if r["event"] == "market_closed"]
        assert len(closed) == 1
        assert closed[0]["reason"] == "Good Friday"
        assert closed[0]["date"] == "20260403"

    def test_weekend_saturday(self):
        engine = _make_engine(datetime(2026, 4, 4, 9, 0))  # Saturday after Good Friday
        asyncio.run(engine._startup())
        assert engine._shutdown is True
        closed = [r for r in engine.logger.records if r["event"] == "market_closed"]
        assert closed[0]["reason"] == "weekend"

    def test_weekend_sunday(self):
        engine = _make_engine(datetime(2026, 4, 5, 9, 0))  # Sunday
        asyncio.run(engine._startup())
        assert engine._shutdown is True
        closed = [r for r in engine.logger.records if r["event"] == "market_closed"]
        assert closed[0]["reason"] == "weekend"

    def test_christmas_2026(self):
        engine = _make_engine(datetime(2026, 12, 25, 9, 0))
        asyncio.run(engine._startup())
        assert engine._shutdown is True
        closed = [r for r in engine.logger.records if r["event"] == "market_closed"]
        assert closed[0]["reason"] == "Christmas"

    def test_normal_trading_day_passes_holiday_check(self):
        """On a normal trading day, holiday check does not set _shutdown."""
        engine = _make_engine(datetime(2026, 3, 23, 9, 0), strategy=_MockStrategyLoaderNoWitching())
        try:
            asyncio.run(engine._startup())
        except (AttributeError, Exception):
            pass  # Expected — mocks incomplete for full startup
        assert engine._shutdown is False


class TestWitchingSkip:
    def test_witching_day_blocked_when_strategy_flags_it(self):
        """Sep 18, 2026 is the quarterly witching day — skip if strategy says so."""
        engine = _make_engine(datetime(2026, 9, 18, 9, 0), strategy=_MockStrategyLoader())
        try:
            asyncio.run(engine._startup())
        except (AttributeError, Exception):
            pass
        assert engine._shutdown is True
        closed = [r for r in engine.logger.records if r["event"] == "market_closed"]
        assert len(closed) == 1
        assert closed[0]["reason"] == "witching_day"

    def test_witching_day_not_blocked_when_strategy_flag_disabled(self):
        """Witching day proceeds when strategy does not enable the gate."""
        engine = _make_engine(datetime(2026, 9, 18, 9, 0), strategy=_MockStrategyLoaderNoWitching())
        try:
            asyncio.run(engine._startup())
        except (AttributeError, Exception):
            pass
        assert engine._shutdown is False
        closed = [r for r in engine.logger.records if r["event"] == "market_closed"]
        assert len(closed) == 0
