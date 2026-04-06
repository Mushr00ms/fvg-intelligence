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


class _MockStrategyLoaderMacroSkip:
    """Strategy with macro event skip enabled (default)."""
    strategy_id = "mock-macro-skip"
    strategy = {"meta": {"hard_gates": {"skip_nfp": True, "skip_cpi": True, "skip_fomc": True}}}

    def load(self):
        pass


class _MockStrategyLoaderMacroAllow:
    """Strategy with macro event skip explicitly disabled."""
    strategy_id = "mock-macro-allow"
    strategy = {"meta": {"hard_gates": {"skip_nfp": False, "skip_cpi": False, "skip_fomc": False}}}

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


class TestMacroEventSkip:
    """Macro gate is per-entry (time-based), not per-day. Engine starts normally
    on macro days; individual entries in blackout windows are rejected."""

    def test_nfp_day_does_not_block_startup(self):
        """2025-01-03 is NFP — engine starts up normally (gate is per-entry)."""
        engine = _make_engine(datetime(2025, 1, 3, 9, 0), strategy=_MockStrategyLoaderMacroSkip())
        try:
            asyncio.run(engine._startup())
        except (AttributeError, Exception):
            pass
        assert engine._shutdown is False

    def test_cpi_day_does_not_block_startup(self):
        """2025-01-15 is CPI — engine starts up normally."""
        engine = _make_engine(datetime(2025, 1, 15, 9, 0), strategy=_MockStrategyLoaderMacroSkip())
        try:
            asyncio.run(engine._startup())
        except (AttributeError, Exception):
            pass
        assert engine._shutdown is False

    def test_fomc_day_does_not_block_startup(self):
        """2025-01-29 is FOMC — engine starts up normally."""
        engine = _make_engine(datetime(2025, 1, 29, 9, 0), strategy=_MockStrategyLoaderMacroSkip())
        try:
            asyncio.run(engine._startup())
        except (AttributeError, Exception):
            pass
        assert engine._shutdown is False

    def test_macro_gate_blocks_in_blackout_window(self):
        """NFP day at 09:45 ET — _macro_gate_allows_entry returns blocked."""
        engine = _make_engine(datetime(2025, 1, 3, 9, 45), strategy=_MockStrategyLoaderMacroSkip())
        allowed, reason = engine._macro_gate_allows_entry()
        assert allowed is False
        assert reason == "macro_nfp_blackout"

    def test_macro_gate_allows_outside_blackout(self):
        """NFP day at 11:00 ET — outside blackout, entry allowed."""
        engine = _make_engine(datetime(2025, 1, 3, 11, 0), strategy=_MockStrategyLoaderMacroSkip())
        allowed, reason = engine._macro_gate_allows_entry()
        assert allowed is True

    def test_macro_gate_allows_when_disabled(self):
        """NFP day at 09:45 ET — but skip_nfp=False, entry allowed."""
        engine = _make_engine(datetime(2025, 1, 3, 9, 45), strategy=_MockStrategyLoaderMacroAllow())
        allowed, reason = engine._macro_gate_allows_entry()
        assert allowed is True

    def test_normal_day_always_allowed(self):
        """Regular day — macro gate never blocks."""
        engine = _make_engine(datetime(2025, 1, 6, 9, 45), strategy=_MockStrategyLoaderMacroSkip())
        allowed, reason = engine._macro_gate_allows_entry()
        assert allowed is True
