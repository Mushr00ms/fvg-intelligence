"""Tests for EOD reconciliation cache reuse in BotEngine."""

import asyncio
from types import SimpleNamespace

from bot.core.engine import BotEngine


class _CaptureLogger:
    def __init__(self):
        self.records = []

    def log(self, event, **kwargs):
        self.records.append({"event": event, **kwargs})


class _StubDB:
    def __init__(self):
        self.inserted = []

    def get_trades(self, date, limit=999):
        return []

    def insert_reconciliation(self, **kwargs):
        self.inserted.append(kwargs)

    def query(self, sql, params):
        return []


async def _empty_backtest(today_fmt, data_dir):
    return []


def test_eod_reconcile_reuses_cached_file_without_waiting(tmp_path, monkeypatch):
    engine = BotEngine.__new__(BotEngine)
    engine.logger = _CaptureLogger()
    engine.daily_state = SimpleNamespace(
        date="2026-04-10",
        realized_pnl=125.0,
        daily_pnl_pct=0.00125,
        start_balance=100000.0,
        filled_trade_count=1,
        kill_switch_active=False,
    )
    engine.db = _StubDB()
    engine.telegram = SimpleNamespace(enabled=False)
    engine.config = SimpleNamespace(paper_mode=False)
    engine.hfoiv_gate = None
    engine._download_data_dir = lambda: str(tmp_path)
    engine._download_today_data = _empty_backtest
    engine._run_reconciliation_backtest = _empty_backtest

    cached = tmp_path / "nq_1secs_20260410.parquet"
    cached.write_text("cached")

    async def _fail_sleep(_seconds):
        raise AssertionError("sleep should be skipped when cache exists")

    async def _fail_download(_date_str, _data_dir):
        raise AssertionError("download should be skipped when cache exists")

    monkeypatch.setattr("bot.core.engine.asyncio.sleep", _fail_sleep)
    engine._download_today_data = _fail_download

    asyncio.run(engine._eod_reconcile())

    assert any(
        r["event"] == "eod_reconcile_download_cached" and r["file"] == str(cached)
        for r in engine.logger.records
    )
    assert len(engine.db.inserted) == 1
