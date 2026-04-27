import asyncio
from datetime import datetime
import pytz

from bot.core.engine import BotEngine


class _Clock:
    def __init__(self):
        self.current = datetime(2026, 4, 27, 8, 20, tzinfo=pytz.timezone("America/New_York"))

    def now(self):
        return self.current


class _Logger:
    def __init__(self):
        self.records = []

    def log(self, event, **kwargs):
        self.records.append((event, kwargs))


def _make_engine():
    engine = BotEngine.__new__(BotEngine)
    engine.clock = _Clock()
    engine.logger = _Logger()
    engine.broker = None
    engine.order_mgr = None
    engine.daily_state = None
    engine._disconnect_flatten_done = True
    engine._last_exec_reconnect_log_ts = 0.0
    engine._suppressed_exec_reconnects = 0
    return engine


def test_exec_reconnect_log_is_throttled():
    engine = _make_engine()

    asyncio.run(engine._on_exec_reconnect())
    asyncio.run(engine._on_exec_reconnect())

    logs = [r for r in engine.logger.records if r[0] == "exec_reconnect"]
    assert len(logs) == 1
    assert logs[0][1]["source"] == "Tradovate order/account websocket"
    assert engine._suppressed_exec_reconnects == 1


def test_exec_reconnect_reports_suppressed_count_after_window():
    engine = _make_engine()

    asyncio.run(engine._on_exec_reconnect())
    asyncio.run(engine._on_exec_reconnect())
    engine.clock.current = datetime(2026, 4, 27, 8, 26, tzinfo=pytz.timezone("America/New_York"))
    asyncio.run(engine._on_exec_reconnect())

    logs = [r for r in engine.logger.records if r[0] == "exec_reconnect"]
    assert len(logs) == 2
    assert logs[1][1]["suppressed"] == 1
