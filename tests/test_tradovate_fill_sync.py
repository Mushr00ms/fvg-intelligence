"""Tests for Tradovate broker-fill accounting."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from bot.core.engine import BotEngine
from bot.execution.broker_adapter import FillSummary
from bot.execution.tradovate.tradovate_adapter import TradovateAdapter
from bot.state.trade_state import DailyState, OrderGroup


def _run(coro):
    return asyncio.run(coro)


class _FakeRest:
    async def get_order_fills(self, order_id):
        if order_id == 100:
            return [
                {
                    "id": 1,
                    "orderId": 100,
                    "timestamp": "2026-04-10T14:30:00Z",
                    "action": "Buy",
                    "qty": 1,
                    "price": 20000.0,
                    "active": True,
                },
                {
                    "id": 2,
                    "orderId": 100,
                    "timestamp": "2026-04-10T14:30:01Z",
                    "action": "Buy",
                    "qty": 1,
                    "price": 20000.5,
                    "active": True,
                },
            ]
        return []

    async def list_fill_fees(self):
        return [
            {"id": 1, "clearingFee": 0.1, "exchangeFee": 0.2, "commission": 0.3},
            {"id": 2, "clearingFee": 0.1, "exchangeFee": 0.2, "commission": 0.3},
        ]


def test_tradovate_fill_summary_uses_weighted_price_and_actual_fees():
    config = SimpleNamespace(
        tradovate_environment="demo",
        tradovate_app_version="1.0",
        tick_size=0.25,
        point_value=20.0,
    )
    adapter = TradovateAdapter(config)
    adapter._rest = _FakeRest()

    summary = _run(adapter.get_order_fill_summary("100"))

    assert summary.quantity == 2
    assert summary.avg_price == 20000.25
    assert summary.commission == 1.2
    assert summary.fee_complete is True


class _DB:
    def __init__(self):
        self.updates = []

    def update_trade_exit(self, group_id, **kwargs):
        self.updates.append((group_id, kwargs))


class _Logger:
    def __init__(self):
        self.records = []

    def log(self, event, **kwargs):
        self.records.append((event, kwargs))


class _StateMgr:
    def __init__(self):
        self.saved = False

    def save(self, state, force=False):
        self.saved = True


class _Broker:
    def __init__(self, fills):
        self.fills = fills

    async def get_order_fill_summary(self, order_id):
        return self.fills.get(str(order_id))


def _closed_order():
    og = OrderGroup(
        group_id="g1",
        fvg_id="f1",
        setup="mit_extreme",
        side="BUY",
        entry_price=20000.0,
        stop_price=19990.0,
        target_price=20020.0,
        risk_pts=10.0,
        n_value=2.0,
        target_qty=2,
        filled_qty=2,
        state="CLOSED",
        close_reason="TP",
        realized_pnl=0.0,
        broker_entry_order_id="100",
        broker_tp_order_id="101",
    )
    return og


def test_eod_sync_corrects_closed_trade_from_broker_fills():
    engine = BotEngine.__new__(BotEngine)
    engine.config = SimpleNamespace(execution_backend="ib_data_tradovate_exec", point_value=20.0)
    engine.daily_state = DailyState(date="2026-04-10", start_balance=100000.0)
    engine.daily_state.closed_trades.append(_closed_order())
    engine.db = _DB()
    engine.logger = _Logger()
    engine.state_mgr = _StateMgr()
    engine.clock = SimpleNamespace(now=lambda: datetime(2026, 4, 10, tzinfo=timezone.utc))
    engine.broker = _Broker({
        "100": FillSummary(
            broker="tradovate",
            order_id="100",
            quantity=2,
            avg_price=20000.25,
            timestamp=datetime(2026, 4, 10, 14, 30, tzinfo=timezone.utc),
            commission=5.0,
            fee_complete=True,
        ),
        "101": FillSummary(
            broker="tradovate",
            order_id="101",
            quantity=2,
            avg_price=20020.75,
            timestamp=datetime(2026, 4, 10, 15, 0, tzinfo=timezone.utc),
            commission=5.0,
            fee_complete=True,
        ),
    })

    errors = _run(engine._sync_closed_trades_from_broker_fills())

    assert errors == []
    assert engine.daily_state.closed_trades[0].realized_pnl == 810.0
    assert engine.daily_state.realized_pnl == 810.0
    assert engine.db.updates[0][1]["actual_entry_price"] == 20000.25
    assert engine.db.updates[0][1]["actual_exit_price"] == 20020.75
    assert engine.db.updates[0][1]["commission"] == 10.0


def test_eod_sync_infers_tp_fill_for_stale_reconciled_close():
    engine = BotEngine.__new__(BotEngine)
    engine.config = SimpleNamespace(execution_backend="ib_data_tradovate_exec", point_value=20.0)
    engine.daily_state = DailyState(date="2026-04-10", start_balance=100000.0)
    og = _closed_order()
    og.close_reason = "RECONCILE_IB_CLOSED"
    og.broker_sl_order_id = "102"
    engine.daily_state.closed_trades.append(og)
    engine.db = _DB()
    engine.logger = _Logger()
    engine.state_mgr = _StateMgr()
    engine.clock = SimpleNamespace(now=lambda: datetime(2026, 4, 10, tzinfo=timezone.utc))
    engine.broker = _Broker({
        "100": FillSummary(
            broker="tradovate",
            order_id="100",
            quantity=2,
            avg_price=20000.25,
            timestamp=datetime(2026, 4, 10, 14, 30, tzinfo=timezone.utc),
            commission=5.0,
            fee_complete=True,
        ),
        "101": FillSummary(
            broker="tradovate",
            order_id="101",
            quantity=2,
            avg_price=20020.75,
            timestamp=datetime(2026, 4, 10, 15, 0, tzinfo=timezone.utc),
            commission=5.0,
            fee_complete=True,
        ),
    })

    errors = _run(engine._sync_closed_trades_from_broker_fills())

    assert errors == []
    assert og.close_reason == "TP"
    assert engine.db.updates[0][1]["exit_reason"] == "TP"
    assert engine.db.updates[0][1]["actual_exit_price"] == 20020.75


def test_eod_sync_reports_error_when_broker_fees_are_incomplete():
    engine = BotEngine.__new__(BotEngine)
    engine.config = SimpleNamespace(execution_backend="ib_data_tradovate_exec", point_value=20.0)
    engine.daily_state = DailyState(date="2026-04-10", start_balance=100000.0)
    engine.daily_state.closed_trades.append(_closed_order())
    engine.db = _DB()
    engine.logger = _Logger()
    engine.state_mgr = _StateMgr()
    engine.broker = _Broker({
        "100": FillSummary("tradovate", "100", 2, 20000.25, commission=0.0, fee_complete=False),
        "101": FillSummary("tradovate", "101", 2, 20020.75, commission=5.0, fee_complete=True),
    })

    errors = _run(engine._sync_closed_trades_from_broker_fills())

    assert errors == ["g1: incomplete broker fill fees"]
    assert engine.db.updates == []
