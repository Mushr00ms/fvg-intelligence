import asyncio
from types import SimpleNamespace

from bot.execution.execution_types import AccountSnapshot, OpenOrderSnapshot, PositionSnapshot, SymbolRules
from crypto_bot.config import CryptoBotConfig
from crypto_bot.engine import CryptoBotEngine
from crypto_bot.fvg import detect_fvg_3bars, parse_ts
from crypto_bot.models import OrderIntent, RuntimeState, new_runtime_state
from crypto_bot.reconciliation import reconcile_runtime_state
from crypto_bot.risk import CryptoRiskManager


class DummyStrategy:
    def find_cell(self, time_period, risk_bps, setup):
        return {"best_n": 1.5}


class FakeClient:
    def __init__(self, *, open_orders=None, open_algo_orders=None, positions=None, orders=None):
        self._open_orders = open_orders or []
        self._open_algo_orders = open_algo_orders or []
        self._positions = positions or []
        self._orders = orders or {}

    async def get_open_orders(self, symbol):
        return list(self._open_orders)

    async def get_open_algo_orders(self, symbol, algo_type=None):
        return list(self._open_algo_orders)

    async def get_positions(self, symbol):
        return list(self._positions)

    async def get_order(self, symbol, *, order_id=None, client_order_id=None):
        return self._orders[client_order_id]


def _config(**overrides):
    base = {
        "symbol": "BTCUSDT",
        "risk_per_trade": 0.01,
        "maker_fee": 0.0,
        "tp_fee": 0.0,
        "stop_fee": 0.0004,
        "leverage": 10,
        "max_daily_loss_pct": 0.10,
        "max_concurrent": 4,
        "max_cumulative_risk_pct": 0.05,
        "max_margin_usage_pct": 0.70,
        "min_liquidation_buffer_pct": 0.05,
        "position_mode": "HEDGE",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _rules():
    return SymbolRules(
        symbol="BTCUSDT",
        price_tick_size=0.1,
        quantity_step_size=0.001,
        min_quantity=0.001,
        min_notional=100.0,
        price_precision=1,
        quantity_precision=3,
    )


def test_order_intent_from_dict_backfills_position_side():
    payload = {
        "group_id": "abc123",
        "fvg_id": "fvg1",
        "symbol": "BTCUSDT",
        "setup": "mit_extreme",
        "side": "BUY",
        "entry_price": 60000.0,
        "stop_price": 59800.0,
        "target_price": 60300.0,
        "risk_bps": 30.0,
        "n_value": 1.5,
        "risk_dollar": 100.0,
        "quantity": 0.1,
        "notional": 6000.0,
        "initial_margin": 600.0,
        "expected_loss": 100.0,
        "expected_profit": 150.0,
        "created_at": "2026-04-01T00:00:00+00:00",
    }

    intent = OrderIntent.from_dict(payload)

    assert intent.position_side == "BOTH"


def test_risk_manager_rejects_opposite_side_in_one_way_mode():
    manager = CryptoRiskManager(_config(position_mode="ONE_WAY"), _rules(), DummyStrategy())
    state = new_runtime_state("BTCUSDT", "strategy", "2026-04-01", 10_000.0)
    state.open_positions.append(
        OrderIntent(
            group_id="g1",
            fvg_id="f1",
            symbol="BTCUSDT",
            setup="mit_extreme",
            side="BUY",
            position_side="BOTH",
            entry_price=60000.0,
            stop_price=59800.0,
            target_price=60300.0,
            risk_bps=30.0,
            n_value=1.5,
            risk_dollar=100.0,
            quantity=0.1,
            notional=6000.0,
            initial_margin=600.0,
            expected_loss=100.0,
            expected_profit=150.0,
            created_at="2026-04-01T00:00:00+00:00",
            status="OPEN",
            filled_qty=0.1,
            avg_entry_price=60000.0,
        )
    )
    new_intent = OrderIntent(
        group_id="g2",
        fvg_id="f2",
        symbol="BTCUSDT",
        setup="mit_extreme",
        side="SELL",
        position_side="BOTH",
        entry_price=60100.0,
        stop_price=60300.0,
        target_price=59800.0,
        risk_bps=30.0,
        n_value=1.5,
        risk_dollar=100.0,
        quantity=0.1,
        notional=6010.0,
        initial_margin=601.0,
        expected_loss=100.0,
        expected_profit=150.0,
        created_at="2026-04-01T00:01:00+00:00",
    )

    reason = manager.can_accept(state, new_intent, available_balance=10_000.0)

    assert "ONE_WAY mode blocks opposite-side exposure" in reason


def test_risk_manager_rejects_margin_usage_breach():
    manager = CryptoRiskManager(_config(max_cumulative_risk_pct=0.25), _rules(), DummyStrategy())
    state = new_runtime_state("BTCUSDT", "strategy", "2026-04-01", 10_000.0)
    intent = OrderIntent(
        group_id="g2",
        fvg_id="f2",
        symbol="BTCUSDT",
        setup="mit_extreme",
        side="BUY",
        position_side="LONG",
        entry_price=60000.0,
        stop_price=59800.0,
        target_price=60300.0,
        risk_bps=30.0,
        n_value=1.5,
        risk_dollar=100.0,
        quantity=0.1,
        notional=6000.0,
        initial_margin=700.0,
        expected_loss=100.0,
        expected_profit=150.0,
        created_at="2026-04-01T00:01:00+00:00",
    )
    snapshot = AccountSnapshot(
        broker="binance_um",
        wallet_balance=1000.0,
        available_balance=900.0,
        margin_balance=1000.0,
        initial_margin=100.0,
    )

    reason = manager.can_accept(
        state,
        intent,
        available_balance=900.0,
        account_snapshot=snapshot,
        exchange_positions=[],
    )

    assert "margin usage" in reason


def test_risk_manager_detects_consecutive_far_conflict():
    manager = CryptoRiskManager(_config(), _rules(), DummyStrategy())
    existing = OrderIntent(
        group_id="near1",
        fvg_id="f1",
        symbol="BTCUSDT",
        setup="mit_extreme",
        side="BUY",
        position_side="LONG",
        entry_price=60100.0,
        stop_price=60000.0,
        target_price=60250.0,
        risk_bps=20.0,
        n_value=1.5,
        risk_dollar=100.0,
        quantity=0.1,
        notional=6010.0,
        initial_margin=601.0,
        expected_loss=100.0,
        expected_profit=150.0,
        created_at="2026-04-01T00:00:00+00:00",
        status="SUBMITTED",
    )
    new_far = OrderIntent(
        group_id="far1",
        fvg_id="f2",
        symbol="BTCUSDT",
        setup="mit_extreme",
        side="BUY",
        position_side="LONG",
        entry_price=60000.0,
        stop_price=59850.0,
        target_price=60225.0,
        risk_bps=25.0,
        n_value=1.5,
        risk_dollar=100.0,
        quantity=0.1,
        notional=6000.0,
        initial_margin=600.0,
        expected_loss=100.0,
        expected_profit=150.0,
        created_at="2026-04-01T00:01:00+00:00",
    )

    reason = manager.consecutive_conflict_reason([existing], new_far)

    assert "consecutive_fvg_far_skip" in reason


def test_reconciliation_resumes_managed_group_and_flags_exit_repair():
    state = new_runtime_state("BTCUSDT", "strategy", "2026-04-01", 10_000.0)
    tp_order = OpenOrderSnapshot(
        broker="binance_um",
        symbol="BTCUSDT",
        side="SELL",
        order_type="LIMIT",
        status="NEW",
        quantity=0.1,
        price=60300.0,
        order_id="42",
        client_order_id="cb_abc123_tp",
        position_side="LONG",
        reduce_only=True,
    )
    position = PositionSnapshot(
        broker="binance_um",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        entry_price=60000.0,
        mark_price=60100.0,
        leverage=10,
        margin_type="CROSSED",
        position_side="LONG",
    )
    client = FakeClient(
        open_orders=[tp_order],
        positions=[position],
        orders={
            "cb_abc123_entry": {
                "orderId": "11",
                "clientOrderId": "cb_abc123_entry",
                "side": "BUY",
                "positionSide": "LONG",
                "origQty": "0.1",
                "executedQty": "0.1",
                "avgPrice": "60000",
                "price": "60000",
                "time": 1711929600000,
            }
        },
    )

    result = asyncio.run(
        reconcile_runtime_state(
            state,
            client=client,
            symbol="BTCUSDT",
            leverage=10,
            resume_managed_positions=True,
        )
    )

    assert result.resumed_groups == ["abc123"]
    assert result.exit_repairs == ["abc123"]
    assert len(state.open_positions) == 1
    assert state.open_positions[0].group_id == "abc123"
    assert state.open_positions[0].position_side == "LONG"


def test_reconciliation_marks_unknown_bot_orders_unmanaged_when_resume_disabled():
    state = RuntimeState(
        version="1.0",
        symbol="BTCUSDT",
        strategy_id="strategy",
        day="2026-04-01",
        start_balance=10_000.0,
        current_balance=10_000.0,
    )
    entry_order = OpenOrderSnapshot(
        broker="binance_um",
        symbol="BTCUSDT",
        side="BUY",
        order_type="LIMIT",
        status="NEW",
        quantity=0.1,
        price=60000.0,
        order_id="42",
        client_order_id="cb_abc123_entry",
        position_side="LONG",
    )
    client = FakeClient(open_orders=[entry_order], positions=[])

    result = asyncio.run(
        reconcile_runtime_state(
            state,
            client=client,
            symbol="BTCUSDT",
            leverage=10,
            resume_managed_positions=False,
        )
    )

    assert result.unmanaged_order_ids == ["cb_abc123_entry"]


def test_seed_recent_history_does_not_emit_historical_trades(tmp_path):
    config = CryptoBotConfig(
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
    )
    engine = CryptoBotEngine(config)
    engine.state = new_runtime_state("BTCUSDT", "strategy", "2026-04-01", 10_000.0)
    engine.clock.sync = lambda: True

    class FakeMarketData:
        async def fetch_klines(self, symbol, interval, limit=500):
            if interval == "5m":
                return [
                    {
                        "interval": "5m",
                        "open_time": "2026-04-01T13:45:00-04:00",
                        "close_time": "2026-04-01T13:49:59.999000-04:00",
                    },
                    {
                        "interval": "5m",
                        "open_time": "2026-04-01T13:50:00-04:00",
                        "close_time": "2026-04-01T13:54:59.999000-04:00",
                    },
                    {
                        "interval": "5m",
                        "open_time": "2026-04-01T13:55:00-04:00",
                        "close_time": "2026-04-01T13:59:59.999000-04:00",
                    },
                    {
                        "interval": "5m",
                        "open_time": "2026-04-01T14:00:00-04:00",
                        "close_time": "2026-04-01T14:04:59.999000-04:00",
                    },
                ]
            return [
                {
                    "interval": "1m",
                    "open_time": "2026-04-01T13:58:00-04:00",
                    "close_time": "2026-04-01T13:58:59.999000-04:00",
                },
                {
                    "interval": "1m",
                    "open_time": "2026-04-01T13:59:00-04:00",
                    "close_time": "2026-04-01T13:59:59.999000-04:00",
                },
                {
                    "interval": "1m",
                    "open_time": "2026-04-01T14:00:00-04:00",
                    "close_time": "2026-04-01T14:00:59.999000-04:00",
                },
            ]

    class FakeFVGManager:
        def __init__(self):
            self.active = []
            self.seeded_bars = None
            self.scanned_bars = []

        def reset(self):
            self.active = []

        def seed_5m(self, bars):
            self.seeded_bars = list(bars)
            self.active = []

        def scan_1m_close(self, bar):
            self.scanned_bars.append(bar)
            return ["would-have-triggered"]

    engine.market_data = FakeMarketData()
    engine.fvg_mgr = FakeFVGManager()
    engine.clock.now_utc = lambda: parse_ts("2026-04-01T13:59:30-04:00")

    asyncio.run(engine._seed_recent_history())

    assert len(engine.fvg_mgr.seeded_bars) == 2
    assert len(engine.fvg_mgr.scanned_bars) == 1
    assert engine.state.closed_trades == []
    assert engine.state.pending_entries == []
    assert engine.state.open_positions == []


def test_crypto_fvg_detection_matches_nq_candle_semantics():
    bar1 = {
        "open_time": "2026-04-01T13:40:00-04:00",
        "close_time": "2026-04-01T13:44:59.999000-04:00",
        "open": 68320.0,
        "high": 68337.7,
        "low": 68326.0,
        "close": 68330.0,
    }
    bar2 = {
        "open_time": "2026-04-01T13:45:00-04:00",
        "close_time": "2026-04-01T13:49:59.999000-04:00",
        "open": 68310.0,
        "high": 68337.7,
        "low": 68254.8,
        "close": 68280.0,
    }
    bar3 = {
        "open_time": "2026-04-01T13:50:00-04:00",
        "close_time": "2026-04-01T13:54:59.999000-04:00",
        "open": 68480.0,
        "high": 68520.0,
        "low": 68470.0,
        "close": 68500.0,
    }

    fvg = detect_fvg_3bars(bar1, bar2, bar3)

    assert fvg is not None
    assert fvg.time_candle1 == "2026-04-01T13:40:00-04:00"
    assert fvg.time_candle2 == "2026-04-01T13:45:00-04:00"
    assert fvg.time_candle3 == "2026-04-01T13:50:00-04:00"
    assert fvg.formation_time == "2026-04-01T13:54:59.999000-04:00"
