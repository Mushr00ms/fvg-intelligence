"""
test_tradovate.py — Unit tests for Tradovate broker adapter components.

Tests cover:
- WebSocket text-frame protocol parsing
- Contract name resolution (expiry → NQM6 mapping)
- Bar completion buffering (detect completed vs in-progress bars)
- State migration (v1.1 ib_*_order_id → v1.2 broker_*_order_id)
- Auth credential dataclass
- REST client URL construction
- Order event dispatch
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import pytz

ET = pytz.timezone("America/New_York")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONTRACT NAME RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

class TestContractNameResolution:
    """Test _expiry_to_tradovate_name() month code mapping."""

    def _resolve(self, symbol, year, month, day=15):
        from bot.execution.tradovate.tradovate_adapter import _expiry_to_tradovate_name
        dt = datetime(year, month, day, tzinfo=ET)
        return _expiry_to_tradovate_name(symbol, dt)

    def test_march_contract(self):
        assert self._resolve("NQ", 2026, 3) == "NQH6"

    def test_june_contract(self):
        assert self._resolve("NQ", 2026, 6) == "NQM6"

    def test_september_contract(self):
        assert self._resolve("NQ", 2026, 9) == "NQU6"

    def test_december_contract(self):
        assert self._resolve("NQ", 2026, 12) == "NQZ6"

    def test_es_symbol(self):
        assert self._resolve("ES", 2026, 6) == "ESM6"

    def test_mid_quarter_maps_to_current_quarter(self):
        # January → March (next quarterly)
        assert self._resolve("NQ", 2026, 1) == "NQH6"
        # February → March
        assert self._resolve("NQ", 2026, 2) == "NQH6"
        # April → June
        assert self._resolve("NQ", 2026, 4) == "NQM6"
        # May → June
        assert self._resolve("NQ", 2026, 5) == "NQM6"
        # July → September
        assert self._resolve("NQ", 2026, 7) == "NQU6"
        # October → December
        assert self._resolve("NQ", 2026, 10) == "NQZ6"
        # November → December
        assert self._resolve("NQ", 2026, 11) == "NQZ6"

    def test_year_digit_wraps(self):
        assert self._resolve("NQ", 2030, 6) == "NQM0"
        assert self._resolve("NQ", 2029, 9) == "NQU9"

    def test_different_year_digits(self):
        assert self._resolve("NQ", 2027, 3) == "NQH7"
        assert self._resolve("NQ", 2028, 12) == "NQZ8"


# ══════════════════════════════════════════════════════════════════════════════
# 2. WEBSOCKET TEXT-FRAME PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════

class TestWebSocketFrameParsing:
    """Test TradovateWebSocket._handle_text_frame() protocol parsing."""

    def _make_ws(self):
        from bot.execution.tradovate.ws_client import TradovateWebSocket
        ws = TradovateWebSocket(name="test")
        ws._is_connected = True
        return ws

    def test_heartbeat_frame_ignored(self):
        ws = self._make_ws()
        # Should not raise or dispatch anything
        ws._handle_text_frame("h")

    def test_data_frame_dispatches_items(self):
        ws = self._make_ws()
        received = []
        ws.add_listener(lambda item: True, lambda item: received.append(item))

        payload = [{"s": 200, "i": 1, "d": {"test": True}}]
        ws._handle_text_frame("a" + json.dumps(payload))

        assert len(received) == 1
        assert received[0]["d"]["test"] is True

    def test_data_frame_resolves_pending_request(self):
        ws = self._make_ws()
        loop = asyncio.new_event_loop()
        fut = loop.create_future()
        ws._pending_requests[42] = fut

        payload = [{"s": 200, "i": 42, "d": {"result": "ok"}}]
        loop.run_until_complete(asyncio.sleep(0))  # Ensure loop is running context
        ws._handle_text_frame("a" + json.dumps(payload))

        assert fut.done()
        assert fut.result()["d"]["result"] == "ok"
        loop.close()

    def test_close_frame_sets_disconnected(self):
        ws = self._make_ws()
        ws._handle_text_frame("c")
        assert ws._is_connected is False

    def test_empty_frame_ignored(self):
        ws = self._make_ws()
        ws._handle_text_frame("")

    def test_unknown_prefix_ignored(self):
        ws = self._make_ws()
        # Should not raise
        ws._handle_text_frame("x{some data}")

    def test_malformed_json_in_data_frame(self):
        ws = self._make_ws()
        # Should log warning but not raise
        ws._handle_text_frame("a{not valid json")

    def test_multiple_items_in_data_frame(self):
        ws = self._make_ws()
        received = []
        ws.add_listener(lambda item: True, lambda item: received.append(item))

        payload = [
            {"s": 200, "i": 1},
            {"s": 200, "i": 2},
            {"s": 200, "i": 3},
        ]
        ws._handle_text_frame("a" + json.dumps(payload))

        assert len(received) == 3

    def test_listener_filter_function(self):
        ws = self._make_ws()
        received = []
        # Only accept items with "e": "md"
        ws.add_listener(
            lambda item: item.get("e") == "md",
            lambda item: received.append(item),
        )

        payload = [
            {"e": "md", "d": {"quotes": []}},
            {"e": "props", "d": {"positions": []}},
            {"e": "md", "d": {"charts": []}},
        ]
        ws._handle_text_frame("a" + json.dumps(payload))

        assert len(received) == 2
        assert all(item["e"] == "md" for item in received)

    def test_unsubscribe_removes_listener(self):
        ws = self._make_ws()
        received = []
        unsub = ws.add_listener(lambda item: True, lambda item: received.append(item))

        ws._handle_text_frame("a" + json.dumps([{"test": 1}]))
        assert len(received) == 1

        unsub()  # Remove listener

        ws._handle_text_frame("a" + json.dumps([{"test": 2}]))
        assert len(received) == 1  # No new items


# ══════════════════════════════════════════════════════════════════════════════
# 3. BAR COMPLETION BUFFERING
# ══════════════════════════════════════════════════════════════════════════════

class TestBarCompletionBuffering:
    """Test that the adapter fires on_bar(completed=True) only on new timestamps."""

    def _make_adapter_and_sub(self):
        from bot.execution.tradovate.tradovate_adapter import TradovateAdapter, _BarSubscription
        from bot.execution.broker_adapter import ContractInfo, BarData

        received = []

        def on_bar(bar, has_new_bar):
            received.append({"bar": bar, "has_new_bar": has_new_bar})

        sub = _BarSubscription(
            sub_id="test_bars_1",
            contract=ContractInfo(
                symbol="NQ", broker_contract_id="123",
                expiry="20260619", exchange="CME",
            ),
            element_size=5,
            on_bar=on_bar,
        )

        config = MagicMock()
        config.tradovate_environment = "demo"
        config.tradovate_app_version = "1.0"
        config.tick_size = 0.25
        config.point_value = 20.0

        adapter = TradovateAdapter(config)
        return adapter, sub, received

    def _make_chart_item(self, timestamp_str, o, h, l, c, up_vol=100, down_vol=50):
        return {
            "d": {
                "charts": [{
                    "id": None,
                    "bars": [{
                        "timestamp": timestamp_str,
                        "open": o, "high": h, "low": l, "close": c,
                        "upVolume": up_vol, "downVolume": down_vol,
                    }],
                }],
            },
        }

    def test_first_bar_not_completed(self):
        adapter, sub, received = self._make_adapter_and_sub()

        item = self._make_chart_item("2026-04-10T10:30:00.000Z", 20000, 20010, 19990, 20005)
        adapter._handle_chart_data(sub, item)

        assert len(received) == 1
        assert received[0]["has_new_bar"] is False  # First bar = seed, not complete

    def test_same_timestamp_updates_not_completed(self):
        adapter, sub, received = self._make_adapter_and_sub()

        ts = "2026-04-10T10:30:00.000Z"
        adapter._handle_chart_data(sub, self._make_chart_item(ts, 20000, 20010, 19990, 20005))
        adapter._handle_chart_data(sub, self._make_chart_item(ts, 20000, 20015, 19990, 20012))
        adapter._handle_chart_data(sub, self._make_chart_item(ts, 20000, 20015, 19985, 20008))

        assert len(received) == 3
        assert all(r["has_new_bar"] is False for r in received)

    def test_new_timestamp_signals_completed(self):
        adapter, sub, received = self._make_adapter_and_sub()

        ts1 = "2026-04-10T10:30:00.000Z"
        ts2 = "2026-04-10T10:35:00.000Z"

        adapter._handle_chart_data(sub, self._make_chart_item(ts1, 20000, 20010, 19990, 20005))
        adapter._handle_chart_data(sub, self._make_chart_item(ts2, 20005, 20020, 20000, 20015))

        assert len(received) == 2
        assert received[0]["has_new_bar"] is False   # First bar
        assert received[1]["has_new_bar"] is True    # New timestamp = prior bar completed

    def test_multiple_bars_sequence(self):
        adapter, sub, received = self._make_adapter_and_sub()

        timestamps = [
            "2026-04-10T10:30:00.000Z",
            "2026-04-10T10:30:00.000Z",  # Update to bar 1
            "2026-04-10T10:35:00.000Z",  # Bar 2 starts → bar 1 complete
            "2026-04-10T10:35:00.000Z",  # Update to bar 2
            "2026-04-10T10:40:00.000Z",  # Bar 3 starts → bar 2 complete
        ]

        for i, ts in enumerate(timestamps):
            adapter._handle_chart_data(sub,
                self._make_chart_item(ts, 20000 + i, 20010 + i, 19990 + i, 20005 + i))

        assert len(received) == 5
        completions = [r["has_new_bar"] for r in received]
        assert completions == [False, False, True, False, True]

    def test_bar_data_has_correct_fields(self):
        adapter, sub, received = self._make_adapter_and_sub()

        item = self._make_chart_item("2026-04-10T14:30:00.000Z", 20100, 20150, 20090, 20130, 500, 200)
        adapter._handle_chart_data(sub, item)

        bar = received[0]["bar"]
        assert bar.open == 20100
        assert bar.high == 20150
        assert bar.low == 20090
        assert bar.close == 20130
        assert bar.volume == 700  # upVolume + downVolume
        assert bar.timestamp.tzinfo is not None  # Timezone-aware


# ══════════════════════════════════════════════════════════════════════════════
# 4. STATE MIGRATION (v1.1 → v1.2)
# ══════════════════════════════════════════════════════════════════════════════

class TestStateMigration:
    """Test order ID field migration from IB-specific to broker-agnostic."""

    def test_v11_ib_order_ids_migrated_to_broker_str(self):
        from bot.state.state_manager import StateManager
        import tempfile

        sm = StateManager(tempfile.mkdtemp())

        v11_data = {
            "version": "1.1",
            "date": "2026-04-10",
            "start_balance": 100000.0,
            "pending_orders": [{
                "group_id": "abc123", "fvg_id": "fvg1",
                "setup": "mit_extreme", "side": "BUY",
                "entry_price": 24000, "stop_price": 23988,
                "target_price": 24024, "risk_pts": 12,
                "n_value": 2.0, "target_qty": 3,
                "ib_entry_order_id": 100,
                "ib_tp_order_id": 101,
                "ib_sl_order_id": 102,
            }],
            "open_positions": [],
            "closed_trades": [],
            "suspended_orders": [],
        }

        migrated = sm._migrate_state(v11_data, "1.1")

        assert migrated["version"] == "1.2"
        og = migrated["pending_orders"][0]
        assert og["broker_entry_order_id"] == "100"
        assert og["broker_tp_order_id"] == "101"
        assert og["broker_sl_order_id"] == "102"
        # Old keys removed
        assert "ib_entry_order_id" not in og
        assert "ib_tp_order_id" not in og
        assert "ib_sl_order_id" not in og

    def test_v11_null_order_ids_stay_null(self):
        from bot.state.state_manager import StateManager
        import tempfile

        sm = StateManager(tempfile.mkdtemp())

        v11_data = {
            "version": "1.1",
            "date": "2026-04-10",
            "start_balance": 100000.0,
            "pending_orders": [{
                "group_id": "abc456", "fvg_id": "fvg2",
                "setup": "mit_extreme", "side": "SELL",
                "entry_price": 20000, "stop_price": 20012,
                "target_price": 19976, "risk_pts": 12,
                "n_value": 2.0, "target_qty": 1,
                "ib_entry_order_id": None,
                "ib_tp_order_id": None,
                "ib_sl_order_id": None,
            }],
            "open_positions": [],
            "closed_trades": [],
            "suspended_orders": [],
        }

        migrated = sm._migrate_state(v11_data, "1.1")
        og = migrated["pending_orders"][0]
        assert og["broker_entry_order_id"] is None
        assert og["broker_tp_order_id"] is None
        assert og["broker_sl_order_id"] is None

    def test_v10_migrates_all_the_way_to_v12(self):
        from bot.state.state_manager import StateManager
        import tempfile

        sm = StateManager(tempfile.mkdtemp())

        v10_data = {
            "version": "1.0",
            "date": "2026-04-10",
            "start_balance": 100000.0,
            "pending_orders": [{
                "group_id": "x", "fvg_id": "f",
                "setup": "mit_extreme", "side": "BUY",
                "entry_price": 24000, "stop_price": 23988,
                "target_price": 24024, "risk_pts": 12,
                "n_value": 2.0, "target_qty": 1,
                "ib_entry_order_id": 50,
                "ib_tp_order_id": 51,
                "ib_sl_order_id": 52,
            }],
            "open_positions": [],
            "closed_trades": [],
        }

        migrated = sm._migrate_state(v10_data, "1.0")

        assert migrated["version"] == "1.2"
        assert "suspended_orders" in migrated
        og = migrated["pending_orders"][0]
        assert og["broker_entry_order_id"] == "50"
        assert og.get("suspended_at") is None

    def test_order_group_from_dict_backward_compat(self):
        """OrderGroup.from_dict reads old ib_*_order_id keys if broker_* absent."""
        from bot.state.trade_state import OrderGroup

        old_dict = {
            "group_id": "g1", "fvg_id": "f1", "setup": "mit_extreme",
            "side": "BUY", "entry_price": 20000, "stop_price": 19990,
            "target_price": 20030, "risk_pts": 10, "n_value": 3.0, "target_qty": 2,
            "ib_entry_order_id": 200,
            "ib_tp_order_id": 201,
            "ib_sl_order_id": 202,
        }

        og = OrderGroup.from_dict(old_dict)
        assert og.broker_entry_order_id == "200"
        assert og.broker_tp_order_id == "201"
        assert og.broker_sl_order_id == "202"

    def test_order_group_from_dict_new_keys_preferred(self):
        """broker_* keys take precedence over ib_* if both present."""
        from bot.state.trade_state import OrderGroup

        mixed_dict = {
            "group_id": "g2", "fvg_id": "f2", "setup": "mid_extreme",
            "side": "SELL", "entry_price": 20000, "stop_price": 20010,
            "target_price": 19970, "risk_pts": 10, "n_value": 3.0, "target_qty": 1,
            "broker_entry_order_id": "tv_1000",
            "broker_tp_order_id": "tv_1001",
            "broker_sl_order_id": "tv_1002",
            "ib_entry_order_id": 300,  # Should be ignored
        }

        og = OrderGroup.from_dict(mixed_dict)
        assert og.broker_entry_order_id == "tv_1000"
        assert og.broker_tp_order_id == "tv_1001"
        assert og.broker_sl_order_id == "tv_1002"


# ══════════════════════════════════════════════════════════════════════════════
# 5. AUTH CREDENTIALS
# ══════════════════════════════════════════════════════════════════════════════

class TestTradovateCredentials:
    """Test TradovateCredentials and environment URL mapping."""

    def test_demo_base_url(self):
        from bot.execution.tradovate.auth import TradovateCredentials
        creds = TradovateCredentials(
            username="user", password="pass", app_id="app",
            app_version="1.0", cid=123, sec="secret", device_id="dev",
            environment="demo",
        )
        assert creds.base_url == "https://demo.tradovateapi.com/v1"

    def test_live_base_url(self):
        from bot.execution.tradovate.auth import TradovateCredentials
        creds = TradovateCredentials(
            username="user", password="pass", app_id="app",
            app_version="1.0", cid=123, sec="secret", device_id="dev",
            environment="live",
        )
        assert creds.base_url == "https://live.tradovateapi.com/v1"

    def test_token_info_expiry(self):
        import time
        from bot.execution.tradovate.auth import TokenInfo

        # Token that expires in 1 hour
        token = TokenInfo(
            access_token="abc123",
            expiration_time=time.time() + 3600,
        )
        assert not token.is_expired
        assert token.seconds_until_expiry > 3500

        # Token that already expired
        expired = TokenInfo(
            access_token="old",
            expiration_time=time.time() - 60,
        )
        assert expired.is_expired
        assert expired.seconds_until_expiry == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 6. ORDER EVENT DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

class TestOrderEventDispatch:
    """Test TradovateAdapter._dispatch_order_event routing."""

    def _make_adapter(self):
        from bot.execution.tradovate.tradovate_adapter import TradovateAdapter
        config = MagicMock()
        config.tradovate_environment = "demo"
        config.tradovate_app_version = "1.0"
        config.tick_size = 0.25
        config.point_value = 20.0
        adapter = TradovateAdapter(config)
        # Reset class-level dict (shared across instances)
        adapter._order_callbacks = {}
        return adapter

    def test_entry_fill_dispatched(self):
        adapter = self._make_adapter()
        fills = {"entry": [], "tp": [], "sl": [], "status": []}

        adapter._order_callbacks["strat_1"] = {
            "on_entry_fill": lambda o: fills["entry"].append(o),
            "on_tp_fill": lambda o: fills["tp"].append(o),
            "on_sl_fill": lambda o: fills["sl"].append(o),
            "on_status_change": lambda o: fills["status"].append(o),
            "_entry_id": 100,
            "_tp_id": 101,
            "_sl_id": 102,
        }

        adapter._dispatch_order_event({"id": 100, "ordStatus": "Filled"})

        assert len(fills["entry"]) == 1
        assert len(fills["tp"]) == 0
        assert len(fills["sl"]) == 0
        assert len(fills["status"]) == 1

    def test_tp_fill_dispatched(self):
        adapter = self._make_adapter()
        fills = {"entry": [], "tp": [], "sl": []}

        adapter._order_callbacks["strat_1"] = {
            "on_entry_fill": lambda o: fills["entry"].append(o),
            "on_tp_fill": lambda o: fills["tp"].append(o),
            "on_sl_fill": lambda o: fills["sl"].append(o),
            "on_status_change": lambda o: None,
            "_entry_id": 100, "_tp_id": 101, "_sl_id": 102,
        }

        adapter._dispatch_order_event({"id": 101, "ordStatus": "Filled"})

        assert len(fills["tp"]) == 1
        assert len(fills["entry"]) == 0
        assert len(fills["sl"]) == 0

    def test_unrelated_order_not_dispatched(self):
        adapter = self._make_adapter()
        fills = {"entry": [], "tp": [], "sl": [], "status": []}

        adapter._order_callbacks["strat_1"] = {
            "on_entry_fill": lambda o: fills["entry"].append(o),
            "on_tp_fill": lambda o: fills["tp"].append(o),
            "on_sl_fill": lambda o: fills["sl"].append(o),
            "on_status_change": lambda o: fills["status"].append(o),
            "_entry_id": 100, "_tp_id": 101, "_sl_id": 102,
        }

        adapter._dispatch_order_event({"id": 999, "ordStatus": "Filled"})

        assert len(fills["entry"]) == 0
        assert len(fills["tp"]) == 0
        assert len(fills["sl"]) == 0
        assert len(fills["status"]) == 0

    def test_empty_status_ignored(self):
        adapter = self._make_adapter()
        adapter._order_callbacks["strat_1"] = {
            "on_entry_fill": lambda o: None,
            "on_tp_fill": lambda o: None,
            "on_sl_fill": lambda o: None,
            "on_status_change": lambda o: None,
            "_entry_id": 100, "_tp_id": 101, "_sl_id": 102,
        }

        # Should not raise
        adapter._dispatch_order_event({"id": 100, "ordStatus": ""})
        adapter._dispatch_order_event({"id": 100})


# ══════════════════════════════════════════════════════════════════════════════
# 7. SECRETS STORE
# ══════════════════════════════════════════════════════════════════════════════

class TestSecretStore:
    """Test SecretStore SSM path construction and env var fallback."""

    def test_param_path_construction(self):
        from bot.secret_store import SecretStore
        store = SecretStore(environment="demo", ssm_prefix="/fvg-bot")
        assert store._param_path("tradovate", "username") == "/fvg-bot/demo/tradovate/username"

    def test_param_path_live(self):
        from bot.secret_store import SecretStore
        store = SecretStore(environment="live", ssm_prefix="/fvg-bot")
        assert store._param_path("tradovate", "cid") == "/fvg-bot/live/tradovate/cid"

    def test_env_var_fallback(self):
        """When SSM is unavailable, env vars provide credentials."""
        from bot.secret_store import SecretStore
        import os

        store = SecretStore(environment="demo")
        # Mock SSM failure by making _get_all_parameters return empty
        store._get_all_parameters = lambda service: {}

        env = {
            "TRADOVATE_USERNAME": "testuser",
            "TRADOVATE_PASSWORD": "testpass",
            "TRADOVATE_CID": "99",
            "TRADOVATE_SEC": "mysecret",
            "TRADOVATE_APP_ID": "myapp",
            "TRADOVATE_DEVICE_ID": "dev-123",
        }
        with patch.dict(os.environ, env):
            secrets = store.load_tradovate()

        assert secrets.username == "testuser"
        assert secrets.password == "testpass"
        assert secrets.cid == 99
        assert secrets.sec == "mysecret"
        assert secrets.app_id == "myapp"
        assert secrets.device_id == "dev-123"

    def test_missing_credentials_raises(self):
        from bot.secret_store import SecretStore, SecretLoadError
        import os

        store = SecretStore(environment="demo")
        store._get_all_parameters = lambda service: {}

        # No env vars set
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SecretLoadError, match="Tradovate credentials not found"):
                store.load_tradovate()


# ══════════════════════════════════════════════════════════════════════════════
# 8. BROKER ADAPTER ABC COMPLIANCE
# ══════════════════════════════════════════════════════════════════════════════

class TestBrokerAdapterABC:
    """Verify both adapters implement the full ABC without errors."""

    def test_tradovate_adapter_instantiates(self):
        from bot.execution.tradovate.tradovate_adapter import TradovateAdapter
        config = MagicMock()
        config.tradovate_environment = "demo"
        config.tradovate_app_version = "1.0"
        config.tick_size = 0.25
        config.point_value = 20.0
        adapter = TradovateAdapter(config)
        assert adapter is not None

    def test_ib_adapter_instantiates(self):
        from bot.execution.ib_adapter import IBAdapter
        config = MagicMock()
        config.ib_host = "127.0.0.1"
        config.ib_port = 7497
        config.ib_client_id = 1
        config.tick_size = 0.25
        config.point_value = 20.0
        config.currency = "USD"
        adapter = IBAdapter(config)
        assert adapter is not None

    def test_broker_adapter_is_abstract(self):
        from bot.execution.broker_adapter import BrokerAdapter
        with pytest.raises(TypeError):
            BrokerAdapter()
