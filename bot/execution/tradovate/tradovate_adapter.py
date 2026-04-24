"""
tradovate_adapter.py — BrokerAdapter implementation for Tradovate.

Cloud-native REST + WebSocket broker. No local bridge needed.
Uses two WebSocket connections:
  1. Order/account WS (demo or live URL) — order events, position sync
  2. Market data WS — bar subscriptions, quote subscriptions
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import pytz

from bot.execution.broker_adapter import (
    BarData,
    BracketOrderResult,
    BrokerAdapter,
    ContractInfo,
    TickData,
)
from bot.execution.execution_types import (
    AccountSnapshot,
    OpenOrderSnapshot,
    PositionSnapshot,
)
from bot.execution.tradovate.auth import TradovateAuth, TradovateCredentials
from bot.execution.tradovate.rest_client import TradovateRestClient
from bot.execution.tradovate.ws_client import TradovateWebSocket

logger = logging.getLogger(__name__)
ET = pytz.timezone("America/New_York")

# WebSocket URLs
WS_URLS = {
    "demo": "wss://demo.tradovateapi.com/v1/websocket",
    "live": "wss://live.tradovateapi.com/v1/websocket",
    "md": "wss://md.tradovateapi.com/v1/websocket",
}

# Tradovate contract month codes
MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}
MONTH_FROM_CODE = {v: k for k, v in MONTH_CODES.items()}


def _expiry_to_tradovate_name(symbol: str, expiry: datetime) -> str:
    """Convert symbol + expiry date to Tradovate contract name.

    E.g., NQ + 2026-06-19 → NQM6
    """
    month = expiry.month
    # Find the nearest quarterly month
    quarterly_months = sorted(MONTH_CODES.keys())
    contract_month = month
    for qm in quarterly_months:
        if qm >= month:
            contract_month = qm
            break
    else:
        contract_month = quarterly_months[0]  # Next year's March

    code = MONTH_CODES[contract_month]
    year_digit = str(expiry.year)[-1]
    return f"{symbol}{code}{year_digit}"


class TradovateAdapter(BrokerAdapter):
    """Tradovate broker adapter — REST + WebSocket, cloud-native."""

    def __init__(self, config, bot_logger=None, clock=None):
        self._config = config
        self._bot_logger = bot_logger
        self._clock = clock

        # Credentials loaded from AWS SSM Parameter Store at connect() time.
        # Nothing sensitive touches config or disk.
        self._creds: Optional[TradovateCredentials] = None
        self._auth: Optional[TradovateAuth] = None
        self._rest: Optional[TradovateRestClient] = None

        # WebSockets
        self._order_ws = TradovateWebSocket(name="order")
        self._md_ws = TradovateWebSocket(name="md")

        # Account state (populated by user sync)
        self._account_id: Optional[int] = None
        self._account_spec: Optional[str] = None
        self._positions: Dict[int, Dict] = {}      # contractId → position
        self._orders: Dict[int, Dict] = {}          # orderId → order
        self._cash_balance: float = 0.0
        self._realized_pnl: float = 0.0

        # Bar subscription tracking
        self._bar_subscriptions: Dict[str, _BarSubscription] = {}
        self._next_sub_id = 1

        # Reconnect callbacks
        self._reconnect_callbacks: List[Callable] = []

        # Connection state
        self._disconnect_time: float = 0.0

        # Order ID counter (Tradovate assigns IDs server-side, but we need
        # pre-allocated IDs for crash-safe state persistence)
        self._local_order_id = 1000000

    # ── Connection ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        # Step 0: Load credentials from AWS SSM Parameter Store
        from bot.secret_store import SecretStore
        env = self._config.tradovate_environment
        store = SecretStore(environment=env)
        secrets = store.load_tradovate()

        self._creds = TradovateCredentials(
            username=secrets.username,
            password=secrets.password,
            app_id=secrets.app_id,
            app_version=secrets.app_version or self._config.tradovate_app_version,
            cid=secrets.cid,
            sec=secrets.sec,
            device_id=secrets.device_id,
            environment=env,
        )
        self._auth = TradovateAuth(self._creds)
        self._rest = TradovateRestClient(self._auth)

        # Step 1: Authenticate via REST
        token = await self._auth.authenticate()
        await self._auth.start_renewal_loop()

        # Step 2: Resolve account
        accounts = await self._rest.list_accounts()
        if not accounts:
            raise ConnectionError("No Tradovate accounts found")
        # Pick the first account (or the one with largest balance)
        account = accounts[0]
        self._account_id = account["id"]
        self._account_spec = account["name"]
        logger.info("Using Tradovate account: %s (id=%d)",
                     self._account_spec, self._account_id)

        # Step 3: Connect WebSockets
        # Order WS: environment-specific URL with access_token
        # Market data WS: always md.tradovateapi.com with mdAccessToken
        #   (md feed is live exchange data regardless of demo/live account)
        env = self._creds.environment
        order_url = WS_URLS[env]
        md_url = WS_URLS["md"]
        md_token = token.md_access_token or token.access_token

        await self._order_ws.connect(order_url, token.access_token)
        await self._md_ws.connect(md_url, md_token)

        # Step 4: Start user sync (positions, orders, accounts)
        await self._start_user_sync()

        # Register reconnect handlers
        self._order_ws.on_reconnect(self._on_order_ws_reconnect)
        self._md_ws.on_reconnect(self._on_md_ws_reconnect)

    async def disconnect(self) -> None:
        await self._order_ws.disconnect()
        await self._md_ws.disconnect()
        await self._rest.close()
        await self._auth.close()

    @property
    def is_connected(self) -> bool:
        return self._order_ws.is_connected and self._md_ws.is_connected

    @property
    def disconnect_seconds(self) -> float:
        if self.is_connected:
            return 0.0
        return max(
            self._order_ws.disconnect_seconds,
            self._md_ws.disconnect_seconds,
        )

    def on_reconnect(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._reconnect_callbacks.append(callback)

    async def _on_order_ws_reconnect(self) -> None:
        await self._start_user_sync()
        for cb in self._reconnect_callbacks:
            await cb()

    async def _on_md_ws_reconnect(self) -> None:
        # Re-subscribe all active bar subscriptions
        for sub in self._bar_subscriptions.values():
            await self._resubscribe_bars(sub)

    # ── User Sync ───────────────────────────────────────────────────────

    async def _start_user_sync(self) -> None:
        """Subscribe to real-time user data (positions, orders, accounts)."""
        self._order_ws.add_listener(
            filter_fn=lambda item: "e" in item or "d" in item,
            callback=self._on_user_sync_event,
        )
        await self._order_ws.request("user/syncrequest", {
            "users": [self._auth.token.user_id],
        })

    def _on_user_sync_event(self, item: Dict) -> None:
        """Handle real-time user data updates."""
        event_type = item.get("e")
        data = item.get("d", item)

        if event_type == "props":
            # Property updates for positions, orders, accounts
            for entity_type, entities in data.items():
                if entity_type == "positions":
                    for pos in entities:
                        cid = pos.get("contractId")
                        if cid:
                            self._positions[cid] = pos
                elif entity_type == "orders":
                    for order in entities:
                        oid = order.get("id")
                        if oid:
                            self._orders[oid] = order
                            self._dispatch_order_event(order)
                elif entity_type == "accounts":
                    for acct in entities:
                        if acct.get("id") == self._account_id:
                            self._cash_balance = acct.get("cashBalance", 0.0)
                elif entity_type == "cashBalances":
                    for cb in entities:
                        if cb.get("accountId") == self._account_id:
                            self._realized_pnl = cb.get("realizedPnL", 0.0)
                            self._cash_balance = cb.get("cashBalance", self._cash_balance)

        # Initial sync also sends data directly
        if "positions" in data:
            for pos in data["positions"]:
                cid = pos.get("contractId")
                if cid:
                    self._positions[cid] = pos
        if "orders" in data:
            for order in data["orders"]:
                oid = order.get("id")
                if oid:
                    self._orders[oid] = order

    # ── Order Event Dispatch ────────────────────────────────────────────

    _order_callbacks: Dict[str, Dict] = {}

    def _dispatch_order_event(self, order: Dict) -> None:
        """Route order status changes to registered callbacks.

        On entry fill, places the OCO exit pair (TP Limit + SL Stop).
        """
        status = order.get("ordStatus", "")
        if not status:
            return

        order_id = order.get("id")

        for bracket_key, cbs in list(self._order_callbacks.items()):
            entry_id = cbs.get("_entry_id")
            tp_id = cbs.get("_tp_id")
            sl_id = cbs.get("_sl_id")

            if order_id == entry_id and status == "Filled":
                cbs["on_entry_fill"](order)
                if not cbs.get("_oco_placed"):
                    cbs["_oco_placed"] = True
                    asyncio.ensure_future(self._place_exit_oco(bracket_key))
            elif order_id == tp_id and status == "Filled":
                cbs["on_tp_fill"](order)
            elif order_id == sl_id and status == "Filled":
                cbs["on_sl_fill"](order)

            if order_id in (entry_id, tp_id, sl_id):
                cbs["on_status_change"](order)

    async def _place_exit_oco(self, bracket_key: str) -> None:
        """Place OCO exit pair (TP Limit + SL Stop) after entry fills."""
        cbs = self._order_callbacks.get(bracket_key)
        if not cbs:
            return

        symbol = cbs["_symbol"]
        reverse = cbs["_reverse_action"]
        qty = cbs["_qty"]
        tp_price = cbs["_tp_price"]
        sl_price = cbs["_sl_price"]

        try:
            result = await self._rest.place_oco(
                account_id=self._account_id,
                account_spec=self._account_spec,
                symbol=symbol,
                action=reverse,
                order_qty=qty,
                order_type="Limit",
                price=tp_price,
                other={
                    "action": reverse,
                    "orderType": "Stop",
                    "stopPrice": sl_price,
                },
            )
            tp_real_id = result.get("orderId")
            sl_real_id = result.get("ocoId")
            cbs["_tp_id"] = tp_real_id
            cbs["_sl_id"] = sl_real_id
            logger.info(
                "OCO exits placed: TP=%s SL=%s for bracket %s",
                tp_real_id, sl_real_id, bracket_key,
            )
        except Exception as e:
            logger.error("Failed to place OCO exits for bracket %s: %s", bracket_key, e)

    # ── Contract Resolution ─────────────────────────────────────────────

    async def resolve_contract(
        self,
        symbol: str,
        exchange: str,
        expiry_hint: Optional[datetime] = None,
    ) -> ContractInfo:
        now = expiry_hint or datetime.now(ET)
        name = _expiry_to_tradovate_name(symbol, now)

        contract = await self._rest.find_contract(name)

        # Get product details for tick size and point value
        product_id = contract.get("contractMaturityId") or contract.get("productId")
        tick_size = self._config.tick_size
        point_value = self._config.point_value

        return ContractInfo(
            symbol=symbol,
            broker_contract_id=str(contract["id"]),
            expiry=contract.get("expirationDate", ""),
            exchange=exchange,
            tick_size=tick_size,
            point_value=point_value,
            name=name,
        )

    # ── Market Data ─────────────────────────────────────────────────────

    async def subscribe_bars(
        self,
        contract: ContractInfo,
        bar_size: str,
        on_bar: Callable[[BarData, bool], Any],
    ) -> str:
        bar_map = {"5 mins": 5, "1 min": 1, "3 mins": 3, "15 mins": 15}
        element_size = bar_map.get(bar_size, 5)

        sub_id = f"bars_{self._next_sub_id}"
        self._next_sub_id += 1

        sub = _BarSubscription(
            sub_id=sub_id,
            contract=contract,
            element_size=element_size,
            on_bar=on_bar,
            current_bar_ts=None,
        )
        self._bar_subscriptions[sub_id] = sub

        await self._subscribe_bars_ws(sub)
        return sub_id

    async def _subscribe_bars_ws(self, sub: _BarSubscription) -> None:
        """Send md/getChart request on the market data WebSocket."""
        # Calculate session start for today
        now = datetime.now(ET)
        session_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now < session_start:
            session_start = session_start.replace(day=session_start.day - 1)

        body = {
            "symbol": sub.contract.name or sub.contract.symbol,
            "chartDescription": {
                "underlyingType": "MinuteBar",
                "elementSize": sub.element_size,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": False,
            },
            "timeRange": {
                "asFarAsTimestamp": session_start.astimezone(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.000Z"
                ),
            },
        }

        response = await self._md_ws.request("md/getChart", body)

        # Extract realtime and historical IDs for filtering
        sub.realtime_id = response.get("d", {}).get("realtimeId")
        sub.historical_id = response.get("d", {}).get("historicalId")

        # Register listener for chart data
        def is_chart_event(item: Dict) -> bool:
            d = item.get("d", {})
            charts = d.get("charts") or []
            for chart in charts:
                cid = chart.get("id")
                if cid in (sub.realtime_id, sub.historical_id):
                    return True
            return False

        def on_chart_data(item: Dict) -> None:
            self._handle_chart_data(sub, item)

        sub.unsub_listener = self._md_ws.add_listener(is_chart_event, on_chart_data)

    async def _resubscribe_bars(self, sub: _BarSubscription) -> None:
        """Re-subscribe bars after reconnection."""
        if sub.unsub_listener:
            sub.unsub_listener()
        await self._subscribe_bars_ws(sub)

    def _handle_chart_data(self, sub: _BarSubscription, item: Dict) -> None:
        """Process incoming chart bar data.

        Tradovate sends bar updates as they tick (in-progress bars).
        We buffer the current bar and fire on_bar(completed=True) only
        when a new timestamp appears.
        """
        d = item.get("d", {})
        charts = d.get("charts") or []

        for chart in charts:
            bars = chart.get("bars") or []
            for bar_data in bars:
                ts_str = bar_data.get("timestamp", "")
                if not ts_str:
                    continue

                try:
                    bar_ts = datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00")
                    ).astimezone(ET)
                except (ValueError, TypeError):
                    continue

                bar = BarData(
                    timestamp=bar_ts,
                    open=bar_data.get("open", 0.0),
                    high=bar_data.get("high", 0.0),
                    low=bar_data.get("low", 0.0),
                    close=bar_data.get("close", 0.0),
                    volume=bar_data.get("upVolume", 0) + bar_data.get("downVolume", 0),
                )

                if sub.current_bar_ts is None:
                    # First bar — seed
                    sub.current_bar_ts = bar_ts
                    sub.on_bar(bar, False)
                elif bar_ts > sub.current_bar_ts:
                    # New bar started — previous bar is complete
                    sub.current_bar_ts = bar_ts
                    sub.on_bar(bar, True)
                else:
                    # Update to current bar (in-progress)
                    sub.on_bar(bar, False)

    async def unsubscribe_bars(self, subscription_id: str) -> None:
        sub = self._bar_subscriptions.pop(subscription_id, None)
        if sub is None:
            return
        if sub.unsub_listener:
            sub.unsub_listener()
        try:
            await self._md_ws.request("md/cancelChart", {
                "subscriptionId": sub.realtime_id,
            })
        except Exception:
            pass

    async def subscribe_ticks(
        self,
        contract: ContractInfo,
        on_tick: Callable[[TickData], Any],
    ) -> Optional[str]:
        """Subscribe to trade ticks via md/subscribeQuote."""
        sub_id = f"ticks_{self._next_sub_id}"
        self._next_sub_id += 1

        response = await self._md_ws.request("md/subscribeQuote", {
            "symbol": contract.name or contract.symbol,
        })

        last_price = [0.0]  # Mutable closure for tracking last trade price

        def is_quote(item: Dict) -> bool:
            d = item.get("d", {})
            return "quotes" in d

        def on_quote(item: Dict) -> None:
            d = item.get("d", {})
            for quote in d.get("quotes", []):
                entries = quote.get("entries", {})
                trade = entries.get("Trade", {})
                price = trade.get("price")
                if price is not None and price != last_price[0]:
                    last_price[0] = price
                    size = trade.get("size", 0)
                    ts_str = quote.get("timestamp", "")
                    try:
                        ts = datetime.fromisoformat(
                            ts_str.replace("Z", "+00:00")
                        ).astimezone(ET)
                    except (ValueError, TypeError):
                        ts = datetime.now(ET)

                    on_tick(TickData(timestamp=ts, price=price, size=float(size)))

        unsub = self._md_ws.add_listener(is_quote, on_quote)
        # Store for cleanup
        self._bar_subscriptions[sub_id] = _BarSubscription(
            sub_id=sub_id,
            contract=contract,
            element_size=0,
            on_bar=lambda *a: None,
            unsub_listener=unsub,
        )
        return sub_id

    async def unsubscribe_ticks(self, subscription_id: str) -> None:
        sub = self._bar_subscriptions.pop(subscription_id, None)
        if sub and sub.unsub_listener:
            sub.unsub_listener()
        try:
            await self._md_ws.request("md/unsubscribeQuote", {
                "symbol": sub.contract.name if sub else "",
            })
        except Exception:
            pass

    # ── Order Management ────────────────────────────────────────────────

    async def place_bracket_order(
        self,
        contract: ContractInfo,
        side: str,
        qty: int,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        on_entry_fill: Callable,
        on_tp_fill: Callable,
        on_sl_fill: Callable,
        on_status_change: Callable,
    ) -> BracketOrderResult:
        """Place bracket: Limit entry, poll for fill, then OCO exits.

        Mirrors the IB adapter pattern — entry is a standalone Limit order.
        A background task polls for the fill, then places a Limit TP + Stop SL
        as an OCO pair. Both exit legs are visible as regular orders.
        """
        action = "Buy" if side == "BUY" else "Sell"
        reverse_action = "Sell" if side == "BUY" else "Buy"
        symbol = contract.name or contract.symbol

        tp_id = self._allocate_local_id()
        sl_id = self._allocate_local_id()

        try:
            result = await self._rest.place_order(
                account_id=self._account_id,
                account_spec=self._account_spec,
                symbol=symbol,
                action=action,
                order_qty=qty,
                order_type="Limit",
                price=entry_price,
            )

            entry_id = result.get("orderId", result.get("id"))
            bracket_key = str(entry_id)

            cbs = {
                "on_entry_fill": on_entry_fill,
                "on_tp_fill": on_tp_fill,
                "on_sl_fill": on_sl_fill,
                "on_status_change": on_status_change,
                "_entry_id": entry_id,
                "_tp_id": None,
                "_sl_id": None,
                "_oco_placed": False,
                "_symbol": symbol,
                "_reverse_action": reverse_action,
                "_qty": qty,
                "_tp_price": tp_price,
                "_sl_price": sl_price,
            }
            self._order_callbacks[bracket_key] = cbs

            asyncio.ensure_future(self._poll_entry_fill(bracket_key))

            return BracketOrderResult(
                success=True,
                entry_order_id=str(entry_id),
                tp_order_id=tp_id,
                sl_order_id=sl_id,
            )

        except Exception as e:
            logger.error("Bracket entry placement failed: %s", e)
            return BracketOrderResult(
                success=False,
                error=str(e),
            )

    async def _poll_entry_fill(self, bracket_key: str, timeout: float = 300) -> None:
        """Poll REST for entry fill, then place OCO exits."""
        cbs = self._order_callbacks.get(bracket_key)
        if not cbs:
            return

        entry_id = cbs["_entry_id"]
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            try:
                order = await self._rest._get("/order/item", {"id": entry_id})
                status = order.get("ordStatus", "")
                if status == "Filled":
                    logger.info("Entry %s filled, placing OCO exits", entry_id)
                    cbs["on_entry_fill"](order)
                    await self._place_exit_oco(bracket_key)
                    return
                if status in ("Cancelled", "Rejected"):
                    logger.warning("Entry %s %s, skipping OCO", entry_id, status)
                    cbs["on_status_change"](order)
                    return
            except Exception as e:
                logger.debug("Poll entry %s: %s", entry_id, e)
            await asyncio.sleep(0.5)

        logger.warning("Entry %s fill poll timed out after %.0fs", entry_id, timeout)

    async def cancel_order(self, order_id: str) -> None:
        oid = int(order_id)
        try:
            await self._rest.cancel_order(oid)
        except Exception as e:
            logger.error("Cancel order %s failed: %s", order_id, e)

        # Cancel the OCO counterpart — Tradovate does not always auto-cancel
        for cbs in self._order_callbacks.values():
            tp_id = cbs.get("_tp_id")
            sl_id = cbs.get("_sl_id")
            if oid == tp_id and sl_id:
                try:
                    await self._rest.cancel_order(sl_id)
                except Exception:
                    pass
            elif oid == sl_id and tp_id:
                try:
                    await self._rest.cancel_order(tp_id)
                except Exception:
                    pass

    async def cancel_bracket_exits(self, entry_order_id: str) -> None:
        """Cancel TP + SL exits for a bracket. Used before flattening."""
        cbs = self._order_callbacks.get(str(entry_order_id))
        if not cbs:
            # Fallback: cancel ALL working orders (flatten safety)
            for o in (await self._rest._get("/order/list") or []):
                if o.get("ordStatus") in ("Working", "Accepted"):
                    try:
                        await self._rest.cancel_order(o["id"])
                    except Exception:
                        pass
            return
        for key in ("_tp_id", "_sl_id"):
            oid = cbs.get(key)
            if oid:
                try:
                    await self._rest.cancel_order(oid)
                except Exception:
                    pass

    async def modify_order_qty(self, order_id: str, new_qty: int) -> None:
        try:
            await self._rest.modify_order(int(order_id), order_qty=new_qty)
        except Exception as e:
            logger.error("Modify order %s qty=%d failed: %s", order_id, new_qty, e)

    async def place_market_order(
        self,
        contract: ContractInfo,
        side: str,
        qty: int,
    ) -> str:
        action = "Buy" if side == "BUY" else "Sell"
        result = await self._rest.place_order(
            account_id=self._account_id,
            account_spec=self._account_spec,
            symbol=contract.name or contract.symbol,
            action=action,
            order_qty=qty,
            order_type="Market",
        )
        return str(result.get("orderId", ""))

    async def get_open_trades(self) -> List[OpenOrderSnapshot]:
        return await self.get_open_orders()

    # ── Account Data ────────────────────────────────────────────────────

    async def get_account_balance(self) -> Optional[float]:
        return self._cash_balance or None

    async def get_positions(self) -> List[PositionSnapshot]:
        result = []
        for cid, pos in self._positions.items():
            net_pos = pos.get("netPos", 0)
            if net_pos == 0:
                continue
            result.append(PositionSnapshot(
                broker="tradovate",
                symbol=str(cid),
                side="BUY" if net_pos > 0 else "SELL",
                quantity=abs(net_pos),
                entry_price=pos.get("netPrice", 0.0),
                unrealized_pnl=pos.get("openPL", 0.0),
                raw=pos,
            ))
        return result

    async def get_open_orders(self) -> List[OpenOrderSnapshot]:
        result = []
        for oid, order in self._orders.items():
            status = order.get("ordStatus", "")
            if status not in ("Working", "Accepted"):
                continue
            result.append(OpenOrderSnapshot(
                broker="tradovate",
                symbol=order.get("contractId", ""),
                side="BUY" if order.get("action") == "Buy" else "SELL",
                order_type=order.get("ordType", ""),
                status=status,
                quantity=order.get("ordQty", 0),
                price=order.get("price", 0.0),
                stop_price=order.get("stopPrice", 0.0),
                order_id=str(oid),
                raw=order,
            ))
        return result

    async def get_margin_per_contract(self, contract: ContractInfo) -> Optional[float]:
        # Tradovate has no whatIfOrder equivalent — return None to trigger
        # config-based margin fallback in MarginTracker
        return None

    async def get_available_funds(self) -> float:
        return self._cash_balance

    # ── Time Sync ───────────────────────────────────────────────────────

    async def get_server_time(self) -> Optional[datetime]:
        # Tradovate doesn't expose a dedicated server time endpoint.
        # We could parse timestamps from WS responses, but for now
        # return None to skip broker time validation.
        return None

    # ── Order ID Allocation ─────────────────────────────────────────────

    def allocate_order_ids(self, count: int) -> List[str]:
        ids = []
        for _ in range(count):
            ids.append(self._allocate_local_id())
        return ids

    def _allocate_local_id(self) -> str:
        """Generate a local order ID placeholder.

        Tradovate assigns real IDs server-side. These local IDs are used
        only for crash-safe state persistence before the server responds.
        """
        oid = f"tv_{self._local_order_id}"
        self._local_order_id += 1
        return oid


# ── Internal Types ──────────────────────────────────────────────────────

class _BarSubscription:
    """Tracks an active bar data subscription."""
    __slots__ = (
        "sub_id", "contract", "element_size", "on_bar",
        "current_bar_ts", "realtime_id", "historical_id", "unsub_listener",
    )

    def __init__(
        self,
        sub_id: str,
        contract: ContractInfo,
        element_size: int,
        on_bar: Callable,
        current_bar_ts: Optional[datetime] = None,
        unsub_listener: Optional[Callable] = None,
    ):
        self.sub_id = sub_id
        self.contract = contract
        self.element_size = element_size
        self.on_bar = on_bar
        self.current_bar_ts = current_bar_ts
        self.realtime_id = None
        self.historical_id = None
        self.unsub_listener = unsub_listener
