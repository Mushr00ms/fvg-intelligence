"""
binance_futures_client.py — Async Binance USDⓈ-M Futures REST/WebSocket client.

Scope for the first integration pass:
- Signed REST requests
- Exchange metadata loading
- Account / balances / positions / open orders
- Leverage + margin configuration helpers
- Order placement / cancellation
- User data stream bootstrap
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from urllib.parse import urlencode

import aiohttp

from bot.execution.execution_types import (
    AccountSnapshot,
    BrokerOrderAck,
    OpenOrderSnapshot,
    PositionSnapshot,
    SymbolRules,
)


class BinanceFuturesError(RuntimeError):
    """Raised when Binance returns an API-level or transport-level error."""

    def __init__(self, message: str, *, status: int = 0, code: int | None = None, payload=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.payload = payload


@dataclass(frozen=True)
class BinanceListenKey:
    """User data stream listen key with creation timestamp."""

    listen_key: str
    created_at_ms: int


class BinanceFuturesClient:
    """Async client for Binance USDⓈ-M Futures."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = "https://fapi.binance.com",
        ws_base_url: str = "wss://fstream.binance.com",
        recv_window: int = 5000,
        logger=None,
        clock=None,
        session: aiohttp.ClientSession | None = None,
    ):
        self._api_key = api_key or ""
        self._api_secret = api_secret or ""
        self._base_url = base_url.rstrip("/")
        self._ws_base_url = ws_base_url.rstrip("/")
        self._recv_window = int(recv_window)
        self._logger = logger
        self._clock = clock
        self._session = session
        self._owned_session = session is None

    async def start(self):
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def close(self):
        if self._owned_session and self._session is not None:
            await self._session.close()
        self._session = None

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    def _timestamp_ms(self) -> int:
        if self._clock is not None:
            return int(self._clock.now().timestamp() * 1000)
        return int(time.time() * 1000)

    def _log(self, event: str, **fields):
        if self._logger is not None:
            self._logger.log(event, broker="binance_um", **fields)

    def _require_auth(self):
        if not self._api_key or not self._api_secret:
            raise BinanceFuturesError("Binance API key/secret not configured")

    def _sign_params(self, params: dict) -> str:
        query = urlencode(params, doseq=True)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{query}&signature={signature}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        signed: bool = False,
    ):
        if self._session is None:
            await self.start()

        params = dict(params or {})
        headers = {}
        query = ""

        if signed:
            self._require_auth()
            params.setdefault("recvWindow", self._recv_window)
            params["timestamp"] = self._timestamp_ms()
            query = self._sign_params(params)
            headers["X-MBX-APIKEY"] = self._api_key
        else:
            if self._api_key:
                headers["X-MBX-APIKEY"] = self._api_key
            query = urlencode(params, doseq=True)

        url = f"{self._base_url}{path}"
        if query:
            url = f"{url}?{query}"

        async with self._session.request(method.upper(), url, headers=headers) as resp:
            text = await resp.text()
            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError:
                payload = {"raw": text}

            if resp.status >= 400:
                msg = payload.get("msg") if isinstance(payload, dict) else str(payload)
                code = payload.get("code") if isinstance(payload, dict) else None
                raise BinanceFuturesError(
                    f"Binance API error {resp.status}: {msg}",
                    status=resp.status,
                    code=code,
                    payload=payload,
                )
            return payload

    async def ping(self) -> bool:
        await self._request("GET", "/fapi/v1/ping")
        return True

    async def get_server_time(self) -> int:
        payload = await self._request("GET", "/fapi/v1/time")
        return int(payload["serverTime"])

    async def get_exchange_info(self) -> dict:
        return await self._request("GET", "/fapi/v1/exchangeInfo")

    async def get_symbol_rules(self, symbol: str) -> SymbolRules:
        info = await self.get_exchange_info()
        for item in info.get("symbols", []):
            if item.get("symbol") != symbol:
                continue
            filters = {f["filterType"]: f for f in item.get("filters", [])}
            price_filter = filters.get("PRICE_FILTER", {})
            lot_size = filters.get("LOT_SIZE", {})
            min_notional = filters.get("MIN_NOTIONAL", {})
            return SymbolRules(
                symbol=symbol,
                price_tick_size=float(price_filter.get("tickSize", "0") or 0.0),
                quantity_step_size=float(lot_size.get("stepSize", "0") or 0.0),
                min_quantity=float(lot_size.get("minQty", "0") or 0.0),
                min_notional=float(min_notional.get("notional", "0") or 0.0),
                price_precision=item.get("pricePrecision"),
                quantity_precision=item.get("quantityPrecision"),
                trigger_protect=float(item.get("triggerProtect", "0") or 0.0),
                contract_type=item.get("contractType", ""),
                base_asset=item.get("baseAsset", ""),
                quote_asset=item.get("quoteAsset", ""),
                margin_asset=item.get("marginAsset", ""),
            )
        raise BinanceFuturesError(f"Symbol not found in exchangeInfo: {symbol}")

    async def get_account_snapshot(self, asset: str = "USDT") -> AccountSnapshot:
        payload = await self._request("GET", "/fapi/v2/account", signed=True)
        asset_entry = None
        for item in payload.get("assets", []):
            if item.get("asset") == asset:
                asset_entry = item
                break
        source = asset_entry or payload
        return AccountSnapshot(
            broker="binance_um",
            wallet_balance=float(source.get("walletBalance", payload.get("totalWalletBalance", 0.0))),
            available_balance=float(source.get("availableBalance", payload.get("availableBalance", 0.0))),
            margin_balance=float(source.get("marginBalance", payload.get("totalMarginBalance", 0.0))),
            unrealized_pnl=float(source.get("unrealizedProfit", payload.get("totalUnrealizedProfit", 0.0))),
            initial_margin=float(source.get("initialMargin", payload.get("totalInitialMargin", 0.0))),
            maintenance_margin=float(source.get("maintMargin", payload.get("totalMaintMargin", 0.0))),
            raw=payload,
        )

    async def get_balances(self) -> list[dict]:
        return await self._request("GET", "/fapi/v2/balance", signed=True)

    async def get_positions(self, symbol: str | None = None) -> list[PositionSnapshot]:
        params = {"symbol": symbol} if symbol else {}
        payload = await self._request("GET", "/fapi/v2/positionRisk", params=params, signed=True)
        positions = []
        for item in payload:
            qty = float(item.get("positionAmt", 0.0))
            if qty == 0:
                continue
            positions.append(
                PositionSnapshot(
                    broker="binance_um",
                    symbol=item["symbol"],
                    side="BUY" if qty > 0 else "SELL",
                    quantity=abs(qty),
                    entry_price=float(item.get("entryPrice", 0.0)),
                    mark_price=float(item.get("markPrice", 0.0)),
                    unrealized_pnl=float(item.get("unRealizedProfit", 0.0)),
                    leverage=int(float(item.get("leverage", 0) or 0)),
                    margin_type=str(item.get("marginType", "")).upper(),
                    position_side=str(item.get("positionSide", "BOTH")).upper(),
                    liquidation_price=float(item.get("liquidationPrice", 0.0)),
                    raw=item,
                )
            )
        return positions

    async def get_open_orders(self, symbol: str | None = None) -> list[OpenOrderSnapshot]:
        params = {"symbol": symbol} if symbol else {}
        payload = await self._request("GET", "/fapi/v1/openOrders", params=params, signed=True)
        orders = []
        for item in payload:
            orders.append(
                OpenOrderSnapshot(
                    broker="binance_um",
                    symbol=item["symbol"],
                    side=item["side"],
                    order_type=item["type"],
                    status=item["status"],
                    quantity=float(item.get("origQty", 0.0)),
                    price=float(item.get("price", 0.0)),
                    stop_price=float(item.get("stopPrice", 0.0)),
                    order_id=str(item.get("orderId", "")),
                    client_order_id=item.get("clientOrderId", ""),
                    position_side=str(item.get("positionSide", "BOTH")).upper(),
                    reduce_only=bool(item.get("reduceOnly", False)),
                    raw=item,
                )
            )
        return orders

    async def get_open_algo_orders(
        self,
        symbol: str | None = None,
        *,
        algo_type: str | None = None,
        algo_id: str | None = None,
    ) -> list[OpenOrderSnapshot]:
        params = {}
        if symbol:
            params["symbol"] = symbol
        if algo_type:
            params["algoType"] = algo_type
        if algo_id:
            params["algoId"] = algo_id
        payload = await self._request("GET", "/fapi/v1/openAlgoOrders", params=params, signed=True)
        orders = []
        for item in payload:
            orders.append(
                OpenOrderSnapshot(
                    broker="binance_um",
                    symbol=item["symbol"],
                    side=item["side"],
                    order_type=item["orderType"],
                    status=item["algoStatus"],
                    quantity=float(item.get("quantity", 0.0)),
                    price=float(item.get("price", 0.0)),
                    stop_price=float(item.get("triggerPrice", 0.0)),
                    order_id=str(item.get("algoId", "")),
                    client_order_id=item.get("clientAlgoId", ""),
                    position_side=str(item.get("positionSide", "BOTH")).upper(),
                    reduce_only=bool(item.get("reduceOnly", False)),
                    raw=item,
                )
            )
        return orders

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        payload = await self._request(
            "POST",
            "/fapi/v1/leverage",
            params={"symbol": symbol, "leverage": int(leverage)},
            signed=True,
        )
        self._log("binance_leverage_set", symbol=symbol, leverage=leverage)
        return payload

    async def set_margin_type(self, symbol: str, margin_type: str = "CROSSED") -> dict:
        payload = await self._request(
            "POST",
            "/fapi/v1/marginType",
            params={"symbol": symbol, "marginType": margin_type.upper()},
            signed=True,
        )
        self._log("binance_margin_type_set", symbol=symbol, margin_type=margin_type.upper())
        return payload

    async def get_position_mode(self) -> bool:
        payload = await self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
        return bool(payload.get("dualSidePosition", False))

    async def set_position_mode(self, hedge_mode: bool) -> dict:
        payload = await self._request(
            "POST",
            "/fapi/v1/positionSide/dual",
            params={"dualSidePosition": "true" if hedge_mode else "false"},
            signed=True,
        )
        self._log("binance_position_mode_set", hedge_mode=hedge_mode)
        return payload

    async def create_order(self, **params) -> BrokerOrderAck:
        payload = await self._request("POST", "/fapi/v1/order", params=params, signed=True)
        return BrokerOrderAck(
            broker="binance_um",
            symbol=payload["symbol"],
            side=payload["side"],
            order_type=payload["type"],
            status=payload.get("status", ""),
            order_id=str(payload.get("orderId", "")),
            client_order_id=payload.get("clientOrderId", ""),
            quantity=float(payload.get("origQty", 0.0)),
            price=float(payload.get("price", 0.0)),
            stop_price=float(payload.get("stopPrice", 0.0)),
            position_side=str(payload.get("positionSide", "")).upper(),
            reduce_only=bool(payload.get("reduceOnly", False)),
            raw=payload,
        )

    async def create_algo_order(self, **params) -> BrokerOrderAck:
        payload = await self._request("POST", "/fapi/v1/algoOrder", params=params, signed=True)
        return BrokerOrderAck(
            broker="binance_um",
            symbol=payload["symbol"],
            side=payload["side"],
            order_type=payload["orderType"],
            status=payload.get("algoStatus", ""),
            order_id=str(payload.get("algoId", "")),
            client_order_id=payload.get("clientAlgoId", ""),
            quantity=float(payload.get("quantity", 0.0)),
            price=float(payload.get("price", 0.0)),
            stop_price=float(payload.get("triggerPrice", 0.0)),
            position_side=str(payload.get("positionSide", "")).upper(),
            reduce_only=bool(payload.get("reduceOnly", False)),
            raw=payload,
        )

    async def get_order(self, symbol: str, *, order_id: str | None = None,
                        client_order_id: str | None = None) -> dict:
        params = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        return await self._request("GET", "/fapi/v1/order", params=params, signed=True)

    async def cancel_order(self, symbol: str, *, order_id: str | None = None,
                           client_order_id: str | None = None) -> dict:
        params = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        if client_order_id is not None:
            params["origClientOrderId"] = client_order_id
        return await self._request("DELETE", "/fapi/v1/order", params=params, signed=True)

    async def cancel_all_orders(self, symbol: str) -> dict:
        return await self._request(
            "DELETE",
            "/fapi/v1/allOpenOrders",
            params={"symbol": symbol},
            signed=True,
        )

    async def cancel_algo_order(self, *, algo_id: str | None = None,
                                client_algo_id: str | None = None) -> dict:
        params = {}
        if algo_id is not None:
            params["algoId"] = algo_id
        if client_algo_id is not None:
            params["clientAlgoId"] = client_algo_id
        return await self._request("DELETE", "/fapi/v1/algoOrder", params=params, signed=True)

    async def create_listen_key(self) -> BinanceListenKey:
        self._require_auth()
        payload = await self._request("POST", "/fapi/v1/listenKey", signed=False)
        key = payload["listenKey"]
        return BinanceListenKey(listen_key=key, created_at_ms=self._timestamp_ms())

    async def keepalive_listen_key(self, listen_key: str) -> dict:
        self._require_auth()
        return await self._request(
            "PUT",
            "/fapi/v1/listenKey",
            params={"listenKey": listen_key},
            signed=False,
        )

    async def close_listen_key(self, listen_key: str) -> dict:
        self._require_auth()
        return await self._request(
            "DELETE",
            "/fapi/v1/listenKey",
            params={"listenKey": listen_key},
            signed=False,
        )

    async def user_stream(self, listen_key: str) -> AsyncIterator[dict]:
        """Yield raw user-data events from Binance Futures."""
        if self._session is None:
            await self.start()
        ws_url = f"{self._ws_base_url}/ws/{listen_key}"
        async with self._session.ws_connect(ws_url, heartbeat=20) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    yield json.loads(msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise BinanceFuturesError(
                        f"User stream websocket error: {ws.exception()}",
                    )
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    break

    async def maintain_listen_key(self, listen_key: str, *, interval_seconds: int = 1800,
                                  stop_event: asyncio.Event | None = None):
        """Keep a listen key alive until stop_event is set."""
        while True:
            if stop_event is not None and stop_event.is_set():
                return
            await asyncio.sleep(interval_seconds)
            await self.keepalive_listen_key(listen_key)
