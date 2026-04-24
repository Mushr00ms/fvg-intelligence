"""
rest_client.py — Tradovate REST API client.

Typed wrappers around the Tradovate HTTP/REST API for order management,
contract resolution, and account queries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import aiohttp

from bot.execution.tradovate.auth import TradovateAuth

logger = logging.getLogger(__name__)


class TradovateRestClient:
    """HTTP REST client for Tradovate API."""

    def __init__(self, auth: TradovateAuth):
        self._auth = auth
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def _base_url(self) -> str:
        return self._auth._creds.base_url

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._auth.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Generic HTTP request to Tradovate API."""
        session = await self._ensure_session()
        url = f"{self._base_url}{path}"

        async with session.request(
            method, url, json=body, params=params, headers=self._headers()
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise TradovateAPIError(
                    f"{method} {path} failed (HTTP {resp.status}): {text}"
                )
            if not text:
                return {}
            return await resp.json(content_type=None)

    async def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        return await self._request("GET", path, params=params)

    async def _post(self, path: str, body: Optional[Dict] = None) -> Any:
        return await self._request("POST", path, body=body)

    # ── Contract Resolution ─────────────────────────────────────────────

    async def find_contract(self, name: str) -> Dict[str, Any]:
        """Find a contract by name (e.g., 'NQM6').

        Returns contract object with id, name, contractMaturityId, etc.
        """
        return await self._get("/contract/find", params={"name": name})

    async def suggest_contracts(self, text: str, n_results: int = 5) -> List[Dict]:
        """Auto-suggest contracts by partial text."""
        return await self._get(
            "/contract/suggest",
            params={"t": text, "l": n_results},
        )

    async def get_contract_item(self, contract_id: int) -> Dict[str, Any]:
        """Get a contract by its ID."""
        return await self._get("/contract/item", params={"id": contract_id})

    async def get_product(self, product_id: int) -> Dict[str, Any]:
        """Get product details (point value, tick size, etc.)."""
        return await self._get("/product/item", params={"id": product_id})

    # ── Account ─────────────────────────────────────────────────────────

    async def list_accounts(self) -> List[Dict[str, Any]]:
        """List all accounts."""
        return await self._get("/account/list")

    async def get_cash_balance(self, account_id: int) -> Dict[str, Any]:
        """Get cash balance snapshot for an account."""
        return await self._get(
            "/cashBalance/getCashBalanceSnapshot",
            params={"accountId": account_id},
        )

    # ── Order Management ────────────────────────────────────────────────

    async def place_order(
        self,
        account_id: int,
        account_spec: str,
        symbol: str,
        action: str,
        order_qty: int,
        order_type: str,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "Day",
    ) -> Dict[str, Any]:
        """Place a single order.

        Args:
            action: "Buy" or "Sell"
            order_type: "Market", "Limit", "Stop", "StopLimit", etc.
            price: Required for Limit/StopLimit orders.
            stop_price: Required for Stop/StopLimit orders.
        """
        body: Dict[str, Any] = {
            "accountSpec": account_spec,
            "accountId": account_id,
            "action": action,
            "symbol": symbol,
            "orderQty": order_qty,
            "orderType": order_type,
            "timeInForce": time_in_force,
            "isAutomated": True,
        }
        if price is not None:
            body["price"] = price
        if stop_price is not None:
            body["stopPrice"] = stop_price

        return await self._post("/order/placeOrder", body)

    async def place_oco(
        self,
        account_id: int,
        account_spec: str,
        symbol: str,
        action: str,
        order_qty: int,
        order_type: str,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        other: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Place an OCO (One-Cancels-Other) order pair.

        The `other` dict describes the second leg of the OCO.
        """
        body: Dict[str, Any] = {
            "accountSpec": account_spec,
            "accountId": account_id,
            "action": action,
            "symbol": symbol,
            "orderQty": order_qty,
            "orderType": order_type,
            "isAutomated": True,
        }
        if price is not None:
            body["price"] = price
        if stop_price is not None:
            body["stopPrice"] = stop_price
        if other is not None:
            body["other"] = other

        return await self._post("/order/placeOCO", body)

    async def start_order_strategy(
        self,
        account_id: int,
        account_spec: str,
        symbol: str,
        action: str,
        order_qty: int,
        order_type: str,
        price: Optional[float] = None,
        brackets: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Place a bracket order via order strategy (orderStrategyTypeId: 2).

        This is the atomic bracket placement — entry + TP + SL as one unit.
        """
        import json as _json

        body: Dict[str, Any] = {
            "accountSpec": account_spec,
            "accountId": account_id,
            "action": action,
            "symbol": symbol,
            "orderQty": order_qty,
            "orderType": order_type,
            "orderStrategyTypeId": 2,
            "isAutomated": True,
        }
        if price is not None:
            body["price"] = price
        if brackets is not None:
            body["params"] = _json.dumps(brackets)

        return await self._post("/orderStrategy/startOrderStrategy", body)

    async def place_oso(
        self,
        account_id: int,
        account_spec: str,
        symbol: str,
        action: str,
        order_qty: int,
        order_type: str,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        bracket1: Optional[Dict] = None,
        bracket2: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Place an OSO (Order-Sends-Order) bracket atomically.

        Server places entry, then sends bracket1 + bracket2 on fill.
        If both brackets are specified they are linked as OCO.
        Response: {orderId, oso1Id, oso2Id}.
        """
        body: Dict[str, Any] = {
            "accountSpec": account_spec,
            "accountId": account_id,
            "action": action,
            "symbol": symbol,
            "orderQty": order_qty,
            "orderType": order_type,
            "isAutomated": True,
        }
        if price is not None:
            body["price"] = price
        if stop_price is not None:
            body["stopPrice"] = stop_price
        if bracket1 is not None:
            body["bracket1"] = bracket1
        if bracket2 is not None:
            body["bracket2"] = bracket2

        return await self._post("/order/placeOSO", body)

    async def cancel_order(self, order_id: int) -> Dict[str, Any]:
        """Cancel an order by its ID."""
        return await self._post("/order/cancelOrder", {"orderId": order_id})

    async def modify_order(
        self,
        order_id: int,
        order_qty: Optional[int] = None,
        order_type: Optional[str] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Modify an existing order.

        Tradovate requires orderType (and price for Limit orders) in every
        modify request. We fetch the current orderVersion to fill in any
        fields not explicitly provided.

        Returns the latest orderVersion after modification.
        """
        ver = await self._get("/orderVersion/item", {"id": order_id})

        body: Dict[str, Any] = {
            "orderId": order_id,
            "orderType": order_type or ver.get("orderType", "Limit"),
            "orderQty": order_qty if order_qty is not None else ver.get("orderQty"),
        }
        eff_type = body["orderType"]

        if eff_type in ("Limit", "StopLimit"):
            body["price"] = price if price is not None else ver.get("price")
        if eff_type in ("Stop", "StopLimit"):
            body["stopPrice"] = stop_price if stop_price is not None else ver.get("stopPrice")

        await self._post("/order/modifyOrder", body)
        return await self.get_latest_order_version(order_id)

    async def get_latest_order_version(self, order_id: int) -> Dict[str, Any]:
        """Get the most recent orderVersion for an order.

        Tradovate creates a new version (with new ID) on each modify.
        /orderVersion/item?id={orderId} always returns the FIRST version.
        """
        versions = await self._get("/orderVersion/deps", {"masterid": order_id})
        if isinstance(versions, list) and versions:
            return versions[-1]
        return await self._get("/orderVersion/item", {"id": order_id})

    async def liquidate_position(
        self, account_id: int, contract_id: int, admin: bool = False
    ) -> Dict[str, Any]:
        """Liquidate a position."""
        return await self._post(
            "/order/liquidatePosition",
            {"accountId": account_id, "contractId": contract_id, "admin": admin},
        )

    # ── Cleanup ─────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


class TradovateAPIError(Exception):
    """Raised when a Tradovate API call fails."""
