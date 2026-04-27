"""
split_adapter.py -- Composite adapter: IB for market data, Tradovate for execution.

Routes market data calls (bars, ticks, quotes) to the IB adapter and order/account
calls (place_bracket_order, get_positions, etc.) to the Tradovate adapter.
Both adapters connect and disconnect together.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, List, Optional

from bot.execution.broker_adapter import (
    BarData,
    BracketOrderResult,
    BrokerAdapter,
    ContractInfo,
    TickData,
)
from bot.execution.execution_types import OpenOrderSnapshot, PositionSnapshot

logger = logging.getLogger(__name__)


class SplitAdapter(BrokerAdapter):
    """IB data + Tradovate execution composite adapter."""

    def __init__(self, data_adapter: BrokerAdapter, exec_adapter: BrokerAdapter, bot_logger=None):
        self._data = data_adapter
        self._exec = exec_adapter
        self._bot_logger = bot_logger
        self._data_contract: Optional[ContractInfo] = None
        self._exec_contract: Optional[ContractInfo] = None

    # ── Connection ─────────────────────────────────────────────────

    async def connect(self) -> None:
        await self._data.connect()
        await self._exec.connect()
        logger.info("SplitAdapter connected: IB (data) + Tradovate (exec)")

    async def disconnect(self) -> None:
        await self._exec.disconnect()
        await self._data.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._data.is_connected and self._exec.is_connected

    @property
    def is_exec_connected(self) -> bool:
        """True if the execution adapter (Tradovate) is connected.

        Cancel/flatten must use this — NOT is_connected — so that a data-only
        disconnect doesn't prevent protective broker actions on live positions.
        """
        return self._exec.is_connected

    @property
    def disconnect_seconds(self) -> float:
        return max(self._data.disconnect_seconds, self._exec.disconnect_seconds)

    def connection_status(self) -> dict:
        """Return per-leg connection status for logging and alerts."""
        return {
            "data": {
                "broker": "IBKR",
                "connected": self._data.is_connected,
                "disconnect_seconds": self._data.disconnect_seconds,
            },
            "execution": {
                "broker": "Tradovate",
                "connected": self._exec.is_connected,
                "disconnect_seconds": self._exec.disconnect_seconds,
            },
        }

    def on_reconnect(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._data.on_reconnect(callback)

    def on_exec_reconnect(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register a callback for execution-side reconnects only."""
        self._exec.on_reconnect(callback)

    # ── Contract resolution ────────────────────────────────────────

    async def resolve_contract(
        self,
        symbol: str,
        exchange: str,
        expiry_hint: Optional[datetime] = None,
    ) -> ContractInfo:
        # IB is the source of truth — it provides market data
        self._data_contract = await self._data.resolve_contract(symbol, exchange, expiry_hint)

        # Drive Tradovate resolution from IB's resolved expiry so both
        # adapters are guaranteed to trade the same contract month
        ib_expiry = self._data_contract.expiry
        tv_hint = _parse_expiry_hint(ib_expiry) if ib_expiry else expiry_hint
        self._exec_contract = await self._exec.resolve_contract(symbol, exchange, tv_hint)

        # Sanity check — should never diverge since Tradovate was driven by IB
        data_exp = ib_expiry[:6] if ib_expiry else ""
        exec_exp = self._exec_contract.expiry[:6] if self._exec_contract.expiry else ""
        if data_exp and exec_exp and data_exp != exec_exp:
            raise RuntimeError(
                f"Contract month mismatch: IB data={ib_expiry} "
                f"vs Tradovate exec={self._exec_contract.expiry}. "
                f"Cannot trade on mismatched contracts near rollover."
            )

        if self._bot_logger:
            self._bot_logger.log(
                "split_contracts_resolved",
                source="split",
                source_of_truth="IBKR",
                ib_symbol=self._data_contract.symbol,
                ib_expiry=ib_expiry,
                ib_conId=self._data_contract.broker_contract_id,
                tv_name=self._exec_contract.name,
                tv_expiry=self._exec_contract.expiry,
                tv_contractId=self._exec_contract.broker_contract_id,
            )

        return self._data_contract

    @property
    def data_contract(self) -> Optional[ContractInfo]:
        return self._data_contract

    @property
    def exec_contract(self) -> Optional[ContractInfo]:
        return self._exec_contract

    # ── IB passthrough (for legacy engine paths) ───────────────────

    @property
    def ib_connection(self):
        return getattr(self._data, "ib_connection", None)

    async def _get_ib_contract(self, contract_info: ContractInfo):
        fn = getattr(self._data, "_get_ib_contract", None)
        if fn and self._data_contract:
            return await fn(self._data_contract)
        return None

    # ── Market data → IB ───────────────────────────────────────────

    async def subscribe_bars(
        self, contract: ContractInfo, bar_size: str,
        on_bar: Callable[[BarData, bool], Any],
    ) -> str:
        return await self._data.subscribe_bars(self._data_contract or contract, bar_size, on_bar)

    async def unsubscribe_bars(self, subscription_id: str) -> None:
        await self._data.unsubscribe_bars(subscription_id)

    async def subscribe_ticks(
        self, contract: ContractInfo,
        on_tick: Callable[[TickData], Any],
    ) -> Optional[str]:
        return await self._data.subscribe_ticks(self._data_contract or contract, on_tick)

    async def unsubscribe_ticks(self, subscription_id: str) -> None:
        await self._data.unsubscribe_ticks(subscription_id)

    # ── Orders → Tradovate ─────────────────────────────────────────

    async def place_bracket_order(
        self, contract: ContractInfo, side: str, qty: int,
        entry_price: float, tp_price: float, sl_price: float,
        on_entry_fill: Callable, on_tp_fill: Callable,
        on_sl_fill: Callable, on_status_change: Callable,
        on_exit_ids: Optional[Callable] = None,
    ) -> BracketOrderResult:
        return await self._exec.place_bracket_order(
            self._exec_contract or contract, side, qty,
            entry_price, tp_price, sl_price,
            on_entry_fill, on_tp_fill, on_sl_fill, on_status_change,
            on_exit_ids=on_exit_ids,
        )

    async def cancel_order(self, order_id: str) -> None:
        await self._exec.cancel_order(order_id)

    async def modify_order_qty(self, order_id: str, new_qty: int) -> None:
        await self._exec.modify_order_qty(order_id, new_qty)

    async def place_market_order(self, contract: ContractInfo, side: str, qty: int) -> str:
        return await self._exec.place_market_order(self._exec_contract or contract, side, qty)

    async def get_open_trades(self) -> List[OpenOrderSnapshot]:
        return await self._exec.get_open_trades()

    async def cancel_bracket_exits(self, entry_order_id: str) -> None:
        await self._exec.cancel_bracket_exits(entry_order_id)

    # ── Account → Tradovate ────────────────────────────────────────

    async def get_account_balance(self) -> Optional[float]:
        return await self._exec.get_account_balance()

    async def get_positions(self) -> List[PositionSnapshot]:
        return await self._exec.get_positions()

    async def get_open_orders(self) -> List[OpenOrderSnapshot]:
        return await self._exec.get_open_orders()

    async def get_margin_per_contract(self, contract: ContractInfo) -> Optional[float]:
        return await self._exec.get_margin_per_contract(self._exec_contract or contract)

    async def get_available_funds(self) -> float:
        return await self._exec.get_available_funds()

    # ── Time → IB ──────────────────────────────────────────────────

    async def get_server_time(self) -> Optional[datetime]:
        return await self._data.get_server_time()

    # ── Order IDs → Tradovate ──────────────────────────────────────

    def allocate_order_ids(self, count: int) -> List[str]:
        return self._exec.allocate_order_ids(count)


def _parse_expiry_hint(ib_expiry: str) -> datetime:
    """Convert IB expiry string (YYYYMM or YYYYMMDD) to a datetime hint.

    Returns the 1st of the expiry month — safely before any quarterly expiry
    so the Tradovate adapter resolves to the same contract.
    """
    year = int(ib_expiry[:4])
    month = int(ib_expiry[4:6])
    return datetime(year, month, 1)
