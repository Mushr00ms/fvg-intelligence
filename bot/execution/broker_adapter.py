"""
broker_adapter.py — Broker-agnostic interface (ABC) for execution backends.

All broker-specific code (IB, Tradovate, Binance) lives behind this interface.
The engine, order manager, and risk gates talk only to BrokerAdapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, List, Optional

from bot.execution.execution_types import (
    AccountSnapshot,
    OpenOrderSnapshot,
    PositionSnapshot,
)


# ---------------------------------------------------------------------------
# Broker-agnostic data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BarData:
    """Broker-agnostic bar. Timestamp is always ET-normalized by the adapter."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class TickData:
    """Broker-agnostic tick. Timestamp is always ET-normalized by the adapter."""
    timestamp: datetime
    price: float
    size: float = 0.0


@dataclass(frozen=True)
class ContractInfo:
    """Resolved instrument/contract."""
    symbol: str                   # e.g. "NQ"
    broker_contract_id: str       # IB conId or Tradovate contract id (as string)
    expiry: str                   # e.g. "20260619"
    exchange: str                 # e.g. "CME"
    tick_size: float = 0.25
    point_value: float = 20.0
    name: str = ""                # Tradovate full name e.g. "NQM6"


@dataclass
class BracketOrderResult:
    """Result of placing a bracket order."""
    success: bool
    entry_order_id: str = ""      # Broker-native ID as string
    tp_order_id: str = ""
    sl_order_id: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Abstract Broker Adapter
# ---------------------------------------------------------------------------

class BrokerAdapter(ABC):
    """Abstract broker interface. All broker-specific code lives behind this."""

    # ── Connection lifecycle ────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the broker. Authenticate, open sockets, etc."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully disconnect from the broker."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True if the broker connection is healthy."""

    @property
    @abstractmethod
    def disconnect_seconds(self) -> float:
        """Seconds since last disconnect, or 0.0 if connected."""

    @abstractmethod
    def on_reconnect(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register a callback to fire after reconnection."""

    # ── Contract resolution ─────────────────────────────────────────────

    @abstractmethod
    async def resolve_contract(
        self,
        symbol: str,
        exchange: str,
        expiry_hint: Optional[datetime] = None,
    ) -> ContractInfo:
        """Resolve a tradeable contract for the given symbol and approximate expiry."""

    # ── Market data ─────────────────────────────────────────────────────

    @abstractmethod
    async def subscribe_bars(
        self,
        contract: ContractInfo,
        bar_size: str,
        on_bar: Callable[[BarData, bool], Any],
    ) -> str:
        """Subscribe to bar data. on_bar(bar, has_new_bar) fires on updates.
        Returns a subscription_id for cleanup."""

    @abstractmethod
    async def unsubscribe_bars(self, subscription_id: str) -> None:
        """Cancel a bar subscription."""

    @abstractmethod
    async def subscribe_ticks(
        self,
        contract: ContractInfo,
        on_tick: Callable[[TickData], Any],
    ) -> Optional[str]:
        """Subscribe to tick-by-tick data. Returns subscription_id or None."""

    @abstractmethod
    async def unsubscribe_ticks(self, subscription_id: str) -> None:
        """Cancel a tick subscription."""

    # ── Order management ────────────────────────────────────────────────

    @abstractmethod
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
        on_exit_ids: Optional[Callable[[str, str], None]] = None,
    ) -> BracketOrderResult:
        """Place an atomic bracket order (entry + TP + SL).
        Callbacks fire on the asyncio event loop.
        on_exit_ids(tp_id, sl_id) fires when real exit order IDs are known."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> None:
        """Cancel a single order by its broker-native ID."""

    @abstractmethod
    async def modify_order_qty(self, order_id: str, new_qty: int) -> None:
        """Modify the quantity of an existing order."""

    @abstractmethod
    async def place_market_order(
        self,
        contract: ContractInfo,
        side: str,
        qty: int,
    ) -> str:
        """Place a market order (used for flatten). Returns order ID."""

    @abstractmethod
    async def get_open_trades(self) -> List[OpenOrderSnapshot]:
        """Return all open/working trades from the broker."""

    # ── Account data ────────────────────────────────────────────────────

    @abstractmethod
    async def get_account_balance(self) -> Optional[float]:
        """Return the account net liquidation value, or None if unavailable."""

    @abstractmethod
    async def get_positions(self) -> List[PositionSnapshot]:
        """Return current open positions from the broker."""

    @abstractmethod
    async def get_open_orders(self) -> List[OpenOrderSnapshot]:
        """Return currently open/working orders."""

    @abstractmethod
    async def get_margin_per_contract(
        self, contract: ContractInfo
    ) -> Optional[float]:
        """Return margin requirement per contract, or None if unsupported.
        Callers fall back to config-based margins on None."""

    @abstractmethod
    async def get_available_funds(self) -> float:
        """Return available funds for new orders."""

    # ── Time sync ───────────────────────────────────────────────────────

    @abstractmethod
    async def get_server_time(self) -> Optional[datetime]:
        """Return the broker's server time (UTC), or None if unsupported."""

    # ── Order ID pre-allocation ─────────────────────────────────────────

    @abstractmethod
    def allocate_order_ids(self, count: int) -> List[str]:
        """Pre-allocate broker-native order IDs for crash-safe state persistence.
        Returns list of string IDs."""

    # ── Optional bracket helpers (not all brokers need these) ──────────

    async def cancel_bracket_exits(self, entry_order_id: str) -> None:
        """Cancel TP + SL exits for a bracket. Default no-op (IB uses OCA)."""
