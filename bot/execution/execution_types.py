"""
execution_types.py — Broker-agnostic execution dataclasses.

These types are shared by broker clients/adapters so the engine can evolve away
from IB-specific payloads without baking Binance response shapes into core code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN


def _quantize_down(value: float, step: float, precision: int | None = None) -> float:
    """Round a positive value down to the nearest exchange step."""
    if step <= 0:
        return round(value, precision or 8)
    dec_value = Decimal(str(value))
    dec_step = Decimal(str(step))
    rounded = (dec_value / dec_step).to_integral_value(rounding=ROUND_DOWN) * dec_step
    if precision is not None:
        quant = Decimal("1").scaleb(-precision)
        rounded = rounded.quantize(quant, rounding=ROUND_DOWN)
    return float(rounded)


@dataclass(frozen=True)
class SymbolRules:
    """Exchange trading rules needed to build valid orders."""

    symbol: str
    price_tick_size: float
    quantity_step_size: float
    min_quantity: float
    min_notional: float = 0.0
    price_precision: int | None = None
    quantity_precision: int | None = None
    trigger_protect: float = 0.0
    contract_type: str = ""
    base_asset: str = ""
    quote_asset: str = ""
    margin_asset: str = ""

    def round_price(self, price: float) -> float:
        return _quantize_down(price, self.price_tick_size, self.price_precision)

    def round_quantity(self, quantity: float) -> float:
        return _quantize_down(quantity, self.quantity_step_size, self.quantity_precision)

    def clamp_quantity(self, quantity: float) -> float:
        rounded = self.round_quantity(quantity)
        if rounded < self.min_quantity:
            return 0.0
        return rounded

    def validate_notional(self, quantity: float, price: float) -> bool:
        if self.min_notional <= 0:
            return True
        return quantity * price >= self.min_notional


@dataclass(frozen=True)
class BrokerOrderAck:
    """Normalized order acknowledgement from a broker."""

    broker: str
    symbol: str
    side: str
    order_type: str
    status: str
    order_id: str
    client_order_id: str
    quantity: float
    price: float = 0.0
    stop_price: float = 0.0
    position_side: str = ""
    reduce_only: bool = False
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AccountSnapshot:
    """Normalized account balances for sizing and risk checks."""

    broker: str
    wallet_balance: float
    available_balance: float
    margin_balance: float
    unrealized_pnl: float = 0.0
    initial_margin: float = 0.0
    maintenance_margin: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PositionSnapshot:
    """Normalized open-position view."""

    broker: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    mark_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: int = 0
    margin_type: str = ""
    position_side: str = ""
    liquidation_price: float = 0.0
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OpenOrderSnapshot:
    """Normalized open order view."""

    broker: str
    symbol: str
    side: str
    order_type: str
    status: str
    quantity: float
    price: float = 0.0
    stop_price: float = 0.0
    order_id: str = ""
    client_order_id: str = ""
    position_side: str = ""
    reduce_only: bool = False
    raw: dict = field(default_factory=dict)
