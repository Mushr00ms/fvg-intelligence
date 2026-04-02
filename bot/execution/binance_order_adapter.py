"""
binance_order_adapter.py — Translate internal OrderGroup objects into Binance
USDⓈ-M Futures order payloads.

Binance UM Futures does not provide a native three-leg bracket equivalent to
IB's parent/child bracket workflow. The safe live pattern is:

1. Place entry LIMIT order
2. Wait for entry fill event on the user-data stream
3. Place reduce-only TP + SL exit orders for the filled quantity
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bot.execution.execution_types import SymbolRules
from bot.state.trade_state import OrderGroup


@dataclass(frozen=True)
class BinanceBracketPlan:
    """Executable Binance Futures order plan for one OrderGroup."""

    symbol: str
    leverage: int
    margin_type: str
    entry_order: dict
    take_profit_order: dict
    stop_loss_order: dict
    activate_exits_on_fill: bool = True
    metadata: dict = field(default_factory=dict)


class BinanceOrderAdapter:
    """Build Binance order payloads from internal order groups."""

    def __init__(
        self,
        symbol: str,
        symbol_rules: SymbolRules,
        *,
        leverage: int = 10,
        margin_type: str = "CROSSED",
        working_type: str = "CONTRACT_PRICE",
    ):
        self.symbol = symbol
        self.rules = symbol_rules
        self.leverage = int(leverage)
        self.margin_type = margin_type.upper()
        self.working_type = working_type

    @staticmethod
    def _reverse_side(side: str) -> str:
        return "SELL" if side == "BUY" else "BUY"

    @staticmethod
    def _client_order_id(group_id: str, suffix: str) -> str:
        raw = f"fvg_{group_id}_{suffix}"
        return raw[:36]

    def _normalized_quantity(self, quantity: float) -> float:
        qty = self.rules.clamp_quantity(quantity)
        if qty <= 0:
            raise ValueError(
                f"Quantity {quantity} rounds below min quantity for {self.symbol}"
            )
        return qty

    def _validate_notional(self, quantity: float, price: float, context: str):
        if not self.rules.validate_notional(quantity, price):
            raise ValueError(
                f"{context} notional below exchange minimum for {self.symbol}: "
                f"qty={quantity} price={price} min_notional={self.rules.min_notional}"
            )

    def build_entry_order(self, order_group: OrderGroup, *, quantity: float | None = None) -> dict:
        qty = self._normalized_quantity(quantity or float(order_group.target_qty))
        price = self.rules.round_price(order_group.entry_price)
        self._validate_notional(qty, price, "Entry order")
        return {
            "symbol": self.symbol,
            "side": order_group.side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": qty,
            "price": price,
            "newClientOrderId": self._client_order_id(order_group.group_id, "entry"),
            "newOrderRespType": "RESULT",
        }

    def build_take_profit_order(self, order_group: OrderGroup, *, quantity: float) -> dict:
        qty = self._normalized_quantity(quantity)
        price = self.rules.round_price(order_group.target_price)
        self._validate_notional(qty, price, "Take-profit order")
        return {
            "symbol": self.symbol,
            "side": self._reverse_side(order_group.side),
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": qty,
            "price": price,
            "reduceOnly": "true",
            "workingType": self.working_type,
            "newClientOrderId": self._client_order_id(order_group.group_id, "tp"),
            "newOrderRespType": "RESULT",
        }

    def build_stop_loss_order(self, order_group: OrderGroup, *, quantity: float) -> dict:
        qty = self._normalized_quantity(quantity)
        stop_price = self.rules.round_price(order_group.stop_price)
        return {
            "symbol": self.symbol,
            "side": self._reverse_side(order_group.side),
            "type": "STOP_MARKET",
            "quantity": qty,
            "stopPrice": stop_price,
            "reduceOnly": "true",
            "workingType": self.working_type,
            "newClientOrderId": self._client_order_id(order_group.group_id, "sl"),
            "newOrderRespType": "RESULT",
        }

    def build_bracket_plan(self, order_group: OrderGroup, *, quantity: float | None = None) -> BinanceBracketPlan:
        qty = self._normalized_quantity(quantity or float(order_group.target_qty))
        entry = self.build_entry_order(order_group, quantity=qty)
        tp = self.build_take_profit_order(order_group, quantity=qty)
        sl = self.build_stop_loss_order(order_group, quantity=qty)

        entry_notional = qty * entry["price"]
        stop_notional = qty * sl["stopPrice"]
        worst_case_loss = abs(entry["price"] - sl["stopPrice"]) * qty

        return BinanceBracketPlan(
            symbol=self.symbol,
            leverage=self.leverage,
            margin_type=self.margin_type,
            entry_order=entry,
            take_profit_order=tp,
            stop_loss_order=sl,
            metadata={
                "entry_notional": round(entry_notional, 8),
                "stop_notional": round(stop_notional, 8),
                "worst_case_price_move": round(worst_case_loss, 8),
                "group_id": order_group.group_id,
            },
        )
