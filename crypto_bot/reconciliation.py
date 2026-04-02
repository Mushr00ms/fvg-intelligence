"""
reconciliation.py - Exchange-driven state reconciliation for the crypto bot.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from bot.execution.binance_futures_client import BinanceFuturesClient, BinanceFuturesError
from bot.execution.execution_types import OpenOrderSnapshot, PositionSnapshot

from crypto_bot.models import OrderIntent, RuntimeState


RUNTIME_TZ = ZoneInfo("America/New_York")


def _iso_from_ms(ms: int | float | None) -> str:
    if not ms:
        return datetime.now(RUNTIME_TZ).isoformat()
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).astimezone(RUNTIME_TZ).isoformat()


def parse_client_order_id(client_order_id: str) -> tuple[str, str] | None:
    if not client_order_id.startswith("cb_"):
        return None
    parts = client_order_id.split("_")
    if len(parts) != 3:
        return None
    _, group_id, leg = parts
    if leg not in {"entry", "tp", "sl"}:
        return None
    return group_id, leg


@dataclass
class ReconciliationResult:
    resumed_groups: list[str] = field(default_factory=list)
    stale_groups: list[str] = field(default_factory=list)
    exit_repairs: list[str] = field(default_factory=list)
    unmanaged_order_ids: list[str] = field(default_factory=list)
    unmanaged_positions: list[str] = field(default_factory=list)


async def reconcile_runtime_state(
    state: RuntimeState,
    *,
    client: BinanceFuturesClient,
    symbol: str,
    leverage: int,
    resume_managed_positions: bool,
) -> ReconciliationResult:
    result = ReconciliationResult()
    open_orders = await client.get_open_orders(symbol)
    open_algo_orders = await client.get_open_algo_orders(symbol, algo_type="CONDITIONAL")
    positions = await client.get_positions(symbol)
    closed_by_group = {intent.group_id for intent in state.closed_trades}

    grouped_open_orders: dict[str, dict[str, OpenOrderSnapshot]] = defaultdict(dict)
    for order in open_orders:
        parsed = parse_client_order_id(order.client_order_id)
        if parsed is None:
            result.unmanaged_order_ids.append(order.client_order_id or order.order_id)
            continue
        group_id, leg = parsed
        grouped_open_orders[group_id][leg] = order
    for order in open_algo_orders:
        parsed = parse_client_order_id(order.client_order_id)
        if parsed is None:
            result.unmanaged_order_ids.append(order.client_order_id or order.order_id)
            continue
        group_id, leg = parsed
        grouped_open_orders[group_id][leg] = order

    known_by_group = {
        intent.group_id: intent
        for intent in state.pending_entries + state.open_positions
    }
    candidate_groups = (set(known_by_group) | set(grouped_open_orders)) - closed_by_group

    if not resume_managed_positions:
        for group_id in sorted(set(grouped_open_orders) - set(known_by_group)):
            for order in grouped_open_orders[group_id].values():
                result.unmanaged_order_ids.append(order.client_order_id or order.order_id)
        candidate_groups = set(known_by_group)

    remaining_position_by_side: dict[str, float] = defaultdict(float)
    for pos in positions:
        key = (pos.position_side or "BOTH").upper()
        remaining_position_by_side[key] += abs(pos.quantity)

    new_pending: list[OrderIntent] = []
    new_open: list[OrderIntent] = []

    for group_id in sorted(candidate_groups):
        intent = known_by_group.get(group_id)
        if intent is None:
            intent = await _reconstruct_intent(client, symbol, leverage, group_id, grouped_open_orders.get(group_id, {}))
            if intent is None:
                continue
            result.resumed_groups.append(group_id)

        await _sync_intent_from_exchange(
            intent,
            client=client,
            symbol=symbol,
            remaining_position_by_side=remaining_position_by_side,
            open_orders=grouped_open_orders.get(group_id, {}),
            exit_repairs=result.exit_repairs,
            stale_groups=result.stale_groups,
            new_pending=new_pending,
            new_open=new_open,
        )

    state.pending_entries = new_pending
    state.open_positions = new_open
    for group_id in result.stale_groups:
        intent = known_by_group.get(group_id)
        if intent is None or group_id in closed_by_group:
            continue
        intent.status = intent.status if intent.status == "CLOSED" else "RECONCILED_STALE"
        if not intent.closed_at:
            intent.closed_at = datetime.now(RUNTIME_TZ).isoformat()
        state.closed_trades.append(intent)

    expected_by_side: dict[str, float] = defaultdict(float)
    for intent in state.open_positions:
        key = (intent.position_side or "BOTH").upper()
        expected_by_side[key] += abs(intent.filled_qty or intent.quantity)
    for pos in positions:
        key = (pos.position_side or "BOTH").upper()
        exchange_qty = abs(pos.quantity)
        if exchange_qty > expected_by_side.get(key, 0.0) + 1e-6:
            result.unmanaged_positions.append(f"{key}:{exchange_qty}")

    return result


async def _sync_intent_from_exchange(
    intent: OrderIntent,
    *,
    client: BinanceFuturesClient,
    symbol: str,
    remaining_position_by_side: dict[str, float],
    open_orders: dict[str, OpenOrderSnapshot],
    exit_repairs: list[str],
    stale_groups: list[str],
    new_pending: list[OrderIntent],
    new_open: list[OrderIntent],
):
    entry_open = open_orders.get("entry")
    tp_open = open_orders.get("tp")
    sl_open = open_orders.get("sl")

    if entry_open is not None:
        intent.entry_order_id = entry_open.order_id
        intent.entry_client_order_id = entry_open.client_order_id
        intent.quantity = entry_open.quantity or intent.quantity
        intent.entry_price = entry_open.price or intent.entry_price
        intent.position_side = entry_open.position_side or intent.position_side
        intent.status = "SUBMITTED"
        new_pending.append(intent)
        return

    if tp_open is not None:
        intent.tp_order_id = tp_open.order_id
        intent.tp_client_order_id = tp_open.client_order_id
        intent.target_price = tp_open.price or intent.target_price
    if sl_open is not None:
        intent.sl_order_id = sl_open.order_id
        intent.sl_client_order_id = sl_open.client_order_id
        intent.stop_price = sl_open.stop_price or intent.stop_price

    entry_details = await _safe_get_order(
        client,
        symbol,
        intent.entry_client_order_id or f"cb_{intent.group_id}_entry",
    )
    if entry_details:
        intent.entry_order_id = str(entry_details.get("orderId", intent.entry_order_id))
        intent.entry_client_order_id = entry_details.get("clientOrderId", intent.entry_client_order_id)
        intent.quantity = float(entry_details.get("origQty", intent.quantity) or intent.quantity)
        intent.filled_qty = float(entry_details.get("executedQty", intent.filled_qty) or intent.filled_qty)
        avg_entry = float(entry_details.get("avgPrice", 0.0) or 0.0)
        price = float(entry_details.get("price", intent.entry_price) or intent.entry_price)
        intent.avg_entry_price = avg_entry or intent.avg_entry_price or price
        intent.entry_price = price or intent.entry_price
        intent.side = entry_details.get("side", intent.side)
        intent.position_side = entry_details.get("positionSide", intent.position_side or "BOTH")

    side_key = (intent.position_side or "BOTH").upper()
    open_qty = remaining_position_by_side.get(side_key, 0.0)
    if intent.filled_qty <= 0 and (tp_open or sl_open):
        intent.filled_qty = max(
            tp_open.quantity if tp_open else 0.0,
            sl_open.quantity if sl_open else 0.0,
            intent.quantity,
        )

    if open_qty > 1e-6 or tp_open is not None or sl_open is not None:
        allocated_qty = min(open_qty, intent.filled_qty or intent.quantity)
        if allocated_qty > 0:
            remaining_position_by_side[side_key] = max(0.0, open_qty - allocated_qty)
        intent.status = "OPEN"
        if not tp_open or not sl_open:
            exit_repairs.append(intent.group_id)
        new_open.append(intent)
        return

    if entry_details:
        intent.status = entry_details.get("status", intent.status or "RECONCILED_STALE")
        if not intent.closed_at and entry_details.get("updateTime"):
            intent.closed_at = _iso_from_ms(entry_details.get("updateTime"))
    stale_groups.append(intent.group_id)


async def _reconstruct_intent(
    client: BinanceFuturesClient,
    symbol: str,
    leverage: int,
    group_id: str,
    open_orders: dict[str, OpenOrderSnapshot],
) -> OrderIntent | None:
    entry_order = await _safe_get_order(client, symbol, f"cb_{group_id}_entry")
    if entry_order is None and "entry" not in open_orders:
        return None

    entry_side = (entry_order or {}).get("side") or (open_orders.get("entry").side if open_orders.get("entry") else "")
    position_side = (entry_order or {}).get("positionSide") or (
        open_orders.get("entry").position_side if open_orders.get("entry") else ""
    ) or "BOTH"
    entry_price = float((entry_order or {}).get("avgPrice", 0.0) or 0.0)
    if entry_price <= 0:
        entry_price = float((entry_order or {}).get("price", 0.0) or 0.0)
    if entry_price <= 0 and open_orders.get("entry") is not None:
        entry_price = open_orders["entry"].price

    quantity = float((entry_order or {}).get("origQty", 0.0) or 0.0)
    filled_qty = float((entry_order or {}).get("executedQty", 0.0) or 0.0)
    if quantity <= 0 and open_orders.get("entry") is not None:
        quantity = open_orders["entry"].quantity
    if filled_qty <= 0:
        filled_qty = max(
            open_orders["tp"].quantity if open_orders.get("tp") else 0.0,
            open_orders["sl"].quantity if open_orders.get("sl") else 0.0,
        )
    quantity = quantity or filled_qty
    if quantity <= 0 or entry_price <= 0:
        return None

    target_price = open_orders["tp"].price if open_orders.get("tp") else entry_price
    stop_price = open_orders["sl"].stop_price if open_orders.get("sl") else entry_price
    notional = quantity * entry_price
    initial_margin = notional / leverage if leverage > 0 else 0.0
    per_unit_loss = abs(entry_price - stop_price) if stop_price > 0 else 0.0
    per_unit_profit = abs(target_price - entry_price) if target_price > 0 else 0.0

    created_at = _iso_from_ms((entry_order or {}).get("time") or (entry_order or {}).get("updateTime"))
    return OrderIntent(
        group_id=group_id,
        fvg_id=f"resumed_{group_id}",
        symbol=symbol,
        setup="resumed",
        side=entry_side,
        position_side=position_side,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_bps=0.0,
        n_value=0.0,
        risk_dollar=0.0,
        quantity=quantity,
        notional=round(notional, 8),
        initial_margin=round(initial_margin, 8),
        expected_loss=round(quantity * per_unit_loss, 8),
        expected_profit=round(quantity * per_unit_profit, 8),
        created_at=created_at,
        status="OPEN" if open_orders.get("tp") or open_orders.get("sl") or filled_qty > 0 else "SUBMITTED",
        entry_order_id=str((entry_order or {}).get("orderId", "")),
        entry_client_order_id=(entry_order or {}).get("clientOrderId", f"cb_{group_id}_entry"),
        tp_order_id=open_orders["tp"].order_id if open_orders.get("tp") else "",
        tp_client_order_id=open_orders["tp"].client_order_id if open_orders.get("tp") else "",
        sl_order_id=open_orders["sl"].order_id if open_orders.get("sl") else "",
        sl_client_order_id=open_orders["sl"].client_order_id if open_orders.get("sl") else "",
        filled_qty=filled_qty,
        avg_entry_price=entry_price,
    )


async def _safe_get_order(client: BinanceFuturesClient, symbol: str, client_order_id: str) -> dict | None:
    try:
        return await client.get_order(symbol, client_order_id=client_order_id)
    except BinanceFuturesError as exc:
        if exc.code in (-2013, -2011):
            return None
        raise
