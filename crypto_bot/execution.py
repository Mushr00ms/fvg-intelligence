"""
execution.py — Live Binance order placement and event handling for the crypto bot.
"""

from __future__ import annotations

from bot.execution.binance_futures_client import BinanceFuturesClient, BinanceFuturesError

from crypto_bot.models import OrderIntent


class BinanceExecutionManager:
    def __init__(self, client: BinanceFuturesClient, config, logger):
        self._client = client
        self._config = config
        self._logger = logger

    async def place_entry(self, intent: OrderIntent):
        intent.entry_client_order_id = f"cb_{intent.group_id}_entry"
        payload = {
            "symbol": intent.symbol,
            "side": intent.side,
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": intent.quantity,
            "price": intent.entry_price,
            "newClientOrderId": intent.entry_client_order_id,
            "newOrderRespType": "RESULT",
        }
        if intent.position_side != "BOTH":
            payload["positionSide"] = intent.position_side
        ack = await self._client.create_order(**payload)
        intent.entry_order_id = ack.order_id
        intent.entry_client_order_id = ack.client_order_id or intent.entry_client_order_id
        intent.status = "SUBMITTED"
        self._logger.log(
            "crypto_order_submitted",
            group_id=intent.group_id,
            order_id=intent.entry_order_id,
            client_order_id=intent.entry_client_order_id,
            quantity=intent.quantity,
            entry_price=intent.entry_price,
            position_side=intent.position_side,
        )
        return ack

    async def cancel_entry_remainder(self, intent: OrderIntent):
        if not intent.entry_order_id and not intent.entry_client_order_id:
            return None
        try:
            return await self._client.cancel_order(
                intent.symbol,
                order_id=intent.entry_order_id or None,
                client_order_id=intent.entry_client_order_id or None,
            )
        except BinanceFuturesError as exc:
            self._logger.log("crypto_cancel_error", group_id=intent.group_id, error=str(exc))
            return None

    async def arm_exits(self, intent: OrderIntent, filled_qty: float):
        intent.tp_client_order_id = f"cb_{intent.group_id}_tp"
        intent.sl_client_order_id = f"cb_{intent.group_id}_sl"
        tp_payload = {
            "symbol": intent.symbol,
            "side": "SELL" if intent.side == "BUY" else "BUY",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": filled_qty,
            "price": intent.target_price,
            "workingType": "CONTRACT_PRICE",
            "newClientOrderId": intent.tp_client_order_id,
            "newOrderRespType": "RESULT",
        }
        if intent.position_side == "BOTH":
            tp_payload["reduceOnly"] = "true"
        if intent.position_side != "BOTH":
            tp_payload["positionSide"] = intent.position_side
        sl_payload = {
            "algoType": "CONDITIONAL",
            "symbol": intent.symbol,
            "side": "SELL" if intent.side == "BUY" else "BUY",
            "type": "STOP_MARKET",
            "quantity": filled_qty,
            "triggerPrice": intent.stop_price,
            "workingType": "CONTRACT_PRICE",
            "clientAlgoId": intent.sl_client_order_id,
            "newOrderRespType": "RESULT",
        }
        if intent.position_side == "BOTH":
            sl_payload["reduceOnly"] = "true"
        if intent.position_side != "BOTH":
            sl_payload["positionSide"] = intent.position_side
        sl_ack = await self._client.create_algo_order(**sl_payload)
        intent.sl_order_id = sl_ack.order_id
        intent.sl_client_order_id = sl_ack.client_order_id or intent.sl_client_order_id
        tp_ack = await self._client.create_order(**tp_payload)
        intent.tp_order_id = tp_ack.order_id
        intent.tp_client_order_id = tp_ack.client_order_id or intent.tp_client_order_id
        self._logger.log(
            "crypto_exit_armed",
            group_id=intent.group_id,
            tp_order_id=intent.tp_order_id,
            sl_order_id=intent.sl_order_id,
            filled_qty=filled_qty,
        )

    async def replace_exits(self, intent: OrderIntent, filled_qty: float):
        await self.cancel_exit_orders(intent)
        await self.arm_exits(intent, filled_qty)

    async def cancel_sibling_exit(self, intent: OrderIntent, *, filled_exit: str):
        sibling_order_id = intent.sl_order_id if filled_exit == "TP" else intent.tp_order_id
        sibling_client_order_id = intent.sl_client_order_id if filled_exit == "TP" else intent.tp_client_order_id
        if not sibling_order_id and not sibling_client_order_id:
            return
        try:
            if filled_exit == "TP":
                await self._client.cancel_algo_order(
                    algo_id=sibling_order_id or None,
                    client_algo_id=sibling_client_order_id or None,
                )
            else:
                await self._client.cancel_order(
                    intent.symbol,
                    order_id=sibling_order_id or None,
                    client_order_id=sibling_client_order_id or None,
                )
        except BinanceFuturesError as exc:
            self._logger.log(
                "crypto_cancel_sibling_error",
                group_id=intent.group_id,
                exit_reason=filled_exit,
                error=str(exc),
            )

    async def cancel_exit_orders(self, intent: OrderIntent):
        for order_id, client_order_id, leg in (
            (intent.tp_order_id, intent.tp_client_order_id, "tp"),
            (intent.sl_order_id, intent.sl_client_order_id, "sl"),
        ):
            if not order_id and not client_order_id:
                continue
            try:
                if leg == "sl":
                    await self._client.cancel_algo_order(
                        algo_id=order_id or None,
                        client_algo_id=client_order_id or None,
                    )
                else:
                    await self._client.cancel_order(
                        intent.symbol,
                        order_id=order_id or None,
                        client_order_id=client_order_id or None,
                    )
            except BinanceFuturesError as exc:
                if exc.code not in (-2011, -4047):
                    self._logger.log("crypto_cancel_error", group_id=intent.group_id, leg=leg, error=str(exc))
        intent.tp_order_id = ""
        intent.tp_client_order_id = ""
        intent.sl_order_id = ""
        intent.sl_client_order_id = ""
