"""
order_manager.py — Bracket order placement, modification, and lifecycle management.

Handles:
- Crash-safe bracket order placement (pre-allocate IDs, save state before placing)
- Partial fill detection and TP/SL qty adjustment
- 5-minute timer for cancelling unfilled entry remainder
- Flatten all (market-order close)
- Cancel all pending entries

Supports two modes:
- Legacy IB mode: direct ib_async calls (when passed an IBConnection)
- Adapter mode: BrokerAdapter interface (IB, Tradovate, SplitAdapter)
"""

import asyncio
from datetime import datetime

import pytz

from bot.state.trade_state import OrderGroup, CLOSE_TP, CLOSE_SL, CLOSE_FLATTEN, CLOSE_CANCEL, CLOSE_EOD, CLOSE_REJECTED
from bot.strategy.trade_calculator import POINT_VALUE

NY_TZ = pytz.timezone("America/New_York")


class OrderManager:
    """
    Manages bracket order lifecycle.

    Accepts either an IBConnection (legacy) or a BrokerAdapter (new).
    Detects mode at init time based on the object passed.
    """

    def __init__(self, connection, contract, state_manager, logger, config, clock=None, db=None,
                 on_kill_switch=None):
        self._contract = contract
        self._state_mgr = state_manager
        self._logger = logger
        self._config = config
        self._clock = clock
        self._db = db
        self._on_kill_switch = on_kill_switch
        self._on_order_resolved = None
        self._partial_timers = {}
        self._adjustment_lock = asyncio.Lock()
        self._loop = None

        # Detect mode: BrokerAdapter or legacy IBConnection
        from bot.execution.broker_adapter import BrokerAdapter
        if isinstance(connection, BrokerAdapter):
            self._adapter = connection
            self._conn = None
            self._use_adapter = True
        else:
            self._adapter = None
            self._conn = connection
            self._use_adapter = False

    def _now_iso(self):
        if self._clock is not None:
            return self._clock.now().isoformat()
        return datetime.now(NY_TZ).isoformat()

    def _safe_callback(self, fn, *args):
        """Marshal a callback into the asyncio event loop (thread-safe)."""
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(fn, *args)
        else:
            fn(*args)

    @property
    def _is_broker_connected(self):
        if self._use_adapter:
            return self._adapter.is_connected
        return self._conn.is_connected if self._conn else False

    @property
    def _is_exec_connected(self):
        """Execution-side connectivity for cancel/flatten (ignores data adapter state)."""
        if self._use_adapter:
            return getattr(self._adapter, 'is_exec_connected', self._adapter.is_connected)
        return self._conn.is_connected if self._conn else False

    async def place_bracket(self, order_group, daily_state):
        """Place a bracket order. Increments trade_count."""
        og = await self._place_bracket_internal(order_group, daily_state)
        if og.state != "CLOSED":
            daily_state.trade_count += 1
            self._state_mgr.save(daily_state)
        return og

    async def _place_bracket_internal(self, order_group, daily_state):
        """Internal bracket placement. Does NOT increment trade_count."""
        og = order_group

        if self._config.dry_run:
            og.state = "SUBMITTED"
            og.submitted_at = self._now_iso()
            daily_state.pending_orders.append(og)
            self._logger.log(
                "order_placed", mode="DRY_RUN", group_id=og.group_id,
                setup=og.setup, side=og.side, entry=og.entry_price,
                stop=og.stop_price, target=og.target_price,
                risk_pts=og.risk_pts, n=og.n_value, contracts=og.target_qty,
            )
            return og

        if self._use_adapter:
            return await self._place_bracket_adapter(og, daily_state)
        else:
            return await self._place_bracket_ib(og, daily_state)

    # ── Adapter-mode bracket placement ─────────────────────────────

    async def _place_bracket_adapter(self, og, daily_state):
        """Place bracket through BrokerAdapter interface."""
        # Step 1: Pre-allocate IDs
        ids = self._adapter.allocate_order_ids(3)
        og.broker_entry_order_id = ids[0]
        og.broker_tp_order_id = ids[1]
        og.broker_sl_order_id = ids[2]

        # Step 2: Save state BEFORE placing
        og.state = "SUBMITTED"
        og.submitted_at = self._now_iso()
        daily_state.pending_orders.append(og)
        self._state_mgr.save(daily_state)

        # Step 3: Place via adapter
        from bot.execution.broker_adapter import ContractInfo
        contract_info = ContractInfo(
            symbol=self._config.ticker, broker_contract_id="",
            expiry="", exchange=self._config.exchange,
            tick_size=self._config.tick_size, point_value=self._config.point_value,
        )

        result = await self._adapter.place_bracket_order(
            contract=contract_info,
            side=og.side,
            qty=og.target_qty,
            entry_price=og.entry_price,
            tp_price=og.target_price,
            sl_price=og.stop_price,
            on_entry_fill=lambda order: self._on_entry_fill_adapter(order, og, daily_state),
            on_tp_fill=lambda order: self._on_tp_fill_adapter(order, og, daily_state),
            on_sl_fill=lambda order: self._on_sl_fill_adapter(order, og, daily_state),
            on_status_change=lambda order: self._on_status_adapter(order, og, daily_state),
            on_exit_ids=lambda tp_id, sl_id: self._on_exit_ids_resolved(tp_id, sl_id, og, daily_state),
        )

        if not result.success:
            self._logger.log("order_placement_failed", group_id=og.group_id, error=result.error)
            daily_state.move_to_closed(og.group_id, CLOSE_REJECTED)
            self._state_mgr.save(daily_state)
            return og

        # Update with real broker IDs
        og.broker_entry_order_id = result.entry_order_id
        og.broker_tp_order_id = result.tp_order_id
        og.broker_sl_order_id = result.sl_order_id
        self._state_mgr.save(daily_state)

        self._logger.log(
            "order_placed", mode="LIVE", group_id=og.group_id,
            setup=og.setup, side=og.side, entry=og.entry_price,
            stop=og.stop_price, target=og.target_price,
            risk_pts=og.risk_pts, n=og.n_value, contracts=og.target_qty,
            broker_entry_id=og.broker_entry_order_id,
        )
        return og

    def _on_exit_ids_resolved(self, tp_id, sl_id, og, daily_state):
        """Persist real Tradovate TP/SL order IDs back to state."""
        og.broker_tp_order_id = tp_id
        og.broker_sl_order_id = sl_id
        self._state_mgr.save(daily_state)

    def _on_entry_fill_adapter(self, order, og, daily_state):
        """Handle entry fill from adapter (Tradovate/split mode)."""
        og.filled_qty = og.target_qty
        og.filled_at = self._now_iso()
        og.actual_entry_price = og.entry_price
        og.entry_commission = 0.0
        daily_state.move_to_open(og.group_id)
        self._notify_order_resolved(released_qty=og.target_qty)
        self._state_mgr.save(daily_state)
        self._logger.log(
            "order_filled", group_id=og.group_id, setup=og.setup,
            side=og.side, qty=og.filled_qty, expected_price=og.entry_price,
            avg_price=og.entry_price, slippage_pts=0,
            risk_pts=og.risk_pts, target_price=og.target_price,
            stop_price=og.stop_price, n_value=og.n_value,
        )

    def _on_tp_fill_adapter(self, order, og, daily_state):
        """Handle TP fill from adapter."""
        if og.close_reason:
            return
        og.close_reason = CLOSE_TP
        self._cleanup_entry_remainder(og)

        qty = og.filled_qty or og.target_qty
        if og.side == "BUY":
            pnl_pts = og.target_price - og.entry_price
        else:
            pnl_pts = og.entry_price - og.target_price
        gross_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
        commission = self._estimate_commission(qty)
        net_pnl = round(gross_pnl - commission, 2)

        og.actual_exit_price = og.target_price
        daily_state.move_to_closed(og.group_id, CLOSE_TP, net_pnl)

        self._logger.log(
            "tp_filled", group_id=og.group_id, setup=og.setup,
            side=og.side, entry_price=og.entry_price, exit_price=og.target_price,
            qty=qty, pnl_pts=round(pnl_pts, 2), net_pnl=net_pnl,
            daily_realized=daily_state.realized_pnl,
        )
        self._write_exit_db(og, og.target_price, "TP", pnl_pts, gross_pnl, commission, net_pnl, daily_state)
        self._check_kill_switch(daily_state, "tp_fill")
        self._state_mgr.save(daily_state)
        self._notify_order_resolved(released_qty=0)

    def _on_sl_fill_adapter(self, order, og, daily_state):
        """Handle SL fill from adapter."""
        if og.close_reason:
            return
        og.close_reason = CLOSE_SL
        self._cleanup_entry_remainder(og)

        qty = og.filled_qty or og.target_qty
        if og.side == "BUY":
            pnl_pts = og.stop_price - og.entry_price
        else:
            pnl_pts = og.entry_price - og.stop_price
        gross_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
        commission = self._estimate_commission(qty)
        net_pnl = round(gross_pnl - commission, 2)

        og.actual_exit_price = og.stop_price
        daily_state.move_to_closed(og.group_id, CLOSE_SL, net_pnl)

        self._logger.log(
            "sl_filled", group_id=og.group_id, setup=og.setup,
            side=og.side, entry_price=og.entry_price, stop_price=og.stop_price,
            exit_price=og.stop_price, qty=qty, pnl_pts=round(pnl_pts, 2),
            net_pnl=net_pnl, daily_realized=daily_state.realized_pnl,
        )
        self._write_exit_db(og, og.stop_price, "SL", pnl_pts, gross_pnl, commission, net_pnl, daily_state)
        self._check_kill_switch(daily_state, "sl_fill")
        self._state_mgr.save(daily_state)
        self._notify_order_resolved(released_qty=0)

    def _on_status_adapter(self, order, og, daily_state):
        """Handle order status changes from adapter."""
        status = order.get("ordStatus", "") if isinstance(order, dict) else ""
        if status in ("Cancelled", "Rejected") and og.state == "SUBMITTED" and og.filled_qty == 0:
            daily_state.move_to_closed(og.group_id, CLOSE_CANCEL)
            self._state_mgr.save(daily_state)

    def _estimate_commission(self, qty):
        """Estimate round-trip commission for adapter mode (no IB fill data)."""
        return round(qty * 2.88 * 2, 2)

    # ── Legacy IB bracket placement ────────────────────────────────

    async def _place_bracket_ib(self, og, daily_state):
        """Place bracket through direct IB connection (legacy path)."""
        ib = self._conn.ib

        og.broker_entry_order_id = str(ib.client.getReqId())
        og.broker_tp_order_id = str(ib.client.getReqId())
        og.broker_sl_order_id = str(ib.client.getReqId())

        og.state = "SUBMITTED"
        og.submitted_at = self._now_iso()
        daily_state.pending_orders.append(og)
        self._state_mgr.save(daily_state)

        from ib_async import LimitOrder, StopOrder

        reverse_side = "SELL" if og.side == "BUY" else "BUY"
        entry_ib_id = int(og.broker_entry_order_id)
        tp_ib_id = int(og.broker_tp_order_id)
        sl_ib_id = int(og.broker_sl_order_id)

        parent = LimitOrder(
            action=og.side, totalQuantity=og.target_qty, lmtPrice=og.entry_price,
            orderId=entry_ib_id, tif='DAY', transmit=False,
        )
        tp = LimitOrder(
            action=reverse_side, totalQuantity=og.target_qty, lmtPrice=og.target_price,
            orderId=tp_ib_id, parentId=entry_ib_id, tif='DAY', transmit=False,
        )
        sl = StopOrder(
            action=reverse_side, totalQuantity=og.target_qty, stopPrice=og.stop_price,
            orderId=sl_ib_id, parentId=entry_ib_id, tif='DAY', transmit=True,
        )

        entry_trade = tp_trade = sl_trade = None
        try:
            entry_trade = ib.placeOrder(self._contract, parent)
            tp_trade = ib.placeOrder(self._contract, tp)
            sl_trade = ib.placeOrder(self._contract, sl)
        except Exception as e:
            self._logger.log(
                "order_placement_failed", group_id=og.group_id, error=str(e),
                entry_placed=entry_trade is not None, tp_placed=tp_trade is not None,
            )
            for placed_order in (parent, tp, sl):
                try:
                    if placed_order.orderId in {t.order.orderId for t in ib.openTrades()}:
                        ib.cancelOrder(placed_order)
                except Exception:
                    pass
            daily_state.move_to_closed(og.group_id, CLOSE_REJECTED)
            self._state_mgr.save(daily_state)
            return og

        self._loop = asyncio.get_event_loop()

        entry_trade.filledEvent += lambda trade: self._safe_callback(
            self._on_entry_fill, trade, og, daily_state)
        entry_trade.statusEvent += lambda trade: self._safe_callback(
            self._on_entry_status, trade, og, daily_state)
        tp_trade.filledEvent += lambda trade: self._safe_callback(
            self._on_tp_fill, trade, og, daily_state)
        sl_trade.filledEvent += lambda trade: self._safe_callback(
            self._on_sl_fill, trade, og, daily_state)

        self._logger.log(
            "order_placed", mode="LIVE", group_id=og.group_id,
            setup=og.setup, side=og.side, entry=og.entry_price,
            stop=og.stop_price, target=og.target_price,
            risk_pts=og.risk_pts, n=og.n_value, contracts=og.target_qty,
            broker_entry_id=og.broker_entry_order_id,
        )
        return og

    # ── Margin Management: Suspend / Reactivate ────────────────────

    async def suspend_order(self, og, daily_state, reason="margin_priority"):
        """Cancel orders for a SUBMITTED order and move to SUSPENDED state."""
        if og.state != "SUBMITTED" or og.filled_qty > 0:
            self._logger.log("suspend_refused", group_id=og.group_id,
                             state=og.state, filled=og.filled_qty)
            return

        if not self._config.dry_run and self._is_exec_connected:
            if og.broker_entry_order_id:
                await self._cancel_single_order(og.broker_entry_order_id)

        result = daily_state.move_to_suspended(og.group_id, reason)
        if result is None:
            self._logger.log("suspend_race", group_id=og.group_id,
                             reason="order not found in pending")
            return

        self._logger.log("order_suspended", group_id=og.group_id,
                         entry=og.entry_price, qty=og.target_qty, reason=reason)
        self._state_mgr.save(daily_state)

    async def reactivate_order(self, og, daily_state):
        """Re-place a SUSPENDED order."""
        reactivated = daily_state.move_suspended_to_pending(og.group_id)
        if reactivated is None:
            return None

        reactivated.ib_entry_order_id = None
        reactivated.ib_tp_order_id = None
        reactivated.ib_sl_order_id = None

        result = await self._place_bracket_internal(reactivated, daily_state)
        self._logger.log("order_reactivated", group_id=og.group_id,
                         entry=og.entry_price, qty=reactivated.target_qty)
        return result

    def _notify_order_resolved(self, released_qty: int = 0):
        if self._on_order_resolved:
            asyncio.ensure_future(self._on_order_resolved(released_qty))

    # ── IB fill callbacks (legacy) ─────────────────────────────────

    def _on_entry_fill(self, trade, og, daily_state):
        filled = trade.orderStatus.filled
        if filled <= 0:
            return

        if filled >= og.target_qty:
            og.filled_qty = og.target_qty
            og.filled_at = self._now_iso()
            avg_fill = trade.orderStatus.avgFillPrice
            slippage = round(abs(avg_fill - og.entry_price), 2)
            og.actual_entry_price = avg_fill
            og.entry_slippage_pts = slippage
            og.entry_commission = self._get_commission(trade)
            daily_state.move_to_open(og.group_id)
            self._notify_order_resolved(released_qty=og.target_qty)
            self._state_mgr.save(daily_state)
            self._logger.log(
                "order_filled", group_id=og.group_id, setup=og.setup,
                side=og.side, qty=og.filled_qty, expected_price=og.entry_price,
                avg_price=avg_fill, slippage_pts=slippage, risk_pts=og.risk_pts,
                target_price=og.target_price, stop_price=og.stop_price,
                n_value=og.n_value,
            )
        else:
            self._handle_partial_fill(trade, og, daily_state, filled)

    def _handle_partial_fill(self, trade, og, daily_state, filled_qty):
        prior_filled = og.filled_qty or 0
        og.filled_qty = filled_qty
        og.state = "PARTIAL"
        og.partial_fill_timer_start = self._now_iso()

        new_increment = filled_qty - prior_filled
        if new_increment > 0:
            self._notify_order_resolved(released_qty=new_increment)

        self._logger.log("partial_fill", group_id=og.group_id,
                         filled=filled_qty, target=og.target_qty,
                         remaining=og.target_qty - filled_qty)

        if not self._config.dry_run and self._is_broker_connected:
            asyncio.ensure_future(self._locked_adjust_child_qty(og, filled_qty))

        if og.group_id not in self._partial_timers:
            timeout = self._config.partial_fill_timeout
            task = asyncio.ensure_future(
                self._partial_fill_timeout(og.group_id, daily_state, timeout))
            self._partial_timers[og.group_id] = task

        self._state_mgr.save(daily_state)

    async def _locked_adjust_child_qty(self, og, new_qty):
        async with self._adjustment_lock:
            await self._adjust_child_qty(og, new_qty)

    async def _adjust_child_qty(self, og, new_qty):
        if self._use_adapter:
            for broker_id in (og.broker_tp_order_id, og.broker_sl_order_id):
                if broker_id:
                    try:
                        await self._adapter.modify_order_qty(broker_id, new_qty)
                    except Exception as e:
                        self._logger.log("partial_fill_adjust_error",
                                         group_id=og.group_id, error=str(e))
        else:
            ib = self._conn.ib
            try:
                tp_ib_id = int(og.broker_tp_order_id) if og.broker_tp_order_id else None
                sl_ib_id = int(og.broker_sl_order_id) if og.broker_sl_order_id else None
                for trade in ib.openTrades():
                    if tp_ib_id and trade.order.orderId == tp_ib_id:
                        trade.order.totalQuantity = new_qty
                        ib.placeOrder(self._contract, trade.order)
                    elif sl_ib_id and trade.order.orderId == sl_ib_id:
                        trade.order.totalQuantity = new_qty
                        ib.placeOrder(self._contract, trade.order)
            except Exception as e:
                self._logger.log("partial_fill_adjust_error",
                                 group_id=og.group_id, error=str(e))
        self._logger.log("partial_fill_adjust", group_id=og.group_id, new_qty=new_qty)

    async def _partial_fill_timeout(self, group_id, daily_state, timeout):
        try:
            await asyncio.sleep(timeout)
            og = None
            for o in daily_state.pending_orders:
                if o.group_id == group_id:
                    og = o
                    break

            if og is None or og.state == "CLOSED":
                return

            if og.filled_qty > 0 and og.filled_qty < og.target_qty:
                unfilled = og.target_qty - og.filled_qty
                await self._cancel_entry_remainder(og)
                self._notify_order_resolved(released_qty=unfilled)
                og.target_qty = og.filled_qty
                og.state = "FILLED"
                og.filled_at = self._now_iso()
                daily_state.move_to_open(og.group_id)
                self._state_mgr.save(daily_state)
                self._logger.log("partial_fill_timeout", group_id=group_id,
                                 kept=og.filled_qty, cancelled=unfilled)
        except asyncio.CancelledError:
            pass
        finally:
            self._partial_timers.pop(group_id, None)

    async def _cancel_entry_remainder(self, og):
        if self._config.dry_run or not self._is_broker_connected:
            return
        if og.broker_entry_order_id:
            await self._cancel_single_order(og.broker_entry_order_id)

    async def restore_partial_fill_timers(self, daily_state):
        for og in list(daily_state.pending_orders):
            if og.state != "PARTIAL" or not og.partial_fill_timer_start:
                continue
            try:
                start_time = datetime.fromisoformat(og.partial_fill_timer_start)
                now = datetime.fromisoformat(self._now_iso())
                elapsed = (now - start_time).total_seconds()
                remaining = self._config.partial_fill_timeout - elapsed

                if remaining <= 0:
                    self._logger.log("partial_fill_timeout_restored",
                                     group_id=og.group_id, action="expired",
                                     elapsed=round(elapsed, 1))
                    await self._cancel_entry_remainder(og)
                    og.target_qty = og.filled_qty
                    og.state = "FILLED"
                    og.filled_at = self._now_iso()
                    daily_state.move_to_open(og.group_id)
                    self._state_mgr.save(daily_state)
                else:
                    self._logger.log("partial_fill_timeout_restored",
                                     group_id=og.group_id, action="resumed",
                                     remaining=round(remaining, 1))
                    task = asyncio.ensure_future(
                        self._partial_fill_timeout(og.group_id, daily_state, remaining))
                    self._partial_timers[og.group_id] = task
            except (ValueError, TypeError) as e:
                self._logger.log("partial_fill_timer_restore_error",
                                 group_id=og.group_id, error=str(e))

    async def verify_child_order_quantities(self, daily_state):
        if self._config.dry_run or not self._is_broker_connected:
            return

        if self._use_adapter:
            return

        ib = self._conn.ib
        open_trades = {t.order.orderId: t for t in ib.openTrades()}
        fixed = 0

        for og in list(daily_state.open_positions) + list(daily_state.pending_orders):
            if og in daily_state.pending_orders and (og.state != "PARTIAL" or og.filled_qty <= 0):
                continue
            expected_qty = og.filled_qty or og.target_qty
            for broker_id in (og.broker_tp_order_id, og.broker_sl_order_id):
                if not broker_id:
                    continue
                ib_id = int(broker_id)
                if ib_id in open_trades:
                    trade = open_trades[ib_id]
                    if trade.order.totalQuantity != expected_qty:
                        self._logger.log("child_qty_mismatch_fixed",
                                         group_id=og.group_id, order_id=broker_id,
                                         was=trade.order.totalQuantity, expected=expected_qty)
                        trade.order.totalQuantity = expected_qty
                        ib.placeOrder(self._contract, trade.order)
                        fixed += 1

        if fixed > 0:
            self._logger.log("child_qty_verify_done", adjustments=fixed)
            self._state_mgr.save(daily_state)

    async def cancel_order_group(self, og, daily_state, reason=CLOSE_CANCEL):
        if not self._config.dry_run and self._is_exec_connected:
            for broker_id in (og.broker_entry_order_id, og.broker_tp_order_id, og.broker_sl_order_id):
                if broker_id:
                    await self._cancel_single_order(broker_id)

        timer = self._partial_timers.pop(og.group_id, None)
        if timer and not timer.done():
            timer.cancel()

        unreleased = max(0, og.target_qty - (og.filled_qty or 0))
        daily_state.move_to_closed(og.group_id, reason)
        self._logger.log("order_cancelled", group_id=og.group_id, reason=reason)
        self._notify_order_resolved(released_qty=unreleased)

    async def cancel_all_pending(self, daily_state, reason=CLOSE_EOD):
        to_cancel = [og for og in daily_state.pending_orders if og.state in ("SUBMITTED", "PARTIAL")]
        for og in to_cancel:
            await self.cancel_order_group(og, daily_state, reason)
        if to_cancel:
            self._logger.log("eod_cancel", count=len(to_cancel), reason=reason)

    async def flatten_all(self, daily_state, reason=CLOSE_FLATTEN):
        await self.cancel_all_pending(daily_state, reason)
        await asyncio.sleep(1.0)

        positions_to_close = list(daily_state.open_positions)
        for og in positions_to_close:
            await self._flatten_single_position(og, daily_state, reason)

        closed_ids = {og.group_id for og in positions_to_close}
        stragglers = [og for og in daily_state.open_positions if og.group_id not in closed_ids]
        if stragglers:
            self._logger.log("flatten_stragglers", count=len(stragglers))
            for og in stragglers:
                await self._flatten_single_position(og, daily_state, reason)

        total = len(positions_to_close) + len(stragglers)
        self._logger.log("flatten", reason=reason, positions=total)

    async def _flatten_single_position(self, og, daily_state, reason):
        qty = og.filled_qty or og.target_qty

        if not self._config.dry_run and self._is_exec_connected:
            # Cancel bracket exits before flattening
            if self._use_adapter:
                await self._adapter.cancel_bracket_exits(og.broker_entry_order_id)
                await asyncio.sleep(0.5)

            reverse_side = "SELL" if og.side == "BUY" else "BUY"

            if self._use_adapter:
                from bot.execution.broker_adapter import ContractInfo
                ci = ContractInfo(symbol=self._config.ticker, broker_contract_id="",
                                  expiry="", exchange=self._config.exchange)
                await self._adapter.place_market_order(ci, reverse_side, qty)
                # Estimate P&L for adapter mode
                if og.side == "BUY":
                    pnl_pts = og.stop_price - og.entry_price
                else:
                    pnl_pts = og.entry_price - og.stop_price
                est_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
                og.actual_exit_price = og.stop_price
                daily_state.move_to_closed(og.group_id, reason, est_pnl)
                self._logger.log(
                    "eod_exit", group_id=og.group_id, setup=og.setup,
                    side=og.side, entry_price=og.entry_price,
                    exit_price=og.stop_price, exit_reason=reason,
                    qty=qty, net_pnl=est_pnl, note="estimated (adapter)",
                )
                self._state_mgr.save(daily_state)
            else:
                from ib_async import MarketOrder
                mkt_order = MarketOrder(action=reverse_side, totalQuantity=qty)
                flatten_trade = self._conn.ib.placeOrder(self._contract, mkt_order)
                flatten_trade.filledEvent += lambda t, _og=og: self._on_flatten_fill(
                    t, _og, daily_state, reason)
        else:
            if og.side == "BUY":
                est_exit = og.stop_price
                pnl_pts = est_exit - og.entry_price
            else:
                est_exit = og.stop_price
                pnl_pts = og.entry_price - est_exit
            est_pnl = round(pnl_pts * qty * POINT_VALUE, 2)

            og.actual_exit_price = est_exit
            daily_state.move_to_closed(og.group_id, reason, est_pnl)
            self._logger.log(
                "eod_exit", group_id=og.group_id, setup=og.setup,
                side=og.side, entry_price=og.entry_price, exit_price=est_exit,
                exit_reason=reason, qty=qty, pnl_pts=round(pnl_pts, 2),
                net_pnl=est_pnl, note="estimated (dry run)",
            )
            if self._db:
                balance_after = daily_state.start_balance + daily_state.realized_pnl
                self._db.update_trade_exit(
                    og.group_id, actual_exit_price=est_exit, exit_reason=reason,
                    pnl_pts=round(pnl_pts, 2), gross_pnl=est_pnl, commission=0,
                    net_pnl=est_pnl, exit_time=self._now_iso(),
                    balance_after=round(balance_after, 2),
                    daily_pnl_after=round(daily_state.realized_pnl, 2),
                )

    def _on_flatten_fill(self, trade, og, daily_state, reason):
        fill_price = trade.orderStatus.avgFillPrice
        qty = og.filled_qty or og.target_qty
        commission = self._get_commission(trade) + og.entry_commission

        if og.side == "BUY":
            pnl_pts = fill_price - og.entry_price
        else:
            pnl_pts = og.entry_price - fill_price

        gross_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
        net_pnl = round(gross_pnl - commission, 2)

        og.actual_exit_price = fill_price
        daily_state.move_to_closed(og.group_id, reason, net_pnl)

        self._logger.log(
            "eod_exit", group_id=og.group_id, setup=og.setup,
            side=og.side, entry_price=og.entry_price, exit_price=fill_price,
            exit_reason=reason, qty=qty, pnl_pts=round(pnl_pts, 2),
            gross_pnl=gross_pnl, commission=commission, net_pnl=net_pnl,
            daily_realized=daily_state.realized_pnl,
        )

        if self._db:
            balance_after = daily_state.start_balance + daily_state.realized_pnl
            self._db.update_trade_exit(
                og.group_id, actual_exit_price=fill_price, exit_reason=reason,
                pnl_pts=round(pnl_pts, 2), gross_pnl=gross_pnl, commission=commission,
                net_pnl=net_pnl, exit_time=self._now_iso(),
                balance_after=round(balance_after, 2),
                daily_pnl_after=round(daily_state.realized_pnl, 2),
            )

        self._state_mgr.save(daily_state)

    def _on_tp_fill(self, trade, og, daily_state):
        if og.close_reason:
            return
        og.close_reason = CLOSE_TP
        self._cleanup_entry_remainder(og)

        fill_price = trade.orderStatus.avgFillPrice
        qty = og.filled_qty or og.target_qty
        commission = self._get_commission(trade) + og.entry_commission

        if og.side == "BUY":
            pnl_pts = fill_price - og.entry_price
        else:
            pnl_pts = og.entry_price - fill_price
        gross_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
        net_pnl = round(gross_pnl - commission, 2)

        duration_str = self._calc_duration(og)
        og.actual_exit_price = fill_price
        daily_state.move_to_closed(og.group_id, CLOSE_TP, net_pnl)

        self._logger.log(
            "tp_filled", group_id=og.group_id, setup=og.setup,
            side=og.side, entry_price=og.entry_price, exit_price=fill_price,
            qty=qty, risk_pts=og.risk_pts, n_value=og.n_value,
            pnl_pts=round(pnl_pts, 2), gross_pnl=gross_pnl,
            commission=commission, net_pnl=net_pnl, duration=duration_str,
            daily_realized=daily_state.realized_pnl, daily_trades=daily_state.trade_count,
        )
        self._write_exit_db(og, fill_price, "TP", pnl_pts, gross_pnl, commission, net_pnl, daily_state, duration_str)
        self._check_kill_switch(daily_state, "tp_fill")
        self._state_mgr.save(daily_state)
        self._notify_order_resolved(released_qty=0)

    def _on_sl_fill(self, trade, og, daily_state):
        if og.close_reason:
            return
        og.close_reason = CLOSE_SL
        self._cleanup_entry_remainder(og)

        fill_price = trade.orderStatus.avgFillPrice
        qty = og.filled_qty or og.target_qty
        commission = self._get_commission(trade) + og.entry_commission
        stop_slippage = round(abs(fill_price - og.stop_price), 2)

        if og.side == "BUY":
            pnl_pts = fill_price - og.entry_price
        else:
            pnl_pts = og.entry_price - fill_price
        gross_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
        net_pnl = round(gross_pnl - commission, 2)

        duration_str = self._calc_duration(og)
        og.actual_exit_price = fill_price
        daily_state.move_to_closed(og.group_id, CLOSE_SL, net_pnl)

        self._logger.log(
            "sl_filled", group_id=og.group_id, setup=og.setup,
            side=og.side, entry_price=og.entry_price, stop_price=og.stop_price,
            exit_price=fill_price, stop_slippage_pts=stop_slippage,
            qty=qty, risk_pts=og.risk_pts, n_value=og.n_value,
            pnl_pts=round(pnl_pts, 2), gross_pnl=gross_pnl,
            commission=commission, net_pnl=net_pnl, duration=duration_str,
            daily_realized=daily_state.realized_pnl, daily_trades=daily_state.trade_count,
        )
        self._write_exit_db(og, fill_price, "SL", pnl_pts, gross_pnl, commission, net_pnl, daily_state, duration_str, stop_slippage)
        self._check_kill_switch(daily_state, "sl_fill")
        self._state_mgr.save(daily_state)
        self._notify_order_resolved(released_qty=0)

    def _cleanup_entry_remainder(self, og):
        timer = self._partial_timers.pop(og.group_id, None)
        if timer and not timer.done():
            timer.cancel()

        if og.filled_qty and og.filled_qty < og.target_qty:
            asyncio.ensure_future(self._cancel_entry_remainder(og))
            self._logger.log("entry_remainder_cancelled", group_id=og.group_id,
                             filled=og.filled_qty, cancelled=og.target_qty - og.filled_qty,
                             reason="tp_sl_filled")

    def _on_entry_status(self, trade, og, daily_state):
        status = trade.orderStatus.status
        if status in ("Cancelled", "Inactive"):
            if og.state == "SUSPENDED":
                self._logger.log("suspend_cancel_echo", group_id=og.group_id, ib_status=status)
                return

            ib_filled = trade.orderStatus.filled
            if ib_filled > 0 and ib_filled > og.filled_qty:
                og.filled_qty = int(ib_filled)

            if og.filled_qty == 0:
                daily_state.move_to_closed(og.group_id, CLOSE_CANCEL)
                self._logger.log("entry_cancelled_ib", group_id=og.group_id)
            else:
                og.target_qty = og.filled_qty
                og.state = "FILLED"
                og.filled_at = self._now_iso()
                daily_state.move_to_open(og.group_id)
                self._logger.log("entry_partial_then_cancelled", group_id=og.group_id,
                                 kept=og.filled_qty)

            self._state_mgr.save(daily_state)

    # ── Shared helpers ─────────────────────────────────────────────

    async def _cancel_single_order(self, broker_id):
        """Cancel a single order by broker ID. Works for both modes."""
        if self._use_adapter:
            try:
                await self._adapter.cancel_order(str(broker_id))
            except Exception:
                pass
        else:
            ib = self._conn.ib
            ib_id = int(broker_id)
            for trade in ib.openTrades():
                if trade.order.orderId == ib_id:
                    ib.cancelOrder(trade.order)
                    break

    def _get_commission(self, trade):
        total = 0.0
        try:
            for fill in trade.fills:
                if fill.commissionReport and fill.commissionReport.commission:
                    c = fill.commissionReport.commission
                    if c < 1e8:
                        total += c
        except Exception:
            pass
        return round(total, 2)

    def _calc_duration(self, og):
        if og.filled_at:
            try:
                entry_t = datetime.fromisoformat(og.filled_at)
                exit_t = datetime.fromisoformat(self._now_iso())
                return str(exit_t - entry_t)
            except Exception:
                pass
        return ""

    def _check_kill_switch(self, daily_state, trigger):
        from bot.risk.risk_gates import RiskGates
        gates = RiskGates(self._config)
        if gates.evaluate_kill_switch(daily_state):
            self._logger.log("kill_switch", reason=daily_state.kill_switch_reason, trigger=trigger)
            if self._on_kill_switch:
                asyncio.ensure_future(self._on_kill_switch())

    def _write_exit_db(self, og, exit_price, reason, pnl_pts, gross_pnl, commission, net_pnl, daily_state, duration_str="", stop_slippage=0):
        if not self._db:
            return
        balance_after = daily_state.start_balance + daily_state.realized_pnl
        dur_sec = None
        if duration_str:
            try:
                parts = duration_str.split(':')
                dur_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))
            except Exception:
                pass
        self._db.update_trade_exit(
            og.group_id,
            actual_entry_price=og.actual_entry_price or og.entry_price,
            actual_exit_price=exit_price,
            entry_slippage_pts=og.entry_slippage_pts,
            stop_slippage_pts=stop_slippage,
            exit_reason=reason,
            pnl_pts=round(pnl_pts, 2),
            gross_pnl=gross_pnl, commission=commission, net_pnl=net_pnl,
            exit_time=self._now_iso(), duration_seconds=dur_sec,
            balance_after=round(balance_after, 2),
            daily_pnl_after=round(daily_state.realized_pnl, 2),
        )
