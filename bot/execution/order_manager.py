"""
order_manager.py — Bracket order placement, modification, and lifecycle management.

Handles:
- Crash-safe bracket order placement (pre-allocate IDs, save state before placing)
- Partial fill detection and TP/SL qty adjustment
- 5-minute timer for cancelling unfilled entry remainder
- Flatten all (market-order close)
- Cancel all pending entries
"""

import asyncio
from datetime import datetime

import pytz

from bot.state.trade_state import OrderGroup, CLOSE_TP, CLOSE_SL, CLOSE_FLATTEN, CLOSE_CANCEL, CLOSE_EOD, CLOSE_REJECTED
from bot.strategy.trade_calculator import POINT_VALUE

NY_TZ = pytz.timezone("America/New_York")


class OrderManager:
    """
    Manages bracket order lifecycle with Interactive Brokers.

    Crash-safe placement protocol:
    1. Pre-allocate 3 order IDs
    2. Set IDs on OrderGroup → save state (IDs persisted BEFORE placing)
    3. Place bracket order with IB
    """

    def __init__(self, ib_connection, contract, state_manager, logger, config, clock=None, db=None,
                 on_kill_switch=None):
        self._conn = ib_connection
        self._contract = contract
        self._state_mgr = state_manager
        self._logger = logger
        self._config = config
        self._clock = clock
        self._db = db
        self._on_kill_switch = on_kill_switch  # Callback for immediate kill switch
        self._partial_timers = {}  # group_id -> asyncio.Task
        self._adjustment_lock = asyncio.Lock()  # Serialize partial fill qty adjustments

    def _now_iso(self):
        if self._clock is not None:
            return self._clock.now().isoformat()
        return datetime.now(NY_TZ).isoformat()

    async def place_bracket(self, order_group, daily_state):
        """
        Place a bracket order (entry + TP + SL) with IB.

        In DRY_RUN mode: logs the order but skips actual placement.
        Returns the updated OrderGroup.
        """
        ib = self._conn.ib
        og = order_group

        if self._config.dry_run:
            og.state = "SUBMITTED"
            og.submitted_at = self._now_iso()
            daily_state.pending_orders.append(og)
            daily_state.trade_count += 1
            self._logger.log(
                "order_placed",
                mode="DRY_RUN",
                group_id=og.group_id,
                setup=og.setup,
                side=og.side,
                entry=og.entry_price,
                stop=og.stop_price,
                target=og.target_price,
                risk_pts=og.risk_pts,
                n=og.n_value,
                contracts=og.target_qty,
            )
            return og

        # Step 1: Pre-allocate IB order IDs
        og.ib_entry_order_id = ib.client.getReqId()
        og.ib_tp_order_id = ib.client.getReqId()
        og.ib_sl_order_id = ib.client.getReqId()

        # Step 2: Save state with IDs BEFORE placing
        og.state = "SUBMITTED"
        og.submitted_at = self._now_iso()
        daily_state.pending_orders.append(og)
        daily_state.trade_count += 1
        self._state_mgr.save(daily_state)

        # Step 3: Place bracket order
        from ib_async import LimitOrder, StopOrder, Order

        reverse_side = "SELL" if og.side == "BUY" else "BUY"

        # Parent: limit entry
        parent = LimitOrder(
            action=og.side,
            totalQuantity=og.target_qty,
            lmtPrice=og.entry_price,
            orderId=og.ib_entry_order_id,
            tif='GTC',
            transmit=False,
        )

        # Take profit: limit at target
        tp = LimitOrder(
            action=reverse_side,
            totalQuantity=og.target_qty,
            lmtPrice=og.target_price,
            orderId=og.ib_tp_order_id,
            parentId=og.ib_entry_order_id,
            tif='GTC',
            transmit=False,
        )

        # Stop loss: stop at stop price
        sl = StopOrder(
            action=reverse_side,
            totalQuantity=og.target_qty,
            stopPrice=og.stop_price,
            orderId=og.ib_sl_order_id,
            parentId=og.ib_entry_order_id,
            tif='GTC',
            transmit=True,  # Transmit the whole bracket
        )

        # Place all three
        entry_trade = ib.placeOrder(self._contract, parent)
        tp_trade = ib.placeOrder(self._contract, tp)
        sl_trade = ib.placeOrder(self._contract, sl)

        # Register fill callbacks
        entry_trade.filledEvent += lambda trade: self._on_entry_fill(trade, og, daily_state)
        entry_trade.statusEvent += lambda trade: self._on_entry_status(trade, og, daily_state)
        tp_trade.filledEvent += lambda trade: self._on_tp_fill(trade, og, daily_state)
        sl_trade.filledEvent += lambda trade: self._on_sl_fill(trade, og, daily_state)

        self._logger.log(
            "order_placed",
            mode="LIVE",
            group_id=og.group_id,
            setup=og.setup,
            side=og.side,
            entry=og.entry_price,
            stop=og.stop_price,
            target=og.target_price,
            risk_pts=og.risk_pts,
            n=og.n_value,
            contracts=og.target_qty,
            ib_entry_id=og.ib_entry_order_id,
        )

        return og

    def _on_entry_fill(self, trade, og, daily_state):
        """Handle entry order fill (full or partial)."""
        filled = trade.orderStatus.filled
        if filled <= 0:
            return

        if filled >= og.target_qty:
            # Full fill
            og.filled_qty = og.target_qty
            og.filled_at = self._now_iso()
            avg_fill = trade.orderStatus.avgFillPrice
            slippage = round(abs(avg_fill - og.entry_price), 2)
            og.actual_entry_price = avg_fill
            og.entry_slippage_pts = slippage
            daily_state.move_to_open(og.group_id)
            self._logger.log(
                "order_filled",
                group_id=og.group_id,
                setup=og.setup,
                side=og.side,
                qty=og.filled_qty,
                expected_price=og.entry_price,
                avg_price=avg_fill,
                slippage_pts=slippage,
                risk_pts=og.risk_pts,
                target_price=og.target_price,
                stop_price=og.stop_price,
                n_value=og.n_value,
            )
        else:
            # Partial fill
            self._handle_partial_fill(trade, og, daily_state, filled)

    def _handle_partial_fill(self, trade, og, daily_state, filled_qty):
        """
        Handle partial fill of entry order:
        1. Adjust TP/SL qty to match filled qty
        2. Start 5-minute timer for unfilled remainder
        """
        og.filled_qty = filled_qty
        og.state = "PARTIAL"
        og.partial_fill_timer_start = self._now_iso()

        self._logger.log(
            "partial_fill",
            group_id=og.group_id,
            filled=filled_qty,
            target=og.target_qty,
            remaining=og.target_qty - filled_qty,
        )

        # Adjust child order quantities (if not dry run)
        # Awaited under lock to prevent race with subsequent partial fills
        if not self._config.dry_run and self._conn.is_connected:
            asyncio.ensure_future(self._locked_adjust_child_qty(og, filled_qty))

        # Start 5-minute timer for unfilled remainder
        if og.group_id not in self._partial_timers:
            timeout = self._config.partial_fill_timeout
            task = asyncio.ensure_future(
                self._partial_fill_timeout(og.group_id, daily_state, timeout)
            )
            self._partial_timers[og.group_id] = task

    async def _locked_adjust_child_qty(self, og, new_qty):
        """Serialize TP/SL qty adjustments to prevent race conditions between partial fills."""
        async with self._adjustment_lock:
            await self._adjust_child_qty(og, new_qty)

    async def _adjust_child_qty(self, og, new_qty):
        """Modify TP and SL order quantities to match partial fill."""
        ib = self._conn.ib
        try:
            # Find and modify TP order
            for trade in ib.openTrades():
                if trade.order.orderId == og.ib_tp_order_id:
                    trade.order.totalQuantity = new_qty
                    ib.placeOrder(self._contract, trade.order)
                elif trade.order.orderId == og.ib_sl_order_id:
                    trade.order.totalQuantity = new_qty
                    ib.placeOrder(self._contract, trade.order)

            self._logger.log(
                "partial_fill_adjust",
                group_id=og.group_id,
                new_qty=new_qty,
            )
        except Exception as e:
            self._logger.log(
                "partial_fill_adjust_error",
                group_id=og.group_id,
                error=str(e),
            )

    async def _partial_fill_timeout(self, group_id, daily_state, timeout):
        """After timeout, cancel unfilled entry remainder."""
        try:
            await asyncio.sleep(timeout)

            # Find the order group
            og = None
            for o in daily_state.pending_orders:
                if o.group_id == group_id:
                    og = o
                    break

            if og is None or og.state == "CLOSED":
                return  # Already resolved

            if og.filled_qty > 0 and og.filled_qty < og.target_qty:
                # Cancel unfilled remainder
                await self._cancel_entry_remainder(og)
                og.target_qty = og.filled_qty
                og.state = "FILLED"
                og.filled_at = self._now_iso()
                daily_state.move_to_open(og.group_id)

                self._logger.log(
                    "partial_fill_timeout",
                    group_id=group_id,
                    kept=og.filled_qty,
                    cancelled=og.target_qty - og.filled_qty,
                )
        except asyncio.CancelledError:
            pass
        finally:
            self._partial_timers.pop(group_id, None)

    async def _cancel_entry_remainder(self, og):
        """Cancel the unfilled portion of an entry order."""
        if self._config.dry_run or not self._conn.is_connected:
            return
        ib = self._conn.ib
        for trade in ib.openTrades():
            if trade.order.orderId == og.ib_entry_order_id:
                ib.cancelOrder(trade.order)
                break

    async def restore_partial_fill_timers(self, daily_state):
        """Restore partial fill timers from persisted state after crash/restart.

        Scans pending orders for PARTIAL state with a timer start timestamp.
        If the timer has expired, immediately cancels the unfilled remainder.
        If time remains, spawns a new timer with the adjusted timeout.
        """
        for og in list(daily_state.pending_orders):
            if og.state != "PARTIAL" or not og.partial_fill_timer_start:
                continue

            try:
                start_time = datetime.fromisoformat(og.partial_fill_timer_start)
                now = datetime.fromisoformat(self._now_iso())
                elapsed = (now - start_time).total_seconds()
                remaining = self._config.partial_fill_timeout - elapsed

                if remaining <= 0:
                    # Timer already expired — cancel remainder immediately
                    self._logger.log(
                        "partial_fill_timeout_restored",
                        group_id=og.group_id,
                        action="expired",
                        elapsed=round(elapsed, 1),
                    )
                    await self._cancel_entry_remainder(og)
                    og.target_qty = og.filled_qty
                    og.state = "FILLED"
                    og.filled_at = self._now_iso()
                    daily_state.move_to_open(og.group_id)
                else:
                    # Timer still active — resume with remaining time
                    self._logger.log(
                        "partial_fill_timeout_restored",
                        group_id=og.group_id,
                        action="resumed",
                        remaining=round(remaining, 1),
                    )
                    task = asyncio.ensure_future(
                        self._partial_fill_timeout(og.group_id, daily_state, remaining)
                    )
                    self._partial_timers[og.group_id] = task
            except (ValueError, TypeError) as e:
                self._logger.log(
                    "partial_fill_timer_restore_error",
                    group_id=og.group_id,
                    error=str(e),
                )

    async def cancel_order_group(self, og, daily_state, reason=CLOSE_CANCEL):
        """Cancel all orders in an order group."""
        if not self._config.dry_run and self._conn.is_connected:
            ib = self._conn.ib
            for order_id in (og.ib_entry_order_id, og.ib_tp_order_id, og.ib_sl_order_id):
                if order_id:
                    for trade in ib.openTrades():
                        if trade.order.orderId == order_id:
                            ib.cancelOrder(trade.order)
                            break

        # Cancel partial fill timer if active
        timer = self._partial_timers.pop(og.group_id, None)
        if timer and not timer.done():
            timer.cancel()

        daily_state.move_to_closed(og.group_id, reason)
        self._logger.log(
            "order_cancelled",
            group_id=og.group_id,
            reason=reason,
        )

    async def cancel_all_pending(self, daily_state, reason=CLOSE_EOD):
        """Cancel all unfilled entry orders (EOD cleanup)."""
        to_cancel = [og for og in daily_state.pending_orders if og.state in ("SUBMITTED", "PARTIAL")]
        for og in to_cancel:
            await self.cancel_order_group(og, daily_state, reason)

        if to_cancel:
            self._logger.log("eod_cancel", count=len(to_cancel), reason=reason)

    async def flatten_all(self, daily_state, reason=CLOSE_FLATTEN):
        """Market-order close all open positions with P&L tracking.

        Safety order: cancel pending entries FIRST (prevents new positions
        appearing during flatten), then flatten open positions, then sweep
        for any stragglers that snuck through.
        """
        # Step 1: Cancel all pending entries FIRST
        await self.cancel_all_pending(daily_state, reason)

        # Step 2: Brief yield for IB to process cancellations
        await asyncio.sleep(1.0)

        # Step 3: Flatten all open positions
        positions_to_close = list(daily_state.open_positions)
        for og in positions_to_close:
            await self._flatten_single_position(og, daily_state, reason)

        # Step 4: Sweep for stragglers (positions that appeared during cancel window)
        closed_ids = {og.group_id for og in positions_to_close}
        stragglers = [
            og for og in daily_state.open_positions
            if og.group_id not in closed_ids
        ]
        if stragglers:
            self._logger.log("flatten_stragglers", count=len(stragglers))
            for og in stragglers:
                await self._flatten_single_position(og, daily_state, reason)

        total = len(positions_to_close) + len(stragglers)
        self._logger.log("flatten", reason=reason, positions=total)

    async def _flatten_single_position(self, og, daily_state, reason):
        """Market-order close a single open position."""
        qty = og.filled_qty or og.target_qty

        if not self._config.dry_run and self._conn.is_connected:
            from ib_async import MarketOrder
            reverse_side = "SELL" if og.side == "BUY" else "BUY"
            mkt_order = MarketOrder(action=reverse_side, totalQuantity=qty)
            flatten_trade = self._conn.ib.placeOrder(self._contract, mkt_order)
            flatten_trade.filledEvent += lambda t, _og=og: self._on_flatten_fill(
                t, _og, daily_state, reason
            )
        else:
            if og.side == "BUY":
                est_exit = og.stop_price
                pnl_pts = est_exit - og.entry_price
            else:
                est_exit = og.stop_price
                pnl_pts = og.entry_price - est_exit
            est_pnl = round(pnl_pts * qty * POINT_VALUE, 2)

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
        """Handle the market order fill from EOD flatten — compute actual P&L."""
        fill_price = trade.orderStatus.avgFillPrice
        qty = og.filled_qty or og.target_qty
        commission = self._get_commission(trade)

        if og.side == "BUY":
            pnl_pts = fill_price - og.entry_price
        else:
            pnl_pts = og.entry_price - fill_price

        gross_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
        net_pnl = round(gross_pnl - commission, 2)

        daily_state.move_to_closed(og.group_id, reason, net_pnl)

        self._logger.log(
            "eod_exit",
            group_id=og.group_id,
            setup=og.setup,
            side=og.side,
            entry_price=og.entry_price,
            exit_price=fill_price,
            exit_reason=reason,
            qty=qty,
            pnl_pts=round(pnl_pts, 2),
            gross_pnl=gross_pnl,
            commission=commission,
            net_pnl=net_pnl,
            daily_realized=daily_state.realized_pnl,
        )

        if self._db:
            balance_after = daily_state.start_balance + daily_state.realized_pnl
            self._db.update_trade_exit(
                og.group_id,
                actual_exit_price=fill_price,
                exit_reason=reason,
                pnl_pts=round(pnl_pts, 2),
                gross_pnl=gross_pnl, commission=commission, net_pnl=net_pnl,
                exit_time=self._now_iso(),
                balance_after=round(balance_after, 2),
                daily_pnl_after=round(daily_state.realized_pnl, 2),
            )

        self._state_mgr.save(daily_state)

    def _on_tp_fill(self, trade, og, daily_state):
        """Take profit filled — log P&L, commissions, trade metrics."""
        # Cancel any unfilled entry remainder + kill partial fill timer
        self._cleanup_entry_remainder(og)

        fill_price = trade.orderStatus.avgFillPrice
        qty = og.filled_qty or og.target_qty
        commission = self._get_commission(trade)

        if og.side == "BUY":
            pnl_pts = fill_price - og.entry_price
        else:
            pnl_pts = og.entry_price - fill_price
        gross_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
        net_pnl = round(gross_pnl - commission, 2)

        # Trade duration
        duration_str = ""
        if og.filled_at:
            try:
                from datetime import datetime
                entry_t = datetime.fromisoformat(og.filled_at)
                exit_t = datetime.fromisoformat(self._now_iso())
                duration_str = str(exit_t - entry_t)
            except Exception:
                pass

        daily_state.move_to_closed(og.group_id, CLOSE_TP, net_pnl)

        self._logger.log(
            "tp_filled",
            group_id=og.group_id,
            setup=og.setup,
            side=og.side,
            entry_price=og.entry_price,
            exit_price=fill_price,
            qty=qty,
            risk_pts=og.risk_pts,
            n_value=og.n_value,
            pnl_pts=round(pnl_pts, 2),
            gross_pnl=gross_pnl,
            commission=commission,
            net_pnl=net_pnl,
            duration=duration_str,
            daily_realized=daily_state.realized_pnl,
            daily_trades=daily_state.trade_count,
        )

        # Write to DB
        if self._db:
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
                actual_exit_price=fill_price,
                entry_slippage_pts=og.entry_slippage_pts,
                stop_slippage_pts=0,
                exit_reason="TP",
                pnl_pts=round(pnl_pts, 2),
                gross_pnl=gross_pnl,
                commission=commission,
                net_pnl=net_pnl,
                exit_time=self._now_iso(),
                duration_seconds=dur_sec,
                balance_after=round(balance_after, 2),
                daily_pnl_after=round(daily_state.realized_pnl, 2),
            )

        # Check kill switch after every close
        from bot.risk.risk_gates import RiskGates
        gates = RiskGates(self._config)
        if gates.evaluate_kill_switch(daily_state):
            self._logger.log("kill_switch", reason=daily_state.kill_switch_reason,
                             trigger="tp_fill")
            if self._on_kill_switch:
                asyncio.ensure_future(self._on_kill_switch())
        self._state_mgr.save(daily_state)

    def _on_sl_fill(self, trade, og, daily_state):
        """Stop loss filled — log P&L, slippage, commissions."""
        # Cancel any unfilled entry remainder + kill partial fill timer
        self._cleanup_entry_remainder(og)

        fill_price = trade.orderStatus.avgFillPrice
        qty = og.filled_qty or og.target_qty
        commission = self._get_commission(trade)
        stop_slippage = round(abs(fill_price - og.stop_price), 2)

        if og.side == "BUY":
            pnl_pts = fill_price - og.entry_price
        else:
            pnl_pts = og.entry_price - fill_price
        gross_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
        net_pnl = round(gross_pnl - commission, 2)

        duration_str = ""
        if og.filled_at:
            try:
                from datetime import datetime
                entry_t = datetime.fromisoformat(og.filled_at)
                exit_t = datetime.fromisoformat(self._now_iso())
                duration_str = str(exit_t - entry_t)
            except Exception:
                pass

        daily_state.move_to_closed(og.group_id, CLOSE_SL, net_pnl)

        self._logger.log(
            "sl_filled",
            group_id=og.group_id,
            setup=og.setup,
            side=og.side,
            entry_price=og.entry_price,
            stop_price=og.stop_price,
            exit_price=fill_price,
            stop_slippage_pts=stop_slippage,
            qty=qty,
            risk_pts=og.risk_pts,
            n_value=og.n_value,
            pnl_pts=round(pnl_pts, 2),
            gross_pnl=gross_pnl,
            commission=commission,
            net_pnl=net_pnl,
            duration=duration_str,
            daily_realized=daily_state.realized_pnl,
            daily_trades=daily_state.trade_count,
        )

        # Write to DB
        if self._db:
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
                actual_exit_price=fill_price,
                entry_slippage_pts=og.entry_slippage_pts,
                stop_slippage_pts=stop_slippage,
                exit_reason="SL",
                pnl_pts=round(pnl_pts, 2),
                gross_pnl=gross_pnl,
                commission=commission,
                net_pnl=net_pnl,
                exit_time=self._now_iso(),
                duration_seconds=dur_sec,
                balance_after=round(balance_after, 2),
                daily_pnl_after=round(daily_state.realized_pnl, 2),
            )

        from bot.risk.risk_gates import RiskGates
        gates = RiskGates(self._config)
        if gates.evaluate_kill_switch(daily_state):
            self._logger.log("kill_switch", reason=daily_state.kill_switch_reason,
                             trigger="sl_fill")
            if self._on_kill_switch:
                asyncio.ensure_future(self._on_kill_switch())
        self._state_mgr.save(daily_state)

    def _cleanup_entry_remainder(self, og):
        """Cancel unfilled entry contracts and kill partial fill timer on TP/SL fill.

        Prevents naked positions: if TP/SL fills while entry is still partially
        unfilled, the remaining entry contracts must be cancelled immediately.
        """
        # Cancel the partial fill timer
        timer = self._partial_timers.pop(og.group_id, None)
        if timer and not timer.done():
            timer.cancel()

        # Cancel unfilled entry remainder on IB
        if og.filled_qty and og.filled_qty < og.target_qty:
            asyncio.ensure_future(self._cancel_entry_remainder(og))
            self._logger.log(
                "entry_remainder_cancelled",
                group_id=og.group_id,
                filled=og.filled_qty,
                cancelled=og.target_qty - og.filled_qty,
                reason="tp_sl_filled",
            )

    def _get_commission(self, trade):
        """Extract total commission from an IB trade fill."""
        total = 0.0
        try:
            for fill in trade.fills:
                if fill.commissionReport and fill.commissionReport.commission:
                    c = fill.commissionReport.commission
                    if c < 1e8:  # IB returns 1e10 for "not yet available"
                        total += c
        except Exception:
            pass
        return round(total, 2)

    def _on_entry_status(self, trade, og, daily_state):
        """Handle entry order status changes (cancelled, rejected, etc.)."""
        status = trade.orderStatus.status
        if status in ("Cancelled", "Inactive"):
            # Update filled qty from IB (our tracker may lag)
            ib_filled = trade.orderStatus.filled
            if ib_filled > 0 and ib_filled > og.filled_qty:
                og.filled_qty = int(ib_filled)

            if og.filled_qty == 0:
                # Clean cancel — no fills
                daily_state.move_to_closed(og.group_id, CLOSE_CANCEL)
                self._logger.log(
                    "order_cancelled",
                    group_id=og.group_id,
                    ib_status=status,
                )
            else:
                # CRITICAL: Entry partially filled then cancelled (race condition).
                # We now have contracts with no TP/SL. Flatten immediately.
                self._logger.log(
                    "cancelled_with_fill",
                    group_id=og.group_id,
                    filled_qty=og.filled_qty,
                    ib_status=status,
                )
                og.state = "FILLED"
                og.filled_at = self._now_iso()
                og.target_qty = og.filled_qty
                daily_state.move_to_open(og.group_id)
                asyncio.ensure_future(
                    self._flatten_single_position(og, daily_state, CLOSE_EOD)
                )
        elif status == "Rejected":
            # IB rejected the order (insufficient margin, invalid price, etc.)
            self._logger.log(
                "order_rejected",
                group_id=og.group_id,
                ib_status=status,
                why_held=getattr(trade.orderStatus, 'whyHeld', ''),
            )
            daily_state.move_to_closed(og.group_id, CLOSE_REJECTED)
            self._state_mgr.save(daily_state)
        elif status not in ("Submitted", "PreSubmitted", "Filled"):
            # Unexpected status — log for investigation
            self._logger.log(
                "order_status_unexpected",
                group_id=og.group_id,
                ib_status=status,
            )
