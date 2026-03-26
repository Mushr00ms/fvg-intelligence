"""
margin_priority.py — Intelligent margin allocation for NQ futures.

When margin is insufficient for a new order, suspends the resting order
furthest from current price to free margin for closer, more actionable setups.
Suspended orders are re-placed when margin becomes available (e.g., after a TP/SL).
"""

import asyncio
import math
from typing import Optional


class MarginPriorityManager:
    """Manages margin allocation across concurrent resting orders."""

    def __init__(self, margin_tracker, order_manager, risk_gates, time_gates,
                 logger, config, clock=None):
        self._margin = margin_tracker
        self._order_mgr = order_manager
        self._risk_gates = risk_gates
        self._time_gates = time_gates
        self._logger = logger
        self._config = config
        self._clock = clock

    async def evaluate_and_place(self, proposed_order, fvg, daily_state, current_price):
        """Main entry point: place order with margin-aware priority.

        Called from engine._process_detection after risk gates pass.

        Returns:
            "PLACED"              — order placed normally
            "PLACED_AFTER_SUSPEND" — placed after suspending farther order(s)
            "SUSPENDED"           — new order itself suspended (farthest from price)
        """
        available = self._margin.get_available_margin()

        # ── Happy path: can afford at least 1 contract ────────────────────
        if self._margin.can_afford(1, available):
            max_qty = self._margin.max_contracts_by_margin(available)
            if max_qty < proposed_order.target_qty:
                self._logger.log(
                    "margin_qty_cap",
                    group_id=proposed_order.group_id,
                    original_qty=proposed_order.target_qty,
                    capped_qty=max(1, max_qty),
                    available=round(available, 2),
                    per_contract=round(self._margin.margin_per_contract, 2),
                )
                proposed_order.target_qty = max(1, max_qty)
            await self._order_mgr.place_bracket(proposed_order, daily_state)
            return "PLACED"

        # ── Can't afford even 1 contract: evaluate priority ───────────────
        needed_for_one = self._margin.margin_required_for(1) * (1.0 + self._config.margin_buffer_pct)
        new_dist = abs(proposed_order.entry_price - current_price)

        # Collect suspendable orders: SUBMITTED only (not PARTIAL/FILLED)
        suspendable = []
        for og in daily_state.pending_orders:
            if og.state == "SUBMITTED" and og.filled_qty == 0:
                dist = abs(og.entry_price - current_price)
                suspendable.append((og, dist))

        # Rank all candidates (existing + proposed) by distance, farthest first
        candidates = list(suspendable)
        candidates.append((proposed_order, new_dist))
        candidates.sort(key=lambda x: x[1], reverse=True)

        # If proposed is the farthest → suspend it (don't displace closer orders)
        if candidates[0][0] is proposed_order:
            self._add_to_suspended(proposed_order, daily_state, fvg,
                                    "farthest_from_price")
            return "SUSPENDED"

        # Greedily suspend farthest orders until enough margin freed
        freed = 0.0
        to_suspend = []
        for og, dist in candidates:
            if og is proposed_order:
                continue
            if available + freed >= needed_for_one:
                break
            freed += self._margin.margin_required_for(og.target_qty)
            to_suspend.append(og)

        # Even suspending everything isn't enough? Try with reduced qty
        total_available = available + freed
        max_qty = self._margin.max_contracts_by_margin(total_available)
        if max_qty < 1:
            self._add_to_suspended(proposed_order, daily_state, fvg,
                                    "insufficient_margin_after_all_suspensions")
            return "SUSPENDED"

        # Execute suspensions
        for og in to_suspend:
            await self._order_mgr.suspend_order(
                og, daily_state,
                reason=f"margin_priority: {proposed_order.group_id} closer "
                       f"({new_dist:.1f} vs {abs(og.entry_price - current_price):.1f} pts)",
            )

        # Brief pause for IB to process cancellations and update margin
        await asyncio.sleep(0.5)

        # Re-check margin after suspensions
        available = self._margin.get_available_margin()
        max_qty = self._margin.max_contracts_by_margin(available)
        if max_qty < 1:
            # Race condition: margin still not freed
            self._add_to_suspended(proposed_order, daily_state, fvg,
                                    "margin_not_freed_after_suspend")
            return "SUSPENDED"

        proposed_order.target_qty = min(proposed_order.target_qty, max(1, max_qty))
        await self._order_mgr.place_bracket(proposed_order, daily_state)

        self._logger.log(
            "margin_priority_resolved",
            group_id=proposed_order.group_id,
            suspended_count=len(to_suspend),
            suspended_ids=[og.group_id for og in to_suspend],
            final_qty=proposed_order.target_qty,
        )
        return "PLACED_AFTER_SUSPEND"

    async def try_reactivate_suspended(self, daily_state, current_price) -> int:
        """Try to re-place suspended orders when margin becomes available.

        Called after any order resolves (TP/SL/cancel).
        Processes closest-to-price suspended orders first.
        Returns count of orders re-placed.
        """
        if not daily_state.suspended_orders:
            return 0

        # Sort by distance to current price (closest first)
        candidates = []
        for og in list(daily_state.suspended_orders):
            dist = abs(og.entry_price - current_price)
            candidates.append((og, dist))
        candidates.sort(key=lambda x: x[1])

        reactivated = 0
        for og, dist in candidates:
            # Check FVG still valid
            fvg = self._find_active_fvg(og.fvg_id, daily_state)
            if fvg is None:
                # FVG mitigated or expired while suspended
                daily_state.move_to_closed(og.group_id, "FVG_EXPIRED")
                self._logger.log(
                    "suspended_fvg_expired",
                    group_id=og.group_id,
                    fvg_id=og.fvg_id,
                )
                continue

            # Check time gate
            allowed, reason = self._time_gates.can_enter()
            if not allowed:
                continue  # Past entry window, leave for EOD cleanup

            # Check risk gates (daily limits may have changed)
            gate_result = self._risk_gates.check_all(daily_state, og)
            if not gate_result.passed:
                self._logger.log(
                    "reactivate_blocked",
                    group_id=og.group_id,
                    gate=gate_result.gate,
                    reason=gate_result.reason,
                )
                continue

            # Check margin
            available = self._margin.get_available_margin()
            if not self._margin.can_afford(1, available):
                break  # No margin left, stop trying

            max_qty = self._margin.max_contracts_by_margin(available)
            og.target_qty = min(og.target_qty, max(1, max_qty))

            # Re-place
            result = await self._order_mgr.reactivate_order(og, daily_state)
            if result:
                reactivated += 1

        return reactivated

    async def clear_all_suspended(self, daily_state, reason="EOD") -> int:
        """Move all suspended orders to CLOSED. Called at EOD."""
        count = 0
        for og in list(daily_state.suspended_orders):
            daily_state.move_to_closed(og.group_id, reason)
            count += 1
        if count:
            self._logger.log("suspended_cleared", count=count, reason=reason)
        return count

    def _add_to_suspended(self, order, daily_state, fvg, reason):
        """Add a not-yet-placed order directly to suspended_orders."""
        order.state = "SUSPENDED"
        order.suspended_at = self._now_iso()
        order.suspend_reason = reason
        daily_state.suspended_orders.append(order)
        # Still link to FVG so we can validate on reactivation
        if order.group_id not in fvg.orders_placed:
            fvg.orders_placed.append(order.group_id)
        self._logger.log(
            "order_suspended",
            group_id=order.group_id,
            fvg_id=fvg.fvg_id,
            entry=order.entry_price,
            qty=order.target_qty,
            reason=reason,
        )

    def _find_active_fvg(self, fvg_id, daily_state):
        """Find an active (non-mitigated) FVG in daily state."""
        for fvg in daily_state.active_fvgs:
            if fvg.fvg_id == fvg_id and not fvg.is_mitigated:
                return fvg
        return None

    def _now_iso(self):
        if self._clock is not None:
            return self._clock.now().isoformat()
        from datetime import datetime
        return datetime.now().isoformat()
