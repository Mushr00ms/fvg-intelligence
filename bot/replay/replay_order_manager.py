"""
replay_order_manager.py — Order manager with tick-accurate fills from Databento data.

Fills are simulated against real Databento tick data (trades schema), not IB
paper fills (which produce garbage slippage) or simple bar touch=fill.

Fill model:
  Entry (limit):  Accumulate volume at exact limit price. Fill when cumulative
                  size >= order qty.  Fill price = exact limit (passive limit).
  Stop (stop):    First trade at/through stop price. Fill price = actual trade
                  price (may be worse than stop → real slippage).
  TP (limit):     First trade at/through target. Fill price = exact target
                  (passive limit).

Includes full margin priority displacement (suspend far, take near, reactivate
on resolution) — same logic as bot/risk/margin_priority.py.
"""

import math
from datetime import datetime

import pytz

from bot.state.trade_state import (
    OrderGroup, CLOSE_TP, CLOSE_SL, CLOSE_FLATTEN,
    CLOSE_CANCEL, CLOSE_EOD, CLOSE_REJECTED,
)
from bot.strategy.trade_calculator import POINT_VALUE, TICK_SIZE

NY_TZ = pytz.timezone("America/New_York")

# IB tiered commission — same formula as backtester
_EXCHANGE_FEE_PER_SIDE = 1.40
_IB_TIERS = [
    (1000,  0.85),
    (10000, 0.65),
    (20000, 0.45),
    (float('inf'), 0.25),
]

# Volume threshold for entry fill confidence (from enrich_volume.py analysis)
FILL_VOLUME_THRESHOLD = 30


def _calc_commission(num_contracts, monthly_contracts=0):
    """Round-trip commission with IB tiered pricing."""
    total = 0.0
    remaining = num_contracts
    vol = monthly_contracts
    for threshold, rate in _IB_TIERS:
        if remaining <= 0:
            break
        available = threshold - vol
        if available <= 0:
            continue
        batch = min(remaining, available)
        total += batch * (rate + _EXCHANGE_FEE_PER_SIDE) * 2
        remaining -= batch
        vol += batch
    return round(total, 2)


class ReplayOrderManager:
    """
    Simulates bracket order fills against Databento tick data.

    Tick-accurate fill model:
    - Entry: volume-aware limit fill (accumulate size at price until qty met)
    - Stop: fill at actual trade price through stop (real slippage)
    - TP: fill at exact target price (passive limit)
    """

    def __init__(self, state_manager, logger, config, clock=None, db=None,
                 on_kill_switch=None):
        self._state_mgr = state_manager
        self._logger = logger
        self._config = config
        self._clock = clock
        self._db = db
        self._on_kill_switch = on_kill_switch
        self._on_order_resolved = None
        self._last_price = 0.0
        self._monthly_contracts = 0
        self._displacement_log = []
        # Track accumulated volume at entry price per order (for volume-aware fills)
        self._entry_volume = {}  # group_id -> cumulative contracts at entry price

    def _now_iso(self):
        if self._clock is not None:
            return self._clock.now().isoformat()
        return datetime.now(NY_TZ).isoformat()

    # ── Public interface (matches OrderManager) ──────────────────────────

    def place_bracket(self, order_group, daily_state):
        """Register bracket order with margin-aware displacement."""
        daily_state.trade_count += 1
        return self._place_internal(order_group, daily_state)

    def _place_internal(self, og, daily_state, is_reactivation=False):
        """Place or re-place an order with margin priority logic."""
        margin_per = self._config.margin_fallback_per_contract
        buffer = self._config.margin_buffer_pct
        buffered_margin = margin_per * (1.0 + buffer)
        balance = daily_state.start_balance + daily_state.realized_pnl

        used_margin = sum(
            (o.filled_qty or o.target_qty) * buffered_margin
            for o in daily_state.pending_orders + daily_state.open_positions
        )
        available = balance - used_margin

        # Happy path: can afford
        if available >= buffered_margin:
            max_qty = max(1, math.floor(available / buffered_margin))
            if max_qty < og.target_qty:
                self._logger.log(
                    "margin_qty_cap", group_id=og.group_id,
                    original_qty=og.target_qty, capped_qty=max_qty,
                    available=round(available, 2),
                )
                og.target_qty = max_qty
            self._submit_order(og, daily_state)
            return og

        # Can't afford: evaluate displacement
        new_dist = abs(og.entry_price - self._last_price) if self._last_price else 0

        suspendable = []
        for existing in daily_state.pending_orders:
            if existing.state == "SUBMITTED" and existing.filled_qty == 0:
                dist = abs(existing.entry_price - self._last_price) if self._last_price else 0
                suspendable.append((existing, dist))

        candidates = list(suspendable)
        candidates.append((og, new_dist))
        candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates[0][0] is og:
            self._suspend_new_order(og, daily_state, "farthest_from_price")
            return og

        freed = 0.0
        to_suspend = []
        for existing, dist in candidates:
            if existing is og:
                continue
            if available + freed >= buffered_margin:
                break
            freed += existing.target_qty * buffered_margin
            to_suspend.append(existing)

        total_available = available + freed
        max_qty = max(1, math.floor(total_available / buffered_margin))
        if max_qty < 1:
            self._suspend_new_order(og, daily_state,
                                    "insufficient_margin_after_all_suspensions")
            return og

        for existing in to_suspend:
            self._displacement_log.append({
                "action": "suspend",
                "suspended_id": existing.group_id,
                "suspended_entry": existing.entry_price,
                "for_id": og.group_id,
                "for_entry": og.entry_price,
                "reason": f"farther ({abs(existing.entry_price - self._last_price):.1f} "
                          f"vs {new_dist:.1f} pts from price {self._last_price})",
                "time": self._now_iso(),
            })
            daily_state.move_to_suspended(
                existing.group_id,
                f"margin_priority: {og.group_id} closer "
                f"({new_dist:.1f} vs {abs(existing.entry_price - self._last_price):.1f} pts)",
            )
            self._logger.log(
                "order_suspended", group_id=existing.group_id,
                entry=existing.entry_price, qty=existing.target_qty,
                reason=f"margin_priority for {og.group_id}",
            )

        og.target_qty = min(og.target_qty, max_qty)
        self._submit_order(og, daily_state)
        self._logger.log(
            "margin_priority_resolved", group_id=og.group_id,
            suspended_count=len(to_suspend),
            suspended_ids=[o.group_id for o in to_suspend],
            final_qty=og.target_qty,
        )
        return og

    def _submit_order(self, og, daily_state):
        """Move order to SUBMITTED and add to pending."""
        og.state = "SUBMITTED"
        og.submitted_at = self._now_iso()
        daily_state.pending_orders.append(og)
        self._entry_volume[og.group_id] = 0  # Reset volume accumulator
        self._logger.log(
            "order_placed", mode="REPLAY", group_id=og.group_id,
            setup=og.setup, side=og.side, entry=og.entry_price,
            stop=og.stop_price, target=og.target_price,
            risk_pts=og.risk_pts, n=og.n_value, contracts=og.target_qty,
        )

    def _suspend_new_order(self, og, daily_state, reason):
        """Suspend a not-yet-placed order."""
        og.state = "SUSPENDED"
        og.suspended_at = self._now_iso()
        og.suspend_reason = reason
        daily_state.suspended_orders.append(og)
        self._displacement_log.append({
            "action": "suspend_new",
            "suspended_id": og.group_id,
            "suspended_entry": og.entry_price,
            "reason": reason,
            "time": self._now_iso(),
        })
        self._logger.log(
            "order_suspended", group_id=og.group_id,
            entry=og.entry_price, qty=og.target_qty, reason=reason,
        )

    def suspend_order(self, og, daily_state, reason="margin_priority"):
        """Suspend a SUBMITTED order."""
        if og.state != "SUBMITTED" or og.filled_qty > 0:
            return
        daily_state.move_to_suspended(og.group_id, reason)
        self._entry_volume.pop(og.group_id, None)
        self._logger.log(
            "order_suspended", group_id=og.group_id,
            entry=og.entry_price, qty=og.target_qty, reason=reason,
        )
        self._state_mgr.save(daily_state)

    def reactivate_order(self, og, daily_state):
        """Re-place a SUSPENDED order (no trade_count increment)."""
        reactivated = daily_state.move_suspended_to_pending(og.group_id)
        if reactivated is None:
            return None
        self._displacement_log.append({
            "action": "reactivate",
            "group_id": reactivated.group_id,
            "entry": reactivated.entry_price,
            "time": self._now_iso(),
        })
        daily_state.pending_orders = [
            o for o in daily_state.pending_orders if o.group_id != reactivated.group_id
        ]
        self._place_internal(reactivated, daily_state, is_reactivation=True)
        self._logger.log(
            "order_reactivated", group_id=reactivated.group_id,
            entry=reactivated.entry_price, qty=reactivated.target_qty,
        )
        return reactivated

    def cancel_order_group(self, og, daily_state, reason=CLOSE_CANCEL):
        """Cancel a single order group."""
        daily_state.move_to_closed(og.group_id, reason)
        self._entry_volume.pop(og.group_id, None)
        self._logger.log("order_cancelled", group_id=og.group_id, reason=reason)
        self._try_reactivate(daily_state)

    def cancel_all_pending(self, daily_state, reason=CLOSE_EOD):
        """Cancel all unfilled entry orders."""
        to_cancel = [og for og in daily_state.pending_orders
                     if og.state in ("SUBMITTED", "PARTIAL")]
        for og in to_cancel:
            daily_state.move_to_closed(og.group_id, reason)
            self._entry_volume.pop(og.group_id, None)
        if to_cancel:
            self._logger.log("eod_cancel", count=len(to_cancel), reason=reason)

    def flatten_all(self, daily_state, reason=CLOSE_FLATTEN):
        """Close all open positions at last known price."""
        self.cancel_all_pending(daily_state, reason)
        for og in list(daily_state.suspended_orders):
            daily_state.move_to_closed(og.group_id, reason)
        for og in list(daily_state.open_positions):
            self._close_position(og, daily_state, self._last_price, reason)
        self._logger.log("flatten", reason=reason,
                         positions=len(daily_state.closed_trades))

    # ── Tick-accurate fill checks ────────────────────────────────────────

    def on_tick(self, price, size, time_str, daily_state):
        """
        Check fills against a single Databento trade tick.

        Args:
            price: trade price
            size: trade size (contracts)
            time_str: timestamp string
            daily_state: current DailyState

        Fill model:
          Entry (limit):  Accumulate volume at exact limit price until
                         cumulative size >= order qty → fill at limit price.
          Stop:           First trade at/through stop → fill at ACTUAL trade
                         price (real slippage, may be worse than stop).
          TP (limit):     First trade at/through target → fill at target price.
        """
        self._last_price = price

        # 1. Entry fills — volume-aware limit simulation
        for og in list(daily_state.pending_orders):
            if og.state != "SUBMITTED":
                continue

            touches_entry = False
            if og.side == "BUY" and price <= og.entry_price:
                touches_entry = True
            elif og.side == "SELL" and price >= og.entry_price:
                touches_entry = True

            if not touches_entry:
                continue

            # Exact price match → accumulate volume for limit fill
            if abs(price - og.entry_price) < 1e-6:
                acc = self._entry_volume.get(og.group_id, 0) + size
                self._entry_volume[og.group_id] = acc
                if acc >= og.target_qty:
                    self._fill_entry(og, og.entry_price, time_str,
                                     daily_state, volume_at_fill=acc)
            else:
                # Price traded THROUGH our limit (marketable) → immediate fill
                self._fill_entry(og, og.entry_price, time_str,
                                 daily_state, volume_at_fill=size)

        # 2. Exit fills — tick-accurate, no same-bar ambiguity
        for og in list(daily_state.open_positions):
            if og.side == "BUY":
                # SL: first trade at/through stop → fill at actual trade price
                if price <= og.stop_price:
                    self._fill_exit(og, price, time_str, CLOSE_SL,
                                    daily_state, slippage_pts=og.stop_price - price)
                # TP: first trade at/through target → fill at target (limit)
                elif price >= og.target_price:
                    self._fill_exit(og, og.target_price, time_str, CLOSE_TP,
                                    daily_state, slippage_pts=0.0)
            else:  # SELL
                if price >= og.stop_price:
                    self._fill_exit(og, price, time_str, CLOSE_SL,
                                    daily_state, slippage_pts=price - og.stop_price)
                elif price <= og.target_price:
                    self._fill_exit(og, og.target_price, time_str, CLOSE_TP,
                                    daily_state, slippage_pts=0.0)

    def on_bar(self, bar_dict, daily_state):
        """Fallback: check fills against bar OHLC (when tick data unavailable).

        Uses conservative SL-priority model (same as backtester).
        """
        high = bar_dict["high"]
        low = bar_dict["low"]
        close = bar_dict["close"]
        bar_time = str(bar_dict.get("date", ""))
        self._last_price = close

        for og in list(daily_state.pending_orders):
            if og.state != "SUBMITTED":
                continue
            filled = False
            if og.side == "BUY" and low <= og.entry_price:
                filled = True
            elif og.side == "SELL" and high >= og.entry_price:
                filled = True
            if filled:
                self._fill_entry(og, og.entry_price, bar_time, daily_state)
                self._check_bar_exit(og, high, low, bar_time, daily_state)

        for og in list(daily_state.open_positions):
            self._check_bar_exit(og, high, low, bar_time, daily_state)

    # ── Displacement log ─────────────────────────────────────────────────

    def get_displacement_log(self):
        """Return the margin displacement event log for validation."""
        return list(self._displacement_log)

    # ── Internal fill mechanics ──────────────────────────────────────────

    def _fill_entry(self, og, fill_price, fill_time, daily_state,
                    volume_at_fill=None):
        """Process entry fill: pending → open."""
        og.filled_qty = og.target_qty
        og.filled_at = fill_time
        og.actual_entry_price = fill_price
        og.entry_slippage_pts = 0.0

        result = daily_state.move_to_open(og.group_id)
        if result is None:
            return

        self._entry_volume.pop(og.group_id, None)

        self._logger.log(
            "order_filled", group_id=og.group_id, setup=og.setup,
            side=og.side, qty=og.filled_qty, avg_price=fill_price,
            slippage_pts=0.0, volume_at_fill=volume_at_fill,
            mode="REPLAY",
        )

    def _fill_exit(self, og, exit_price, exit_time, reason, daily_state,
                   slippage_pts=0.0):
        """Process TP/SL exit: open → closed with P&L.

        For stops, exit_price is the actual trade price (may differ from
        stop_price due to gap-through → real slippage).
        """
        qty = og.filled_qty or og.target_qty
        entry = og.actual_entry_price or og.entry_price

        if og.side == "BUY":
            pnl_pts = exit_price - entry
        else:
            pnl_pts = entry - exit_price

        commission = _calc_commission(qty, self._monthly_contracts)
        self._monthly_contracts += qty
        gross_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
        net_pnl = round(gross_pnl - commission, 2)

        og.actual_exit_price = exit_price
        daily_state.move_to_closed(og.group_id, reason, net_pnl)

        event = "tp_filled" if reason == CLOSE_TP else "sl_filled"
        self._logger.log(
            event, group_id=og.group_id, setup=og.setup,
            side=og.side, entry_price=entry, exit_price=exit_price,
            stop_price=og.stop_price, qty=qty,
            pnl_pts=round(pnl_pts, 2), gross_pnl=gross_pnl,
            commission=commission, net_pnl=net_pnl,
            slippage_pts=round(slippage_pts, 2),
            mode="REPLAY",
        )

        self._state_mgr.save(daily_state)
        self._check_kill_switch(daily_state)
        self._try_reactivate(daily_state)

    def _check_bar_exit(self, og, high, low, bar_time, daily_state):
        """Check TP/SL on a bar. SL has priority (conservative, matches backtester)."""
        if og.side == "BUY":
            if low <= og.stop_price:
                self._fill_exit(og, og.stop_price, bar_time, CLOSE_SL, daily_state)
            elif high >= og.target_price:
                self._fill_exit(og, og.target_price, bar_time, CLOSE_TP, daily_state)
        else:
            if high >= og.stop_price:
                self._fill_exit(og, og.stop_price, bar_time, CLOSE_SL, daily_state)
            elif low <= og.target_price:
                self._fill_exit(og, og.target_price, bar_time, CLOSE_TP, daily_state)

    def _close_position(self, og, daily_state, price, reason):
        """Flatten a single position at given price."""
        qty = og.filled_qty or og.target_qty
        entry = og.actual_entry_price or og.entry_price
        if og.side == "BUY":
            pnl_pts = price - entry
        else:
            pnl_pts = entry - price
        commission = _calc_commission(qty, self._monthly_contracts)
        self._monthly_contracts += qty
        gross_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
        net_pnl = round(gross_pnl - commission, 2)
        og.actual_exit_price = price
        daily_state.move_to_closed(og.group_id, reason, net_pnl)
        self._logger.log(
            "eod_exit", group_id=og.group_id, setup=og.setup,
            side=og.side, entry_price=entry, exit_price=price,
            exit_reason=reason, qty=qty, pnl_pts=round(pnl_pts, 2),
            net_pnl=net_pnl, mode="REPLAY",
        )

    def _check_kill_switch(self, daily_state):
        """Evaluate kill switch: (realized + worst_unrealized) / start <= threshold."""
        if daily_state.kill_switch_active or daily_state.start_balance <= 0:
            return
        worst_unrealized = sum(
            og.risk_pts * (og.filled_qty or og.target_qty) * POINT_VALUE
            for og in daily_state.open_positions
        )
        total_worst = daily_state.realized_pnl - worst_unrealized
        pct = total_worst / daily_state.start_balance
        if pct <= self._config.kill_switch_pct:
            daily_state.kill_switch_active = True
            daily_state.kill_switch_reason = (
                f"Replay kill switch: {pct:.1%} "
                f"(realized {daily_state.realized_pnl:.0f} + "
                f"worst_unrealized {-worst_unrealized:.0f})"
            )
            self._logger.log(
                "kill_switch", action="triggered",
                reason=daily_state.kill_switch_reason,
                pnl=daily_state.realized_pnl,
            )

    def _try_reactivate(self, daily_state):
        """Try to reactivate suspended orders after margin freed."""
        if not daily_state.suspended_orders:
            return
        current_price = self._last_price
        if current_price <= 0:
            return

        candidates = sorted(
            list(daily_state.suspended_orders),
            key=lambda og: abs(og.entry_price - current_price),
        )
        reactivated = 0
        for og in candidates:
            fvg = self._find_active_fvg(og.fvg_id, daily_state)
            if fvg is None:
                daily_state.move_to_closed(og.group_id, "FVG_EXPIRED")
                self._displacement_log.append({
                    "action": "reactivate_blocked",
                    "group_id": og.group_id,
                    "reason": "FVG expired",
                    "time": self._now_iso(),
                })
                continue

            margin_per = self._config.margin_fallback_per_contract
            buffer = self._config.margin_buffer_pct
            buffered = margin_per * (1.0 + buffer)
            balance = daily_state.start_balance + daily_state.realized_pnl
            used = sum(
                (o.filled_qty or o.target_qty) * buffered
                for o in daily_state.pending_orders + daily_state.open_positions
            )
            available = balance - used
            if available < buffered:
                break

            max_qty = max(1, math.floor(available / buffered))
            og.target_qty = min(og.target_qty, max_qty)
            result = self.reactivate_order(og, daily_state)
            if result:
                reactivated += 1

        if reactivated:
            self._logger.log("suspended_reactivated", count=reactivated)

    def _find_active_fvg(self, fvg_id, daily_state):
        """Find an active FVG in daily state."""
        for fvg in daily_state.active_fvgs:
            if fvg.fvg_id == fvg_id and not fvg.is_mitigated:
                return fvg
        return None
