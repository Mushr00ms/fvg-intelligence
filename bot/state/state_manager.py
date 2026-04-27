"""
state_manager.py — JSON state persistence and crash recovery.

Writes bot_state.json atomically every 30 seconds and on state changes.
On startup, loads state if today's date matches, then reconciles with IB.
"""

import json
import os
import time
from datetime import datetime, date

import pytz

from bot.state.trade_state import DailyState, STATE_VERSION

NY_TZ = pytz.timezone("America/New_York")


class StateManager:
    """
    Manages bot state persistence to disk.

    Features:
    - Atomic writes (tmp file + os.replace)
    - Debounced saves (max once per second for event-triggered saves)
    - Periodic saves (every 30s via engine scheduler)
    - Crash recovery: load + reconcile with IB
    """

    def __init__(self, state_dir, logger=None, clock=None):
        self._state_dir = state_dir
        self._logger = logger
        self._clock = clock
        self._state_path = os.path.join(state_dir, "bot_state.json")
        self._last_save_time = 0
        self._dirty = False
        os.makedirs(state_dir, exist_ok=True)

    def _now(self):
        if self._clock is not None:
            return self._clock.now()
        return datetime.now(NY_TZ)

    def save(self, daily_state, force=False):
        """
        Save state to disk. Debounced to max once per second unless force=True.
        """
        now = time.time()
        if not force and (now - self._last_save_time) < 1.0:
            self._dirty = True
            return

        daily_state.last_updated = self._now().isoformat()
        data = daily_state.to_dict()

        # Atomic write
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, self._state_path)

        self._last_save_time = now
        self._dirty = False

    def save_if_dirty(self, daily_state):
        """Save if there are pending changes from debounced saves."""
        if self._dirty:
            self.save(daily_state, force=True)

    def load(self):
        """
        Load state from disk if file exists and date matches today.

        Returns:
            DailyState if found and current, None otherwise.
        """
        if not os.path.exists(self._state_path):
            return None

        try:
            with open(self._state_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            if self._logger:
                self._logger.log("state_load_error", error=str(e))
            return None

        # Check date
        today = self._now().strftime("%Y-%m-%d")
        if data.get("date") != today:
            if self._logger:
                self._logger.log(
                    "state_stale",
                    state_date=data.get("date"),
                    today=today,
                )
            return None

        # Check state schema version
        file_version = data.get("version", "0.0")
        if file_version != STATE_VERSION:
            if self._logger:
                self._logger.log(
                    "state_version_mismatch",
                    file_version=file_version,
                    expected=STATE_VERSION,
                )
            # Back up old state before migration
            backup_path = self._state_path + f".v{file_version}.bak"
            try:
                import shutil
                shutil.copy2(self._state_path, backup_path)
            except OSError:
                pass
            data = self._migrate_state(data, file_version)

        state = DailyState.from_dict(data)

        if self._logger:
            self._logger.log(
                "state_loaded",
                date=state.date,
                realized_pnl=state.realized_pnl,
                trade_count=state.trade_count,
                kill_switch=state.kill_switch_active,
                active_fvgs=len(state.active_fvgs),
                pending_orders=len(state.pending_orders),
                open_positions=len(state.open_positions),
                suspended_orders=len(state.suspended_orders),
            )

        return state

    def create_new(self, start_balance):
        """Create a fresh DailyState for today."""
        today = self._now().strftime("%Y-%m-%d")
        state = DailyState(
            date=today,
            start_balance=start_balance,
        )
        if self._logger:
            self._logger.log(
                "state_created",
                date=today,
                start_balance=start_balance,
            )
        return state

    def reconcile_with_ib(self, daily_state, ib_orders, ib_positions):
        """
        Reconcile saved state with IB's actual orders and positions.
        IB is the source of truth.

        Cases:
        1. State has order ID, IB doesn't → remove from state (never placed / crash)
        2. IB has order not in state → log warning (add if matches our contract)
        3. Both have it → update state to match IB status

        Args:
            daily_state: DailyState from disk
            ib_orders: list from ib.openOrders()
            ib_positions: list from ib.positions()

        Returns:
            Reconciled DailyState
        """
        ib_order_ids = set()
        for order in ib_orders:
            if hasattr(order, "orderId"):
                ib_order_ids.add(str(order.orderId))
            elif hasattr(order, "order_id"):
                ib_order_ids.add(str(order.order_id))

        # Check pending orders against broker open orders AND positions.
        # With Tradovate OSO, after entry fills only TP/SL remain as working
        # orders — the entry itself disappears. A pending order whose entry
        # filled while offline must be promoted to OPEN, not orphaned.
        orphaned = []
        promoted = []
        for og in daily_state.pending_orders:
            if not og.broker_entry_order_id:
                continue
            if og.broker_entry_order_id in ib_order_ids:
                continue
            # Entry not in working orders — either never placed, or already filled.
            # Check if TP or SL are still working (proves entry filled + exits active).
            tp_alive = og.broker_tp_order_id and og.broker_tp_order_id in ib_order_ids
            sl_alive = og.broker_sl_order_id and og.broker_sl_order_id in ib_order_ids
            if tp_alive or sl_alive:
                promoted.append(og.group_id)
            else:
                orphaned.append(og.group_id)

        for gid in promoted:
            og = next((o for o in daily_state.pending_orders if o.group_id == gid), None)
            if og:
                og.filled_qty = og.target_qty
                og.filled_at = og.submitted_at
                og.actual_entry_price = og.entry_price
                daily_state.move_to_open(gid)
                if self._logger:
                    self._logger.log(
                        "reconciliation",
                        action="promoted_to_open",
                        group_id=gid,
                        note="entry filled while offline, TP/SL still working",
                    )

        for gid in orphaned:
            daily_state.move_to_closed(gid, "RECONCILE_ORPHAN")
            if self._logger:
                self._logger.log(
                    "reconciliation",
                    action="removed_orphan",
                    group_id=gid,
                )

        # Validate open positions against IB — detect positions that closed
        # while the bot was down (TP/SL filled without callback reaching us).
        # IB positions are net per-contract; we check if our expected side
        # has any quantity at IB.
        ib_net_position = 0
        for pos in ib_positions:
            # Support both raw IB Position (.position) and PositionSnapshot (.quantity/.side)
            if hasattr(pos, 'position'):
                ib_net_position += int(pos.position)
            elif hasattr(pos, 'quantity'):
                signed = int(pos.quantity)
                if hasattr(pos, 'side') and pos.side == "SELL":
                    signed = -signed
                ib_net_position += signed

        # Count what our state thinks we hold
        state_net_position = 0
        for og in daily_state.open_positions:
            qty = og.filled_qty or og.target_qty
            if og.side == "BUY":
                state_net_position += qty
            else:
                state_net_position -= qty

        stale_positions = []
        if daily_state.open_positions and ib_net_position == 0:
            # IB flat but we think we have positions — they were closed without us
            stale_positions = [og.group_id for og in daily_state.open_positions]
        elif daily_state.open_positions and abs(ib_net_position) < abs(state_net_position):
            # IB has fewer contracts than we expect — some positions closed.
            # Close positions from the oldest until our count matches IB.
            excess = abs(state_net_position) - abs(ib_net_position)
            sorted_positions = sorted(
                daily_state.open_positions,
                key=lambda og: og.filled_at or og.submitted_at or "",
            )
            closed_qty = 0
            for og in sorted_positions:
                if closed_qty >= excess:
                    break
                stale_positions.append(og.group_id)
                closed_qty += og.filled_qty or og.target_qty

        for gid in stale_positions:
            og = next((o for o in daily_state.open_positions if o.group_id == gid), None)
            est_pnl = 0.0
            exit_type = "unknown"
            if og:
                tp_alive = og.broker_tp_order_id and og.broker_tp_order_id in ib_order_ids
                sl_alive = og.broker_sl_order_id and og.broker_sl_order_id in ib_order_ids
                qty = og.filled_qty or og.target_qty
                if not tp_alive and sl_alive:
                    exit_type = "TP"
                    og.actual_exit_price = og.target_price
                    pts = (og.target_price - og.entry_price) if og.side == "BUY" else (og.entry_price - og.target_price)
                    est_pnl = round(pts * qty * 20, 2)
                elif not sl_alive and tp_alive:
                    exit_type = "SL"
                    og.actual_exit_price = og.stop_price
                    pts = (og.stop_price - og.entry_price) if og.side == "BUY" else (og.entry_price - og.stop_price)
                    est_pnl = round(pts * qty * 20, 2)
            daily_state.move_to_closed(gid, "RECONCILE_IB_CLOSED", est_pnl)
            if self._logger:
                self._logger.log(
                    "reconciliation",
                    action="position_closed_at_ib",
                    group_id=gid,
                    exit_type=exit_type,
                    est_pnl=est_pnl,
                    note="TP/SL likely filled while bot was offline",
                )

        if self._logger:
            self._logger.log(
                "reconciliation",
                action="complete",
                orphans_removed=len(orphaned),
                stale_positions_closed=len(stale_positions),
                ib_orders=len(ib_orders),
                ib_positions=len(ib_positions),
                ib_net_qty=ib_net_position,
                state_net_qty=state_net_position,
            )

        return daily_state

    def _migrate_state(self, data, old_version):
        """Migrate state data from old schema versions.

        Each version bump should add a migration step here.
        Migrations are applied sequentially: 0.0 → 1.0 → future versions.
        """
        if old_version == "0.0":
            # Pre-versioned state: add version field, all fields are compatible
            data["version"] = "1.0"
            if self._logger:
                self._logger.log("state_migrated", from_version="0.0", to_version="1.0")

        if data.get("version") == "1.0":
            # v1.1: Add suspended_orders list and new OrderGroup fields
            data["suspended_orders"] = []
            for key in ("pending_orders", "open_positions", "closed_trades"):
                for og in data.get(key, []):
                    og.setdefault("suspended_at", None)
                    og.setdefault("suspend_reason", "")
            data["version"] = "1.1"
            if self._logger:
                self._logger.log("state_migrated", from_version="1.0", to_version="1.1")

        if data.get("version") == "1.1":
            # v1.2: Rename ib_*_order_id to broker_*_order_id (int→str)
            for key in ("pending_orders", "open_positions", "closed_trades", "suspended_orders"):
                for og in data.get(key, []):
                    for old, new in (
                        ("ib_entry_order_id", "broker_entry_order_id"),
                        ("ib_tp_order_id", "broker_tp_order_id"),
                        ("ib_sl_order_id", "broker_sl_order_id"),
                    ):
                        if old in og and new not in og:
                            val = og.pop(old)
                            og[new] = str(val) if val is not None else None
            data["version"] = "1.2"
            if self._logger:
                self._logger.log("state_migrated", from_version="1.1", to_version="1.2")

        if data.get("version") == "1.2":
            # v1.3: Persist broker exit market-order id for flatten/EOD fills.
            for key in ("pending_orders", "open_positions", "closed_trades", "suspended_orders"):
                for og in data.get(key, []):
                    og.setdefault("broker_exit_order_id", None)
            data["version"] = "1.3"
            if self._logger:
                self._logger.log("state_migrated", from_version="1.2", to_version="1.3")

        return data
