"""
Tests for the Intelligent Margin Management system.

Covers:
- MarginTracker: margin fetching, available margin, contract affordability
- MarginPriorityManager: suspend/reactivate logic, priority ranking
- State transitions: SUSPENDED state, move_to_suspended, move_suspended_to_pending
- E2E: full detection → margin evaluation → suspend/place → reactivate flow
- State migration: v1.0 → v1.1

Uses the project's standard test patterns: no unittest.mock, hand-written stubs,
real RiskGates/StrategyLoader in DRY_RUN mode.
"""

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest
import pytz

from bot.bot_config import BotConfig
from bot.risk.margin_tracker import MarginTracker
from bot.risk.margin_priority import MarginPriorityManager
from bot.risk.risk_gates import RiskGates, GateResult
from bot.risk.time_gates import TimeGates
from bot.state.trade_state import (
    DailyState, OrderGroup, FVGRecord, _new_id, STATE_VERSION,
    CLOSE_TP, CLOSE_SL, CLOSE_EOD,
)

NY = pytz.timezone("America/New_York")


# ── Stubs ────────────────────────────────────────────────────────────────────


class _CaptureLogger:
    def __init__(self):
        self.records = []

    def log(self, event, **kwargs):
        self.records.append({"event": event, **kwargs})

    def events(self, name):
        return [r for r in self.records if r["event"] == name]


class _FakeClock:
    def __init__(self, hour=10, minute=30):
        self._now = NY.localize(datetime(2026, 3, 26, hour, minute))

    def now(self):
        return self._now

    def advance(self, minutes=5):
        self._now += timedelta(minutes=minutes)


class _NoOpStateMgr:
    def __init__(self):
        self.save_count = 0

    def save(self, state, force=False):
        self.save_count += 1


class _FakeIBConn:
    """Fake IB connection with configurable account values."""
    is_connected = True

    def __init__(self, available_funds=100000.0):
        self._available_funds = available_funds
        self._cancelled_order_ids = []
        self.ib = _FakeIB(available_funds, self._cancelled_order_ids)


class _FakeIB:
    """Fake ib_async.IB with accountValues and openTrades."""
    def __init__(self, available_funds, cancelled_ids):
        self._available_funds = available_funds
        self._cancelled_ids = cancelled_ids
        self._next_id = 100

    def set_available_funds(self, amount):
        self._available_funds = amount

    def accountValues(self):
        return [
            _FakeAccountValue("AvailableFunds-C", str(self._available_funds), "USD"),
            _FakeAccountValue("NetLiquidation", "100000", "USD"),
        ]

    def openTrades(self):
        return []

    def cancelOrder(self, order):
        self._cancelled_ids.append(getattr(order, "orderId", None))

    class client:
        _id_counter = 100

        @classmethod
        def getReqId(cls):
            cls._id_counter += 1
            return cls._id_counter


@dataclass
class _FakeAccountValue:
    tag: str
    value: str
    currency: str
    account: str = ""


class _FakeContract:
    symbol = "NQ"
    conId = 12345


class _FakeOrderManager:
    """Tracks place/suspend/reactivate calls without IB."""
    def __init__(self):
        self.placed = []
        self.suspended = []
        self.reactivated = []
        self._on_order_resolved = None

    async def place_bracket(self, order_group, daily_state):
        order_group.state = "SUBMITTED"
        order_group.submitted_at = datetime.now().isoformat()
        daily_state.pending_orders.append(order_group)
        daily_state.trade_count += 1
        self.placed.append(order_group.group_id)
        return order_group

    async def suspend_order(self, og, daily_state, reason=""):
        if og.state != "SUBMITTED" or og.filled_qty > 0:
            return
        daily_state.move_to_suspended(og.group_id, reason)
        self.suspended.append(og.group_id)

    async def reactivate_order(self, og, daily_state):
        reactivated = daily_state.move_suspended_to_pending(og.group_id)
        if reactivated is None:
            return None
        reactivated.state = "SUBMITTED"
        reactivated.submitted_at = datetime.now().isoformat()
        daily_state.pending_orders.append(reactivated)
        self.reactivated.append(og.group_id)
        return reactivated


# ── Helpers ──────────────────────────────────────────────────────────────────


def _round_tick(p):
    return round(p * 4) / 4


def _make_config(**overrides):
    defaults = dict(
        dry_run=True,
        paper_mode=True,
        ib_port=7497,
        risk_per_trade=0.01,
        use_risk_tiers=False,
        max_concurrent=3,
        max_daily_trades=15,
        kill_switch_pct=-0.03,
        max_cumulative_risk_pct=0.05,
        point_value=20.0,
        tick_size=0.25,
        max_trade_loss_pct=0.015,
        margin_intraday_maintenance=25000.0,
        margin_overnight_initial=50000.0,
        margin_fallback_per_contract=25000.0,
        margin_buffer_pct=0.05,
        margin_refresh_interval=1800,
        margin_management_enabled=True,
        state_dir=tempfile.mkdtemp(),
        log_dir=tempfile.mkdtemp(),
        strategy_dir=tempfile.mkdtemp(),
    )
    defaults.update(overrides)
    return BotConfig(**defaults)


def _make_state(balance=100000.0, **kwargs):
    defaults = dict(date="2026-03-26", start_balance=balance)
    defaults.update(kwargs)
    return DailyState(**defaults)


def _make_order(entry_price=24000.0, risk_pts=12.0, qty=2, side="BUY",
                state="SUBMITTED", fvg_id=None, **kwargs):
    defaults = dict(
        group_id=_new_id(),
        fvg_id=fvg_id or _new_id(),
        setup="mit_extreme",
        side=side,
        entry_price=entry_price,
        stop_price=entry_price - risk_pts if side == "BUY" else entry_price + risk_pts,
        target_price=entry_price + risk_pts * 2 if side == "BUY" else entry_price - risk_pts * 2,
        risk_pts=risk_pts,
        n_value=2.0,
        target_qty=qty,
        state=state,
    )
    defaults.update(kwargs)
    return OrderGroup(**defaults)


def _make_fvg(fvg_id=None, zone_low=24000.0, zone_high=24010.0,
              time_period="10:30-11:00"):
    return FVGRecord(
        fvg_id=fvg_id or _new_id(),
        fvg_type="bullish",
        zone_low=zone_low, zone_high=zone_high,
        time_candle1="2026-03-26T10:30:00-04:00",
        time_candle2="2026-03-26T10:35:00-04:00",
        time_candle3="2026-03-26T10:40:00-04:00",
        middle_open=24005.0, middle_low=23998.0, middle_high=24015.0,
        first_open=23995.0,
        time_period=time_period,
        formation_date="2026-03-26",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRADE STATE: SUSPENDED transitions
# ══════════════════════════════════════════════════════════════════════════════


class TestSuspendedState:
    def test_move_to_suspended(self):
        state = _make_state()
        order = _make_order()
        state.pending_orders.append(order)

        result = state.move_to_suspended(order.group_id, "margin_priority")

        assert result is not None
        assert result.state == "SUSPENDED"
        assert result.suspend_reason == "margin_priority"
        assert result.suspended_at is not None
        assert len(state.pending_orders) == 0
        assert len(state.suspended_orders) == 1

    def test_move_suspended_to_pending(self):
        state = _make_state()
        order = _make_order(state="SUSPENDED")
        order.suspended_at = "2026-03-26T10:00:00"
        order.suspend_reason = "test"
        state.suspended_orders.append(order)

        result = state.move_suspended_to_pending(order.group_id)

        assert result is not None
        assert result.state == "PENDING"
        assert result.suspended_at is None
        assert result.suspend_reason == ""
        assert len(state.suspended_orders) == 0

    def test_move_suspended_to_closed(self):
        state = _make_state()
        order = _make_order(state="SUSPENDED")
        state.suspended_orders.append(order)

        result = state.move_to_closed(order.group_id, "EOD")

        assert result is not None
        assert result.state == "CLOSED"
        assert result.close_reason == "EOD"
        assert len(state.suspended_orders) == 0
        assert len(state.closed_trades) == 1

    def test_suspended_not_in_active_order_count(self):
        state = _make_state()
        state.pending_orders.append(_make_order())
        state.open_positions.append(_make_order(state="FILLED"))
        state.suspended_orders.append(_make_order(state="SUSPENDED"))

        assert state.active_order_count == 2  # pending + open, NOT suspended

    def test_suspended_not_found_returns_none(self):
        state = _make_state()
        assert state.move_to_suspended("nonexistent") is None
        assert state.move_suspended_to_pending("nonexistent") is None

    def test_serialization_round_trip(self):
        state = _make_state()
        order = _make_order(state="SUSPENDED")
        order.suspended_at = "2026-03-26T10:00:00"
        order.suspend_reason = "margin_priority"
        state.suspended_orders.append(order)

        data = state.to_dict()
        restored = DailyState.from_dict(data)

        assert len(restored.suspended_orders) == 1
        og = restored.suspended_orders[0]
        assert og.state == "SUSPENDED"
        assert og.suspended_at == "2026-03-26T10:00:00"
        assert og.suspend_reason == "margin_priority"


# ══════════════════════════════════════════════════════════════════════════════
# 2. STATE MIGRATION v1.0 → v1.1
# ══════════════════════════════════════════════════════════════════════════════


class TestStateMigration:
    def test_v10_to_v11_migration(self):
        from bot.state.state_manager import StateManager
        logger = _CaptureLogger()
        sm = StateManager(tempfile.mkdtemp(), logger=logger)

        v10_data = {
            "version": "1.0",
            "date": "2026-03-26",
            "start_balance": 100000.0,
            "realized_pnl": 500.0,
            "trade_count": 3,
            "kill_switch_active": False,
            "kill_switch_reason": "",
            "active_fvgs": [],
            "pending_orders": [{
                "group_id": "abc123", "fvg_id": "fvg1",
                "setup": "mit_extreme", "side": "BUY",
                "entry_price": 24000, "stop_price": 23988,
                "target_price": 24024, "risk_pts": 12,
                "n_value": 2.0, "target_qty": 3,
            }],
            "open_positions": [],
            "closed_trades": [],
            "last_updated": "",
        }

        migrated = sm._migrate_state(v10_data, "1.0")

        assert migrated["version"] == "1.1"
        assert "suspended_orders" in migrated
        assert migrated["suspended_orders"] == []
        # Existing OrderGroups get new fields
        og = migrated["pending_orders"][0]
        assert og["suspended_at"] is None
        assert og["suspend_reason"] == ""

    def test_v00_to_v11_migration(self):
        from bot.state.state_manager import StateManager
        sm = StateManager(tempfile.mkdtemp(), logger=_CaptureLogger())

        v00_data = {
            "date": "2026-03-26",
            "start_balance": 100000.0,
            "pending_orders": [],
            "open_positions": [],
            "closed_trades": [],
        }

        migrated = sm._migrate_state(v00_data, "0.0")

        assert migrated["version"] == "1.1"
        assert "suspended_orders" in migrated


# ══════════════════════════════════════════════════════════════════════════════
# 3. MARGIN TRACKER
# ══════════════════════════════════════════════════════════════════════════════


class TestMarginTracker:
    def test_dry_run_uses_intraday_fallback(self):
        config = _make_config(dry_run=True, margin_intraday_maintenance=30000.0)
        logger = _CaptureLogger()
        # Clock at 10:30 ET = intraday window
        tracker = MarginTracker(
            _FakeIBConn(), _FakeContract(), logger, config, clock=_FakeClock(10, 30)
        )

        margin = asyncio.new_event_loop().run_until_complete(tracker.initialize())

        assert margin == 30000.0
        assert tracker.margin_per_contract == 30000.0
        assert tracker.is_initialized

    def test_available_margin_dry_run(self):
        config = _make_config(dry_run=True, margin_intraday_maintenance=25000.0)
        tracker = MarginTracker(
            _FakeIBConn(), _FakeContract(), _CaptureLogger(), config,
            clock=_FakeClock(10, 30),
        )
        asyncio.new_event_loop().run_until_complete(tracker.initialize())

        avail = tracker.get_available_margin()
        # Dry run during intraday: intraday_maintenance * 3
        assert avail == 75000.0

    def test_max_contracts_by_margin(self):
        config = _make_config(
            dry_run=True,
            margin_intraday_maintenance=25000.0,
            margin_buffer_pct=0.0,  # No buffer for clean math
        )
        tracker = MarginTracker(
            _FakeIBConn(100000), _FakeContract(), _CaptureLogger(), config,
            clock=_FakeClock(10, 30),
        )
        asyncio.new_event_loop().run_until_complete(tracker.initialize())

        # 100k available / 25k per contract = 4 contracts
        assert tracker.max_contracts_by_margin(100000.0) == 4
        assert tracker.max_contracts_by_margin(50000.0) == 2
        assert tracker.max_contracts_by_margin(20000.0) == 0

    def test_max_contracts_with_buffer(self):
        config = _make_config(
            dry_run=True,
            margin_intraday_maintenance=25000.0,
            margin_buffer_pct=0.25,  # 25% buffer
        )
        tracker = MarginTracker(
            _FakeIBConn(), _FakeContract(), _CaptureLogger(), config,
            clock=_FakeClock(10, 30),
        )
        asyncio.new_event_loop().run_until_complete(tracker.initialize())

        # 100k / (25k * 1.25) = 100k / 31250 = 3.2 → 3
        assert tracker.max_contracts_by_margin(100000.0) == 3

    def test_can_afford(self):
        config = _make_config(
            dry_run=True,
            margin_intraday_maintenance=25000.0,
            margin_buffer_pct=0.0,
        )
        tracker = MarginTracker(
            _FakeIBConn(), _FakeContract(), _CaptureLogger(), config,
            clock=_FakeClock(10, 30),
        )
        asyncio.new_event_loop().run_until_complete(tracker.initialize())

        assert tracker.can_afford(2, 60000.0) is True
        assert tracker.can_afford(3, 60000.0) is False

    def test_margin_required_for(self):
        config = _make_config(dry_run=True, margin_intraday_maintenance=33000.0)
        tracker = MarginTracker(
            _FakeIBConn(), _FakeContract(), _CaptureLogger(), config,
            clock=_FakeClock(10, 30),
        )
        asyncio.new_event_loop().run_until_complete(tracker.initialize())

        assert tracker.margin_required_for(4) == 132000.0

    def test_overnight_fallback_outside_rth(self):
        config = _make_config(
            dry_run=True,
            margin_intraday_maintenance=22924.0,
            margin_overnight_initial=46373.0,
        )
        # Clock at 17:00 ET = outside intraday window
        tracker = MarginTracker(
            _FakeIBConn(), _FakeContract(), _CaptureLogger(), config,
            clock=_FakeClock(17, 0),
        )
        margin = asyncio.new_event_loop().run_until_complete(tracker.initialize())

        assert margin == 46373.0
        assert tracker.margin_per_contract == 46373.0

    def test_intraday_fallback_during_rth(self):
        config = _make_config(
            dry_run=True,
            margin_intraday_maintenance=22924.0,
            margin_overnight_initial=46373.0,
        )
        # Clock at 12:00 ET = inside intraday window
        tracker = MarginTracker(
            _FakeIBConn(), _FakeContract(), _CaptureLogger(), config,
            clock=_FakeClock(12, 0),
        )
        margin = asyncio.new_event_loop().run_until_complete(tracker.initialize())

        assert margin == 22924.0
        assert tracker.margin_per_contract == 22924.0


# ══════════════════════════════════════════════════════════════════════════════
# 4. MARGIN PRIORITY MANAGER
# ══════════════════════════════════════════════════════════════════════════════


def _make_margin_priority(available_funds=100000.0, margin_per_contract=25000.0,
                           buffer_pct=0.0):
    """Create a MarginPriorityManager with a fake margin tracker.

    Uses dry_run=False so the tracker reads AvailableFunds from the fake IB
    connection rather than returning the dry_run fallback. The _FakeOrderManager
    handles order placement without IB.
    """
    config = _make_config(
        dry_run=False,  # So MarginTracker reads from fake IB, not fallback
        margin_fallback_per_contract=margin_per_contract,
        margin_buffer_pct=buffer_pct,
    )
    logger = _CaptureLogger()
    clock = _FakeClock()

    conn = _FakeIBConn(available_funds)
    tracker = MarginTracker(conn, _FakeContract(), logger, config, clock=clock)
    # Manually set margin (skip whatIfOrder which needs real IB)
    tracker._margin_per_contract = margin_per_contract
    tracker._initialized = True
    tracker._last_fetch_time = clock.now()

    order_mgr = _FakeOrderManager()
    risk_gates = RiskGates(config)
    time_gates = TimeGates(config, clock=clock)

    priority = MarginPriorityManager(
        tracker, order_mgr, risk_gates, time_gates,
        logger, config, clock=clock,
    )
    return priority, order_mgr, logger, tracker


class TestMarginPriority:
    def test_happy_path_enough_margin(self):
        """Enough margin → order placed directly."""
        priority, order_mgr, logger, _ = _make_margin_priority(
            available_funds=100000, margin_per_contract=25000,
        )
        state = _make_state()
        fvg = _make_fvg()
        order = _make_order(entry_price=24000, qty=2, state="PENDING", fvg_id=fvg.fvg_id)

        result = asyncio.new_event_loop().run_until_complete(
            priority.evaluate_and_place(order, fvg, state, current_price=24050.0)
        )

        assert result == "PLACED"
        assert len(order_mgr.placed) == 1
        assert len(state.pending_orders) == 1

    def test_happy_path_caps_qty_by_margin(self):
        """If risk sizing gives 4 but margin only supports 2 → cap to 2."""
        priority, order_mgr, _, _ = _make_margin_priority(
            available_funds=55000, margin_per_contract=25000,
        )
        state = _make_state()
        fvg = _make_fvg()
        order = _make_order(entry_price=24000, qty=4, state="PENDING", fvg_id=fvg.fvg_id)

        result = asyncio.new_event_loop().run_until_complete(
            priority.evaluate_and_place(order, fvg, state, current_price=24050.0)
        )

        assert result == "PLACED"
        assert order.target_qty == 2  # Capped from 4 to 2

    def test_suspend_farthest_order(self):
        """New order closer to price → suspend the farther resting order.

        Simulates: first order consumed all margin (IB shows 0 available),
        then a closer FVG fires — should suspend the far order, freeing margin.
        """
        # available_funds=0: first order consumed all margin at IB
        priority, order_mgr, logger, tracker = _make_margin_priority(
            available_funds=0, margin_per_contract=25000,
        )
        state = _make_state()

        # Existing resting order: far from current price (24200 vs price at 24050)
        fvg_far = _make_fvg(zone_low=24200.0, zone_high=24210.0)
        far_order = _make_order(entry_price=24200.0, qty=1, fvg_id=fvg_far.fvg_id)
        far_order.submitted_at = datetime.now().isoformat()
        state.pending_orders.append(far_order)

        # New order: closer to price (24060 vs price at 24050)
        fvg_near = _make_fvg(zone_low=24060.0, zone_high=24070.0)
        new_order = _make_order(entry_price=24060.0, qty=1, state="PENDING",
                                fvg_id=fvg_near.fvg_id)

        # After suspension, IB will show freed margin
        original_avail = tracker._conn.ib._available_funds

        async def _run():
            # Monkey-patch: after suspend_order runs, simulate IB freeing margin
            orig_suspend = order_mgr.suspend_order
            async def _suspend_and_free(og, ds, reason=""):
                await orig_suspend(og, ds, reason)
                tracker._conn.ib.set_available_funds(30000)  # Margin freed
            order_mgr.suspend_order = _suspend_and_free

            return await priority.evaluate_and_place(
                new_order, fvg_near, state, current_price=24050.0
            )

        result = asyncio.new_event_loop().run_until_complete(_run())

        assert result == "PLACED_AFTER_SUSPEND"
        assert far_order.group_id in order_mgr.suspended
        assert new_order.group_id in order_mgr.placed
        assert len(state.suspended_orders) == 1
        assert state.suspended_orders[0].group_id == far_order.group_id

    def test_new_order_farthest_gets_suspended(self):
        """New order is farthest from price → suspend it, don't displace closer."""
        # available_funds=0: existing order consumed all margin
        priority, order_mgr, logger, _ = _make_margin_priority(
            available_funds=0, margin_per_contract=25000,
        )
        state = _make_state()

        # Existing resting order: closer to price (24060 vs price 24050)
        fvg_near = _make_fvg(zone_low=24060.0, zone_high=24070.0)
        near_order = _make_order(entry_price=24060.0, qty=1, fvg_id=fvg_near.fvg_id)
        near_order.submitted_at = datetime.now().isoformat()
        state.pending_orders.append(near_order)

        # New order: far from price (24300 vs price 24050)
        fvg_far = _make_fvg(zone_low=24300.0, zone_high=24310.0)
        new_order = _make_order(entry_price=24300.0, qty=1, state="PENDING",
                                fvg_id=fvg_far.fvg_id)

        result = asyncio.new_event_loop().run_until_complete(
            priority.evaluate_and_place(new_order, fvg_far, state, current_price=24050.0)
        )

        assert result == "SUSPENDED"
        assert len(order_mgr.suspended) == 0  # Existing was NOT suspended
        assert len(state.suspended_orders) == 1
        assert state.suspended_orders[0].group_id == new_order.group_id
        # Existing order untouched
        assert len(state.pending_orders) == 1
        assert state.pending_orders[0].group_id == near_order.group_id

    def test_partial_fill_not_suspended(self):
        """Orders in PARTIAL state (has fills) must NEVER be suspended."""
        # available_funds=0: margin exhausted
        priority, order_mgr, _, _ = _make_margin_priority(
            available_funds=0, margin_per_contract=25000,
        )
        state = _make_state()

        # Partially filled order: has 1 fill, state=PARTIAL
        fvg_partial = _make_fvg()
        partial_order = _make_order(entry_price=24060.0, qty=3, state="PARTIAL",
                                     fvg_id=fvg_partial.fvg_id)
        partial_order.filled_qty = 1
        state.pending_orders.append(partial_order)

        # New order that can't be placed (insufficient margin)
        fvg_new = _make_fvg(zone_low=24100.0, zone_high=24110.0)
        new_order = _make_order(entry_price=24100.0, qty=1, state="PENDING",
                                fvg_id=fvg_new.fvg_id)

        result = asyncio.new_event_loop().run_until_complete(
            priority.evaluate_and_place(new_order, fvg_new, state, current_price=24050.0)
        )

        # Partial order was NOT suspended (no SUBMITTED with 0 fills available)
        assert result == "SUSPENDED"
        assert len(order_mgr.suspended) == 0
        assert partial_order.state == "PARTIAL"  # Untouched

    def test_reactivate_closest_first(self):
        """When margin frees up, reactivate the closest suspended order first."""
        priority, order_mgr, logger, tracker = _make_margin_priority(
            available_funds=60000, margin_per_contract=25000,
        )
        state = _make_state()

        # Two suspended orders
        fvg1 = _make_fvg(fvg_id="fvg-close")
        fvg2 = _make_fvg(fvg_id="fvg-far")
        state.active_fvgs = [fvg1, fvg2]

        close_order = _make_order(entry_price=24060.0, qty=1, state="SUSPENDED",
                                   fvg_id="fvg-close")
        close_order.suspended_at = "2026-03-26T10:00:00"
        far_order = _make_order(entry_price=24200.0, qty=1, state="SUSPENDED",
                                 fvg_id="fvg-far")
        far_order.suspended_at = "2026-03-26T10:00:00"

        state.suspended_orders = [far_order, close_order]  # Reversed to test sorting

        count = asyncio.new_event_loop().run_until_complete(
            priority.try_reactivate_suspended(state, current_price=24050.0)
        )

        assert count == 2
        # Close order should be reactivated first (but both should be reactivated)
        assert close_order.group_id in order_mgr.reactivated
        assert far_order.group_id in order_mgr.reactivated

    def test_reactivate_skips_expired_fvg(self):
        """Suspended order whose FVG was mitigated → closed, not re-placed."""
        priority, order_mgr, logger, _ = _make_margin_priority(
            available_funds=60000, margin_per_contract=25000,
        )
        state = _make_state()
        state.active_fvgs = []  # FVG is gone (mitigated/expired)

        susp_order = _make_order(entry_price=24060.0, qty=1, state="SUSPENDED",
                                  fvg_id="expired-fvg")
        susp_order.suspended_at = "2026-03-26T10:00:00"
        state.suspended_orders = [susp_order]

        count = asyncio.new_event_loop().run_until_complete(
            priority.try_reactivate_suspended(state, current_price=24050.0)
        )

        assert count == 0
        assert len(state.suspended_orders) == 0
        assert len(state.closed_trades) == 1
        assert state.closed_trades[0].close_reason == "FVG_EXPIRED"

    def test_reactivate_stops_when_margin_exhausted(self):
        """Only reactivate as many as margin allows."""
        priority, order_mgr, _, tracker = _make_margin_priority(
            available_funds=30000, margin_per_contract=25000,
        )
        state = _make_state()

        fvg1 = _make_fvg(fvg_id="fvg1")
        fvg2 = _make_fvg(fvg_id="fvg2")
        state.active_fvgs = [fvg1, fvg2]

        order1 = _make_order(entry_price=24060.0, qty=1, state="SUSPENDED", fvg_id="fvg1")
        order1.suspended_at = "2026-03-26T10:00:00"
        order2 = _make_order(entry_price=24100.0, qty=1, state="SUSPENDED", fvg_id="fvg2")
        order2.suspended_at = "2026-03-26T10:00:00"
        state.suspended_orders = [order1, order2]

        # After first reactivation, reduce available funds to simulate margin consumption
        orig_reactivate = order_mgr.reactivate_order
        async def _reactivate_and_consume(og, ds):
            result = await orig_reactivate(og, ds)
            tracker._conn.ib.set_available_funds(5000)  # Margin consumed
            return result
        order_mgr.reactivate_order = _reactivate_and_consume

        count = asyncio.new_event_loop().run_until_complete(
            priority.try_reactivate_suspended(state, current_price=24050.0)
        )

        # Only 1 can be reactivated (after first, margin drops to 5k < 25k)
        assert count == 1
        assert len(order_mgr.reactivated) == 1

    def test_clear_all_suspended_eod(self):
        """EOD clears all suspended orders."""
        priority, _, logger, _ = _make_margin_priority()
        state = _make_state()

        state.suspended_orders = [
            _make_order(state="SUSPENDED"),
            _make_order(state="SUSPENDED"),
        ]

        count = asyncio.new_event_loop().run_until_complete(
            priority.clear_all_suspended(state, "EOD")
        )

        assert count == 2
        assert len(state.suspended_orders) == 0
        assert len(state.closed_trades) == 2
        assert all(t.close_reason == "EOD" for t in state.closed_trades)


# ══════════════════════════════════════════════════════════════════════════════
# 5. ORDER MANAGER: suspend/reactivate
# ══════════════════════════════════════════════════════════════════════════════


class TestOrderManagerSuspend:
    def test_suspend_submitted_order(self):
        """Suspending a SUBMITTED order moves it to suspended_orders."""
        from bot.execution.order_manager import OrderManager

        config = _make_config(dry_run=True)
        logger = _CaptureLogger()
        conn = _FakeIBConn()
        state_mgr = _NoOpStateMgr()
        mgr = OrderManager(conn, _FakeContract(), state_mgr, logger, config)

        state = _make_state()
        order = _make_order(entry_price=24000, qty=2)
        state.pending_orders.append(order)

        asyncio.new_event_loop().run_until_complete(
            mgr.suspend_order(order, state, "test_reason")
        )

        assert len(state.pending_orders) == 0
        assert len(state.suspended_orders) == 1
        assert state.suspended_orders[0].state == "SUSPENDED"

        logs = logger.events("order_suspended")
        assert len(logs) == 1
        assert logs[0]["reason"] == "test_reason"

    def test_suspend_refuses_partial(self):
        """Cannot suspend an order with fills."""
        from bot.execution.order_manager import OrderManager

        config = _make_config(dry_run=True)
        logger = _CaptureLogger()
        mgr = OrderManager(_FakeIBConn(), _FakeContract(), _NoOpStateMgr(), logger, config)

        state = _make_state()
        order = _make_order(state="PARTIAL")
        order.filled_qty = 1
        state.pending_orders.append(order)

        asyncio.new_event_loop().run_until_complete(
            mgr.suspend_order(order, state, "should_fail")
        )

        assert len(state.pending_orders) == 1  # Not moved
        assert len(state.suspended_orders) == 0
        assert len(logger.events("suspend_refused")) == 1

    def test_place_bracket_increments_trade_count(self):
        """place_bracket increments trade_count, _place_bracket_internal does not."""
        from bot.execution.order_manager import OrderManager

        config = _make_config(dry_run=True)
        mgr = OrderManager(_FakeIBConn(), _FakeContract(), _NoOpStateMgr(),
                           _CaptureLogger(), config)

        state = _make_state()
        order = _make_order(state="PENDING")

        asyncio.new_event_loop().run_until_complete(
            mgr.place_bracket(order, state)
        )
        assert state.trade_count == 1

        # Internal placement does NOT increment
        state2 = _make_state()
        order2 = _make_order(state="PENDING")
        asyncio.new_event_loop().run_until_complete(
            mgr._place_bracket_internal(order2, state2)
        )
        assert state2.trade_count == 0

    def test_reactivate_order(self):
        """Reactivating a suspended order re-places it without incrementing trade_count."""
        from bot.execution.order_manager import OrderManager

        config = _make_config(dry_run=True)
        logger = _CaptureLogger()
        mgr = OrderManager(_FakeIBConn(), _FakeContract(), _NoOpStateMgr(), logger, config)

        state = _make_state()
        state.trade_count = 5  # Already counted
        order = _make_order(state="SUSPENDED")
        order.suspended_at = "2026-03-26T10:00:00"
        state.suspended_orders.append(order)

        result = asyncio.new_event_loop().run_until_complete(
            mgr.reactivate_order(order, state)
        )

        assert result is not None
        assert result.state == "SUBMITTED"
        assert len(state.suspended_orders) == 0
        assert len(state.pending_orders) == 1
        assert state.trade_count == 5  # NOT incremented

        logs = logger.events("order_reactivated")
        assert len(logs) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 6. E2E: Full Detection → Margin Evaluation → Place/Suspend
# ══════════════════════════════════════════════════════════════════════════════


class TestE2EMarginFlow:
    def _make_engine_shell(self, config, strategy, logger=None, clock=None,
                            margin_priority=None, order_mgr=None):
        """Wire up a minimal engine shell for _process_detection with margin support."""
        import types
        from bot.core.engine import BotEngine
        from bot.strategy.fvg_detector import ActiveFVGManager

        logger = logger or _CaptureLogger()
        clock = clock or _FakeClock()

        @dataclass
        class _NoOpDB:
            def insert_trade(self, **kw): pass
            def insert_fvg(self, **kw): pass
            def update_fvg(self, fvg_id, **kw): pass

        @dataclass
        class _Shell:
            config: object
            strategy: object
            risk_gates: object
            time_gates: object
            daily_state: object
            order_mgr: object
            margin_priority: object
            margin_tracker: object
            fvg_mgr: object
            state_mgr: object
            logger: object
            db: object
            clock: object
            ib_conn: object
            telegram: object = field(default_factory=lambda: type("T", (), {"enabled": False})())
            _reconciliation_complete: bool = True
            _detection_lock: object = field(default_factory=asyncio.Lock)
            _bars_5min: object = None

        state = DailyState(date="2026-03-26", start_balance=100000.0)
        shell = _Shell(
            config=config,
            strategy=strategy,
            risk_gates=RiskGates(config),
            time_gates=TimeGates(config, clock=clock),
            daily_state=state,
            order_mgr=order_mgr,
            margin_priority=margin_priority,
            margin_tracker=getattr(margin_priority, '_margin', None) if margin_priority else None,
            fvg_mgr=ActiveFVGManager(strategy, getattr(config, 'min_fvg_size', 0.25), logger),
            state_mgr=_NoOpStateMgr(),
            logger=logger,
            db=_NoOpDB(),
            clock=clock,
            ib_conn=_FakeIBConn(),
        )

        shell._process_detection = types.MethodType(BotEngine._process_detection, shell)
        shell._get_current_price = types.MethodType(BotEngine._get_current_price, shell)

        return shell

    def _make_strategy_with_cell(self, time_period="10:30-11:00", risk_range="10-15",
                                  setup="mit_extreme", rr_target=2.0):
        from bot.strategy.strategy_loader import StrategyLoader
        strategy = {
            "schema_version": "1.0",
            "meta": {
                "id": "margin-test", "name": "Margin Test", "description": "",
                "created_at": "2026-03-26", "updated_at": "2026-03-26",
                "source_dataset": "test", "ticker": "NQ", "timeframe": "5min",
            },
            "filters": {"min_samples": 10, "require_all_evs_positive": False},
            "cells": [{
                "time_period": time_period, "risk_range": risk_range,
                "setup": setup, "rr_target": rr_target,
                "ev": 0.20, "win_rate": 35.0, "samples": 300,
                "trades_per_day": 0.2, "median_risk": 12.0,
                "enabled": True, "notes": "",
            }],
            "stats": {},
        }
        sdir = tempfile.mkdtemp()
        with open(os.path.join(sdir, "margin-test.json"), "w") as f:
            json.dump(strategy, f)
        with open(os.path.join(sdir, "manifest.json"), "w") as f:
            json.dump({
                "strategies": [{"id": "margin-test", "name": "Margin Test"}],
                "active_strategy": "margin-test",
                "last_updated": "2026-03-26",
            }, f)
        loader = StrategyLoader(sdir)
        loader.load()
        return loader

    def test_e2e_dry_run_places_without_margin_check(self):
        """In dry_run mode, _process_detection skips margin evaluation and places directly."""
        strategy = self._make_strategy_with_cell()
        config = _make_config(
            dry_run=True,
            strategy_dir=strategy._strategy_dir,
        )
        logger = _CaptureLogger()
        order_mgr = _FakeOrderManager()

        shell = self._make_engine_shell(
            config, strategy, logger=logger,
            margin_priority=None,  # No margin priority in dry_run
            order_mgr=order_mgr,
        )

        # Bullish FVG: zone_high=24112, middle_low=24100 → risk=12 → bucket "10-15" ✓
        fvg = FVGRecord(
            fvg_id=_new_id(), fvg_type="bullish",
            zone_low=24100.0, zone_high=24112.0,
            time_candle1="2026-03-26T10:30:00-04:00",
            time_candle2="2026-03-26T10:35:00-04:00",
            time_candle3="2026-03-26T10:40:00-04:00",
            middle_open=24105.0, middle_low=24100.0, middle_high=24120.0,
            first_open=24095.0,
            time_period="10:30-11:00",
            formation_date="2026-03-26",
        )
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.new_event_loop().run_until_complete(
            shell._process_detection(fvg)
        )

        accepted = logger.events("setup_accepted")
        assert len(accepted) == 1
        # In dry_run, order placed via order_mgr (no margin evaluation)
        assert len(order_mgr.placed) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 7. CONFIG FIELDS
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigFields:
    def test_margin_config_defaults(self):
        config = BotConfig(
            state_dir=tempfile.mkdtemp(),
            log_dir=tempfile.mkdtemp(),
            strategy_dir=tempfile.mkdtemp(),
        )
        assert config.margin_intraday_maintenance == 22924.0
        assert config.margin_overnight_initial == 46373.0
        assert config.margin_fallback_per_contract == 22924.0
        assert config.margin_intraday_start == "09:30"
        assert config.margin_intraday_end == "16:00"
        assert config.margin_buffer_pct == 0.05
        assert config.margin_refresh_interval == 1800
        assert config.margin_management_enabled is True

    def test_margin_config_overrides(self):
        config = BotConfig(
            margin_fallback_per_contract=22000.0,
            margin_buffer_pct=0.10,
            margin_management_enabled=False,
            state_dir=tempfile.mkdtemp(),
            log_dir=tempfile.mkdtemp(),
            strategy_dir=tempfile.mkdtemp(),
        )
        assert config.margin_fallback_per_contract == 22000.0
        assert config.margin_buffer_pct == 0.10
        assert config.margin_management_enabled is False
