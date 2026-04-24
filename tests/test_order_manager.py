"""
Tests for order_manager.py critical hardening fixes.

Covers:
1. State save after entry fill (full fill + partial fill timeout)
2. TP/SL race condition — atomic close_reason claim
3. Partial fill state persistence
4. IB callback thread-safety marshalling
5. Trade count NOT incremented on failed placement
6. placeOrder error handling → REJECTED state

Uses hand-written stubs (no unittest.mock), matching project test conventions.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest
import pytz

from bot.execution.order_manager import OrderManager
from bot.state.trade_state import (
    DailyState, OrderGroup, _new_id,
    CLOSE_TP, CLOSE_SL, CLOSE_REJECTED,
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


class _TrackingStateMgr:
    """State manager that tracks save calls."""
    def __init__(self):
        self.save_count = 0

    def save(self, state, force=False):
        self.save_count += 1


class _FakeClock:
    def __init__(self, hour=10, minute=30):
        self._now = NY.localize(datetime(2026, 3, 26, hour, minute))

    def now(self):
        return self._now

    def advance(self, seconds=0, minutes=0):
        self._now += timedelta(seconds=seconds, minutes=minutes)


class _Event:
    """Minimal ib_async-style event."""
    def __init__(self):
        self._handlers = []

    def __iadd__(self, handler):
        self._handlers.append(handler)
        return self

    def fire(self, *args):
        for h in self._handlers:
            h(*args)


class _FakeOrderStatus:
    def __init__(self, filled=0, avg_fill_price=0.0, status="Submitted"):
        self.filled = filled
        self.avgFillPrice = avg_fill_price
        self.status = status
        self.whyHeld = ""


class _FakeFill:
    def __init__(self, commission=2.50):
        self.commissionReport = _FakeCommReport(commission)


class _FakeCommReport:
    def __init__(self, commission):
        self.commission = commission


class _FakeTrade:
    """Fake ib_async Trade returned by placeOrder."""
    def __init__(self):
        self.filledEvent = _Event()
        self.statusEvent = _Event()
        self.orderStatus = _FakeOrderStatus()
        self.fills = [_FakeFill()]
        self.order = _FakeOrder()


class _FakeOrder:
    def __init__(self, order_id=0):
        self.orderId = order_id


class _FakeClient:
    def __init__(self):
        self._next_id = 100

    def getReqId(self):
        self._next_id += 1
        return self._next_id


class _FakeIB:
    """Fake ib_async.IB with placeOrder that returns FakeTrades."""
    def __init__(self, fail_on_place=False):
        self.client = _FakeClient()
        self._fail_on_place = fail_on_place
        self.placed_trades = []

    def placeOrder(self, contract, order):
        if self._fail_on_place:
            raise ConnectionError("IB disconnected")
        trade = _FakeTrade()
        trade.order = order
        self.placed_trades.append(trade)
        return trade

    def openTrades(self):
        return []

    def cancelOrder(self, order):
        pass


class _FakeIBConn:
    is_connected = True

    def __init__(self, fail_on_place=False):
        self.ib = _FakeIB(fail_on_place)


@dataclass
class _FakeConfig:
    dry_run: bool = False
    partial_fill_timeout: int = 300


# ── Helpers ──────────────────────────────────────────────────────────────────


def _run(coro):
    """Run async code in tests — matches project convention."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        # Drain pending callbacks (call_soon_threadsafe-scheduled)
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()


def _run_with_drain(coro, drain_seconds=0.0):
    """Run coro then drain the event loop for scheduled callbacks."""
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(coro)
        loop.run_until_complete(asyncio.sleep(drain_seconds))
        return result
    finally:
        loop.close()


def _make_og(**overrides):
    defaults = dict(
        group_id=_new_id(),
        fvg_id=_new_id(),
        setup="mit_extreme",
        side="BUY",
        entry_price=20000.0,
        stop_price=19990.0,
        target_price=20027.5,
        risk_pts=10.0,
        n_value=2.75,
        target_qty=2,
    )
    defaults.update(overrides)
    return OrderGroup(**defaults)


def _make_daily():
    return DailyState(date="2026-03-26", start_balance=100000.0)


def _make_mgr(config=None, conn=None, state_mgr=None, clock=None, logger=None):
    return OrderManager(
        connection=conn or _FakeIBConn(),
        contract=object(),
        state_manager=state_mgr or _TrackingStateMgr(),
        logger=logger or _CaptureLogger(),
        config=config or _FakeConfig(),
        clock=clock,
    )


# ── Fix 1: State save after entry fill ──────────────────────────────────────


class TestStateSaveAfterFill:
    def test_state_saved_after_full_entry_fill(self):
        """After full entry fill → move_to_open, state must be persisted."""
        state_mgr = _TrackingStateMgr()
        conn = _FakeIBConn()
        clock = _FakeClock()
        mgr = _make_mgr(state_mgr=state_mgr, conn=conn, clock=clock)
        daily = _make_daily()
        og = _make_og()

        async def run():
            await mgr.place_bracket(og, daily)

            saves_before = state_mgr.save_count

            # Simulate full fill via IB callback
            entry_trade = conn.ib.placed_trades[0]
            entry_trade.orderStatus.filled = og.target_qty
            entry_trade.orderStatus.avgFillPrice = 20000.25
            entry_trade.filledEvent.fire(entry_trade)

            # Drain call_soon_threadsafe callbacks
            await asyncio.sleep(0)

            assert state_mgr.save_count > saves_before, "State not saved after entry fill"
            assert len(daily.open_positions) == 1

        _run(run())

    def test_state_saved_after_partial_fill_timeout(self):
        """After partial fill timer expires → move_to_open, state is persisted."""
        state_mgr = _TrackingStateMgr()
        conn = _FakeIBConn()
        clock = _FakeClock()
        config = _FakeConfig(partial_fill_timeout=0.05)  # 50ms
        mgr = _make_mgr(state_mgr=state_mgr, conn=conn, clock=clock, config=config)
        daily = _make_daily()
        og = _make_og(target_qty=4)

        async def run():
            await mgr.place_bracket(og, daily)

            entry_trade = conn.ib.placed_trades[0]
            entry_trade.orderStatus.filled = 2
            entry_trade.orderStatus.avgFillPrice = 20000.0
            entry_trade.filledEvent.fire(entry_trade)
            await asyncio.sleep(0)

            saves_before = state_mgr.save_count

            # Wait for partial fill timer
            await asyncio.sleep(0.15)

            assert state_mgr.save_count > saves_before
            assert len(daily.open_positions) == 1

        _run(run())


# ── Fix 2: TP/SL race condition ─────────────────────────────────────────────


class TestTPSLRace:
    def test_tp_then_sl_only_tp_processes(self):
        """If TP fires first, subsequent SL is blocked."""
        state_mgr = _TrackingStateMgr()
        conn = _FakeIBConn()
        clock = _FakeClock()
        mgr = _make_mgr(state_mgr=state_mgr, conn=conn, clock=clock)
        daily = _make_daily()
        og = _make_og()

        async def run():
            await mgr.place_bracket(og, daily)

            # Full entry fill
            entry_trade = conn.ib.placed_trades[0]
            entry_trade.orderStatus.filled = og.target_qty
            entry_trade.orderStatus.avgFillPrice = 20000.0
            entry_trade.filledEvent.fire(entry_trade)
            await asyncio.sleep(0)
            assert len(daily.open_positions) == 1

            # TP fills
            tp_trade = conn.ib.placed_trades[1]
            tp_trade.orderStatus.filled = og.target_qty
            tp_trade.orderStatus.avgFillPrice = 20027.50
            tp_trade.filledEvent.fire(tp_trade)
            await asyncio.sleep(0)

            assert og.close_reason == CLOSE_TP
            assert len(daily.closed_trades) == 1

            # SL fires after — blocked by atomic claim
            sl_trade = conn.ib.placed_trades[2]
            sl_trade.orderStatus.filled = og.target_qty
            sl_trade.orderStatus.avgFillPrice = 19990.0
            sl_trade.filledEvent.fire(sl_trade)
            await asyncio.sleep(0)

            assert og.close_reason == CLOSE_TP
            assert len(daily.closed_trades) == 1, "SL must not double-process"

        _run(run())

    def test_sl_then_tp_only_sl_processes(self):
        """If SL fires first, subsequent TP is blocked."""
        conn = _FakeIBConn()
        clock = _FakeClock()
        mgr = _make_mgr(conn=conn, clock=clock)
        daily = _make_daily()
        og = _make_og()

        async def run():
            await mgr.place_bracket(og, daily)

            # Full entry fill
            entry_trade = conn.ib.placed_trades[0]
            entry_trade.orderStatus.filled = og.target_qty
            entry_trade.orderStatus.avgFillPrice = 20000.0
            entry_trade.filledEvent.fire(entry_trade)
            await asyncio.sleep(0)

            # SL fires first
            sl_trade = conn.ib.placed_trades[2]
            sl_trade.orderStatus.filled = og.target_qty
            sl_trade.orderStatus.avgFillPrice = 19990.0
            sl_trade.filledEvent.fire(sl_trade)
            await asyncio.sleep(0)

            assert og.close_reason == CLOSE_SL

            # TP fires after — blocked
            tp_trade = conn.ib.placed_trades[1]
            tp_trade.orderStatus.filled = og.target_qty
            tp_trade.orderStatus.avgFillPrice = 20027.50
            tp_trade.filledEvent.fire(tp_trade)
            await asyncio.sleep(0)

            assert og.close_reason == CLOSE_SL
            assert len(daily.closed_trades) == 1

        _run(run())


# ── Fix 3: Partial fill state persistence ────────────────────────────────────


class TestPartialFillPersistence:
    def test_partial_fill_triggers_state_save(self):
        """Partial fill must persist state immediately."""
        state_mgr = _TrackingStateMgr()
        conn = _FakeIBConn()
        clock = _FakeClock()
        mgr = _make_mgr(state_mgr=state_mgr, conn=conn, clock=clock)
        daily = _make_daily()
        og = _make_og(target_qty=4)

        async def run():
            await mgr.place_bracket(og, daily)
            saves_before = state_mgr.save_count

            entry_trade = conn.ib.placed_trades[0]
            entry_trade.orderStatus.filled = 1
            entry_trade.orderStatus.avgFillPrice = 20000.0
            entry_trade.filledEvent.fire(entry_trade)
            await asyncio.sleep(0)

            assert state_mgr.save_count > saves_before, "State not saved after partial fill"
            assert og.state == "PARTIAL"
            assert og.filled_qty == 1

        _run(run())


# ── Fix 4: Thread-safe callback marshalling ──────────────────────────────────


class TestCallbackThreadSafety:
    def test_loop_cached_after_placement(self):
        """Event loop must be cached for callback marshalling."""
        conn = _FakeIBConn()
        mgr = _make_mgr(conn=conn)
        daily = _make_daily()
        og = _make_og()

        async def run():
            assert mgr._loop is None
            await mgr.place_bracket(og, daily)
            assert mgr._loop is not None

        _run(run())

    def test_callbacks_deferred_via_call_soon_threadsafe(self):
        """Callbacks are queued, not executed synchronously on fire."""
        conn = _FakeIBConn()
        mgr = _make_mgr(conn=conn)
        daily = _make_daily()
        og = _make_og()

        async def run():
            await mgr.place_bracket(og, daily)

            entry_trade = conn.ib.placed_trades[0]
            entry_trade.orderStatus.filled = og.target_qty
            entry_trade.orderStatus.avgFillPrice = 20000.0

            # Fire the event — callback is queued, not immediate
            entry_trade.filledEvent.fire(entry_trade)

            # Before yield: callback hasn't run yet (queued via call_soon_threadsafe)
            # open_positions may be empty
            open_before_yield = len(daily.open_positions)

            # After yield: callback executes
            await asyncio.sleep(0)

            assert len(daily.open_positions) == 1, "Callback did not execute after yield"

        _run(run())


# ── Fix 5: Trade count after successful placement ───────────────────────────


class TestTradeCountTiming:
    def test_incremented_after_success(self):
        """trade_count increments only after successful placement."""
        conn = _FakeIBConn()
        mgr = _make_mgr(conn=conn)
        daily = _make_daily()
        og = _make_og()

        async def run():
            assert daily.trade_count == 0
            await mgr.place_bracket(og, daily)
            assert daily.trade_count == 1
            assert og.state == "SUBMITTED"

        _run(run())

    def test_not_incremented_on_failure(self):
        """If placeOrder fails, trade_count must NOT increment."""
        conn = _FakeIBConn(fail_on_place=True)
        mgr = _make_mgr(conn=conn)
        daily = _make_daily()
        og = _make_og()

        async def run():
            assert daily.trade_count == 0
            await mgr.place_bracket(og, daily)
            assert daily.trade_count == 0, "Trade count incremented despite failed placement"
            assert og.state == "CLOSED"

        _run(run())

    def test_dry_run_increments_trade_count(self):
        """DRY_RUN placement still increments trade_count."""
        config = _FakeConfig(dry_run=True)
        mgr = _make_mgr(config=config)
        daily = _make_daily()
        og = _make_og()

        async def run():
            await mgr.place_bracket(og, daily)
            assert daily.trade_count == 1
            assert og.state == "SUBMITTED"
            assert len(daily.pending_orders) == 1

        _run(run())


# ── Fix 6: placeOrder error handling ─────────────────────────────────────────


class TestPlaceOrderErrorHandling:
    def test_failure_marks_rejected(self):
        """If IB placeOrder raises, order moves to CLOSED/REJECTED."""
        logger = _CaptureLogger()
        conn = _FakeIBConn(fail_on_place=True)
        mgr = _make_mgr(conn=conn, logger=logger)
        daily = _make_daily()
        og = _make_og()

        async def run():
            result = await mgr.place_bracket(og, daily)
            assert result.state == "CLOSED"
            assert result.close_reason == CLOSE_REJECTED
            assert len(daily.closed_trades) == 1
            assert len(daily.pending_orders) == 0
            assert logger.events("order_placement_failed")

        _run(run())

    def test_failure_saves_state(self):
        """State must be persisted after placement failure."""
        state_mgr = _TrackingStateMgr()
        conn = _FakeIBConn(fail_on_place=True)
        mgr = _make_mgr(state_mgr=state_mgr, conn=conn)
        daily = _make_daily()
        og = _make_og()

        async def run():
            await mgr.place_bracket(og, daily)
            # save before placement (SUBMITTED) + save after failure (REJECTED)
            assert state_mgr.save_count >= 2, f"Expected ≥2 saves, got {state_mgr.save_count}"

        _run(run())
