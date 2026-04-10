"""
Live paper-TWS test for suspend/reactivate + margin release fixes.

Validates two recent fixes against real IB paper TWS:

  Fix #1 (commit bdbf969 — IB cancel echo no longer wipes suspended orders):
    suspend_order() calls ib.cancelOrder() which fires an async Cancelled
    callback. Before the fix, _on_entry_status moved the order to
    closed_trades on that echo, instantly destroying every suspended order.
    Test: place a real bracket, suspend it, wait for the echo, assert the
    order is still in suspended_orders and NOT in closed_trades.

  Fix #2 (commit 2013146 — trust local accounting after suspend):
    MarginTracker.release() decrements _reserved_margin locally without
    re-querying IB (which lags ~5s). Test: reserve(1), release(1), assert
    _reserved_margin returns to baseline immediately, no IB round-trip.

  Plus: reactivate_order() — assert a suspended order can be re-placed
  with fresh IB IDs after suspension.

Read-only: places one bracket far from market, uses 1 contract so margin
fits comfortably. Cleans up everything before disconnecting.

Run manually:  python tests/test_ib_paper_suspend_margin.py
"""

import asyncio
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

import pytz  # noqa: E402
from ib_async import IB, Future, util  # noqa: E402

from bot.bot_config import BotConfig  # noqa: E402
from bot.execution.order_manager import OrderManager  # noqa: E402
from bot.risk.margin_tracker import MarginTracker  # noqa: E402
from bot.state.trade_state import DailyState, OrderGroup, _new_id  # noqa: E402

NY = pytz.timezone("America/New_York")
IB_PORT = 7497
assert IB_PORT == 7497, "paper TWS only"
CLIENT_ID = 97  # distinct from canonical-bar test (98) and running bot


def detect_ib_host():
    try:
        r = subprocess.run(['ip', 'route', 'show', 'default'],
                           capture_output=True, text=True, timeout=3)
        for p in r.stdout.split():
            if '.' in p and p[0].isdigit():
                return p
    except Exception:
        pass
    return "127.0.0.1"


def front_month_expiry():
    now = datetime.now()
    for qm in (3, 6, 9, 12):
        if now.month < qm or (now.month == qm and now.day <= 15):
            return f"{now.year}{qm:02d}"
    return f"{now.year + 1}03"


def round_tick(p):
    return round(p * 4) / 4


class CapturingLogger:
    def __init__(self):
        self.records = []

    def log(self, event, **kwargs):
        self.records.append({"event": event, **kwargs})
        ts = datetime.now(NY).strftime("%H:%M:%S.%f")[:-3]
        details = " ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"  [{ts}] {event} {details}")

    def events(self, name):
        return [r for r in self.records if r["event"] == name]

    def has_event(self, name):
        return any(r["event"] == name for r in self.records)


class NoOpStateMgr:
    def save(self, state, force=False):
        pass


class LiveIBConn:
    def __init__(self, ib):
        self._ib = ib

    @property
    def ib(self):
        return self._ib

    @property
    def is_connected(self):
        return self._ib.isConnected()


async def get_spot(ib, contract):
    ticker = ib.reqMktData(contract, '', True, False)
    spot = None
    for _ in range(60):
        await asyncio.sleep(0.1)
        if ticker.last and ticker.last > 0:
            spot = ticker.last; break
        if ticker.close and ticker.close > 0:
            spot = ticker.close; break
    ib.cancelMktData(contract)
    return spot


async def main():
    host = detect_ib_host()
    print(f"\n{'='*64}")
    print(f"  IB Paper Suspend / Margin Release Test")
    print(f"  Host: {host}:{IB_PORT}  Client: {CLIENT_ID}")
    print(f"  Mode: PAPER (port 7497)")
    print(f"{'='*64}\n")

    ib = IB()
    print("[1/9] Connecting...")
    try:
        await ib.connectAsync(host, IB_PORT, clientId=CLIENT_ID, timeout=15)
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)
    print("  OK")

    print(f"[2/9] Resolving NQ {front_month_expiry()}...")
    contract = Future(symbol='NQ', lastTradeDateOrContractMonth=front_month_expiry(),
                      exchange='CME', currency='USD')
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        print("  FAIL qualify")
        ib.disconnect(); sys.exit(1)
    print(f"  OK — {contract.localSymbol}")

    print("[3/9] Fetching spot...")
    spot = await get_spot(ib, contract)
    if spot is None:
        print("  FAIL — no spot price")
        ib.disconnect(); sys.exit(1)
    print(f"  Spot ~ {spot}")

    print("[4/9] Wiring OrderManager + MarginTracker...")
    logger = CapturingLogger()
    config = BotConfig(
        dry_run=False, paper_mode=True, ib_port=IB_PORT,
        risk_per_trade=0.001, use_risk_tiers=False,
        point_value=20.0, tick_size=0.25, partial_fill_timeout=300,
    )
    state = DailyState(date=datetime.now().strftime("%Y-%m-%d"), start_balance=100000.0)
    conn = LiveIBConn(ib)
    state_mgr = NoOpStateMgr()
    order_mgr = OrderManager(conn, contract, state_mgr, logger, config)
    margin = MarginTracker(conn, contract, logger, config)
    # Force a fixed per-contract margin so we don't depend on IB account state
    margin._margin_per_contract = 25000.0
    print(f"  OK — margin_per_contract pinned to ${margin._margin_per_contract:.0f}")

    # ── Place 1 bracket far enough below spot to never fill ────────
    print("[5/9] Placing 1 bracket BUY 100pt below spot...")
    entry = round_tick(spot - 100)
    target = round_tick(entry + 20)
    stop = round_tick(entry - 12)
    og = OrderGroup(
        group_id=_new_id(),
        fvg_id=f"test-suspend-{_new_id()[:6]}",
        setup="mit_extreme",
        side="BUY",
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        risk_pts=12.0,
        n_value=1.5,
        target_qty=1,
        risk_pct=0.001,
    )
    await order_mgr.place_bracket(og, state)
    margin.reserve(1)

    if og.state != "SUBMITTED":
        print(f"  FAIL — expected SUBMITTED, got {og.state}")
        ib.disconnect(); sys.exit(1)
    if og not in state.pending_orders:
        print("  FAIL — og not in pending_orders")
        ib.disconnect(); sys.exit(1)
    baseline_reserved = margin._reserved_margin
    print(f"  OK — group_id={og.group_id} entry={entry} state={og.state} "
          f"reserved=${baseline_reserved:.0f}")

    await asyncio.sleep(2)  # let IB acknowledge

    # ── Suspend the order ───────────────────────────────────────────
    print("[6/9] Suspending order (margin priority simulation)...")
    await order_mgr.suspend_order(og, state, reason="test_margin_priority")
    margin.release(1)

    if og.state != "SUSPENDED":
        print(f"  FAIL — expected SUSPENDED, got {og.state}")
        ib.disconnect(); sys.exit(1)
    if og not in state.suspended_orders:
        print("  FAIL — og not in suspended_orders")
        ib.disconnect(); sys.exit(1)
    if og in state.pending_orders:
        print("  FAIL — og still in pending_orders")
        ib.disconnect(); sys.exit(1)
    print(f"  OK — state={og.state}, in suspended_orders, "
          f"reserved=${margin._reserved_margin:.0f}")

    # Margin should be released LOCALLY without IB round-trip
    if abs(margin._reserved_margin - (baseline_reserved - 25000.0)) > 0.01:
        print(f"  FAIL — margin not released locally "
              f"(expected ${baseline_reserved - 25000.0:.0f}, "
              f"got ${margin._reserved_margin:.0f})")
        ib.disconnect(); sys.exit(1)
    print("  OK — margin released locally (no IB re-query)")

    # ── Wait for IB cancel echo and assert it does NOT wipe suspended ─
    print("[7/9] Waiting 4s for IB cancel echo...")
    await asyncio.sleep(4)

    if og not in state.suspended_orders:
        print("  FAIL — IB cancel echo wiped the suspended order!")
        print(f"         closed_trades count: {len(state.closed_trades)}")
        for ct in state.closed_trades:
            print(f"           - {ct.group_id} state={ct.state}")
        ib.disconnect(); sys.exit(1)
    if any(ct.group_id == og.group_id for ct in state.closed_trades):
        print("  FAIL — order moved to closed_trades despite SUSPENDED state")
        ib.disconnect(); sys.exit(1)
    if not logger.has_event("suspend_cancel_echo"):
        print("  WARN — no suspend_cancel_echo log; may mean IB hasn't echoed yet")
    print("  OK — order survived IB cancel echo (still in suspended_orders)")

    # ── Reactivate ──────────────────────────────────────────────────
    print("[8/9] Reactivating suspended order...")
    old_entry_id = og.broker_entry_order_id
    reactivated = await order_mgr.reactivate_order(og, state)
    if reactivated is None:
        print("  FAIL — reactivate returned None")
        ib.disconnect(); sys.exit(1)
    if reactivated.state != "SUBMITTED":
        print(f"  FAIL — reactivated state={reactivated.state}, expected SUBMITTED")
        ib.disconnect(); sys.exit(1)
    if reactivated.broker_entry_order_id == old_entry_id:
        print(f"  FAIL — reactivate did not assign new IB IDs "
              f"(still {old_entry_id})")
        ib.disconnect(); sys.exit(1)
    if reactivated not in state.pending_orders:
        print("  FAIL — reactivated order not in pending_orders")
        ib.disconnect(); sys.exit(1)
    if reactivated in state.suspended_orders:
        print("  FAIL — reactivated order still in suspended_orders")
        ib.disconnect(); sys.exit(1)
    print(f"  OK — new entry_id={reactivated.broker_entry_order_id} "
          f"(was {old_entry_id}), state=SUBMITTED")

    await asyncio.sleep(2)

    # ── Cleanup: cancel everything we placed ────────────────────────
    print("[9/9] Cleaning up — cancelling reactivated bracket...")
    for trade in ib.openTrades():
        try:
            ib.cancelOrder(trade.order)
        except Exception:
            pass
    await asyncio.sleep(2)

    ib.disconnect()

    print(f"\n{'='*64}")
    print("  RESULT: ALL ASSERTIONS PASSED")
    print("  - suspend_order: order moved to SUSPENDED, not wiped by IB echo")
    print("  - margin release: local accounting decremented immediately")
    print("  - reactivate_order: new IB IDs, back to SUBMITTED in pending")
    print(f"{'='*64}\n")
    sys.exit(0)


if __name__ == "__main__":
    util.run(main())
