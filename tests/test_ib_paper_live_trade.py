"""
Live paper e2e test: market entry → monitor TP/SL → report lifecycle.

Places a REAL bracket order on IB paper using the bot's OrderManager.
Entry fills at market, then monitors the trade exactly as the bot would:
  - Entry fill → move_to_open → log slippage
  - TP fill → move_to_closed → log P&L
  - SL fill → move_to_closed → log P&L

Uses a tight 1:1 R:R (3 pts = $60/contract) so the trade resolves in seconds.
Safety: 60-second timeout flattens if neither TP nor SL hits.

Run manually:  python tests/test_ib_paper_live_trade.py [--side BUY|SELL] [--risk 3]
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

import pytz
from ib_async import IB, Future, MarketOrder, util

from bot.bot_config import BotConfig
from bot.execution.order_manager import OrderManager
from bot.state.trade_state import DailyState, OrderGroup, _new_id

NY = pytz.timezone("America/New_York")

# ── Config ───────────────────────────────────────────────────────────────

IB_PORT = 7497  # Paper ONLY
CLIENT_ID = 98
MAX_WAIT_SECONDS = 90  # safety timeout


def detect_ib_host():
    try:
        import subprocess
        result = subprocess.run(['ip', 'route', 'show', 'default'],
                                capture_output=True, text=True, timeout=3)
        for part in result.stdout.split():
            if '.' in part and part[0].isdigit():
                return part
    except Exception:
        pass
    return "127.0.0.1"


class LiveLogger:
    """Logger that prints and captures events."""
    def __init__(self):
        self.records = []

    def log(self, event, **kwargs):
        self.records.append({"event": event, **kwargs})
        ts = datetime.now(NY).strftime("%H:%M:%S.%f")[:-3]
        details = " ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"  [{ts}] {event.upper()} {details}")

    def events(self, name):
        return [r for r in self.records if r["event"] == name]


class NoOpStateMgr:
    def save(self, state, force=False):
        pass


class LiveIBConn:
    """Wraps ib_async.IB to match IBConnection interface."""
    def __init__(self, ib):
        self._ib = ib

    @property
    def ib(self):
        return self._ib

    @property
    def is_connected(self):
        return self._ib.isConnected()


async def get_market_price(ib, contract):
    """Get current market price from a snapshot."""
    ticker = ib.reqMktData(contract, '', True, False)
    for _ in range(50):  # wait up to 5 seconds
        await asyncio.sleep(0.1)
        if ticker.last and ticker.last > 0:
            price = ticker.last
            ib.cancelMktData(contract)
            return price
        if ticker.close and ticker.close > 0:
            price = ticker.close
            ib.cancelMktData(contract)
            return price
    ib.cancelMktData(contract)
    return None


def round_tick(p):
    return round(p * 4) / 4


async def run_trade(side="BUY", risk_pts=3.0):
    host = detect_ib_host()

    print(f"\n{'='*60}")
    print(f"  Live Paper Trade Test — 1:1 R:R")
    print(f"  Side: {side}  Risk: {risk_pts} pts (${ risk_pts * 20:.0f})")
    print(f"  Host: {host}:{IB_PORT}  Client: {CLIENT_ID}")
    print(f"  Safety timeout: {MAX_WAIT_SECONDS}s")
    print(f"{'='*60}\n")

    ib = IB()

    # ── Connect ──────────────────────────────────────────────
    print("[1/7] Connecting to IB paper...")
    try:
        await ib.connectAsync(host, IB_PORT, clientId=CLIENT_ID, timeout=15)
    except Exception as e:
        print(f"  FAIL: {e}")
        return False
    print(f"  OK — connected")

    # ── Resolve contract ─────────────────────────────────────
    print("[2/7] Resolving NQ front-month...")
    now = datetime.now()
    quarter_months = [3, 6, 9, 12]
    for qm in quarter_months:
        if now.month < qm or (now.month == qm and now.day <= 15):
            exp_month = f"{now.year}{qm:02d}"
            break
    else:
        exp_month = f"{now.year + 1}03"

    contract = Future(symbol='NQ', lastTradeDateOrContractMonth=exp_month,
                      exchange='CME', currency='USD')
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        print(f"  FAIL: Could not qualify NQ {exp_month}")
        ib.disconnect()
        return False
    print(f"  OK — {contract.localSymbol}")

    # ── Get market price ─────────────────────────────────────
    print("[3/7] Getting market price...")
    market_price = await get_market_price(ib, contract)
    if market_price is None:
        print("  FAIL: Could not get market price")
        ib.disconnect()
        return False
    print(f"  OK — last: {market_price}")

    # ── Build order group ────────────────────────────────────
    print("[4/7] Building bracket order...")

    if side == "BUY":
        # Limit at bid - 1 tick: sits just below bid for fast fill
        entry_price = round_tick(market_price - 0.25)
        target_price = round_tick(entry_price + risk_pts)
        stop_price = round_tick(entry_price - risk_pts)
    else:
        # Limit at ask + 1 tick: sits just above ask for fast fill
        entry_price = round_tick(market_price + 0.25)
        target_price = round_tick(entry_price - risk_pts)
        stop_price = round_tick(entry_price + risk_pts)

    og = OrderGroup(
        group_id=_new_id(),
        fvg_id=f"test-live-{_new_id()[:6]}",
        setup="mit_extreme",
        side=side,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_pts=risk_pts,
        n_value=1.0,  # 1:1 R:R
        target_qty=1,
        risk_pct=0.001,
    )

    print(f"  Side:   {side}")
    print(f"  Entry:  {entry_price} (limit, should fill immediately)")
    print(f"  Target: {target_price} (+{risk_pts} pts)")
    print(f"  Stop:   {stop_price} (-{risk_pts} pts)")
    print(f"  Risk:   ${risk_pts * 20:.0f} (1 contract)")

    # ── Set up bot infrastructure ────────────────────────────
    logger = LiveLogger()
    config = BotConfig(
        dry_run=False,  # LIVE placement on paper
        paper_mode=True,
        ib_port=IB_PORT,
        risk_per_trade=0.001,
        use_risk_tiers=False,
        point_value=20.0,
        tick_size=0.25,
        partial_fill_timeout=300,
    )

    state = DailyState(date=datetime.now().strftime("%Y-%m-%d"), start_balance=76000.0)
    conn = LiveIBConn(ib)
    state_mgr = NoOpStateMgr()

    order_mgr = OrderManager(conn, contract, state_mgr, logger, config)

    # ── Place bracket via real OrderManager ──────────────────
    print("\n[5/7] Placing bracket order via OrderManager.place_bracket()...")

    trade_resolved = asyncio.Event()
    trade_result = {"outcome": None, "pnl": None}

    # Monkey-patch the TP/SL handlers to signal completion
    _orig_tp = order_mgr._on_tp_fill
    _orig_sl = order_mgr._on_sl_fill

    def _on_tp(trade, og=og, ds=state):
        _orig_tp(trade, og, ds)
        trade_result["outcome"] = "TP"
        trade_result["pnl"] = og.realized_pnl
        trade_resolved.set()

    def _on_sl(trade, og=og, ds=state):
        _orig_sl(trade, og, ds)
        trade_result["outcome"] = "SL"
        trade_result["pnl"] = og.realized_pnl
        trade_resolved.set()

    await order_mgr.place_bracket(og, state)

    # Replace callbacks with our wrapped versions
    for trade in ib.openTrades():
        if og.broker_tp_order_id and trade.order.orderId == int(og.broker_tp_order_id):
            trade.filledEvent.clear()
            trade.filledEvent += _on_tp
        elif og.broker_sl_order_id and trade.order.orderId == int(og.broker_sl_order_id):
            trade.filledEvent.clear()
            trade.filledEvent += _on_sl

    assert og.state == "SUBMITTED", f"Expected SUBMITTED, got {og.state}"
    assert len(state.pending_orders) == 1

    # ── Wait for entry fill ──────────────────────────────────
    print("\n[6/7] Waiting for entry fill...")
    entry_filled = False
    for _ in range(100):  # 10 seconds max
        await asyncio.sleep(0.1)
        if og.state == "FILLED" or len(state.open_positions) > 0:
            entry_filled = True
            break
        # Check if entry fill happened via callback
        if og.filled_qty > 0 and og.state != "SUBMITTED":
            entry_filled = True
            break

    if not entry_filled:
        # Check open positions — callback may have moved it
        if len(state.open_positions) > 0:
            entry_filled = True

    if entry_filled:
        fills = logger.events("order_filled")
        if fills:
            f = fills[0]
            print(f"  FILLED: {f.get('qty', '?')}x @ {f.get('avg_price', '?')}")
            print(f"  Slippage: {f.get('slippage_pts', 0)} pts")
        else:
            print(f"  FILLED (state: {og.state}, positions: {len(state.open_positions)})")
    else:
        print("  WARN: Entry not filled after 10s — cancelling...")
        for trade in ib.openTrades():
            if og.broker_entry_order_id and trade.order.orderId == int(og.broker_entry_order_id):
                ib.cancelOrder(trade.order)
        await asyncio.sleep(2)
        ib.disconnect()
        return False

    # ── Monitor trade ────────────────────────────────────────
    print(f"\n[7/7] Monitoring trade (timeout {MAX_WAIT_SECONDS}s)...")
    print(f"  Waiting for TP ({target_price}) or SL ({stop_price})...")

    try:
        await asyncio.wait_for(trade_resolved.wait(), timeout=MAX_WAIT_SECONDS)
    except asyncio.TimeoutError:
        print(f"\n  TIMEOUT after {MAX_WAIT_SECONDS}s — flattening...")
        # Market order to close
        reverse = "SELL" if side == "BUY" else "BUY"
        flatten = MarketOrder(action=reverse, totalQuantity=1)
        flatten_trade = ib.placeOrder(contract, flatten)
        await asyncio.sleep(3)
        trade_result["outcome"] = "TIMEOUT_FLATTEN"

        # Cancel remaining bracket legs
        for trade in ib.openTrades():
            if trade.order.orderId in (int(og.broker_tp_order_id or 0), int(og.broker_sl_order_id or 0)):
                try:
                    ib.cancelOrder(trade.order)
                except Exception:
                    pass
        await asyncio.sleep(2)

    # ── Report ───────────────────────────────────────────────
    outcome = trade_result["outcome"]
    print(f"\n{'='*60}")
    print(f"  TRADE RESULT: {outcome}")

    if outcome == "TP":
        tp_events = logger.events("tp_filled")
        if tp_events:
            e = tp_events[0]
            print(f"  Fill price:   {e.get('fill_price', '?')}")
            print(f"  P&L (gross):  ${e.get('gross_pnl', '?')}")
            print(f"  P&L (net):    ${e.get('net_pnl', '?')}")
            print(f"  Commission:   ${e.get('commission', '?')}")
            print(f"  Duration:     {e.get('duration', '?')}")
    elif outcome == "SL":
        sl_events = logger.events("sl_filled")
        if sl_events:
            e = sl_events[0]
            print(f"  Fill price:   {e.get('fill_price', '?')}")
            print(f"  Stop slip:    {e.get('stop_slippage', 0)} pts")
            print(f"  P&L (gross):  ${e.get('gross_pnl', '?')}")
            print(f"  P&L (net):    ${e.get('net_pnl', '?')}")
            print(f"  Commission:   ${e.get('commission', '?')}")
            print(f"  Duration:     {e.get('duration', '?')}")
    else:
        print(f"  Flattened due to timeout")

    # State summary
    print(f"\n  State summary:")
    print(f"    Pending:  {len(state.pending_orders)}")
    print(f"    Open:     {len(state.open_positions)}")
    print(f"    Closed:   {len(state.closed_trades)}")
    print(f"    P&L:      ${state.realized_pnl:.2f}")
    print(f"    Trades:   {state.trade_count}")

    if state.closed_trades:
        ct = state.closed_trades[0]
        print(f"\n  Closed trade detail:")
        print(f"    group_id:     {ct.group_id}")
        print(f"    close_reason: {ct.close_reason}")
        print(f"    realized_pnl: ${ct.realized_pnl:.2f}")
        print(f"    state:        {ct.state}")

    print(f"{'='*60}\n")

    # ── Cleanup ──────────────────────────────────────────────
    # Cancel any remaining orders just in case
    for trade in ib.openTrades():
        if trade.order.orderId in (int(og.broker_entry_order_id or 0), int(og.broker_tp_order_id or 0), int(og.broker_sl_order_id or 0)):
            try:
                ib.cancelOrder(trade.order)
            except Exception:
                pass
    await asyncio.sleep(1)
    ib.disconnect()

    return outcome in ("TP", "SL")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live paper trade test")
    parser.add_argument("--side", default="BUY", choices=["BUY", "SELL"])
    parser.add_argument("--risk", type=float, default=3.0, help="Risk in points (default 3)")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        success = loop.run_until_complete(run_trade(args.side, args.risk))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    finally:
        loop.close()
