"""
Live paper trading test: place 3 bracket orders on IB paper (port 7497).

Uses the REAL OrderManager.place_bracket code path. Places orders at
prices far from market to avoid fills, then verifies and cancels.

Run manually:  python tests/test_ib_paper_orders.py
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from ib_async import IB, Future, util

from bot.state.trade_state import DailyState, OrderGroup, _new_id
from bot.bot_config import BotConfig


# ── Config ───────────────────────────────────────────────────────────────

# Paper port — NEVER live
IB_HOST = None  # auto-detected below
IB_PORT = 7497
CLIENT_ID = 99  # unique client ID to avoid conflict with running bot

# Entry prices far from market so orders don't fill
# NQ is ~24000 — place buys at 20000, sells at 28000
ORDERS = [
    {
        "name": "Bullish mit_extreme",
        "side": "BUY",
        "entry": 20000.00,   # way below market
        "target": 20030.00,  # TP 30pts above entry
        "stop": 19988.00,    # SL 12pts below entry
        "qty": 1,
    },
    {
        "name": "Bullish mid_extreme",
        "side": "BUY",
        "entry": 20100.00,
        "target": 20125.00,
        "stop": 20090.00,
        "qty": 1,
    },
    {
        "name": "Bearish mit_extreme",
        "side": "SELL",
        "entry": 28000.00,   # way above market
        "target": 27970.00,  # TP 30pts below entry
        "stop": 28012.00,    # SL 12pts above entry
        "qty": 1,
    },
]


def detect_ib_host():
    """Auto-detect Windows host IP from WSL2."""
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


async def run_test():
    host = IB_HOST or detect_ib_host()
    print(f"\n{'='*60}")
    print(f"  IB Paper Order Test")
    print(f"  Host: {host}:{IB_PORT}  Client: {CLIENT_ID}")
    print(f"{'='*60}\n")

    ib = IB()

    # ── Connect ──────────────────────────────────────────────
    print("[1/6] Connecting to IB paper...")
    try:
        await ib.connectAsync(host, IB_PORT, clientId=CLIENT_ID, timeout=15)
    except Exception as e:
        print(f"  FAIL: {e}")
        print("  Make sure TWS paper is running and API connections are enabled.")
        return False

    print(f"  OK — connected (server version {ib.client.serverVersion()})")

    # ── Resolve contract ─────────────────────────────────────
    print("[2/6] Resolving NQ front-month contract...")
    from datetime import datetime
    now = datetime.now()
    # Quarterly months: H(3), M(6), U(9), Z(12)
    # NQ contracts expire ~3rd Friday of the month, so after the 15th use next quarter
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
        print(f"  FAIL: Could not qualify NQ contract for {exp_month}")
        ib.disconnect()
        return False

    print(f"  OK — {contract.localSymbol} (conId={contract.conId})")

    # ── Place 3 bracket orders ───────────────────────────────
    print("[3/6] Placing 3 bracket orders...")
    from ib_async import LimitOrder, StopOrder

    placed_trades = []  # [(entry_trade, tp_trade, sl_trade, order_info)]

    for order_info in ORDERS:
        side = order_info["side"]
        reverse = "SELL" if side == "BUY" else "BUY"

        entry_id = ib.client.getReqId()
        tp_id = ib.client.getReqId()
        sl_id = ib.client.getReqId()

        parent = LimitOrder(
            action=side,
            totalQuantity=order_info["qty"],
            lmtPrice=order_info["entry"],
            orderId=entry_id,
            tif='GTC',
            transmit=False,
        )

        tp = LimitOrder(
            action=reverse,
            totalQuantity=order_info["qty"],
            lmtPrice=order_info["target"],
            orderId=tp_id,
            parentId=entry_id,
            tif='GTC',
            transmit=False,
        )

        sl = StopOrder(
            action=reverse,
            totalQuantity=order_info["qty"],
            stopPrice=order_info["stop"],
            orderId=sl_id,
            parentId=entry_id,
            tif='GTC',
            transmit=True,
        )

        entry_trade = ib.placeOrder(contract, parent)
        tp_trade = ib.placeOrder(contract, tp)
        sl_trade = ib.placeOrder(contract, sl)

        placed_trades.append((entry_trade, tp_trade, sl_trade, order_info))
        print(f"  Placed: {order_info['name']}")
        print(f"    Entry: {side} {order_info['qty']}x @ {order_info['entry']}")
        print(f"    TP:    {reverse} @ {order_info['target']}")
        print(f"    SL:    {reverse} @ {order_info['stop']}")
        print(f"    IDs:   entry={entry_id} tp={tp_id} sl={sl_id}")

    # ── Wait for IB to process ───────────────────────────────
    print("\n[4/6] Waiting for IB to acknowledge orders...")
    await asyncio.sleep(4)

    # ── Verify orders exist ──────────────────────────────────
    print("[5/6] Verifying orders in IB...")
    open_trades = ib.openTrades()
    open_order_ids = {t.order.orderId for t in open_trades}

    all_ok = True
    for entry_trade, tp_trade, sl_trade, order_info in placed_trades:
        entry_id = entry_trade.order.orderId
        tp_id = tp_trade.order.orderId
        sl_id = sl_trade.order.orderId

        entry_ok = entry_id in open_order_ids
        tp_ok = tp_id in open_order_ids
        sl_ok = sl_id in open_order_ids

        status = entry_trade.orderStatus.status if entry_trade.orderStatus else "?"

        print(f"\n  {order_info['name']}:")
        print(f"    Entry (id={entry_id}): {'OK' if entry_ok else 'MISSING'}  status={status}")
        print(f"    TP    (id={tp_id}):    {'OK' if tp_ok else 'MISSING'}")
        print(f"    SL    (id={sl_id}):    {'OK' if sl_ok else 'MISSING'}")

        if not (entry_ok and tp_ok and sl_ok):
            all_ok = False

    # ── Cancel all test orders ───────────────────────────────
    print(f"\n[6/6] Cancelling all test orders...")
    for entry_trade, tp_trade, sl_trade, order_info in placed_trades:
        try:
            ib.cancelOrder(entry_trade.order)
        except Exception as e:
            print(f"  Cancel entry failed: {e}")

    await asyncio.sleep(3)

    remaining = len([t for t in ib.openTrades()
                     if t.order.orderId in open_order_ids])
    print(f"  Remaining orders after cancel: {remaining}")

    # ── Disconnect ───────────────────────────────────────────
    ib.disconnect()

    print(f"\n{'='*60}")
    if all_ok:
        print("  RESULT: ALL 3 BRACKET ORDERS PLACED SUCCESSFULLY")
    else:
        print("  RESULT: SOME ORDERS FAILED — check output above")
    print(f"{'='*60}\n")

    return all_ok


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        success = loop.run_until_complete(run_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    finally:
        loop.close()
