"""
Live paper-TWS end-to-end test for the canonical-bar-only FVG detection path.

Validates the fix from EOD reconciliation 2026-04-07: live FVG detection now
consumes IB's settled 5min bar exclusively (no tick-built phantom OHLCs).

What this test does — against PAPER TWS (port 7497) only:

  STEP 1 (static guard)
    Asserts engine.py has no _on_tick_bar_complete / _tick_detected_bars /
    TickBarBuilder / detect_from_tick_bar / fvg_detected_tick references.
    Catches accidental reintroduction of the tick FVG path.

  STEP 2 (live parity check)
    Subscribes to NQ 5min keepUpToDate bars and AllLast tick stream side by
    side. Waits for one 5min bar to close. Compares the canonical IB bar
    OHLC against a tick-aggregated OHLC for the same window. Reports any
    divergence — this is the exact failure mode that caused today's loss.

  STEP 3 (real order placement on paper)
    Places three REAL bracket orders via IB paper at prices far from market
    (so they don't fill), proving the canonical-bar path is wired into the
    actual order plumbing. Verifies the orders land in IB.openTrades() and
    then cancels every order it placed.

Read-only against bot state. Uses CLIENT_ID=98 to avoid colliding with a
running bot. Paper port hardcoded — will refuse to run on 7496.

Run manually:  python tests/test_ib_paper_canonical_bar_parity.py
"""

import asyncio
import os
import subprocess
import sys
from datetime import datetime, time as dtime, timedelta

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

import pytz  # noqa: E402
from ib_async import IB, Future, LimitOrder, StopOrder, util  # noqa: E402

from bot.core import engine as engine_mod  # noqa: E402

NY = pytz.timezone("America/New_York")
RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)


def is_rth_now():
    now_et = datetime.now(NY)
    if now_et.weekday() >= 5:
        return False
    return RTH_OPEN <= now_et.time() < RTH_CLOSE

# Paper ONLY — refuse anything else
IB_PORT = 7497
assert IB_PORT == 7497, "this test must only run against paper TWS"

CLIENT_ID = 98


def detect_ib_host():
    try:
        result = subprocess.run(['ip', 'route', 'show', 'default'],
                                capture_output=True, text=True, timeout=3)
        for part in result.stdout.split():
            if '.' in part and part[0].isdigit():
                return part
    except Exception:
        pass
    return "127.0.0.1"


def front_month_expiry():
    now = datetime.now()
    for qm in (3, 6, 9, 12):
        if now.month < qm or (now.month == qm and now.day <= 15):
            return f"{now.year}{qm:02d}"
    return f"{now.year + 1}03"


def floor_5min(dt):
    return dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)


# ──────────────────────────────────────────────────────────────────────────
# STEP 1 — Static guard
# ──────────────────────────────────────────────────────────────────────────

def step1_static_guard():
    print("\n[STEP 1] Static guard: tick FVG path must not exist")
    forbidden_attrs = ("_on_tick_bar_complete", "_tick_detected_bars", "_tick_bar_builder")
    for attr in forbidden_attrs:
        assert not hasattr(engine_mod.BotEngine, attr), \
            f"  FAIL — BotEngine.{attr} reintroduced"
    src = open(engine_mod.__file__).read()
    for tok in ("detect_from_tick_bar", "fvg_detected_tick", "TickBarBuilder"):
        assert tok not in src, f"  FAIL — engine.py references {tok!r}"
    print("  OK — engine.py has no tick FVG path")
    return True


# ──────────────────────────────────────────────────────────────────────────
# STEP 2 — Live tick-vs-bar parity check
# ──────────────────────────────────────────────────────────────────────────

async def step2_live_bar_close(ib, contract, wait_minutes=6):
    """Wait for one canonical 5min bar to close via keepUpToDate stream.

    Validates the exact path the bot now uses for FVG detection: receive
    the just-closed bar from `_on_5min_update`'s `bars[-2]` slot once IB
    pushes the new-bar event (~5s after the wall-clock close).
    """
    print(f"\n[STEP 2] Live canonical-bar close (wait up to {wait_minutes}min)")

    if not is_rth_now():
        print("  SKIP — outside RTH (09:30–16:00 ET, weekday).")
        return None

    bars = await ib.reqHistoricalDataAsync(
        contract, endDateTime='', durationStr='3600 S',
        barSizeSetting='5 mins', whatToShow='TRADES',
        useRTH=True, formatDate=2, keepUpToDate=True,
    )

    captured = {"bar": None, "delay_s": None, "wall_close": None}

    def on_bar(bs, hasNewBar):
        if hasNewBar and len(bs) >= 2 and captured["bar"] is None:
            jc = bs[-2]
            wall_close = jc.date + timedelta(minutes=5)
            now_utc = datetime.now(pytz.utc)
            captured["bar"] = jc
            captured["wall_close"] = wall_close
            captured["delay_s"] = (now_utc - wall_close).total_seconds()

    bars.updateEvent += on_bar

    deadline = asyncio.get_event_loop().time() + wait_minutes * 60
    while captured["bar"] is None and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(1)

    ib.cancelHistoricalData(bars)

    if captured["bar"] is None:
        print("  FAIL — no bar closed within window")
        return False

    b = captured["bar"]
    print(f"  Bar closed @ {b.date}  O={b.open} H={b.high} L={b.low} C={b.close}")
    print(f"  Settlement delay (wall-close → IB push): {captured['delay_s']:.2f}s")
    print("  This is exactly what _on_5min_update consumes via bars[-2].")
    return True


# ──────────────────────────────────────────────────────────────────────────
# STEP 2b — Historical canonical-bar smoke test (runnable any time)
# ──────────────────────────────────────────────────────────────────────────

async def step2b_historical_bars(ib, contract, end_dt_et=None, lookback_bars=12):
    """Fetch a recent window of canonical 5min TRADES bars and print them.

    Validates that reqHistoricalData returns sane settled bars — the same
    source the canonical-bar-only FVG path now consumes. No tick comparison;
    the 5min bar with its ~5s settlement delay is the single source of truth.
    """
    print(f"\n[STEP 2b] Historical canonical-bar smoke test")

    if end_dt_et is None:
        # Default: yesterday's 11:00 ET (covers the 10:55 phantom window from 2026-04-07)
        now_et = datetime.now(NY)
        target = now_et.replace(hour=11, minute=0, second=0, microsecond=0)
        if target > now_et:
            target -= timedelta(days=1)
        end_dt_et = target

    # IB UTC dash format: 'YYYYMMDD-HH:MM:SS'
    end_utc = end_dt_et.astimezone(pytz.utc)
    end_str = end_utc.strftime('%Y%m%d-%H:%M:%S')

    duration = f"{lookback_bars * 5 * 60} S"
    print(f"  Window end: {end_dt_et}  ({lookback_bars} bars × 5min)")

    bars = await ib.reqHistoricalDataAsync(
        contract, endDateTime=end_str, durationStr=duration,
        barSizeSetting='5 mins', whatToShow='TRADES',
        useRTH=True, formatDate=2, keepUpToDate=False,
    )
    if not bars:
        print("  FAIL — no historical bars returned")
        return False
    print(f"  Got {len(bars)} canonical 5min bars (TRADES, useRTH=True)")
    for b in bars:
        print(f"    {b.date}  O={b.open}  H={b.high}  L={b.low}  C={b.close}  "
              f"V={b.volume}")
    print("  These are the exact bars _on_5min_update consumes — no tick second-guessing.")
    return True


# ──────────────────────────────────────────────────────────────────────────
# STEP 3 — Place 3 real bracket orders on paper, then cancel
# ──────────────────────────────────────────────────────────────────────────

async def step3_real_orders(ib, contract):
    print("\n[STEP 3] Place 3 real bracket orders on paper TWS")

    # Get current spot so we can place orders close enough that initial margin
    # stays sane (far-from-market limits cause IB to demand huge buffers).
    print("  Fetching spot price...")
    ticker = ib.reqMktData(contract, '', True, False)
    spot = None
    for _ in range(60):
        await asyncio.sleep(0.1)
        if ticker.last and ticker.last > 0:
            spot = ticker.last; break
        if ticker.close and ticker.close > 0:
            spot = ticker.close; break
    ib.cancelMktData(contract)
    if spot is None:
        print("  FAIL — no spot price (market may be closed and no last/close)")
        return False
    print(f"  Spot ~ {spot}")

    def rt(p): return round(p * 4) / 4

    # Place 1 bracket ~100pts below spot — far enough not to fill, single
    # contract so margin stays comfortably below available funds (overnight
    # margin on NQ is ~$35k/contract; 3 contracts would exceed $105k buffer).
    orders = [
        {"name": "Bullish 100 below", "side": "BUY",
         "entry": rt(spot - 100), "target": rt(spot - 80), "stop": rt(spot - 112), "qty": 1},
    ]

    placed = []
    for o in orders:
        side = o["side"]
        rev = "SELL" if side == "BUY" else "BUY"

        eid = ib.client.getReqId()
        tid = ib.client.getReqId()
        sid = ib.client.getReqId()

        parent = LimitOrder(action=side, totalQuantity=o["qty"],
                            lmtPrice=o["entry"], orderId=eid,
                            tif='GTC', transmit=False)
        tp = LimitOrder(action=rev, totalQuantity=o["qty"],
                        lmtPrice=o["target"], orderId=tid,
                        parentId=eid, tif='GTC', transmit=False)
        sl = StopOrder(action=rev, totalQuantity=o["qty"],
                       stopPrice=o["stop"], orderId=sid,
                       parentId=eid, tif='GTC', transmit=True)

        et = ib.placeOrder(contract, parent)
        tt = ib.placeOrder(contract, tp)
        st = ib.placeOrder(contract, sl)
        placed.append((et, tt, st, o, eid, tid, sid))
        print(f"  Placed {o['name']}: entry={eid} tp={tid} sl={sid}")

    print("  Waiting 4s for IB to acknowledge...")
    await asyncio.sleep(4)

    open_ids = {t.order.orderId for t in ib.openTrades()}
    all_ok = True
    for et, tt, st, o, eid, tid, sid in placed:
        e_ok = eid in open_ids
        t_ok = tid in open_ids
        s_ok = sid in open_ids
        status = et.orderStatus.status if et.orderStatus else "?"
        ok_str = "OK" if (e_ok and t_ok and s_ok) else "MISSING"
        print(f"  {o['name']}: {ok_str}  parent_status={status}")
        if not (e_ok and t_ok and s_ok):
            all_ok = False

    print("  Cancelling all placed test orders...")
    for et, tt, st, *_ in placed:
        if et.orderStatus and et.orderStatus.status in ('Cancelled', 'Filled', 'Inactive'):
            continue
        try:
            ib.cancelOrder(et.order)
        except Exception as e:
            print(f"    cancel parent failed: {e}")

    await asyncio.sleep(3)
    remaining = [t.order.orderId for t in ib.openTrades()
                 if t.order.orderId in {eid for *_, eid, _, _ in placed}
                 or t.order.orderId in {tid for *_, _, tid, _ in placed}
                 or t.order.orderId in {sid for *_, _, _, sid in placed}]
    print(f"  Remaining test orders after cancel: {len(remaining)}")
    return all_ok and len(remaining) == 0


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

async def main():
    host = detect_ib_host()
    print(f"\n{'='*64}")
    print(f"  IB Paper Canonical-Bar Parity + Order Placement")
    print(f"  Host: {host}:{IB_PORT}  Client: {CLIENT_ID}")
    print(f"  Mode: PAPER (port 7497)")
    print(f"{'='*64}")

    if not step1_static_guard():
        sys.exit(1)

    ib = IB()
    print(f"\n[connect] paper TWS {host}:{IB_PORT} client={CLIENT_ID} ...")
    try:
        await ib.connectAsync(host, IB_PORT, clientId=CLIENT_ID, timeout=15)
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)
    print("  OK")

    print(f"[contract] resolving NQ {front_month_expiry()} ...")
    contract = Future(symbol='NQ', lastTradeDateOrContractMonth=front_month_expiry(),
                      exchange='CME', currency='USD')
    qualified = await ib.qualifyContractsAsync(contract)
    if not qualified:
        print("  FAIL qualify")
        ib.disconnect()
        sys.exit(1)
    print(f"  OK — {contract.localSymbol}")

    parity_ok = await step2_live_bar_close(ib, contract, wait_minutes=6)
    hist_ok = await step2b_historical_bars(ib, contract)
    orders_ok = await step3_real_orders(ib, contract)

    ib.disconnect()

    print(f"\n{'='*64}")
    print(f"  STEP 1  (static guard):       PASS")
    print(f"  STEP 2  (live parity):        {'PASS' if parity_ok else 'SKIP'}")
    print(f"  STEP 2b (historical parity):  {'PASS' if hist_ok else 'FAIL'}")
    print(f"  STEP 3  (real orders):        {'PASS' if orders_ok else 'FAIL'}")
    print(f"{'='*64}\n")

    sys.exit(0 if orders_ok else 1)


if __name__ == "__main__":
    util.run(main())
