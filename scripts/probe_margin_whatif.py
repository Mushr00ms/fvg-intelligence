"""
probe_margin_whatif.py — empirical test of whatIfOrderAsync probe shapes
against paper TWS, to verify the TIF fix for the UNSET_DOUBLE / [] bug.

Connects to 127.0.0.1:7497 with a distinct clientId so it does not collide
with the running bot. Runs four probe variants and prints the raw
initMarginChange (or [] / error) for each, so we can see exactly which
shape IB resolves cleanly.
"""

import asyncio
import subprocess
import sys
from datetime import datetime

from ib_async import IB, Future, LimitOrder, MarketOrder

IB_PORT = 7497
CLIENT_ID = 199  # distinct from bot + other tests


def detect_host():
    try:
        r = subprocess.run(["ip", "route", "show", "default"],
                           capture_output=True, text=True, timeout=3)
        for p in r.stdout.split():
            if "." in p and p[0].isdigit():
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


async def get_spot(ib, contract):
    t = ib.reqMktData(contract, "", True, False)
    spot = None
    for _ in range(60):
        await asyncio.sleep(0.1)
        if t.last and t.last > 0:
            spot = t.last; break
        if t.bid and t.ask and t.bid > 0 and t.ask > 0:
            spot = (t.bid + t.ask) / 2; break
        if t.close and t.close > 0:
            spot = t.close; break
    ib.cancelMktData(contract)
    return spot


def describe(probe_name, what_if):
    if isinstance(what_if, list):
        return f"{probe_name}: BROKEN — returned {what_if!r} (UNSET_DOUBLE)"
    raw = getattr(what_if, "initMarginChange", "") or ""
    maint = getattr(what_if, "maintMarginChange", "") or ""
    eq = getattr(what_if, "equityWithLoanChange", "") or ""
    return (f"{probe_name}: OK — initMarginChange={raw!r} "
            f"maintMarginChange={maint!r} equityWithLoanChange={eq!r}")


async def run_probe(ib, contract, name, order):
    try:
        wi = await asyncio.wait_for(ib.whatIfOrderAsync(contract, order), timeout=6.0)
        print("  " + describe(name, wi))
    except asyncio.TimeoutError:
        print(f"  {name}: TIMEOUT after 6s")
    except Exception as e:
        print(f"  {name}: ERROR {type(e).__name__}: {e}")


async def main():
    host = detect_host()
    print(f"\n=== whatIf probe diagnostics ===")
    print(f"Host: {host}:{IB_PORT}  clientId={CLIENT_ID}\n")

    ib = IB()
    try:
        await ib.connectAsync(host, IB_PORT, clientId=CLIENT_ID, timeout=15)
    except Exception as e:
        print(f"connect FAIL: {e}")
        sys.exit(1)
    print("connected")

    # Pick primary account (largest NLV)
    primary_account = None
    try:
        avs = ib.accountValues()
        per_acc = {}
        for av in avs:
            if av.tag == "NetLiquidation" and av.currency == "USD":
                try:
                    per_acc[av.account] = float(av.value)
                except (TypeError, ValueError):
                    pass
        if per_acc:
            primary_account = max(per_acc, key=lambda a: per_acc[a])
            print(f"primary_account={primary_account} NLV={per_acc[primary_account]:.2f}")
            if len(per_acc) > 1:
                print(f"  (multi-account: {per_acc})")
    except Exception as e:
        print(f"account scan err: {e}")

    expiry = front_month_expiry()
    print(f"\nResolving NQ {expiry}...")
    contract = Future(symbol="NQ", lastTradeDateOrContractMonth=expiry,
                      exchange="CME", currency="USD")
    q = await ib.qualifyContractsAsync(contract)
    if not q:
        print("qualify FAIL"); ib.disconnect(); sys.exit(1)
    print(f"  -> {contract.localSymbol}  conId={contract.conId}")

    spot = await get_spot(ib, contract)
    if not spot:
        print("no spot"); ib.disconnect(); sys.exit(1)
    spot = round(spot * 4) / 4
    print(f"spot={spot}\n")

    # ── Variant A: bare LimitOrder (current code, expected broken) ──
    a = LimitOrder(action="BUY", totalQuantity=1, lmtPrice=spot)
    print("[A] bare LimitOrder (no tif):")
    await run_probe(ib, contract, "A", a)

    # ── Variant B: LimitOrder + tif="DAY" ──
    b = LimitOrder(action="BUY", totalQuantity=1, lmtPrice=spot)
    b.tif = "DAY"
    b.outsideRth = False
    print("\n[B] LimitOrder + tif=DAY:")
    await run_probe(ib, contract, "B", b)

    # ── Variant C: LimitOrder + tif=DAY + account ──
    if primary_account:
        c = LimitOrder(action="BUY", totalQuantity=1, lmtPrice=spot)
        c.tif = "DAY"
        c.outsideRth = False
        c.account = primary_account
        print("\n[C] LimitOrder + tif=DAY + account:")
        await run_probe(ib, contract, "C", c)

    # ── Variant D: MarketOrder + tif=DAY (+account) ──
    d = MarketOrder(action="BUY", totalQuantity=1)
    d.tif = "DAY"
    d.outsideRth = False
    if primary_account:
        d.account = primary_account
    print("\n[D] MarketOrder + tif=DAY:")
    await run_probe(ib, contract, "D", d)

    ib.disconnect()
    print("\ndone")


if __name__ == "__main__":
    asyncio.run(main())
