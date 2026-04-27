#!/usr/bin/env python3
"""Simulate what the BTC bot would have traded today (UTC day) using the locked strategy.

Fetches live 5m + 1m BTCUSDC bars from Binance, runs the same FVG detection,
mitigation scan, cell lookup, and capital-cap gate that the bot uses.
Reports every qualified signal with cell match and accept/reject status.
"""
import asyncio, aiohttp, json, sys, os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from crypto_bot.fvg import parse_ts, hourly_period

STRATEGY = "bot/strategies/btc-5min-locked-ev007-s30-both.json"
CAPITAL = 50_000
RISK_PCT = 0.0007
LEVERAGE = 1.0
MAKER_FEE = 0.0
STOP_FEE = 0.0004
MIT_WINDOW_5M = 90
MARKET_TZ = "UTC"
RISK_BINS = [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 994]


def risk_bucket(bps):
    for i in range(len(RISK_BINS) - 1):
        if RISK_BINS[i] <= bps < RISK_BINS[i + 1]:
            return f"{RISK_BINS[i]}-{RISK_BINS[i+1]}"
    return None


async def fetch_bars(session, symbol, interval, start_ms):
    out = []
    cursor = start_ms
    while True:
        r = await session.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "startTime": cursor, "limit": 1000},
        )
        data = await r.json()
        if not data:
            break
        out.extend(data)
        cursor = data[-1][0] + 1
        if len(data) < 1000:
            break
    return out


async def main():
    # Load strategy
    with open(STRATEGY) as f:
        strat = json.load(f)
    # Build cell lookup: (time_period, risk_range, setup) -> best_n
    cells = {}
    for c in strat["cells"]:
        cells[(c["time_period"], c["risk_range"], c["setup"])] = c
    print(f"Loaded {len(cells)} cells from {strat['meta']['id']}")

    # Today's UTC day start
    today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_ms = int(today_utc.timestamp() * 1000)
    print(f"Simulating from {today_utc.isoformat()} to now ({datetime.now(timezone.utc).isoformat()})")

    async with aiohttp.ClientSession() as s:
        bars_5m = await fetch_bars(s, "BTCUSDC", "5m", start_ms)
        bars_1m = await fetch_bars(s, "BTCUSDC", "1m", start_ms)
    print(f"  Fetched {len(bars_5m)} 5m bars, {len(bars_1m)} 1m bars")

    # Only use CLOSED bars (closeTime < now)
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    bars_5m = [b for b in bars_5m if b[6] < now_ms]
    bars_1m = [b for b in bars_1m if b[6] < now_ms]
    print(f"  Closed: {len(bars_5m)} 5m, {len(bars_1m)} 1m")

    # Detect FVGs on 5m (3-bar pattern, min 5 bps)
    fvgs = []
    for i in range(2, len(bars_5m)):
        c1 = bars_5m[i-2]
        c2 = bars_5m[i-1]
        c3 = bars_5m[i]
        c1_high, c1_low = float(c1[2]), float(c1[3])
        c2_close, c2_high, c2_low = float(c2[4]), float(c2[2]), float(c2[3])
        c3_high, c3_low = float(c3[2]), float(c3[3])
        c2_mid_low, c2_mid_high = c2_low, c2_high

        if c3_low > c1_high:
            z_low, z_high = c1_high, c3_low
            fvg_type = "bullish"
        elif c3_high < c1_low:
            z_low, z_high = c3_high, c1_low
            fvg_type = "bearish"
        else:
            continue
        size_bps = (z_high - z_low) / c2_close * 10000
        if size_bps < 5:
            continue
        fvgs.append({
            "formation_idx": i,
            "formation_ts": datetime.fromtimestamp(c3[0] / 1000, tz=timezone.utc),
            "type": fvg_type,
            "zone_low": z_low,
            "zone_high": z_high,
            "ref_price": c2_close,
            "size_bps": round(size_bps, 2),
            "middle_low": c2_mid_low,
            "middle_high": c2_mid_high,
            "mitigated": False,
        })

    print(f"\nDetected {len(fvgs)} FVGs today (5bps+)")

    # Scan 1m bars for mitigation of each FVG
    mitigated_events = []
    for fvg in fvgs:
        form_ms = int(fvg["formation_ts"].timestamp() * 1000)
        form_end_ms = form_ms + 300_000  # 5m bar close time
        expire_ms = form_ms + MIT_WINDOW_5M * 300_000
        for bar in bars_1m:
            open_ms = bar[0]
            if open_ms < form_end_ms:
                continue
            if open_ms > expire_ms:
                break
            b_low, b_high = float(bar[3]), float(bar[2])
            hit = False
            if fvg["type"] == "bullish":
                # Mitigation when price re-enters the zone from above
                if b_low <= fvg["zone_high"]:
                    hit = True
            else:
                if b_high >= fvg["zone_low"]:
                    hit = True
            if hit:
                fvg["mitigated"] = True
                fvg["mit_ts"] = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc)
                fvg["mit_price"] = b_high if fvg["type"] == "bearish" else b_low
                mitigated_events.append(fvg)
                break

    print(f"  Mitigated: {len(mitigated_events)}")

    # For each mitigation → build both setups, lookup cell, check capital cap
    bal = CAPITAL
    open_positions = []  # (release_ts, notional)
    signals = []
    for fvg in sorted(mitigated_events, key=lambda f: f["mit_ts"]):
        for setup in ("mit_extreme", "mid_extreme"):
            # Entry/stop calc matches crypto_bot/risk.py
            if fvg["type"] == "bullish":
                entry = fvg["zone_high"] if setup == "mit_extreme" else (fvg["zone_low"] + fvg["zone_high"]) / 2
                stop = fvg["middle_low"]
                side = "BUY"
            else:
                entry = fvg["zone_low"] if setup == "mit_extreme" else (fvg["zone_low"] + fvg["zone_high"]) / 2
                stop = fvg["middle_high"]
                side = "SELL"
            risk_bps = abs(entry - stop) / fvg["ref_price"] * 10000
            rb = risk_bucket(risk_bps)
            if rb is None:
                continue
            tp = hourly_period(fvg["formation_ts"], MARKET_TZ)
            key = (tp, rb, setup)
            cell = cells.get(key)
            if cell is None:
                continue
            # Capital-cap: release expired open positions
            mit_ts = fvg["mit_ts"]
            open_positions = [(r, n) for r, n in open_positions if r > mit_ts]
            open_notional = sum(n for _, n in open_positions)

            risk_dollar = bal * RISK_PCT
            per_unit_loss = abs(entry - stop) + (entry * MAKER_FEE) + (stop * STOP_FEE)
            qty = risk_dollar / per_unit_loss
            notional = qty * entry

            accepted = open_notional + notional <= bal * LEVERAGE
            signals.append({
                "time": mit_ts.strftime("%H:%M UTC"),
                "form": fvg["formation_ts"].strftime("%H:%M"),
                "type": fvg["type"][:4],
                "setup": setup,
                "bps": round(risk_bps, 1),
                "entry": round(entry, 2),
                "stop": round(stop, 2),
                "best_n": cell["best_n"],
                "cell_ev": cell["ev"],
                "notional": round(notional, 0),
                "cap": round(bal * LEVERAGE, 0),
                "open_before": round(open_notional, 0),
                "accepted": accepted,
                "cell_key": f"{tp} {rb} {setup[:4]}",
            })
            if accepted:
                # Lock until expiration (4h walk window)
                release = mit_ts + timedelta(minutes=240)
                open_positions.append((release, notional))

    # Report
    accepted_sigs = [s for s in signals if s["accepted"]]
    rejected_sigs = [s for s in signals if not s["accepted"]]
    print(f"\n{'='*110}")
    print(f"  Today's simulation @ 1x, $50k, 0.07% risk, UTC cells")
    print(f"{'='*110}")
    print(f"  FVGs detected:      {len(fvgs)}")
    print(f"  Mitigated:          {len(mitigated_events)}")
    print(f"  Qualified signals:  {len(signals)}  (both setups × cell match)")
    print(f"  Accepted by cap:    {len(accepted_sigs)}")
    print(f"  Rejected by cap:    {len(rejected_sigs)}")

    if signals:
        print(f"\n  {'mit':>8}  {'form':>5}  {'dir':>4}  {'setup':<11}  {'bps':>6}  {'N':>5}  {'ev':>6}  {'notional':>9}  {'accept':>7}  cell")
        print(f"  {'-'*120}")
        for s in signals:
            flag = "✓" if s["accepted"] else "REJ"
            print(f"  {s['time']:>8}  {s['form']:>5}  {s['type']:>4}  {s['setup']:<11}  {s['bps']:>6.1f}  {s['best_n']:>5.2f}  {s['cell_ev']:>6.3f}  ${s['notional']:>7,.0f}  {flag:>7}  {s['cell_key']}")


if __name__ == "__main__":
    asyncio.run(main())
