"""
validate_crypto_fvg_live.py - Read-only production-market validator for crypto FVG detection.

This script never uses private Binance endpoints and never places orders.
It validates the live 5m detector by:
1. fetching recent BTCUSDC 5m futures bars from Binance production
2. seeding ActiveFVGManager with historical bars
3. waiting for the next closed 5m production websocket bar
4. comparing the live detector output against a direct offline 3-bar recomputation
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from datetime import datetime, timezone
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from crypto_bot.fvg import ActiveFVGManager, detect_fvg_3bars, parse_ts
from crypto_bot.market_data import BinanceMarketData

PROD_BASE_URL = "https://fapi.binance.com"
PROD_WS_BASE_URL = "wss://fstream.binance.com"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate live crypto FVG detection on Binance production data")
    parser.add_argument("--symbol", default="BTCUSDC", help="Binance USD-M symbol")
    parser.add_argument("--min-fvg-bps", type=float, default=5.0, help="Minimum FVG size in bps")
    parser.add_argument("--mitigation-window", type=int, default=90, help="Mitigation window in 5m bars")
    parser.add_argument("--seed-bars", type=int, default=96, help="Historical 5m bars used to seed the detector")
    parser.add_argument("--max-wait-seconds", type=int, default=420, help="Maximum time to wait for the next closed 5m bar")
    return parser.parse_args()


def summarize_fvg(fvg):
    if fvg is None:
        return None
    return {
        "fvg_type": fvg.fvg_type,
        "zone_low": fvg.zone_low,
        "zone_high": fvg.zone_high,
        "time_period": fvg.time_period,
        "formation_time": fvg.formation_time,
        "reference_price": fvg.reference_price,
    }


async def wait_for_next_closed_5m_bar(
    market_data: BinanceMarketData,
    symbol: str,
    max_wait_seconds: int,
    *,
    after_close_time: str,
):
    start = datetime.now(timezone.utc)
    async for bar in market_data.stream_closed_klines(symbol, ["5m"]):
        age = (datetime.now(timezone.utc) - start).total_seconds()
        if age > max_wait_seconds:
            raise TimeoutError(f"Timed out waiting for next closed 5m bar after {max_wait_seconds}s")
        if parse_ts(bar["close_time"]) <= parse_ts(after_close_time):
            continue
        return bar
    raise RuntimeError("5m stream closed before a bar was received")


async def main():
    args = parse_args()
    market_data = BinanceMarketData(PROD_BASE_URL, PROD_WS_BASE_URL, market_timezone="America/New_York")
    await market_data.start()
    try:
        bars = await market_data.fetch_klines(args.symbol, "5m", limit=max(args.seed_bars, 3) + 2)
        seed_bars = bars[-max(args.seed_bars, 3):]

        mgr = ActiveFVGManager(
            min_fvg_bps=args.min_fvg_bps,
            mitigation_window_5m=args.mitigation_window,
        )
        recent = deque(maxlen=3)
        historical_detections = []
        for bar in seed_bars:
            recent.append(bar)
            live_detected = mgr.on_5m_close(bar)
            if live_detected is not None:
                historical_detections.append(summarize_fvg(live_detected))

        print("=" * 72)
        print(f"PRODUCTION FVG VALIDATION | symbol={args.symbol} | seed_bars={len(seed_bars)}")
        print(f"Last historical bar close: {seed_bars[-1]['close_time']}")
        print(f"Historical detections during seed: {len(historical_detections)}")
        print("=" * 72)

        recent.clear()
        for bar in seed_bars[-2:]:
            recent.append(bar)

        print("Waiting for next closed production 5m bar...")
        live_bar = await wait_for_next_closed_5m_bar(
            market_data,
            args.symbol,
            max_wait_seconds=args.max_wait_seconds,
            after_close_time=seed_bars[-1]["close_time"],
        )

        recent.append(live_bar)
        offline_detected = detect_fvg_3bars(
            recent[0],
            recent[1],
            recent[2],
            min_fvg_bps=args.min_fvg_bps,
        )
        live_detected = mgr.on_5m_close(live_bar)

        offline_summary = summarize_fvg(offline_detected)
        live_summary = summarize_fvg(live_detected)
        matched = offline_summary == live_summary

        print(f"New live 5m bar close: {live_bar['close_time']}")
        print(f"Offline recomputation: {offline_summary}")
        print(f"Live detector output: {live_summary}")
        print(f"Match: {matched}")

        if not matched:
            raise SystemExit(1)
    finally:
        await market_data.close()


if __name__ == "__main__":
    asyncio.run(main())
