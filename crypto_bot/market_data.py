"""
market_data.py — Public Binance websocket and REST helpers for market data.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import aiohttp


class BinanceMarketData:
    def __init__(self, base_url: str, ws_base_url: str, market_timezone: str = "America/New_York"):
        self._base_url = base_url.rstrip("/")
        self._ws_base_url = ws_base_url.rstrip("/")
        self._market_tz = ZoneInfo(market_timezone)
        self._session = None

    def _ms_to_iso(self, ms: int) -> str:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(self._market_tz).isoformat()

    def _parse_kline(self, k: dict) -> dict:
        return {
            "open_time": self._ms_to_iso(int(k["t"])),
            "close_time": self._ms_to_iso(int(k["T"])),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "interval": k["i"],
        }

    def _parse_agg_trade(self, payload: dict) -> dict:
        return {
            "trade_time": self._ms_to_iso(int(payload["T"])),
            "event_time": self._ms_to_iso(int(payload["E"])),
            "price": float(payload["p"]),
            "quantity": float(payload["q"]),
            "symbol": payload["s"],
        }

    async def start(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def fetch_klines(self, symbol: str, interval: str, limit: int = 500) -> list[dict]:
        if self._session is None:
            await self.start()
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        async with self._session.get(f"{self._base_url}/fapi/v1/klines", params=params) as resp:
            resp.raise_for_status()
            rows = await resp.json()
        bars = []
        for row in rows:
            bars.append({
                "open_time": self._ms_to_iso(int(row[0])),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": self._ms_to_iso(int(row[6])),
                "interval": interval,
            })
        return bars

    async def stream_closed_klines(self, symbol: str, intervals: list[str]):
        if self._session is None:
            await self.start()
        stream = "/".join(f"{symbol.lower()}@kline_{interval}" for interval in intervals)
        url = f"{self._ws_base_url}/stream?streams={stream}"
        async with self._session.ws_connect(url, heartbeat=20) as ws:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                payload = json.loads(msg.data)
                data = payload.get("data", {})
                if data.get("e") != "kline":
                    continue
                k = data.get("k", {})
                if not k.get("x"):
                    continue
                yield self._parse_kline(k)

    async def stream_market_events(self, symbol: str, intervals: list[str], *, include_agg_trade: bool = False):
        if self._session is None:
            await self.start()
        streams = [f"{symbol.lower()}@kline_{interval}" for interval in intervals]
        if include_agg_trade:
            streams.append(f"{symbol.lower()}@aggTrade")
        url = f"{self._ws_base_url}/stream?streams={'/'.join(streams)}"
        async with self._session.ws_connect(url, heartbeat=20) as ws:
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                payload = json.loads(msg.data)
                data = payload.get("data", {})
                event_type = data.get("e")
                if event_type == "kline":
                    kline = data.get("k", {})
                    if not kline.get("x"):
                        continue
                    yield {"type": "kline", "bar": self._parse_kline(kline)}
                elif include_agg_trade and event_type == "aggTrade":
                    yield {"type": "agg_trade", "tick": self._parse_agg_trade(data)}
