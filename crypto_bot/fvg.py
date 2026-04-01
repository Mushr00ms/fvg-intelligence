"""
fvg.py — BTC FVG detection and mitigation tracking for the crypto bot.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from crypto_bot.models import FVGRecord, _new_id


def parse_ts(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts.astimezone(timezone.utc)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)


def hourly_period(ts, market_tz: str = "America/New_York") -> str:
    dt = parse_ts(ts)
    dt = dt.astimezone(ZoneInfo(market_tz))
    start = dt.replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=1)
    return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"


def _format_market_ts(ts, market_tz: str) -> str:
    return parse_ts(ts).astimezone(ZoneInfo(market_tz)).isoformat()


def _format_bar_label(bar: dict, market_tz: str) -> str:
    return parse_ts(bar["open_time"]).astimezone(ZoneInfo(market_tz)).strftime("%H:%M")


def _format_candle_label(ts: str, market_tz: str) -> str:
    return parse_ts(ts).astimezone(ZoneInfo(market_tz)).strftime("%H:%M")


def _format_bar_range(bar: dict, market_tz: str) -> str:
    start = parse_ts(bar["open_time"]).astimezone(ZoneInfo(market_tz)).strftime("%H:%M")
    close_dt = parse_ts(bar["close_time"]).astimezone(ZoneInfo(market_tz)) + timedelta(milliseconds=1)
    end = close_dt.strftime("%H:%M")
    return f"{start}-{end}"


def detect_fvg_3bars(
    bar1: dict,
    bar2: dict,
    bar3: dict,
    min_fvg_bps: float = 5.0,
    market_tz: str = "America/New_York",
) -> FVGRecord | None:
    first_high = float(bar1["high"])
    first_low = float(bar1["low"])
    first_open = float(bar1["open"])
    middle_open = float(bar2["open"])
    middle_high = float(bar2["high"])
    middle_low = float(bar2["low"])
    ref_price = float(bar2["close"])
    third_low = float(bar3["low"])
    third_high = float(bar3["high"])
    time_candle1 = _format_market_ts(bar1["open_time"], market_tz)
    time_candle2 = _format_market_ts(bar2["open_time"], market_tz)
    time_candle3 = _format_market_ts(bar3["open_time"], market_tz)
    formation_date = time_candle3[:10]

    if ref_price <= 0:
        return None

    zone_low = zone_high = None
    fvg_type = None
    if third_low > first_high:
        zone_low = first_high
        zone_high = third_low
        fvg_type = "bullish"
    elif third_high < first_low:
        zone_low = third_high
        zone_high = first_low
        fvg_type = "bearish"
    else:
        return None

    fvg_size_bps = (zone_high - zone_low) / ref_price * 10_000
    if fvg_size_bps < min_fvg_bps:
        return None

    return FVGRecord(
        fvg_id=_new_id(),
        fvg_type=fvg_type,
        zone_low=round(zone_low, 8),
        zone_high=round(zone_high, 8),
        time_candle1=time_candle1,
        time_candle2=time_candle2,
        time_candle3=time_candle3,
        reference_price=round(ref_price, 8),
        time_period=hourly_period(parse_ts(bar3["open_time"]), market_tz),
        formation_time=_format_market_ts(bar3["close_time"], market_tz),
        formation_date=formation_date,
        first_open=first_open,
        middle_open=middle_open,
        middle_high=middle_high,
        middle_low=middle_low,
        mitigation_deadline=(
            parse_ts(bar3["close_time"]).astimezone(ZoneInfo(market_tz)) + timedelta(minutes=5 * 90)
        ).isoformat(),
    )


def is_mitigated(bar_1m: dict, fvg: FVGRecord) -> bool:
    return float(bar_1m["low"]) <= fvg.zone_high and float(bar_1m["high"]) >= fvg.zone_low


class ActiveFVGManager:
    def __init__(
        self,
        min_fvg_bps: float = 5.0,
        mitigation_window_5m: int = 90,
        logger=None,
        market_timezone: str = "America/New_York",
        symbol: str = "",
    ):
        self._min_fvg_bps = min_fvg_bps
        self._mitigation_window = mitigation_window_5m
        self._logger = logger
        self._market_timezone = market_timezone
        self._symbol = symbol
        self._recent_5m = deque(maxlen=128)
        self._active = {}
        self._last_5m_close_time = None

    def reset(self):
        self._recent_5m.clear()
        self._active.clear()
        self._last_5m_close_time = None

    def seed_5m(self, bars: list[dict]):
        for bar in bars:
            self.on_5m_close(bar, source="seed")

    def on_5m_close(self, bar: dict, *, source: str = "live") -> FVGRecord | None:
        bar_close_time = parse_ts(bar["close_time"])
        if self._last_5m_close_time is not None:
            if bar_close_time < self._last_5m_close_time:
                return None
            if bar_close_time == self._last_5m_close_time:
                if self._recent_5m:
                    self._recent_5m[-1] = bar
                return None

        self._recent_5m.append(bar)
        self._last_5m_close_time = bar_close_time
        self.expire_old(bar_close_time)
        if len(self._recent_5m) < 3:
            return None

        recent_bars = list(self._recent_5m)[-3:]
        formation_bars = " | ".join(
            f"C{index}={_format_bar_range(recent_bar, self._market_timezone)}"
            for index, recent_bar in enumerate(recent_bars, start=1)
        )

        fvg = detect_fvg_3bars(
            recent_bars[0],
            recent_bars[1],
            recent_bars[2],
            self._min_fvg_bps,
            self._market_timezone,
        )
        if fvg is None:
            return None
        fvg.mitigation_deadline = (
            parse_ts(bar["close_time"]).astimezone(ZoneInfo(self._market_timezone))
            + timedelta(minutes=5 * self._mitigation_window)
        ).isoformat()
        self._active[fvg.fvg_id] = fvg
        if self._logger and source == "live":
            self._logger.log(
                "crypto_fvg_detected",
                symbol=self._symbol,
                fvg_id=fvg.fvg_id,
                fvg_type=fvg.fvg_type,
                time_candle1=fvg.time_candle1,
                time_candle2=fvg.time_candle2,
                time_candle3=fvg.time_candle3,
                c1_high=float(recent_bars[0]["high"]),
                c1_low=float(recent_bars[0]["low"]),
                c2_high=float(recent_bars[1]["high"]),
                c2_low=float(recent_bars[1]["low"]),
                c3_high=float(recent_bars[2]["high"]),
                c3_low=float(recent_bars[2]["low"]),
                zone_low=fvg.zone_low,
                zone_high=fvg.zone_high,
                time_period=fvg.time_period,
                confirmed_at=_format_market_ts(bar["close_time"], self._market_timezone),
                display_time=_format_candle_label(fvg.time_candle3, self._market_timezone),
                formation_bars=(
                    f"C1={_format_candle_label(fvg.time_candle1, self._market_timezone)}-"
                    f"{_format_candle_label(fvg.time_candle2, self._market_timezone)} | "
                    f"C2={_format_candle_label(fvg.time_candle2, self._market_timezone)}-"
                    f"{_format_candle_label(fvg.time_candle3, self._market_timezone)} | "
                    f"C3={_format_candle_label(fvg.time_candle3, self._market_timezone)}-"
                    f"{(parse_ts(fvg.formation_time).astimezone(ZoneInfo(self._market_timezone)) + timedelta(milliseconds=1)).strftime('%H:%M')}"
                ),
                reference_price=fvg.reference_price,
            )
        return fvg

    def expire_old(self, now_ts: datetime):
        expired = []
        for fvg in self._active.values():
            if not fvg.is_mitigated and parse_ts(fvg.mitigation_deadline) < now_ts:
                fvg.expired = True
                expired.append(fvg.fvg_id)
        for fvg_id in expired:
            self._active.pop(fvg_id, None)

    def scan_1m_close(self, bar: dict):
        mitigated = []
        now = parse_ts(bar["close_time"])
        self.expire_old(now)
        for fvg in list(self._active.values()):
            if fvg.is_mitigated or fvg.expired:
                continue
            if is_mitigated(bar, fvg):
                fvg.is_mitigated = True
                fvg.mitigation_time = now.astimezone(ZoneInfo(self._market_timezone)).isoformat()
                mitigated.append(fvg)
                self._active.pop(fvg.fvg_id, None)
        return mitigated

    @property
    def active(self) -> list[FVGRecord]:
        return list(self._active.values())
