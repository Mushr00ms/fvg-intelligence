"""
margin_tracker.py — IB margin query wrapper for intelligent margin management.

Time-aware: uses IB's intraday maintenance margin during RTH (09:30–16:00 ET)
and overnight initial margin outside RTH. Since the bot flattens before overnight,
the intraday rate is what matters for position sizing.

At startup, probes IB with whatIfOrder to get the exact initial margin per contract
for the current front-month NQ contract. Tracks available margin via accountValues().
Falls back to time-aware configurable estimates in dry_run mode or on API failure.
"""

import math
from datetime import datetime, time, timedelta
from typing import Optional


class MarginTracker:
    """Tracks per-contract margin requirements and available margin."""

    def __init__(self, ib_connection, contract, logger, config, clock=None):
        self._conn = ib_connection
        self._contract = contract
        self._logger = logger
        self._config = config
        self._clock = clock
        self._margin_per_contract: float = 0.0
        self._last_fetch_time: Optional[datetime] = None
        self._initialized = False

        # Parse intraday window from config (ET times)
        self._intraday_start = _parse_time(config.margin_intraday_start)
        self._intraday_end = _parse_time(config.margin_intraday_end)

    def _now(self):
        if self._clock is not None:
            return self._clock.now()
        return datetime.now()

    def _is_intraday(self) -> bool:
        """True if current time falls within the intraday margin window (ET)."""
        now = self._now()
        t = now.time()
        return self._intraday_start <= t < self._intraday_end

    def _time_aware_fallback(self) -> float:
        """Return the correct fallback margin for current time of day."""
        if self._is_intraday():
            return self._config.margin_intraday_maintenance
        return self._config.margin_overnight_initial

    async def initialize(self) -> float:
        """Fetch initial margin per contract at startup. Returns margin_per_contract."""
        margin = await self._fetch_margin_per_contract()
        self._initialized = True
        return margin

    async def refresh_if_stale(self):
        """Re-fetch margin if older than refresh_interval."""
        if self._last_fetch_time is None:
            return
        elapsed = (self._now() - self._last_fetch_time).total_seconds()
        if elapsed >= self._config.margin_refresh_interval:
            await self._fetch_margin_per_contract()

    async def _fetch_margin_per_contract(self) -> float:
        """Use ib.whatIfOrderAsync to get initMarginChange for 1 contract."""
        fallback = self._time_aware_fallback()

        if self._config.dry_run or not self._conn.is_connected:
            self._margin_per_contract = fallback
            self._last_fetch_time = self._now()
            self._logger.log(
                "margin_fetched",
                mode="fallback",
                per_contract=fallback,
                period="intraday" if self._is_intraday() else "overnight",
            )
            return fallback

        try:
            from ib_async import MarketOrder

            ib = self._conn.ib
            probe_order = MarketOrder(action="BUY", totalQuantity=1)
            what_if = await ib.whatIfOrderAsync(self._contract, probe_order)

            # whatIfOrder returns an OrderState with initMarginChange
            init_margin_str = getattr(what_if, "initMarginChange", "0")
            init_margin = float(init_margin_str) if init_margin_str else 0.0

            if init_margin <= 0:
                self._logger.log(
                    "margin_fetch_invalid",
                    raw_value=init_margin_str,
                    using_fallback=True,
                )
                self._margin_per_contract = fallback
            else:
                self._margin_per_contract = init_margin
                self._logger.log(
                    "margin_fetched",
                    mode="live",
                    per_contract=round(init_margin, 2),
                )

        except Exception as e:
            self._logger.log(
                "margin_fetch_error",
                error=str(e),
                using_fallback=True,
            )
            self._margin_per_contract = fallback

        self._last_fetch_time = self._now()
        return self._margin_per_contract

    def get_available_margin(self) -> float:
        """Query ib.accountValues() for available funds in the Commodities segment.

        Priority: AvailableFunds-C → AvailableFunds → 0.0
        """
        if self._config.dry_run or not self._conn.is_connected:
            return self._time_aware_fallback() * 3  # Assume 3-contract capacity in dry_run

        try:
            account_values = self._conn.ib.accountValues()
            # Try Commodities segment first (exact match for futures margin)
            for tag in ("AvailableFunds-C", "AvailableFunds"):
                for av in account_values:
                    if av.tag == tag and av.currency == "USD":
                        return float(av.value)
        except Exception:
            pass
        return 0.0

    def max_contracts_by_margin(self, available_margin: Optional[float] = None) -> int:
        """How many contracts can be opened with available margin (including buffer)."""
        if available_margin is None:
            available_margin = self.get_available_margin()
        if self._margin_per_contract <= 0:
            return 0
        buffered_margin = self._margin_per_contract * (1.0 + self._config.margin_buffer_pct)
        return max(0, math.floor(available_margin / buffered_margin))

    def margin_required_for(self, qty: int) -> float:
        """Margin required for qty contracts (without buffer)."""
        return qty * self._margin_per_contract

    def can_afford(self, qty: int, available_margin: Optional[float] = None) -> bool:
        """True if available margin covers qty contracts including safety buffer."""
        if available_margin is None:
            available_margin = self.get_available_margin()
        buffered = self._margin_per_contract * qty * (1.0 + self._config.margin_buffer_pct)
        return available_margin >= buffered

    @property
    def margin_per_contract(self) -> float:
        return self._margin_per_contract

    @property
    def is_initialized(self) -> bool:
        return self._initialized


def _parse_time(s: str) -> time:
    """Parse 'HH:MM' string to datetime.time."""
    h, m = s.split(":")
    return time(int(h), int(m))
