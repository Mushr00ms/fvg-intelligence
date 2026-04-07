"""
margin_tracker.py — IB margin query wrapper for intelligent margin management.

Time-aware: uses IB's intraday maintenance margin during RTH (09:30–16:00 ET)
and overnight initial margin outside RTH. Since the bot flattens before overnight,
the intraday rate is what matters for position sizing.

Probe strategy for per-contract margin (in order):
  1. whatIfOrderAsync with LimitOrder at current market price — most accurate
  2. whatIfOrderAsync with MarketOrder — fallback if price unavailable
  3. Derive from open NQ positions: InitMarginReq-C / position_size
  4. Time-aware configured fallback estimate

Also tracks locally-reserved margin for pending orders to avoid race conditions
where two orders are evaluated against the same pre-order available funds before
IB updates accountValues.
"""

import math
from datetime import datetime, time, timedelta
from typing import Optional

# IB uses DBL_MAX as a sentinel meaning "field not computed/applicable"
_IB_MAX_DOUBLE = 1.7976931348623157e+308


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
        self._reserved_margin: float = 0.0  # locally reserved for pending orders
        self._primary_account: Optional[str] = None  # set by initialize()

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
        """Return the correct fallback margin for current time of day.

        Uses INITIAL margin (not maintenance) because this is used to check
        whether we can OPEN new positions. Maintenance is lower and would
        allow orders that IB rejects.
        """
        if self._is_intraday():
            return self._config.margin_intraday_initial
        return self._config.margin_overnight_initial

    async def initialize(self) -> float:
        """Fetch initial margin per contract at startup. Returns margin_per_contract."""
        margin = await self._fetch_margin_per_contract()
        self._initialized = True

        # Log account segment state so we can see the real margin capacity.
        # IB returns one row per (account, tag, currency) tuple. With multiple
        # managed accounts or model/master separation, naïvely doing
        # `vals[tag] = float(av.value)` overwrites with whichever row IB
        # iterated last — frequently a near-zero subaccount. We pick the
        # account with the largest NetLiquidation as the "primary" and read
        # all tags from that account only.
        if not self._config.dry_run and self._conn.is_connected:
            try:
                avs = self._conn.ib.accountValues()
                wanted_tags = {
                    "NetLiquidation", "NetLiquidation-C",
                    "AvailableFunds-C", "InitMarginReq-C",
                }

                # Build per-account view: {account: {tag: value}}
                per_account: dict = {}
                for av in avs:
                    if av.currency != "USD" or av.tag not in wanted_tags:
                        continue
                    try:
                        v = float(av.value)
                    except (TypeError, ValueError):
                        continue
                    per_account.setdefault(av.account, {})[av.tag] = v

                # Pick the account with the largest NetLiquidation as primary.
                # Falls back to NetLiquidation-C, then any account, then empty.
                def _nlv(acc_vals):
                    return acc_vals.get("NetLiquidation", 0) or acc_vals.get("NetLiquidation-C", 0)

                primary_account = None
                vals: dict = {}
                if per_account:
                    primary_account = max(per_account, key=lambda a: _nlv(per_account[a]))
                    vals = per_account[primary_account]
                    self._primary_account = primary_account

                # If multiple accounts exist, surface that — single biggest cause
                # of the "nlv=0.63" symptom.
                if len(per_account) > 1:
                    self._logger.log(
                        "margin_account_multi",
                        accounts=sorted(per_account.keys()),
                        primary=primary_account,
                        nlvs={a: round(_nlv(v), 2) for a, v in per_account.items()},
                    )

                nlv_c = vals.get("NetLiquidation-C", 0)
                avail_c = vals.get("AvailableFunds-C", 0)
                max_contracts = int(avail_c / (margin * (1 + self._config.margin_buffer_pct))) if margin > 0 else 0
                self._logger.log(
                    "margin_account_state",
                    account=primary_account,
                    nlv=round(vals.get("NetLiquidation", 0), 2),
                    nlv_c=round(nlv_c, 2),
                    avail_c=round(avail_c, 2),
                    init_margin_c=round(vals.get("InitMarginReq-C", 0), 2),
                    per_contract=round(margin, 2),
                    max_new_contracts=max_contracts,
                )
            except Exception as e:
                self._logger.log("margin_account_state_error", error=str(e))

        return margin

    async def refresh_if_stale(self):
        """Re-fetch margin if older than refresh_interval."""
        if self._last_fetch_time is None:
            return
        elapsed = (self._now() - self._last_fetch_time).total_seconds()
        if elapsed >= self._config.margin_refresh_interval:
            await self._fetch_margin_per_contract()

    def reserve(self, qty: int):
        """Reserve margin for a pending order immediately after placement.

        Prevents a race condition where two orders are evaluated against the
        same pre-order accountValues snapshot before IB updates AvailableFunds.
        """
        reserved = self._margin_per_contract * qty
        self._reserved_margin += reserved
        self._logger.log(
            "margin_reserved",
            qty=qty,
            reserved=round(reserved, 2),
            total_reserved=round(self._reserved_margin, 2),
        )

    def release(self, qty: int):
        """Release reserved margin when an order fills or cancels."""
        released = self._margin_per_contract * qty
        self._reserved_margin = max(0.0, self._reserved_margin - released)
        self._logger.log(
            "margin_released",
            qty=qty,
            released=round(released, 2),
            total_reserved=round(self._reserved_margin, 2),
        )

    async def _fetch_margin_per_contract(self) -> float:
        """Fetch initial margin per NQ contract via IB API.

        Probe order:
          1. LimitOrder at current market price (most reliable for futures)
          2. MarketOrder (fallback if no price available)
          3. Derive from open NQ positions via InitMarginReq-C
          4. Time-aware configured fallback
        """
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

        import asyncio
        from ib_async import LimitOrder, MarketOrder

        ib = self._conn.ib

        # ── Step 1: get current market price for limit order probe ──────────
        # First try passive tickers (already subscribed via bar feed).
        # If nothing available, request an active snapshot — IB needs a price
        # to compute initMarginChange for futures; MarketOrder returns empty.
        current_price = None
        try:
            tickers = ib.tickers()
            for t in tickers:
                symbol = getattr(getattr(t, "contract", None), "symbol", "") or ""
                if "NQ" not in symbol:
                    continue
                last = getattr(t, "last", None)
                bid = getattr(t, "bid", None)
                ask = getattr(t, "ask", None)
                if last and last > 0:
                    current_price = round(last * 4) / 4
                    break
                if bid and ask and bid > 0 and ask > 0:
                    current_price = round(((bid + ask) / 2) * 4) / 4
                    break
        except Exception:
            pass

        if not current_price:
            # Active snapshot — waits up to 3s for bid/ask/last/close
            try:
                snap = ib.reqMktData(self._contract, snapshot=True, regulatorySnapshot=False)
                for _ in range(15):  # 15 × 0.2s = 3s max
                    await asyncio.sleep(0.2)
                    _bid = getattr(snap, "bid", None)
                    _ask = getattr(snap, "ask", None)
                    _last = getattr(snap, "last", None)
                    _close = getattr(snap, "close", None)
                    if _last and _last > 0:
                        current_price = round(_last * 4) / 4
                        break
                    if _bid and _ask and _bid > 0 and _ask > 0:
                        current_price = round(((_bid + _ask) / 2) * 4) / 4
                        break
                    if _close and _close > 0:
                        current_price = round(_close * 4) / 4
                        break
                ib.cancelMktData(self._contract)
                if current_price:
                    self._logger.log("margin_price_snapshot", price=current_price)
            except Exception as e:
                self._logger.log("margin_price_snapshot_error", error=str(e))

        # ── Step 2: whatIfOrder probes ──────────────────────────────────────
        probes = []
        if current_price:
            probes.append((
                "LimitOrder",
                LimitOrder(action="BUY", totalQuantity=1, lmtPrice=current_price),
            ))
        probes.append(("MarketOrder", MarketOrder(action="BUY", totalQuantity=1)))

        for probe_name, probe_order in probes:
            for attempt in range(2):
                try:
                    what_if = await asyncio.wait_for(
                        ib.whatIfOrderAsync(self._contract, probe_order),
                        timeout=5.0,
                    )

                    # ib_async returns [] when IB sends UNSET_DOUBLE for
                    # initMarginChange (wrapper never resolves the future)
                    if isinstance(what_if, list):
                        self._logger.log(
                            "margin_fetch_invalid",
                            probe=probe_name,
                            raw_value="[] (whatIfOrder broken — UNSET_DOUBLE)",
                            attempt=attempt + 1,
                        )
                        break  # no point retrying same probe type

                    raw = getattr(what_if, "initMarginChange", "") or ""
                    raw = str(raw).strip()

                    try:
                        val = float(raw) if raw else 0.0
                    except (ValueError, OverflowError):
                        val = 0.0

                    # IB returns DBL_MAX as sentinel for "not computed"
                    if val >= _IB_MAX_DOUBLE * 0.99:
                        val = 0.0

                    if val > 1000:  # sanity: NQ margin must be > $1,000
                        self._margin_per_contract = val
                        self._last_fetch_time = self._now()
                        self._logger.log(
                            "margin_fetched",
                            mode="live",
                            probe=probe_name,
                            per_contract=round(val, 2),
                        )
                        return self._margin_per_contract

                    self._logger.log(
                        "margin_fetch_invalid",
                        probe=probe_name,
                        raw_value=raw,
                        parsed=round(val, 2),
                        attempt=attempt + 1,
                    )
                    if attempt == 0:
                        await asyncio.sleep(1.5)

                except asyncio.TimeoutError:
                    self._logger.log(
                        "margin_fetch_error",
                        probe=probe_name,
                        error="timeout_5s",
                        attempt=attempt + 1,
                    )
                except Exception as e:
                    self._logger.log(
                        "margin_fetch_error",
                        probe=probe_name,
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    if attempt == 0:
                        await asyncio.sleep(1.0)

        # ── Step 3: derive from open NQ positions ───────────────────────────
        try:
            positions = ib.positions()
            nq_qty = sum(
                abs(int(p.position)) for p in positions
                if "NQ" in (getattr(p.contract, "symbol", "") or "")
                   and p.position != 0
            )
            if nq_qty > 0:
                account_values = ib.accountValues()
                for av in account_values:
                    if av.tag in ("InitMarginReq-C", "FullInitMarginReq-C") and av.currency == "USD":
                        total_req = float(av.value)
                        if total_req > 1000:
                            per_contract = total_req / nq_qty
                            self._margin_per_contract = per_contract
                            self._last_fetch_time = self._now()
                            self._logger.log(
                                "margin_fetched",
                                mode="from_positions",
                                nq_qty=nq_qty,
                                total_req=round(total_req, 2),
                                per_contract=round(per_contract, 2),
                            )
                            return self._margin_per_contract
        except Exception as e:
            self._logger.log(
                "margin_fetch_error", probe="from_positions", error=str(e)
            )

        # ── Step 4: configured fallback ─────────────────────────────────────
        self._margin_per_contract = fallback
        self._last_fetch_time = self._now()
        self._logger.log(
            "margin_fetched",
            mode="fallback",
            per_contract=fallback,
            period="intraday" if self._is_intraday() else "overnight",
        )
        return fallback

    def get_available_margin(self) -> float:
        """Query IB accountValues for available funds, minus locally reserved margin.

        Subtracts _reserved_margin to account for pending orders that have been
        placed but not yet reflected in IB's accountValues snapshot.

        Priority: AvailableFunds-C → AvailableFunds → 0.0
        """
        if self._config.dry_run or not self._conn.is_connected:
            return self._time_aware_fallback() * 3  # Assume 3-contract capacity in dry_run

        raw_available = 0.0
        try:
            account_values = self._conn.ib.accountValues()
            for tag in ("AvailableFunds-C", "AvailableFunds"):
                for av in account_values:
                    if av.tag != tag or av.currency != "USD":
                        continue
                    # Pin to the primary account discovered at init. Without
                    # this, IB's row order can hand us a near-zero subaccount
                    # and silently block all trades.
                    if self._primary_account and av.account != self._primary_account:
                        continue
                    try:
                        raw_available = float(av.value)
                    except (TypeError, ValueError):
                        continue
                    break
                if raw_available > 0:
                    break
        except Exception:
            pass

        return max(0.0, raw_available - self._reserved_margin)

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
