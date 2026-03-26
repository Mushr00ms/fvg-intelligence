"""
engine.py — BotEngine: central coordinator wiring all components.

Manages the event loop: bar subscriptions, FVG detection, mitigation scanning,
trade evaluation, order placement, state persistence, and EOD scheduling.
"""

import asyncio
import signal
import sys
import os
from collections import deque
from datetime import datetime, timedelta

import pytz

# Add project root for logic imports
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bot.bot_config import BotConfig
from bot.clock import Clock
from bot.bot_logging.bot_logger import BotLogger
from bot.state.state_manager import StateManager
from bot.state.trade_state import DailyState
from bot.strategy.strategy_loader import StrategyLoader
from bot.strategy.fvg_detector import ActiveFVGManager
from bot.strategy.mitigation_scanner import scan_active_fvgs
from bot.strategy.tick_bar_builder import TickBarBuilder
from bot.strategy.trade_calculator import calculate_setup
from bot.risk.risk_gates import RiskGates
from bot.risk.time_gates import TimeGates
from bot.execution.ib_connection import IBConnection
from bot.execution.order_manager import OrderManager
from bot.execution.position_tracker import get_account_balance, get_ib_open_orders, get_ib_positions
from bot.alerts.telegram import TelegramAlerter
from bot.db import TradeDB

NY_TZ = pytz.timezone("America/New_York")
CME_TZ = pytz.timezone("America/Chicago")  # CME exchange timezone


def _bar_date_to_et(bar_date):
    """Convert IB bar date (naive, in CME/Central time) to ET datetime."""
    if bar_date is None:
        return bar_date
    if hasattr(bar_date, 'tzinfo') and bar_date.tzinfo is not None:
        # Already timezone-aware — just convert to ET
        return bar_date.astimezone(NY_TZ)
    # Naive datetime from IB — assume CME (Central) timezone
    return CME_TZ.localize(bar_date).astimezone(NY_TZ)


def _tick_time_to_et(tick_time):
    """Convert IB tick-by-tick timestamp to ET datetime.

    IB tick-by-tick AllLast times are UTC (naive datetime from ib_async).
    """
    if tick_time is None:
        return None
    if hasattr(tick_time, 'tzinfo') and tick_time.tzinfo is not None:
        return tick_time.astimezone(NY_TZ)
    # Naive datetime from IB ticks — assume UTC
    return pytz.utc.localize(tick_time).astimezone(NY_TZ)


class BotEngine:
    """
    Central coordinator for the FVG trading bot.

    Lifecycle:
    1. Load config, strategy, init all components
    2. Connect to IB, resolve contract
    3. Load/reconcile state
    4. Subscribe to bar data
    5. Schedule EOD actions + periodic saves
    6. Run event loop
    7. Graceful shutdown
    """

    def __init__(self, config: BotConfig):
        self.config = config

        # NTP-synced clock — all time-sensitive operations use this
        self.clock = Clock()
        self.clock.sync()

        self.logger = BotLogger(config.log_dir, clock=self.clock)
        self.state_mgr = StateManager(config.state_dir, self.logger, clock=self.clock)
        self.strategy = StrategyLoader(config.strategy_dir, self.logger)
        self.risk_gates = RiskGates(config)
        self.time_gates = TimeGates(config, clock=self.clock)
        self.ib_conn = IBConnection(
            config.ib_host, config.ib_port, config.ib_client_id, self.logger,
            clock=self.clock,
        )
        self.db = TradeDB(os.path.join(config.state_dir, "bot_trades.db"))
        self.telegram = TelegramAlerter(
            config.telegram_bot_token, config.telegram_chat_id, self.logger,
            db=self.db,
        )
        self.fvg_mgr = None         # Initialized after strategy loads
        self.order_mgr = None        # Initialized after IB connects
        self.margin_tracker = None   # Initialized after contract resolution
        self.margin_priority = None  # Initialized after margin tracker
        self.daily_state = None
        self._contract = None
        self._bars_5min = None
        self._bars_1min = None
        self._tick_bar_builder = TickBarBuilder()
        self._tick_subscription = None
        self._tick_detected_bars = set()  # window_start datetimes already tick-detected
        self._shutdown = False
        self._reconciliation_complete = False  # Block trading until startup finishes
        self._detection_lock = asyncio.Lock()
        self._eod_tasks = []
        self._periodic_tasks = []

    async def run(self):
        """Main entry point. Runs the bot until session end or kill switch."""
        try:
            await self._startup()
            await self._event_loop()
        except KeyboardInterrupt:
            self.logger.log("bot_stop", reason="keyboard_interrupt")
        except Exception as e:
            self.logger.log("bot_error", error=str(e))
            raise
        finally:
            await self._shutdown_gracefully()

    async def _startup(self):
        """Initialize all components and prepare for trading."""
        self.logger.log("bot_start", mode="PAPER" if self.config.paper_mode else "LIVE",
                        dry_run=self.config.dry_run)

        # 1. Load strategy
        self.strategy.load()
        self.fvg_mgr = ActiveFVGManager(self.strategy, self.config.min_fvg_size, self.logger)

        # 2. Connect to IB
        if not self.config.dry_run:
            await self.ib_conn.connect()

            # 3. Resolve NQ front-month contract
            self._contract = await self._resolve_contract()

            # 4. Initialize order manager
            self.order_mgr = OrderManager(
                self.ib_conn, self._contract, self.state_mgr, self.logger, self.config,
                clock=self.clock, db=self.db,
                on_kill_switch=self._trigger_kill_switch,
            )

            # 4b. Initialize margin management
            if self.config.margin_management_enabled:
                from bot.risk.margin_tracker import MarginTracker
                from bot.risk.margin_priority import MarginPriorityManager
                self.margin_tracker = MarginTracker(
                    self.ib_conn, self._contract, self.logger, self.config,
                    clock=self.clock,
                )
                await self.margin_tracker.initialize()
                self.margin_priority = MarginPriorityManager(
                    self.margin_tracker, self.order_mgr,
                    self.risk_gates, self.time_gates,
                    self.logger, self.config, clock=self.clock,
                )
                # Register margin re-evaluation callback on order resolution
                self.order_mgr._on_order_resolved = self._on_order_resolved

        # 5. Load or create daily state
        self.daily_state = self.state_mgr.load()
        if self.daily_state is None:
            balance = 76000.0  # Default; overridden by IB query if connected
            if not self.config.dry_run and self.ib_conn.is_connected:
                ib_balance = await get_account_balance(self.ib_conn)
                if ib_balance:
                    balance = ib_balance
            self.daily_state = self.state_mgr.create_new(balance)
        else:
            # Reconcile with IB
            if not self.config.dry_run and self.ib_conn.is_connected:
                ib_orders = await get_ib_open_orders(self.ib_conn)
                ib_positions = await get_ib_positions(self.ib_conn)
                self.daily_state = self.state_mgr.reconcile_with_ib(
                    self.daily_state, ib_orders, ib_positions
                )

            # Restore active FVGs
            self.fvg_mgr.restore(self.daily_state.active_fvgs)

        # Restore partial fill timers from persisted state
        if self.order_mgr and self.daily_state:
            await self.order_mgr.restore_partial_fill_timers(self.daily_state)

        # Check kill switch from previous run
        if self.daily_state.kill_switch_active:
            self.logger.log(
                "kill_switch",
                action="already_active",
                reason=self.daily_state.kill_switch_reason,
            )
            self._shutdown = True
            return

        # 5b. Cross-validate clock with IB server time
        if not self.config.dry_run and self.ib_conn.is_connected:
            await self.clock.validate_with_ib(self.ib_conn)

        # 6. Subscribe to bar data
        if not self.config.dry_run and self.ib_conn.is_connected:
            await self._subscribe_bars()
            self._seed_recent_bars()
            self._subscribe_ticks()

            # 6b. Backfill FVGs from today's existing 5-min bars
            # On crash/restart mid-session, the bot needs to know about
            # FVGs that formed before it came back online.
            await self._backfill_fvgs()

            # 6c. Register reconnect handler (re-subscribe bars if IB drops)
            self.ib_conn.on_reconnect(self._on_reconnect)

        # 7. Schedule EOD actions
        self._schedule_eod()

        # 8. Schedule periodic tasks
        self._schedule_periodic()

        # 9. Telegram notification
        if self.telegram.enabled:
            await self.telegram.alert_bot_start(self.config, self.strategy.strategy_id)

        # Mark reconciliation complete — trading is now allowed
        self._reconciliation_complete = True

        # Process detection for backfilled FVGs that don't yet have orders
        # (must run AFTER _reconciliation_complete = True so time/risk gates allow it)
        await self._process_backfilled_fvgs()

        self.logger.log(
            "startup_complete",
            strategy=self.strategy.strategy_id,
            cells=self.strategy.cell_count,
            active_fvgs=self.fvg_mgr.active_count,
        )

    async def _event_loop(self):
        """Run until shutdown signal, session end, or kill switch.

        Uses ib.sleep() instead of asyncio.sleep() to ensure ib_async
        processes its internal events (bar updates, fill callbacks, etc.).
        """
        while not self._shutdown:
            if self.daily_state and self.daily_state.kill_switch_active:
                await self._trigger_kill_switch()
                break

            if not self.time_gates.is_session_active():
                # Before session start: wait for market open instead of exiting
                now_t = self.time_gates._now().time()
                if now_t < self.time_gates.session_start:
                    secs = self.time_gates.seconds_until(self.time_gates.session_start)
                    if secs > 0:
                        self.logger.log("waiting_for_session", seconds=round(secs))
                        await asyncio.sleep(min(secs, 30))  # Re-check every 30s max
                        continue
                # After session end: actually stop
                self.logger.log("bot_stop", reason="session_ended")
                break

            # Let asyncio process events (IB callbacks, timers, etc.)
            # Yield to event loop — ib_async processes bar updates and fill
            # callbacks during asyncio.sleep via nest_asyncio patching
            await asyncio.sleep(0.1)

    def _get_current_price(self) -> float:
        """Get the most recent trade price from bar data."""
        if self._bars_5min and len(self._bars_5min) > 0:
            return self._bars_5min[-1].close
        return 0.0

    async def _on_order_resolved(self):
        """Called when any order resolves (TP/SL/cancel). Re-evaluates suspended orders."""
        if not self.margin_priority or not self.daily_state or not self.daily_state.suspended_orders:
            return
        async with self._detection_lock:
            current_price = self._get_current_price()
            count = await self.margin_priority.try_reactivate_suspended(
                self.daily_state, current_price,
            )
            if count > 0:
                self.logger.log("suspended_reactivated", count=count)
                self.state_mgr.save(self.daily_state)

    async def _resolve_contract(self):
        """Resolve the NQ front-month futures contract."""
        from logic.utils.contract_utils import (
            generate_nq_expirations,
            get_contract_for_date,
        )
        from ib_async import Future

        now = self.clock.now()
        expirations = generate_nq_expirations(now.year - 1, now.year + 1)
        exp_date = get_contract_for_date(now, expirations, roll_days=8)

        # Try exact expiry date first, then month-only format (for newly listed contracts)
        # Suppress IB error 200 ("no security definition") during this probe
        ib = self.ib_conn.ib
        _suppress_200 = [True]
        def _quiet_error(reqId, errorCode, errorString, contract):
            if errorCode == 200 and _suppress_200[0]:
                return  # Expected during contract probing
        ib.errorEvent += _quiet_error

        for date_fmt in [exp_date.strftime("%Y%m%d"), exp_date.strftime("%Y%m")]:
            contract = Future(
                symbol=self.config.ticker,
                lastTradeDateOrContractMonth=date_fmt,
                exchange=self.config.exchange,
                currency=self.config.currency,
            )
            qualified = await ib.qualifyContractsAsync(contract)
            if qualified and contract.conId > 0:
                _suppress_200[0] = False
                ib.errorEvent -= _quiet_error
                self.logger.log(
                    "contract_resolved",
                    symbol=contract.symbol,
                    expiry=date_fmt,
                    conId=contract.conId,
                )
                return contract

        _suppress_200[0] = False
        ib.errorEvent -= _quiet_error
        raise RuntimeError(f"Failed to qualify NQ contract for expiry {exp_date}")

    async def _subscribe_bars(self):
        """Subscribe to 5min and 1min keepUpToDate historical bars."""
        ib = self.ib_conn.ib

        # 5-minute bars (use async version to avoid event loop conflict)
        self._bars_5min = await ib.reqHistoricalDataAsync(
            self._contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="5 mins",
            whatToShow="TRADES",
            useRTH=True,
            keepUpToDate=True,
        )
        self._bars_5min.updateEvent += self._on_5min_update

        # 1-minute bars
        self._bars_1min = await ib.reqHistoricalDataAsync(
            self._contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=True,
            keepUpToDate=True,
        )
        self._bars_1min.updateEvent += self._on_1min_update

        self.logger.log("bars_subscribed", timeframes=["5min", "1min"])

    def _seed_recent_bars(self):
        """Seed ActiveFVGManager._recent_bars from historical bars.

        At startup, _recent_bars is empty. keepUpToDate only fires for NEW bars.
        The tick path needs bar1/bar2 in _recent_bars immediately to detect FVGs
        on the very first bar completion. Without seeding, FVGs formed from the
        first few bars after startup would be missed.
        """
        if not self._bars_5min or not self.fvg_mgr:
            return

        # All bars except the last one (which is incomplete with keepUpToDate)
        completed = self._bars_5min[:-1] if len(self._bars_5min) > 1 else []

        # Seed the last N completed bars (deque maxlen=10 handles overflow)
        for bar in completed[-10:]:
            bar_dict = {
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "date": _bar_date_to_et(bar.date),
            }
            self.fvg_mgr.append_bar(bar_dict)

        self.logger.log("recent_bars_seeded", count=len(self.fvg_mgr._recent_bars))

    def _subscribe_ticks(self):
        """Subscribe to tick-by-tick trade data for sub-ms FVG detection.

        Optional enhancement — if subscription fails, keepUpToDate bars
        still handle detection with ~5 sec delay.
        """
        try:
            ib = self.ib_conn.ib
            self._tick_subscription = ib.reqTickByTickData(
                self._contract, tickType='AllLast',
                numberOfTicks=0, ignoreSize=True,
            )
            self._tick_subscription.updateEvent += self._on_tick_update
            self._tick_bar_builder.reset()
            self.logger.log("ticks_subscribed")
        except Exception as e:
            self.logger.log("ticks_subscribe_failed", error=str(e))
            self._tick_subscription = None

    def _on_tick_update(self, ticker):
        """Callback for tick-by-tick trade data. Hot path — must be fast."""
        ticks = ticker.tickByTicks
        if not ticks:
            return

        for tick in ticks:
            tick_time_et = _tick_time_to_et(tick.time)
            if tick_time_et is None:
                continue

            # Tick-based mitigation: check every trade against active FVGs
            self._check_tick_mitigation(tick.price, tick_time_et)

            completed = self._tick_bar_builder.on_tick(tick.price, tick_time_et)
            if completed is not None:
                self._on_tick_bar_complete(completed)

        # Clear processed ticks to prevent unbounded memory growth
        ticks.clear()

    def _check_tick_mitigation(self, price, tick_time_et):
        """Check a single trade tick against active FVGs for mitigation."""
        if self.fvg_mgr is None or self.fvg_mgr.active_count == 0:
            return

        mit_time = str(tick_time_et)
        mitigated = []

        for fvg in self.fvg_mgr.active_fvgs:
            if fvg.is_mitigated:
                continue
            if fvg.zone_low <= price <= fvg.zone_high:
                fvg.is_mitigated = True
                fvg.mitigation_time = mit_time
                mitigated.append(fvg)

        for fvg in mitigated:
            self.logger.log(
                "mitigation",
                fvg_id=fvg.fvg_id,
                type=fvg.fvg_type,
                zone=[fvg.zone_low, fvg.zone_high],
                tick_price=price,
                source="tick",
            )
            self.db.update_fvg(fvg.fvg_id, mitigated=1, mitigation_time=mit_time)
            self.fvg_mgr.remove(fvg.fvg_id)

        if mitigated:
            self.daily_state.active_fvgs = self.fvg_mgr.active_fvgs
            self.state_mgr.save(self.daily_state)

    def _on_tick_bar_complete(self, tick_ohlc):
        """Merge tick OHLC with IB's keepUpToDate bar for conservative FVG detection.

        ALWAYS appends bar3 to _recent_bars so subsequent tick detections
        have correct bar1/bar2 — cannot rely on keepUpToDate if it's broken.
        """
        if self.fvg_mgr is None:
            return

        # Read IB's keepUpToDate incomplete bar (continuously updated ~every 5 sec)
        ib_bar = self._bars_5min[-1] if self._bars_5min and len(self._bars_5min) > 0 else None

        if ib_bar is not None:
            # Hybrid merge: most conservative extremes from both sources
            bar3 = {
                "open":  ib_bar.open,
                "high":  max(ib_bar.high, tick_ohlc["high"]),
                "low":   min(ib_bar.low, tick_ohlc["low"]),
                "close": tick_ohlc["close"],
                "date":  _bar_date_to_et(ib_bar.date),
            }
        else:
            bar3 = tick_ohlc

        # Always append bar3 to _recent_bars so next tick detection has correct bar1/bar2.
        # detect_from_tick_bar uses [-2] and [-1] BEFORE this append.
        if len(self.fvg_mgr._recent_bars) >= 2:
            fvg = self.fvg_mgr.detect_from_tick_bar(bar3)
        else:
            fvg = None

        # Append AFTER detection (so [-2] and [-1] were the correct bar1/bar2)
        self.fvg_mgr.append_bar(bar3)
        self._tick_detected_bars.add(tick_ohlc["date"])

        if fvg:
            self.daily_state.active_fvgs = self.fvg_mgr.active_fvgs
            self.state_mgr.save(self.daily_state)
            self.db.insert_fvg(
                fvg_id=fvg.fvg_id, trade_date=fvg.formation_date,
                fvg_type=fvg.fvg_type, zone_low=fvg.zone_low,
                zone_high=fvg.zone_high, fvg_size=round(fvg.zone_high - fvg.zone_low, 2),
                time_period=fvg.time_period, formation_time=fvg.time_candle3,
            )
            self.logger.log(
                "fvg_detected_tick", fvg_id=fvg.fvg_id, type=fvg.fvg_type,
                zone=[fvg.zone_low, fvg.zone_high], source="tick",
            )
            asyncio.ensure_future(self._guarded_process_detection(fvg))

    async def _backfill_fvgs(self):
        """
        Scan today's existing 5-min bars for un-mitigated FVGs.

        On crash/restart mid-session, the bot loses its active FVG list.
        This replays the already-loaded 5-min bars (from the keepUpToDate
        subscription which includes today's history) to rebuild the list.
        Then checks 1-min bars to see if any were already mitigated.
        """
        if self._bars_5min is None or len(self._bars_5min) < 3:
            return

        from bot.strategy.fvg_detector import check_fvg_3bars, _assign_time_period, SESSION_INTERVALS
        from bot.strategy.trade_calculator import round_to_tick

        # Build set of already-traded zones to prevent duplicate entries on restart
        _traded_zones = set()
        for og in (self.daily_state.pending_orders
                   + self.daily_state.open_positions
                   + self.daily_state.closed_trades):
            _traded_zones.add((round(og.entry_price, 2), round(og.stop_price, 2)))

        # Replay all completed 5-min bars to detect FVGs
        detected = 0
        for i in range(2, len(self._bars_5min)):
            bar1 = self._bars_5min[i - 2]
            bar2 = self._bars_5min[i - 1]
            bar3 = self._bars_5min[i]

            b1 = {"open": bar1.open, "high": bar1.high, "low": bar1.low, "close": bar1.close, "date": _bar_date_to_et(bar1.date)}
            b2 = {"open": bar2.open, "high": bar2.high, "low": bar2.low, "close": bar2.close, "date": _bar_date_to_et(bar2.date)}
            b3 = {"open": bar3.open, "high": bar3.high, "low": bar3.low, "close": bar3.close, "date": _bar_date_to_et(bar3.date)}

            fvg = check_fvg_3bars(b1, b2, b3, self.config.min_fvg_size)
            if fvg is None:
                continue

            fvg.time_period = _assign_time_period(b3["date"], SESSION_INTERVALS)
            if not fvg.time_period:
                continue

            fvg.formation_date = str(b3["date"].date()) if hasattr(b3["date"], "date") else ""

            # Check if any strategy cell matches this time period
            has_cell = any(
                tp == fvg.time_period for tp, _ in self.strategy._lookup.keys()
            )
            if not has_cell:
                continue

            # Check if already mitigated by scanning 1-min bars after formation
            # FVG is only confirmed when bar3 CLOSES (5 min after bar3 open).
            # Mitigation scan must start after bar3 close, not bar3 open.
            formation_time = b3["date"] + timedelta(minutes=5)
            already_mitigated = False
            if self._bars_1min:
                for bar_1m in self._bars_1min:
                    bar_1m_et = _bar_date_to_et(bar_1m.date)
                    if bar_1m_et <= formation_time:
                        continue
                    if bar_1m.low <= fvg.zone_high and bar_1m.high >= fvg.zone_low:
                        already_mitigated = True
                        break

            if not already_mitigated:
                # Check if this zone was already traded (prevents duplicate on restart)
                mit_entry = round_to_tick(fvg.zone_high if fvg.fvg_type == "bullish" else fvg.zone_low)
                mid_entry = round_to_tick((fvg.zone_high + fvg.zone_low) / 2)
                stop = round_to_tick(fvg.middle_low if fvg.fvg_type == "bullish" else fvg.middle_high)
                if (round(mit_entry, 2), round(stop, 2)) in _traded_zones or \
                   (round(mid_entry, 2), round(stop, 2)) in _traded_zones:
                    continue  # Already traded this zone

                self.fvg_mgr._active[fvg.fvg_id] = fvg
                detected += 1

        if detected > 0:
            self.daily_state.active_fvgs = self.fvg_mgr.active_fvgs
            self.state_mgr.save(self.daily_state)

        self.logger.log(
            "fvg_backfill",
            bars_scanned=len(self._bars_5min),
            fvgs_found=detected,
            total_active=self.fvg_mgr.active_count,
        )

    async def _process_backfilled_fvgs(self):
        """Place orders for backfilled FVGs that don't already have pending/open orders."""
        if self.fvg_mgr.active_count == 0:
            self.logger.log("backfill_process", action="skip", reason="no_active_fvgs")
            return

        # Build set of fvg_ids that already have ACTIVE or SUCCESSFULLY FILLED orders.
        # Cancelled/rejected orders don't count — the zone was never actually traded.
        _skip_reasons = {"CANCEL", "RECONCILE_ORPHAN", "REJECTED"}
        ordered_fvg_ids = set()
        for og in self.daily_state.pending_orders + self.daily_state.open_positions:
            ordered_fvg_ids.add(og.fvg_id)
        for og in self.daily_state.closed_trades:
            if og.close_reason not in _skip_reasons:
                ordered_fvg_ids.add(og.fvg_id)

        # Process detection for FVGs without orders (copy list — detection may modify active)
        processed = 0
        skipped = 0
        for fvg in list(self.fvg_mgr.active_fvgs):
            if fvg.fvg_id not in ordered_fvg_ids:
                self.logger.log(
                    "backfill_process", action="evaluating",
                    fvg_id=fvg.fvg_id, type=fvg.fvg_type,
                    zone=[fvg.zone_low, fvg.zone_high],
                    time_period=fvg.time_period,
                )
                await self._guarded_process_detection(fvg)
                processed += 1
            else:
                skipped += 1

        self.logger.log(
            "backfill_process", action="done",
            processed=processed, skipped_ordered=skipped,
        )

    async def _on_reconnect(self):
        """Re-subscribe bars/ticks and backfill FVGs after IB reconnect."""
        self.logger.log("reconnect_resubscribe")
        try:
            await self._subscribe_bars()
            self._seed_recent_bars()
            self._subscribe_ticks()
            self._tick_detected_bars.clear()
            await self._backfill_fvgs()
            await self._process_backfilled_fvgs()
            self.logger.log("reconnect_resubscribe_done",
                            active_fvgs=self.fvg_mgr.active_count)
        except Exception as e:
            self.logger.log("reconnect_resubscribe_error", error=str(e))

    def _on_5min_update(self, bars, hasNewBar):
        """Callback when 5min bar data updates.

        With keepUpToDate=True, hasNewBar=True means a new bar STARTED.
        bars[-1] is the new incomplete bar, bars[-2] is the just-completed bar.
        We detect FVGs on the COMPLETED bar (bars[-2]), never on the incomplete one.

        If the tick path already detected an FVG for this bar window,
        we still append the authoritative bar to _recent_bars but skip re-detection.
        """
        if not hasNewBar or len(bars) < 2:
            return

        # Use the just-completed bar, NOT the new incomplete one
        bar = bars[-2]
        bar_dict = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "date": _bar_date_to_et(bar.date),
        }

        # Check if tick path already handled this bar window
        bar_window = _bar_date_to_et(bar.date)
        if bar_window in self._tick_detected_bars:
            # Tick path already appended bar3 and ran detection.
            # Replace the tick-built bar with IB's authoritative version
            # so future bar1/bar2 references are exact.
            if len(self.fvg_mgr._recent_bars) > 0:
                self.fvg_mgr._recent_bars[-1] = bar_dict
            self._tick_detected_bars.discard(bar_window)
            return

        # Normal path: detect FVG from keepUpToDate (fallback for non-tick path)
        fvg = self.fvg_mgr.on_5min_bar(bar_dict)
        if fvg:
            self.daily_state.active_fvgs = self.fvg_mgr.active_fvgs
            self.state_mgr.save(self.daily_state)
            self.db.insert_fvg(
                fvg_id=fvg.fvg_id, trade_date=fvg.formation_date,
                fvg_type=fvg.fvg_type, zone_low=fvg.zone_low,
                zone_high=fvg.zone_high, fvg_size=round(fvg.zone_high - fvg.zone_low, 2),
                time_period=fvg.time_period, formation_time=fvg.time_candle3,
            )

            # Evaluate strategy and place limit order immediately at detection time
            # (IB fills the limit order when price reaches the zone — that IS the mitigation)
            asyncio.ensure_future(self._guarded_process_detection(fvg))

    def _on_1min_update(self, bars, hasNewBar):
        """Callback when 1min bar data updates.

        Same as 5min: bars[-1] is incomplete, bars[-2] is just-completed.
        Mitigation scanning uses the completed bar.
        """
        if not hasNewBar or len(bars) < 2:
            return

        bar = bars[-2]
        bar_dict = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "date": _bar_date_to_et(bar.date),
        }

        # Scan for mitigations
        mitigated = scan_active_fvgs(bar_dict, self.fvg_mgr.active_fvgs)

        for fvg, mit_time in mitigated:
            self.logger.log(
                "mitigation",
                fvg_id=fvg.fvg_id,
                type=fvg.fvg_type,
                zone=[fvg.zone_low, fvg.zone_high],
                candle_low=bar_dict["low"],
                candle_high=bar_dict["high"],
            )
            self.db.update_fvg(fvg.fvg_id, mitigated=1, mitigation_time=mit_time)

            # Remove from active list (order was already placed at detection time)
            self.fvg_mgr.remove(fvg.fvg_id)

        if mitigated:
            self.daily_state.active_fvgs = self.fvg_mgr.active_fvgs
            self.state_mgr.save(self.daily_state)

    async def _guarded_process_detection(self, fvg):
        """Serialize detection-time order processing to prevent concurrent position limit bypass.

        Time gate is checked BEFORE acquiring the lock (pure clock check, no shared state).
        Risk gates + order placement happen INSIDE the lock to ensure check-then-act atomicity.
        """
        if not self._reconciliation_complete:
            self.logger.log("setup_rejected", fvg_id=fvg.fvg_id,
                            gate="startup", reason="reconciliation not complete")
            return

        # Time gate check outside lock — pure clock check, no DailyState reads
        allowed, reason = self.time_gates.can_enter()
        if not allowed:
            self.logger.log("setup_rejected", fvg_id=fvg.fvg_id, gate="time", reason=reason)
            # Remove from active FVGs — past entry window, won't trade
            self.fvg_mgr.remove(fvg.fvg_id)
            self.daily_state.active_fvgs = self.fvg_mgr.active_fvgs
            return

        async with self._detection_lock:
            await self._process_detection(fvg)

    async def _process_detection(self, fvg):
        """
        Evaluate and potentially place a limit bracket order for a newly detected FVG.
        Called with _detection_lock held — risk gate reads and order placement are atomic.

        The entry limit order sits with IB; it fills when price reaches the zone level
        (which is equivalent to mitigation).
        """
        from bot.strategy.trade_calculator import risk_to_range

        # Try each setup type: the strategy determines which setup applies
        # First compute risk for both setups and check which has a strategy cell
        for setup_type in ["mit_extreme", "mid_extreme"]:
            # Compute a provisional entry to determine risk
            if setup_type == "mit_extreme":
                entry = fvg.zone_high if fvg.fvg_type == "bullish" else fvg.zone_low
            else:
                entry = (fvg.zone_high + fvg.zone_low) / 2

            stop = fvg.middle_low if fvg.fvg_type == "bullish" else fvg.middle_high
            risk_pts = round(abs(entry - stop) * 4) / 4  # round to tick

            if risk_pts <= 0:
                self.logger.log(
                    "setup_skipped_strategy",
                    fvg_id=fvg.fvg_id,
                    strategy_id=self.strategy.strategy_id,
                    time_period=fvg.time_period,
                    setup=setup_type,
                    risk_pts=risk_pts,
                    reason="non_positive_risk",
                )
                continue

            # Look up strategy cell
            risk_range = risk_to_range(risk_pts)
            if not risk_range:
                self.logger.log(
                    "setup_skipped_strategy",
                    fvg_id=fvg.fvg_id,
                    strategy_id=self.strategy.strategy_id,
                    time_period=fvg.time_period,
                    setup=setup_type,
                    risk_pts=risk_pts,
                    reason="risk_out_of_strategy_bins",
                )
                continue

            cell = self.strategy.find_cell(fvg.time_period, risk_pts)
            if cell is None:
                self.logger.log(
                    "setup_skipped_strategy",
                    fvg_id=fvg.fvg_id,
                    strategy_id=self.strategy.strategy_id,
                    time_period=fvg.time_period,
                    setup=setup_type,
                    risk_pts=risk_pts,
                    risk_range=risk_range,
                    reason="no_strategy_cell",
                )
                continue

            # Only proceed if cell matches this setup type
            if cell["setup"] != setup_type:
                self.logger.log(
                    "setup_skipped_strategy",
                    fvg_id=fvg.fvg_id,
                    strategy_id=self.strategy.strategy_id,
                    time_period=fvg.time_period,
                    setup=setup_type,
                    matched_setup=cell["setup"],
                    risk_pts=risk_pts,
                    risk_range=risk_range,
                    reason="cell_setup_mismatch",
                )
                continue

            # Calculate full setup
            balance = self.daily_state.start_balance + self.daily_state.realized_pnl
            order = calculate_setup(fvg, cell, balance, self.config.risk_per_trade, config=self.config)
            if order is None:
                self.logger.log(
                    "setup_rejected", fvg_id=fvg.fvg_id,
                    gate="calculation", reason="invalid setup"
                )
                continue

            # Check risk gates
            result = self.risk_gates.check_all(self.daily_state, order)
            if not result.passed:
                self.logger.log(
                    "setup_rejected",
                    fvg_id=fvg.fvg_id,
                    setup=setup_type,
                    gate=result.gate,
                    reason=result.reason,
                )
                continue

            # Drawdown scaling — reduce size on bad days, never block
            dd_mult = self.risk_gates.drawdown_multiplier(self.daily_state)
            if dd_mult < 1.0:
                scaled_qty = max(1, int(order.target_qty * dd_mult))
                self.logger.log(
                    "dd_scale",
                    group_id=order.group_id,
                    original_qty=order.target_qty,
                    scaled_qty=scaled_qty,
                    multiplier=dd_mult,
                    daily_pnl=round(self.daily_state.realized_pnl, 2),
                )
                order.target_qty = scaled_qty

            # Place the bracket order
            risk_dollars = round(order.risk_pts * order.target_qty * 20, 2)
            reward_dollars = round(order.risk_pts * order.n_value * order.target_qty * 20, 2)
            self.logger.log(
                "setup_accepted",
                fvg_id=fvg.fvg_id,
                fvg_type=fvg.fvg_type,
                time_period=fvg.time_period,
                setup=setup_type,
                side=order.side,
                entry=order.entry_price,
                stop=order.stop_price,
                target=order.target_price,
                risk_pts=order.risk_pts,
                risk_range=risk_range,
                n=order.n_value,
                qty=order.target_qty,
                risk_dollars=risk_dollars,
                reward_dollars=reward_dollars,
                balance=round(balance, 2),
                cell_ev=cell["ev"],
                cell_wr=cell["win_rate"],
            )

            if self.order_mgr and (self.config.dry_run or self.ib_conn.is_connected):
                if self.margin_priority and not self.config.dry_run:
                    current_price = self._get_current_price()
                    margin_result = await self.margin_priority.evaluate_and_place(
                        order, fvg, self.daily_state, current_price,
                    )
                    if margin_result == "SUSPENDED":
                        self.logger.log(
                            "setup_suspended_margin",
                            group_id=order.group_id,
                            fvg_id=fvg.fvg_id,
                            entry=order.entry_price,
                            reason=order.suspend_reason,
                        )
                        # Still save state and link FVG (done below), but skip DB insert
                        fvg.orders_placed.append(order.group_id)
                        self.state_mgr.save(self.daily_state)
                        return
                else:
                    await self.order_mgr.place_bracket(order, self.daily_state)
            elif self.order_mgr and not self.ib_conn.is_connected:
                self.logger.log(
                    "setup_rejected", fvg_id=fvg.fvg_id,
                    gate="connection", reason="IB not connected",
                )
                continue
            else:
                # Dry run without IB connection
                order.state = "SUBMITTED"
                order.submitted_at = self.clock.now().isoformat()
                self.daily_state.pending_orders.append(order)
                self.daily_state.trade_count += 1
                self.logger.log(
                    "order_placed",
                    mode="DRY_RUN",
                    group_id=order.group_id,
                    setup=order.setup,
                    side=order.side,
                    entry=order.entry_price,
                    stop=order.stop_price,
                    target=order.target_price,
                    risk_pts=order.risk_pts,
                    n=order.n_value,
                    contracts=order.target_qty,
                )

            # Write trade to DB
            self.db.insert_trade(
                group_id=order.group_id, fvg_id=fvg.fvg_id,
                trade_date=fvg.formation_date, fvg_type=fvg.fvg_type,
                zone_low=fvg.zone_low, zone_high=fvg.zone_high,
                fvg_size=round(fvg.zone_high - fvg.zone_low, 2),
                time_period=fvg.time_period, risk_range=risk_range,
                setup=setup_type, n_value=order.n_value,
                cell_ev=cell["ev"], cell_win_rate=cell["win_rate"],
                side=order.side, contracts=order.target_qty,
                entry_price=order.entry_price, stop_price=order.stop_price,
                target_price=order.target_price, risk_pts=order.risk_pts,
                entry_time=order.submitted_at,
                balance_before=round(balance, 2),
                strategy_id=self.strategy.strategy_id,
                mode="PAPER" if self.config.paper_mode else "LIVE",
            )
            self.db.update_fvg(fvg.fvg_id, trade_placed=1)

            # Link order to FVG for expiration cancellation
            fvg.orders_placed.append(order.group_id)
            self.state_mgr.save(self.daily_state)
            return  # Only one setup per FVG

        # No strategy cell matched — remove FVG from active tracking
        self.fvg_mgr.remove(fvg.fvg_id)
        self.daily_state.active_fvgs = self.fvg_mgr.active_fvgs

    async def _trigger_kill_switch(self):
        """Execute kill switch: cancel all, flatten all, halt."""
        self.logger.log(
            "kill_switch",
            action="triggered",
            reason=self.daily_state.kill_switch_reason,
            pnl=self.daily_state.realized_pnl,
        )

        if self.order_mgr:
            await self.order_mgr.flatten_all(self.daily_state, "KILL_SWITCH")

        self.state_mgr.save(self.daily_state, force=True)

        if self.telegram.enabled:
            await self.telegram.alert_kill_switch(
                self.daily_state.kill_switch_reason,
                self.daily_state.realized_pnl,
                self.daily_state.start_balance + self.daily_state.realized_pnl,
            )

        self._shutdown = True

    def _schedule_eod(self):
        """Schedule end-of-day actions."""
        loop = asyncio.get_event_loop()
        schedule = self.time_gates.get_eod_schedule()

        for delay, action in schedule:
            if action == "cancel_unfilled":
                handle = loop.call_later(delay, lambda: asyncio.ensure_future(self._eod_cancel()))
            elif action == "flatten_all":
                handle = loop.call_later(delay, lambda: asyncio.ensure_future(self._eod_flatten()))
            elif action == "session_end":
                handle = loop.call_later(delay, lambda: asyncio.ensure_future(self._eod_cleanup()))
            else:
                continue
            self._eod_tasks.append(handle)

        self.logger.log("eod_scheduled", actions=len(schedule))

    def _schedule_periodic(self):
        """Schedule periodic tasks (state save, strategy reload)."""
        loop = asyncio.get_event_loop()

        async def periodic_save():
            while not self._shutdown:
                await asyncio.sleep(self.config.state_save_interval)
                if self.daily_state:
                    self.state_mgr.save_if_dirty(self.daily_state)
                    self.state_mgr.save(self.daily_state)

        async def periodic_strategy_check():
            while not self._shutdown:
                await asyncio.sleep(self.config.strategy_reload_interval)
                self.strategy.check_reload()

        async def periodic_clock_sync():
            while not self._shutdown:
                await asyncio.sleep(300)  # Re-sync NTP every 5 minutes
                self.clock.check_resync()

        async def periodic_alert_retry():
            while not self._shutdown:
                await asyncio.sleep(60)  # Retry unsent alerts every 60 seconds
                try:
                    sent = await self.telegram.retry_unsent()
                    if sent > 0:
                        self.logger.log("alerts_retried", count=sent)
                except Exception as e:
                    self.logger.log("alert_retry_error", error=str(e))

        async def periodic_margin_refresh():
            while not self._shutdown:
                await asyncio.sleep(self.config.margin_refresh_interval)
                if self.margin_tracker:
                    await self.margin_tracker.refresh_if_stale()

        self._periodic_tasks.append(asyncio.ensure_future(periodic_save()))
        self._periodic_tasks.append(asyncio.ensure_future(periodic_strategy_check()))
        self._periodic_tasks.append(asyncio.ensure_future(periodic_clock_sync()))
        self._periodic_tasks.append(asyncio.ensure_future(periodic_alert_retry()))
        if self.margin_tracker:
            self._periodic_tasks.append(asyncio.ensure_future(periodic_margin_refresh()))

    async def _eod_cancel(self):
        """Cancel all unfilled entry orders at configured EOD cancel time."""
        self.logger.log("eod_cancel", time=self.config.cancel_unfilled_time)
        if self.order_mgr:
            await self.order_mgr.cancel_all_pending(self.daily_state, "EOD")
        # Clear all suspended orders at EOD
        if self.margin_priority:
            count = await self.margin_priority.clear_all_suspended(self.daily_state, "EOD")
            if count > 0:
                self.logger.log("eod_suspended_cleared", count=count)
        self.state_mgr.save(self.daily_state, force=True)

    async def _eod_flatten(self):
        """Flatten all open positions at configured EOD flatten time."""
        self.logger.log("eod_flatten", time=self.config.flatten_time)
        if self.order_mgr:
            await self.order_mgr.flatten_all(self.daily_state, "EOD")
        self.state_mgr.save(self.daily_state, force=True)

    def _calculate_max_drawdown(self, start_balance, closed_trades):
        """Calculate max intraday drawdown from the trade sequence.

        Walks trades in chronological order tracking peak equity.
        Drawdown = peak - trough (always positive, represents max loss from peak).
        """
        running = start_balance
        peak = start_balance
        max_dd = 0.0

        for trade in sorted(closed_trades, key=lambda t: t.closed_at or ""):
            running += trade.realized_pnl
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        return round(max_dd, 2)

    async def _eod_cleanup(self):
        """16:00 — Expire FVGs, daily summary, final state save."""
        expired = self.fvg_mgr.expire_all()
        self.daily_state.active_fvgs = []
        self._tick_detected_bars.clear()

        # Compute comprehensive daily stats
        closed = self.daily_state.closed_trades
        wins = [t for t in closed if t.close_reason == "TP"]
        losses = [t for t in closed if t.close_reason == "SL"]
        eod_exits = [t for t in closed if t.close_reason in ("EOD", "FLATTEN")]

        gross_profit = sum(t.realized_pnl for t in closed if t.realized_pnl > 0)
        gross_loss = sum(t.realized_pnl for t in closed if t.realized_pnl < 0)
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

        # Per-cell breakdown
        cell_pnl = {}
        for t in closed:
            key = f"{t.setup}"
            cell_pnl[key] = cell_pnl.get(key, 0) + t.realized_pnl

        # Total commissions (estimated if not tracked: $4.50/contract round-trip)
        total_contracts = sum(t.filled_qty or t.target_qty for t in closed)

        self.logger.log(
            "daily_summary",
            date=self.daily_state.date,
            start_balance=self.daily_state.start_balance,
            end_balance=self.daily_state.start_balance + self.daily_state.realized_pnl,
            total_trades=len(closed),
            wins=len(wins),
            losses=len(losses),
            eod_exits=len(eod_exits),
            win_rate=round(len(wins) / len(closed) * 100, 1) if closed else 0,
            gross_profit=round(gross_profit, 2),
            gross_loss=round(gross_loss, 2),
            net_pnl=round(self.daily_state.realized_pnl, 2),
            pnl_pct=round(self.daily_state.daily_pnl_pct * 100, 2),
            profit_factor=round(profit_factor, 2) if profit_factor != float('inf') else "inf",
            avg_win=round(gross_profit / len(wins), 2) if wins else 0,
            avg_loss=round(gross_loss / len(losses), 2) if losses else 0,
            total_contracts=total_contracts,
            fvgs_detected=self.daily_state.trade_count,  # approximation
            fvgs_expired=len(expired),
            setup_pnl=cell_pnl,
            kill_switch=self.daily_state.kill_switch_active,
        )

        self.state_mgr.save(self.daily_state, force=True)

        # Calculate intraday max drawdown
        max_dd = self._calculate_max_drawdown(self.daily_state.start_balance, closed)

        # Save daily stats to DB
        pf = round(profit_factor, 2) if profit_factor != float('inf') else None
        self.db.insert_daily_stats(
            trade_date=self.daily_state.date,
            start_balance=self.daily_state.start_balance,
            end_balance=self.daily_state.start_balance + self.daily_state.realized_pnl,
            net_pnl=round(self.daily_state.realized_pnl, 2),
            pnl_pct=round(self.daily_state.daily_pnl_pct * 100, 2),
            total_trades=len(closed),
            wins=len(wins),
            losses=len(losses),
            eod_exits=len(eod_exits),
            win_rate=round(len(wins) / len(closed) * 100, 1) if closed else 0,
            gross_profit=round(gross_profit, 2),
            gross_loss=round(gross_loss, 2),
            profit_factor=pf,
            avg_win=round(gross_profit / len(wins), 2) if wins else 0,
            avg_loss=round(gross_loss / len(losses), 2) if losses else 0,
            total_contracts=total_contracts,
            max_drawdown=max_dd,
            kill_switch_hit=1 if self.daily_state.kill_switch_active else 0,
            strategy_id=self.strategy.strategy_id,
        )

        if self.telegram.enabled:
            await self.telegram.alert_daily_summary(self.daily_state)

        self._shutdown = True

    async def _shutdown_gracefully(self):
        """Clean shutdown: cancel tasks, save state, disconnect."""
        # Cancel periodic tasks
        for task in self._periodic_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Cancel EOD tasks
        for handle in self._eod_tasks:
            handle.cancel()

        # Cancel tick subscription
        if self._tick_subscription is not None and self.ib_conn.is_connected:
            try:
                self.ib_conn.ib.cancelTickByTickData(self._tick_subscription)
            except Exception:
                pass

        # Save final state
        if self.daily_state:
            self.state_mgr.save(self.daily_state, force=True)

        # Disconnect from IB
        if self.ib_conn.is_connected:
            await self.ib_conn.disconnect()

        self.logger.log("bot_stop", reason="shutdown")
        self.logger.close()
