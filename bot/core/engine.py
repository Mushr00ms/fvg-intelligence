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
from bot.state.trade_state import DailyState, CLOSE_CANCEL, CLOSE_EOD, CLOSE_REJECTED
from bot.strategy.strategy_loader import StrategyLoader
from bot.strategy.fvg_detector import ActiveFVGManager
from bot.strategy.mitigation_scanner import scan_active_fvgs
from bot.strategy.tick_imbalance_accumulator import TickImbalanceAccumulator
from bot.strategy.trade_calculator import calculate_setup
from bot.risk.risk_gates import RiskGates
from bot.risk.time_gates import TimeGates
from bot.execution.ib_connection import IBConnection
from bot.execution.order_manager import OrderManager
from bot.execution.position_tracker import get_account_balance, get_ib_open_orders, get_ib_positions
from bot.alerts.telegram import TelegramAlerter
from bot.db import TradeDB
from bot.risk.calendar_gates import WitchingGateConfig, is_blocked_by_witching_gate
from bot.backtest.us_holidays import is_trading_day, US_MARKET_HOLIDAYS

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
        if not self.clock.is_trusted:
            # Logger not yet initialized — print to stderr as last resort
            import sys
            print("CRITICAL: NTP sync failed on startup — clock untrusted. "
                  "New trade entries will be BLOCKED until NTP succeeds.", file=sys.stderr)

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
        self._tick_subscription = None
        self.hfoiv_gate = None              # Initialized in _startup if enabled
        self._imbalance_accumulator = None  # TickImbalanceAccumulator for live HFOIV
        self._shutdown = False
        self._reconciliation_complete = False  # Block trading until startup finishes
        self._detection_lock = asyncio.Lock()
        self._eod_tasks = []
        self._periodic_tasks = []
        self._eod_cancel_done = False
        self._eod_flatten_done = False
        self._eod_cleanup_done = False
        self._session_seeded = False  # True after bars re-seeded at session open

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
        # 0. Calendar gate — skip non-trading days (mirrors backtester behavior)
        today_str = self.clock.now().strftime("%Y%m%d")
        if not self.config.test_connection and not is_trading_day(today_str):
            reason = US_MARKET_HOLIDAYS.get(today_str, "weekend")
            self.logger.log("market_closed", date=today_str, reason=reason)
            self._shutdown = True
            return

        # 0a. Test-connection mode — connect, print account info, exit
        if self.config.test_connection:
            await self._run_connection_test()
            self._shutdown = True
            return

        if self.config.execution_backend != "ib":
            raise NotImplementedError(
                "execution_backend='binance_um' is scaffolded at the client/adapter layer "
                "but not yet wired into BotEngine"
            )

        self.logger.log("bot_start", mode="PAPER" if self.config.paper_mode else "LIVE",
                        dry_run=self.config.dry_run)

        # 1. Load strategy
        self.strategy.load()
        self.fvg_mgr = ActiveFVGManager(self.strategy, self.config.min_fvg_size, self.logger)

        # 1b. Initialize HFOIV gate (optional — config in strategy meta)
        strategy_meta = self.strategy.strategy.get("meta", {}) if self.strategy.strategy else {}
        hfoiv_cfg = strategy_meta.get("hfoiv_gate", {})
        if hfoiv_cfg.get("enabled", False):
            from bot.risk.hfoiv_gate import HFOIVGate, HFOIVConfig
            self.hfoiv_gate = HFOIVGate(HFOIVConfig(
                enabled=True,
                rolling_bars=hfoiv_cfg.get("rolling_bars", 6),
                lookback_sessions=hfoiv_cfg.get("lookback_sessions", 90),
                bucket_minutes=hfoiv_cfg.get("bucket_minutes", 30),
                thresholds=[tuple(t) for t in hfoiv_cfg.get("thresholds", [(70, 0.25)])],
            ))
            self._imbalance_accumulator = TickImbalanceAccumulator()
            self.logger.log("hfoiv_gate_init",
                            config_label=hfoiv_cfg.get("config_label", ""),
                            rolling=self.hfoiv_gate.config.rolling_bars,
                            lookback=self.hfoiv_gate.config.lookback_sessions)

        # 1c. Witching gate — skip day if strategy hard_gates flag it (mirrors backtester)
        allowed, cal_reason = self._calendar_gate_allows_entry()
        if not allowed:
            self.logger.log("market_closed", date=today_str, reason=cal_reason)
            self._shutdown = True
            return

        # 2. Connect to IB
        if not self.config.dry_run:
            await self.ib_conn.connect()
            self._register_ib_error_handler()

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

                # Reseed margin tracker for orders that survived reconciliation.
                # _reserved_margin starts at 0 on every boot; without this, the
                # local "release on suspend" path is a no-op for any order
                # restored from disk, and a mid-session restart undercounts
                # reserved margin (oversizing the next entry).
                #
                # Reseed ONLY the unfilled remainder of pending/PARTIAL orders.
                # Open (fully filled) positions consume real IB margin already
                # visible in AvailableFunds — re-reserving them locally would
                # double-count and starve the next entry's sizing.
                if self.margin_tracker:
                    reseeded_qty = 0
                    for og in self.daily_state.pending_orders:
                        qty = og.target_qty - (og.filled_qty or 0)
                        if qty > 0:
                            self.margin_tracker.reserve(qty)
                            reseeded_qty += qty
                    if reseeded_qty > 0:
                        self.logger.log(
                            "margin_reseeded_after_reconcile",
                            total_qty=reseeded_qty,
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

            # 6a-ii. Pre-populate HFOIV normalization history from stored parquets
            if self.hfoiv_gate is not None:
                self._load_hfoiv_history()

            # 6b. Backfill FVGs from today's existing 5-min bars
            # On crash/restart mid-session, the bot needs to know about
            # FVGs that formed before it came back online.
            await self._backfill_fvgs()

            # If session is already active (mid-day launch), mark seeded so
            # the event loop doesn't redundantly re-seed at the top.
            if self.time_gates.is_session_active():
                self._session_seeded = True

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

        # Kick the suspended-order queue. On a mid-session restart, suspended
        # orders persisted from the previous run would otherwise sit forever
        # because try_reactivate_suspended is only called on order resolution.
        if self.daily_state and self.daily_state.suspended_orders and self.margin_priority:
            await self._on_order_resolved(released_qty=0)

        self.logger.log(
            "startup_complete",
            strategy=self.strategy.strategy_id,
            cells=self.strategy.cell_count,
            active_fvgs=self.fvg_mgr.active_count,
        )

    async def _event_loop(self):
        """Run until shutdown signal, session end, or kill switch.

        EOD actions (cancel/flatten/cleanup) are triggered by NTP-corrected
        wall clock checks every iteration, NOT by call_later — which uses
        the monotonic clock and drifts on WSL2 over a full trading day.

        IMPORTANT: _check_eod_actions() runs BEFORE the session-active check
        so that the 16:00 cleanup (including EOD reconciliation) fires before
        the loop exits on session_end.
        """
        self._disconnect_flatten_done = False
        while not self._shutdown:
            if self.daily_state and self.daily_state.kill_switch_active:
                await self._trigger_kill_switch()
                break

            # EOD actions — checked FIRST against NTP wall clock every tick.
            # Must run before the session-active check because session_end and
            # eod_cleanup both trigger at 16:00; if session check runs first
            # the loop breaks before cleanup (and reconciliation) ever fires.
            await self._check_eod_actions()

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

            # On first loop iteration after session opens, re-seed bars.
            # When the bot starts pre-market, _seed_recent_bars() finds 0
            # today bars (they don't exist yet). By 9:30 IB has delivered
            # the first bars, so re-seeding here populates _recent_bars
            # and enables FVG detection from the very first bar close.
            if not self._session_seeded:
                self._session_seeded = True
                self._seed_recent_bars()
                await self._backfill_fvgs()
                self.logger.log("session_open_reseed",
                                bars=len(self.fvg_mgr._recent_bars) if self.fvg_mgr else 0)

            # Disconnect safety: flatten all if IB has been down too long
            await self._check_disconnect_timeout()

            # Yield to event loop — ib_async processes bar updates and fill
            # callbacks during asyncio.sleep via nest_asyncio patching
            await asyncio.sleep(0.1)

    async def _check_disconnect_timeout(self):
        """Flatten all positions if IB has been disconnected beyond max_disconnect_minutes.

        Fires once per disconnect event (reset on reconnect by _on_reconnect).
        This prevents ghost positions sitting unmonitored while the connection is down.
        """
        if self._disconnect_flatten_done:
            return
        if self.config.dry_run or not self.ib_conn:
            return

        max_seconds = self.config.max_disconnect_minutes * 60
        disc_secs = self.ib_conn.disconnect_seconds
        if disc_secs >= max_seconds:
            self._disconnect_flatten_done = True
            self.logger.log(
                "disconnect_timeout_flatten",
                disconnect_seconds=round(disc_secs),
                threshold_minutes=self.config.max_disconnect_minutes,
            )
            if self.order_mgr and self.daily_state:
                await self.order_mgr.flatten_all(self.daily_state, "DISCONNECT_TIMEOUT")
                self.state_mgr.save(self.daily_state, force=True)
            if self.telegram.enabled:
                await self.telegram.alert_connection_lost(
                    f"IB disconnected for {round(disc_secs)}s — all positions flattened"
                )

    async def _check_eod_actions(self):
        """Fire EOD actions based on NTP-corrected wall clock time."""
        now_t = self.time_gates._now().time()

        if not self._eod_cancel_done and now_t >= self.time_gates.cancel_unfilled:
            self._eod_cancel_done = True
            await self._eod_cancel()

        if not self._eod_flatten_done and now_t >= self.time_gates.flatten_time:
            self._eod_flatten_done = True
            await self._eod_flatten()

        if not self._eod_cleanup_done and now_t >= self.time_gates.session_end:
            self._eod_cleanup_done = True
            await self._eod_cleanup()

    def _get_current_price(self) -> float:
        """Get the most recent trade price from bar data."""
        if self._bars_5min and len(self._bars_5min) > 0:
            return self._bars_5min[-1].close
        return 0.0

    async def _run_connection_test(self):
        """Connect to IB and print account/position info, then disconnect."""
        print(f"\nConnecting to IB at {self.config.ib_host}:{self.config.ib_port} ...")
        try:
            await self.ib_conn.connect()
        except Exception as e:
            print(f"CONNECTION FAILED: {e}")
            return

        ib = self.ib_conn.ib
        print("Connected.\n")

        # Account summary
        try:
            account_values = await ib.accountSummaryAsync()
            want = {"NetLiquidation", "AvailableFunds", "TotalCashValue", "UnrealizedPnL"}
            print("--- Account Summary ---")
            for av in account_values:
                if av.tag in want:
                    print(f"  {av.tag}: {av.value} {av.currency}")
        except Exception as e:
            print(f"  (Could not fetch account summary: {e})")

        # Open positions
        try:
            positions = await ib.reqPositionsAsync()
            print("\n--- Open Positions ---")
            if positions:
                for p in positions:
                    print(f"  {p.contract.symbol} {p.contract.lastTradeDateOrContractMonth}"
                          f"  qty={p.position}  avgCost={p.avgCost}")
            else:
                print("  (none)")
        except Exception as e:
            print(f"  (Could not fetch positions: {e})")

        print("\nConnection test complete.")
        await self.ib_conn.disconnect()

    def _calendar_gate_allows_entry(self):
        """Check strategy hard calendar gates (e.g., witching filters)."""
        strategy_meta = self.strategy.strategy.get("meta", {}) if self.strategy and self.strategy.strategy else {}
        hard_gates = strategy_meta.get("hard_gates", {})
        cfg = WitchingGateConfig(
            no_trade_witching_day=bool(hard_gates.get("no_trade_witching_day", False)),
            no_trade_witching_day_minus_1=bool(hard_gates.get("no_trade_witching_day_minus_1", False)),
        )
        today = self.clock.now().date()
        blocked, reason = is_blocked_by_witching_gate(today, cfg)
        if blocked:
            return False, reason
        return True, ""

    async def _on_order_resolved(self, released_qty: int = 0):
        """Called when any order resolves (TP/SL/cancel). Releases reserved margin
        and re-evaluates suspended orders."""
        if self.margin_tracker and released_qty > 0:
            self.margin_tracker.release(released_qty)
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

    def _register_ib_error_handler(self):
        """Register global IB error handler for graceful logging of known benign codes."""
        import logging

        ib = self.ib_conn.ib
        logger = self.logger

        # Codes handled by the bot logger instead of raw ib_async output
        _HANDLED_CODES = {
            202: "oca_cancel",   # OCA sibling cancelled (TP cancelled when SL fills, etc.)
            399: "ib_reprice",   # Bracket order self-trade prevention — informational
        }

        def _on_ib_error(reqId, errorCode, errorString, contract):
            event_name = _HANDLED_CODES.get(errorCode)
            if event_name:
                logger.log(event_name, reqId=reqId, code=errorCode, msg=errorString.strip())

        ib.errorEvent += _on_ib_error
        self._ib_error_handler = _on_ib_error  # prevent GC

        # Suppress raw ib_async log lines for these codes
        class _BenignCodeFilter(logging.Filter):
            def filter(self, record):
                msg = record.getMessage()
                for code in _HANDLED_CODES:
                    if f"Error {code}" in msg or f"Warning {code}" in msg:
                        return False
                return True

        for name in ("ib_async.ib", "ib_async.wrapper", "ib_async.client", "ib_async"):
            logging.getLogger(name).addFilter(_BenignCodeFilter())

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

        IMPORTANT: Only seeds bars from TODAY's session. With useRTH=True and
        durationStr="1 D", IB may return prior-day bars (e.g. Friday bars on
        Monday). Seeding cross-day bars causes false FVG detections from
        weekend/overnight gaps.
        """
        if not self._bars_5min or not self.fvg_mgr:
            return

        today = datetime.now(NY_TZ).date()

        # All bars except the last one (which is incomplete with keepUpToDate)
        completed = self._bars_5min[:-1] if len(self._bars_5min) > 1 else []

        # Seed the last N completed bars from TODAY only
        # (deque maxlen=10 handles overflow)
        seeded = 0
        for bar in completed[-10:]:
            bar_dict = {
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "date": _bar_date_to_et(bar.date),
            }
            bar_date = bar_dict["date"]
            if hasattr(bar_date, "date") and bar_date.date() != today:
                continue
            self.fvg_mgr.append_bar(bar_dict)
            seeded += 1

        self.logger.log("recent_bars_seeded", count=seeded)

    def _load_hfoiv_history(self):
        """Load historical imbalance parquets to pre-populate HFOIV normalization.

        Reads the last N (lookback_sessions) parquets from bot/data/imbalance/,
        feeds each day's bars through reset_day() + update() to build the
        cross-session percentile history.  Calls reset_day() one final time
        to prepare for today's live session.
        """
        import glob
        import pandas as pd

        imbalance_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "imbalance"
        )
        if not os.path.isdir(imbalance_dir):
            self.logger.log("hfoiv_history_skip", reason="no_imbalance_dir")
            return

        pattern = os.path.join(imbalance_dir, "nq_imbalance_5min_*.parquet")
        files = sorted(glob.glob(pattern))
        lookback = self.hfoiv_gate.config.lookback_sessions
        files = files[-lookback:] if len(files) > lookback else files

        loaded = 0
        for fpath in files:
            try:
                df = pd.read_parquet(fpath)
                # Validate schema before committing to gate state
                if "date" not in df.columns or "imbalance" not in df.columns:
                    self.logger.log("hfoiv_history_error",
                                    file=os.path.basename(fpath),
                                    error="missing date/imbalance columns")
                    continue
                if df.empty:
                    continue
                self.hfoiv_gate.reset_day()
                for _, row in df.iterrows():
                    bar_dt = row["date"]
                    bar_min = bar_dt.hour * 60 + bar_dt.minute
                    self.hfoiv_gate.update(bar_min, float(row["imbalance"]))
                loaded += 1
            except Exception as e:
                self.logger.log("hfoiv_history_error",
                                file=os.path.basename(fpath), error=str(e))

        # Final reset for today's live session
        self.hfoiv_gate.reset_day()
        self.logger.log("hfoiv_history_loaded",
                        sessions=loaded,
                        session_count=self.hfoiv_gate._session_count)

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

            # Feed imbalance accumulator for HFOIV gate (ETH 08:30-16:00)
            if self._imbalance_accumulator is not None:
                tick_size = getattr(tick, 'size', None)
                if tick_size is not None and tick_size > 0:
                    imb_bar = self._imbalance_accumulator.on_tick(
                        tick.price, tick_size, tick_time_et)
                    if imb_bar is not None and self.hfoiv_gate is not None:
                        self.hfoiv_gate.update(
                            imb_bar["bar_minutes"], imb_bar["imbalance"])

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

    async def _backfill_fvgs(self):
        """
        Scan today's COMPLETED 5-min bars for un-mitigated FVGs.

        On crash/restart mid-session, the bot loses its active FVG list.
        This replays the already-loaded 5-min bars (from the keepUpToDate
        subscription which includes today's history) to rebuild the list.
        Then checks 1-min bars to see if any were already mitigated.

        IMPORTANT: With useRTH=True and durationStr="1 D", IB may return
        prior-day bars (e.g. Friday bars on Monday). We filter to today's
        bars only and exclude the last bar (incomplete with keepUpToDate)
        to prevent false FVGs from weekend/overnight gaps.
        """
        if self._bars_5min is None or len(self._bars_5min) < 3:
            return

        from bot.strategy.fvg_detector import check_fvg_3bars, _assign_time_period, SESSION_INTERVALS
        from bot.strategy.trade_calculator import round_to_tick

        today = datetime.now(NY_TZ).date()

        # Build today-only completed bars list.
        # Exclude the last bar (incomplete with keepUpToDate).
        completed_bars = []
        source = self._bars_5min[:-1] if len(self._bars_5min) > 1 else []
        for bar in source:
            bar_et = _bar_date_to_et(bar.date)
            if hasattr(bar_et, "date") and bar_et.date() != today:
                continue
            completed_bars.append({
                "open": bar.open, "high": bar.high,
                "low": bar.low, "close": bar.close,
                "date": bar_et,
            })

        if len(completed_bars) < 3:
            self.logger.log(
                "fvg_backfill",
                bars_scanned=len(completed_bars),
                fvgs_found=0,
                total_active=self.fvg_mgr.active_count,
            )
            return

        # Build set of already-traded zones to prevent duplicate entries on restart
        _traded_zones = set()
        for og in (self.daily_state.pending_orders
                   + self.daily_state.open_positions
                   + self.daily_state.closed_trades):
            _traded_zones.add((round(og.entry_price, 2), round(og.stop_price, 2)))

        # Replay today's completed 5-min bars to detect FVGs
        detected = 0
        for i in range(2, len(completed_bars)):
            b1 = completed_bars[i - 2]
            b2 = completed_bars[i - 1]
            b3 = completed_bars[i]

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
            bars_scanned=len(completed_bars),
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
        # Suspended orders also "own" their FVG — backfill must NOT create a
        # duplicate OrderGroup for the same zone. The suspended-queue kick at
        # startup is responsible for reactivating these via margin_priority.
        for og in self.daily_state.suspended_orders:
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
        self._disconnect_flatten_done = False  # Reset for next potential disconnect
        self.logger.log("reconnect_resubscribe")
        try:
            await self._subscribe_bars()
            self._seed_recent_bars()
            self._subscribe_ticks()
            # Reset imbalance accumulator (tick rule state stale after disconnect)
            if self._imbalance_accumulator is not None:
                self._imbalance_accumulator.reset()
            await self._backfill_fvgs()
            await self._process_backfilled_fvgs()
            # Verify TP/SL quantities match filled_qty (fix failed adjustments)
            if self.order_mgr and self.daily_state:
                await self.order_mgr.verify_child_order_quantities(self.daily_state)
            self.logger.log("reconnect_resubscribe_done",
                            active_fvgs=self.fvg_mgr.active_count)
        except Exception as e:
            self.logger.log("reconnect_resubscribe_error", error=str(e))

    def _on_5min_update(self, bars, hasNewBar):
        """Callback when 5min bar data updates.

        With keepUpToDate=True, hasNewBar=True means a new bar STARTED.
        bars[-1] is the new incomplete bar, bars[-2] is the just-completed bar.
        We detect FVGs on the COMPLETED bar (bars[-2]), never on the incomplete one.

        This is the SOLE source of FVG detection — the tick path was removed
        because it could produce OHLC values that diverged from the canonical
        post-close bar, breaking live/backtest parity.
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

        # Block entries if NTP has never synced — session timing cannot be trusted
        if not self.clock.is_trusted:
            self.logger.log("setup_rejected", fvg_id=fvg.fvg_id,
                            gate="clock", reason="NTP never synced — clock untrusted")
            return

        allowed, reason = self._calendar_gate_allows_entry()
        if not allowed:
            self.logger.log("setup_rejected", fvg_id=fvg.fvg_id, gate="calendar", reason=reason)
            self.fvg_mgr.remove(fvg.fvg_id)
            self.daily_state.active_fvgs = self.fvg_mgr.active_fvgs
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

            # HFOIV gate — reduce size when imbalance volatility is elevated
            hfoiv_mult = 1.0
            hfoiv_info = {}
            if self.hfoiv_gate is not None:
                _now = self.clock.now()
                minutes_now = _now.hour * 60 + _now.minute if _now else 0
                hfoiv_mult, hfoiv_info = self.hfoiv_gate.get_size_multiplier(minutes_now)
                if hfoiv_mult < 1.0:
                    pre_hfoiv_qty = order.target_qty
                    import math as _math
                    order.target_qty = max(1, _math.floor(order.target_qty * hfoiv_mult))
                    self.logger.log(
                        "hfoiv_scale",
                        group_id=order.group_id,
                        original_qty=pre_hfoiv_qty,
                        scaled_qty=order.target_qty,
                        multiplier=hfoiv_mult,
                        percentile=hfoiv_info.get("percentile"),
                        bucket=hfoiv_info.get("bucket"),
                        hfoiv=hfoiv_info.get("hfoiv"),
                    )

            # Place the bracket order
            risk_dollars = round(order.risk_pts * order.target_qty * 20, 2)
            reward_dollars = round(order.risk_pts * order.n_value * order.target_qty * 20, 2)
            setup_log = dict(
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
            if hfoiv_mult < 1.0:
                setup_log["hfoiv_mult"] = hfoiv_mult
                setup_log["hfoiv_pct"] = hfoiv_info.get("percentile")
            self.logger.log("setup_accepted", **setup_log)

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
        """Log EOD schedule. Actions are fired by wall-clock checks in _check_eod_actions."""
        schedule = [
            (self.time_gates.cancel_unfilled, "cancel_unfilled"),
            (self.time_gates.flatten_time, "flatten_all"),
            (self.time_gates.session_end, "session_end"),
        ]
        for action_time, action_name in schedule:
            self.logger.log("eod_scheduled", action=action_name, time=action_time.strftime("%H:%M"))


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

        # Flush today's HFOIV values to history + reset for next session
        if self.hfoiv_gate is not None:
            self.hfoiv_gate.reset_day()
            self.logger.log("hfoiv_eod_flush",
                            session_count=self.hfoiv_gate._session_count)
        if self._imbalance_accumulator is not None:
            self._imbalance_accumulator.reset()

        # Compute comprehensive daily stats — exclude unfilled cancels
        all_closed = self.daily_state.closed_trades
        closed = [
            t for t in all_closed
            if not (t.close_reason in (CLOSE_CANCEL, CLOSE_EOD, CLOSE_REJECTED)
                    and t.filled_qty == 0)
        ]
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

        # Run EOD reconciliation (daily summary is merged into the recon message)
        try:
            await self._eod_reconcile()
        except Exception as e:
            self.logger.log("eod_reconcile_error", error=str(e), phase="orchestration")
            # Reconciliation failed — still send daily summary as fallback
            if self.telegram.enabled:
                await self.telegram.alert_daily_summary(self.daily_state)

        self._shutdown = True

    # ── EOD Reconciliation ────────────────────────────────────────────────

    async def _eod_reconcile(self):
        """Download today's data, run backtest, compare to live trades, report."""
        from bot.backtest.eod_reconciler import (
            match_trades, format_telegram_report, build_backtest_config,
            build_weekly_summary, result_to_db_kwargs, ReconciliationResult,
            validate_fills, has_bad_fills,
        )

        today = self.daily_state.date          # "2026-03-31"
        today_fmt = today.replace("-", "")      # "20260331"

        self.logger.log("eod_reconcile_start", date=today)

        # 1. Wait for IB data to become available
        await asyncio.sleep(180)

        # 2. Download 1-second bars via Windows subprocess
        data_dir = self._download_data_dir()
        data_file = await self._download_today_data(today_fmt, data_dir)
        def _stamp_daily(r):
            """Attach daily summary fields to a ReconciliationResult."""
            r.daily_pnl = self.daily_state.realized_pnl
            r.daily_pnl_pct = self.daily_state.daily_pnl_pct * 100
            r.balance = self.daily_state.start_balance + self.daily_state.realized_pnl
            r.filled_trades = self.daily_state.filled_trade_count
            return r

        if data_file is None:
            err = "1-second bar download failed after 3 attempts"
            self.logger.log("eod_reconcile_skip", reason=err)
            result = _stamp_daily(ReconciliationResult(
                date=today, live_count=0, backtest_count=0,
                matched_count=0, error=err))
            self.db.insert_reconciliation(**result_to_db_kwargs(result))
            if self.telegram.enabled:
                await self.telegram.alert_reconciliation(
                    format_telegram_report(result))
            return

        # 3. Run backtester
        try:
            bt_trades = await self._run_reconciliation_backtest(
                today_fmt, data_dir)
        except Exception as e:
            err = f"Backtest failed: {e}"
            self.logger.log("eod_reconcile_error", error=str(e), phase="backtest")
            result = _stamp_daily(ReconciliationResult(
                date=today, live_count=0, backtest_count=0,
                matched_count=0, error=err))
            self.db.insert_reconciliation(**result_to_db_kwargs(result))
            if self.telegram.enabled:
                await self.telegram.alert_reconciliation(
                    format_telegram_report(result))
            return

        # 4. Load live trades from DB (only closed trades with exit_reason)
        live_trades = self.db.get_trades(date=today, limit=999)
        live_trades = [t for t in live_trades if t.get("exit_reason")]

        # 5. Compare
        hfoiv_on = self.hfoiv_gate is not None
        result = match_trades(live_trades, bt_trades, hfoiv_active=hfoiv_on)
        result.date = today
        result.kill_switch_active = self.daily_state.kill_switch_active
        _stamp_daily(result)

        # 6. Tick-validate paper fills (skip in live mode — IB fills are real)
        fills_garbage = False
        if live_trades and self.config.paper_mode:
            try:
                ticks_by_window = await self._fetch_fill_ticks(
                    today_fmt, live_trades)
                if ticks_by_window:
                    checks = validate_fills(live_trades, ticks_by_window)
                    fills_garbage = has_bad_fills(checks)
                    bad_count = sum(1 for c in checks if not c.valid)
                    self.logger.log("eod_reconcile_tick_validation",
                                    checked=len(checks), bad=bad_count)
            except Exception as e:
                self.logger.log("eod_reconcile_tick_error", error=str(e))

        # 7. Save to DB
        self.db.insert_reconciliation(**result_to_db_kwargs(result))

        # 8. Build weekly summary (Fridays only)
        balance = self.daily_state.start_balance + self.daily_state.realized_pnl
        weekly_html = build_weekly_summary(self.db, today, balance)

        # 9. Send Telegram report
        self.logger.log(
            "eod_reconcile_done",
            date=today,
            live_trades=result.live_count,
            backtest_trades=result.backtest_count,
            matched=result.matched_count,
            divergences=len(result.divergences),
            weekly_report=weekly_html is not None,
        )

        if self.telegram.enabled:
            msg = format_telegram_report(result, weekly_html,
                                         fills_garbage=fills_garbage)
            await self.telegram.alert_reconciliation(msg)

    def _download_data_dir(self):
        """Return the bot/data/ directory path (creating if needed)."""
        d = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(d, exist_ok=True)
        return d

    async def _download_today_data(self, date_str, data_dir):
        """Download today's 1-second bars directly via ib_async.

        Connects with a separate clientId (20) so it doesn't interfere
        with the bot's main connection. Returns parquet path or None.
        """
        out_file = os.path.join(data_dir, f"nq_1secs_{date_str}.parquet")

        # If already downloaded, skip
        if os.path.exists(out_file):
            self.logger.log("eod_reconcile_download_cached", file=out_file)
            return out_file

        loop = asyncio.get_event_loop()

        def _fetch():
            import time as _time
            from datetime import date as _date
            import pandas as _pd
            import pytz as _pz
            from ib_async import IB, Future

            _ET = _pz.timezone('US/Eastern')
            _UTC = _pz.utc

            def _to_utc_str(naive_et_dt):
                return _ET.localize(naive_et_dt).astimezone(_UTC).strftime('%Y%m%d-%H:%M:%S')

            # Resolve NQ contract for the date
            from bot.bridge.ib_data_fetcher import get_nq_contract_for_date
            trade_date = _date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
            exp_date = get_nq_contract_for_date(trade_date)
            contract = Future('NQ', exchange='CME',
                              lastTradeDateOrContractMonth=exp_date.strftime('%Y%m'))

            ib = IB()
            ib.connect(self.config.ib_host, self.config.ib_port,
                       clientId=20, timeout=15)

            try:
                ib.qualifyContracts(contract)

                # Paginate in 30-min chunks (IB limit for 1-sec bars)
                all_records = []
                chunks = []
                end_hour, end_min = 16, 0
                for _ in range(14):
                    end_dt = datetime(trade_date.year, trade_date.month,
                                      trade_date.day, end_hour, end_min, 0)
                    chunks.append(end_dt)
                    end_min -= 30
                    if end_min < 0:
                        end_min += 60
                        end_hour -= 1
                    if end_hour < 9 or (end_hour == 9 and end_min < 30):
                        break
                chunks.reverse()

                pacing_error = [False]
                def on_error(reqId, errorCode, errorString, contract):
                    if errorCode == 162:
                        pacing_error[0] = True
                ib.errorEvent += on_error

                for ci, end_dt in enumerate(chunks):
                    end_str = _to_utc_str(end_dt)
                    bars = None
                    for attempt in range(4):
                        pacing_error[0] = False
                        try:
                            bars = ib.reqHistoricalData(
                                contract, endDateTime=end_str,
                                durationStr='1800 S', barSizeSetting='1 secs',
                                whatToShow='TRADES', useRTH=True,
                                formatDate=2, timeout=60)
                        except Exception:
                            bars = None
                        if pacing_error[0] or (bars is not None and len(bars) == 0):
                            _time.sleep(20 * (attempt + 1))
                            continue
                        break
                    if bars:
                        for bar in bars:
                            all_records.append({
                                'date': str(bar.date), 'open': bar.open,
                                'high': bar.high, 'low': bar.low,
                                'close': bar.close, 'volume': bar.volume,
                            })
                    _time.sleep(11)  # IB pacing

                ib.errorEvent -= on_error

                if not all_records:
                    return None

                # Deduplicate + sort
                seen = set()
                deduped = []
                for r in all_records:
                    if r['date'] not in seen:
                        seen.add(r['date'])
                        deduped.append(r)
                deduped.sort(key=lambda x: x['date'])

                # Write parquet
                df = _pd.DataFrame(deduped)
                df.to_parquet(out_file, index=False)
                return out_file

            finally:
                ib.disconnect()

        for attempt in range(3):
            try:
                result = await loop.run_in_executor(None, _fetch)
                if result:
                    self.logger.log("eod_reconcile_download_ok",
                                    file=out_file)
                    return result
                self.logger.log("eod_reconcile_download_retry",
                                attempt=attempt + 1, reason="no bars returned")
                await asyncio.sleep(60)
            except Exception as e:
                self.logger.log("eod_reconcile_download_retry",
                                attempt=attempt + 1, error=str(e))
                await asyncio.sleep(60)

        return None

    async def _fetch_fill_ticks(self, date_str, live_trades):
        """Fetch tick data around each live trade's fill times for validation.

        Returns dict mapping (group_id, "entry"|"exit") -> list of tick dicts.
        Each tick: {"price": float, "size": int, "time_utc": str}
        """
        from datetime import date as _date, timedelta as _td
        from bot.bridge.ib_data_fetcher import get_nq_contract_for_date

        trade_date = _date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        WINDOW_SECS = 10  # +/- 10 seconds around each fill

        # Collect unique (fill_time, fill_type, group_id) windows
        windows = []
        for t in live_trades:
            for fill_type, time_key in [("entry", "entry_time"), ("exit", "exit_time")]:
                ft = t.get(time_key)
                if not ft:
                    continue
                windows.append((t["group_id"], fill_type, ft))

        if not windows:
            return {}

        loop = asyncio.get_event_loop()

        def _fetch():
            import time as _time
            import pytz as _pz
            from ib_async import IB, Future
            from dateutil import parser as dtparser

            _ET = _pz.timezone('US/Eastern')
            _UTC = _pz.utc

            exp_date = get_nq_contract_for_date(trade_date)
            contract = Future('NQ', exchange='CME',
                              lastTradeDateOrContractMonth=exp_date.strftime('%Y%m'))

            ib = IB()
            ib.connect(self.config.ib_host, self.config.ib_port,
                       clientId=21, timeout=15)

            try:
                ib.qualifyContracts(contract)
                result = {}

                for gid, fill_type, fill_time_str in windows:
                    # Parse fill time to UTC
                    ft = dtparser.parse(fill_time_str)
                    if ft.tzinfo is not None:
                        ft_utc = ft.astimezone(_UTC)
                    else:
                        ft_utc = _ET.localize(ft).astimezone(_UTC)

                    start_utc = ft_utc - _td(seconds=WINDOW_SECS)
                    start_str = start_utc.strftime('%Y%m%d-%H:%M:%S')

                    try:
                        ticks = ib.reqHistoricalTicks(
                            contract,
                            startDateTime=start_str,
                            endDateTime='',
                            numberOfTicks=1000,
                            whatToShow='TRADES',
                            useRth=True,
                        )
                    except Exception:
                        ticks = []

                    end_utc = ft_utc + _td(seconds=WINDOW_SECS)
                    filtered = []
                    for tick in (ticks or []):
                        t_utc = tick.time
                        if t_utc.tzinfo is None:
                            t_utc = _UTC.localize(t_utc)
                        if start_utc <= t_utc <= end_utc:
                            filtered.append({
                                "price": tick.price,
                                "size": tick.size,
                                "time_utc": t_utc.strftime('%Y-%m-%d %H:%M:%S.%f'),
                            })

                    result[(gid, fill_type)] = filtered
                    _time.sleep(2)  # Pacing between tick requests

                return result
            finally:
                ib.disconnect()

        return await loop.run_in_executor(None, _fetch)

    async def _run_reconciliation_backtest(self, date_str, data_dir):
        """Run the backtester on today's data with the current strategy.

        Returns list of backtester.Trade objects.
        """
        from bot.backtest.backtester import load_1s_bars, run_backtest
        from bot.backtest.eod_reconciler import build_backtest_config, load_margin_schedule

        strategy_dict = self.strategy.strategy
        margin_schedule = load_margin_schedule(self.config.log_dir, date_str)
        config = build_backtest_config(
            self.config, strategy_dict, self.daily_state.start_balance,
            margin_schedule=margin_schedule)

        loop = asyncio.get_event_loop()

        def _run():
            df = load_1s_bars(data_dir, start_date=date_str, end_date=date_str)
            trades, _ = run_backtest(df, strategy_dict, config)
            return trades

        return await loop.run_in_executor(None, _run)

    async def _shutdown_gracefully(self):
        """Clean shutdown: cancel tasks, save state, disconnect."""
        # Cancel periodic tasks
        for task in self._periodic_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # EOD tasks are now wall-clock driven (no call_later handles to cancel)

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
