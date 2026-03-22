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
from datetime import datetime

import pytz

# Add project root for logic imports
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bot.bot_config import BotConfig
from bot.clock import Clock
from bot.logging.bot_logger import BotLogger
from bot.state.state_manager import StateManager
from bot.state.trade_state import DailyState
from bot.strategy.strategy_loader import StrategyLoader
from bot.strategy.fvg_detector import ActiveFVGManager
from bot.strategy.mitigation_scanner import scan_active_fvgs
from bot.strategy.trade_calculator import calculate_setup
from bot.risk.risk_gates import RiskGates
from bot.risk.time_gates import TimeGates
from bot.execution.ib_connection import IBConnection
from bot.execution.order_manager import OrderManager
from bot.execution.position_tracker import get_account_balance, get_ib_open_orders, get_ib_positions
from bot.alerts.telegram import TelegramAlerter
from bot.db import TradeDB

NY_TZ = pytz.timezone("America/New_York")


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
        self.telegram = TelegramAlerter(
            config.telegram_bot_token, config.telegram_chat_id, self.logger
        )
        self.db = TradeDB(os.path.join(config.state_dir, "bot_trades.db"))
        self.fvg_mgr = None         # Initialized after strategy loads
        self.order_mgr = None        # Initialized after IB connects
        self.daily_state = None
        self._contract = None
        self._bars_5min = None
        self._bars_1min = None
        self._shutdown = False
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
            )

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

        # 7. Schedule EOD actions
        self._schedule_eod()

        # 8. Schedule periodic tasks
        self._schedule_periodic()

        # 9. Telegram notification
        if self.telegram.enabled:
            await self.telegram.alert_bot_start(self.config, self.strategy.strategy_id)

        self.logger.log(
            "startup_complete",
            strategy=self.strategy.strategy_id,
            cells=self.strategy.cell_count,
            active_fvgs=self.fvg_mgr.active_count,
        )

    async def _event_loop(self):
        """Run until shutdown signal, session end, or kill switch."""
        while not self._shutdown:
            if self.daily_state and self.daily_state.kill_switch_active:
                await self._trigger_kill_switch()
                break

            if not self.time_gates.is_session_active():
                self.logger.log("bot_stop", reason="session_ended")
                break

            # Let asyncio process events (IB callbacks, timers, etc.)
            await asyncio.sleep(0.1)

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

        contract = Future(
            symbol=self.config.ticker,
            lastTradeDateOrContractMonth=exp_date.strftime("%Y%m%d"),
            exchange=self.config.exchange,
            currency=self.config.currency,
        )

        qualified = await self.ib_conn.ib.qualifyContractsAsync(contract)
        if qualified:
            self.logger.log(
                "contract_resolved",
                symbol=contract.symbol,
                expiry=exp_date.strftime("%Y%m%d"),
                conId=contract.conId,
            )
            return contract
        else:
            raise RuntimeError(f"Failed to qualify contract: {contract}")

    async def _subscribe_bars(self):
        """Subscribe to 5min and 1min keepUpToDate historical bars."""
        ib = self.ib_conn.ib

        # 5-minute bars
        self._bars_5min = ib.reqHistoricalData(
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
        self._bars_1min = ib.reqHistoricalData(
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

    def _on_5min_update(self, bars, hasNewBar):
        """Callback when 5min bar data updates."""
        if not hasNewBar or len(bars) < 1:
            return

        bar = bars[-1]
        bar_dict = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "date": bar.date,
        }

        # Detect FVG on the completed bar
        fvg = self.fvg_mgr.on_5min_bar(bar_dict)
        if fvg:
            self.daily_state.active_fvgs = self.fvg_mgr.active_fvgs
            self.state_mgr.save(self.daily_state)
            # Log to DB
            self.db.insert_fvg(
                fvg_id=fvg.fvg_id, trade_date=fvg.formation_date,
                fvg_type=fvg.fvg_type, zone_low=fvg.zone_low,
                zone_high=fvg.zone_high, fvg_size=round(fvg.zone_high - fvg.zone_low, 2),
                time_period=fvg.time_period, formation_time=fvg.time_candle3,
            )

    def _on_1min_update(self, bars, hasNewBar):
        """Callback when 1min bar data updates."""
        if not hasNewBar or len(bars) < 1:
            return

        bar = bars[-1]
        bar_dict = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "date": bar.date,
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

            # Process trade setup (synchronous — sequential for race safety)
            asyncio.ensure_future(self._process_mitigation(fvg, bar_dict))

            # Remove from active list
            self.fvg_mgr.remove(fvg.fvg_id)

        if mitigated:
            self.daily_state.active_fvgs = self.fvg_mgr.active_fvgs
            self.state_mgr.save(self.daily_state)

    async def _process_mitigation(self, fvg, bar):
        """
        Evaluate and potentially place a trade for a mitigated FVG.
        """
        # Check time gate
        allowed, reason = self.time_gates.can_enter()
        if not allowed:
            self.logger.log("setup_rejected", fvg_id=fvg.fvg_id, gate="time", reason=reason)
            return

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
                continue

            # Look up strategy cell
            cell = self.strategy.find_cell(fvg.time_period, risk_pts)
            if cell is None:
                continue

            # Only proceed if cell matches this setup type
            if cell["setup"] != setup_type:
                continue

            # Calculate full setup
            balance = self.daily_state.start_balance + self.daily_state.realized_pnl
            order = calculate_setup(fvg, cell, balance, self.config.risk_per_trade)
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

            if self.order_mgr:
                await self.order_mgr.place_bracket(order, self.daily_state)
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

            self.state_mgr.save(self.daily_state)
            break  # Only one setup per FVG

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

        self._periodic_tasks.append(asyncio.ensure_future(periodic_save()))
        self._periodic_tasks.append(asyncio.ensure_future(periodic_strategy_check()))
        self._periodic_tasks.append(asyncio.ensure_future(periodic_clock_sync()))

    async def _eod_cancel(self):
        """15:50 — Cancel all unfilled entry orders."""
        self.logger.log("eod_cancel", time="15:50")
        if self.order_mgr:
            await self.order_mgr.cancel_all_pending(self.daily_state, "EOD")
        self.state_mgr.save(self.daily_state, force=True)

    async def _eod_flatten(self):
        """15:55 — Flatten all open positions at market."""
        self.logger.log("eod_flatten", time="15:55")
        if self.order_mgr:
            await self.order_mgr.flatten_all(self.daily_state, "EOD")
        self.state_mgr.save(self.daily_state, force=True)

    async def _eod_cleanup(self):
        """16:00 — Expire FVGs, daily summary, final state save."""
        expired = self.fvg_mgr.expire_all()
        self.daily_state.active_fvgs = []

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

        # Save final state
        if self.daily_state:
            self.state_mgr.save(self.daily_state, force=True)

        # Disconnect from IB
        if self.ib_conn.is_connected:
            await self.ib_conn.disconnect()

        self.logger.log("bot_stop", reason="shutdown")
        self.logger.close()
