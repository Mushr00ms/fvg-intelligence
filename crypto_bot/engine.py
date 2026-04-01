"""
engine.py - Standalone Binance/BTC trading engine.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from bot.alerts.telegram import TelegramAlerter
from bot.bot_logging.bot_logger import BotLogger
from bot.clock import Clock
from bot.execution.binance_futures_client import BinanceFuturesClient, BinanceFuturesError
from bot.execution.execution_types import AccountSnapshot, PositionSnapshot

from crypto_bot.execution import BinanceExecutionManager
from crypto_bot.fvg import ActiveFVGManager, parse_ts
from crypto_bot.market_data import BinanceMarketData
from crypto_bot.models import OrderIntent, new_runtime_state
from crypto_bot.reconciliation import reconcile_runtime_state
from crypto_bot.risk import CryptoRiskManager
from crypto_bot.state_store import StateStore
from crypto_bot.strategy import BTCStrategyLoader


class CryptoBotEngine:
    def __init__(self, config):
        self.config = config
        self.clock = Clock()
        self.logger = BotLogger(config.log_dir, clock=self.clock)
        self.state_store = StateStore(config.state_dir)
        self.strategy = BTCStrategyLoader(config.strategy_path)
        self.market_data = BinanceMarketData(
            config.base_url,
            config.ws_base_url,
            market_timezone=config.market_timezone,
        )
        self.client = BinanceFuturesClient(
            config.api_key,
            config.api_secret,
            base_url=config.base_url,
            ws_base_url=config.ws_base_url,
            recv_window=config.recv_window,
            logger=self.logger,
            clock=self.clock,
        )
        self.telegram = TelegramAlerter(
            config.telegram_bot_token,
            config.telegram_chat_id,
            self.logger,
        )
        self.state = None
        self.fvg_mgr = ActiveFVGManager(
            min_fvg_bps=config.min_fvg_bps,
            mitigation_window_5m=config.mitigation_window_5m,
            logger=self.logger,
            market_timezone=config.market_timezone,
            symbol=config.symbol,
        )
        self.rules = None
        self.risk_mgr = None
        self.exec_mgr = None
        self.account_asset = config.account_asset or "USDT"
        self._shutdown = False
        self._listen_key = None
        self._listen_key_stop = asyncio.Event()
        self._background_tasks = []
        self._reset_tz = ZoneInfo(config.daily_reset_timezone)
        self._market_tz = ZoneInfo(config.market_timezone)

    async def run(self):
        try:
            await self._startup()
            await self._run_forever()
        finally:
            await self._shutdown_gracefully()

    async def _startup(self):
        self.clock.sync()
        self.strategy.load()
        await self.market_data.start()
        await self.client.start()
        self.rules = await self.client.get_symbol_rules(self.config.symbol)
        self.account_asset = self.config.account_asset or self.rules.margin_asset or "USDT"
        self.risk_mgr = CryptoRiskManager(self.config, self.rules, self.strategy)
        self.exec_mgr = BinanceExecutionManager(self.client, self.config, self.logger)

        live_snapshot = None
        if not self.config.dry_run:
            if not self.config.api_key or not self.config.api_secret:
                raise RuntimeError("Live mode requires Binance API credentials in crypto_bot_config.json")
            live_snapshot = await self._configure_live_account()

        self._load_runtime_state(live_snapshot)
        await self._seed_recent_history()

        if not self.config.dry_run:
            await self._reconcile_live_state(startup=True)
            await self._start_user_stream()
            self._background_tasks.append(asyncio.create_task(self._periodic_reconcile_loop()))

        self._background_tasks.append(asyncio.create_task(self._periodic_save_loop()))
        self.logger.log(
            "crypto_bot_started",
            mode=self.config.execution_mode,
            symbol=self.config.symbol,
            market_base_url=self.config.base_url,
            strategy_id=self.strategy.strategy_id,
            leverage=self.config.leverage,
            balance=self.state.current_balance,
            position_mode=self.config.position_mode,
        )
        if self.telegram.enabled:
            await self.telegram.send(
                "<b>CRYPTO BOT STARTED</b>\n\n"
                f"Mode: {self.config.execution_mode.upper()}\n"
                f"Symbol: {self.config.symbol}\n"
                f"Strategy: {self.strategy.strategy_id}\n"
                f"Leverage: {self.config.leverage}x\n"
                f"Position Mode: {self.config.position_mode}"
            )

    async def _configure_live_account(self) -> AccountSnapshot:
        positions = await self.client.get_positions(self.config.symbol)
        open_orders = await self.client.get_open_orders(self.config.symbol)
        current_hedge_mode = await self.client.get_position_mode()
        desired_hedge_mode = self.config.position_mode == "HEDGE"

        if positions or open_orders:
            if current_hedge_mode != desired_hedge_mode:
                raise RuntimeError(
                    f"Binance position mode mismatch with live exposure present. "
                    f"Exchange hedge_mode={current_hedge_mode}, config requires {desired_hedge_mode}."
                )
        elif current_hedge_mode != desired_hedge_mode:
            await self.client.set_position_mode(desired_hedge_mode)

        await self.client.set_leverage(self.config.symbol, self.config.leverage)
        try:
            await self.client.set_margin_type(self.config.symbol, self.config.margin_type)
        except BinanceFuturesError as exc:
            if exc.code not in (-4046,):
                raise

        return await self.client.get_account_snapshot(self.account_asset)

    def _load_runtime_state(self, live_snapshot: AccountSnapshot | None):
        state = self.state_store.load()
        today = self._current_day()
        balance = live_snapshot.wallet_balance if live_snapshot is not None else self.config.starting_balance

        if state is None:
            self.state = new_runtime_state(self.config.symbol, self.strategy.strategy_id, today, balance)
        else:
            self.state = state
            if self.state.day != today:
                self.state.reset_for_new_day(today)
            if live_snapshot is not None and balance > 0:
                self.state.current_balance = balance
                if self.state.start_balance <= 0:
                    self.state.start_balance = balance

    async def _seed_recent_history(self):
        self.fvg_mgr.reset()

        now_utc = self.clock.now_utc()
        five_min_limit = min(max(self.config.mitigation_window_5m + 8, 16), 1000)
        one_min_limit = min(max((self.config.mitigation_window_5m * 5) + 25, 60), 1000)

        bars_5m = await self.market_data.fetch_klines(self.config.symbol, "5m", limit=five_min_limit)
        completed_5m = [
            bar for bar in bars_5m
            if parse_ts(bar["close_time"]) <= now_utc
        ]
        self.fvg_mgr.seed_5m(completed_5m)

        bars_1m = await self.market_data.fetch_klines(self.config.symbol, "1m", limit=one_min_limit)
        completed_1m = [
            bar for bar in bars_1m
            if parse_ts(bar["close_time"]) <= now_utc
        ]
        for bar in completed_1m:
            self.fvg_mgr.scan_1m_close(bar)

        self.state.active_fvgs = self.fvg_mgr.active
        self.logger.log(
            "crypto_fvg_backfill",
            symbol=self.config.symbol,
            window_5m=self.config.mitigation_window_5m,
            bars_5m_scanned=len(completed_5m),
            bars_1m_scanned=len(completed_1m),
            active_fvgs=len(self.state.active_fvgs),
        )
        self.state_store.save(self.state, force=True)

    async def _start_user_stream(self):
        await self._create_listen_key()
        self._background_tasks.append(asyncio.create_task(self._user_stream_loop()))

    async def _create_listen_key(self):
        listen_key = await self.client.create_listen_key()
        self._listen_key = listen_key.listen_key
        self._listen_key_stop = asyncio.Event()
        self._background_tasks.append(
            asyncio.create_task(
                self.client.maintain_listen_key(
                    self._listen_key,
                    interval_seconds=self.config.user_stream_keepalive,
                    stop_event=self._listen_key_stop,
                )
            )
        )

    async def _reconcile_live_state(self, *, startup: bool):
        positions = await self.client.get_positions(self.config.symbol)
        open_orders = await self.client.get_open_orders(self.config.symbol)
        has_exchange_state = bool(positions or open_orders)
        if startup and has_exchange_state and not self.config.allow_start_with_open_positions:
            raise RuntimeError(
                "Refusing live startup with existing Binance positions/open orders. "
                "Flatten/cancel manually or set allow_start_with_open_positions=true."
            )
        if not has_exchange_state and not (self.state.pending_entries or self.state.open_positions):
            snapshot = await self.client.get_account_snapshot(self.account_asset)
            self.state.current_balance = snapshot.wallet_balance or self.state.current_balance
            self.state.pending_entries = []
            self.state.open_positions = []
            self.state_store.save(self.state)
            return

        reconciliation = await reconcile_runtime_state(
            self.state,
            client=self.client,
            symbol=self.config.symbol,
            leverage=self.config.leverage,
            resume_managed_positions=self.config.resume_managed_positions,
        )
        if reconciliation.unmanaged_order_ids or reconciliation.unmanaged_positions:
            raise RuntimeError(
                "Exchange contains unmanaged crypto state. "
                f"orders={reconciliation.unmanaged_order_ids} positions={reconciliation.unmanaged_positions}"
            )

        snapshot = await self.client.get_account_snapshot(self.account_asset)
        self.state.current_balance = snapshot.wallet_balance or self.state.current_balance
        self.logger.log(
            "crypto_reconcile_complete",
            startup=startup,
            resumed=len(reconciliation.resumed_groups),
            repaired=len(reconciliation.exit_repairs),
            stale=len(reconciliation.stale_groups),
            pending=len(self.state.pending_entries),
            open_positions=len(self.state.open_positions),
        )

        for group_id in reconciliation.exit_repairs:
            intent = next((i for i in self.state.open_positions if i.group_id == group_id), None)
            if intent is None:
                continue
            filled_qty = intent.filled_qty or intent.quantity
            if filled_qty <= 0:
                continue
            await self.exec_mgr.replace_exits(intent, filled_qty)
            self.logger.log("crypto_exit_repaired", group_id=group_id, filled_qty=filled_qty)

        self.state_store.save(self.state, force=True)

    async def _run_forever(self):
        while not self._shutdown:
            try:
                async for bar in self.market_data.stream_closed_klines(self.config.symbol, ["1m", "5m"]):
                    if self._shutdown:
                        break
                    self._roll_day_if_needed()
                    if bar["interval"] == "5m":
                        self.fvg_mgr.on_5m_close(bar)
                        self.state.active_fvgs = self.fvg_mgr.active
                    else:
                        mitigated = self.fvg_mgr.scan_1m_close(bar)
                        self.state.active_fvgs = self.fvg_mgr.active
                        await self._process_mitigations(mitigated)
                if not self._shutdown:
                    raise RuntimeError("market data stream closed")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._handle_stream_failure("market_data", exc)

    async def _process_mitigations(self, mitigated: list):
        if not mitigated:
            return

        available_balance = self.state.current_balance
        account_snapshot = None
        exchange_positions: list[PositionSnapshot] = []
        if not self.config.dry_run:
            account_snapshot = await self.client.get_account_snapshot(self.account_asset)
            exchange_positions = await self.client.get_positions(self.config.symbol)
            available_balance = account_snapshot.available_balance or account_snapshot.wallet_balance
            self.state.current_balance = account_snapshot.wallet_balance or self.state.current_balance

        candidates = []
        for fvg in mitigated:
            for setup in ("mit_extreme", "mid_extreme"):
                intent = self.risk_mgr.build_intent(fvg, setup, balance=available_balance)
                if intent is not None:
                    candidates.append((fvg, setup, intent))

        candidates.sort(key=lambda item: self._mitigation_priority(item[2]))
        batch_active_intents = list(self.state.pending_entries) + list(self.state.open_positions)

        for fvg, setup, intent in candidates:
            consecutive_reason = self.risk_mgr.consecutive_conflict_reason(batch_active_intents, intent)
            if consecutive_reason:
                self.logger.log(
                    "crypto_setup_rejected",
                    group_id=intent.group_id,
                    fvg_id=fvg.fvg_id,
                    setup=setup,
                    reason=consecutive_reason,
                )
                continue

            reason = self.risk_mgr.can_accept(
                self.state,
                intent,
                available_balance=available_balance,
                account_snapshot=account_snapshot,
                exchange_positions=exchange_positions,
            )
            if reason:
                self.logger.log(
                    "crypto_setup_rejected",
                    group_id=intent.group_id,
                    fvg_id=fvg.fvg_id,
                    setup=setup,
                    reason=reason,
                )
                continue

            fvg.orders_placed.append(setup)
            if self.config.dry_run:
                intent.status = "DRY_RUN"
                self.state.closed_trades.append(intent)
                self.state.trade_count += 1
                self.logger.log(
                    "crypto_dry_run_intent",
                    group_id=intent.group_id,
                    fvg_id=fvg.fvg_id,
                    setup=setup,
                    side=intent.side,
                    position_side=intent.position_side,
                    quantity=intent.quantity,
                    entry=intent.entry_price,
                    stop=intent.stop_price,
                    target=intent.target_price,
                    notional=intent.notional,
                    expected_loss=intent.expected_loss,
                )
            else:
                await self.exec_mgr.place_entry(intent)
                self.state.pending_entries.append(intent)
                self.state.trade_count += 1
                self.logger.log(
                    "crypto_setup_submitted",
                    group_id=intent.group_id,
                    fvg_id=fvg.fvg_id,
                    setup=setup,
                    side=intent.side,
                    position_side=intent.position_side,
                )
                batch_active_intents.append(intent)

            if self.config.dry_run:
                batch_active_intents.append(intent)
            self.state_store.save(self.state)

    async def _on_mitigation(self, fvg):
        available_balance = self.state.current_balance
        account_snapshot = None
        exchange_positions: list[PositionSnapshot] = []

        if not self.config.dry_run:
            account_snapshot = await self.client.get_account_snapshot(self.account_asset)
            exchange_positions = await self.client.get_positions(self.config.symbol)
            available_balance = account_snapshot.available_balance or account_snapshot.wallet_balance
            self.state.current_balance = account_snapshot.wallet_balance or self.state.current_balance

        for setup in ("mit_extreme", "mid_extreme"):
            intent = self.risk_mgr.build_intent(fvg, setup, balance=available_balance)
            if intent is None:
                continue
            reason = self.risk_mgr.can_accept(
                self.state,
                intent,
                available_balance=available_balance,
                account_snapshot=account_snapshot,
                exchange_positions=exchange_positions,
            )
            if reason:
                self.logger.log(
                    "crypto_setup_rejected",
                    group_id=intent.group_id,
                    fvg_id=fvg.fvg_id,
                    setup=setup,
                    reason=reason,
                )
                continue

            fvg.orders_placed.append(setup)
            if self.config.dry_run:
                intent.status = "DRY_RUN"
                self.state.closed_trades.append(intent)
                self.state.trade_count += 1
                self.logger.log(
                    "crypto_dry_run_intent",
                    group_id=intent.group_id,
                    fvg_id=fvg.fvg_id,
                    setup=setup,
                    side=intent.side,
                    position_side=intent.position_side,
                    quantity=intent.quantity,
                    entry=intent.entry_price,
                    stop=intent.stop_price,
                    target=intent.target_price,
                    notional=intent.notional,
                    expected_loss=intent.expected_loss,
                )
            else:
                await self.exec_mgr.place_entry(intent)
                self.state.pending_entries.append(intent)
                self.state.trade_count += 1
                self.logger.log(
                    "crypto_setup_submitted",
                    group_id=intent.group_id,
                    fvg_id=fvg.fvg_id,
                    setup=setup,
                    side=intent.side,
                    position_side=intent.position_side,
                )

            self.state_store.save(self.state)

    def _mitigation_priority(self, intent: OrderIntent):
        if intent.side == "BUY":
            return (0, -intent.entry_price, intent.stop_price)
        return (1, intent.entry_price, -intent.stop_price)

    async def _user_stream_loop(self):
        while not self._shutdown:
            try:
                if not self._listen_key:
                    await self._create_listen_key()
                async for event in self.client.user_stream(self._listen_key):
                    if self._shutdown:
                        break
                    await self._handle_user_stream_event(event)
                if not self._shutdown:
                    raise RuntimeError("user stream closed")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._handle_stream_failure("user_stream", exc)
                await self._reset_user_stream()
                await self._reconcile_live_state(startup=False)

    async def _handle_user_stream_event(self, event: dict):
        if event.get("e") == "ORDER_TRADE_UPDATE":
            order_update = event["o"]
            event_key = (
                f"order:{order_update.get('i', '')}:{order_update.get('x', '')}:"
                f"{order_update.get('X', '')}:{order_update.get('z', '')}:{event.get('E', '')}"
            )
            if self.state.has_processed_event(event_key):
                return
            await self._handle_order_trade_update(order_update)
            self.state.mark_event_processed(event_key)
            self.state_store.save(self.state)
        elif event.get("e") == "ALGO_UPDATE":
            algo_update = event["o"]
            event_key = f"algo:{algo_update.get('aid', '')}:{algo_update.get('X', '')}:{event.get('E', '')}"
            if self.state.has_processed_event(event_key):
                return
            await self._handle_algo_update(algo_update)
            self.state.mark_event_processed(event_key)
            self.state_store.save(self.state)
        elif event.get("e") == "ACCOUNT_UPDATE":
            event_key = f"account:{event.get('E', '')}"
            if self.state.has_processed_event(event_key):
                return
            await self._handle_account_update(event["a"])
            self.state.mark_event_processed(event_key)
            self.state_store.save(self.state)

    async def _handle_account_update(self, account_update: dict):
        for balance in account_update.get("B", []):
            if balance.get("a") == self.account_asset:
                self.state.current_balance = float(balance.get("wb", self.state.current_balance))
                return

    async def _handle_order_trade_update(self, order_update: dict):
        intent = self.state.find_order(
            client_order_id=order_update.get("c", ""),
            order_id=str(order_update.get("i", "")),
        )
        if intent is None:
            self.logger.log(
                "crypto_unmatched_order_update",
                client_order_id=order_update.get("c", ""),
                order_id=order_update.get("i", ""),
                status=order_update.get("X", ""),
            )
            return

        status = order_update.get("X", "")
        executed_qty = float(order_update.get("z", 0.0))
        avg_price = float(order_update.get("ap", 0.0))
        client_order_id = order_update.get("c", "")

        if client_order_id == intent.entry_client_order_id:
            await self._handle_entry_update(intent, status, executed_qty, avg_price)
        elif client_order_id in {intent.tp_client_order_id, intent.sl_client_order_id}:
            await self._handle_exit_update(intent, status, executed_qty, avg_price, client_order_id)

    async def _handle_algo_update(self, algo_update: dict):
        client_algo_id = algo_update.get("caid", "")
        intent = self.state.find_order(client_order_id=client_algo_id)
        if intent is None:
            self.logger.log(
                "crypto_unmatched_algo_update",
                client_algo_id=client_algo_id,
                algo_id=algo_update.get("aid", ""),
                status=algo_update.get("X", ""),
            )
            return

        status = algo_update.get("X", "")
        intent.sl_order_id = str(algo_update.get("aid", intent.sl_order_id or ""))
        intent.sl_client_order_id = client_algo_id or intent.sl_client_order_id
        self.logger.log(
            "crypto_algo_update",
            group_id=intent.group_id,
            algo_id=intent.sl_order_id,
            status=status,
            order_type=algo_update.get("o", ""),
        )

    async def _handle_entry_update(self, intent: OrderIntent, status: str, executed_qty: float, avg_price: float):
        if status in {"PARTIALLY_FILLED", "FILLED"}:
            intent.filled_qty = executed_qty
            intent.avg_entry_price = avg_price or intent.avg_entry_price or intent.entry_price
            if status == "PARTIALLY_FILLED" and intent.status != "OPEN":
                await self.exec_mgr.cancel_entry_remainder(intent)
                await self.exec_mgr.replace_exits(intent, executed_qty)
                intent.status = "OPEN"
                self._move_pending_to_open(intent)
            elif status == "FILLED":
                if intent.status != "OPEN":
                    await self.exec_mgr.replace_exits(intent, executed_qty)
                    self._move_pending_to_open(intent)
                intent.status = "OPEN"
            self.logger.log(
                "crypto_entry_update",
                group_id=intent.group_id,
                status=status,
                filled_qty=executed_qty,
                avg_entry_price=intent.avg_entry_price,
            )
        elif status in {"CANCELED", "EXPIRED", "REJECTED"}:
            if intent.status == "OPEN":
                return
            intent.status = status
            self._move_pending_to_closed(intent)
            self.logger.log("crypto_entry_closed", group_id=intent.group_id, status=status)
            if status == "REJECTED" and self.telegram.enabled:
                await self.telegram.alert_order_rejected(intent.group_id, "Binance rejected entry order")

    async def _handle_exit_update(
        self,
        intent: OrderIntent,
        status: str,
        executed_qty: float,
        avg_price: float,
        client_order_id: str,
    ):
        if status != "FILLED":
            return
        exit_reason = "TP" if client_order_id == intent.tp_client_order_id else "SL"
        intent.exit_reason = exit_reason
        intent.exit_price = avg_price
        intent.closed_at = datetime.now(self._market_tz).isoformat()
        entry_price = intent.avg_entry_price or intent.entry_price
        price_delta = (avg_price - entry_price) if intent.side == "BUY" else (entry_price - avg_price)
        intent.realized_pnl = round(
            (price_delta * executed_qty) - self._exit_fees(intent, executed_qty, avg_price, exit_reason),
            8,
        )
        intent.status = "CLOSED"
        self.state.realized_pnl += intent.realized_pnl
        self.state.current_balance += intent.realized_pnl
        await self.exec_mgr.cancel_sibling_exit(intent, filled_exit=exit_reason)
        self._move_open_to_closed(intent)
        self.logger.log(
            "crypto_position_closed",
            group_id=intent.group_id,
            exit_reason=exit_reason,
            realized_pnl=intent.realized_pnl,
            current_balance=self.state.current_balance,
        )

    def _exit_fees(self, intent: OrderIntent, qty: float, exit_price: float, exit_reason: str) -> float:
        entry_price = intent.avg_entry_price or intent.entry_price
        entry_fee = entry_price * qty * self.config.maker_fee
        exit_fee_rate = self.config.tp_fee if exit_reason == "TP" else self.config.stop_fee
        exit_fee = exit_price * qty * exit_fee_rate
        return entry_fee + exit_fee

    def _move_pending_to_open(self, intent: OrderIntent):
        self.state.pending_entries = [i for i in self.state.pending_entries if i.group_id != intent.group_id]
        if not any(i.group_id == intent.group_id for i in self.state.open_positions):
            self.state.open_positions.append(intent)

    def _move_open_to_closed(self, intent: OrderIntent):
        self.state.open_positions = [i for i in self.state.open_positions if i.group_id != intent.group_id]
        if not any(i.group_id == intent.group_id for i in self.state.closed_trades):
            self.state.closed_trades.append(intent)

    def _move_pending_to_closed(self, intent: OrderIntent):
        self.state.pending_entries = [i for i in self.state.pending_entries if i.group_id != intent.group_id]
        if not any(i.group_id == intent.group_id for i in self.state.closed_trades):
            self.state.closed_trades.append(intent)

    def _current_day(self) -> str:
        return self.clock.now_utc().astimezone(self._reset_tz).strftime("%Y-%m-%d")

    def _roll_day_if_needed(self):
        today = self._current_day()
        if self.state.day != today:
            self.logger.log(
                "crypto_new_day",
                old_day=self.state.day,
                new_day=today,
                end_balance=self.state.current_balance,
            )
            self.state.reset_for_new_day(today)
            self.state_store.save(self.state, force=True)

    async def _periodic_save_loop(self):
        while not self._shutdown:
            await asyncio.sleep(self.config.save_interval_seconds)
            self.state_store.save_if_dirty(self.state)

    async def _periodic_reconcile_loop(self):
        while not self._shutdown:
            await asyncio.sleep(self.config.reconcile_interval_seconds)
            try:
                await self._reconcile_live_state(startup=False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.log("crypto_reconcile_error", error=str(exc))

    async def _handle_stream_failure(self, stream_name: str, exc: Exception):
        self.logger.log("crypto_stream_error", stream=stream_name, error=str(exc))
        if self.telegram.enabled:
            await self.telegram.alert_connection_lost(self.config.stream_reconnect_seconds)
        await asyncio.sleep(self.config.stream_reconnect_seconds)

    async def _reset_user_stream(self):
        self._listen_key_stop.set()
        if self._listen_key:
            with suppress(Exception):
                await self.client.close_listen_key(self._listen_key)
        self._listen_key = None

    async def _shutdown_gracefully(self):
        self._shutdown = True
        self._listen_key_stop.set()
        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            with suppress(Exception):
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
        if self.state is not None:
            self.state_store.save_if_dirty(self.state)
        if self._listen_key:
            with suppress(Exception):
                await self.client.close_listen_key(self._listen_key)
        await self.client.close()
        await self.market_data.close()
        self.logger.close()
