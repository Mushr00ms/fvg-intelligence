"""
ib_adapter.py — BrokerAdapter implementation for Interactive Brokers.

Wraps IBConnection and all IB-specific logic (contract resolution, bar/tick
subscriptions, order placement, account queries, margin probes) into the
broker-agnostic BrokerAdapter interface.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

import pytz

from bot.execution.broker_adapter import (
    BarData,
    BracketOrderResult,
    BrokerAdapter,
    ContractInfo,
    TickData,
)
from bot.execution.execution_types import (
    OpenOrderSnapshot,
    PositionSnapshot,
)
from bot.execution.ib_connection import IBConnection

logger = logging.getLogger(__name__)
NY_TZ = pytz.timezone("America/New_York")
CME_TZ = pytz.timezone("America/Chicago")


def _bar_date_to_et(bar_date):
    """Convert IB bar date (naive, in CME/Central time) to ET datetime."""
    if bar_date is None:
        return bar_date
    if hasattr(bar_date, 'tzinfo') and bar_date.tzinfo is not None:
        return bar_date.astimezone(NY_TZ)
    return CME_TZ.localize(bar_date).astimezone(NY_TZ)


def _tick_time_to_et(tick_time):
    """Convert IB tick timestamp (naive UTC) to ET datetime."""
    if tick_time is None:
        return None
    if hasattr(tick_time, 'tzinfo') and tick_time.tzinfo is not None:
        return tick_time.astimezone(NY_TZ)
    return pytz.utc.localize(tick_time).astimezone(NY_TZ)


class IBAdapter(BrokerAdapter):
    """Interactive Brokers adapter via ib_async."""

    def __init__(self, config, bot_logger=None, clock=None):
        self._config = config
        self._bot_logger = bot_logger
        self._clock = clock
        self._conn = IBConnection(
            host=config.ib_host,
            port=config.ib_port,
            client_id=config.ib_client_id,
            logger=bot_logger,
            clock=clock,
        )
        # Subscription tracking
        self._bar_subs: Dict[str, _IBBarSub] = {}
        self._tick_sub = None
        self._tick_callback = None
        self._next_sub_id = 1
        # Reconnect
        self._reconnect_callbacks: List[Callable] = []
        # IB error handler ref (prevent GC)
        self._ib_error_handler = None

    @property
    def ib_connection(self) -> IBConnection:
        """Expose underlying IBConnection for code still in transition."""
        return self._conn

    # ── Connection ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        await self._conn.connect()
        self._register_ib_error_handler()
        self._conn.on_reconnect(self._on_ib_reconnect)

    async def disconnect(self) -> None:
        # Cancel tick subscription
        if self._tick_sub is not None and self._conn.is_connected:
            try:
                self._conn.ib.cancelTickByTickData(self._tick_sub)
            except Exception:
                pass
            self._tick_sub = None
        await self._conn.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._conn.is_connected

    @property
    def disconnect_seconds(self) -> float:
        return self._conn.disconnect_seconds

    def on_reconnect(self, callback: Callable[[], Awaitable[None]]) -> None:
        self._reconnect_callbacks.append(callback)

    async def _on_ib_reconnect(self) -> None:
        for cb in self._reconnect_callbacks:
            await cb()

    # ── Contract Resolution ─────────────────────────────────────────────

    async def resolve_contract(
        self,
        symbol: str,
        exchange: str,
        expiry_hint: Optional[datetime] = None,
    ) -> ContractInfo:
        from logic.utils.contract_utils import (
            generate_nq_expirations,
            get_contract_for_date,
        )
        from ib_async import Future

        now = expiry_hint or (self._clock.now() if self._clock else datetime.now(NY_TZ))
        expirations = generate_nq_expirations(now.year - 1, now.year + 1)
        exp_date = get_contract_for_date(now, expirations, roll_days=8)

        ib = self._conn.ib
        currency = self._config.currency

        # Suppress IB error 200 during contract probing
        _suppress_200 = [True]
        def _quiet_error(reqId, errorCode, errorString, contract):
            if errorCode == 200 and _suppress_200[0]:
                return
        ib.errorEvent += _quiet_error

        try:
            for date_fmt in [exp_date.strftime("%Y%m%d"), exp_date.strftime("%Y%m")]:
                contract = Future(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=date_fmt,
                    exchange=exchange,
                    currency=currency,
                )
                qualified = await ib.qualifyContractsAsync(contract)
                if qualified and contract.conId > 0:
                    if self._bot_logger:
                        self._bot_logger.log(
                            "contract_resolved",
                            symbol=contract.symbol,
                            expiry=date_fmt,
                            conId=contract.conId,
                        )
                    return ContractInfo(
                        symbol=symbol,
                        broker_contract_id=str(contract.conId),
                        expiry=date_fmt,
                        exchange=exchange,
                        tick_size=self._config.tick_size,
                        point_value=self._config.point_value,
                    )
        finally:
            _suppress_200[0] = False
            ib.errorEvent -= _quiet_error

        raise RuntimeError(f"Failed to qualify {symbol} contract for expiry {exp_date}")

    # ── Market Data ─────────────────────────────────────────────────────

    async def subscribe_bars(
        self,
        contract: ContractInfo,
        bar_size: str,
        on_bar: Callable[[BarData, bool], Any],
    ) -> str:
        ib = self._conn.ib
        # Resolve the IB contract object from conId
        ib_contract = await self._get_ib_contract(contract)

        bars = await ib.reqHistoricalDataAsync(
            ib_contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            keepUpToDate=True,
        )

        sub_id = f"ib_bars_{self._next_sub_id}"
        self._next_sub_id += 1

        sub = _IBBarSub(
            sub_id=sub_id,
            bars=bars,
            on_bar=on_bar,
            ib_contract=ib_contract,
        )
        self._bar_subs[sub_id] = sub

        def _on_bars_update(bars, has_new_bar):
            if not bars:
                return
            bar = bars[-1]
            bar_et = _bar_date_to_et(bar.date)
            bd = BarData(
                timestamp=bar_et,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=getattr(bar, 'volume', 0),
            )
            on_bar(bd, has_new_bar)

        bars.updateEvent += _on_bars_update
        sub.update_handler = _on_bars_update

        if self._bot_logger:
            self._bot_logger.log("bars_subscribed", bar_size=bar_size, sub_id=sub_id)

        return sub_id

    async def unsubscribe_bars(self, subscription_id: str) -> None:
        sub = self._bar_subs.pop(subscription_id, None)
        if sub and sub.bars and sub.update_handler:
            sub.bars.updateEvent -= sub.update_handler

    async def subscribe_ticks(
        self,
        contract: ContractInfo,
        on_tick: Callable[[TickData], Any],
    ) -> Optional[str]:
        try:
            ib = self._conn.ib
            ib_contract = await self._get_ib_contract(contract)
            self._tick_sub = ib.reqTickByTickData(
                ib_contract, tickType='AllLast',
                numberOfTicks=0, ignoreSize=True,
            )
            self._tick_callback = on_tick

            def _on_tick_update(ticker):
                if not hasattr(ticker, 'tickByTicks') or not ticker.tickByTicks:
                    return
                for tick in ticker.tickByTicks:
                    ts = _tick_time_to_et(tick.time)
                    on_tick(TickData(timestamp=ts, price=tick.price, size=float(tick.size)))

            self._tick_sub.updateEvent += _on_tick_update
            if self._bot_logger:
                self._bot_logger.log("ticks_subscribed")
            return "ib_ticks"
        except Exception as e:
            if self._bot_logger:
                self._bot_logger.log("ticks_subscribe_failed", error=str(e))
            return None

    async def unsubscribe_ticks(self, subscription_id: str) -> None:
        if self._tick_sub is not None and self._conn.is_connected:
            try:
                self._conn.ib.cancelTickByTickData(self._tick_sub)
            except Exception:
                pass
            self._tick_sub = None

    # ── Order Management ────────────────────────────────────────────────

    async def place_bracket_order(
        self,
        contract: ContractInfo,
        side: str,
        qty: int,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        on_entry_fill: Callable,
        on_tp_fill: Callable,
        on_sl_fill: Callable,
        on_status_change: Callable,
    ) -> BracketOrderResult:
        from ib_async import LimitOrder, StopOrder

        ib = self._conn.ib
        ib_contract = await self._get_ib_contract(contract)
        reverse_side = "SELL" if side == "BUY" else "BUY"

        entry_id = ib.client.getReqId()
        tp_id = ib.client.getReqId()
        sl_id = ib.client.getReqId()

        parent = LimitOrder(
            action=side, totalQuantity=qty, lmtPrice=entry_price,
            orderId=entry_id, tif='DAY', transmit=False,
        )
        tp = LimitOrder(
            action=reverse_side, totalQuantity=qty, lmtPrice=tp_price,
            orderId=tp_id, parentId=entry_id, tif='DAY', transmit=False,
        )
        sl = StopOrder(
            action=reverse_side, totalQuantity=qty, stopPrice=sl_price,
            orderId=sl_id, parentId=entry_id, tif='DAY', transmit=True,
        )

        loop = asyncio.get_event_loop()

        def _safe(fn, *args):
            if loop.is_running():
                loop.call_soon_threadsafe(fn, *args)
            else:
                fn(*args)

        try:
            entry_trade = ib.placeOrder(ib_contract, parent)
            tp_trade = ib.placeOrder(ib_contract, tp)
            sl_trade = ib.placeOrder(ib_contract, sl)
        except Exception as e:
            for order in (parent, tp, sl):
                try:
                    if order.orderId in {t.order.orderId for t in ib.openTrades()}:
                        ib.cancelOrder(order)
                except Exception:
                    pass
            return BracketOrderResult(success=False, error=str(e))

        entry_trade.filledEvent += lambda trade: _safe(on_entry_fill, trade)
        entry_trade.statusEvent += lambda trade: _safe(on_status_change, trade)
        tp_trade.filledEvent += lambda trade: _safe(on_tp_fill, trade)
        sl_trade.filledEvent += lambda trade: _safe(on_sl_fill, trade)

        return BracketOrderResult(
            success=True,
            entry_order_id=str(entry_id),
            tp_order_id=str(tp_id),
            sl_order_id=str(sl_id),
        )

    async def cancel_order(self, order_id: str) -> None:
        ib = self._conn.ib
        ib_id = int(order_id)
        for trade in ib.openTrades():
            if trade.order.orderId == ib_id:
                ib.cancelOrder(trade.order)
                return

    async def modify_order_qty(self, order_id: str, new_qty: int) -> None:
        ib = self._conn.ib
        ib_id = int(order_id)
        ib_contract = None
        for trade in ib.openTrades():
            if trade.order.orderId == ib_id:
                trade.order.totalQuantity = new_qty
                ib.placeOrder(trade.contract, trade.order)
                return

    async def place_market_order(
        self, contract: ContractInfo, side: str, qty: int
    ) -> str:
        from ib_async import MarketOrder
        ib = self._conn.ib
        ib_contract = await self._get_ib_contract(contract)
        order = MarketOrder(action=side, totalQuantity=qty)
        trade = ib.placeOrder(ib_contract, order)
        return str(trade.order.orderId)

    async def get_open_trades(self) -> List[OpenOrderSnapshot]:
        return await self.get_open_orders()

    # ── Account Data ────────────────────────────────────────────────────

    async def get_account_balance(self) -> Optional[float]:
        if not self._conn.is_connected:
            return None
        ib = self._conn.ib
        for tag in ["AvailableFunds", "TotalCashValue", "NetLiquidation"]:
            for av in ib.accountValues():
                if av.tag == tag and av.currency == "USD":
                    return float(av.value)
        return None

    async def get_positions(self) -> List[PositionSnapshot]:
        if not self._conn.is_connected:
            return []
        result = []
        for pos in self._conn.ib.positions():
            result.append(PositionSnapshot(
                broker="ib",
                symbol=pos.contract.symbol if hasattr(pos.contract, 'symbol') else "",
                side="BUY" if pos.position > 0 else "SELL",
                quantity=abs(int(pos.position)),
                entry_price=pos.avgCost / (self._config.point_value or 20.0),
                raw={"position": pos.position, "avgCost": pos.avgCost},
            ))
        return result

    async def get_open_orders(self) -> List[OpenOrderSnapshot]:
        if not self._conn.is_connected:
            return []
        result = []
        for order in self._conn.ib.openOrders():
            result.append(OpenOrderSnapshot(
                broker="ib",
                symbol=getattr(order, 'symbol', ''),
                side=getattr(order, 'action', ''),
                order_type=getattr(order, 'orderType', ''),
                status="Working",
                quantity=getattr(order, 'totalQuantity', 0),
                price=getattr(order, 'lmtPrice', 0.0),
                order_id=str(getattr(order, 'orderId', '')),
            ))
        return result

    async def get_margin_per_contract(self, contract: ContractInfo) -> Optional[float]:
        """Attempt IB whatIfOrder margin probe. Returns None on failure."""
        if not self._conn.is_connected:
            return None
        try:
            from ib_async import LimitOrder
            ib = self._conn.ib
            ib_contract = await self._get_ib_contract(contract)

            # Get a price reference for the limit order probe
            price = None
            for ticker in ib.tickers():
                if hasattr(ticker, 'last') and ticker.last and ticker.last > 0:
                    price = ticker.last
                    break
            if price is None:
                return None

            probe = LimitOrder(action="BUY", totalQuantity=1, lmtPrice=price, tif="DAY")
            what_if = await ib.whatIfOrderAsync(ib_contract, probe)
            if what_if and hasattr(what_if, 'initMarginChange'):
                margin = float(what_if.initMarginChange)
                if 0 < margin < 1e9:
                    return margin
        except Exception:
            pass
        return None

    async def get_available_funds(self) -> float:
        balance = await self.get_account_balance()
        return balance or 0.0

    # ── Time Sync ───────────────────────────────────────────────────────

    async def get_server_time(self) -> Optional[datetime]:
        if not self._conn.is_connected:
            return None
        try:
            return await self._conn.ib.reqCurrentTimeAsync()
        except Exception:
            return None

    # ── Order ID Allocation ─────────────────────────────────────────────

    def allocate_order_ids(self, count: int) -> List[str]:
        ib = self._conn.ib
        return [str(ib.client.getReqId()) for _ in range(count)]

    # ── IB-Specific Helpers ─────────────────────────────────────────────

    async def _get_ib_contract(self, contract_info: ContractInfo):
        """Resolve a ContractInfo back to an ib_async Future object."""
        from ib_async import Future
        ib = self._conn.ib

        # Try conId first (fastest)
        con_id = int(contract_info.broker_contract_id)
        contract = Future(conId=con_id, exchange=contract_info.exchange)
        qualified = await ib.qualifyContractsAsync(contract)
        if qualified and contract.conId > 0:
            return contract

        # Fallback: resolve by symbol + expiry
        contract = Future(
            symbol=contract_info.symbol,
            lastTradeDateOrContractMonth=contract_info.expiry,
            exchange=contract_info.exchange,
            currency=self._config.currency,
        )
        qualified = await ib.qualifyContractsAsync(contract)
        if qualified and contract.conId > 0:
            return contract

        raise RuntimeError(
            f"Cannot resolve IB contract for {contract_info.symbol} "
            f"(conId={contract_info.broker_contract_id})"
        )

    def _register_ib_error_handler(self):
        """Register IB error handler for known benign codes."""
        import logging as _logging

        ib = self._conn.ib
        bot_logger = self._bot_logger

        _HANDLED_CODES = {
            202: "oca_cancel",
            399: "ib_reprice",
        }

        def _on_ib_error(reqId, errorCode, errorString, contract):
            event_name = _HANDLED_CODES.get(errorCode)
            if event_name and bot_logger:
                bot_logger.log(event_name, reqId=reqId, code=errorCode,
                               msg=errorString.strip())

        ib.errorEvent += _on_ib_error
        self._ib_error_handler = _on_ib_error

        class _BenignCodeFilter(_logging.Filter):
            def filter(self, record):
                msg = record.getMessage()
                for code in _HANDLED_CODES:
                    if f"Error {code}" in msg or f"Warning {code}" in msg:
                        return False
                return True

        for name in ("ib_async.ib", "ib_async.wrapper", "ib_async.client", "ib_async"):
            _logging.getLogger(name).addFilter(_BenignCodeFilter())

    def get_raw_ib_positions(self):
        """Return raw IB position objects for state reconciliation."""
        if not self._conn.is_connected:
            return []
        return self._conn.ib.positions()

    def get_raw_ib_open_orders(self):
        """Return raw IB open order objects for state reconciliation."""
        if not self._conn.is_connected:
            return []
        return self._conn.ib.openOrders()


class _IBBarSub:
    """Tracks an active IB bar subscription."""
    __slots__ = ("sub_id", "bars", "on_bar", "ib_contract", "update_handler")

    def __init__(self, sub_id, bars, on_bar, ib_contract, update_handler=None):
        self.sub_id = sub_id
        self.bars = bars
        self.on_bar = on_bar
        self.ib_contract = ib_contract
        self.update_handler = update_handler
