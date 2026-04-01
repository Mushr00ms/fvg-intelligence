"""
models.py — Dataclasses for the standalone crypto bot runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


RUNTIME_TZ = ZoneInfo("America/New_York")


def _iso_now() -> str:
    return datetime.now(RUNTIME_TZ).isoformat()


@dataclass
class FVGRecord:
    fvg_id: str
    fvg_type: str
    zone_low: float
    zone_high: float
    time_candle1: str
    time_candle2: str
    time_candle3: str
    reference_price: float
    time_period: str
    formation_time: str
    formation_date: str
    first_open: float
    middle_open: float
    middle_high: float
    middle_low: float
    mitigation_deadline: str
    is_mitigated: bool = False
    mitigation_time: str | None = None
    expired: bool = False
    orders_placed: list[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "fvg_id": self.fvg_id,
            "fvg_type": self.fvg_type,
            "zone_low": self.zone_low,
            "zone_high": self.zone_high,
            "time_candle1": self.time_candle1,
            "time_candle2": self.time_candle2,
            "time_candle3": self.time_candle3,
            "reference_price": self.reference_price,
            "time_period": self.time_period,
            "formation_time": self.formation_time,
            "formation_date": self.formation_date,
            "first_open": self.first_open,
            "middle_open": self.middle_open,
            "middle_high": self.middle_high,
            "middle_low": self.middle_low,
            "mitigation_deadline": self.mitigation_deadline,
            "is_mitigated": self.is_mitigated,
            "mitigation_time": self.mitigation_time,
            "expired": self.expired,
            "orders_placed": list(self.orders_placed),
        }

    @classmethod
    def from_dict(cls, data):
        payload = dict(data)
        fallback_time = payload.get("formation_time", "")
        payload.setdefault("time_candle1", fallback_time)
        payload.setdefault("time_candle2", fallback_time)
        payload.setdefault("time_candle3", fallback_time)
        payload.setdefault("formation_date", fallback_time[:10] if fallback_time else "")
        payload.setdefault("first_open", payload.get("middle_open", 0.0))
        return cls(**payload)


@dataclass
class OrderIntent:
    group_id: str
    fvg_id: str
    symbol: str
    setup: str
    side: str
    position_side: str
    entry_price: float
    stop_price: float
    target_price: float
    risk_bps: float
    n_value: float
    risk_dollar: float
    quantity: float
    notional: float
    initial_margin: float
    expected_loss: float
    expected_profit: float
    created_at: str
    status: str = "PENDING"
    entry_order_id: str = ""
    entry_client_order_id: str = ""
    tp_order_id: str = ""
    tp_client_order_id: str = ""
    sl_order_id: str = ""
    sl_client_order_id: str = ""
    filled_qty: float = 0.0
    avg_entry_price: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    realized_pnl: float = 0.0
    closed_at: str = ""

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, data):
        payload = dict(data)
        payload.setdefault("position_side", "BOTH")
        return cls(**payload)


@dataclass
class RuntimeState:
    version: str
    symbol: str
    strategy_id: str
    day: str
    start_balance: float
    current_balance: float
    realized_pnl: float = 0.0
    trade_count: int = 0
    daily_loss_halt: bool = False
    daily_loss_reason: str = ""
    active_fvgs: list[FVGRecord] = field(default_factory=list)
    pending_entries: list[OrderIntent] = field(default_factory=list)
    open_positions: list[OrderIntent] = field(default_factory=list)
    closed_trades: list[OrderIntent] = field(default_factory=list)
    processed_event_keys: list[str] = field(default_factory=list)
    last_updated: str = field(default_factory=_iso_now)

    def to_dict(self):
        return {
            "version": self.version,
            "symbol": self.symbol,
            "strategy_id": self.strategy_id,
            "day": self.day,
            "start_balance": self.start_balance,
            "current_balance": self.current_balance,
            "realized_pnl": self.realized_pnl,
            "trade_count": self.trade_count,
            "daily_loss_halt": self.daily_loss_halt,
            "daily_loss_reason": self.daily_loss_reason,
            "active_fvgs": [f.to_dict() for f in self.active_fvgs],
            "pending_entries": [o.to_dict() for o in self.pending_entries],
            "open_positions": [o.to_dict() for o in self.open_positions],
            "closed_trades": [o.to_dict() for o in self.closed_trades],
            "processed_event_keys": list(self.processed_event_keys),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            version=data["version"],
            symbol=data["symbol"],
            strategy_id=data["strategy_id"],
            day=data["day"],
            start_balance=data["start_balance"],
            current_balance=data["current_balance"],
            realized_pnl=data.get("realized_pnl", 0.0),
            trade_count=data.get("trade_count", 0),
            daily_loss_halt=data.get("daily_loss_halt", False),
            daily_loss_reason=data.get("daily_loss_reason", ""),
            active_fvgs=[FVGRecord.from_dict(f) for f in data.get("active_fvgs", [])],
            pending_entries=[OrderIntent.from_dict(o) for o in data.get("pending_entries", [])],
            open_positions=[OrderIntent.from_dict(o) for o in data.get("open_positions", [])],
            closed_trades=[OrderIntent.from_dict(o) for o in data.get("closed_trades", [])],
            processed_event_keys=data.get("processed_event_keys", []),
            last_updated=data.get("last_updated", _iso_now()),
        )

    def touch(self):
        self.last_updated = _iso_now()

    def reset_for_new_day(self, day: str):
        self.day = day
        self.start_balance = self.current_balance
        self.realized_pnl = 0.0
        self.trade_count = 0
        self.daily_loss_halt = False
        self.daily_loss_reason = ""
        self.processed_event_keys = []
        self.touch()

    @property
    def active_order_count(self) -> int:
        return len(self.pending_entries) + len(self.open_positions)

    def find_order(self, *, client_order_id: str = "", order_id: str = "") -> OrderIntent | None:
        for intent in self.pending_entries + self.open_positions:
            if client_order_id and client_order_id in {
                intent.entry_client_order_id,
                intent.tp_client_order_id,
                intent.sl_client_order_id,
            }:
                return intent
            if order_id and order_id in {
                intent.entry_order_id,
                intent.tp_order_id,
                intent.sl_order_id,
            }:
                return intent
        return None

    def has_processed_event(self, event_key: str) -> bool:
        return event_key in self.processed_event_keys

    def mark_event_processed(self, event_key: str, *, max_items: int = 2000):
        if event_key in self.processed_event_keys:
            return
        self.processed_event_keys.append(event_key)
        if len(self.processed_event_keys) > max_items:
            self.processed_event_keys = self.processed_event_keys[-max_items:]


def new_runtime_state(symbol: str, strategy_id: str, day: str, balance: float) -> RuntimeState:
    return RuntimeState(
        version="1.0",
        symbol=symbol,
        strategy_id=strategy_id,
        day=day,
        start_balance=balance,
        current_balance=balance,
    )
