"""
trade_state.py — Dataclasses for all bot state: FVGs, orders, positions, daily state.

All classes have to_dict() / from_dict() for JSON serialization.
No external dependencies — pure stdlib.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional
import uuid


def _now_iso(clock=None):
    """Get current timestamp as ISO string. Uses clock if provided, else system time."""
    if clock is not None:
        return clock.now().isoformat()
    return datetime.now().isoformat()


def _new_id():
    return uuid.uuid4().hex[:12]


# State schema version — increment on breaking changes, add migration in state_manager.py
STATE_VERSION = "1.1"


# ---------------------------------------------------------------------------
# FVG Record
# ---------------------------------------------------------------------------

@dataclass
class FVGRecord:
    """A detected Fair Value Gap tracked by the bot."""
    fvg_id: str
    fvg_type: str                   # "bullish" | "bearish"
    zone_low: float
    zone_high: float
    time_candle1: str               # ISO datetime string
    time_candle2: str
    time_candle3: str
    middle_open: float
    middle_low: float
    middle_high: float
    first_open: float
    time_period: str                # "10:30-11:00"
    formation_date: str             # "2026-03-22"
    is_mitigated: bool = False
    mitigation_time: Optional[str] = None
    orders_placed: list = field(default_factory=list)

    def to_dict(self):
        return {
            "fvg_id": self.fvg_id,
            "fvg_type": self.fvg_type,
            "zone_low": self.zone_low,
            "zone_high": self.zone_high,
            "time_candle1": self.time_candle1,
            "time_candle2": self.time_candle2,
            "time_candle3": self.time_candle3,
            "middle_open": self.middle_open,
            "middle_low": self.middle_low,
            "middle_high": self.middle_high,
            "first_open": self.first_open,
            "time_period": self.time_period,
            "formation_date": self.formation_date,
            "is_mitigated": self.is_mitigated,
            "mitigation_time": self.mitigation_time,
            "orders_placed": list(self.orders_placed),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            fvg_id=d["fvg_id"],
            fvg_type=d["fvg_type"],
            zone_low=d["zone_low"],
            zone_high=d["zone_high"],
            time_candle1=d["time_candle1"],
            time_candle2=d["time_candle2"],
            time_candle3=d["time_candle3"],
            middle_open=d["middle_open"],
            middle_low=d["middle_low"],
            middle_high=d["middle_high"],
            first_open=d["first_open"],
            time_period=d["time_period"],
            formation_date=d["formation_date"],
            is_mitigated=d.get("is_mitigated", False),
            mitigation_time=d.get("mitigation_time"),
            orders_placed=d.get("orders_placed", []),
        )


# ---------------------------------------------------------------------------
# Order Group (bracket: entry + TP + SL)
# ---------------------------------------------------------------------------

# State machine: PENDING → SUBMITTED → PARTIAL → FILLED → CLOSED
#                               ↓                          ↑
#                          SUSPENDED ─── (reactivate) ─────┘
ORDER_STATES = ("PENDING", "SUBMITTED", "PARTIAL", "FILLED", "SUSPENDED", "CLOSED")

# Close reasons
CLOSE_TP = "TP"
CLOSE_SL = "SL"
CLOSE_FLATTEN = "FLATTEN"
CLOSE_CANCEL = "CANCEL"
CLOSE_EOD = "EOD"
CLOSE_REJECTED = "REJECTED"
CLOSE_MARGIN_SUSPEND = "MARGIN_SUSPEND"


@dataclass
class OrderGroup:
    """A bracket order group: entry + take profit + stop loss."""
    group_id: str
    fvg_id: str
    setup: str                      # "mit_extreme" | "mid_extreme"
    side: str                       # "BUY" | "SELL"
    entry_price: float
    stop_price: float
    target_price: float
    risk_pts: float
    n_value: float                  # R:R target (e.g. 2.75)
    target_qty: int                 # Desired number of contracts
    risk_pct: float = 0.01          # Actual risk fraction used (respects 3-tier)
    filled_qty: int = 0
    state: str = "PENDING"
    ib_entry_order_id: Optional[int] = None
    ib_tp_order_id: Optional[int] = None
    ib_sl_order_id: Optional[int] = None
    submitted_at: Optional[str] = None
    filled_at: Optional[str] = None
    closed_at: Optional[str] = None
    close_reason: str = ""
    realized_pnl: float = 0.0
    partial_fill_timer_start: Optional[str] = None
    actual_entry_price: float = 0.0
    actual_exit_price: float = 0.0
    entry_slippage_pts: float = 0.0
    entry_commission: float = 0.0
    suspended_at: Optional[str] = None
    suspend_reason: str = ""

    def to_dict(self):
        return {
            "group_id": self.group_id,
            "fvg_id": self.fvg_id,
            "setup": self.setup,
            "side": self.side,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "risk_pts": self.risk_pts,
            "n_value": self.n_value,
            "target_qty": self.target_qty,
            "risk_pct": self.risk_pct,
            "filled_qty": self.filled_qty,
            "state": self.state,
            "ib_entry_order_id": self.ib_entry_order_id,
            "ib_tp_order_id": self.ib_tp_order_id,
            "ib_sl_order_id": self.ib_sl_order_id,
            "submitted_at": self.submitted_at,
            "filled_at": self.filled_at,
            "closed_at": self.closed_at,
            "close_reason": self.close_reason,
            "realized_pnl": self.realized_pnl,
            "partial_fill_timer_start": self.partial_fill_timer_start,
            "actual_entry_price": self.actual_entry_price,
            "actual_exit_price": self.actual_exit_price,
            "close_price": self.actual_exit_price or None,
            "entry_slippage_pts": self.entry_slippage_pts,
            "entry_commission": self.entry_commission,
            "suspended_at": self.suspended_at,
            "suspend_reason": self.suspend_reason,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            group_id=d["group_id"],
            fvg_id=d["fvg_id"],
            setup=d["setup"],
            side=d["side"],
            entry_price=d["entry_price"],
            stop_price=d["stop_price"],
            target_price=d["target_price"],
            risk_pts=d["risk_pts"],
            n_value=d["n_value"],
            target_qty=d["target_qty"],
            risk_pct=d.get("risk_pct", 0.01),
            filled_qty=d.get("filled_qty", 0),
            state=d.get("state", "PENDING"),
            ib_entry_order_id=d.get("ib_entry_order_id"),
            ib_tp_order_id=d.get("ib_tp_order_id"),
            ib_sl_order_id=d.get("ib_sl_order_id"),
            submitted_at=d.get("submitted_at"),
            filled_at=d.get("filled_at"),
            closed_at=d.get("closed_at"),
            close_reason=d.get("close_reason", ""),
            realized_pnl=d.get("realized_pnl", 0.0),
            partial_fill_timer_start=d.get("partial_fill_timer_start"),
            actual_entry_price=d.get("actual_entry_price", 0.0),
            actual_exit_price=d.get("actual_exit_price", 0.0) or d.get("close_price", 0.0),
            entry_slippage_pts=d.get("entry_slippage_pts", 0.0),
            entry_commission=d.get("entry_commission", 0.0),
            suspended_at=d.get("suspended_at"),
            suspend_reason=d.get("suspend_reason", ""),
        )

    @property
    def is_active(self):
        """True if the order group is in a state that counts toward position limits."""
        return self.state in ("SUBMITTED", "PARTIAL", "FILLED")


# ---------------------------------------------------------------------------
# Daily State
# ---------------------------------------------------------------------------

@dataclass
class DailyState:
    """Full bot state for one trading day."""
    date: str                       # "2026-03-22"
    start_balance: float
    realized_pnl: float = 0.0
    trade_count: int = 0
    kill_switch_active: bool = False
    kill_switch_reason: str = ""
    active_fvgs: list = field(default_factory=list)       # List[FVGRecord]
    pending_orders: list = field(default_factory=list)     # List[OrderGroup] in SUBMITTED/PARTIAL
    open_positions: list = field(default_factory=list)     # List[OrderGroup] in FILLED
    closed_trades: list = field(default_factory=list)      # List[OrderGroup] in CLOSED
    suspended_orders: list = field(default_factory=list)   # List[OrderGroup] in SUSPENDED
    last_updated: str = ""

    def to_dict(self):
        return {
            "version": STATE_VERSION,
            "date": self.date,
            "start_balance": self.start_balance,
            "realized_pnl": self.realized_pnl,
            "trade_count": self.trade_count,
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
            "active_fvgs": [f.to_dict() for f in self.active_fvgs],
            "pending_orders": [o.to_dict() for o in self.pending_orders],
            "open_positions": [o.to_dict() for o in self.open_positions],
            "closed_trades": [o.to_dict() for o in self.closed_trades],
            "suspended_orders": [o.to_dict() for o in self.suspended_orders],
            "last_updated": self.last_updated or _now_iso(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            date=d["date"],
            start_balance=d["start_balance"],
            realized_pnl=d.get("realized_pnl", 0.0),
            trade_count=d.get("trade_count", 0),
            kill_switch_active=d.get("kill_switch_active", False),
            kill_switch_reason=d.get("kill_switch_reason", ""),
            active_fvgs=[FVGRecord.from_dict(f) for f in d.get("active_fvgs", [])],
            pending_orders=[OrderGroup.from_dict(o) for o in d.get("pending_orders", [])],
            open_positions=[OrderGroup.from_dict(o) for o in d.get("open_positions", [])],
            closed_trades=[OrderGroup.from_dict(o) for o in d.get("closed_trades", [])],
            suspended_orders=[OrderGroup.from_dict(o) for o in d.get("suspended_orders", [])],
            last_updated=d.get("last_updated", ""),
        )

    @property
    def active_order_count(self):
        """Number of positions counting toward the concurrent max."""
        return len(self.pending_orders) + len(self.open_positions)

    @property
    def filled_trade_count(self):
        """Number of trades that actually filled (excludes unfilled cancels)."""
        unfilled_cancels = sum(
            1 for t in self.closed_trades
            if t.close_reason in (CLOSE_CANCEL, CLOSE_EOD, CLOSE_REJECTED)
            and t.filled_qty == 0
        )
        return len(self.closed_trades) - unfilled_cancels

    @property
    def daily_pnl_pct(self):
        """Realized P&L as percentage of start balance."""
        if self.start_balance <= 0:
            return 0.0
        return self.realized_pnl / self.start_balance

    def find_order_by_ib_id(self, ib_order_id):
        """Find an OrderGroup by any of its IB order IDs."""
        for og in self.pending_orders + self.open_positions:
            if ib_order_id in (
                og.ib_entry_order_id, og.ib_tp_order_id, og.ib_sl_order_id
            ):
                return og
        return None

    def move_to_open(self, group_id):
        """Move an order group from pending to open positions."""
        for i, og in enumerate(self.pending_orders):
            if og.group_id == group_id:
                og.state = "FILLED"
                self.open_positions.append(self.pending_orders.pop(i))
                return og
        return None

    def move_to_closed(self, group_id, reason, pnl=0.0):
        """Move an order group to closed trades."""
        for lst in (self.pending_orders, self.open_positions, self.suspended_orders):
            for i, og in enumerate(lst):
                if og.group_id == group_id:
                    og.state = "CLOSED"
                    og.close_reason = reason
                    og.closed_at = _now_iso()
                    og.realized_pnl = pnl
                    self.closed_trades.append(lst.pop(i))
                    self.realized_pnl += pnl
                    return og
        return None

    def move_to_suspended(self, group_id, reason=""):
        """Move a SUBMITTED order from pending_orders to suspended_orders."""
        for i, og in enumerate(self.pending_orders):
            if og.group_id == group_id:
                og.state = "SUSPENDED"
                og.suspended_at = _now_iso()
                og.suspend_reason = reason
                self.suspended_orders.append(self.pending_orders.pop(i))
                return og
        return None

    def move_suspended_to_pending(self, group_id):
        """Pop a suspended order for re-placement. Caller must re-place via place_bracket."""
        for i, og in enumerate(self.suspended_orders):
            if og.group_id == group_id:
                og.state = "PENDING"
                og.suspended_at = None
                og.suspend_reason = ""
                return self.suspended_orders.pop(i)
        return None
