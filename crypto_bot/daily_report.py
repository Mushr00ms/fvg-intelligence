"""
daily_report.py - Daily Telegram reconciliation formatting for the crypto bot.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from statistics import mean
from zoneinfo import ZoneInfo

from crypto_bot.models import OrderIntent, RuntimeState


def build_daily_reconciliation_report(state: RuntimeState, *, mode: str, report_tz: str) -> str:
    tz = ZoneInfo(report_tz)
    report_day = state.day

    filled_trades = sorted(
        _filter_by_day(state.closed_trades + state.open_positions, "opened_at", report_day, tz),
        key=_sort_key("opened_at"),
    )
    closed_trades = sorted(
        [
            trade for trade in _filter_by_day(state.closed_trades, "closed_at", report_day, tz)
            if trade.exit_reason
        ],
        key=_sort_key("closed_at"),
    )
    rejected_or_canceled = sorted(
        [
            trade for trade in _filter_by_day(state.closed_trades, "created_at", report_day, tz)
            if trade.status in {"REJECTED", "CANCELED", "EXPIRED"}
        ],
        key=_sort_key("created_at"),
    )
    open_carry = sorted(state.open_positions, key=_sort_key("opened_at"))
    pending_carry = sorted(state.pending_entries, key=_sort_key("created_at"))

    wins = [trade for trade in closed_trades if trade.realized_pnl > 0]
    losses = [trade for trade in closed_trades if trade.realized_pnl < 0]
    gross_profit = sum(trade.realized_pnl for trade in wins)
    gross_loss = sum(trade.realized_pnl for trade in losses)
    trade_count = len(closed_trades)
    win_rate = (len(wins) / trade_count * 100.0) if trade_count else 0.0
    pnl_pct = (state.realized_pnl / state.start_balance * 100.0) if state.start_balance else 0.0
    avg_win = mean([trade.realized_pnl for trade in wins]) if wins else 0.0
    avg_loss = mean([trade.realized_pnl for trade in losses]) if losses else 0.0

    lines = [
        "<b>CRYPTO DAILY RECONCILIATION</b>",
        "",
        f"Date: {escape(report_day)}",
        f"Mode: {escape(mode.upper())}",
        f"Symbol: {escape(state.symbol)}",
        f"Strategy: <code>{escape(state.strategy_id)}</code>",
        f"P&amp;L: {_fmt_signed_money(state.realized_pnl)} ({_fmt_signed_pct(pnl_pct)})",
        f"Balance: {_fmt_money(state.start_balance)} -> {_fmt_money(state.current_balance)}",
        (
            "Trades taken: "
            f"{len(filled_trades)} filled / {len(closed_trades)} closed / {len(open_carry)} open"
        ),
        f"Setups submitted: {state.trade_count}",
        f"Wins-Losses: {len(wins)}-{len(losses)} ({win_rate:.1f}%)",
        f"Avg win-loss: {_fmt_signed_money(avg_win)} / {_fmt_signed_money(avg_loss)}",
        f"Rejected-canceled: {len(rejected_or_canceled)}",
    ]

    if gross_profit or gross_loss:
        lines.append(
            f"Gross: {_fmt_signed_money(gross_profit)} / {_fmt_signed_money(gross_loss)}"
        )

    if closed_trades:
        lines.extend(["", "<b>Closed Trades</b>"])
        for index, trade in enumerate(closed_trades[:12], start=1):
            lines.append(_format_closed_trade(index, trade, tz))
        if len(closed_trades) > 12:
            lines.append(f"... {len(closed_trades) - 12} more closed trades")

    if open_carry:
        lines.extend(["", "<b>Open Carry</b>"])
        for index, trade in enumerate(open_carry[:8], start=1):
            lines.append(_format_open_trade(index, trade, tz))
        if len(open_carry) > 8:
            lines.append(f"... {len(open_carry) - 8} more open positions")

    if pending_carry:
        lines.extend(["", "<b>Pending Entries</b>"])
        for index, trade in enumerate(pending_carry[:8], start=1):
            lines.append(_format_pending_trade(index, trade, tz))
        if len(pending_carry) > 8:
            lines.append(f"... {len(pending_carry) - 8} more pending entries")

    if not closed_trades and not open_carry and not pending_carry:
        lines.extend(["", "No trades taken."])

    return "\n".join(lines)


def _filter_by_day(trades: list[OrderIntent], attr: str, report_day: str, tz: ZoneInfo) -> list[OrderIntent]:
    filtered = []
    for trade in trades:
        value = getattr(trade, attr, "")
        if not value:
            continue
        if _to_day(value, tz) == report_day:
            filtered.append(trade)
    return filtered


def _sort_key(attr: str):
    def key(trade: OrderIntent):
        value = getattr(trade, attr, "")
        if not value:
            return datetime.min.replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(value)

    return key


def _to_day(timestamp: str, tz: ZoneInfo) -> str:
    return datetime.fromisoformat(timestamp).astimezone(tz).strftime("%Y-%m-%d")


def _to_hm(timestamp: str, tz: ZoneInfo) -> str:
    return datetime.fromisoformat(timestamp).astimezone(tz).strftime("%H:%M")


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _fmt_signed_money(value: float) -> str:
    if abs(value) < 1e-9:
        return "$0.00"
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.2f}"


def _fmt_signed_pct(value: float) -> str:
    return f"{value:+.2f}%"


def _fmt_price(value: float) -> str:
    return f"{value:,.2f}"


def _format_closed_trade(index: int, trade: OrderIntent, tz: ZoneInfo) -> str:
    opened = _to_hm(trade.opened_at or trade.created_at, tz)
    closed = _to_hm(trade.closed_at, tz)
    return (
        f"{index}. {opened}-{closed} {escape(trade.side)} {escape(trade.setup)} "
        f"{trade.risk_bps:.0f}bps n={trade.n_value:.2f} -> {escape(trade.exit_reason)} "
        f"{_fmt_signed_money(trade.realized_pnl)}"
    )


def _format_open_trade(index: int, trade: OrderIntent, tz: ZoneInfo) -> str:
    opened = _to_hm(trade.opened_at or trade.created_at, tz)
    return (
        f"{index}. {opened} {escape(trade.side)} {escape(trade.setup)} "
        f"entry {_fmt_price(trade.entry_price)} stop {_fmt_price(trade.stop_price)} "
        f"target {_fmt_price(trade.target_price)}"
    )


def _format_pending_trade(index: int, trade: OrderIntent, tz: ZoneInfo) -> str:
    created = _to_hm(trade.created_at, tz)
    return (
        f"{index}. {created} {escape(trade.side)} {escape(trade.setup)} "
        f"entry {_fmt_price(trade.entry_price)} stop {_fmt_price(trade.stop_price)} "
        f"target {_fmt_price(trade.target_price)}"
    )
