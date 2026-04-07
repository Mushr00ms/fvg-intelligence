"""
risk.py — BTCUSDT sizing and risk checks for the crypto bot.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from bot.execution.execution_types import AccountSnapshot, PositionSnapshot, SymbolRules

from crypto_bot.models import FVGRecord, OrderIntent, RuntimeState, _new_id
from crypto_bot.strategy import BTCStrategyLoader


RUNTIME_TZ = ZoneInfo("America/New_York")


class CryptoRiskManager:
    def __init__(self, config, symbol_rules: SymbolRules, strategy: BTCStrategyLoader):
        self._config = config
        self._rules = symbol_rules
        self._strategy = strategy

    def evaluate_daily_halt(self, state: RuntimeState):
        if state.start_balance <= 0:
            return False
        if state.daily_loss_halt:
            return True
        pnl_pct = state.realized_pnl / state.start_balance
        if pnl_pct <= -abs(self._config.max_daily_loss_pct):
            state.daily_loss_halt = True
            state.daily_loss_reason = (
                f"Realized daily PnL {pnl_pct*100:.1f}% breached "
                f"-{abs(self._config.max_daily_loss_pct)*100:.1f}%"
            )
            return True
        return False

    def build_intent(self, fvg: FVGRecord, setup: str, *, balance: float) -> OrderIntent | None:
        if setup in fvg.orders_placed:
            return None

        if fvg.fvg_type == "bullish":
            entry_price = fvg.zone_high if setup == "mit_extreme" else (fvg.zone_low + fvg.zone_high) / 2
            stop_price = fvg.middle_low
            side = "BUY"
        else:
            entry_price = fvg.zone_low if setup == "mit_extreme" else (fvg.zone_low + fvg.zone_high) / 2
            stop_price = fvg.middle_high
            side = "SELL"

        risk_bps = abs(entry_price - stop_price) / fvg.reference_price * 10_000
        cell = self._strategy.find_cell(fvg.time_period, risk_bps, setup)
        if cell is None:
            return None

        entry_price = self._rules.round_price(entry_price)
        stop_price = self._rules.round_price(stop_price)
        if side == "BUY":
            target_price = self._rules.round_price(entry_price + cell["best_n"] * abs(entry_price - stop_price))
        else:
            target_price = self._rules.round_price(entry_price - cell["best_n"] * abs(entry_price - stop_price))

        risk_dollar = balance * self._config.risk_per_trade
        per_unit_loss = abs(entry_price - stop_price) + (entry_price * self._config.maker_fee) + (stop_price * self._config.stop_fee)
        if per_unit_loss <= 0:
            return None

        quantity = self._rules.clamp_quantity(risk_dollar / per_unit_loss)
        if quantity <= 0:
            return None
        if not self._rules.validate_notional(quantity, entry_price):
            return None

        notional = quantity * entry_price
        initial_margin = notional / self._config.leverage
        expected_loss = quantity * per_unit_loss
        per_unit_profit = abs(target_price - entry_price) - (entry_price * self._config.maker_fee) - (target_price * self._config.tp_fee)
        expected_profit = quantity * max(per_unit_profit, 0.0)

        return OrderIntent(
            group_id=_new_id(),
            fvg_id=fvg.fvg_id,
            symbol=self._config.symbol,
            setup=setup,
            side=side,
            position_side=self._position_side_for(side),
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            risk_bps=round(risk_bps, 3),
            n_value=cell["best_n"],
            risk_dollar=round(risk_dollar, 8),
            quantity=quantity,
            notional=round(notional, 8),
            initial_margin=round(initial_margin, 8),
            expected_loss=round(expected_loss, 8),
            expected_profit=round(expected_profit, 8),
            created_at=datetime.now(RUNTIME_TZ).isoformat(),
        )

    def can_accept(
        self,
        state: RuntimeState,
        intent: OrderIntent,
        *,
        available_balance: float,
        account_snapshot: AccountSnapshot | None = None,
        exchange_positions: list[PositionSnapshot] | None = None,
    ) -> str | None:
        if self.evaluate_daily_halt(state):
            return state.daily_loss_reason

        if self._config.position_mode == "ONE_WAY":
            active_sides = {
                item.side
                for item in state.pending_entries + state.open_positions
                if item.status not in {"CLOSED", "CANCELED", "EXPIRED", "REJECTED"}
            }
            if active_sides and intent.side not in active_sides:
                return "ONE_WAY mode blocks opposite-side exposure while orders/positions are active"

        if state.active_order_count >= self._config.max_concurrent:
            return f"max_concurrent reached ({state.active_order_count}/{self._config.max_concurrent})"

        # Capital-cap notional check: open notional must fit within equity * leverage.
        # Backtest enforces this strictly; without it the bot will take more positions
        # than the backtest and the realized P&L distribution will diverge.
        open_notional = sum(i.notional for i in state.pending_entries + state.open_positions)
        notional_cap = available_balance * self._config.leverage
        if open_notional + intent.notional > notional_cap:
            return (
                f"capital-cap reject: open_notional={open_notional:.2f} + new={intent.notional:.2f} "
                f"exceeds cap {notional_cap:.2f} (equity={available_balance:.2f} × lev={self._config.leverage})"
            )

        open_risk = sum(i.expected_loss for i in state.pending_entries + state.open_positions)
        max_open_risk = available_balance * self._config.max_cumulative_risk_pct
        if open_risk + intent.expected_loss > max_open_risk * 1.01:
            return (
                f"cumulative risk {open_risk + intent.expected_loss:.2f} exceeds "
                f"{self._config.max_cumulative_risk_pct*100:.1f}% of available balance"
            )

        reserved_margin = sum(i.initial_margin for i in state.pending_entries + state.open_positions)
        if intent.initial_margin > max(0.0, available_balance - reserved_margin):
            return (
                f"insufficient free margin for new intent "
                f"(need {intent.initial_margin:.2f}, free {max(0.0, available_balance - reserved_margin):.2f})"
            )

        if account_snapshot is not None:
            total_initial_margin = account_snapshot.initial_margin + reserved_margin + intent.initial_margin
            margin_balance = max(account_snapshot.margin_balance, account_snapshot.wallet_balance, 0.0)
            if margin_balance > 0:
                margin_usage = total_initial_margin / margin_balance
                if margin_usage > self._config.max_margin_usage_pct:
                    return (
                        f"margin usage {margin_usage*100:.1f}% exceeds "
                        f"{self._config.max_margin_usage_pct*100:.1f}% cap"
                    )

        if exchange_positions:
            liquidation_risk = self._worst_liquidation_buffer(exchange_positions)
            if liquidation_risk is not None and liquidation_risk < self._config.min_liquidation_buffer_pct:
                return (
                    f"liquidation buffer {liquidation_risk*100:.2f}% is below "
                    f"{self._config.min_liquidation_buffer_pct*100:.2f}% threshold"
                )

        return None

    def _position_side_for(self, side: str) -> str:
        if self._config.position_mode == "HEDGE":
            return "LONG" if side == "BUY" else "SHORT"
        return "BOTH"

    def consecutive_conflict_reason(
        self,
        active_intents: list[OrderIntent],
        intent: OrderIntent,
    ) -> str | None:
        for existing in active_intents:
            if existing.side != intent.side:
                continue
            if existing.position_side != intent.position_side:
                continue
            if self._same_price(existing.stop_price, intent.entry_price):
                return (
                    f"consecutive_fvg_far_skip existing_group={existing.group_id} "
                    f"existing_stop={existing.stop_price:.8f} == new_entry={intent.entry_price:.8f}"
                )
        return None

    def _same_price(self, left: float, right: float) -> bool:
        tick = self._rules.price_tick_size or 0.0
        tolerance = tick / 2 if tick > 0 else 1e-9
        return abs(left - right) <= tolerance

    @staticmethod
    def _worst_liquidation_buffer(exchange_positions: list[PositionSnapshot]) -> float | None:
        buffers = []
        for pos in exchange_positions:
            if pos.liquidation_price <= 0 or pos.mark_price <= 0:
                continue
            if pos.side == "BUY":
                buffer_pct = (pos.mark_price - pos.liquidation_price) / pos.mark_price
            else:
                buffer_pct = (pos.liquidation_price - pos.mark_price) / pos.mark_price
            buffers.append(buffer_pct)
        if not buffers:
            return None
        return min(buffers)
