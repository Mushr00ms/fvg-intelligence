"""
risk_gates.py — All risk management checks run before every order placement.

Gates are checked as a pipeline: first failure blocks the trade.
"""

from dataclasses import dataclass


@dataclass
class GateResult:
    """Result of a risk gate check."""
    passed: bool
    gate: str = ""
    reason: str = ""

    @staticmethod
    def ok():
        return GateResult(passed=True)

    @staticmethod
    def fail(gate, reason):
        return GateResult(passed=False, gate=gate, reason=reason)


class RiskGates:
    """
    Risk management gate pipeline.

    All gates are checked before placing any order. A single failure
    blocks the trade. The check order matters: kill switch first (cheapest),
    daily limits next, then per-trade checks.
    """

    def __init__(self, config):
        self.risk_per_trade = config.risk_per_trade        # 0.01
        self.max_trade_loss_pct = config.max_trade_loss_pct  # 0.015
        self.max_concurrent = config.max_concurrent        # 3
        self.max_daily_trades = config.max_daily_trades    # 15
        self.kill_switch_pct = config.kill_switch_pct      # -0.03
        self.point_value = config.point_value              # 20.0
        self.max_cumulative_risk_pct = getattr(config, 'max_cumulative_risk_pct', 0.05)  # 5%

    def check_all(self, daily_state, proposed_order):
        """
        Run all gates in order. Returns first failure or GateResult.ok().

        Args:
            daily_state: DailyState
            proposed_order: OrderGroup
        """
        checks = [
            self._check_kill_switch,
            self._check_daily_trades,
            self._check_concurrent_positions,
            self._check_cumulative_risk,
            self._check_per_trade_risk,
            self._check_max_trade_loss,
        ]
        for check in checks:
            result = check(daily_state, proposed_order)
            if not result.passed:
                return result
        return GateResult.ok()

    def _check_kill_switch(self, daily_state, proposed_order):
        """Reject if kill switch is already active."""
        if daily_state.kill_switch_active:
            return GateResult.fail(
                "kill_switch",
                f"Kill switch active: {daily_state.kill_switch_reason}"
            )
        return GateResult.ok()

    def _check_daily_trades(self, daily_state, proposed_order):
        """Reject if daily trade count exceeded."""
        if daily_state.trade_count >= self.max_daily_trades:
            return GateResult.fail(
                "daily_trades",
                f"Daily trade limit reached: {daily_state.trade_count}/{self.max_daily_trades}"
            )
        return GateResult.ok()

    def _check_concurrent_positions(self, daily_state, proposed_order):
        """Reject if max concurrent positions exceeded."""
        active = daily_state.active_order_count
        if active >= self.max_concurrent:
            return GateResult.fail(
                "concurrent_positions",
                f"Max concurrent positions reached: {active}/{self.max_concurrent}"
            )
        return GateResult.ok()

    def _check_cumulative_risk(self, daily_state, proposed_order):
        """Reject if total open risk (all positions + proposed) exceeds cumulative limit."""
        balance = daily_state.start_balance + daily_state.realized_pnl
        if balance <= 0:
            return GateResult.ok()

        # Sum risk dollars from all open positions
        total_risk = 0.0
        for og in daily_state.open_positions:
            qty = og.filled_qty or og.target_qty
            total_risk += og.risk_pts * self.point_value * qty
        for og in daily_state.pending_orders:
            qty = og.target_qty
            total_risk += og.risk_pts * self.point_value * qty

        # Add proposed order risk
        proposed_risk = (
            proposed_order.risk_pts * self.point_value * proposed_order.target_qty
        )
        total_with_new = total_risk + proposed_risk
        max_risk = balance * self.max_cumulative_risk_pct

        if total_with_new > max_risk * 1.01:  # 1% rounding tolerance
            return GateResult.fail(
                "cumulative_risk",
                f"Cumulative risk ${total_with_new:.0f} exceeds "
                f"{self.max_cumulative_risk_pct*100:.1f}% of balance (max ${max_risk:.0f})"
            )
        return GateResult.ok()

    def _check_per_trade_risk(self, daily_state, proposed_order):
        """
        Reject if trade risk exceeds the order's risk tier % of current balance.
        Uses proposed_order.risk_pct (set by 3-tier sizing) instead of uniform %.
        """
        balance = daily_state.start_balance + daily_state.realized_pnl
        actual_risk_pct = getattr(proposed_order, 'risk_pct', self.risk_per_trade)
        max_risk = balance * actual_risk_pct
        trade_risk = (
            proposed_order.risk_pts * self.point_value * proposed_order.target_qty
        )

        if trade_risk > max_risk * 1.01:  # 1% tolerance for rounding
            return GateResult.fail(
                "per_trade_risk",
                f"Trade risk ${trade_risk:.0f} exceeds {actual_risk_pct*100:.1f}% "
                f"of balance ${balance:.0f} (max ${max_risk:.0f})"
            )
        return GateResult.ok()

    def _check_max_trade_loss(self, daily_state, proposed_order):
        """
        Reject if potential max loss (with slippage) exceeds risk tier + 0.5% buffer.
        """
        balance = daily_state.start_balance + daily_state.realized_pnl
        actual_risk_pct = getattr(proposed_order, 'risk_pct', self.risk_per_trade)
        max_loss = balance * (actual_risk_pct + 0.005)  # risk tier + 0.5% slippage buffer
        # Worst case: 1 tick slippage on stop
        worst_risk_pts = proposed_order.risk_pts + 0.25
        potential_loss = worst_risk_pts * self.point_value * proposed_order.target_qty

        if potential_loss > max_loss * 1.01:
            return GateResult.fail(
                "max_trade_loss",
                f"Potential loss ${potential_loss:.0f} (with slippage) exceeds "
                f"{(actual_risk_pct+0.005)*100:.1f}% of balance (max ${max_loss:.0f})"
            )
        return GateResult.ok()

    def evaluate_kill_switch(self, daily_state):
        """
        Check if daily P&L has breached the kill switch threshold.

        Includes worst-case unrealized P&L: assumes every open position
        is at its stop loss (maximum possible loss per position).

        Returns:
            True if kill switch should be triggered.
        """
        if daily_state.kill_switch_active:
            return False

        if daily_state.start_balance <= 0:
            return False

        # Worst-case unrealized: every open position at its stop
        worst_unrealized = 0.0
        for og in daily_state.open_positions:
            qty = og.filled_qty or og.target_qty
            worst_unrealized -= og.risk_pts * self.point_value * qty

        total_pnl = daily_state.realized_pnl + worst_unrealized
        pnl_pct = total_pnl / daily_state.start_balance

        if pnl_pct <= self.kill_switch_pct:
            daily_state.kill_switch_active = True
            daily_state.kill_switch_reason = (
                f"Daily P&L (realized ${daily_state.realized_pnl:.0f} + "
                f"worst unrealized ${worst_unrealized:.0f}) = "
                f"{pnl_pct*100:.1f}% breached {self.kill_switch_pct*100:.0f}% threshold"
            )
            return True

        return False
