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

    def _check_per_trade_risk(self, daily_state, proposed_order):
        """
        Reject if trade risk exceeds 1% of current balance.
        risk = risk_pts * point_value * qty
        """
        balance = daily_state.start_balance + daily_state.realized_pnl
        max_risk = balance * self.risk_per_trade
        trade_risk = (
            proposed_order.risk_pts * self.point_value * proposed_order.target_qty
        )

        if trade_risk > max_risk * 1.01:  # 1% tolerance for rounding
            return GateResult.fail(
                "per_trade_risk",
                f"Trade risk ${trade_risk:.0f} exceeds {self.risk_per_trade*100:.0f}% "
                f"of balance ${balance:.0f} (max ${max_risk:.0f})"
            )
        return GateResult.ok()

    def _check_max_trade_loss(self, daily_state, proposed_order):
        """
        Reject if potential max loss exceeds 1.5% of balance.
        This accounts for slippage on the stop.
        """
        balance = daily_state.start_balance + daily_state.realized_pnl
        max_loss = balance * self.max_trade_loss_pct
        # Worst case: 1 tick slippage on stop
        worst_risk_pts = proposed_order.risk_pts + 0.25
        potential_loss = worst_risk_pts * self.point_value * proposed_order.target_qty

        if potential_loss > max_loss * 1.01:
            return GateResult.fail(
                "max_trade_loss",
                f"Potential loss ${potential_loss:.0f} (with slippage) exceeds "
                f"{self.max_trade_loss_pct*100:.1f}% of balance (max ${max_loss:.0f})"
            )
        return GateResult.ok()

    def evaluate_kill_switch(self, daily_state):
        """
        Check if daily P&L has breached the kill switch threshold.

        Returns:
            True if kill switch should be triggered (daily loss >= threshold).
        """
        if daily_state.kill_switch_active:
            return False  # Already active

        if daily_state.start_balance <= 0:
            return False

        pnl_pct = daily_state.daily_pnl_pct
        if pnl_pct <= self.kill_switch_pct:
            daily_state.kill_switch_active = True
            daily_state.kill_switch_reason = (
                f"Daily P&L {pnl_pct*100:.1f}% breached "
                f"{self.kill_switch_pct*100:.0f}% threshold"
            )
            return True

        return False
