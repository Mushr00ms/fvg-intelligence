"""Tests for risk management gates."""

import pytest
from bot.risk.risk_gates import RiskGates, GateResult
from bot.state.trade_state import DailyState, OrderGroup


def _make_config():
    """Minimal config-like object for RiskGates."""
    class C:
        risk_per_trade = 0.01
        max_trade_loss_pct = 0.015
        max_concurrent = 3
        max_daily_trades = 15
        kill_switch_pct = -0.03
        point_value = 20.0
        max_cumulative_risk_pct = 0.05  # 5%
    return C()


def _make_state(balance=76000, pnl=0, trades=0, kill=False, pending=0, open_pos=0):
    state = DailyState(date="2026-03-22", start_balance=balance)
    state.realized_pnl = pnl
    state.trade_count = trades
    state.kill_switch_active = kill
    state.kill_switch_reason = "test" if kill else ""

    for i in range(pending):
        og = OrderGroup(
            group_id=f"pending-{i}", fvg_id="fvg", setup="mit_extreme",
            side="BUY", entry_price=19500, stop_price=19490, target_price=19530,
            risk_pts=10, n_value=2.0, target_qty=3, state="SUBMITTED",
        )
        state.pending_orders.append(og)

    for i in range(open_pos):
        og = OrderGroup(
            group_id=f"open-{i}", fvg_id="fvg", setup="mit_extreme",
            side="BUY", entry_price=19500, stop_price=19490, target_price=19530,
            risk_pts=10, n_value=2.0, target_qty=3, filled_qty=3, state="FILLED",
        )
        state.open_positions.append(og)

    return state


def _make_order(risk_pts=10, qty=3):
    return OrderGroup(
        group_id="test-order", fvg_id="fvg", setup="mit_extreme",
        side="BUY", entry_price=19500, stop_price=19490, target_price=19530,
        risk_pts=risk_pts, n_value=2.0, target_qty=qty,
    )


class TestRiskGates:
    """Test each gate individually and the pipeline."""

    def test_all_gates_pass(self):
        gates = RiskGates(_make_config())
        state = _make_state()
        order = _make_order(risk_pts=10, qty=3)  # risk = 10 * 20 * 3 = $600 < $760
        result = gates.check_all(state, order)
        assert result.passed is True

    def test_kill_switch_blocks(self):
        gates = RiskGates(_make_config())
        state = _make_state(kill=True)
        order = _make_order()
        result = gates.check_all(state, order)
        assert result.passed is False
        assert result.gate == "emergency_halt"

    def test_drawdown_multiplier_full_size(self):
        gates = RiskGates(_make_config())
        state = _make_state(balance=100000, pnl=-1500)  # -1.5%, above -2%
        assert gates.drawdown_multiplier(state) == 1.0

    def test_drawdown_multiplier_half_size(self):
        gates = RiskGates(_make_config())
        state = _make_state(balance=100000, pnl=-2500)  # -2.5%, between -2% and -4%
        assert gates.drawdown_multiplier(state) == 0.50

    def test_drawdown_multiplier_quarter_size(self):
        gates = RiskGates(_make_config())
        state = _make_state(balance=100000, pnl=-5000)  # -5%, below -4%
        assert gates.drawdown_multiplier(state) == 0.25

    def test_drawdown_multiplier_no_pnl(self):
        gates = RiskGates(_make_config())
        state = _make_state(balance=100000, pnl=500)  # positive day
        assert gates.drawdown_multiplier(state) == 1.0

    def test_daily_trade_limit(self):
        gates = RiskGates(_make_config())
        state = _make_state(trades=15)
        order = _make_order()
        result = gates.check_all(state, order)
        assert result.passed is False
        assert result.gate == "daily_trades"

    def test_concurrent_positions_pending(self):
        """3 pending orders = max reached."""
        gates = RiskGates(_make_config())
        state = _make_state(pending=3)
        order = _make_order()
        result = gates.check_all(state, order)
        assert result.passed is False
        assert result.gate == "concurrent_positions"

    def test_concurrent_positions_open(self):
        """3 open positions = max reached."""
        gates = RiskGates(_make_config())
        state = _make_state(open_pos=3)
        order = _make_order()
        result = gates.check_all(state, order)
        assert result.passed is False
        assert result.gate == "concurrent_positions"

    def test_concurrent_positions_mixed(self):
        """1 pending + 2 open = 3 = max reached."""
        gates = RiskGates(_make_config())
        state = _make_state(pending=1, open_pos=2)
        order = _make_order()
        result = gates.check_all(state, order)
        assert result.passed is False
        assert result.gate == "concurrent_positions"

    def test_concurrent_positions_under_limit(self):
        """2 open positions — room for 1 more."""
        gates = RiskGates(_make_config())
        state = _make_state(open_pos=2)
        order = _make_order()
        result = gates.check_all(state, order)
        assert result.passed is True

    def test_per_trade_risk_exceeded(self):
        """Risk = 25 * 20 * 3 = $1500 > $760 (1% of $76k)."""
        gates = RiskGates(_make_config())
        state = _make_state()
        order = _make_order(risk_pts=25, qty=3)
        result = gates.check_all(state, order)
        assert result.passed is False
        assert result.gate == "per_trade_risk"

    def test_per_trade_risk_with_losses(self):
        """Balance decreased by losses: $76k - $2k = $74k. 1% = $740."""
        gates = RiskGates(_make_config())
        state = _make_state(pnl=-2000)
        order = _make_order(risk_pts=12, qty=3)  # 12 * 20 * 3 = $720 < $740
        result = gates.check_all(state, order)
        assert result.passed is True

    def test_max_trade_loss_exceeded(self):
        """With slippage, potential loss = (25 + 0.25) * 20 * 3 = $1515 > $1140."""
        gates = RiskGates(_make_config())
        state = _make_state()
        order = _make_order(risk_pts=25, qty=3)
        result = gates.check_all(state, order)
        assert result.passed is False

    def test_max_trade_loss_ok(self):
        """With slippage: (7.5 + 0.25) * 20 * 5 = $775 < $1140."""
        gates = RiskGates(_make_config())
        state = _make_state()
        order = _make_order(risk_pts=7.5, qty=5)
        result = gates.check_all(state, order)
        assert result.passed is True


class TestCumulativeRisk:
    """Tests for cumulative risk gate (total open exposure)."""

    def test_cumulative_risk_pass_no_positions(self):
        """No open positions — proposed order risk is well within 5%."""
        gates = RiskGates(_make_config())
        state = _make_state()  # $76k balance, no positions
        order = _make_order(risk_pts=10, qty=3)  # $600 < $3800 (5%)
        result = gates._check_cumulative_risk(state, order)
        assert result.passed is True

    def test_cumulative_risk_fail_with_open_positions(self):
        """2 open positions + new order exceeds 5% cumulative risk.
        Each open: 10 * 20 * 3 = $600. Two = $1200.
        New order: 10 * 20 * 10 = $2000. Total = $3200.
        But with 10 contracts: 10 * 20 * 10 = $2000. 1200 + 2000 = $3200 < $3800.
        Use higher qty to trigger: 10 * 20 * 15 = $3000. 1200 + 3000 = $4200 > $3800.
        """
        gates = RiskGates(_make_config())
        state = _make_state(open_pos=2)  # 2 * $600 = $1200 at risk
        order = _make_order(risk_pts=10, qty=15)  # $3000 proposed
        result = gates._check_cumulative_risk(state, order)
        assert result.passed is False
        assert result.gate == "cumulative_risk"

    def test_cumulative_risk_includes_pending(self):
        """Pending orders count toward cumulative risk."""
        gates = RiskGates(_make_config())
        state = _make_state(pending=2)  # 2 * $600 = $1200 at risk
        order = _make_order(risk_pts=10, qty=15)  # $3000 proposed
        result = gates._check_cumulative_risk(state, order)
        assert result.passed is False

    def test_cumulative_risk_pass_with_losses(self):
        """Balance reduced by losses, but risk still within limit.
        Balance: $76k - $2k = $74k. 5% = $3700.
        Open: $600. Proposed: $600. Total: $1200 < $3700.
        """
        gates = RiskGates(_make_config())
        state = _make_state(pnl=-2000, open_pos=1)
        order = _make_order(risk_pts=10, qty=3)
        result = gates._check_cumulative_risk(state, order)
        assert result.passed is True


class TestKillSwitch:
    """Tests for kill switch evaluation."""

    def test_trigger_at_minus_3pct(self):
        gates = RiskGates(_make_config())
        state = _make_state(pnl=-2280)  # -2280/76000 = -3%
        triggered = gates.evaluate_kill_switch(state)
        assert triggered is True
        assert state.kill_switch_active is True

    def test_no_trigger_above_threshold(self):
        gates = RiskGates(_make_config())
        state = _make_state(pnl=-2000)  # -2.6%
        triggered = gates.evaluate_kill_switch(state)
        assert triggered is False
        assert state.kill_switch_active is False

    def test_no_double_trigger(self):
        gates = RiskGates(_make_config())
        state = _make_state(pnl=-3000, kill=True)
        triggered = gates.evaluate_kill_switch(state)
        assert triggered is False  # Already active, don't re-trigger
