"""Tests for position P&L calculation."""

import pytest
from bot.execution.position_tracker import calculate_pnl, determine_close_reason
from bot.state.trade_state import OrderGroup, CLOSE_TP, CLOSE_SL


def _make_order(side="BUY", entry=19500, stop=19490, target=19530, qty=3):
    return OrderGroup(
        group_id="test", fvg_id="fvg", setup="mit_extreme",
        side=side, entry_price=entry, stop_price=stop,
        target_price=target, risk_pts=abs(entry - stop),
        n_value=2.0, target_qty=qty, filled_qty=qty,
    )


class TestCalculatePnl:
    """Tests for P&L calculation."""

    def test_bullish_win(self):
        """BUY at 19500, TP at 19530: (19530-19500)*3*20 = $1800."""
        og = _make_order(side="BUY", entry=19500, target=19530, qty=3)
        pnl = calculate_pnl(og, fill_price=19530)
        assert pnl == 1800.0

    def test_bullish_loss(self):
        """BUY at 19500, SL at 19490: (19490-19500)*3*20 = -$600."""
        og = _make_order(side="BUY", entry=19500, stop=19490, qty=3)
        pnl = calculate_pnl(og, fill_price=19490)
        assert pnl == -600.0

    def test_bearish_win(self):
        """SELL at 19600, TP at 19570: (19600-19570)*2*20 = $1200."""
        og = _make_order(side="SELL", entry=19600, target=19570, qty=2)
        pnl = calculate_pnl(og, fill_price=19570)
        assert pnl == 1200.0

    def test_bearish_loss(self):
        """SELL at 19600, SL at 19615: (19600-19615)*2*20 = -$600."""
        og = _make_order(side="SELL", entry=19600, stop=19615, qty=2)
        pnl = calculate_pnl(og, fill_price=19615)
        assert pnl == -600.0

    def test_slippage_loss(self):
        """SL hit with 1 tick slippage: (19489.75-19500)*3*20 = -$615."""
        og = _make_order(side="BUY", entry=19500, stop=19490, qty=3)
        pnl = calculate_pnl(og, fill_price=19489.75)
        assert pnl == -615.0

    def test_zero_pnl(self):
        """Fill at entry price."""
        og = _make_order(side="BUY", entry=19500, qty=1)
        pnl = calculate_pnl(og, fill_price=19500)
        assert pnl == 0.0

    def test_different_point_value(self):
        """ES with $50/point."""
        og = _make_order(side="BUY", entry=5000, target=5010, qty=2)
        pnl = calculate_pnl(og, fill_price=5010, point_value=50.0)
        assert pnl == 1000.0


class TestDetermineCloseReason:
    """Tests for detecting TP vs SL fills."""

    def test_bullish_tp(self):
        og = _make_order(side="BUY", entry=19500, target=19530, stop=19490)
        assert determine_close_reason(og, 19530) == CLOSE_TP

    def test_bullish_sl(self):
        og = _make_order(side="BUY", entry=19500, target=19530, stop=19490)
        assert determine_close_reason(og, 19490) == CLOSE_SL

    def test_bearish_tp(self):
        og = _make_order(side="SELL", entry=19600, target=19570, stop=19615)
        assert determine_close_reason(og, 19570) == CLOSE_TP

    def test_bearish_sl(self):
        og = _make_order(side="SELL", entry=19600, target=19570, stop=19615)
        assert determine_close_reason(og, 19615) == CLOSE_SL

    def test_tp_with_slight_slippage(self):
        """Fill within 1 tick of target should be TP."""
        og = _make_order(side="BUY", entry=19500, target=19530, stop=19490)
        assert determine_close_reason(og, 19529.75) == CLOSE_TP
