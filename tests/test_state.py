"""Tests for trade state dataclasses and state manager."""

import json
import os
import pytest
from bot.state.trade_state import FVGRecord, OrderGroup, DailyState
from bot.state.state_manager import StateManager


class TestFVGRecord:
    """Tests for FVG record serialization."""

    def test_round_trip(self):
        fvg = FVGRecord(
            fvg_id="abc123", fvg_type="bullish",
            zone_low=19500, zone_high=19515,
            time_candle1="2026-03-22T10:30:00",
            time_candle2="2026-03-22T10:35:00",
            time_candle3="2026-03-22T10:40:00",
            middle_open=19505, middle_low=19490, middle_high=19530,
            first_open=19480, time_period="10:30-11:00",
            formation_date="2026-03-22",
        )
        d = fvg.to_dict()
        loaded = FVGRecord.from_dict(d)
        assert loaded.fvg_id == "abc123"
        assert loaded.zone_low == 19500
        assert loaded.fvg_type == "bullish"
        assert loaded.time_period == "10:30-11:00"

    def test_json_serializable(self):
        fvg = FVGRecord(
            fvg_id="x", fvg_type="bearish",
            zone_low=100, zone_high=110,
            time_candle1="t1", time_candle2="t2", time_candle3="t3",
            middle_open=105, middle_low=95, middle_high=115,
            first_open=102, time_period="10:00-10:30",
            formation_date="2026-03-22",
        )
        json_str = json.dumps(fvg.to_dict())
        assert '"fvg_type": "bearish"' in json_str


class TestOrderGroup:
    """Tests for order group serialization and state."""

    def test_round_trip(self):
        og = OrderGroup(
            group_id="og1", fvg_id="fvg1", setup="mit_extreme",
            side="BUY", entry_price=19500, stop_price=19490,
            target_price=19530, risk_pts=10, n_value=2.0, target_qty=3,
            filled_qty=2, state="PARTIAL",
            ib_entry_order_id=100, ib_tp_order_id=101, ib_sl_order_id=102,
        )
        d = og.to_dict()
        loaded = OrderGroup.from_dict(d)
        assert loaded.group_id == "og1"
        assert loaded.filled_qty == 2
        assert loaded.state == "PARTIAL"
        assert loaded.ib_entry_order_id == 100

    def test_is_active(self):
        og = OrderGroup(
            group_id="x", fvg_id="f", setup="mit_extreme",
            side="BUY", entry_price=100, stop_price=90,
            target_price=120, risk_pts=10, n_value=2.0, target_qty=1,
        )
        og.state = "SUBMITTED"
        assert og.is_active is True
        og.state = "FILLED"
        assert og.is_active is True
        og.state = "CLOSED"
        assert og.is_active is False
        og.state = "PENDING"
        assert og.is_active is False


class TestDailyState:
    """Tests for daily state operations."""

    def _make_state(self):
        state = DailyState(date="2026-03-22", start_balance=76000)
        og = OrderGroup(
            group_id="og1", fvg_id="fvg1", setup="mit_extreme",
            side="BUY", entry_price=19500, stop_price=19490,
            target_price=19530, risk_pts=10, n_value=2.0, target_qty=3,
            state="SUBMITTED",
        )
        state.pending_orders.append(og)
        return state

    def test_round_trip(self):
        state = self._make_state()
        state.realized_pnl = 500
        state.trade_count = 3
        d = state.to_dict()
        loaded = DailyState.from_dict(d)
        assert loaded.date == "2026-03-22"
        assert loaded.start_balance == 76000
        assert loaded.realized_pnl == 500
        assert loaded.trade_count == 3
        assert len(loaded.pending_orders) == 1

    def test_active_order_count(self):
        state = self._make_state()
        assert state.active_order_count == 1
        # Add an open position
        og2 = OrderGroup(
            group_id="og2", fvg_id="fvg2", setup="mid_extreme",
            side="SELL", entry_price=19600, stop_price=19610,
            target_price=19580, risk_pts=10, n_value=2.0, target_qty=2,
            filled_qty=2, state="FILLED",
        )
        state.open_positions.append(og2)
        assert state.active_order_count == 2

    def test_daily_pnl_pct(self):
        state = DailyState(date="2026-03-22", start_balance=76000)
        state.realized_pnl = -2280
        assert abs(state.daily_pnl_pct - (-0.03)) < 0.001

    def test_move_to_open(self):
        state = self._make_state()
        og = state.move_to_open("og1")
        assert og is not None
        assert og.state == "FILLED"
        assert len(state.pending_orders) == 0
        assert len(state.open_positions) == 1

    def test_move_to_closed(self):
        state = self._make_state()
        og = state.move_to_closed("og1", "TP", pnl=600)
        assert og is not None
        assert og.state == "CLOSED"
        assert og.close_reason == "TP"
        assert og.realized_pnl == 600
        assert state.realized_pnl == 600
        assert len(state.closed_trades) == 1

    def test_find_order_by_ib_id(self):
        state = self._make_state()
        state.pending_orders[0].ib_entry_order_id = 42
        state.pending_orders[0].ib_tp_order_id = 43
        found = state.find_order_by_ib_id(42)
        assert found is not None
        assert found.group_id == "og1"
        found2 = state.find_order_by_ib_id(43)
        assert found2 is not None
        assert state.find_order_by_ib_id(999) is None


class TestStateManager:
    """Tests for state persistence."""

    def test_save_and_load(self, tmp_dir):
        mgr = StateManager(tmp_dir)
        state = DailyState(
            date="2026-03-22", start_balance=76000,
            realized_pnl=500, trade_count=2,
        )
        mgr.save(state, force=True)
        loaded = mgr.load()
        assert loaded is not None
        assert loaded.start_balance == 76000
        assert loaded.realized_pnl == 500

    def test_load_returns_none_for_stale_date(self, tmp_dir):
        mgr = StateManager(tmp_dir)
        state = DailyState(date="2020-01-01", start_balance=76000)
        mgr.save(state, force=True)
        loaded = mgr.load()
        assert loaded is None  # Different date than today

    def test_load_returns_none_if_no_file(self, tmp_dir):
        mgr = StateManager(tmp_dir)
        assert mgr.load() is None

    def test_create_new(self, tmp_dir):
        mgr = StateManager(tmp_dir)
        state = mgr.create_new(76000)
        assert state.start_balance == 76000
        assert state.realized_pnl == 0
        assert state.trade_count == 0

    def test_kill_switch_persists(self, tmp_dir):
        mgr = StateManager(tmp_dir)
        from datetime import datetime
        import pytz
        today = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
        state = DailyState(date=today, start_balance=76000)
        state.kill_switch_active = True
        state.kill_switch_reason = "Daily loss exceeded -3%"
        mgr.save(state, force=True)
        loaded = mgr.load()
        assert loaded is not None
        assert loaded.kill_switch_active is True
        assert "3%" in loaded.kill_switch_reason

    def test_atomic_write(self, tmp_dir):
        """No .tmp file should remain after save."""
        mgr = StateManager(tmp_dir)
        state = DailyState(date="2026-03-22", start_balance=76000)
        mgr.save(state, force=True)
        files = os.listdir(tmp_dir)
        assert not any(f.endswith(".tmp") for f in files)
