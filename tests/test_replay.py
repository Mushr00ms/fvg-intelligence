"""
Tests for the Databento tick-accurate replay module.

Covers: ReplayClock, ReplayOrderManager (tick fills + volume + slippage +
displacement), scenario selector, result exporter, comparator.
"""

import json
import math
import os
import sys
import pytest
from datetime import datetime, time, timedelta
from unittest.mock import MagicMock

import pytz

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

NY_TZ = pytz.timezone("America/New_York")


# ── ReplayClock ──────────────────────────────────────────────────────────

class TestReplayClock:
    def _make_clock(self):
        from bot.replay.replay_clock import ReplayClock
        return ReplayClock()

    def test_initial_state_returns_epoch(self):
        clock = self._make_clock()
        assert clock.now().year == 2000

    def test_update_advances_time(self):
        clock = self._make_clock()
        t1 = NY_TZ.localize(datetime(2024, 3, 15, 10, 30, 0))
        t2 = NY_TZ.localize(datetime(2024, 3, 15, 10, 35, 0))
        clock.update(t1)
        assert clock.now() == t1
        clock.update(t2)
        assert clock.now() == t2

    def test_update_does_not_go_backwards(self):
        clock = self._make_clock()
        t1 = NY_TZ.localize(datetime(2024, 3, 15, 10, 35, 0))
        t2 = NY_TZ.localize(datetime(2024, 3, 15, 10, 30, 0))
        clock.update(t1)
        clock.update(t2)
        assert clock.now() == t1

    def test_today_str(self):
        clock = self._make_clock()
        clock.update(NY_TZ.localize(datetime(2024, 6, 20, 14, 0, 0)))
        assert clock.today_str() == "2024-06-20"

    def test_update_from_ib(self):
        clock = self._make_clock()
        ib_time = datetime(2024, 3, 15, 14, 30, 0)  # UTC = 10:30 ET
        clock.update_from_ib(ib_time)
        assert clock.now().hour == 10

    def test_sync_is_noop(self):
        clock = self._make_clock()
        assert clock.sync() is True
        assert clock.is_synced is True
        assert clock.offset_ms == 0

    def test_eod_watchdog_forces_session_end(self):
        from bot.replay.replay_clock import ReplayClock
        import time as _time
        clock = ReplayClock()
        clock.update(NY_TZ.localize(datetime(2024, 3, 15, 15, 55, 0)))
        clock._last_update_wall = _time.time() - 31.0
        assert clock.now().hour == 16
        assert clock.now().minute == 0


# ── ReplayOrderManager — Tick-Accurate Fills ────────────────────────────

class TestReplayOrderManager:
    def _make_state(self, balance=76000):
        from bot.state.trade_state import DailyState
        return DailyState(date="2024-03-15", start_balance=balance)

    def _make_order(self, side="BUY", entry=17400.0, stop=17390.0,
                    target=17430.0, qty=2, setup="mit_extreme",
                    fvg_id="test_fvg_1", risk_pts=10.0, n_value=3.0):
        from bot.state.trade_state import OrderGroup
        import uuid
        return OrderGroup(
            group_id=uuid.uuid4().hex[:12],
            fvg_id=fvg_id, setup=setup, side=side,
            entry_price=entry, stop_price=stop, target_price=target,
            risk_pts=risk_pts, n_value=n_value, target_qty=qty,
        )

    def _make_mgr(self):
        from bot.replay.replay_order_manager import ReplayOrderManager
        from bot.bot_config import BotConfig
        config = BotConfig(
            replay_mode=True,
            margin_fallback_per_contract=33000.0,
            margin_buffer_pct=0.05,
            kill_switch_pct=-0.10,
        )
        logger = MagicMock()
        logger.log = MagicMock()
        state_mgr = MagicMock()
        state_mgr.save = MagicMock()
        return ReplayOrderManager(
            state_manager=state_mgr, logger=logger, config=config,
        )

    def test_place_bracket_submits_order(self):
        mgr = self._make_mgr()
        state = self._make_state()
        order = self._make_order()
        mgr.place_bracket(order, state)
        assert len(state.pending_orders) == 1
        assert state.pending_orders[0].state == "SUBMITTED"
        assert state.trade_count == 1

    def test_tick_entry_fill_volume_aware(self):
        """Entry fills when cumulative volume at limit price >= order qty."""
        mgr = self._make_mgr()
        state = self._make_state()
        order = self._make_order(side="BUY", entry=17400.0, qty=2)
        order.state = "SUBMITTED"
        order.submitted_at = "2024-03-15T10:00:00"
        state.pending_orders.append(order)
        mgr._entry_volume[order.group_id] = 0

        # Tick 1: 1 contract at entry price — not enough yet
        mgr.on_tick(17400.0, 1, "2024-03-15T10:05:00", state)
        assert len(state.pending_orders) == 1  # Still pending

        # Tick 2: 1 more contract at entry — now cumulative >= qty
        mgr.on_tick(17400.0, 1, "2024-03-15T10:05:01", state)
        assert len(state.open_positions) == 1
        assert state.open_positions[0].actual_entry_price == 17400.0

    def test_tick_entry_fill_through_price(self):
        """Price trading THROUGH our limit → immediate fill."""
        mgr = self._make_mgr()
        state = self._make_state()
        order = self._make_order(side="BUY", entry=17400.0, qty=2)
        order.state = "SUBMITTED"
        order.submitted_at = "2024-03-15T10:00:00"
        state.pending_orders.append(order)
        mgr._entry_volume[order.group_id] = 0

        # Price trades through our limit (below it)
        mgr.on_tick(17398.0, 5, "2024-03-15T10:05:00", state)
        assert len(state.open_positions) == 1
        # Fill at our limit price, not the trade price
        assert state.open_positions[0].actual_entry_price == 17400.0

    def test_stop_fill_with_real_slippage(self):
        """Stop fills at ACTUAL trade price, not our stop price → real slippage."""
        mgr = self._make_mgr()
        state = self._make_state()
        order = self._make_order(side="BUY", entry=17400.0, stop=17390.0,
                                 target=17430.0, qty=1, risk_pts=10.0)
        order.state = "FILLED"
        order.filled_qty = 1
        order.actual_entry_price = 17400.0
        state.open_positions.append(order)

        # Price gaps through our stop (trades at 17387, not 17390)
        mgr.on_tick(17387.0, 3, "2024-03-15T10:30:00", state)

        assert len(state.closed_trades) == 1
        assert state.closed_trades[0].close_reason == "SL"
        # Fill at actual trade price (3 pts slippage)
        assert state.closed_trades[0].actual_exit_price == 17387.0

    def test_tp_fill_at_exact_target(self):
        """TP fills at exact target price (passive limit order)."""
        mgr = self._make_mgr()
        state = self._make_state()
        order = self._make_order(side="BUY", entry=17400.0, target=17430.0,
                                 stop=17390.0, qty=1)
        order.state = "FILLED"
        order.filled_qty = 1
        order.actual_entry_price = 17400.0
        state.open_positions.append(order)

        # Price reaches target
        mgr.on_tick(17432.0, 2, "2024-03-15T10:30:00", state)

        assert len(state.closed_trades) == 1
        assert state.closed_trades[0].close_reason == "TP"
        # Fill at target, not at the trade price
        assert state.closed_trades[0].actual_exit_price == 17430.0

    def test_tick_sl_before_tp(self):
        """With ticks, the first trigger in time wins."""
        mgr = self._make_mgr()
        state = self._make_state()
        order = self._make_order(side="BUY", entry=17400.0, target=17430.0,
                                 stop=17390.0, qty=1)
        order.state = "FILLED"
        order.filled_qty = 1
        order.actual_entry_price = 17400.0
        state.open_positions.append(order)

        # SL tick arrives first
        mgr.on_tick(17389.0, 1, "2024-03-15T10:30:00", state)
        assert state.closed_trades[0].close_reason == "SL"

    def test_bar_fallback_sl_priority(self):
        """Bar-based fallback: SL wins on same-bar ambiguity."""
        mgr = self._make_mgr()
        state = self._make_state()
        order = self._make_order(side="BUY", entry=17400.0, target=17430.0,
                                 stop=17390.0, qty=1)
        order.state = "FILLED"
        order.filled_qty = 1
        order.actual_entry_price = 17400.0
        state.open_positions.append(order)

        bar = {"open": 17400, "high": 17435, "low": 17385, "close": 17420,
               "date": "2024-03-15T10:30:00"}
        mgr.on_bar(bar, state)
        assert state.closed_trades[0].close_reason == "SL"

    def test_flatten_all(self):
        mgr = self._make_mgr()
        mgr._last_price = 17405.0
        state = self._make_state()
        order = self._make_order(side="BUY", entry=17400.0, qty=1)
        order.state = "FILLED"
        order.filled_qty = 1
        order.actual_entry_price = 17400.0
        state.open_positions.append(order)

        mgr.flatten_all(state, "EOD")
        assert len(state.open_positions) == 0
        assert len(state.closed_trades) == 1
        assert state.closed_trades[0].close_reason == "EOD"

    def test_margin_displacement_suspends_farthest(self):
        """When margin is tight, the farthest order is suspended."""
        mgr = self._make_mgr()
        mgr._last_price = 17450.0
        state = self._make_state(balance=40000)

        order1 = self._make_order(side="BUY", entry=17350.0, qty=1, fvg_id="far")
        mgr.place_bracket(order1, state)
        assert len(state.pending_orders) == 1

        order2 = self._make_order(side="BUY", entry=17440.0, qty=1, fvg_id="near")
        mgr.place_bracket(order2, state)

        assert len(state.suspended_orders) == 1
        assert state.suspended_orders[0].entry_price == 17350.0
        assert any(o.entry_price == 17440.0 for o in state.pending_orders)

        log = mgr.get_displacement_log()
        assert len(log) >= 1
        assert log[0]["action"] == "suspend"


# ── Scenario Selector ────────────────────────────────────────────────────

class TestScenarioSelector:
    def _bt(self, trades):
        return {"meta": {"balance": 76000}, "trades": trades, "daily_pnl": []}

    def test_detect_eod_flatten(self):
        from bot.replay.scenario_selector import select_replay_dates
        bt = self._bt([
            {"date": "2024-01-15", "side": "BUY", "entry_price": 17400,
             "setup": "mit_extreme", "exit_reason": "EOD", "pnl_dollars": -100,
             "entry_time": "2024-01-15T10:00:00", "exit_time": "2024-01-15T16:00:00",
             "dd_note": "", "risk_range": "10-15", "risk_pts": 12},
        ])
        selected = select_replay_dates(bt)
        assert any("EOD flatten" in tag for s in selected for tag in s[1])

    def test_detect_high_trade_day(self):
        from bot.replay.scenario_selector import select_replay_dates
        trades = [
            {"date": "2024-02-10", "side": "BUY", "entry_price": 17400 + i,
             "setup": "mit_extreme", "exit_reason": "TP", "pnl_dollars": 100,
             "entry_time": f"2024-02-10T{10+i}:00:00",
             "exit_time": f"2024-02-10T{10+i}:30:00",
             "dd_note": "", "risk_range": "10-15", "risk_pts": 12}
            for i in range(8)
        ]
        selected = select_replay_dates(self._bt(trades))
        assert any("High-trade" in tag for s in selected for tag in s[1])

    def test_empty_backtest(self):
        from bot.replay.scenario_selector import select_replay_dates
        assert select_replay_dates(self._bt([])) == []


# ── Comparator ───────────────────────────────────────────────────────────

class TestComparator:
    def test_match(self):
        from bot.replay.comparator import compare_day
        trade = {"date": "2024-01-15", "side": "BUY", "entry_price": 17400.0,
                 "setup": "mit_extreme", "exit_reason": "TP",
                 "exit_price": 17430.0, "pnl_dollars": 500, "contracts": 1}
        result = compare_day([trade], [trade.copy()])
        assert len(result["matches"]) == 1

    def test_diff_exit_reason(self):
        from bot.replay.comparator import compare_day
        r = {"date": "2024-01-15", "side": "BUY", "entry_price": 17400.0,
             "setup": "mit_extreme", "exit_reason": "SL",
             "exit_price": 17390.0, "pnl_dollars": -200, "contracts": 1}
        b = {**r, "exit_reason": "TP", "exit_price": 17430.0, "pnl_dollars": 500}
        result = compare_day([r], [b])
        assert len(result["diffs"]) == 1

    def test_only_in_replay(self):
        from bot.replay.comparator import compare_day
        r = {"date": "2024-01-15", "side": "SELL", "entry_price": 17500.0,
             "setup": "mid_extreme", "exit_reason": "TP",
             "exit_price": 17470.0, "pnl_dollars": 300, "contracts": 1}
        result = compare_day([r], [])
        assert len(result["only_replay"]) == 1

    def test_pnl_tolerance(self):
        from bot.replay.comparator import compare_day
        r = {"date": "2024-01-15", "side": "BUY", "entry_price": 17400.0,
             "setup": "mit_extreme", "exit_reason": "TP",
             "exit_price": 17430.0, "pnl_dollars": 499.50, "contracts": 1}
        b = {**r, "pnl_dollars": 500.25}
        result = compare_day([r], [b])
        assert len(result["matches"]) == 1


# ── Result Exporter ──────────────────────────────────────────────────────

class TestResultExporter:
    def test_export_creates_json(self, tmp_path):
        from bot.state.trade_state import DailyState, OrderGroup
        from bot.bot_config import BotConfig
        from bot.replay.result_exporter import export_session

        state = DailyState(date="2024-03-15", start_balance=76000, realized_pnl=500.0)
        og = OrderGroup(
            group_id="test123", fvg_id="fvg_001", setup="mit_extreme",
            side="BUY", entry_price=17400.0, stop_price=17390.0,
            target_price=17430.0, risk_pts=10.0, n_value=3.0, target_qty=2,
            filled_qty=2, actual_entry_price=17400.0, actual_exit_price=17430.0,
            close_reason="TP", realized_pnl=500.0, state="CLOSED",
        )
        state.closed_trades.append(og)

        config = BotConfig(replay_mode=True)
        filepath = export_session(state, config, output_dir=str(tmp_path))

        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert data["meta"]["source"] == "replay"
        assert len(data["trades"]) == 1
        assert data["trades"][0]["exit_reason"] == "TP"
        assert data["summary"]["net_pnl"] == 500.0
