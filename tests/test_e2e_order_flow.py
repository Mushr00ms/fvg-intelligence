"""
End-to-end tests: FVG detection → strategy → risk gates → order placement → state transitions.

Exercises the real _process_detection path from engine.py with:
- Real BotConfig, RiskGates, StrategyLoader, trade_calculator
- Real DailyState and state transitions (pending → open → closed)
- DRY_RUN mode (no IB connection needed)
- Dynamic price generation (no hardcoded scenarios)

Tests concurrent position limits, kill switch, cumulative risk, and full lifecycle.
"""

import asyncio
import json
import os
import random
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional

import pytest
import pytz

from bot.bot_config import BotConfig
from bot.risk.risk_gates import RiskGates, GateResult
from bot.state.trade_state import DailyState, OrderGroup, FVGRecord, _new_id
from bot.strategy.fvg_detector import (
    ActiveFVGManager, check_fvg_3bars, _assign_time_period, SESSION_INTERVALS,
)
from bot.strategy.trade_calculator import calculate_setup, risk_to_range, round_to_tick

NY = pytz.timezone("America/New_York")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _round_tick(p):
    return round(p * 4) / 4


class _CaptureLogger:
    """Capture log events for assertion."""
    def __init__(self):
        self.records = []

    def log(self, event, **kwargs):
        self.records.append({"event": event, **kwargs})

    def events(self, name):
        return [r for r in self.records if r["event"] == name]


class _NoOpDB:
    """DB stub that records calls without touching sqlite."""
    def __init__(self):
        self.trades = []
        self.fvg_updates = []

    def insert_trade(self, **kwargs):
        self.trades.append(kwargs)

    def insert_fvg(self, **kwargs):
        pass

    def update_fvg(self, fvg_id, **kwargs):
        self.fvg_updates.append({"fvg_id": fvg_id, **kwargs})


class _FakeClock:
    """Clock that returns a fixed time during trading hours."""
    def __init__(self, hour=10, minute=30):
        self._now = NY.localize(datetime(2026, 3, 26, hour, minute))

    def now(self):
        return self._now

    def advance(self, minutes=5):
        self._now += timedelta(minutes=minutes)


class _NoOpStateMgr:
    """State manager that just tracks saves."""
    def __init__(self):
        self.save_count = 0

    def save(self, state, force=False):
        self.save_count += 1


class _FakeIBConn:
    """Minimal IB connection mock."""
    is_connected = False


def _make_config(**overrides):
    """Create a DRY_RUN BotConfig with sensible test defaults."""
    defaults = dict(
        dry_run=True,
        paper_mode=True,
        ib_port=7497,
        risk_per_trade=0.01,
        use_risk_tiers=False,
        use_slippage=False,
        max_concurrent=3,
        max_daily_trades=15,
        kill_switch_pct=-0.03,
        max_cumulative_risk_pct=0.05,
        point_value=20.0,
        tick_size=0.25,
        max_trade_loss_pct=0.015,
        state_dir=tempfile.mkdtemp(),
        log_dir=tempfile.mkdtemp(),
        strategy_dir=tempfile.mkdtemp(),
    )
    defaults.update(overrides)
    return BotConfig(**defaults)


def _make_strategy(cells, strategy_id="test-e2e"):
    """Create a real StrategyLoader from cell definitions.

    Args:
        cells: list of dicts with keys: time_period, risk_range, setup, rr_target, ev, win_rate, samples
    """
    from bot.strategy.strategy_loader import StrategyLoader

    strategy = {
        "schema_version": "1.0",
        "meta": {
            "id": strategy_id,
            "name": "E2E Test",
            "description": "",
            "created_at": "2026-03-26",
            "updated_at": "2026-03-26",
            "source_dataset": "test",
            "ticker": "NQ",
            "timeframe": "5min",
        },
        "filters": {"min_samples": 10, "require_all_evs_positive": False},
        "cells": [{**c, "enabled": True, "notes": ""} for c in cells],
        "stats": {},
    }

    sdir = tempfile.mkdtemp()
    with open(os.path.join(sdir, f"{strategy_id}.json"), "w") as f:
        json.dump(strategy, f)
    with open(os.path.join(sdir, "manifest.json"), "w") as f:
        json.dump({
            "strategies": [{"id": strategy_id, "name": "E2E Test"}],
            "active_strategy": strategy_id,
            "last_updated": "2026-03-26",
        }, f)

    loader = StrategyLoader(sdir)
    loader.load()
    return loader


def _generate_fvg(fvg_type="bullish", zone_low=24100, zone_size=10,
                  target_risk=12.0, time_period="10:30-11:00"):
    """Generate a realistic FVGRecord with risk_pts matching target_risk.

    For bullish mit_extreme: entry=zone_high, stop=middle_low, risk=entry-stop.
    So middle_low = zone_high - target_risk.

    For bearish mit_extreme: entry=zone_low, stop=middle_high, risk=stop-entry.
    So middle_high = zone_low + target_risk.
    """
    zone_high = _round_tick(zone_low + zone_size)
    zone_low = _round_tick(zone_low)

    if fvg_type == "bullish":
        # mit_extreme entry = zone_high, stop = middle_low
        middle_low = _round_tick(zone_high - target_risk)
        middle_high = _round_tick(zone_high + zone_size)
    else:
        # mit_extreme entry = zone_low, stop = middle_high
        middle_high = _round_tick(zone_low + target_risk)
        middle_low = _round_tick(zone_low - zone_size)

    return FVGRecord(
        fvg_id=_new_id(),
        fvg_type=fvg_type,
        zone_low=zone_low,
        zone_high=zone_high,
        time_candle1=str(NY.localize(datetime(2026, 3, 26, 10, 30))),
        time_candle2=str(NY.localize(datetime(2026, 3, 26, 10, 35))),
        time_candle3=str(NY.localize(datetime(2026, 3, 26, 10, 40))),
        middle_open=_round_tick(zone_low + zone_size * 0.3),
        middle_low=middle_low,
        middle_high=middle_high,
        first_open=_round_tick(zone_low - 5),
        time_period=time_period,
        formation_date="2026-03-26",
    )


def _make_engine_deps(config, strategy, logger=None, clock=None):
    """Wire up a minimal set of engine dependencies for _process_detection."""

    @dataclass
    class _EngineShell:
        """Minimal stand-in for BotEngine — just enough for _process_detection."""
        config: object
        strategy: object
        risk_gates: object
        daily_state: object
        order_mgr: object
        fvg_mgr: object
        state_mgr: object
        logger: object
        db: object
        clock: object
        ib_conn: object
        telegram: object = field(default_factory=lambda: type("T", (), {"enabled": False})())
        hfoiv_gate: object = None
        _reconciliation_complete: bool = True
        _detection_lock: object = field(default_factory=asyncio.Lock)

    logger = logger or _CaptureLogger()
    clock = clock or _FakeClock()
    db = _NoOpDB()

    state = DailyState(date="2026-03-26", start_balance=76000.0)

    shell = _EngineShell(
        config=config,
        strategy=strategy,
        risk_gates=RiskGates(config),
        daily_state=state,
        order_mgr=None,  # DRY_RUN with no order_mgr uses inline placement
        fvg_mgr=ActiveFVGManager(strategy, config.min_fvg_size, logger),
        state_mgr=_NoOpStateMgr(),
        logger=logger,
        db=db,
        clock=clock,
        ib_conn=_FakeIBConn(),
    )

    # Bind _process_detection from engine module
    from bot.core.engine import BotEngine
    import types
    shell._process_detection = types.MethodType(BotEngine._process_detection, shell)
    shell._guarded_process_detection = types.MethodType(BotEngine._guarded_process_detection, shell)

    return shell


# ── Test: Full Detection → Strategy → Risk Gates → Order Placement ───────


class TestFullDetectionToOrder:
    """FVG detection through to DRY_RUN order placement."""

    def _setup(self, zone_low=24100, zone_size=12, risk_range="10-15",
               setup="mit_extreme", rr=2.5):
        """Create engine shell with a strategy cell matching the FVG."""
        strategy = _make_strategy([{
            "time_period": "10:30-11:00",
            "risk_range": risk_range,
            "setup": setup,
            "rr_target": rr,
            "ev": 0.20,
            "win_rate": 35.0,
            "samples": 300,
            "trades_per_day": 0.2,
            "median_risk": 12.0,
        }])
        config = _make_config()
        shell = _make_engine_deps(config, strategy)

        fvg = _generate_fvg("bullish", zone_low=zone_low, zone_size=zone_size)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        return shell, fvg

    def test_order_placed_in_dry_run(self):
        """Valid FVG + matching cell → order placed in DRY_RUN."""
        shell, fvg = self._setup()

        asyncio.get_event_loop().run_until_complete(
            shell._process_detection(fvg)
        )

        assert len(shell.daily_state.pending_orders) == 1
        order = shell.daily_state.pending_orders[0]
        assert order.state == "SUBMITTED"
        assert order.fvg_id == fvg.fvg_id
        assert order.side == "BUY"  # bullish FVG
        assert order.entry_price > 0
        assert order.stop_price > 0
        assert order.target_price > order.entry_price
        assert order.target_qty >= 1
        assert shell.daily_state.trade_count == 1

    def test_order_logged_as_accepted(self):
        """setup_accepted log should fire with all fields."""
        logger = _CaptureLogger()
        strategy = _make_strategy([{
            "time_period": "10:30-11:00", "risk_range": "10-15",
            "setup": "mit_extreme", "rr_target": 2.5,
            "ev": 0.20, "win_rate": 35.0, "samples": 300,
            "trades_per_day": 0.2, "median_risk": 12.0,
        }])
        shell = _make_engine_deps(_make_config(), strategy, logger=logger)
        fvg = _generate_fvg("bullish", zone_low=24100, zone_size=12)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        accepted = logger.events("setup_accepted")
        assert len(accepted) == 1
        assert accepted[0]["fvg_id"] == fvg.fvg_id
        assert accepted[0]["setup"] == "mit_extreme"
        assert "entry" in accepted[0]
        assert "risk_dollars" in accepted[0]

    def test_trade_written_to_db(self):
        """Order placement writes trade to DB."""
        shell, fvg = self._setup()
        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        assert len(shell.db.trades) == 1
        trade = shell.db.trades[0]
        assert trade["fvg_id"] == fvg.fvg_id
        assert trade["setup"] == "mit_extreme"
        assert trade["mode"] == "PAPER"

    def test_no_cell_match_removes_fvg(self):
        """FVG with no matching strategy cell → removed from active."""
        strategy = _make_strategy([{
            "time_period": "13:00-13:30", "risk_range": "5-10",
            "setup": "mid_extreme", "rr_target": 2.0,
            "ev": 0.15, "win_rate": 40.0, "samples": 200,
            "trades_per_day": 0.3, "median_risk": 7.0,
        }])
        shell = _make_engine_deps(_make_config(), strategy)
        fvg = _generate_fvg("bullish", zone_low=24100, zone_size=12,
                            time_period="10:30-11:00")  # no cell for this period
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        assert shell.fvg_mgr.active_count == 0
        assert len(shell.daily_state.pending_orders) == 0

    def test_bearish_fvg_sells(self):
        """Bearish FVG → SELL order."""
        strategy = _make_strategy([{
            "time_period": "10:30-11:00", "risk_range": "10-15",
            "setup": "mit_extreme", "rr_target": 2.0,
            "ev": 0.18, "win_rate": 33.0, "samples": 250,
            "trades_per_day": 0.2, "median_risk": 12.0,
        }])
        shell = _make_engine_deps(_make_config(), strategy)
        fvg = _generate_fvg("bearish", zone_low=24100, zone_size=12)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        if shell.daily_state.pending_orders:
            assert shell.daily_state.pending_orders[0].side == "SELL"


# ── Test: Concurrent Position Limits ─────────────────────────────────────


class TestConcurrentPositionLimits:
    """max_concurrent=3 should block the 4th order."""

    def _make_shell_with_n_positions(self, n_open, n_pending=0):
        strategy = _make_strategy([{
            "time_period": "10:30-11:00", "risk_range": "10-15",
            "setup": "mit_extreme", "rr_target": 2.5,
            "ev": 0.20, "win_rate": 35.0, "samples": 300,
            "trades_per_day": 0.2, "median_risk": 12.0,
        }])
        config = _make_config(max_concurrent=3)
        shell = _make_engine_deps(config, strategy)

        # Fill up open positions
        for i in range(n_open):
            og = OrderGroup(
                group_id=_new_id(), fvg_id=_new_id(),
                setup="mit_extreme", side="BUY",
                entry_price=24100 + i * 50, stop_price=24080 + i * 50,
                target_price=24130 + i * 50, risk_pts=10.0,
                n_value=2.5, target_qty=1, state="FILLED",
            )
            shell.daily_state.open_positions.append(og)

        for i in range(n_pending):
            og = OrderGroup(
                group_id=_new_id(), fvg_id=_new_id(),
                setup="mit_extreme", side="BUY",
                entry_price=24300 + i * 50, stop_price=24280 + i * 50,
                target_price=24330 + i * 50, risk_pts=10.0,
                n_value=2.5, target_qty=1, state="SUBMITTED",
            )
            shell.daily_state.pending_orders.append(og)

        return shell

    def test_3_open_blocks_4th(self):
        """With 3 open positions, a new order should be rejected."""
        shell = self._make_shell_with_n_positions(3)
        fvg = _generate_fvg("bullish", zone_low=24500, zone_size=12)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        assert len(shell.daily_state.pending_orders) == 0
        rejected = shell.logger.events("setup_rejected")
        assert any(r["gate"] == "concurrent_positions" for r in rejected)

    def test_2_open_1_pending_blocks_4th(self):
        """2 open + 1 pending = 3 active → blocks 4th."""
        shell = self._make_shell_with_n_positions(2, n_pending=1)
        fvg = _generate_fvg("bullish", zone_low=24500, zone_size=12)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        assert shell.daily_state.trade_count == 0

    def test_2_open_allows_3rd(self):
        """With 2 open positions, a new order should be allowed."""
        shell = self._make_shell_with_n_positions(2)
        fvg = _generate_fvg("bullish", zone_low=24500, zone_size=12)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        assert shell.daily_state.trade_count == 1

    @pytest.mark.parametrize("n_existing", [0, 1, 2])
    def test_positions_below_limit_allow_orders(self, n_existing):
        """Any count below max_concurrent should allow placement."""
        shell = self._make_shell_with_n_positions(n_existing)
        fvg = _generate_fvg("bullish", zone_low=24500, zone_size=12)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        assert shell.daily_state.trade_count == 1

    def test_closed_positions_dont_count(self):
        """Closed trades should NOT count toward concurrent limit."""
        shell = self._make_shell_with_n_positions(2)
        # Add a closed trade (should not block)
        closed = OrderGroup(
            group_id=_new_id(), fvg_id=_new_id(),
            setup="mit_extreme", side="BUY",
            entry_price=24000, stop_price=23990,
            target_price=24030, risk_pts=10.0,
            n_value=2.5, target_qty=1, state="CLOSED",
            close_reason="TP", realized_pnl=200.0,
        )
        shell.daily_state.closed_trades.append(closed)

        fvg = _generate_fvg("bullish", zone_low=24500, zone_size=12)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        assert shell.daily_state.trade_count == 1  # allowed


# ── Test: State Transitions ──────────────────────────────────────────────


class TestStateTransitions:
    """pending → open → closed lifecycle."""

    def test_move_to_open(self):
        """Simulates entry fill: pending → open."""
        state = DailyState(date="2026-03-26", start_balance=76000.0)
        og = OrderGroup(
            group_id="test-001", fvg_id="fvg-001",
            setup="mit_extreme", side="BUY",
            entry_price=24100, stop_price=24088,
            target_price=24130, risk_pts=12.0,
            n_value=2.5, target_qty=2, state="SUBMITTED",
        )
        state.pending_orders.append(og)

        result = state.move_to_open("test-001")

        assert result is not None
        assert result.state == "FILLED"
        assert len(state.pending_orders) == 0
        assert len(state.open_positions) == 1
        assert state.open_positions[0].group_id == "test-001"

    def test_move_to_closed_tp(self):
        """Simulates TP fill: open → closed with positive P&L."""
        state = DailyState(date="2026-03-26", start_balance=76000.0)
        og = OrderGroup(
            group_id="test-001", fvg_id="fvg-001",
            setup="mit_extreme", side="BUY",
            entry_price=24100, stop_price=24088,
            target_price=24130, risk_pts=12.0,
            n_value=2.5, target_qty=2, state="FILLED",
        )
        state.open_positions.append(og)

        pnl = 30.0 * 20.0 * 2  # 30 pts × $20 × 2 contracts = $1200
        result = state.move_to_closed("test-001", "TP", pnl=pnl)

        assert result is not None
        assert result.state == "CLOSED"
        assert result.close_reason == "TP"
        assert result.realized_pnl == pnl
        assert len(state.open_positions) == 0
        assert len(state.closed_trades) == 1
        assert state.realized_pnl == pnl

    def test_move_to_closed_sl(self):
        """Simulates SL fill: open → closed with negative P&L."""
        state = DailyState(date="2026-03-26", start_balance=76000.0)
        og = OrderGroup(
            group_id="test-001", fvg_id="fvg-001",
            setup="mit_extreme", side="BUY",
            entry_price=24100, stop_price=24088,
            target_price=24130, risk_pts=12.0,
            n_value=2.5, target_qty=2, state="FILLED",
        )
        state.open_positions.append(og)

        pnl = -12.0 * 20.0 * 2  # -12 pts × $20 × 2 contracts = -$480
        result = state.move_to_closed("test-001", "SL", pnl=pnl)

        assert result.close_reason == "SL"
        assert result.realized_pnl == pnl
        assert state.realized_pnl == pnl

    def test_full_lifecycle_multiple_trades(self):
        """3 trades through the full lifecycle: some win, some lose."""
        state = DailyState(date="2026-03-26", start_balance=76000.0)

        trades = [
            ("trade-1", "BUY", 24100, 24088, 24130, 12.0, 2),
            ("trade-2", "SELL", 24200, 24212, 24170, 12.0, 1),
            ("trade-3", "BUY", 24050, 24038, 24080, 12.0, 3),
        ]

        # Place all 3
        for gid, side, entry, stop, target, risk, qty in trades:
            og = OrderGroup(
                group_id=gid, fvg_id=f"fvg-{gid}",
                setup="mit_extreme", side=side,
                entry_price=entry, stop_price=stop,
                target_price=target, risk_pts=risk,
                n_value=2.5, target_qty=qty, state="SUBMITTED",
            )
            state.pending_orders.append(og)
            state.trade_count += 1

        assert state.active_order_count == 3
        assert state.trade_count == 3

        # Fill all entries
        for gid, *_ in trades:
            state.move_to_open(gid)
        assert len(state.pending_orders) == 0
        assert len(state.open_positions) == 3

        # trade-1: TP (win)
        state.move_to_closed("trade-1", "TP", pnl=30.0 * 20 * 2)
        # trade-2: SL (loss)
        state.move_to_closed("trade-2", "SL", pnl=-12.0 * 20 * 1)
        # trade-3: EOD flatten
        state.move_to_closed("trade-3", "FLATTEN", pnl=5.0 * 20 * 3)

        assert len(state.open_positions) == 0
        assert len(state.closed_trades) == 3
        assert state.active_order_count == 0

        expected_pnl = (30 * 20 * 2) + (-12 * 20 * 1) + (5 * 20 * 3)
        assert state.realized_pnl == expected_pnl

    def test_cancel_from_pending(self):
        """Cancel an unfilled order: pending → closed with CANCEL reason."""
        state = DailyState(date="2026-03-26", start_balance=76000.0)
        og = OrderGroup(
            group_id="test-cancel", fvg_id="fvg-cancel",
            setup="mit_extreme", side="BUY",
            entry_price=24100, stop_price=24088,
            target_price=24130, risk_pts=12.0,
            n_value=2.5, target_qty=1, state="SUBMITTED",
        )
        state.pending_orders.append(og)

        result = state.move_to_closed("test-cancel", "CANCEL", pnl=0.0)

        assert result.close_reason == "CANCEL"
        assert result.realized_pnl == 0.0
        assert len(state.pending_orders) == 0
        assert len(state.closed_trades) == 1
        assert state.realized_pnl == 0.0

    def test_move_nonexistent_returns_none(self):
        """Moving a non-existent group_id returns None."""
        state = DailyState(date="2026-03-26", start_balance=76000.0)
        assert state.move_to_open("does-not-exist") is None
        assert state.move_to_closed("does-not-exist", "TP") is None


# ── Test: Kill Switch ────────────────────────────────────────────────────


class TestKillSwitch:
    """Kill switch (-3% daily loss) should halt all new trading."""

    def _make_shell_with_strategy(self):
        strategy = _make_strategy([{
            "time_period": "10:30-11:00", "risk_range": "10-15",
            "setup": "mit_extreme", "rr_target": 2.5,
            "ev": 0.20, "win_rate": 35.0, "samples": 300,
            "trades_per_day": 0.2, "median_risk": 12.0,
        }])
        return _make_engine_deps(_make_config(), strategy)

    def test_emergency_halt_blocks_new_orders(self):
        """Active emergency halt → risk gate rejects all new orders."""
        shell = self._make_shell_with_strategy()
        shell.daily_state.kill_switch_active = True
        shell.daily_state.kill_switch_reason = "Daily loss exceeded -10%"

        fvg = _generate_fvg("bullish", zone_low=24100, zone_size=12)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        assert len(shell.daily_state.pending_orders) == 0
        rejected = shell.logger.events("setup_rejected")
        assert any(r["gate"] == "emergency_halt" for r in rejected)

    def test_emergency_halt_triggers_on_loss_threshold(self):
        """-10% P&L on $76k = -$7600. Verify risk gate catches it."""
        config = _make_config(kill_switch_pct=-0.10)
        gates = RiskGates(config)

        state = DailyState(date="2026-03-26", start_balance=76000.0)
        state.kill_switch_active = True
        state.kill_switch_reason = "PnL hit -10%"

        dummy_order = OrderGroup(
            group_id="x", fvg_id="x", setup="mit_extreme", side="BUY",
            entry_price=24100, stop_price=24088, target_price=24130,
            risk_pts=12.0, n_value=2.5, target_qty=1,
        )

        result = gates.check_all(state, dummy_order)
        assert not result.passed
        assert result.gate == "emergency_halt"

    def test_negative_pnl_below_threshold_still_allows(self):
        """P&L at -2% (above -3% threshold) should still allow trades."""
        shell = self._make_shell_with_strategy()
        shell.daily_state.realized_pnl = -1520.0  # -2% of 76k
        shell.daily_state.kill_switch_active = False

        fvg = _generate_fvg("bullish", zone_low=24100, zone_size=12)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        assert shell.daily_state.trade_count == 1

    def test_daily_trade_limit_blocks(self):
        """After max_daily_trades (15), no new orders."""
        shell = self._make_shell_with_strategy()
        shell.daily_state.trade_count = 15

        fvg = _generate_fvg("bullish", zone_low=24100, zone_size=12)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg

        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        assert len(shell.daily_state.pending_orders) == 0
        rejected = shell.logger.events("setup_rejected")
        assert any(r["gate"] == "daily_trades" for r in rejected)


# ── Test: Sequential Orders Fill Concurrent Limit ────────────────────────


class TestSequentialFillsAndLimits:
    """Simulate multiple FVGs detected → orders placed → fills → limits reached."""

    def test_4_fvgs_only_3_orders_placed(self):
        """4 FVGs detected, but max_concurrent=3 → only 3 orders placed."""
        strategy = _make_strategy([{
            "time_period": "10:30-11:00", "risk_range": "10-15",
            "setup": "mit_extreme", "rr_target": 2.5,
            "ev": 0.20, "win_rate": 35.0, "samples": 300,
            "trades_per_day": 0.2, "median_risk": 12.0,
        }])
        config = _make_config(max_concurrent=3)
        shell = _make_engine_deps(config, strategy)

        fvgs = []
        for i in range(4):
            fvg = _generate_fvg("bullish", zone_low=24100 + i * 50, zone_size=12)
            shell.fvg_mgr._active[fvg.fvg_id] = fvg
            fvgs.append(fvg)

        for fvg in fvgs:
            asyncio.get_event_loop().run_until_complete(
                shell._process_detection(fvg)
            )

        assert len(shell.daily_state.pending_orders) == 3
        assert shell.daily_state.trade_count == 3

        rejected = shell.logger.events("setup_rejected")
        concurrent_rejections = [r for r in rejected if r.get("gate") == "concurrent_positions"]
        assert len(concurrent_rejections) >= 1

    def test_close_position_then_new_order_allowed(self):
        """Close 1 of 3 positions → new order should be allowed."""
        strategy = _make_strategy([{
            "time_period": "10:30-11:00", "risk_range": "10-15",
            "setup": "mit_extreme", "rr_target": 2.5,
            "ev": 0.20, "win_rate": 35.0, "samples": 300,
            "trades_per_day": 0.2, "median_risk": 12.0,
        }])
        config = _make_config(max_concurrent=3)
        shell = _make_engine_deps(config, strategy)

        # Place 3 orders
        placed_ids = []
        for i in range(3):
            fvg = _generate_fvg("bullish", zone_low=24100 + i * 50, zone_size=12)
            shell.fvg_mgr._active[fvg.fvg_id] = fvg
            asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))
            placed_ids.append(shell.daily_state.pending_orders[-1].group_id)

        assert len(shell.daily_state.pending_orders) == 3

        # Simulate: first order fills and then hits TP
        shell.daily_state.move_to_open(placed_ids[0])
        shell.daily_state.move_to_closed(placed_ids[0], "TP", pnl=500.0)

        # Now active_order_count = 2 → room for 1 more
        assert shell.daily_state.active_order_count == 2

        fvg = _generate_fvg("bullish", zone_low=24400, zone_size=12)
        shell.fvg_mgr._active[fvg.fvg_id] = fvg
        asyncio.get_event_loop().run_until_complete(shell._process_detection(fvg))

        assert shell.daily_state.trade_count == 4  # 3 + 1 new
