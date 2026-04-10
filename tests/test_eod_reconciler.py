"""
test_eod_reconciler.py — Tests for EOD reconciliation trade matching and reporting.
"""

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime

import pytest

from bot.backtest.eod_reconciler import (
    Divergence,
    FillCheck,
    ReconciliationResult,
    WEEKLY_EXPECTATIONS,
    match_trades,
    format_telegram_report,
    build_backtest_config,
    build_weekly_summary,
    result_to_db_kwargs,
    validate_fills,
    has_bad_fills,
)
from bot.db import TradeDB


# ── Fake backtest Trade objects ──────────────────────────────────────────

@dataclass
class FakeTrade:
    """Minimal backtest Trade for testing (mirrors bot.backtest.backtester.Trade)."""
    trade_id: int = 1
    date: str = "2026-03-31"
    fvg_type: str = "bearish"
    time_period: str = "10:30-11:00"
    risk_range: str = "15-20"
    setup: str = "mit_extreme"
    side: str = "SELL"
    entry_price: float = 23500.0
    stop_price: float = 23518.25
    target_price: float = 23481.75
    risk_pts: float = 18.25
    n_value: float = 1.0
    contracts: int = 2
    zone_high: float = 23505.0
    zone_low: float = 23500.0
    formation_time: str = "2026-03-31 10:35:00"
    entry_time: str = "2026-03-31 10:45:00"
    exit_time: str = "2026-03-31 11:10:00"
    exit_price: float = 23481.75
    exit_reason: str = "TP"
    pnl_pts: float = 18.25
    pnl_dollars: float = 730.0
    is_win: bool = True
    tp_touched: bool = True
    runner_exit_reason: str = ""
    runner_exit_price: float = 0.0
    tp_exit_contracts: int = 0
    runner_contracts: int = 0
    excursion_pts: float = 0.0
    excursion_r: float = 0.0
    dd_note: str = ""


def _live_trade(**overrides):
    """Build a live trade dict (mirrors DB row)."""
    base = {
        "id": 1,
        "group_id": "abc123",
        "fvg_id": "fvg456",
        "trade_date": "2026-03-31",
        "fvg_type": "bearish",
        "zone_low": 23500.0,
        "zone_high": 23505.0,
        "fvg_size": 5.0,
        "time_period": "10:30-11:00",
        "risk_range": "15-20",
        "setup": "mit_extreme",
        "n_value": 1.0,
        "cell_ev": 0.15,
        "cell_win_rate": 50.0,
        "side": "SELL",
        "contracts": 2,
        "entry_price": 23500.0,
        "stop_price": 23518.25,
        "target_price": 23481.75,
        "risk_pts": 18.25,
        "actual_entry_price": 23500.0,
        "entry_slippage_pts": 0.0,
        "actual_exit_price": 23481.75,
        "stop_slippage_pts": 0.0,
        "exit_reason": "TP",
        "pnl_pts": 18.25,
        "gross_pnl": 730.0,
        "commission": 0.0,
        "net_pnl": 730.0,
        "entry_time": "2026-03-31T10:45:00",
        "exit_time": "2026-03-31T11:10:00",
        "duration_seconds": 1500,
        "balance_before": 100000.0,
        "balance_after": 100730.0,
        "daily_pnl_after": 730.0,
        "strategy_id": "test-strategy",
        "mode": "PAPER",
    }
    base.update(overrides)
    return base


# ── Tests ────────────────────────────────────────────────────────────────

class TestMatchTrades:
    """Test trade matching logic."""

    def test_perfect_match(self):
        """All trades match with identical fields."""
        live = [_live_trade()]
        bt = [FakeTrade()]
        result = match_trades(live, bt)

        assert result.live_count == 1
        assert result.backtest_count == 1
        assert result.matched_count == 1
        assert len(result.divergences) == 0

    def test_missed_backtest(self):
        """Backtest has a trade that live bot didn't take."""
        live = [_live_trade()]
        bt = [
            FakeTrade(),
            FakeTrade(trade_id=2, time_period="12:00-12:30",
                      entry_price=23400.0, entry_time="2026-03-31 12:15:00"),
        ]
        result = match_trades(live, bt)

        assert result.matched_count == 1
        missed = [d for d in result.divergences if d.severity == "MISSED_BACKTEST"]
        assert len(missed) == 1
        assert "12:00-12:30" in missed[0].cell_key

    def test_missed_live(self):
        """Live bot took a trade that backtest didn't produce."""
        live = [
            _live_trade(),
            _live_trade(id=2, group_id="def789", time_period="13:00-13:30",
                        entry_price=23350.0, entry_time="2026-03-31T13:05:00"),
        ]
        bt = [FakeTrade()]
        result = match_trades(live, bt)

        assert result.matched_count == 1
        missed = [d for d in result.divergences if d.severity == "MISSED_LIVE"]
        assert len(missed) == 1
        assert "13:00-13:30" in missed[0].cell_key

    def test_exit_mismatch(self):
        """Same setup matched, but different exit reason."""
        live = [_live_trade(exit_reason="SL", pnl_pts=-18.25, net_pnl=-730.0,
                            actual_exit_price=23518.25)]
        bt = [FakeTrade()]  # exits TP
        result = match_trades(live, bt)

        assert result.matched_count == 1
        exit_mis = [d for d in result.divergences if d.severity == "EXIT_MISMATCH"]
        assert len(exit_mis) == 1
        assert "SL" in exit_mis[0].live_detail
        assert "TP" in exit_mis[0].backtest_detail

    def test_price_drift(self):
        """Matched trade with entry price drift beyond tolerance."""
        live = [_live_trade(actual_entry_price=23500.0, pnl_pts=18.25)]
        bt = [FakeTrade(entry_price=23500.0, pnl_pts=21.0)]  # pnl differs by 2.75 > 2.0 tol
        result = match_trades(live, bt)

        assert result.matched_count == 1
        drifts = [d for d in result.divergences if d.severity == "PRICE_DRIFT"]
        assert len(drifts) == 1

    def test_no_trades_either_side(self):
        """No trades on either side."""
        result = match_trades([], [])
        assert result.live_count == 0
        assert result.backtest_count == 0
        assert result.matched_count == 0
        assert len(result.divergences) == 0

    def test_only_unfilled_live_trades_excluded(self):
        """Live trades without exit_reason are excluded."""
        live = [
            _live_trade(),
            _live_trade(id=2, group_id="unfilled", exit_reason=None,
                        pnl_pts=0, net_pnl=0),
        ]
        bt = [FakeTrade()]
        result = match_trades(live, bt)

        assert result.live_count == 1  # unfilled excluded
        assert result.matched_count == 1

    def test_multiple_trades_same_cell(self):
        """Multiple trades in the same cell matched by entry price."""
        live = [
            _live_trade(entry_price=23500.0, actual_entry_price=23500.0,
                        entry_time="2026-03-31T10:45:00"),
            _live_trade(id=2, group_id="second", entry_price=23510.0,
                        actual_entry_price=23510.0,
                        entry_time="2026-03-31T10:55:00"),
        ]
        bt = [
            FakeTrade(trade_id=1, entry_price=23500.0,
                      entry_time="2026-03-31 10:45:00"),
            FakeTrade(trade_id=2, entry_price=23510.0,
                      entry_time="2026-03-31 10:55:00"),
        ]
        result = match_trades(live, bt)

        assert result.matched_count == 2
        assert len(result.divergences) == 0

    def test_kill_switch_day(self):
        """Live stopped early (kill switch), backtest continued.
        Extra backtest trades should appear as MISSED_BACKTEST."""
        live = [_live_trade()]
        bt = [
            FakeTrade(),
            FakeTrade(trade_id=2, time_period="14:00-14:30",
                      entry_price=23300.0, entry_time="2026-03-31 14:10:00"),
            FakeTrade(trade_id=3, time_period="15:00-15:30",
                      entry_price=23250.0, entry_time="2026-03-31 15:05:00"),
        ]
        result = match_trades(live, bt)

        assert result.matched_count == 1
        missed = [d for d in result.divergences if d.severity == "MISSED_BACKTEST"]
        assert len(missed) == 2

    def test_contract_mismatch(self):
        """Same trade matched but different contract count."""
        live = [_live_trade(contracts=3)]
        bt = [FakeTrade(contracts=2)]
        result = match_trades(live, bt)

        assert result.matched_count == 1
        mis = [d for d in result.divergences if d.severity == "EXIT_MISMATCH"]
        assert len(mis) == 1
        assert "3ct" in mis[0].live_detail
        assert "2ct" in mis[0].backtest_detail

    def test_contract_mismatch_hfoiv_scaled(self):
        """Contract scaled down by HFOIV gate is expected, not severe."""
        live = [_live_trade(contracts=1)]
        bt = [FakeTrade(contracts=2)]
        result = match_trades(live, bt, hfoiv_active=True)

        assert result.matched_count == 1
        # HFOIV-caused contract reduction is not a real divergence
        severe = [d for d in result.divergences if d.severity == "EXIT_MISMATCH"]
        assert len(severe) == 0
        hfoiv = [d for d in result.divergences if d.severity == "HFOIV_EXPECTED"]
        assert len(hfoiv) == 1
        assert "HFOIV" in hfoiv[0].live_detail
        assert "1ct" in hfoiv[0].live_detail

    def test_contract_mismatch_no_hfoiv_tag_when_live_larger(self):
        """HFOIV tag only appears when live has FEWER contracts."""
        live = [_live_trade(contracts=3)]
        bt = [FakeTrade(contracts=2)]
        result = match_trades(live, bt, hfoiv_active=True)

        mis = [d for d in result.divergences if d.severity == "EXIT_MISMATCH"]
        assert len(mis) == 1
        assert "HFOIV" not in mis[0].live_detail


class TestFormatTelegramReport:
    """Test Telegram report formatting."""

    def test_no_trades(self):
        result = ReconciliationResult(
            date="2026-03-31", live_count=0, backtest_count=0, matched_count=0)
        msg = format_telegram_report(result)
        assert "No trades" in msg
        assert "2026-03-31" in msg

    def test_error_report(self):
        result = ReconciliationResult(
            date="2026-03-31", live_count=0, backtest_count=0, matched_count=0,
            error="Data download failed")
        msg = format_telegram_report(result)
        assert "SKIPPED" in msg
        assert "Data download failed" in msg

    def test_clean_report(self):
        result = ReconciliationResult(
            date="2026-03-31", live_count=3, backtest_count=3, matched_count=3,
            live_net_pnl=1500.0, backtest_net_pnl=1480.0)
        msg = format_telegram_report(result)
        assert "CLEAN" in msg
        assert "Live 3" in msg
        assert "Matched 3" in msg
        assert "All 3 trades match" in msg

    def test_divergence_report(self):
        divs = [
            Divergence("MISSED_BACKTEST", "10:30-11:00 | 15-20 | mit_extreme | SELL",
                        "N/A", "TP @ 23500.00, 2ct, +18.2pts"),
            Divergence("EXIT_MISMATCH", "12:00-12:30 | 10-15 | mit_extreme | BUY",
                        "SL", "TP"),
        ]
        result = ReconciliationResult(
            date="2026-03-31", live_count=3, backtest_count=4, matched_count=2,
            divergences=divs, live_net_pnl=-200.0, backtest_net_pnl=500.0)
        msg = format_telegram_report(result)
        assert "2 divergences" in msg
        assert "10:30-11:00" in msg
        assert "12:00-12:30" in msg
        assert "Delta" in msg

    def test_hfoiv_report_shows_corrected_pnl(self):
        result = ReconciliationResult(
            date="2026-04-09",
            live_count=8,
            backtest_count=8,
            matched_count=8,
            live_net_pnl=3745.0,
            backtest_net_pnl=8487.0,
            corrected_net_pnl=3775.0,
            divergences=[
                Divergence(
                    "HFOIV_EXPECTED",
                    "09:30-10:00 | 20-25 | mid_extreme | SELL",
                    "1ct ⬇ HFOIV",
                    "2ct",
                ),
            ],
        )
        msg = format_telegram_report(result)
        assert "Corrected" in msg
        assert "$+3,775" in msg
        assert "Residual" in msg
        assert "$-30" in msg

    def test_with_weekly_html(self):
        result = ReconciliationResult(
            date="2026-03-31", live_count=0, backtest_count=0, matched_count=0)
        msg = format_telegram_report(result, weekly_html="<b>WEEKLY</b>")
        assert "<b>WEEKLY</b>" in msg

    def test_kill_switch_note(self):
        result = ReconciliationResult(
            date="2026-03-31", live_count=1, backtest_count=3, matched_count=1,
            kill_switch_active=True,
            divergences=[
                Divergence("MISSED_BACKTEST", "cell", "N/A", "detail"),
            ])
        msg = format_telegram_report(result)
        assert "Kill switch" in msg


class TestBuildBacktestConfig:
    """Test backtest config builder."""

    def test_basic_config(self):
        from bot.bot_config import BotConfig
        bc = BotConfig()
        strategy = {"meta": {"id": "test", "name": "Test Strategy"}, "cells": []}
        config = build_backtest_config(bc, strategy, 100000)

        assert config["balance"] == 100000
        assert config["risk_pct"] == 0.01
        assert config["max_concurrent"] == 3
        assert config["max_daily_trades"] == 15
        assert config["slip"] is False
        assert config["margin_per_contract"] == 36750.0

    def test_risk_tiers_from_strategy(self):
        from bot.bot_config import BotConfig
        bc = BotConfig(use_risk_tiers=True)
        strategy = {
            "meta": {"id": "test", "risk_rules": {"small": 0.005}},
            "cells": [],
        }
        config = build_backtest_config(bc, strategy, 50000)
        assert config["risk_tiers"] is True


class TestWeeklySummary:
    """Test weekly health check generation."""

    def _make_db(self):
        """Create a temp DB with daily_stats for a test week."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db = TradeDB(tmp.name)
        return db, tmp.name

    def test_not_friday_returns_none(self):
        db, path = self._make_db()
        try:
            # 2026-03-30 is a Monday
            result = build_weekly_summary(db, "2026-03-30", 100000)
            assert result is None
        finally:
            os.unlink(path)

    def test_friday_with_stats(self):
        db, path = self._make_db()
        try:
            # Insert stats for the week of 2026-03-27 (Friday)
            # 2026-03-27 is actually a Thursday — let me find a real Friday
            # 2026-03-27 weekday: datetime(2026,3,27).weekday() = 4? No, let me check
            # Actually let me just use a date I know is Friday
            from datetime import date
            # March 27, 2026 is a Friday (2026-03-23=Mon ... 2026-03-27=Fri)
            friday = "2026-03-27"
            d = datetime.strptime(friday, "%Y-%m-%d")
            assert d.weekday() == 4, f"Expected Friday, got weekday={d.weekday()}"

            # Insert some daily stats for the week
            for day_offset, (trades, wins, pnl) in enumerate([
                (4, 2, 500),    # Monday
                (5, 2, -300),   # Tuesday
                (3, 1, 200),    # Wednesday
                (6, 3, 800),    # Thursday
                (4, 1, -100),   # Friday
            ]):
                day = datetime(2026, 3, 23 + day_offset).strftime("%Y-%m-%d")
                db.insert_daily_stats(
                    trade_date=day,
                    start_balance=100000,
                    end_balance=100000 + pnl,
                    net_pnl=pnl,
                    pnl_pct=pnl / 1000,
                    total_trades=trades,
                    wins=wins,
                    losses=trades - wins,
                )

            result = build_weekly_summary(db, friday, 100000)
            assert result is not None
            assert "WEEKLY HEALTH CHECK" in result
            assert "Trades: 22" in result
            assert "Reminder:" in result
        finally:
            os.unlink(path)

    def test_scaling_to_balance(self):
        db, path = self._make_db()
        try:
            friday = "2026-03-27"
            db.insert_daily_stats(
                trade_date="2026-03-23",
                start_balance=50000,
                end_balance=49000,
                net_pnl=-1000,
                pnl_pct=-2.0,
                total_trades=5,
                wins=1,
                losses=4,
            )
            # At $50k, normal PnL range is half of $100k range
            result = build_weekly_summary(db, friday, 50000)
            assert result is not None
            # P&L thresholds should be scaled
            assert "$" in result
        finally:
            os.unlink(path)

    def test_no_stats_recorded(self):
        db, path = self._make_db()
        try:
            friday = "2026-03-27"
            result = build_weekly_summary(db, friday, 100000)
            assert result is not None
            assert "No daily stats" in result
        finally:
            os.unlink(path)

    def test_thursday_before_holiday_friday(self):
        """2026-04-02 is Thursday, 2026-04-03 is Good Friday (holiday)."""
        db, path = self._make_db()
        try:
            thursday = "2026-04-02"
            d = datetime.strptime(thursday, "%Y-%m-%d")
            assert d.weekday() == 3, f"Expected Thursday, got weekday={d.weekday()}"

            db.insert_daily_stats(
                trade_date="2026-03-30",
                start_balance=100000,
                end_balance=100200,
                net_pnl=200,
                pnl_pct=0.2,
                total_trades=3,
                wins=2,
                losses=1,
            )

            result = build_weekly_summary(db, thursday, 100000)
            assert result is not None
            assert "WEEKLY HEALTH CHECK" in result
            # Header should show Thu, not Fri
            assert "Thu 2026-04-02" in result
        finally:
            os.unlink(path)

    def test_thursday_normal_friday_returns_none(self):
        """Thursday with a normal (non-holiday) Friday should return None."""
        db, path = self._make_db()
        try:
            # 2026-03-26 is a Thursday, 2026-03-27 (Friday) is not a holiday
            result = build_weekly_summary(db, "2026-03-26", 100000)
            assert result is None
        finally:
            os.unlink(path)


class TestResultToDbKwargs:
    """Test serialization for DB storage."""

    def test_basic_serialization(self):
        result = ReconciliationResult(
            date="2026-03-31",
            live_count=3,
            backtest_count=4,
            matched_count=2,
            divergences=[
                Divergence("MISSED_BACKTEST", "cell1", "N/A", "detail1"),
            ],
            live_net_pnl=500.0,
            backtest_net_pnl=700.0,
        )
        kwargs = result_to_db_kwargs(result)

        assert kwargs["trade_date"] == "2026-03-31"
        assert kwargs["divergence_count"] == 1
        assert kwargs["live_net_pnl"] == 500.0

        divs = json.loads(kwargs["divergences_json"])
        assert len(divs) == 1
        assert divs[0]["severity"] == "MISSED_BACKTEST"
