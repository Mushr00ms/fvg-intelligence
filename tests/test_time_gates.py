"""Tests for time gates and session clock."""

import pytest
from datetime import time, datetime
from bot.risk.time_gates import TimeGates, _parse_time

import pytz
NY_TZ = pytz.timezone("America/New_York")


def _make_config():
    class C:
        session_start = "09:30"
        session_end = "16:00"
        last_entry_time = "15:45"
        cancel_unfilled_time = "15:50"
        flatten_time = "15:55"
    return C()


def _make_dt(hour, minute):
    """Create a NY-aware datetime for testing."""
    return NY_TZ.localize(datetime(2026, 3, 22, hour, minute))


class TestCanEnter:
    """Tests for entry time window."""

    def test_before_session(self):
        gates = TimeGates(_make_config())
        allowed, reason = gates.can_enter(_make_dt(9, 29))
        assert allowed is False
        assert "session start" in reason.lower() or "Before" in reason

    def test_at_session_start(self):
        gates = TimeGates(_make_config())
        allowed, _ = gates.can_enter(_make_dt(9, 30))
        assert allowed is True

    def test_mid_session(self):
        gates = TimeGates(_make_config())
        allowed, _ = gates.can_enter(_make_dt(12, 0))
        assert allowed is True

    def test_just_before_last_entry(self):
        gates = TimeGates(_make_config())
        allowed, _ = gates.can_enter(_make_dt(15, 44))
        assert allowed is True

    def test_at_last_entry_time(self):
        """15:45 should be REJECTED (>= boundary)."""
        gates = TimeGates(_make_config())
        allowed, _ = gates.can_enter(_make_dt(15, 45))
        assert allowed is False

    def test_after_last_entry(self):
        gates = TimeGates(_make_config())
        allowed, _ = gates.can_enter(_make_dt(15, 50))
        assert allowed is False


class TestIsSessionActive:
    """Tests for session activity check."""

    def test_before_session(self):
        gates = TimeGates(_make_config())
        assert gates.is_session_active(_make_dt(9, 0)) is False

    def test_during_session(self):
        gates = TimeGates(_make_config())
        assert gates.is_session_active(_make_dt(12, 0)) is True

    def test_at_session_end(self):
        gates = TimeGates(_make_config())
        assert gates.is_session_active(_make_dt(16, 0)) is False

    def test_at_session_start(self):
        gates = TimeGates(_make_config())
        assert gates.is_session_active(_make_dt(9, 30)) is True


class TestEodSchedule:
    """Tests for EOD action scheduling."""

    def test_all_actions_scheduled_before_1550(self):
        gates = TimeGates(_make_config())
        schedule = gates.get_eod_schedule(_make_dt(15, 0))
        action_names = [name for _, name in schedule]
        assert "cancel_unfilled" in action_names
        assert "flatten_all" in action_names
        assert "session_end" in action_names
        assert len(schedule) == 3

    def test_cancel_already_passed(self):
        gates = TimeGates(_make_config())
        schedule = gates.get_eod_schedule(_make_dt(15, 51))
        action_names = [name for _, name in schedule]
        assert "cancel_unfilled" not in action_names
        assert "flatten_all" in action_names
        assert "session_end" in action_names

    def test_delays_are_positive(self):
        gates = TimeGates(_make_config())
        schedule = gates.get_eod_schedule(_make_dt(15, 0))
        for delay, _ in schedule:
            assert delay > 0

    def test_no_actions_after_session(self):
        gates = TimeGates(_make_config())
        schedule = gates.get_eod_schedule(_make_dt(16, 5))
        assert len(schedule) == 0


class TestParseTime:
    def test_parse_string(self):
        t = _parse_time("15:45")
        assert t == time(15, 45)

    def test_parse_time_object(self):
        t = _parse_time(time(9, 30))
        assert t == time(9, 30)
