"""Tests for macro event gate (NFP/CPI/FOMC blackout windows)."""

from datetime import date, time

from bot.risk.macro_gate import (
    MacroGateConfig,
    is_blocked_by_macro_gate,
    is_macro_event_day,
    get_blackout_windows,
)


class TestIsMacroEventDay:
    def test_nfp_day_detected(self):
        is_macro, event = is_macro_event_day(date(2024, 1, 5))
        assert is_macro is True
        assert event == "nfp"

    def test_cpi_day_detected(self):
        is_macro, event = is_macro_event_day(date(2024, 1, 11))
        assert is_macro is True
        assert event == "cpi"

    def test_fomc_day_detected(self):
        is_macro, event = is_macro_event_day(date(2024, 1, 31))
        assert is_macro is True
        assert event == "fomc"

    def test_normal_day_not_detected(self):
        is_macro, event = is_macro_event_day(date(2024, 1, 15))
        assert is_macro is False
        assert event == ""


class TestBlackoutWindows:
    """Test that blackout windows are returned correctly per event type."""

    def test_nfp_windows(self):
        cfg = MacroGateConfig()
        windows = get_blackout_windows(date(2024, 1, 5), cfg)  # NFP
        assert len(windows) == 2
        starts = [(w[0], w[1]) for w in windows]
        assert (time(9, 30), time(11, 0)) in starts
        assert (time(15, 30), time(16, 0)) in starts

    def test_cpi_windows(self):
        cfg = MacroGateConfig()
        windows = get_blackout_windows(date(2024, 1, 11), cfg)  # CPI
        assert len(windows) == 2
        starts = [(w[0], w[1]) for w in windows]
        assert (time(9, 30), time(10, 30)) in starts
        assert (time(12, 0), time(12, 30)) in starts

    def test_fomc_windows(self):
        cfg = MacroGateConfig()
        windows = get_blackout_windows(date(2024, 1, 31), cfg)  # FOMC
        assert len(windows) == 3

    def test_normal_day_no_windows(self):
        cfg = MacroGateConfig()
        windows = get_blackout_windows(date(2024, 1, 15), cfg)
        assert windows == []

    def test_disabled_event_no_windows(self):
        cfg = MacroGateConfig(skip_nfp=False)
        windows = get_blackout_windows(date(2024, 1, 5), cfg)  # NFP but disabled
        assert windows == []


class TestMacroGateBlocking:
    """Test time-level blocking in blackout windows."""

    def test_nfp_blocked_at_0945(self):
        cfg = MacroGateConfig()
        blocked, reason = is_blocked_by_macro_gate(date(2024, 1, 5), time(9, 45), cfg)
        assert blocked is True
        assert reason == "macro_nfp_blackout"

    def test_nfp_allowed_at_1100(self):
        cfg = MacroGateConfig()
        blocked, _ = is_blocked_by_macro_gate(date(2024, 1, 5), time(11, 0), cfg)
        assert blocked is False

    def test_nfp_blocked_at_1530(self):
        cfg = MacroGateConfig()
        blocked, reason = is_blocked_by_macro_gate(date(2024, 1, 5), time(15, 30), cfg)
        assert blocked is True
        assert reason == "macro_nfp_blackout"

    def test_nfp_allowed_at_1400(self):
        cfg = MacroGateConfig()
        blocked, _ = is_blocked_by_macro_gate(date(2024, 1, 5), time(14, 0), cfg)
        assert blocked is False

    def test_cpi_blocked_at_1000(self):
        cfg = MacroGateConfig()
        blocked, reason = is_blocked_by_macro_gate(date(2024, 1, 11), time(10, 0), cfg)
        assert blocked is True
        assert reason == "macro_cpi_blackout"

    def test_cpi_allowed_at_1030(self):
        cfg = MacroGateConfig()
        blocked, _ = is_blocked_by_macro_gate(date(2024, 1, 11), time(10, 30), cfg)
        assert blocked is False

    def test_cpi_blocked_at_1215(self):
        cfg = MacroGateConfig()
        blocked, reason = is_blocked_by_macro_gate(date(2024, 1, 11), time(12, 15), cfg)
        assert blocked is True

    def test_cpi_allowed_at_1100(self):
        cfg = MacroGateConfig()
        blocked, _ = is_blocked_by_macro_gate(date(2024, 1, 11), time(11, 0), cfg)
        assert blocked is False

    def test_fomc_blocked_at_1230(self):
        cfg = MacroGateConfig()
        blocked, reason = is_blocked_by_macro_gate(date(2024, 1, 31), time(12, 30), cfg)
        assert blocked is True
        assert reason == "macro_fomc_blackout"

    def test_fomc_allowed_at_1400(self):
        """Post-FOMC release window (14:00-14:30) is allowed — it's profitable."""
        cfg = MacroGateConfig()
        blocked, _ = is_blocked_by_macro_gate(date(2024, 1, 31), time(14, 0), cfg)
        assert blocked is False

    def test_normal_day_never_blocked(self):
        cfg = MacroGateConfig()
        blocked, _ = is_blocked_by_macro_gate(date(2024, 1, 15), time(9, 45), cfg)
        assert blocked is False

    def test_all_disabled_never_blocked(self):
        cfg = MacroGateConfig(skip_nfp=False, skip_cpi=False, skip_fomc=False)
        blocked, _ = is_blocked_by_macro_gate(date(2024, 1, 5), time(9, 45), cfg)
        assert blocked is False

    def test_selective_skip_nfp_only(self):
        cfg = MacroGateConfig(skip_nfp=True, skip_cpi=False, skip_fomc=False)
        # NFP day, in window → blocked
        blocked, _ = is_blocked_by_macro_gate(date(2024, 1, 5), time(9, 45), cfg)
        assert blocked is True
        # CPI day, in window → allowed (skip_cpi=False)
        blocked, _ = is_blocked_by_macro_gate(date(2024, 1, 11), time(9, 45), cfg)
        assert blocked is False
