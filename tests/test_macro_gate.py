"""Tests for macro event date detection.

Blackout windows are currently empty (backtesting showed filtering
costs PnL due to compounding effects). The infrastructure remains
for future use if windows are re-enabled.
"""

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

    def test_2026_cpi_april_10(self):
        is_macro, event = is_macro_event_day(date(2026, 4, 10))
        assert is_macro is True
        assert event == "cpi"


class TestBlackoutWindowsEmpty:
    """All blackout windows are currently empty — no entries are blocked."""

    def test_no_windows_on_any_event_day(self):
        cfg = MacroGateConfig()
        for d, ev in [(date(2024, 1, 5), "nfp"), (date(2024, 1, 11), "cpi"), (date(2024, 1, 31), "fomc")]:
            assert get_blackout_windows(d, cfg) == [], f"Expected no windows for {ev}"

    def test_never_blocked(self):
        cfg = MacroGateConfig()
        for d in [date(2024, 1, 5), date(2024, 1, 11), date(2024, 1, 31), date(2024, 1, 15)]:
            for t in [time(9, 30), time(10, 0), time(12, 0), time(14, 0)]:
                blocked, _ = is_blocked_by_macro_gate(d, t, cfg)
                assert blocked is False, f"Should not block {d} {t}"
