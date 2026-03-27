from datetime import date

from bot.risk.calendar_gates import (
    WitchingGateConfig,
    is_blocked_by_witching_gate,
    is_witching_day,
    is_witching_day_minus_1,
)


def test_witching_day_detection():
    assert is_witching_day(date(2025, 9, 19))
    assert not is_witching_day(date(2025, 9, 18))


def test_witching_day_minus_1_detection():
    assert is_witching_day_minus_1(date(2025, 9, 18))
    assert not is_witching_day_minus_1(date(2025, 9, 17))


def test_gate_blocks_both_days_when_enabled():
    cfg = WitchingGateConfig(
        no_trade_witching_day=True,
        no_trade_witching_day_minus_1=True,
    )
    blocked, reason = is_blocked_by_witching_gate(date(2025, 9, 19), cfg)
    assert blocked is True
    assert reason == "witching_day"

    blocked, reason = is_blocked_by_witching_gate(date(2025, 9, 18), cfg)
    assert blocked is True
    assert reason == "witching_day_minus_1"
