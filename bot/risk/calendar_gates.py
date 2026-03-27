"""
calendar_gates.py - Utilities for quarterly witching calendar hard gates.

Witching day is defined as the third Friday of March, June, September, December.
The "-1" day is the previous weekday before that Friday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


_WITCHING_MONTHS = (3, 6, 9, 12)


def _third_friday(year: int, month: int) -> date:
    """Return the third Friday for the given month."""
    d = date(year, month, 1)
    while d.weekday() != 4:  # Friday
        d += timedelta(days=1)
    return d + timedelta(days=14)


def _previous_weekday(d: date) -> date:
    """Return the previous weekday (Mon-Fri), skipping weekend only."""
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def is_witching_day(d: date) -> bool:
    """True if date is quarterly witching day (third Fri of Mar/Jun/Sep/Dec)."""
    if d.month not in _WITCHING_MONTHS:
        return False
    return d == _third_friday(d.year, d.month)


def is_witching_day_minus_1(d: date) -> bool:
    """True if date is the prior weekday before quarterly witching day."""
    if d.month not in _WITCHING_MONTHS:
        return False
    return d == _previous_weekday(_third_friday(d.year, d.month))


@dataclass(frozen=True)
class WitchingGateConfig:
    no_trade_witching_day: bool = False
    no_trade_witching_day_minus_1: bool = False


def is_blocked_by_witching_gate(d: date, cfg: WitchingGateConfig) -> tuple[bool, str]:
    """
    Evaluate witching hard gates.

    Returns:
        (blocked, reason)
    """
    if cfg.no_trade_witching_day and is_witching_day(d):
        return True, "witching_day"
    if cfg.no_trade_witching_day_minus_1 and is_witching_day_minus_1(d):
        return True, "witching_day_minus_1"
    return False, ""
