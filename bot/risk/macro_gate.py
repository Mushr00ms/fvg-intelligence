"""
macro_gate.py — Block entries during high-impact macro event windows.

NFP (8:30 AM ET), CPI (8:30 AM ET), and FOMC rate decisions (2:00 PM ET)
cause extreme volatility in specific time windows around the release.

Backtest analysis (2020-2026, 6857 trades) identified these blackout windows:

  NFP:  09:30-11:00 ET (post-release vol) + 15:30-16:00 ET (EOD noise)
        61 trades, -$13.4k PnL.  Rest of day: 239 trades, +$25k.

  CPI:  09:30-10:30 ET (post-release vol) + 12:00-12:30 ET (midday reversal)
        57 trades, -$17.8k PnL.  Rest of day: 273 trades, +$53k.

  FOMC: 09:30-10:00 ET + 12:00-13:30 ET (pre-release) + 14:30-15:00 ET
        74 trades, -$13.8k PnL.  Rest of day: 162 trades, +$24k.

Dates are loaded from logic/configs/macro_events.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, time
from typing import Set


_MACRO_EVENTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "logic", "configs", "macro_events.json",
)

# Blackout windows per event type (entry ET times that are blocked).
# Each tuple is (start_inclusive, end_exclusive).
_BLACKOUT_WINDOWS: dict[str, list[tuple[time, time]]] = {
    "nfp": [
        (time(9, 30), time(11, 0)),   # post-release volatility
        (time(15, 30), time(16, 0)),   # EOD noise
    ],
    "cpi": [
        (time(9, 30), time(10, 30)),   # post-release volatility
        (time(12, 0), time(12, 30)),   # midday reversal
    ],
    "fomc": [
        (time(9, 30), time(10, 0)),    # morning uncertainty
        (time(12, 0), time(13, 30)),   # pre-release positioning
        (time(14, 30), time(15, 0)),   # post-release whipsaw
    ],
}


@dataclass(frozen=True)
class MacroGateConfig:
    skip_nfp: bool = True
    skip_cpi: bool = True
    skip_fomc: bool = True


def _load_macro_dates() -> dict[str, Set[date]]:
    """Load macro event dates from JSON. Returns {event_type: set of dates}."""
    path = os.path.normpath(_MACRO_EVENTS_PATH)
    if not os.path.exists(path):
        return {"nfp": set(), "cpi": set(), "fomc": set()}

    with open(path) as f:
        raw = json.load(f)

    result = {}
    for key in ("nfp", "cpi", "fomc"):
        result[key] = {date.fromisoformat(d) for d in raw.get(key, [])}
    return result


# Module-level cache — loaded once on first import
_MACRO_DATES: dict[str, Set[date]] | None = None


def _get_macro_dates() -> dict[str, Set[date]]:
    global _MACRO_DATES
    if _MACRO_DATES is None:
        _MACRO_DATES = _load_macro_dates()
    return _MACRO_DATES


def is_macro_event_day(d: date) -> tuple[bool, str]:
    """
    Check if a date is a macro event day.

    Returns:
        (is_macro_day, event_type) — event_type is "nfp", "cpi", "fomc", or "".
    """
    dates = _get_macro_dates()
    for event_type in ("nfp", "cpi", "fomc"):
        if d in dates.get(event_type, set()):
            return True, event_type
    return False, ""


def is_blocked_by_macro_gate(
    d: date,
    entry_time_et: time,
    cfg: MacroGateConfig,
) -> tuple[bool, str]:
    """
    Check if an entry at (date, time ET) falls in a macro blackout window.

    Returns:
        (blocked, reason) — reason is e.g. "macro_nfp_blackout"
    """
    dates = _get_macro_dates()

    checks = []
    if cfg.skip_nfp:
        checks.append("nfp")
    if cfg.skip_cpi:
        checks.append("cpi")
    if cfg.skip_fomc:
        checks.append("fomc")

    for event_type in checks:
        if d not in dates.get(event_type, set()):
            continue
        for win_start, win_end in _BLACKOUT_WINDOWS[event_type]:
            if win_start <= entry_time_et < win_end:
                return True, f"macro_{event_type}_blackout"

    return False, ""


def get_blackout_windows(d: date, cfg: MacroGateConfig) -> list[tuple[time, time, str]]:
    """
    Return active blackout windows for a given date.

    Returns list of (start, end, event_type) tuples, or empty list if
    no macro events or all skips disabled.
    """
    dates = _get_macro_dates()
    windows = []

    checks = []
    if cfg.skip_nfp:
        checks.append("nfp")
    if cfg.skip_cpi:
        checks.append("cpi")
    if cfg.skip_fomc:
        checks.append("fomc")

    for event_type in checks:
        if d in dates.get(event_type, set()):
            for win_start, win_end in _BLACKOUT_WINDOWS[event_type]:
                windows.append((win_start, win_end, event_type))

    return windows
