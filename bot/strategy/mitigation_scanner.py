"""
mitigation_scanner.py — Scan 1-minute bars against active FVGs for mitigation.

Mitigation = price touches the FVG zone (wick-based).
Only the first mitigation counts. FVG is removed from active list after.
"""

from datetime import datetime


def check_mitigation(bar_1min, fvg):
    """
    Check if a 1-minute bar mitigates (touches) an FVG zone.

    Mitigation condition (from fvg_analysis.py line 568):
        bar.low <= zone_high AND bar.high >= zone_low

    This is wick-based: even a wick touching the zone edge counts.

    Args:
        bar_1min: dict with {open, high, low, close, date}
        fvg: FVGRecord

    Returns:
        True if mitigated, False otherwise.
    """
    return bar_1min["low"] <= fvg.zone_high and bar_1min["high"] >= fvg.zone_low


def scan_active_fvgs(bar_1min, active_fvgs):
    """
    Check all active FVGs against a 1-minute bar for mitigation.

    Args:
        bar_1min: dict with {open, high, low, close, date}
        active_fvgs: list of FVGRecord

    Returns:
        List of (FVGRecord, mitigation_time_str) for mitigated FVGs.
        Only un-mitigated FVGs are checked.
    """
    mitigated = []
    mit_time = str(bar_1min["date"])

    for fvg in active_fvgs:
        if fvg.is_mitigated:
            continue

        if check_mitigation(bar_1min, fvg):
            fvg.is_mitigated = True
            fvg.mitigation_time = mit_time
            mitigated.append((fvg, mit_time))

    return mitigated
