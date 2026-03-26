"""
us_holidays.py — US market holiday calendar for filtering non-trading days.

CME/NYSE equity market closures. Used by backtester and data fetcher
to skip days with no market data.
"""

from datetime import datetime, timedelta


# Full NYSE/CME equity market closures (not early closes)
# Source: NYSE holiday calendar + CME Holiday Calendar
US_MARKET_HOLIDAYS = {
    # 2024
    "20240101": "New Year's Day",
    "20240115": "MLK Day",
    "20240219": "Presidents Day",
    "20240329": "Good Friday",
    "20240527": "Memorial Day",
    "20240619": "Juneteenth",
    "20240704": "Independence Day",
    "20240902": "Labor Day",
    "20241128": "Thanksgiving",
    "20241225": "Christmas",
    # 2025
    "20250101": "New Year's Day",
    "20250120": "MLK Day",
    "20250217": "Presidents Day",
    "20250418": "Good Friday",
    "20250526": "Memorial Day",
    "20250619": "Juneteenth",
    "20250704": "Independence Day",
    "20250901": "Labor Day",
    "20251127": "Thanksgiving",
    "20251225": "Christmas",
    # 2026
    "20260101": "New Year's Day",
    "20260119": "MLK Day",
    "20260216": "Presidents Day",
    "20260403": "Good Friday",
    "20260525": "Memorial Day",
    "20260619": "Juneteenth",
    "20260703": "Independence Day (observed)",
    "20260907": "Labor Day",
    "20261126": "Thanksgiving",
    "20261225": "Christmas",
    # 2027
    "20270101": "New Year's Day",
    "20270118": "MLK Day",
    "20270215": "Presidents Day",
    "20270326": "Good Friday",
    "20270531": "Memorial Day",
    "20270618": "Juneteenth (observed)",
    "20270705": "Independence Day (observed)",
    "20270906": "Labor Day",
    "20271125": "Thanksgiving",
    "20271224": "Christmas (observed)",
}


def is_trading_day(date_str):
    """
    Check if a YYYYMMDD date is a trading day (weekday + not a market holiday).

    Args:
        date_str: Date string in YYYYMMDD format

    Returns:
        True if the date is a trading day
    """
    d = datetime.strptime(date_str, "%Y%m%d").date()
    # Skip weekends
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    # Skip holidays
    return date_str not in US_MARKET_HOLIDAYS


def trading_days_in_range(start_date, end_date):
    """
    Yield YYYYMMDD date strings that are trading days in the given range (inclusive).

    Args:
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format

    Yields:
        YYYYMMDD strings for each trading day
    """
    current = datetime.strptime(start_date, "%Y%m%d").date()
    end = datetime.strptime(end_date, "%Y%m%d").date()
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        if is_trading_day(date_str):
            yield date_str
        current += timedelta(days=1)
