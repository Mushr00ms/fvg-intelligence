from datetime import datetime, time, timedelta

import pandas as pd
import pytz

try:
    # Try relative import first (when used as a module)
    from ..config import ny_tz
except ImportError:
    # Fall back to absolute import (when run from logic directory)
    from config import ny_tz


def format_timedelta_analysis(td):
    """Custom formatter: Omit '0 days' and limit to HH:MM:SS."""
    if pd.isnull(td) or td is pd.NaT:
        return ""
    days = td.days
    seconds = int(td.total_seconds()) % (24 * 3600)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days > 0:
        return f"{days} days {hours:02}:{minutes:02}:{seconds:02}"
    else:
        return f"{hours:02}:{minutes:02}:{seconds:02}"


def ensure_ny_timezone(df):
    """Centralized function to ensure the DataFrame index is in NY timezone."""
    if df.empty:
        return df
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    return df


def get_session_based_data_end(
    now_utc, active_end_utc, session_start_time, session_end_time, debug=False
):
    """
    Determine the effective end time for data fetching based on session times.
    Only considers data within configured session windows to avoid unnecessary fetching.
    """

    now_ny = now_utc.astimezone(ny_tz)
    weekday = now_ny.weekday()  # 0=Monday, 6=Sunday

    # Get the last complete trading session end time
    if weekday < 5:  # Monday to Friday
        today_session_end = now_ny.replace(
            hour=session_end_time.hour,
            minute=session_end_time.minute,
            second=0,
            microsecond=0,
        )

        # If current time is after session end, today's session is complete
        if now_ny >= today_session_end:
            last_session_end = today_session_end
        else:
            # Session is ongoing, use previous trading day's session end
            if weekday == 0:  # Monday, go back to Friday
                last_session_end = (now_ny - timedelta(days=3)).replace(
                    hour=session_end_time.hour,
                    minute=session_end_time.minute,
                    second=0,
                    microsecond=0,
                )
            else:
                last_session_end = (now_ny - timedelta(days=1)).replace(
                    hour=session_end_time.hour,
                    minute=session_end_time.minute,
                    second=0,
                    microsecond=0,
                )
    elif weekday == 5:  # Saturday
        # Last session was Friday
        last_session_end = (now_ny - timedelta(days=1)).replace(
            hour=session_end_time.hour,
            minute=session_end_time.minute,
            second=0,
            microsecond=0,
        )
    else:  # Sunday
        # Last session was Friday
        last_session_end = (now_ny - timedelta(days=2)).replace(
            hour=session_end_time.hour,
            minute=session_end_time.minute,
            second=0,
            microsecond=0,
        )

    # Convert to UTC and ensure it doesn't exceed active_end_utc
    last_session_end_utc = last_session_end.astimezone(pytz.utc)
    effective_end_utc = min(last_session_end_utc, active_end_utc)

    if debug:
        print(
            f"[DEBUG] Session-based data end: {effective_end_utc} (last complete session within {session_start_time}-{session_end_time})"
        )

    return effective_end_utc


def check_session_data_coverage(
    df, session_start_time, session_end_time, target_date_utc, debug=False
):
    """
    Check if cached data adequately covers the session time window for a given date.
    Returns True if session is fully covered, False otherwise.
    """

    if df.empty:
        return False

    # Convert target date to NY timezone
    target_date_ny = target_date_utc.astimezone(ny_tz).date()

    # Define session start and end times for the target date
    session_start_dt = ny_tz.localize(
        datetime.combine(target_date_ny, session_start_time)
    )
    session_end_dt = ny_tz.localize(datetime.combine(target_date_ny, session_end_time))

    # Filter dataframe to the target date and session times
    session_data = df[(df.index >= session_start_dt) & (df.index <= session_end_dt)]

    if session_data.empty:
        if debug:
            print(
                f"[DEBUG] No data found for session {session_start_dt} to {session_end_dt}"
            )
        return False

    # Check if we have data covering the full session (allowing for some gaps)
    data_start = session_data.index.min()
    data_end = session_data.index.max()

    # Allow for small gaps at the beginning and end (e.g., 10 minutes)
    tolerance = timedelta(minutes=10)
    session_adequately_covered = data_start <= (
        session_start_dt + tolerance
    ) and data_end >= (session_end_dt - tolerance)

    if debug:
        print(
            f"[DEBUG] Session coverage check for {target_date_ny}: "
            f"data_start={data_start}, data_end={data_end}, "
            f"session_start={session_start_dt}, session_end={session_end_dt}, "
            f"adequately_covered={session_adequately_covered}"
        )

    return session_adequately_covered


def create_time_intervals(start_time, end_time, interval_minutes=30):
    """Create time intervals for analysis (e.g., 30-minute intervals)."""
    intervals = []
    current_start = start_time

    while current_start < end_time:
        # Calculate end time for current interval
        current_end_dt = datetime.combine(datetime.today(), current_start) + timedelta(
            minutes=interval_minutes
        )
        current_end = current_end_dt.time()

        # Ensure we don't go beyond the session end time
        if current_end > end_time:
            current_end = end_time

        intervals.append((current_start, current_end))

        # Move to next interval
        if current_end == end_time:
            break
        current_start = current_end

    return intervals


def assign_time_period_to_fvgs(df_fvgs, time_intervals):
    """Assign time periods to FVGs based on their time_candle3 (formation time)."""

    def assign_time_period(dt, intervals):
        if pd.isna(dt):
            return None
        t = dt.time()
        for start, end in intervals:
            # Use inclusive end for the last interval to catch boundary cases
            if start <= t < end or (t == end and end == intervals[-1][1]):
                return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
        return None

    df_fvgs["time_period"] = df_fvgs["time_candle3"].apply(
        assign_time_period, intervals=time_intervals
    )
    return df_fvgs
