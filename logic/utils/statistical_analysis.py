from datetime import timedelta, time

import numpy as np
import pandas as pd

from .fvg_analysis import calculate_expansion_after_mitigation
from .visualization_utils import plot_fvg_expansion


def calculate_session_performance_metrics(df, session_definitions=None):
    """
    Calculate comprehensive performance metrics for NQ futures broken down by trading sessions.
    
    Args:
        df: DataFrame with OHLC price data (must include 'open', 'high', 'low', 'close' columns)
        session_definitions: Dict mapping session names to (start_time, end_time) tuples
                           Default: Asian (18:00-01:00), London (02:00-09:00), NY (09:30-16:00)
    
    Returns:
        DataFrame with performance metrics for each session including:
        - Session returns (open to close, percentage and points)
        - Win rate (percentage of positive sessions)
        - Total profit/loss
        - Average price movement (high-low range)
        - Session high/low statistics
        - Session range (high - low)
        - Cumulative performance
        - Volatility metrics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Default session definitions for NQ futures (in ET)
    if session_definitions is None:
        session_definitions = {
            'Asian': (time(18, 0), time(1, 0)),    # 6PM - 1AM ET (crosses midnight)
            'London': (time(2, 0), time(9, 0)),    # 2AM - 9AM ET
            'NY': (time(9, 30), time(16, 0))       # 9:30AM - 4PM ET
        }
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        elif 'time' in df.columns:
            df.set_index('time', inplace=True)
    
    results = []
    
    for session_name, (start_time, end_time) in session_definitions.items():
        # Handle sessions that cross midnight
        if start_time > end_time:
            # Session crosses midnight (e.g., Asian session)
            session_mask = (df.index.time >= start_time) | (df.index.time <= end_time)
        else:
            # Normal session within same day
            session_mask = (df.index.time >= start_time) & (df.index.time <= end_time)
        
        session_data = df[session_mask].copy()
        
        if session_data.empty:
            continue
        
        # Group by date to get daily session metrics
        session_data['date'] = session_data.index.date
        
        # For sessions crossing midnight, group appropriately
        if start_time > end_time:
            # Assign late night hours to previous day's session
            session_data.loc[session_data.index.time >= start_time, 'date'] = (
                session_data.loc[session_data.index.time >= start_time].index - pd.Timedelta(days=1)
            ).date
        
        daily_metrics = []
        
        for date in session_data['date'].unique():
            day_data = session_data[session_data['date'] == date]
            
            if len(day_data) < 2:  # Need at least 2 data points
                continue
            
            # Calculate session metrics
            session_open = day_data.iloc[0]['open']
            session_close = day_data.iloc[-1]['close']
            session_high = day_data['high'].max()
            session_low = day_data['low'].min()
            
            # Returns
            points_return = session_close - session_open
            pct_return = (points_return / session_open) * 100 if session_open != 0 else 0
            
            # Range metrics
            session_range = session_high - session_low
            
            # Intraday metrics
            high_from_open = session_high - session_open
            low_from_open = session_open - session_low
            close_from_high = session_high - session_close
            close_from_low = session_close - session_low
            
            daily_metrics.append({
                'date': date,
                'session': session_name,
                'open': session_open,
                'high': session_high,
                'low': session_low,
                'close': session_close,
                'points_return': points_return,
                'pct_return': pct_return,
                'range': session_range,
                'high_from_open': high_from_open,
                'low_from_open': low_from_open,
                'close_from_high': close_from_high,
                'close_from_low': close_from_low,
                'is_bullish': points_return > 0,
                'volume': day_data['volume'].sum() if 'volume' in day_data.columns else None
            })
        
        if not daily_metrics:
            continue
        
        # Create DataFrame for this session
        session_df = pd.DataFrame(daily_metrics)
        
        # Calculate aggregate statistics
        total_sessions = len(session_df)
        winning_sessions = session_df['is_bullish'].sum()
        losing_sessions = total_sessions - winning_sessions
        
        win_rate = (winning_sessions / total_sessions * 100) if total_sessions > 0 else 0
        
        # Performance metrics
        total_points = session_df['points_return'].sum()
        avg_points = session_df['points_return'].mean()
        std_points = session_df['points_return'].std()
        
        avg_pct_return = session_df['pct_return'].mean()
        std_pct_return = session_df['pct_return'].std()
        
        # Win/Loss metrics
        avg_win = session_df[session_df['is_bullish']]['points_return'].mean() if winning_sessions > 0 else 0
        avg_loss = session_df[~session_df['is_bullish']]['points_return'].mean() if losing_sessions > 0 else 0
        
        # Range metrics
        avg_range = session_df['range'].mean()
        max_range = session_df['range'].max()
        min_range = session_df['range'].min()
        
        # Calculate cumulative performance
        session_df['cumulative_points'] = session_df['points_return'].cumsum()
        session_df['cumulative_pct'] = ((1 + session_df['pct_return']/100).cumprod() - 1) * 100
        
        final_cumulative_points = session_df['cumulative_points'].iloc[-1]
        final_cumulative_pct = session_df['cumulative_pct'].iloc[-1]
        
        # Calculate max drawdown
        cumulative = session_df['cumulative_points'].values
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        sharpe_ratio = (avg_pct_return / std_pct_return * np.sqrt(252)) if std_pct_return != 0 else 0
        
        # Calculate percentiles
        returns_25 = session_df['points_return'].quantile(0.25)
        returns_50 = session_df['points_return'].quantile(0.50)
        returns_75 = session_df['points_return'].quantile(0.75)
        
        results.append({
            'session': session_name,
            'start_time': start_time.strftime('%H:%M'),
            'end_time': end_time.strftime('%H:%M'),
            'total_sessions': total_sessions,
            'winning_sessions': winning_sessions,
            'losing_sessions': losing_sessions,
            'win_rate': round(win_rate, 2),
            'total_points': round(total_points, 2),
            'avg_points_per_session': round(avg_points, 2),
            'std_points': round(std_points, 2),
            'avg_pct_return': round(avg_pct_return, 3),
            'std_pct_return': round(std_pct_return, 3),
            'avg_winning_session': round(avg_win, 2),
            'avg_losing_session': round(avg_loss, 2),
            'profit_factor': round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else float('inf'),
            'avg_range': round(avg_range, 2),
            'max_range': round(max_range, 2),
            'min_range': round(min_range, 2),
            'cumulative_points': round(final_cumulative_points, 2),
            'cumulative_pct_return': round(final_cumulative_pct, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'returns_25_percentile': round(returns_25, 2),
            'returns_median': round(returns_50, 2),
            'returns_75_percentile': round(returns_75, 2),
            'best_session_points': round(session_df['points_return'].max(), 2),
            'worst_session_points': round(session_df['points_return'].min(), 2),
            'avg_high_from_open': round(session_df['high_from_open'].mean(), 2),
            'avg_low_from_open': round(session_df['low_from_open'].mean(), 2)
        })
    
    return pd.DataFrame(results)


def calculate_day_of_week_stats(filtered_fvgs):
    """
    Calculate mitigation statistics by day of week (Monday-Friday).

    Args:
        filtered_fvgs: DataFrame with FVG data

    Returns:
        dict: Day-of-week statistics with mitigation percentage and average time
    """
    if filtered_fvgs.empty:
        return {}

    # Add day of week column (0=Monday, 6=Sunday) based on formation time
    filtered_fvgs = filtered_fvgs.copy()
    filtered_fvgs["day_of_week"] = filtered_fvgs["time_candle3"].dt.dayofweek

    # Only consider Monday-Friday (0-4)
    weekday_fvgs = filtered_fvgs[filtered_fvgs["day_of_week"] < 5]

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    day_stats = {}

    for day_num, day_name in enumerate(day_names):
        day_fvgs = weekday_fvgs[weekday_fvgs["day_of_week"] == day_num]

        if len(day_fvgs) == 0:
            day_stats[day_name] = {
                "total_fvgs": 0,
                "mitigated_fvgs": 0,
                "mitigation_percentage": 0.0,
                "average_mitigation_time": None,
            }
            continue

        total = len(day_fvgs)
        mitigated = day_fvgs["is_mitigated"].sum()
        mitigation_pct = (mitigated / total * 100) if total > 0 else 0.0

        avg_time = None
        if mitigated > 0:
            mitigated_df = day_fvgs[day_fvgs["is_mitigated"]]
            avg_time = mitigated_df["time_to_mitigation"].mean()

        day_stats[day_name] = {
            "total_fvgs": total,
            "mitigated_fvgs": mitigated,
            "mitigation_percentage": round(mitigation_pct, 2),
            "average_mitigation_time": avg_time,
        }

    return day_stats


def calculate_time_period_stats(
    filtered_fvgs, fvg_filter_start_time, fvg_filter_end_time
):
    """
    Calculate mitigation statistics by 30-minute time periods.

    Args:
        filtered_fvgs: DataFrame with FVG data
        fvg_filter_start_time: Start time for filtering (e.g., time(9, 30))
        fvg_filter_end_time: End time for filtering (e.g., time(16, 0))

    Returns:
        dict: Time period statistics with mitigation percentage and average time
    """
    if filtered_fvgs.empty:
        return {}

    # Convert time objects to datetime for easier manipulation
    from datetime import datetime, timedelta

    # Create 30-minute periods
    periods = []
    current_time = datetime.combine(datetime.today(), fvg_filter_start_time)
    end_time = datetime.combine(datetime.today(), fvg_filter_end_time)

    while current_time < end_time:
        period_end = current_time + timedelta(minutes=30)
        if period_end > end_time:
            period_end = end_time

        periods.append(
            {
                "start": current_time.time(),
                "end": period_end.time(),
                "label": f"{current_time.strftime('%H:%M')}-{period_end.strftime('%H:%M')}",
            }
        )

        current_time = period_end

    period_stats = {}

    for period in periods:
        # Filter FVGs that formed within this time period (based on formation time)
        period_fvgs = filtered_fvgs[
            (filtered_fvgs["time_candle3"].dt.time >= period["start"])
            & (filtered_fvgs["time_candle3"].dt.time < period["end"])
        ]

        if len(period_fvgs) == 0:
            period_stats[period["label"]] = {
                "total_fvgs": 0,
                "mitigated_fvgs": 0,
                "mitigation_percentage": 0.0,
                "average_mitigation_time": None,
            }
            continue

        total = len(period_fvgs)
        mitigated = period_fvgs["is_mitigated"].sum()
        mitigation_pct = (mitigated / total * 100) if total > 0 else 0.0

        avg_time = None
        if mitigated > 0:
            mitigated_df = period_fvgs[period_fvgs["is_mitigated"]]
            avg_time = mitigated_df["time_to_mitigation"].mean()

        # NEW: Calculate percentiles for expansion size and time for valid expansions
        exp_size_percentiles = None
        exp_time_percentiles = None

        # Check if expansion columns exist before calculating percentiles
        required_exp_cols = [
            "is_expansion_valid",
            "expansion_size",
            "expansion_time_seconds",
        ]
        if all(col in period_fvgs.columns for col in required_exp_cols):
            valid_expansions = period_fvgs[
                period_fvgs["is_mitigated"] & period_fvgs["is_expansion_valid"]
            ]
            if not valid_expansions.empty:
                exp_size_percentiles = (
                    valid_expansions["expansion_size"]
                    .quantile([0.25, 0.5, 0.75])
                    .to_dict()
                )
                exp_time_percentiles = (
                    valid_expansions["expansion_time_seconds"]
                    .quantile([0.25, 0.5, 0.75])
                    .to_dict()
                )

        period_stats[period["label"]] = {
            "total_fvgs": total,
            "mitigated_fvgs": mitigated,
            "mitigation_percentage": round(mitigation_pct, 2),
            "average_mitigation_time": avg_time,
            "expansion_size_percentiles": exp_size_percentiles,
            "expansion_time_percentiles": exp_time_percentiles,
        }

    return period_stats


def calculate_combined_time_stats(
    filtered_fvgs, fvg_filter_start_time, fvg_filter_end_time
):
    """
    Calculate combined day-of-week and time period statistics.

    Args:
        filtered_fvgs: DataFrame with FVG data
        fvg_filter_start_time: Start time for filtering
        fvg_filter_end_time: End time for filtering

    Returns:
        dict: Combined statistics with both day-of-week and time period breakdowns
    """
    day_stats = calculate_day_of_week_stats(filtered_fvgs)
    time_period_stats = calculate_time_period_stats(
        filtered_fvgs, fvg_filter_start_time, fvg_filter_end_time
    )

    # Calculate day-of-week breakdown for each time period
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    combined_stats = {}

    for day_name in day_names:
        day_fvgs = filtered_fvgs[
            filtered_fvgs["time_candle3"].dt.dayofweek == day_names.index(day_name)
        ]
        day_period_stats = calculate_time_period_stats(
            day_fvgs, fvg_filter_start_time, fvg_filter_end_time
        )
        combined_stats[day_name] = day_period_stats

    return {
        "day_of_week_stats": day_stats,
        "time_period_stats": time_period_stats,
        "combined_day_time_stats": combined_stats,
    }


def crosscheck_invalidated_fvgs(df_fvgs_per_bm, df_1min, df_5min, fvg_filter_end_time, session_start_time=None):
    """Crosscheck and plot all invalidated FVGs (mitigated but invalid expansions) per body_multiplier."""
    print(
        "\n[INFO] Starting crosscheck: Plotting all invalidated FVGs per body_multiplier..."
    )
    for bm, df_fvgs in df_fvgs_per_bm:
        # Filter to mitigated FVGs first
        mitigated_df = df_fvgs[df_fvgs["is_mitigated"]].copy()

        # Recompute validity with current logic to avoid cached inconsistencies
        print(
            f"\n[INFO] Body Multiplier {bm}: Recomputing expansion validity for {len(mitigated_df)} mitigated FVGs..."
        )
        truly_invalidated = []

        for idx, fvg_row in mitigated_df.iterrows():
            exp_size, exp_time, is_valid, expansion_details, _, _, _, _, _ = (
                calculate_expansion_after_mitigation(
                    df_1min, df_5min, fvg_row, fvg_filter_end_time, debug=False
                )
            )
            if not is_valid:  # Only collect truly invalid ones
                truly_invalidated.append(
                    (idx, fvg_row, exp_size, exp_time, expansion_details)
                )

        num_invalid = len(truly_invalidated)
        print(
            f"[INFO] Body Multiplier {bm}: {num_invalid} truly invalidated FVGs found after recomputation."
        )

        for counter, (idx, fvg_row, exp_size, exp_time, expansion_details) in enumerate(
            truly_invalidated, start=1
        ):
            print(
                f"\n[INFO] Displaying invalidated FVG {counter}/{num_invalid} for BM {bm}..."
            )
            # Show details with debug for this specific FVG
            _, _, is_valid_debug, _, _, _, _, _, _ = calculate_expansion_after_mitigation(
                df_1min, df_5min, fvg_row, fvg_filter_end_time, debug=True
            )
            print(
                f"FVG Details: Type={fvg_row['fvg_type']}, Time={fvg_row['time_candle3']}, Attempted Expansion Size={exp_size}, Time to Stop={exp_time}"
            )
            # Plot extended to session end
            plot_fvg_expansion(
                df_5min,
                fvg_row,
                exp_size,
                exp_time,
                expansion_details,
                window=20,
                fvg_filter_end_time=fvg_filter_end_time,
                extend_to_session_end=True,
                session_start_time=session_start_time,
            )
        print(f"[INFO] Finished crosscheck for BM {bm}.")


def crosscheck_valid_expansions(
    valid_fvgs_per_bm, df_1min, df_5min, fvg_filter_end_time, session_start_time=None, debug=False
):
    """Crosscheck and plot all valid expansions per body_multiplier."""
    print(
        "\n[INFO] Starting crosscheck: Plotting all valid expansions per body_multiplier..."
    )
    for bm, valid_fvgs in valid_fvgs_per_bm:
        num_valid = len(valid_fvgs)
        print(f"\n[INFO] Body Multiplier {bm}: {num_valid} valid expansions found.")
        for counter, (idx, fvg_row) in enumerate(valid_fvgs.iterrows(), start=1):
            print(
                f"\n[INFO] Displaying valid expansion {counter}/{num_valid} for BM {bm}..."
            )
            # Recompute for confirmation
            exp_size, exp_time, is_valid, expansion_details, _, _, _, _, _ = (
                calculate_expansion_after_mitigation(
                    df_1min, df_5min, fvg_row, fvg_filter_end_time, debug=False
                )
            )
            if is_valid:
                num_exp_candles = len(expansion_details)
                fvg_size = fvg_row["zone_high"] - fvg_row["zone_low"]
                print(
                    f"FVG Statistics:\n  Type: {fvg_row['fvg_type']}\n  Formation Time: {fvg_row['time_candle3']}\n  Zone Low: {fvg_row['zone_low']:.2f}\n  Zone High: {fvg_row['zone_high']:.2f}\n  FVG Size: {fvg_size:.2f}\n  Middle Open: {fvg_row['middle_open']:.2f}\n  Middle Low: {fvg_row['middle_low']:.2f}\n  Middle High: {fvg_row['middle_high']:.2f}\n  Mitigation Time: {fvg_row['mitigation_time']}\n  Time to Mitigation: {fvg_row['time_to_mitigation']}\n  Expansion Size: {exp_size}\n  Expansion Time: {exp_time}\n  Expansion Candles Count: {num_exp_candles}"
                )
                plot_fvg_expansion(
                    df_5min,
                    fvg_row,
                    exp_size,
                    exp_time,
                    expansion_details,
                    window=20,
                    fvg_filter_end_time=fvg_filter_end_time,
                    extend_to_session_end=True,
                    session_start_time=session_start_time,
                )
            else:
                print("Skipping: Expansion no longer valid on recompute.")
        print(f"[INFO] Finished crosscheck for BM {bm}.")


def crosscheck_all_fvgs(size_range, time_period, df_1min, df_5min, df_fvgs, fvg_filter_end_time, session_start_time=None, validation_filter="all"):
    """
    Merged function that displays all FVGs one by one (both valid expansions and invalidated FVGs).
    
    Args:
        size_range: Tuple of (min_size, max_size) for FVG filtering
        time_period: Tuple of (start_time, end_time) as time objects for filtering FVGs by formation time
        df_1min: 1-minute market data
        df_5min: 5-minute market data
        df_fvgs: DataFrame containing all FVG data
        fvg_filter_end_time: End time for filtering
        session_start_time: Session start time (optional)
        validation_filter: Filter by validation status - "valid", "invalid", or "all" (default: "all")
    """
    min_size, max_size = size_range
    start_time, end_time = time_period
    
    validation_text = validation_filter.upper() if validation_filter != "all" else "ALL"
    print(f"\n[INFO] Starting merged crosscheck: Displaying {validation_text} FVGs with size {min_size}-{max_size}, time {start_time}-{end_time}...")
    
    # Filter FVGs by size range
    df_fvgs['fvg_size'] = df_fvgs["zone_high"] - df_fvgs["zone_low"]
    size_filtered_fvgs = df_fvgs[
        (df_fvgs['fvg_size'] >= min_size) & 
        (df_fvgs['fvg_size'] <= max_size)
    ].copy()
    
    # Filter FVGs by time period (formation time)
    time_filtered_fvgs = size_filtered_fvgs[
        (size_filtered_fvgs["time_candle3"].dt.time >= start_time) &
        (size_filtered_fvgs["time_candle3"].dt.time <= end_time)
    ].copy()
    
    if time_filtered_fvgs.empty:
        print(f"[WARNING] No FVGs found with size {min_size}-{max_size} and time {start_time}-{end_time}")
        return
    
    # Apply validation filtering
    if validation_filter.lower() == "valid":
        # Only mitigated FVGs with valid expansions
        mitigated_fvgs = time_filtered_fvgs[time_filtered_fvgs["is_mitigated"]].copy()
        if mitigated_fvgs.empty:
            print(f"[WARNING] No mitigated FVGs found to check for valid expansions")
            return
            
        # Compute expansion validity for filtering
        valid_fvgs = []
        for idx, fvg_row in mitigated_fvgs.iterrows():
            _, _, is_valid, _, _, _, _, _, _ = calculate_expansion_after_mitigation(
                df_1min, df_5min, fvg_row, fvg_filter_end_time, debug=False
            )
            if is_valid:
                valid_fvgs.append((idx, fvg_row))
        
        if not valid_fvgs:
            print(f"[WARNING] No valid expansions found")
            return
            
        filtered_fvgs = pd.DataFrame([row for _, row in valid_fvgs])
        
    elif validation_filter.lower() == "invalid":
        # Only mitigated FVGs with invalid expansions
        mitigated_fvgs = time_filtered_fvgs[time_filtered_fvgs["is_mitigated"]].copy()
        if mitigated_fvgs.empty:
            print(f"[WARNING] No mitigated FVGs found to check for invalid expansions")
            return
            
        # Compute expansion validity for filtering
        invalid_fvgs = []
        for idx, fvg_row in mitigated_fvgs.iterrows():
            _, _, is_valid, _, _, _, _, _, _ = calculate_expansion_after_mitigation(
                df_1min, df_5min, fvg_row, fvg_filter_end_time, debug=False
            )
            if not is_valid:
                invalid_fvgs.append((idx, fvg_row))
        
        if not invalid_fvgs:
            print(f"[WARNING] No invalid expansions found")
            return
            
        filtered_fvgs = pd.DataFrame([row for _, row in invalid_fvgs])
        
    else:
        # Show all FVGs (default)
        filtered_fvgs = time_filtered_fvgs
    
    total_fvgs = len(filtered_fvgs)
    mitigated_fvgs = filtered_fvgs[filtered_fvgs["is_mitigated"]].copy()
    
    print(f"[INFO] Found {total_fvgs} {validation_text} FVGs with size {min_size}-{max_size} and time {start_time}-{end_time}")
    print(f"[INFO] {len(mitigated_fvgs)} are mitigated")
    
    # Process all FVGs one by one
    for counter, (idx, fvg_row) in enumerate(filtered_fvgs.iterrows(), start=1):
        print(f"\n{'='*60}")
        print(f"[INFO] Processing FVG {counter}/{total_fvgs}")
        
        # Basic FVG information
        fvg_size = fvg_row["zone_high"] - fvg_row["zone_low"]
        print(f"FVG Basic Info:")
        print(f"  Type: {fvg_row['fvg_type']}")
        print(f"  Formation Time: {fvg_row['time_candle3']}")
        print(f"  Zone Low: {fvg_row['zone_low']:.2f}")
        print(f"  Zone High: {fvg_row['zone_high']:.2f}")
        print(f"  FVG Size: {fvg_size:.2f}")
        print(f"  Middle Open: {fvg_row['middle_open']:.2f}")
        print(f"  Middle Low: {fvg_row['middle_low']:.2f}")
        print(f"  Middle High: {fvg_row['middle_high']:.2f}")
        print(f"  Is Mitigated: {fvg_row['is_mitigated']}")
        
        if fvg_row['is_mitigated']:
            print(f"  Mitigation Time: {fvg_row['mitigation_time']}")
            print(f"  Time to Mitigation: {fvg_row['time_to_mitigation']}")
            
            # Recompute expansion for current validation
            exp_size, exp_time, is_valid, expansion_details, _, _, _, _, _ = (
                calculate_expansion_after_mitigation(
                    df_1min, df_5min, fvg_row, fvg_filter_end_time, debug=False
                )
            )

            print(f"  Expansion Size: {exp_size}")
            print(f"  Expansion Time: {exp_time}")
            print(f"  Is Valid Expansion: {is_valid}")
            
            if expansion_details:
                print(f"  Expansion Candles Count: {len(expansion_details)}")
            
            # Determine category and display accordingly
            if is_valid:
                print(f"[INFO] ✅ VALID EXPANSION - Plotting...")
            else:
                print(f"[INFO] ❌ INVALIDATED FVG - Plotting...")
                # Show debug details for invalidated FVGs
                _, _, _, _, _, _, _, _, _ = calculate_expansion_after_mitigation(
                    df_1min, df_5min, fvg_row, fvg_filter_end_time, debug=True
                )
            
            # Plot the FVG expansion with focused window
            plot_fvg_expansion(
                df_5min,
                fvg_row,
                exp_size,
                exp_time,
                expansion_details,
                window=30,  # Show more context around FVG
                fvg_filter_end_time=fvg_filter_end_time,
                extend_to_session_end=False,  # Focus on FVG area instead of entire session
                session_start_time=session_start_time,
            )
        else:
            print(f"[INFO] 🟡 NON-MITIGATED FVG - No expansion analysis needed")
        
        # Wait for user input to continue to next FVG
        user_input = input(f"\nPress Enter to continue to next FVG, 'q' to quit, or 's' to skip to summary: ").strip().lower()
        if user_input == 'q':
            print("[INFO] User requested quit. Stopping FVG analysis.")
            break
        elif user_input == 's':
            print("[INFO] User requested skip to summary. Jumping to summary...")
            break
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"[INFO] SUMMARY for size {min_size}-{max_size}, time {start_time}-{end_time}:")
    
    mitigated_count = len(mitigated_fvgs)
    if mitigated_count > 0:
        # Recompute all expansions for final statistics
        valid_count = 0
        invalid_count = 0
        
        for _, fvg_row in mitigated_fvgs.iterrows():
            _, _, is_valid, _, _, _, _, _, _ = calculate_expansion_after_mitigation(
                df_1min, df_5min, fvg_row, fvg_filter_end_time, debug=False
            )
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        
        mitigation_rate = (mitigated_count / total_fvgs) * 100
        invalidation_rate = (invalid_count / mitigated_count) * 100 if mitigated_count > 0 else 0
        
        print(f"  Total FVGs: {total_fvgs}")
        print(f"  Mitigated FVGs: {mitigated_count} ({mitigation_rate:.1f}%)")
        print(f"  Valid Expansions: {valid_count}")
        print(f"  Invalidated FVGs: {invalid_count} ({invalidation_rate:.1f}%)")
    else:
        print(f"  Total FVGs: {total_fvgs}")
        print(f"  No mitigated FVGs found")
    
    print(f"[INFO] Merged crosscheck analysis complete!")
