import os
from datetime import datetime, time, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from matplotlib.patches import Rectangle

try:
    # Try relative import first (when used as a module)
    from ..config import (
        fvg_filter_end_time,
        min_expansion_size,
        min_search_bars_1min,
        min_search_bars_5min,
        mitigation_same_day,
        ny_tz,
    )
except ImportError:
    # Fall back to absolute import (when run from logic directory)
    from config import (
        fvg_filter_end_time,
        min_expansion_size,
        min_search_bars_1min,
        min_search_bars_5min,
        mitigation_same_day,
        ny_tz,
    )

from .fvg_detection import detect_fvg
from .time_utils import assign_time_period_to_fvgs, create_time_intervals


def study_specific_fvg(
    target_date,
    target_time,
    contract_symbol="NQU5",
    lookback_period=10,
    chart_hours_before=2,
    chart_hours_after=4,
    debug=True,
):
    """
    Study a specific FVG by date and time, providing detailed analysis and visualization.

    Args:
        target_date (str): Date in format 'YYYY-MM-DD' (e.g., '2024-12-15')
        target_time (str): Time in format 'HH:MM' (e.g., '10:30') in NY timezone
        contract_symbol (str): Contract symbol (default: 'NQU5')
        lookback_period (int): Lookback period for average body size calculation (default: 10)
        chart_hours_before (int): Hours of data to show before FVG formation (default: 2)
        chart_hours_after (int): Hours of data to show after FVG formation (default: 4)
        debug (bool): Enable debug output (default: True)

    Returns:
        dict: Detailed FVG analysis including formation data, mitigation info, and chart
    """

    try:
        # Parse target datetime
        target_datetime_str = f"{target_date} {target_time}"
        target_datetime_ny = ny_tz.localize(
            datetime.strptime(target_datetime_str, "%Y-%m-%d %H:%M")
        )
        target_datetime_utc = target_datetime_ny.astimezone(pytz.utc)

        if debug:
            print(
                f"[DEBUG] Searching for FVG formed at: {target_datetime_ny} (NY) / {target_datetime_utc} (UTC)"
            )

        # Define search window (look for FVG formation within ±30 minutes)
        search_window = timedelta(minutes=30)
        search_start = target_datetime_utc - search_window
        search_end = target_datetime_utc + search_window

        # Load cached data
        cache_file = f"data_cache/{contract_symbol}_5mins.parquet"
        if not os.path.exists(cache_file):
            print(f"[ERROR] Cache file not found: {cache_file}")
            return None

        df_5min = pd.read_parquet(cache_file)
        df_5min.index = pd.to_datetime(df_5min.index, utc=True)

        # Filter data around target time for analysis
        chart_start = target_datetime_utc - timedelta(hours=chart_hours_before)
        chart_end = target_datetime_utc + timedelta(hours=chart_hours_after)
        chart_data = df_5min[
            (df_5min.index >= chart_start) & (df_5min.index <= chart_end)
        ].copy()

        if chart_data.empty:
            print(f"[ERROR] No data found in time range {chart_start} to {chart_end}")
            return None

        # Reset index for FVG detection
        chart_data.reset_index(inplace=True)
        chart_data.rename(columns={"index": "date"}, inplace=True)

        if debug:
            print(
                f"[DEBUG] Chart data range: {chart_data['date'].min()} to {chart_data['date'].max()}"
            )
            print(f"[DEBUG] Total candles in chart: {len(chart_data)}")

        # Detect FVGs in the chart data
        fvg_list = detect_fvg(chart_data)

        # Find FVGs near the target time
        target_fvgs = []
        for i, fvg in enumerate(fvg_list):
            if fvg is not None:
                (
                    fvg_type,
                    y0,
                    y1,
                    time_candle1,
                    time_candle2,
                    time_candle3,
                    idx,
                    middle_open,
                    middle_low,
                    middle_high,
                    first_open,
                ) = fvg

                # Check if any of the formation candles are within our search window
                formation_times = [time_candle1, time_candle2, time_candle3]
                for formation_time in formation_times:
                    if search_start <= formation_time <= search_end:
                        target_fvgs.append((i, fvg))
                        break

        if not target_fvgs:
            print(f"[ERROR] No FVG found near target time {target_datetime_ny}")
            print(f"[INFO] Search window: {search_start} to {search_end}")

            # Show nearby FVGs for reference
            nearby_fvgs = []
            for i, fvg in enumerate(fvg_list):
                if fvg is not None:
                    (
                        fvg_type,
                        y0,
                        y1,
                        time_candle1,
                        time_candle2,
                        time_candle3,
                        idx,
                        middle_open,
                        middle_low,
                        middle_high,
                        first_open,
                    ) = fvg
                    time_diff = abs(
                        (time_candle2 - target_datetime_utc).total_seconds() / 60
                    )  # minutes
                    if time_diff <= 120:  # Within 2 hours
                        nearby_fvgs.append((time_candle2, fvg_type, time_diff))

            if nearby_fvgs:
                print(f"[INFO] Nearby FVGs found:")
                for fvg_time, fvg_type, time_diff in sorted(
                    nearby_fvgs, key=lambda x: x[2]
                ):
                    fvg_time_ny = fvg_time.astimezone(ny_tz)
                    print(
                        f"  - {fvg_type.upper()} FVG at {fvg_time_ny.strftime('%Y-%m-%d %H:%M')} (±{time_diff:.0f} min)"
                    )

            return None

        # If multiple FVGs found, select the closest one
        if len(target_fvgs) > 1:
            closest_fvg = min(
                target_fvgs,
                key=lambda x: abs((x[1][4] - target_datetime_utc).total_seconds()),
            )
            target_fvgs = [closest_fvg]
            if debug:
                print(f"[DEBUG] Multiple FVGs found, selected closest one")

        fvg_index, target_fvg = target_fvgs[0]
        fvg_type, y0, y1, time_candle1, time_candle2, time_candle3, idx, middle_open, middle_low, middle_high, first_open = (
            target_fvg
        )

        # Convert times to NY timezone for display
        time_candle1_ny = time_candle1.astimezone(ny_tz)
        time_candle2_ny = time_candle2.astimezone(ny_tz)
        time_candle3_ny = time_candle3.astimezone(ny_tz)

        # Calculate FVG metrics
        fvg_size = abs(y1 - y0)
        fvg_midpoint = (y0 + y1) / 2

        # Get the middle candle data for body size calculation
        middle_candle = chart_data.iloc[idx - 1]
        middle_body = abs(middle_candle["close"] - middle_candle["open"])

        # Calculate average body size for context
        start_idx = max(0, idx - 1 - lookback_period)
        if start_idx < idx - 1:
            prev_bodies = (
                chart_data["close"].iloc[start_idx : idx - 1]
                - chart_data["open"].iloc[start_idx : idx - 1]
            ).abs()
            avg_body_size = prev_bodies.mean()
        else:
            avg_body_size = 0.001

        body_ratio = middle_body / avg_body_size if avg_body_size > 0 else 0

        # Load 1-minute data for mitigation analysis
        cache_file_1min = f"data_cache/{contract_symbol}_1min.parquet"
        mitigation_info = None

        if os.path.exists(cache_file_1min):
            df_1min = pd.read_parquet(cache_file_1min)
            df_1min.index = pd.to_datetime(df_1min.index, utc=True)

            # Find mitigation in 1-minute data
            mitigation_start = time_candle3
            mitigation_end = mitigation_start + timedelta(hours=8)  # Look 8 hours ahead

            mitigation_data = df_1min[
                (df_1min.index >= mitigation_start) & (df_1min.index <= mitigation_end)
            ]

            if not mitigation_data.empty:
                mitigation_info = analyze_fvg_mitigation(
                    mitigation_data, fvg_type, y0, y1, fvg_size
                )

        # Create detailed analysis
        analysis = {
            "fvg_info": {
                "type": fvg_type,
                "formation_time": time_candle3_ny.strftime("%Y-%m-%d %H:%M:%S"),
                "candle1_time": time_candle1_ny.strftime("%Y-%m-%d %H:%M:%S"),
                "candle2_time": time_candle2_ny.strftime("%Y-%m-%d %H:%M:%S"),
                "candle3_time": time_candle3_ny.strftime("%Y-%m-%d %H:%M:%S"),
                "upper_level": max(y0, y1),
                "lower_level": min(y0, y1),
                "midpoint": fvg_midpoint,
                "size": fvg_size,
                "middle_body_size": middle_body,
                "avg_body_size": avg_body_size,
                "body_ratio": body_ratio,
            },
            "mitigation_info": mitigation_info,
            "chart_data": chart_data,
            "fvg_index": fvg_index,
        }

        # Print detailed analysis
        print(f"\n{'='*60}")
        print(f"FVG ANALYSIS REPORT")
        print(f"{'='*60}")
        print(f"FVG Type: {fvg_type.upper()}")
        print(f"Formation Time: {time_candle3_ny.strftime('%Y-%m-%d %H:%M:%S')} (NY)")
        print(f"Formation Candles:")
        print(f"  Candle 1: {time_candle1_ny.strftime('%H:%M:%S')}")
        print(f"  Candle 2: {time_candle2_ny.strftime('%H:%M:%S')} (middle/trigger)")
        print(f"  Candle 3: {time_candle3_ny.strftime('%H:%M:%S')}")
        print(f"\nFVG Levels:")
        print(f"  Upper: {max(y0, y1):.2f}")
        print(f"  Lower: {min(y0, y1):.2f}")
        print(f"  Midpoint: {fvg_midpoint:.2f}")
        print(f"  Size: {fvg_size:.2f} points")
        print(f"\nBody Analysis:")
        print(f"  Middle candle body: {middle_body:.2f}")
        print(f"  Average body size: {avg_body_size:.2f}")
        print(f"  Body ratio: {body_ratio:.2f}x")
        print(f"  Qualification: ✓ PASSED (size-based detection only)")

        if mitigation_info:
            print(f"\nMitigation Analysis:")
            print(f"  Status: {mitigation_info['status']}")
            print(f"  Time to mitigation: {mitigation_info['time_to_mitigation']}")
            print(f"  Mitigation price: {mitigation_info['mitigation_price']:.2f}")
            print(f"  Max expansion: {mitigation_info['max_expansion']:.2f}")
            print(f"  Expansion ratio: {mitigation_info['expansion_ratio']:.2f}")
        else:
            print(f"\nMitigation Analysis: No 1-minute data available")

        # Create visualization
        create_fvg_chart(analysis, chart_hours_before, chart_hours_after)

        return analysis

    except Exception as e:
        print(f"[ERROR] Error in study_specific_fvg: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def analyze_fvg_mitigation(mitigation_data, fvg_type, y0, y1, fvg_size):
    """
    Analyze FVG mitigation in 1-minute data.
    """
    upper_level = max(y0, y1)
    lower_level = min(y0, y1)
    midpoint = (upper_level + lower_level) / 2

    mitigation_price = None
    mitigation_time = None
    max_expansion = 0

    for i, (timestamp, row) in enumerate(mitigation_data.iterrows()):
        # Check for mitigation
        if fvg_type == "bullish":
            # Bullish FVG is mitigated when price touches lower level
            if row["low"] <= lower_level and mitigation_price is None:
                mitigation_price = lower_level
                mitigation_time = timestamp
            # Track maximum upward expansion
            expansion = row["high"] - upper_level
            max_expansion = max(max_expansion, expansion)
        else:
            # Bearish FVG is mitigated when price touches upper level
            if row["high"] >= upper_level and mitigation_price is None:
                mitigation_price = upper_level
                mitigation_time = timestamp
            # Track maximum downward expansion
            expansion = lower_level - row["low"]
            max_expansion = max(max_expansion, expansion)

    if mitigation_time is not None:
        time_to_mitigation = mitigation_time - mitigation_data.index[0]
        status = "MITIGATED"
    else:
        time_to_mitigation = None
        status = "NOT MITIGATED"

    expansion_ratio = max_expansion / fvg_size if fvg_size > 0 else 0

    return {
        "status": status,
        "mitigation_price": mitigation_price,
        "mitigation_time": mitigation_time,
        "time_to_mitigation": str(time_to_mitigation) if time_to_mitigation else None,
        "max_expansion": max_expansion,
        "expansion_ratio": expansion_ratio,
    }


def create_fvg_chart(analysis, chart_hours_before, chart_hours_after):
    """
    Create a detailed chart for the specific FVG analysis.
    """
    chart_data = analysis["chart_data"]
    fvg_info = analysis["fvg_info"]
    fvg_index = analysis["fvg_index"]

    # Convert dates to matplotlib format
    dates = [pd.to_datetime(d).to_pydatetime() for d in chart_data["date"]]

    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot candlesticks
    for i in range(len(chart_data)):
        row = chart_data.iloc[i]
        date = dates[i]

        # Determine candle color
        color = "green" if row["close"] > row["open"] else "red"

        # Draw the wick
        ax.plot([date, date], [row["low"], row["high"]], color="black", linewidth=1)

        # Draw the body
        body_height = abs(row["close"] - row["open"])
        body_bottom = min(row["open"], row["close"])

        rect = Rectangle(
            (mdates.date2num(date) - 0.0001, body_bottom),
            0.0002,
            body_height,
            facecolor=color,
            alpha=0.7,
        )
        ax.add_patch(rect)

    # Highlight FVG formation candles
    formation_indices = [
        fvg_index - 2,
        fvg_index - 1,
        fvg_index,
    ]  # candle1, candle2, candle3
    colors = ["blue", "orange", "purple"]
    labels = ["Candle 1", "Candle 2 (Trigger)", "Candle 3"]

    for i, (idx, color, label) in enumerate(zip(formation_indices, colors, labels)):
        if 0 <= idx < len(chart_data):
            date = dates[idx]
            row = chart_data.iloc[idx]

            # Highlight the candle
            rect = Rectangle(
                (mdates.date2num(date) - 0.0002, row["low"]),
                0.0004,
                row["high"] - row["low"],
                facecolor=color,
                alpha=0.3,
                label=label,
            )
            ax.add_patch(rect)

    # Draw FVG zone
    upper_level = fvg_info["upper_level"]
    lower_level = fvg_info["lower_level"]

    # FVG rectangle
    fvg_start = dates[0]
    fvg_end = dates[-1]
    fvg_rect = Rectangle(
        (mdates.date2num(fvg_start), lower_level),
        mdates.date2num(fvg_end) - mdates.date2num(fvg_start),
        upper_level - lower_level,
        facecolor="yellow",
        alpha=0.2,
        label="FVG Zone",
    )
    ax.add_patch(fvg_rect)

    # Draw FVG levels
    ax.axhline(
        y=upper_level,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Upper Level ({upper_level:.2f})",
    )
    ax.axhline(
        y=lower_level,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Lower Level ({lower_level:.2f})",
    )
    ax.axhline(
        y=fvg_info["midpoint"],
        color="orange",
        linestyle=":",
        alpha=0.6,
        label=f'Midpoint ({fvg_info["midpoint"]:.2f})',
    )

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Labels and title
    ax.set_xlabel("Time (NY)")
    ax.set_ylabel("Price")
    ax.set_title(
        f'{fvg_info["type"].upper()} FVG Analysis - {fvg_info["formation_time"]}\n'
        f'Size: {fvg_info["size"]:.2f} pts | Body Ratio: {fvg_info["body_ratio"]:.2f}x | Size-Based Detection'
    )

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Adjust layout and save
    plt.tight_layout()

    # Save the chart
    chart_filename = f'fvg_study_{fvg_info["formation_time"].replace(":", "").replace("-", "").replace(" ", "_")}.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches="tight")
    print(f"\n[INFO] Chart saved as: {chart_filename}")

    plt.show()


def find_fvg_mitigations(
    df_1min,
    df_5min,
    fvg_list_5min,
    fvg_filter_start_time,
    fvg_filter_end_time,
    debug=False,
    search_bars_primary=None,
    search_bars_fallback=None,
    primary_label=None,
    fallback_label=None,
):
    if search_bars_primary is None:
        search_bars_primary = min_search_bars_1min
    if search_bars_fallback is None:
        search_bars_fallback = min_search_bars_5min
    if primary_label is None:
        primary_label = "1min"
    if fallback_label is None:
        fallback_label = "5min"
    fvg_results = []
    for fvg in fvg_list_5min:
        if fvg is None:
            continue
        (
            fvg_type,
            y0,
            y1,
            time_candle1,
            time_candle2,
            time_candle3,
            idx_third_candle_5min,
            middle_open,
            middle_low,
            middle_high,
            first_open,
        ) = fvg
        zone_low, zone_high = sorted([y0, y1])
        fvg_data = {
            "fvg_type": fvg_type,
            "zone_low": zone_low,
            "zone_high": zone_high,
            "time_candle1": time_candle1,
            "time_candle2": time_candle2,
            "time_candle3": time_candle3,
            "fvg_5min_idx": idx_third_candle_5min,
            "middle_open": middle_open,
            "middle_low": middle_low,
            "middle_high": middle_high,
            "first_open": first_open,
            "is_mitigated": False,
            "mitigation_time": None,
            "time_to_mitigation": None,
            "mitigation_idx": None,
            "mitigation_method": None,
        }
        if idx_third_candle_5min + 1 >= len(df_5min):
            fvg_results.append(fvg_data)
            continue
        search_start_time_1min = df_5min["date"].iloc[idx_third_candle_5min + 1]
        start_search_1min_idx = df_1min["date"].searchsorted(
            search_start_time_1min, side="left"
        )

        # Calculate max search end for primary timeframe
        end_search_1min_idx = min(
            start_search_1min_idx + search_bars_primary, len(df_1min)
        )
        if mitigation_same_day:
            formation_date = time_candle3.date()
            same_day_end = ny_tz.localize(
                datetime.combine(formation_date, time(17, 00, 00))
            )
            same_day_idx = df_1min["date"].searchsorted(same_day_end, side="right")
            end_search_1min_idx = min(end_search_1min_idx, same_day_idx)

        mitigation_found = False
        for j in range(start_search_1min_idx, end_search_1min_idx):
            if j >= len(df_1min):
                break
            low_1min = df_1min["low"].iloc[j]
            high_1min = df_1min["high"].iloc[j]
            fill_time = df_1min["date"].iloc[j]
            if low_1min <= zone_high and high_1min >= zone_low:
                time_to_mitigation = fill_time - time_candle3
                fvg_data.update(
                    {
                        "is_mitigated": True,
                        "mitigation_time": fill_time,
                        "time_to_mitigation": time_to_mitigation,
                        "mitigation_idx": j,
                        "mitigation_method": primary_label,
                    }
                )
                mitigation_found = True
                break
        if not mitigation_found:
            # Calculate end time of 1min window
            search_end_time_1min = (
                df_1min["date"].iloc[end_search_1min_idx - 1]
                if end_search_1min_idx > start_search_1min_idx
                else search_start_time_1min
            )
            start_search_5min_idx = df_5min["date"].searchsorted(
                search_end_time_1min, side="right"
            )
            end_search_5min_idx = min(
                start_search_5min_idx + search_bars_fallback, len(df_5min)
            )
            if mitigation_same_day:
                formation_date = time_candle3.date()
                same_day_end = ny_tz.localize(
                    datetime.combine(formation_date, time(23, 59, 59))
                )
                same_day_idx = df_5min["date"].searchsorted(same_day_end, side="right")
                end_search_5min_idx = min(end_search_5min_idx, same_day_idx)

            for k in range(start_search_5min_idx, end_search_5min_idx):
                if k >= len(df_5min):
                    break
                low_5min = df_5min["low"].iloc[k]
                high_5min = df_5min["high"].iloc[k]
                fill_time = df_5min["date"].iloc[k]
                if low_5min <= zone_high and high_5min >= zone_low:
                    time_to_mitigation = fill_time - time_candle3
                    fvg_data.update(
                        {
                            "is_mitigated": True,
                            "mitigation_time": fill_time,
                            "time_to_mitigation": time_to_mitigation,
                            "mitigation_idx": k,
                            "mitigation_method": fallback_label,
                        }
                    )
                    mitigation_found = True
                    break
        # Print only if FVG is within the filtered time range
        # Extract time component from formation timestamp for comparison with time objects
        time_candle3_time = time_candle3.time()
        if (
            time_candle3_time >= fvg_filter_start_time
            and time_candle3_time <= fvg_filter_end_time
        ):
            if debug:
                if mitigation_found:
                    print(
                        f"✅ {fvg_type.capitalize()} FVG mitigated (FVG Date: {time_candle1.strftime('%Y-%m-%d')}, C1:{time_candle1.strftime('%H:%M')}, C2:{time_candle2.strftime('%H:%M')}, C3:{time_candle3.strftime('%H:%M')}) mitigated at {fill_time.strftime('%Y-%m-%d %H:%M:%S %Z')} (Time to mitigation: {time_to_mitigation}) via {fvg_data['mitigation_method']}"
                    )
                else:
                    print(
                        f"⚠️ {fvg_type.capitalize()} FVG (FVG Date: {time_candle1.strftime('%Y-%m-%d')}, C1:{time_candle1.strftime('%H:%M')}, C2:{time_candle2.strftime('%H:%M')}, C3:{time_candle3.strftime('%H:%M')}) was NOT mitigated within extended search windows."
                    )
        fvg_results.append(fvg_data)
    return fvg_results


def calculate_expansion_after_mitigation(
    df_1min, df_5min, fvg_data, fvg_filter_end_time, debug=False,
    expansion_size_threshold=None,
):
    if debug:
        print(
            f"Starting expansion calculation for {fvg_data['fvg_type']} FVG at {fvg_data['time_candle3']}"
        )
    if not fvg_data["is_mitigated"]:
        if debug:
            print("FVG not mitigated, skipping.")
        return None, None, False, [], 0.0, 0, 0, None, None, 0.0, 0.0
    mitigation_time = fvg_data["mitigation_time"]
    fvg_type = fvg_data["fvg_type"]
    zone_low = fvg_data["zone_low"]
    zone_high = fvg_data["zone_high"]
    middle_open = fvg_data["middle_open"]
    middle_low = fvg_data["middle_low"]
    middle_high = fvg_data["middle_high"]
    first_open = fvg_data.get("first_open", None)
    fvg_size = zone_high - zone_low
    if debug:
        print(f"FVG Size: {fvg_size}")
    if fvg_type == "bullish":
        fill_price = zone_high
    else:
        fill_price = zone_low
    if debug:
        print(f"Fill Price: {fill_price}")
    start_idx = df_5min["date"].searchsorted(mitigation_time, side="right")
    if start_idx >= len(df_5min):
        if debug:
            print("No post-mitigation data.")
        return 0.0, timedelta(0), False, [], 0.0, 0, 0, None, None, 0.0, 0.0
    mitigation_date = mitigation_time.date()
    # Handle case where fvg_filter_end_time is already a time object
    if hasattr(fvg_filter_end_time, "time"):
        end_time_obj = fvg_filter_end_time.time()
    else:
        end_time_obj = fvg_filter_end_time
    end_time = datetime.combine(mitigation_date, end_time_obj)
    end_time = ny_tz.localize(end_time)
    if mitigation_time >= end_time:
        if debug:
            print("Mitigation after end time.")
        return 0.0, timedelta(0), False, [], 0.0, 0, 0, None, None, 0.0, 0.0
    end_idx = df_5min["date"].searchsorted(end_time, side="right")
    post_mitigation_df = df_5min.iloc[start_idx:end_idx]
    if post_mitigation_df.empty:
        if debug:
            print("Empty post-mitigation dataframe.")
        return 0.0, timedelta(0), False, [], 0.0, 0, 0, None, None, 0.0, 0.0
    max_expansion = 0.0
    exp_time = timedelta(0)
    time_to_target = None  # Time to first reach expansion_size_threshold
    extrema = fill_price
    is_valid = False
    reached_size = False
    was_invalidated = False  # Track if expansion was actually invalidated
    time_to_invalidation = None  # Time from mitigation to invalidation
    expansion_details = []
    max_expansion_before_invalidation = 0.0
    exp_time_before_invalidation = timedelta(0)
    max_penetration_depth = 0.0
    penetration_candle_count = 0
    midpoint_crossing_count = 0
    fvg_midpoint = (zone_high + zone_low) / 2

    # Risk-Reward: compute risk (entry-to-stop distance, touch-based)
    if fvg_type == "bullish":
        risk_points = round((zone_high - middle_low) * 4) / 4
    else:
        risk_points = round((middle_high - zone_low) * 4) / 4
    stop_touched = False
    max_expansion_before_stop = 0.0

    # Compute max possible penetration depth (from mitigation level to Candle 1 open)
    if first_open is not None:
        if fvg_type == "bullish":
            max_possible_penetration = zone_high - first_open
        else:
            max_possible_penetration = first_open - zone_low
        if max_possible_penetration <= 0:
            max_possible_penetration = 0.0
    else:
        max_possible_penetration = None

    # Track only contiguous penetration (stop once a candle exits the zone)
    penetration_ended = False

    # Check the 5-min candle containing the mitigation (skipped by the main loop)
    if max_possible_penetration is not None and max_possible_penetration > 0 and start_idx > 0:
        mit_candle = df_5min.iloc[start_idx - 1]
        if fvg_type == "bullish" and mit_candle["low"] < zone_high:
            penetration_candle_count += 1
            if mit_candle["low"] <= fvg_midpoint:
                midpoint_crossing_count += 1
            depth = min(zone_high - mit_candle["low"], max_possible_penetration)
            max_penetration_depth = max(max_penetration_depth, depth)
        elif fvg_type == "bearish" and mit_candle["high"] > zone_low:
            penetration_candle_count += 1
            if mit_candle["high"] >= fvg_midpoint:
                midpoint_crossing_count += 1
            depth = min(mit_candle["high"] - zone_low, max_possible_penetration)
            max_penetration_depth = max(max_penetration_depth, depth)

    if debug:
        print("Processing post-mitigation candles:")
    for idx, row in post_mitigation_df.iterrows():
        if debug:
            print(
                f"Candle {idx}: Time={row['date']}, Open={row['open']}, High={row['high']}, Low={row['low']}, Close={row['close']}"
            )
        low = row["low"]
        high = row["high"]
        close = row["close"]
        open_ = row["open"]
        current_time = row["date"]
        is_bullish_candle = close > open_
        is_bearish_candle = close < open_

        # Touch-based stop check (for RR calculation — stop order fires on wick)
        if not stop_touched and risk_points > 0:
            if (fvg_type == "bullish" and low <= middle_low) or \
               (fvg_type == "bearish" and high >= middle_high):
                stop_touched = True

        # Track contiguous penetration depth (stop once a candle exits the zone)
        if not penetration_ended and max_possible_penetration is not None and max_possible_penetration > 0:
            is_penetrating = (fvg_type == "bullish" and low < zone_high) or (fvg_type == "bearish" and high > zone_low)
            if is_penetrating:
                penetration_candle_count += 1
                if fvg_type == "bullish":
                    if low <= fvg_midpoint:
                        midpoint_crossing_count += 1
                    depth = min(zone_high - low, max_possible_penetration)
                else:
                    if high >= fvg_midpoint:
                        midpoint_crossing_count += 1
                    depth = min(high - zone_low, max_possible_penetration)
                max_penetration_depth = max(max_penetration_depth, depth)
            else:
                penetration_ended = True

        # Check for FVG invalidation BEFORE calculating expansion for this candle
        # Bullish FVG invalidated when close < middle candle low
        # Bearish FVG invalidated when close > middle candle high
        if (fvg_type == "bearish" and close > middle_high) or (
            fvg_type == "bullish" and close < middle_low
        ):
            if debug:
                print("Stopped: Invalidation condition met.")
            was_invalidated = True
            time_to_invalidation = current_time - mitigation_time
            # Use expansion from before this invalidating candle
            max_expansion = max_expansion_before_invalidation
            exp_time = exp_time_before_invalidation
            break

        # Calculate expansion based on FVG type
        if fvg_type == "bullish":
            new_extrema = max(extrema, high)
            current_exp = new_extrema - fill_price
            if debug:
                print(
                    f"Updated extrema to {new_extrema}, current expansion: {current_exp}"
                )
        else:
            new_extrema = min(extrema, low)
            current_exp = fill_price - new_extrema
            if debug:
                print(
                    f"Updated extrema to {new_extrema}, current expansion: {current_exp}"
                )

        # Update extrema for next iteration
        extrema = new_extrema

        # Track the maximum expansion (this becomes the "before invalidation" value for next iteration)
        if current_exp > max_expansion:
            max_expansion = current_exp
            exp_time = current_time - mitigation_time
            # Record first time expansion crosses the target threshold
            if time_to_target is None and expansion_size_threshold is not None and current_exp >= expansion_size_threshold:
                time_to_target = exp_time
            if debug:
                print(f"New max expansion: {max_expansion} at time {exp_time}")

        # Track max expansion before stop is touched (for RR calculation)
        if not stop_touched:
            max_expansion_before_stop = max(max_expansion_before_stop, current_exp)

        # Store the current max expansion as "before invalidation" for potential next iteration
        max_expansion_before_invalidation = max_expansion
        exp_time_before_invalidation = exp_time

        expansion_details.append(
            {"time": current_time, "expansion_size": round(current_exp * 4) / 4}
        )
        # Removed break for size reached

    effective_min_expansion = expansion_size_threshold if expansion_size_threshold is not None else min_expansion_size
    meets_min_size = max_expansion >= effective_min_expansion
    # Valid if size requirements are met (before any potential invalidation)
    is_valid = meets_min_size
    # Note: was_invalidated is tracked separately but does not affect validity
    rounded_exp = round(max_expansion * 4) / 4 if max_expansion > 0 else 0.0
    if debug:
        print(
            f"Final Expansion Size: {rounded_exp}, Time: {exp_time}, Meets Min Size: {meets_min_size}, Was Invalidated: {was_invalidated}, Valid: {is_valid}"
        )
    rounded_pen = round(max_penetration_depth * 4) / 4 if max_penetration_depth > 0 else 0.0
    time_to_target_seconds = time_to_target.total_seconds() if time_to_target is not None else None
    time_to_invalidation_seconds = time_to_invalidation.total_seconds() if time_to_invalidation is not None else None
    # Compute achieved RR based on max expansion before stop was touched
    rounded_exp_before_stop = round(max_expansion_before_stop * 4) / 4 if max_expansion_before_stop > 0 else 0.0
    if risk_points > 0 and rounded_exp_before_stop > 0:
        achieved_rr = round(rounded_exp_before_stop / risk_points, 4)
    else:
        achieved_rr = 0.0
    return rounded_exp, exp_time, is_valid, expansion_details, rounded_pen, penetration_candle_count, midpoint_crossing_count, time_to_target_seconds, time_to_invalidation_seconds, risk_points, achieved_rr


def calculate_fvg_size(df_fvgs):
    """Calculate FVG size in points: zone_high - zone_low for each FVG, rounded to nearest 0.25."""
    if (
        df_fvgs.empty
        or "zone_high" not in df_fvgs.columns
        or "zone_low" not in df_fvgs.columns
    ):
        return df_fvgs
    df_fvgs["fvg_size"] = df_fvgs["zone_high"] - df_fvgs["zone_low"]
    df_fvgs["fvg_size"] = (round(df_fvgs["fvg_size"] * 4) / 4).where(
        df_fvgs["fvg_size"].notna(), None
    )
    return df_fvgs


def get_random_fvg_with_expansion(
    filtered_fvgs,
    df_1min,
    df_5min,
    fvg_filter_end_time,
    debug=False,
    plot=False,
):
    if filtered_fvgs.empty:
        if debug:
            print("No FVGs available.")
        return None, None, None, None  # Return None for bm too

    mitigated_fvgs = filtered_fvgs[filtered_fvgs["is_mitigated"]]
    if mitigated_fvgs.empty:
        if debug:
            print("No mitigated FVGs available.")
        return None, None, None, None

    max_attempts = len(mitigated_fvgs) * 2
    attempt = 0
    while attempt < max_attempts:
        random_fvg = mitigated_fvgs.sample(1).iloc[0]
        exp_size, exp_time, is_valid, expansion_details, _, _, _, _, _, _, _ = (
            calculate_expansion_after_mitigation(
                df_1min, df_5min, random_fvg, fvg_filter_end_time, debug=debug
            )
        )
        if is_valid:
            if debug:
                print("Selected FVG with Valid Expansion:")
            for key, value in random_fvg.items():
                print(f"{key}: {value}")
                print(f"Expansion Size: {exp_size}")
                print(f"Expansion Time: {exp_time}")
                print("Expansion Candles Details:")
                for detail in expansion_details:
                    print(
                        f"Time: {detail['time'].strftime('%Y-%m-%d %H:%M')}, Size: {detail['expansion_size']}"
                    )
            if plot:
                from .visualization_utils import plot_fvg_expansion

                plot_fvg_expansion(
                    df_5min,
                    random_fvg,
                    exp_size,
                    exp_time,
                    expansion_details,
                    window=20,
                    fvg_filter_end_time=fvg_filter_end_time,
                )
            return (
                random_fvg,
                exp_size,
                exp_time,
                None,  # Placeholder for backward compatibility
            )
        attempt += 1
    if debug:
        print("No FVG with valid expansion found after multiple attempts.")
    return None, None, None, None


def analyze_fvgs_by_time_period(df_fvgs, time_intervals):
    """Analyze FVG performance by time periods."""

    # Assign time periods to FVGs
    df_fvgs = assign_time_period_to_fvgs(df_fvgs, time_intervals)

    # Filter out FVGs without time periods
    df_fvgs_with_time = df_fvgs[df_fvgs["time_period"].notna()].copy()

    if df_fvgs_with_time.empty:
        return pd.DataFrame()

    results = []

    for time_start, time_end in time_intervals:
        time_period = f"{time_start.strftime('%H:%M')}-{time_end.strftime('%H:%M')}"

        # Filter FVGs for this time period
        period_fvgs = df_fvgs_with_time[
            df_fvgs_with_time["time_period"] == time_period
        ].copy()

        if period_fvgs.empty:
            continue

        # Calculate basic statistics
        total_fvgs = len(period_fvgs)
        mitigated_fvgs = int(period_fvgs["is_mitigated"].sum())
        mitigation_rate = (mitigated_fvgs / total_fvgs * 100) if total_fvgs > 0 else 0.0

        # Calculate expansion statistics for mitigated FVGs
        mitigated_df = period_fvgs[period_fvgs["is_mitigated"]].copy()

        valid_expansions = 0
        invalidated_expansions = 0
        expansion_sizes = []
        expansion_times = []

        if len(mitigated_df) > 0:
            # Check if expansion columns exist
            if "is_expansion_valid" in mitigated_df.columns:
                valid_exp_df = mitigated_df[mitigated_df["is_expansion_valid"]]
                valid_expansions = len(valid_exp_df)
                invalidated_expansions = len(mitigated_df) - valid_expansions

                if "expansion_size" in valid_exp_df.columns:
                    expansion_sizes = valid_exp_df["expansion_size"].dropna().tolist()
                if "expansion_time_seconds" in valid_exp_df.columns:
                    expansion_times = (
                        valid_exp_df["expansion_time_seconds"].dropna().tolist()
                    )

        # Calculate invalidation rate
        total_attempts = valid_expansions + invalidated_expansions
        invalidation_rate = (
            (invalidated_expansions / total_attempts * 100)
            if total_attempts > 0
            else 0.0
        )

        # Calculate expansion statistics
        avg_expansion = np.mean(expansion_sizes) if expansion_sizes else 0.0
        median_expansion = np.median(expansion_sizes) if expansion_sizes else 0.0

        # Calculate FVG size statistics
        if "fvg_size" in period_fvgs.columns:
            fvg_sizes = period_fvgs["fvg_size"].dropna()
            avg_fvg_size = fvg_sizes.mean() if len(fvg_sizes) > 0 else 0.0
            median_fvg_size = fvg_sizes.median() if len(fvg_sizes) > 0 else 0.0
        else:
            avg_fvg_size = 0.0
            median_fvg_size = 0.0

        # Calculate mitigation time statistics
        if len(mitigated_df) > 0 and "time_to_mitigation" in mitigated_df.columns:
            mitigation_times = mitigated_df["time_to_mitigation"].dropna()
            avg_mitigation_time = (
                mitigation_times.mean()
                if len(mitigation_times) > 0
                else pd.Timedelta(0)
            )
            p75_mitigation_time = (
                mitigation_times.quantile(0.75)
                if len(mitigation_times) > 0
                else pd.Timedelta(0)
            )
        else:
            avg_mitigation_time = pd.Timedelta(0)
            p75_mitigation_time = pd.Timedelta(0)

        # Calculate expansion efficiency
        expansion_efficiency = (
            (avg_expansion / avg_fvg_size) if avg_fvg_size > 0 else 0.0
        )

        # Calculate penetration depth statistics
        avg_penetration_depth = 0.0
        p75_penetration_depth = 0.0
        avg_penetration_candle_count = 0.0
        if len(mitigated_df) > 0 and "max_penetration_depth" in mitigated_df.columns:
            pen_depths = mitigated_df["max_penetration_depth"].dropna()
            if len(pen_depths) > 0:
                avg_penetration_depth = pen_depths.mean()
                p75_penetration_depth = pen_depths.quantile(0.75)
        if len(mitigated_df) > 0 and "penetration_candle_count" in mitigated_df.columns:
            pen_counts = mitigated_df["penetration_candle_count"].dropna()
            if len(pen_counts) > 0:
                avg_penetration_candle_count = pen_counts.mean()

        avg_midpoint_crossing_count = 0.0
        midpoint_crossed_pct = 0.0
        if len(mitigated_df) > 0 and "midpoint_crossing_count" in mitigated_df.columns:
            mid_counts = mitigated_df["midpoint_crossing_count"].dropna()
            if len(mid_counts) > 0:
                avg_midpoint_crossing_count = mid_counts.mean()
                midpoint_crossed_pct = round((mid_counts > 0).sum() / total_fvgs * 100, 2) if total_fvgs > 0 else 0.0

        result = {
            "time_period": time_period,
            "total_fvgs": total_fvgs,
            "mitigated_fvgs": mitigated_fvgs,
            "mitigation_rate": round(mitigation_rate, 2),
            "valid_expansions": valid_expansions,
            "invalidated_expansions": invalidated_expansions,
            "invalidation_rate": round(invalidation_rate, 2),
            "avg_expansion_size": round(avg_expansion, 2),
            "median_expansion_size": round(median_expansion, 2),
            "avg_fvg_size": round(avg_fvg_size, 2),
            "median_fvg_size": round(median_fvg_size, 2),
            "expansion_efficiency": round(expansion_efficiency, 2),
            "avg_mitigation_time": avg_mitigation_time,
            "p75_mitigation_time": p75_mitigation_time,
            "sample_size": len(period_fvgs),
            "avg_penetration_depth": round(avg_penetration_depth, 2),
            "p75_penetration_depth": round(p75_penetration_depth, 2),
            "avg_penetration_candle_count": round(avg_penetration_candle_count, 1),
            "avg_midpoint_crossing_count": round(avg_midpoint_crossing_count, 1),
            "midpoint_crossed_pct": midpoint_crossed_pct,
        }

        results.append(result)

    return pd.DataFrame(results)


def analyze_fvg_size_time_distribution(
    df_fvgs,
    min_fvg_sizes=None,
    fvg_filter_start_time=None,
    fvg_filter_end_time=None,
    interval_minutes=30,
    max_samples_per_cell=None,
    size_filtering_method="cumulative",
):
    """
    Analyze FVG invalidation rates across different minimum FVG size thresholds AND time periods.
    
    This simplified version works with already-detected FVGs and filters them by size thresholds
    instead of re-detecting with different body multipliers.

    Args:
        df_fvgs: DataFrame with detected FVGs (from main analysis)
        min_fvg_sizes: Array of minimum FVG sizes to test (in points)
        fvg_filter_start_time: Session start time for filtering
        fvg_filter_end_time: Session end time for filtering
        interval_minutes: Minutes for time intervals (default: 30)
        max_samples_per_cell: Maximum number of FVGs to analyze per size/time combination
        size_filtering_method: "cumulative" (fvg_size >= threshold) or "bins" (discrete size ranges)

    Returns:
        DataFrame: Comprehensive results with invalidation rates across size thresholds and time periods
    """
    # Set default minimum FVG sizes if not provided
    if min_fvg_sizes is None:
        from config import get_min_fvg_sizes_range
        min_fvg_sizes = get_min_fvg_sizes_range()

    # Set default time filters if not provided
    if fvg_filter_start_time is None or fvg_filter_end_time is None:
        from config import fvg_filter_start_time as default_start, fvg_filter_end_time as default_end
        fvg_filter_start_time = fvg_filter_start_time or default_start
        fvg_filter_end_time = fvg_filter_end_time or default_end

    # Create time intervals
    time_intervals = create_time_intervals(
        fvg_filter_start_time, fvg_filter_end_time, interval_minutes
    )

    # Ensure df_fvgs has FVG size calculated
    df_fvgs_processed = calculate_fvg_size(df_fvgs.copy())

    # Assign time periods to FVGs
    df_fvgs_processed = assign_time_period_to_fvgs(df_fvgs_processed, time_intervals)

    total_combinations = len(time_intervals) * len(min_fvg_sizes)

    print(
        f"[INFO] Analyzing FVG size distribution across {len(min_fvg_sizes)} minimum sizes and {len(time_intervals)} time intervals"
    )
    print(f"[INFO] Size filtering method: {size_filtering_method}")
    if size_filtering_method == "cumulative":
        print("[INFO] Using cumulative filtering: each threshold includes all FVGs >= size")
    elif size_filtering_method == "bins":
        print("[INFO] Using discrete bins: each FVG belongs to exactly one size range")
    print(f"[INFO] Total combinations to process: {total_combinations}")
    print(
        f"[INFO] Minimum FVG sizes: {min_fvg_sizes[0]:.2f} to {min_fvg_sizes[-1]:.2f} points"
    )
    print(
        f"[INFO] Time intervals: {len(time_intervals)} x {interval_minutes}-minute periods"
    )
    # Add detailed breakdown of FVG filtering
    original_count = len(df_fvgs)
    processed_count = len(df_fvgs_processed)
    time_filtered_count = len(df_fvgs_processed[df_fvgs_processed["time_period"].notna()])
    
    print(f"[INFO] === FVG FILTERING BREAKDOWN ===")
    print(f"[INFO] Original FVGs detected: {original_count:,}")
    print(f"[INFO] FVGs after preprocessing: {processed_count:,}")
    print(f"[INFO] FVGs with valid time periods: {time_filtered_count:,}")
    print(f"[INFO] FVGs excluded from analysis: {original_count - time_filtered_count:,}")
    
    if time_filtered_count != original_count:
        exclusion_reasons = []
        if processed_count < original_count:
            exclusion_reasons.append(f"{original_count - processed_count:,} lost during preprocessing")
        if time_filtered_count < processed_count:
            exclusion_reasons.append(f"{processed_count - time_filtered_count:,} outside session time window")
        print(f"[INFO] Exclusion reasons: {'; '.join(exclusion_reasons)}")
    
    # Add detailed size distribution analysis
    time_filtered_fvgs = df_fvgs_processed[df_fvgs_processed["time_period"].notna()]
    if not time_filtered_fvgs.empty and "fvg_size" in time_filtered_fvgs.columns:
        valid_sizes = time_filtered_fvgs["fvg_size"].notna()
        fvgs_with_valid_sizes = time_filtered_fvgs[valid_sizes]
        
        print(f"[INFO] === SIZE DISTRIBUTION ANALYSIS ===")
        print(f"[INFO] FVGs with valid sizes: {len(fvgs_with_valid_sizes):,}")
        print(f"[INFO] FVGs with missing/invalid sizes: {len(time_filtered_fvgs) - len(fvgs_with_valid_sizes):,}")
        
        if len(fvgs_with_valid_sizes) > 0:
            min_actual_size = fvgs_with_valid_sizes["fvg_size"].min()
            max_actual_size = fvgs_with_valid_sizes["fvg_size"].max()
            print(f"[INFO] Actual size range: {min_actual_size:.2f} to {max_actual_size:.2f} points")
            
            # Check how many FVGs fall within the analysis size range
            min_analysis_size = min(min_fvg_sizes)
            max_analysis_size = max(min_fvg_sizes)
            within_range = fvgs_with_valid_sizes[
                (fvgs_with_valid_sizes["fvg_size"] >= min_analysis_size) & 
                (fvgs_with_valid_sizes["fvg_size"] <= max_analysis_size)
            ]
            print(f"[INFO] FVGs within analysis range ({min_analysis_size:.2f}-{max_analysis_size:.2f}): {len(within_range):,}")
            print(f"[INFO] FVGs outside analysis range: {len(fvgs_with_valid_sizes) - len(within_range):,}")

    results = []
    total_processed_fvgs = 0  # Counter to track FVGs actually processed in bins
    
    # Iterate through each combination of time period and minimum size
    for time_period in df_fvgs_processed["time_period"].unique():
        if pd.isna(time_period):
            continue
            
        # Filter FVGs for this time period
        period_fvgs = df_fvgs_processed[df_fvgs_processed["time_period"] == time_period]
        
        for i, min_size in enumerate(min_fvg_sizes):
            if size_filtering_method == "cumulative":
                # Cumulative approach: all FVGs >= threshold
                size_filtered_fvgs = period_fvgs[period_fvgs["fvg_size"] >= min_size]
                size_range = f"{min_size:.2f}"
            elif size_filtering_method == "bins":
                # Discrete bins approach: FVGs in specific size range
                if i < len(min_fvg_sizes) - 1:
                    next_size = min_fvg_sizes[i + 1]
                    size_filtered_fvgs = period_fvgs[
                        (period_fvgs["fvg_size"] >= min_size) & 
                        (period_fvgs["fvg_size"] < next_size)
                    ]
                    size_range = f"{min_size:.2f}-{next_size:.2f}"
                else:
                    # Last bin: use the step size to determine upper bound
                    if len(min_fvg_sizes) > 1:
                        step_size = min_fvg_sizes[1] - min_fvg_sizes[0]
                        upper_bound = min_size + step_size
                        size_filtered_fvgs = period_fvgs[
                            (period_fvgs["fvg_size"] >= min_size) & 
                            (period_fvgs["fvg_size"] < upper_bound)
                        ]
                        size_range = f"{min_size:.2f}-{upper_bound:.2f}"
                    else:
                        # Single threshold case, use unbounded
                        size_filtered_fvgs = period_fvgs[period_fvgs["fvg_size"] >= min_size]
                        size_range = f"{min_size:.2f}+"
            else:
                raise ValueError(f"Unknown size_filtering_method: {size_filtering_method}. Use 'cumulative' or 'bins'.")
            
            # Apply max samples limit if specified
            if max_samples_per_cell and len(size_filtered_fvgs) > max_samples_per_cell:
                size_filtered_fvgs = size_filtered_fvgs.sample(n=max_samples_per_cell, random_state=42)
            
            # Calculate statistics
            total_fvgs = len(size_filtered_fvgs)
            total_processed_fvgs += total_fvgs  # Add to running counter
            if total_fvgs == 0:
                results.append({
                    "time_period": time_period,
                    "min_fvg_size": min_size,
                    "size_range": size_range,
                    "total_fvgs": 0,
                    "mitigated_fvgs": 0,
                    "mitigation_rate": 0.0,
                    "invalidation_rate": 0.0,
                    "avg_expansion_size": np.nan,
                    "p75_expansion_size": np.nan,
                    "expansion_efficiency": np.nan,
                    "p75_expansion_efficiency": np.nan,
                    "p75_mitigation_time": np.nan,
                    "p75_expansion_time": np.nan,
                    "p75_time_to_target": np.nan,
                    "p75_time_to_invalidation": np.nan,
                    "optimal_target": np.nan,
                    "optimal_ev": np.nan,
                    "avg_penetration_depth": np.nan,
                    "p75_penetration_depth": np.nan,
                    "avg_penetration_candle_count": np.nan,
                    "avg_penetration_depth_ratio": np.nan,
                    "avg_midpoint_crossing_count": np.nan,
                    "midpoint_crossed_pct": np.nan,
                    "avg_risk_points": np.nan,
                    "avg_rr": np.nan,
                    "rr_1_0_hit_rate": np.nan,
                    "rr_1_5_hit_rate": np.nan,
                    "rr_2_0_hit_rate": np.nan,
                })
                continue
            
            # Mitigated FVGs
            mitigated_fvgs = size_filtered_fvgs[size_filtered_fvgs["is_mitigated"]]
            mitigated_count = len(mitigated_fvgs)
            mitigation_rate = (mitigated_count / total_fvgs * 100) if total_fvgs > 0 else 0.0
            
            # Invalidation analysis (for mitigated FVGs)
            if mitigated_count > 0:
                valid_expansions = mitigated_fvgs["is_expansion_valid"].sum()
                invalidated_count = mitigated_count - valid_expansions
                invalidation_rate = (invalidated_count / mitigated_count * 100)
                
                # Expansion statistics
                expansion_data = mitigated_fvgs[mitigated_fvgs["expansion_size"].notna()]
                if len(expansion_data) > 0:
                    avg_expansion_size = expansion_data["expansion_size"].mean()
                    p75_expansion_size = expansion_data["expansion_size"].quantile(0.75)
                    # Calculate expansion efficiency (expansion/fvg_size ratio)
                    expansion_efficiency = (expansion_data["expansion_size"] / expansion_data["fvg_size"]).mean()
                    p75_expansion_efficiency = (expansion_data["expansion_size"] / expansion_data["fvg_size"]).quantile(0.75)
                    # Optimal take-profit target
                    opt = optimize_expansion_target(expansion_data["expansion_size"])
                    optimal_target = opt["optimal_target"]
                    optimal_ev = opt["optimal_ev"]
                else:
                    avg_expansion_size = np.nan
                    p75_expansion_size = np.nan
                    expansion_efficiency = np.nan
                    p75_expansion_efficiency = np.nan
                    optimal_target = np.nan
                    optimal_ev = np.nan

                # P75 mitigation time
                mitigation_times = mitigated_fvgs["time_to_mitigation"].dropna()
                if len(mitigation_times) > 0:
                    p75_mitigation_time = mitigation_times.quantile(0.75).total_seconds() / 60  # Convert to minutes
                else:
                    p75_mitigation_time = np.nan

                # P75 expansion time (seconds → minutes) — time to peak expansion
                if "expansion_time_seconds" in expansion_data.columns:
                    exp_times = expansion_data["expansion_time_seconds"].dropna()
                    p75_expansion_time = (exp_times.quantile(0.75) / 60) if len(exp_times) > 0 else np.nan
                else:
                    p75_expansion_time = np.nan

                # P75 time to target (seconds → minutes) — time to first reach expansion threshold
                if "time_to_target_seconds" in expansion_data.columns:
                    ttt = expansion_data["time_to_target_seconds"].dropna()
                    p75_time_to_target = (ttt.quantile(0.75) / 60) if len(ttt) > 0 else np.nan
                else:
                    p75_time_to_target = np.nan

                # P75 time to invalidation (seconds → minutes) — how long trade survives before invalidation
                if "time_to_invalidation_seconds" in expansion_data.columns:
                    tti = expansion_data["time_to_invalidation_seconds"].dropna()
                    p75_time_to_invalidation = (tti.quantile(0.75) / 60) if len(tti) > 0 else np.nan
                else:
                    p75_time_to_invalidation = np.nan

                # Penetration depth statistics
                if "max_penetration_depth" in expansion_data.columns:
                    pen_depths = expansion_data["max_penetration_depth"].dropna()
                    avg_penetration_depth = pen_depths.mean() if len(pen_depths) > 0 else np.nan
                    p75_penetration_depth = pen_depths.quantile(0.75) if len(pen_depths) > 0 else np.nan
                else:
                    avg_penetration_depth = np.nan
                    p75_penetration_depth = np.nan
                if "penetration_candle_count" in expansion_data.columns:
                    pen_counts = expansion_data["penetration_candle_count"].dropna()
                    avg_penetration_candle_count = pen_counts.mean() if len(pen_counts) > 0 else np.nan
                else:
                    avg_penetration_candle_count = np.nan
                # Penetration depth ratio (depth / max_possible_penetration)
                if "max_penetration_depth" in expansion_data.columns and "first_open" in expansion_data.columns:
                    pen_data = expansion_data[expansion_data["max_penetration_depth"].notna()].copy()
                    if len(pen_data) > 0:
                        # Compute max possible penetration per FVG
                        max_poss = np.where(
                            pen_data["fvg_type"].values == "bullish",
                            pen_data["zone_high"].values - pen_data["first_open"].values,
                            pen_data["first_open"].values - pen_data["zone_low"].values,
                        )
                        valid_mask = max_poss > 0
                        if valid_mask.any():
                            avg_penetration_depth_ratio = (pen_data["max_penetration_depth"].values[valid_mask] / max_poss[valid_mask]).mean()
                        else:
                            avg_penetration_depth_ratio = np.nan
                    else:
                        avg_penetration_depth_ratio = np.nan
                else:
                    avg_penetration_depth_ratio = np.nan
                if "midpoint_crossing_count" in expansion_data.columns:
                    mid_counts = expansion_data["midpoint_crossing_count"].dropna()
                    avg_midpoint_crossing_count = mid_counts.mean() if len(mid_counts) > 0 else np.nan
                    midpoint_crossed_pct = round((mid_counts > 0).sum() / total_fvgs * 100, 2) if total_fvgs > 0 and len(mid_counts) > 0 else np.nan
                else:
                    avg_midpoint_crossing_count = np.nan
                    midpoint_crossed_pct = np.nan
                # Risk-Reward statistics (touch-based stop)
                # Hit rates: target = N × fvg_size (not N × risk), since stop > fvg
                # avg_rr: actual RR = expansion_before_stop / risk (accounts for wider stop)
                if "risk_points" in expansion_data.columns and "achieved_rr" in expansion_data.columns and "fvg_size" in expansion_data.columns:
                    rr_data = expansion_data[expansion_data["risk_points"].notna() & (expansion_data["risk_points"] > 0) & (expansion_data["fvg_size"].notna())].copy()
                    if len(rr_data) > 0:
                        avg_risk_points = round(rr_data["risk_points"].mean(), 2)
                        # Actual RR = expansion / risk (lower than FVG multiple since risk > fvg_size)
                        avg_rr = round(rr_data["achieved_rr"].mean(), 4)
                        # Expansion before stop = achieved_rr * risk_points
                        exp_before_stop = rr_data["achieved_rr"] * rr_data["risk_points"]
                        fvg_sizes = rr_data["fvg_size"]
                        rr_1_0_hit_rate = round((exp_before_stop >= 1.0 * fvg_sizes).sum() / len(rr_data) * 100, 2)
                        rr_1_5_hit_rate = round((exp_before_stop >= 1.5 * fvg_sizes).sum() / len(rr_data) * 100, 2)
                        rr_2_0_hit_rate = round((exp_before_stop >= 2.0 * fvg_sizes).sum() / len(rr_data) * 100, 2)
                    else:
                        avg_risk_points = avg_rr = rr_1_0_hit_rate = rr_1_5_hit_rate = rr_2_0_hit_rate = np.nan
                else:
                    avg_risk_points = avg_rr = rr_1_0_hit_rate = rr_1_5_hit_rate = rr_2_0_hit_rate = np.nan
            else:
                invalidation_rate = 0.0
                avg_expansion_size = np.nan
                p75_expansion_size = np.nan
                expansion_efficiency = np.nan
                p75_expansion_efficiency = np.nan
                p75_mitigation_time = np.nan
                p75_expansion_time = np.nan
                p75_time_to_target = np.nan
                p75_time_to_invalidation = np.nan
                optimal_target = np.nan
                optimal_ev = np.nan
                avg_penetration_depth = np.nan
                p75_penetration_depth = np.nan
                avg_penetration_candle_count = np.nan
                avg_penetration_depth_ratio = np.nan
                avg_midpoint_crossing_count = np.nan
                midpoint_crossed_pct = np.nan
                avg_risk_points = np.nan
                avg_rr = np.nan
                rr_1_0_hit_rate = np.nan
                rr_1_5_hit_rate = np.nan
                rr_2_0_hit_rate = np.nan

            results.append({
                "time_period": time_period,
                "min_fvg_size": min_size,
                "size_range": size_range,
                "total_fvgs": total_fvgs,
                "mitigated_fvgs": mitigated_count,
                "mitigation_rate": round(mitigation_rate, 2),
                "invalidation_rate": round(invalidation_rate, 2),
                "avg_expansion_size": avg_expansion_size,
                "p75_expansion_size": p75_expansion_size,
                "expansion_efficiency": expansion_efficiency,
                "p75_expansion_efficiency": p75_expansion_efficiency,
                "p75_mitigation_time": p75_mitigation_time,
                "p75_expansion_time": p75_expansion_time,
                "p75_time_to_target": p75_time_to_target,
                "p75_time_to_invalidation": p75_time_to_invalidation,
                "optimal_target": optimal_target,
                "optimal_ev": optimal_ev,
                "avg_penetration_depth": avg_penetration_depth,
                "p75_penetration_depth": p75_penetration_depth,
                "avg_penetration_candle_count": avg_penetration_candle_count,
                "avg_penetration_depth_ratio": avg_penetration_depth_ratio,
                "avg_midpoint_crossing_count": avg_midpoint_crossing_count,
                "midpoint_crossed_pct": midpoint_crossed_pct,
                "avg_risk_points": avg_risk_points,
                "avg_rr": avg_rr,
                "rr_1_0_hit_rate": rr_1_0_hit_rate,
                "rr_1_5_hit_rate": rr_1_5_hit_rate,
                "rr_2_0_hit_rate": rr_2_0_hit_rate,
            })

    results_df = pd.DataFrame(results)
    
    # Add summary of binned FVG counts
    if not results_df.empty:
        total_binned_fvgs = results_df["total_fvgs"].sum()
        total_binned_mitigated = results_df["mitigated_fvgs"].sum()
        print(f"[INFO] === BINNED ANALYSIS SUMMARY ===")
        print(f"[INFO] Total FVGs processed in bins: {total_processed_fvgs:,}")
        print(f"[INFO] Total FVGs in final results: {total_binned_fvgs:,}")
        print(f"[INFO] Total mitigated in bins: {total_binned_mitigated:,}")
        print(f"[INFO] Binned coverage: {(total_binned_fvgs/original_count*100):.1f}% of original FVGs")
        print(f"[INFO] Processing efficiency: {(total_binned_fvgs/time_filtered_count*100):.1f}% of time-filtered FVGs")
    
    print(f"[INFO] Completed analysis of {len(results)} combinations")
    return results_df


def optimize_expansion_target(expansion_values, step=1.0):
    """
    Find the expansion target T that maximizes expected value = T × hit_rate(T).

    Args:
        expansion_values: array/Series of expansion_size for mitigated FVGs
        step: increment in points (default 1.0)

    Returns:
        dict: optimal_target, optimal_ev, optimal_hit_rate, curve_df
    """
    expansion_values = pd.Series(expansion_values).dropna()
    if len(expansion_values) == 0:
        return {
            "optimal_target": np.nan,
            "optimal_ev": np.nan,
            "optimal_hit_rate": np.nan,
            "curve_df": pd.DataFrame(columns=["target", "hit_rate", "expected_value"]),
        }

    max_target = np.percentile(expansion_values, 95)
    if max_target <= 0:
        return {
            "optimal_target": np.nan,
            "optimal_ev": np.nan,
            "optimal_hit_rate": np.nan,
            "curve_df": pd.DataFrame(columns=["target", "hit_rate", "expected_value"]),
        }

    targets = np.arange(step, max_target + step, step)
    n = len(expansion_values)
    rows = []
    for t in targets:
        hit_rate = (expansion_values >= t).sum() / n
        ev = t * hit_rate
        rows.append({"target": t, "hit_rate": hit_rate, "expected_value": ev})

    curve_df = pd.DataFrame(rows)
    best_idx = curve_df["expected_value"].idxmax()
    best = curve_df.loc[best_idx]

    return {
        "optimal_target": best["target"],
        "optimal_ev": best["expected_value"],
        "optimal_hit_rate": best["hit_rate"],
        "curve_df": curve_df,
    }


def debug_fvg_invalidation(df_1min, df_5min, fvg_data, fvg_filter_end_time):
    """Debug function to show detailed invalidation logic for a specific FVG."""
    print(f"\n{'='*60}")
    print(f"FVG INVALIDATION DEBUG")
    print(f"{'='*60}")

    fvg_type = fvg_data["fvg_type"]
    middle_open = fvg_data["middle_open"]
    middle_low = fvg_data["middle_low"]
    middle_high = fvg_data["middle_high"]
    zone_low = fvg_data["zone_low"]
    zone_high = fvg_data["zone_high"]

    print(f"FVG Type: {fvg_type}")
    print(f"Middle Open: {middle_open:.2f}")
    print(f"Middle Low: {middle_low:.2f}")
    print(f"Middle High: {middle_high:.2f}")
    print(f"Zone Low: {zone_low:.2f}")
    print(f"Zone High: {zone_high:.2f}")

    if fvg_type == "bearish":
        print(f"Invalidation Rule: Close > Middle High ({middle_high:.2f})")
    else:
        print(f"Invalidation Rule: Close < Middle Low ({middle_low:.2f})")

    print(f"\nTesting invalidation logic:")

    # Run the expansion logic with detailed invalidation tracking
    mitigation_time = fvg_data["mitigation_time"]
    start_idx = df_5min["date"].searchsorted(mitigation_time, side="right")

    if start_idx >= len(df_5min):
        print("No post-mitigation data.")
        return

    mitigation_date = mitigation_time.date()
    # Handle case where fvg_filter_end_time is already a time object
    if hasattr(fvg_filter_end_time, "time"):
        end_time_obj = fvg_filter_end_time.time()
    else:
        end_time_obj = fvg_filter_end_time
    end_time = datetime.combine(mitigation_date, end_time_obj)
    end_time = ny_tz.localize(end_time)
    end_idx = df_5min["date"].searchsorted(end_time, side="right")
    post_mitigation_df = df_5min.iloc[start_idx:end_idx]

    if post_mitigation_df.empty:
        print("Empty post-mitigation dataframe.")
        return

    invalidation_candle = None
    for idx, row in post_mitigation_df.iterrows():
        close = row["close"]
        current_time = row["date"]

        # Check for invalidation
        # Bearish FVG invalidated when close > middle candle high
        if fvg_type == "bearish" and close > middle_high:
            invalidation_candle = row
            print(f"INVALIDATION FOUND at {current_time}:")
            print(f"  Close: {close:.2f}")
            print(f"  Middle High: {middle_high:.2f}")
            print(f"  Difference: {close - middle_high:.2f}")
            print(
                f"  Condition: {close:.2f} > {middle_high:.2f} = {close > middle_high}"
            )
            break
        # Bullish FVG invalidated when close < middle candle low
        elif fvg_type == "bullish" and close < middle_low:
            invalidation_candle = row
            print(f"INVALIDATION FOUND at {current_time}:")
            print(f"  Close: {close:.2f}")
            print(f"  Middle Low: {middle_low:.2f}")
            print(f"  Difference: {middle_low - close:.2f}")
            print(
                f"  Condition: {close:.2f} < {middle_low:.2f} = {close < middle_low}"
            )
            break
        else:
            print(
                f"Candle {idx}: Time={current_time}, Close={close:.2f}, No invalidation"
            )

    if invalidation_candle is None:
        print("No invalidation found in the expansion period.")

    print(f"{'='*60}")
