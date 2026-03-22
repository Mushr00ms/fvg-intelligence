import os
import warnings
from datetime import datetime, time

# Use non-interactive backend to prevent GUI windows
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Handle optional imports for analysis functions
try:
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[WARNING] Seaborn not available - some plots may be limited")


try:
    # Try relative import first (when used as a module)
    from ..config import ny_tz
except ImportError:
    # Fall back to absolute import (when run from logic directory)
    from config import ny_tz


def sort_pivot_columns_by_numeric(pivot_df):
    """Sort pivot table columns by their numeric value instead of alphabetically."""
    if pivot_df.empty:
        return pivot_df
    
    def extract_sort_key(size_range_str):
        """Extract numeric sort key from size_range string."""
        try:
            if '-' in str(size_range_str):
                # Bins format: "0.25-1.25" -> sort by first value (0.25)
                return float(str(size_range_str).split('-')[0])
            else:
                # Cumulative format: "0.25" -> sort by value (0.25)
                return float(size_range_str)
        except (ValueError, TypeError):
            # If conversion fails, return original string for alphabetic sorting
            return str(size_range_str)
    
    try:
        sorted_cols = sorted(pivot_df.columns, key=extract_sort_key)
        return pivot_df[sorted_cols]
    except (ValueError, TypeError):
        # If all else fails, return original
        return pivot_df


def plot_fvg_expansion(
    df_5min,
    fvg_data,
    expansion_size,
    expansion_time,
    expansion_details,
    window=20,
    fvg_filter_end_time=None,
    extend_to_session_end=False,
    session_start_time=None,
):
    """Plot 5-min chart with FVG, mitigation, and expansion annotations. Optionally extend to session end."""
    mitigation_time = fvg_data["mitigation_time"]
    time_candle2 = fvg_data["time_candle2"]  # Start shading from 2nd candle

    # Use provided session start time or default to 09:30
    mitigation_date = mitigation_time.date()
    if session_start_time is None:
        session_start_time_obj = time(9, 30)  # Default to US session start
    elif isinstance(session_start_time, str):
        # Convert string time format "HH:MM" to time object
        hour, minute = map(int, session_start_time.split(":"))
        session_start_time_obj = time(hour, minute)
    else:
        session_start_time_obj = session_start_time
    
    session_start = ny_tz.localize(
        datetime.combine(mitigation_date, session_start_time_obj)
    )
    start_idx = df_5min["date"].searchsorted(session_start, side="left")
    start_idx = max(0, start_idx)  # Ensure non-negative

    # Calculate end_idx: Normally 1 candle after expansion/mitigation, but extend if requested
    mitigation_idx = df_5min["date"].searchsorted(mitigation_time, side="left")
    if extend_to_session_end and fvg_filter_end_time:
        session_end = ny_tz.localize(
            datetime.combine(mitigation_date, fvg_filter_end_time)
        )
        end_idx = df_5min["date"].searchsorted(session_end, side="right")
    elif expansion_time:
        exp_end_time = mitigation_time + expansion_time
        exp_end_idx = df_5min["date"].searchsorted(exp_end_time, side="left")
        end_idx = min(len(df_5min), exp_end_idx + 2)  # Expansion end + 1 candle after
    else:
        end_idx = min(
            len(df_5min), mitigation_idx + 2
        )  # Mitigation + 1 candle after (fallback)

    # Slice plot_df accordingly
    plot_df = df_5min.iloc[start_idx:end_idx].copy()

    # Limit plot_df to session end time if provided (safety)
    if fvg_filter_end_time:
        session_end = ny_tz.localize(
            datetime.combine(mitigation_date, fvg_filter_end_time)
        )
        plot_df = plot_df[plot_df["date"] <= session_end]
    
    # Check if plot_df is empty after filtering
    if plot_df.empty:
        print(f"[WARNING] No data to plot for FVG at {fvg_data['time_candle2']} - "
              f"session bounds: {session_start_time_obj} to {fvg_filter_end_time}")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    # NEW: Apply custom color template
    background_color = "#131722"  # Dark background
    up_color = "#fbc02d"  # Yellow for up candles (body, borders, wick)
    down_color = "#787b86"  # Gray for down candles (body, borders, wick)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    ax.grid(
        True, color="gray", linestyle="--", linewidth=0.5, alpha=0.5
    )  # Gray grid on dark bg
    ax.tick_params(axis="x", colors="white")  # White x-ticks for visibility
    ax.tick_params(axis="y", colors="white")  # White y-ticks
    plt.rcParams["text.color"] = "white"  # Default text color white

    # Candlestick plot with custom colors
    for _, row in plot_df.iterrows():
        if row["close"] > row["open"]:
            body_color = up_color
            wick_color = up_color
        else:
            body_color = down_color
            wick_color = down_color
        # Wick (high-low)
        ax.vlines(row["date"], row["low"], row["high"], color=wick_color, linewidth=1)
        # Body (open-close, thicker)
        ax.vlines(row["date"], row["open"], row["close"], color=body_color, linewidth=3)

    # Annotate FVG zone: Shade from time_candle2 to mitigation_time with yellow color
    zone_low, zone_high = fvg_data["zone_low"], fvg_data["zone_high"]
    fvg_color = "yellow"  # Use yellow for all FVG zones
    shading_start = time_candle2
    shading_end = mitigation_time
    ax.fill_between(
        [shading_start, shading_end],
        zone_low,
        zone_high,
        color=fvg_color,
        alpha=0.3,
        label="FVG Zone",
    )  # Increased opacity to 0.3 for better visibility

    # Mark mitigation with small arrow (instead of vline)
    mitigation_level = zone_high if fvg_data["fvg_type"] == "bullish" else zone_low
    ax.annotate(
        "",
        xy=(mitigation_time, mitigation_level),
        xytext=(
            mitigation_time,
            (
                mitigation_level + 5
                if fvg_data["fvg_type"] == "bullish"
                else mitigation_level - 5
            ),
        ),
        arrowprops=dict(
            facecolor="blue",
            edgecolor="blue",
            linewidth=1,
            width=1,
            headwidth=5,
            headlength=5,
            alpha=0.8,
        ),
    )
    ax.text(
        mitigation_time,
        (
            mitigation_level + 10
            if fvg_data["fvg_type"] == "bullish"
            else mitigation_level - 10
        ),
        "Mitigation",
        color="blue",
        fontsize=8,
    )

    # NEW: Horizontal line for mitigation level
    ax.axhline(
        mitigation_level,
        color="blue",
        linestyle="-",
        linewidth=1,
        alpha=0.7,
        label="Mitigation Level",
    )

    # NEW: Label mitigation level on y-axis
    ax.text(
        ax.get_xlim()[0],
        mitigation_level,
        f"{mitigation_level:.2f}",
        color="blue",
        va="center",
        ha="right",
        fontsize=8,
        bbox=dict(facecolor=background_color, edgecolor="none", alpha=0.7),
    )  # Match bg for blending

    # Mark expansion end with small arrow (if valid)
    if expansion_time:
        exp_end_time = mitigation_time + expansion_time
        exp_end_level = (
            mitigation_level + expansion_size
            if fvg_data["fvg_type"] == "bullish"
            else mitigation_level - expansion_size
        )
        ax.annotate(
            "",
            xy=(exp_end_time, exp_end_level),
            xytext=(
                exp_end_time,
                (
                    exp_end_level + 5
                    if fvg_data["fvg_type"] == "bullish"
                    else exp_end_level - 5
                ),
            ),
            arrowprops=dict(
                facecolor="purple",
                edgecolor="purple",
                linewidth=1,
                width=1,
                headwidth=5,
                headlength=5,
                alpha=0.8,
            ),
        )
        ax.text(
            exp_end_time,
            (
                exp_end_level + 10
                if fvg_data["fvg_type"] == "bullish"
                else exp_end_level - 10
            ),
            f"Expansion ({expansion_size})",
            color="purple",
            fontsize=8,
        )

        # NEW: Horizontal line for expansion end level
        ax.axhline(
            exp_end_level,
            color="purple",
            linestyle="-",
            linewidth=1,
            alpha=0.7,
            label="Expansion End Level",
        )

        # NEW: Label expansion end level on y-axis
        ax.text(
            ax.get_xlim()[0],
            exp_end_level,
            f"{exp_end_level:.2f}",
            color="purple",
            va="center",
            ha="right",
            fontsize=8,
            bbox=dict(facecolor=background_color, edgecolor="none", alpha=0.7),
        )  # Match bg

        # NEW: Visible line between mitigation (high/low of expansion range)
        exp_start_y = mitigation_level  # Start at mitigation level
        exp_end_y = exp_end_level  # End at expansion extrema
        ax.plot(
            [mitigation_time, exp_end_time],
            [exp_start_y, exp_end_y],
            color="yellow",
            linestyle="-",
            linewidth=2,
            label="Expansion Line",
        )  # Visible yellow line

    # NEW: Limit x-axis to session end time
    if fvg_filter_end_time and not plot_df.empty:
        left_limit = plot_df["date"].min()
        if pd.notna(left_limit):  # Check for valid datetime
            ax.set_xlim(left=left_limit, right=session_end)

    # Format x-axis for 5-minute intervals in NY time
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ny_tz))
    ax.xaxis.set_major_locator(
        mdates.MinuteLocator(interval=15)
    )  # Show every 15 minutes for 5-min charts
    ax.xaxis.set_minor_locator(
        mdates.MinuteLocator(interval=5)
    )  # Minor ticks every 5 minutes
    plt.setp(
        ax.xaxis.get_majorticklabels(), rotation=45, ha="right"
    )  # Rotate labels for better readability

    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor=background_color,
        edgecolor="white",
        labelcolor="white",
    )  # Legend matches dark bg
    plt.title(f"{fvg_data['fvg_type'].capitalize()} FVG Expansion Check", color="white")
    plt.xlabel("Time (NY)", color="white")
    plt.ylabel("Price", color="white")
    plt.tight_layout()  # Better spacing
    plt.show()


def create_time_based_heatmaps(results_df, filename_prefix: str, save_plots=True):
    """Create comprehensive heatmaps for time-based FVG analysis."""
    if not PLOTTING_AVAILABLE:
        print("[WARNING] Plotting not available - skipping visualization")
        return
    valid_results = results_df[results_df["total_fvgs"] > 0].copy()
    if valid_results.empty:
        print("[WARNING] No valid results to plot")
        return
    # Create pivot tables for heatmaps using size_range instead of min_fvg_size
    pivot_invalidation = valid_results.pivot_table(
        index="time_period",
        columns="size_range",
        values="invalidation_rate",
        aggfunc="mean",
    )
    pivot_expansion = valid_results.pivot_table(
        index="time_period",
        columns="size_range",
        values="avg_expansion_size",
        aggfunc="mean",
    )
    pivot_total_fvgs = valid_results.pivot_table(
        index="time_period", columns="size_range", values="total_fvgs", aggfunc="sum"
    )
    pivot_efficiency = valid_results.pivot_table(
        index="time_period",
        columns="size_range",
        values="p75_expansion_efficiency",
        aggfunc="mean",
    )
    
    # Sort columns by numerical value (not alphabetically)
    pivot_invalidation = sort_pivot_columns_by_numeric(pivot_invalidation)
    pivot_expansion = sort_pivot_columns_by_numeric(pivot_expansion)
    pivot_total_fvgs = sort_pivot_columns_by_numeric(pivot_total_fvgs)
    pivot_efficiency = sort_pivot_columns_by_numeric(pivot_efficiency)

    # --- Static PNG heatmaps ---
    # Infer interval from time_period column (e.g. "09:30-10:30" → 60 min)
    _interval_label = "Time"
    if "time_period" in valid_results.columns and not valid_results["time_period"].empty:
        _tp = str(valid_results["time_period"].iloc[0])
        if "-" in _tp:
            try:
                _start, _end = _tp.split("-")
                _sh, _sm = map(int, _start.strip().split(":"))
                _eh, _em = map(int, _end.strip().split(":"))
                _interval_label = f"{(_eh * 60 + _em) - (_sh * 60 + _sm)}-Minute"
            except (ValueError, IndexError):
                pass
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(
        f"FVG Time-Based Analysis: {_interval_label} Interval Performance",
        fontsize=16,
        fontweight="bold",
    )
    ax1 = axes[0, 0]
    sns.heatmap(
        pivot_invalidation,
        annot=True,
        cmap="RdYlBu_r",
        fmt=".1f",
        ax=ax1,
        cbar_kws={"label": "Invalidation Rate (%)"},
    )
    ax1.set_title("Invalidation Rate by Time Period and Minimum FVG Size Threshold")
    ax1.set_xlabel("Minimum FVG Size Threshold (points)")
    ax1.set_ylabel("Time Period")
    ax2 = axes[0, 1]
    # Use power normalization (gamma < 1) to spread color resolution across the upper range
    from matplotlib.colors import PowerNorm
    _exp_values = pivot_expansion.values[~np.isnan(pivot_expansion.values)]
    _exp_vmin = np.min(_exp_values) if len(_exp_values) > 0 else 0
    _exp_vmax = np.max(_exp_values) if len(_exp_values) > 0 else 1
    sns.heatmap(
        pivot_expansion,
        annot=True,
        cmap="YlOrRd",
        fmt=".1f",
        ax=ax2,
        norm=PowerNorm(gamma=2.5, vmin=_exp_vmin, vmax=_exp_vmax),
        cbar_kws={"label": "Avg Expansion Size (points)"},
    )
    ax2.set_title("Average Expansion Size by Time Period and Minimum FVG Size Threshold")
    ax2.set_xlabel("Minimum FVG Size Threshold (points)")
    ax2.set_ylabel("Time Period")
    ax3 = axes[1, 0]
    sns.heatmap(
        pivot_total_fvgs,
        annot=True,
        cmap="Blues",
        fmt=".0f",
        ax=ax3,
        cbar_kws={"label": "Total FVGs"},
    )
    ax3.set_title("Total FVGs by Time Period and Minimum FVG Size Threshold")
    ax3.set_xlabel("Minimum FVG Size Threshold (points)")
    ax3.set_ylabel("Time Period")
    ax4 = axes[1, 1]
    sns.heatmap(
        pivot_efficiency,
        annot=True,
        cmap="viridis",
        fmt=".2f",
        ax=ax4,
        cbar_kws={"label": "P75 Expansion Efficiency"},
    )
    ax4.set_title("P75 Expansion Efficiency by Time Period and Minimum FVG Size Threshold")
    ax4.set_xlabel("Minimum FVG Size Threshold (points)")
    ax4.set_ylabel("Time Period")
    plt.tight_layout()
    if save_plots:
        # Ensure charts directory exists
        os.makedirs("charts", exist_ok=True)
        filename = f"{filename_prefix}_time_heatmaps.png"
        plt.savefig(os.path.join("charts", filename), dpi=300, bbox_inches="tight")
        print(f"[INFO] Time-based heatmaps saved as {filename}")
    plt.close()


def create_time_performance_charts(results_df, filename_prefix: str, save_plots=True):
    """Create performance charts showing time-based trends."""

    if not PLOTTING_AVAILABLE:
        print("[WARNING] Plotting not available - skipping visualization")
        return

    # Filter valid results
    valid_results = results_df[results_df["total_fvgs"] > 0].copy()

    if valid_results.empty:
        print("[WARNING] No valid results to plot")
        return

    # Group by time period for analysis
    time_summary = (
        valid_results.groupby("time_period")
        .agg(
            {
                "total_fvgs": "sum",
                "invalidation_rate": "mean",
                "avg_expansion_size": "mean",
                "expansion_efficiency": "mean",
                "p75_expansion_efficiency": "mean",
                "mitigation_rate": "mean",
            }
        )
        .reset_index()
    )

    # Infer interval from time_period column
    _interval_label = "Time"
    if not time_summary["time_period"].empty:
        _tp = str(time_summary["time_period"].iloc[0])
        if "-" in _tp:
            try:
                _start, _end = _tp.split("-")
                _sh, _sm = map(int, _start.strip().split(":"))
                _eh, _em = map(int, _end.strip().split(":"))
                _interval_label = f"{(_eh * 60 + _em) - (_sh * 60 + _sm)}-Minute"
            except (ValueError, IndexError):
                pass

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"FVG Performance by Time Period ({_interval_label} Intervals)",
        fontsize=16,
        fontweight="bold",
    )

    # Chart 1: Total FVGs by Time Period
    ax1 = axes[0, 0]
    ax1.bar(
        time_summary["time_period"],
        time_summary["total_fvgs"],
        color="skyblue",
        alpha=0.7,
    )
    ax1.set_title("Total FVGs by Time Period")
    ax1.set_xlabel("Time Period")
    ax1.set_ylabel("Total FVGs")
    ax1.tick_params(axis="x", rotation=45)

    # Chart 2: Invalidation Rate by Time Period
    ax2 = axes[0, 1]
    ax2.plot(
        time_summary["time_period"],
        time_summary["invalidation_rate"],
        marker="o",
        color="red",
        linewidth=2,
        markersize=8,
    )
    ax2.set_title("Invalidation Rate by Time Period")
    ax2.set_xlabel("Time Period")
    ax2.set_ylabel("Invalidation Rate (%)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # Chart 3: Average Expansion Size by Time Period
    ax3 = axes[1, 0]
    ax3.plot(
        time_summary["time_period"],
        time_summary["avg_expansion_size"],
        marker="s",
        color="green",
        linewidth=2,
        markersize=8,
    )
    ax3.set_title("Average Expansion Size by Time Period")
    ax3.set_xlabel("Time Period")
    ax3.set_ylabel("Average Expansion Size (points)")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)

    # Chart 4: Expansion Efficiency Comparison by Time Period
    ax4 = axes[1, 1]
    ax4.plot(
        time_summary["time_period"],
        time_summary["expansion_efficiency"],
        marker="^",
        color="purple",
        linewidth=2,
        markersize=8,
        label="Avg Efficiency",
    )
    ax4.plot(
        time_summary["time_period"],
        time_summary["p75_expansion_efficiency"],
        marker="v",
        color="blue",
        linewidth=2,
        markersize=8,
        label="P75 Efficiency",
    )
    ax4.set_title("Expansion Efficiency Comparison by Time Period")
    ax4.set_xlabel("Time Period")
    ax4.set_ylabel("Expansion Efficiency")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    if save_plots:
        # Ensure charts directory exists
        os.makedirs("charts", exist_ok=True)
        filename = f"{filename_prefix}_time_performance.png"
        plt.savefig(os.path.join("charts", filename), dpi=300, bbox_inches="tight")
        print(f"[INFO] Time performance charts saved as {filename}")

    plt.close()


def create_analysis_plots(results_df, filename_prefix: str, save_plots=True):
    """Create comprehensive plots for FVG size and time analysis."""

    if not PLOTTING_AVAILABLE:
        print("[WARNING] Plotting not available - skipping visualization")
        return

    # Set up the plotting style
    plt.style.use("default")
    if PLOTTING_AVAILABLE:
        sns.set_palette("husl")

    # Create the main analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "FVG Size and Time Analysis: Overall Performance Metrics",
        fontsize=16,
        fontweight="bold",
    )

    # Filter out rows with zero FVGs for better visualization
    valid_results = results_df[results_df["total_fvgs"] > 0].copy()

    if valid_results.empty:
        print("[WARNING] No valid results to plot")
        return

    # Plot 1: Total FVGs vs Min Size (aggregated across all time periods)
    ax1 = axes[0, 0]
    size_summary = (
        valid_results.groupby("size_range")["total_fvgs"].sum().reset_index()
    )
    ax1.plot(
        size_summary["size_range"],
        size_summary["total_fvgs"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="skyblue",
    )
    ax1.set_xlabel("Minimum FVG Size Threshold (points)")
    ax1.set_ylabel("Total FVGs")
    ax1.set_title("Total FVGs by Size Range")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Average Invalidation Rate vs Min Size
    ax2 = axes[0, 1]
    invalidation_summary = (
        valid_results.groupby("size_range")["invalidation_rate"].mean().reset_index()
    )
    plot_data = invalidation_summary[invalidation_summary["invalidation_rate"] > 0]
    if not plot_data.empty:
        ax2.plot(
            plot_data["size_range"],
            plot_data["invalidation_rate"],
            marker="o",
            linewidth=2,
            markersize=6,
            color="lightcoral",
        )
        ax2.axhline(
            y=plot_data["invalidation_rate"].mean(),
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f'Average: {plot_data["invalidation_rate"].mean():.1f}%',
        )
    ax2.set_xlabel("Minimum FVG Size Threshold (points)")
    ax2.set_ylabel("Invalidation Rate (%)")
    ax2.set_title("Average Invalidation Rate by Size Range")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Average Expansion Size vs Min Size
    ax3 = axes[0, 2]
    expansion_summary = (
        valid_results.groupby("size_range")["avg_expansion_size"].mean().reset_index()
    )
    expansion_data = expansion_summary[expansion_summary["avg_expansion_size"] > 0]
    if not expansion_data.empty:
        ax3.plot(
            expansion_data["size_range"],
            expansion_data["avg_expansion_size"],
            marker="s",
            linewidth=2,
            markersize=6,
            color="lightgreen",
        )
        ax3.axhline(
            y=expansion_data["avg_expansion_size"].mean(),
            color="g",
            linestyle="--",
            alpha=0.7,
            label=f'Average: {expansion_data["avg_expansion_size"].mean():.1f}',
        )
    ax3.set_xlabel("Minimum FVG Size Threshold (points)")
    ax3.set_ylabel("Average Expansion Size (points)")
    ax3.set_title("Average Expansion Size by Size Range")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Expansion Efficiency vs Min Size
    ax4 = axes[1, 0]
    efficiency_summary = (
        valid_results.groupby("size_range")["expansion_efficiency"]
        .mean()
        .reset_index()
    )
    efficiency_data = efficiency_summary[efficiency_summary["expansion_efficiency"] > 0]
    if not efficiency_data.empty:
        ax4.plot(
            efficiency_data["size_range"],
            efficiency_data["expansion_efficiency"],
            marker="^",
            linewidth=2,
            markersize=6,
            color="mediumpurple",
        )
        ax4.axhline(
            y=efficiency_data["expansion_efficiency"].mean(),
            color="purple",
            linestyle="--",
            alpha=0.7,
            label=f'Average: {efficiency_data["expansion_efficiency"].mean():.2f}',
        )
    ax4.set_xlabel("Minimum FVG Size Threshold (points)")
    ax4.set_ylabel("Expansion Efficiency (Expansion/FVG Size)")
    ax4.set_title("Expansion Efficiency by Size Range")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Mitigation Rate vs Min Size
    ax5 = axes[1, 1]
    mitigation_summary = (
        valid_results.groupby("size_range")["mitigation_rate"].mean().reset_index()
    )
    mitigation_data = mitigation_summary[mitigation_summary["mitigation_rate"] > 0]
    if not mitigation_data.empty:
        ax5.plot(
            mitigation_data["size_range"],
            mitigation_data["mitigation_rate"],
            marker="D",
            linewidth=2,
            markersize=6,
            color="orange",
        )
        ax5.axhline(
            y=mitigation_data["mitigation_rate"].mean(),
            color="orange",
            linestyle="--",
            alpha=0.7,
            label=f'Average: {mitigation_data["mitigation_rate"].mean():.1f}%',
        )
    ax5.set_xlabel("Minimum FVG Size Threshold (points)")
    ax5.set_ylabel("Mitigation Rate (%)")
    ax5.set_title("Mitigation Rate by Size Range")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Plot 6: Time Period Performance Summary
    ax6 = axes[1, 2]
    time_performance = (
        valid_results.groupby("time_period")
        .agg({"invalidation_rate": "mean", "avg_expansion_size": "mean"})
        .reset_index()
    )

    # Create dual-axis plot
    ax6_twin = ax6.twinx()

    # Plot invalidation rate
    line1 = ax6.plot(
        time_performance["time_period"],
        time_performance["invalidation_rate"],
        "r-",
        marker="o",
        linewidth=2,
        markersize=6,
        label="Invalidation Rate",
    )
    ax6.set_ylabel("Invalidation Rate (%)", color="r")
    ax6.tick_params(axis="y", labelcolor="r")

    # Plot expansion size on secondary axis
    line2 = ax6_twin.plot(
        time_performance["time_period"],
        time_performance["avg_expansion_size"],
        "g-",
        marker="s",
        linewidth=2,
        markersize=6,
        label="Expansion Size",
    )
    ax6_twin.set_ylabel("Average Expansion Size (points)", color="g")
    ax6_twin.tick_params(axis="y", labelcolor="g")

    ax6.set_xlabel("Time Period")
    ax6.set_title("Performance by Time Period")
    ax6.tick_params(axis="x", rotation=45)
    ax6.grid(True, alpha=0.3)

    # Add legend
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()

    if save_plots:
        # Ensure charts directory exists
        os.makedirs("charts", exist_ok=True)
        filename = f"{filename_prefix}_size_time_analysis.png"
        plt.savefig(os.path.join("charts", filename), dpi=300, bbox_inches="tight")
        print(f"[INFO] Analysis plots saved as {filename}")

    plt.close()


def create_mitigation_time_heatmap(results_df, filename_prefix: str, save_plots=True):
    """Create heatmaps for P75 mitigation time, mitigation rate, and sample size by time period and minimum FVG size."""

    if not PLOTTING_AVAILABLE:
        print("[WARNING] Plotting not available - skipping visualization")
        return

    valid_results = results_df[results_df["mitigated_fvgs"] > 0].copy()
    all_results = results_df[results_df["total_fvgs"] > 0].copy()

    if valid_results.empty or all_results.empty:
        print("[WARNING] No valid results to plot")
        return

    pivot_mitigation_time = valid_results.pivot_table(
        index="time_period",
        columns="size_range",
        values="p75_mitigation_time",
        aggfunc="mean",
    )

    pivot_mitigation_rate = valid_results.pivot_table(
        index="time_period",
        columns="size_range",
        values="mitigation_rate",
        aggfunc="mean",
    )

    # Create sample size pivot table using all results
    pivot_sample_size = all_results.pivot_table(
        index="time_period", columns="size_range", values="total_fvgs", aggfunc="sum"
    )
    
    # Sort all pivot columns by numeric value
    pivot_mitigation_time = sort_pivot_columns_by_numeric(pivot_mitigation_time)
    pivot_mitigation_rate = sort_pivot_columns_by_numeric(pivot_mitigation_rate)
    pivot_sample_size = sort_pivot_columns_by_numeric(pivot_sample_size)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

    # Heatmap 1: P75 Mitigation Time
    sns.heatmap(
        pivot_mitigation_time,
        annot=True,
        cmap="coolwarm",
        fmt=".1f",
        ax=ax1,
        cbar_kws={"label": "P75 Mitigation Time (minutes)"},
    )
    ax1.set_title("P75 Mitigation Time by Time Period and Minimum FVG Size Threshold")
    ax1.set_xlabel("Minimum FVG Size Threshold (points)")
    ax1.set_ylabel("Time Period")

    # Heatmap 2: Mitigation Rate
    sns.heatmap(
        pivot_mitigation_rate,
        annot=True,
        cmap="YlGnBu",
        fmt=".1f",
        ax=ax2,
        cbar_kws={"label": "Mitigation Rate (%)"},
    )
    ax2.set_title("Mitigation Rate by Time Period and Minimum FVG Size Threshold")
    ax2.set_xlabel("Minimum FVG Size Threshold (points)")
    ax2.set_ylabel("Time Period")

    # Heatmap 3: Sample Size (Total FVGs)
    sns.heatmap(
        pivot_sample_size,
        annot=True,
        cmap="viridis",
        fmt=".0f",
        ax=ax3,
        cbar_kws={"label": "Total FVGs (Sample Size)"},
    )
    ax3.set_title("Sample Size (Total FVGs) by Time Period and Minimum FVG Size Threshold")
    ax3.set_xlabel("Minimum FVG Size Threshold (points)")
    ax3.set_ylabel("Time Period")

    plt.tight_layout()

    if save_plots:
        # Ensure charts directory exists
        os.makedirs("charts", exist_ok=True)
        filename = f"{filename_prefix}_mitigation_time_heatmap.png"
        plt.savefig(os.path.join("charts", filename), dpi=300, bbox_inches="tight")
        print(f"[INFO] Mitigation time heatmap with sample size saved as {filename}")

    plt.close()


def create_body_multiplier_charts(results_df, filename_prefix: str, save_plots=True):
    """Create charts for body multiplier analysis results."""

    if not PLOTTING_AVAILABLE:
        print("[WARNING] Plotting not available - skipping visualization")
        return

    # Filter valid results
    valid_results = results_df[results_df["total_fvgs"] > 0].copy()

    if valid_results.empty:
        print("[WARNING] No valid results to plot")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("FVG Body Multiplier Analysis Results", fontsize=16, fontweight="bold")

    # Plot 1: Total FVGs vs Body Multiplier
    ax1 = axes[0, 0]
    ax1.plot(
        valid_results["body_multiplier"],
        valid_results["total_fvgs"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="skyblue",
    )
    ax1.set_xlabel("Body Multiplier")
    ax1.set_ylabel("Total FVGs")
    ax1.set_title("Total FVGs vs Body Multiplier")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mitigation Rate vs Body Multiplier
    ax2 = axes[0, 1]
    ax2.plot(
        valid_results["body_multiplier"],
        valid_results["mitigation_percentage"],
        marker="s",
        linewidth=2,
        markersize=6,
        color="lightgreen",
    )
    ax2.set_xlabel("Body Multiplier")
    ax2.set_ylabel("Mitigation Rate (%)")
    ax2.set_title("Mitigation Rate vs Body Multiplier")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average Expansion Size vs Body Multiplier
    ax3 = axes[0, 2]
    ax3.plot(
        valid_results["body_multiplier"],
        valid_results["avg_expansion"],
        marker="^",
        linewidth=2,
        markersize=6,
        color="orange",
    )
    ax3.set_xlabel("Body Multiplier")
    ax3.set_ylabel("Average Expansion Size (points)")
    ax3.set_title("Average Expansion Size vs Body Multiplier")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Invalidation Rate vs Body Multiplier
    ax4 = axes[1, 0]
    ax4.plot(
        valid_results["body_multiplier"],
        valid_results["invalidated_percentage"],
        marker="D",
        linewidth=2,
        markersize=6,
        color="red",
    )
    ax4.set_xlabel("Body Multiplier")
    ax4.set_ylabel("Invalidation Rate (%)")
    ax4.set_title("Invalidation Rate vs Body Multiplier")
    ax4.grid(True, alpha=0.3)

    # Plot 5: Valid Expansions vs Body Multiplier
    ax5 = axes[1, 1]
    ax5.plot(
        valid_results["body_multiplier"],
        valid_results["valid_expansions"],
        marker="v",
        linewidth=2,
        markersize=6,
        color="purple",
    )
    ax5.set_xlabel("Body Multiplier")
    ax5.set_ylabel("Valid Expansions Count")
    ax5.set_title("Valid Expansions vs Body Multiplier")
    ax5.grid(True, alpha=0.3)

    # Plot 6: FVG Size Distribution
    ax6 = axes[1, 2]
    fvg_size_data = valid_results[
        ["fvg_size_25pct", "fvg_size_50pct", "fvg_size_75pct"]
    ].dropna()
    if not fvg_size_data.empty:
        ax6.plot(
            valid_results["body_multiplier"],
            valid_results["fvg_size_25pct"],
            marker="o",
            linewidth=1,
            markersize=4,
            label="25th percentile",
            alpha=0.7,
        )
        ax6.plot(
            valid_results["body_multiplier"],
            valid_results["fvg_size_50pct"],
            marker="s",
            linewidth=2,
            markersize=6,
            label="50th percentile (median)",
        )
        ax6.plot(
            valid_results["body_multiplier"],
            valid_results["fvg_size_75pct"],
            marker="^",
            linewidth=1,
            markersize=4,
            label="75th percentile",
            alpha=0.7,
        )
    ax6.set_xlabel("Body Multiplier")
    ax6.set_ylabel("FVG Size (points)")
    ax6.set_title("FVG Size Distribution vs Body Multiplier")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()

    if save_plots:
        # Ensure charts directory exists
        os.makedirs("charts", exist_ok=True)
        filename = f"{filename_prefix}_body_multiplier_analysis.png"
        plt.savefig(os.path.join("charts", filename), dpi=300, bbox_inches="tight")
        print(f"[INFO] Body multiplier analysis charts saved as {filename}")

    plt.close()


def create_expansion_percentile_charts(
    results_df, filename_prefix: str, save_plots=True
):
    """Create charts showing expansion size and time percentiles."""

    if not PLOTTING_AVAILABLE:
        print("[WARNING] Plotting not available - skipping visualization")
        return

    # Filter valid results
    valid_results = results_df[results_df["total_fvgs"] > 0].copy()

    if valid_results.empty:
        print("[WARNING] No valid results to plot")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "FVG Expansion Analysis: Size and Time Percentiles",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Expansion Size Percentiles
    ax1 = axes[0, 0]
    expansion_size_data = valid_results[
        ["expansion_size_25pct", "expansion_size_50pct", "expansion_size_75pct"]
    ].dropna()
    if not expansion_size_data.empty:
        ax1.plot(
            valid_results["body_multiplier"],
            valid_results["expansion_size_25pct"],
            marker="o",
            linewidth=1,
            markersize=4,
            label="25th percentile",
            alpha=0.7,
        )
        ax1.plot(
            valid_results["body_multiplier"],
            valid_results["expansion_size_50pct"],
            marker="s",
            linewidth=2,
            markersize=6,
            label="50th percentile (median)",
        )
        ax1.plot(
            valid_results["body_multiplier"],
            valid_results["expansion_size_75pct"],
            marker="^",
            linewidth=1,
            markersize=4,
            label="75th percentile",
            alpha=0.7,
        )
    ax1.set_xlabel("Body Multiplier")
    ax1.set_ylabel("Expansion Size (points)")
    ax1.set_title("Expansion Size Percentiles vs Body Multiplier")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Mitigation Duration Percentiles
    ax2 = axes[0, 1]
    # Convert timedelta to minutes for plotting
    duration_data = valid_results[
        ["duration_25pct", "duration_50pct", "duration_75pct"]
    ].dropna()
    if not duration_data.empty:
        dur_25_min = [
            d.total_seconds() / 60 if pd.notna(d) else np.nan
            for d in valid_results["duration_25pct"]
        ]
        dur_50_min = [
            d.total_seconds() / 60 if pd.notna(d) else np.nan
            for d in valid_results["duration_50pct"]
        ]
        dur_75_min = [
            d.total_seconds() / 60 if pd.notna(d) else np.nan
            for d in valid_results["duration_75pct"]
        ]

        ax2.plot(
            valid_results["body_multiplier"],
            dur_25_min,
            marker="o",
            linewidth=1,
            markersize=4,
            label="25th percentile",
            alpha=0.7,
        )
        ax2.plot(
            valid_results["body_multiplier"],
            dur_50_min,
            marker="s",
            linewidth=2,
            markersize=6,
            label="50th percentile (median)",
        )
        ax2.plot(
            valid_results["body_multiplier"],
            dur_75_min,
            marker="^",
            linewidth=1,
            markersize=4,
            label="75th percentile",
            alpha=0.7,
        )
    ax2.set_xlabel("Body Multiplier")
    ax2.set_ylabel("Mitigation Duration (minutes)")
    ax2.set_title("Mitigation Duration Percentiles vs Body Multiplier")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Expansion Efficiency Metrics
    ax3 = axes[1, 0]
    efficiency_data = valid_results[["expansion_ratio_75pct"]].dropna()
    if not efficiency_data.empty:
        ax3.plot(
            valid_results["body_multiplier"],
            valid_results["expansion_ratio_75pct"],
            marker="D",
            linewidth=2,
            markersize=6,
            color="purple",
        )
    ax3.set_xlabel("Body Multiplier")
    ax3.set_ylabel("Expansion Ratio (75th percentile)")
    ax3.set_title("Expansion Efficiency vs Body Multiplier")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary Performance Score
    ax4 = axes[1, 1]
    # Calculate a simple performance score: high expansion, low invalidation
    performance_score = (
        valid_results["avg_expansion"] / valid_results["avg_expansion"].max()
    ) * (1 - valid_results["invalidated_percentage"] / 100)

    ax4.plot(
        valid_results["body_multiplier"],
        performance_score,
        marker="*",
        linewidth=2,
        markersize=8,
        color="gold",
    )
    ax4.set_xlabel("Body Multiplier")
    ax4.set_ylabel("Performance Score (0-1)")
    ax4.set_title("Overall Performance Score vs Body Multiplier")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        # Ensure charts directory exists
        os.makedirs("charts", exist_ok=True)
        filename = f"{filename_prefix}_expansion_percentiles.png"
        plt.savefig(os.path.join("charts", filename), dpi=300, bbox_inches="tight")
        print(f"[INFO] Expansion percentile charts saved as {filename}")

    plt.close()


def create_time_period_heatmap(
    time_period_results_df, filename_prefix: str, save_plots=True
):
    """Create heatmap showing FVG performance by time period and body multiplier."""
    if not PLOTTING_AVAILABLE:
        print("[WARNING] Plotting not available - skipping visualization")
        return
    if time_period_results_df.empty:
        print("[WARNING] No time period results to plot")
        return
    # Create pivot tables for different metrics
    pivot_total_fvgs = time_period_results_df.pivot_table(
        index="time_period",
        columns="size_range",
        values="total_fvgs",
        aggfunc="sum",
    )
    pivot_invalidation = time_period_results_df.pivot_table(
        index="time_period",
        columns="size_range",
        values="invalidation_rate",
        aggfunc="mean",
    )
    pivot_expansion = time_period_results_df.pivot_table(
        index="time_period",
        columns="size_range",
        values="avg_expansion_size",
        aggfunc="mean",
    )
    pivot_mitigation = time_period_results_df.pivot_table(
        index="time_period",
        columns="size_range",
        values="mitigation_rate",
        aggfunc="mean",
    )
    
    # Sort all pivot columns by numeric value
    pivot_total_fvgs = sort_pivot_columns_by_numeric(pivot_total_fvgs)
    pivot_invalidation = sort_pivot_columns_by_numeric(pivot_invalidation)
    pivot_expansion = sort_pivot_columns_by_numeric(pivot_expansion)
    pivot_mitigation = sort_pivot_columns_by_numeric(pivot_mitigation)

    # --- Static PNG heatmaps ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(
        "FVG Performance by Time Period and Body Multiplier",
        fontsize=16,
        fontweight="bold",
    )
    ax1 = axes[0, 0]
    sns.heatmap(
        pivot_total_fvgs,
        annot=True,
        cmap="Blues",
        fmt=".0f",
        ax=ax1,
        cbar_kws={"label": "Total FVGs"},
    )
    ax1.set_title("Total FVGs by Time Period and Body Multiplier")
    ax1.set_xlabel("Body Multiplier")
    ax1.set_ylabel("Time Period")
    ax2 = axes[0, 1]
    sns.heatmap(
        pivot_invalidation,
        annot=True,
        cmap="Reds",
        fmt=".1f",
        ax=ax2,
        cbar_kws={"label": "Invalidation Rate (%)"},
    )
    ax2.set_title("Invalidation Rate by Time Period and Body Multiplier")
    ax2.set_xlabel("Body Multiplier")
    ax2.set_ylabel("Time Period")
    ax3 = axes[1, 0]
    sns.heatmap(
        pivot_expansion,
        annot=True,
        cmap="Greens",
        fmt=".1f",
        ax=ax3,
        cbar_kws={"label": "Avg Expansion Size (points)"},
    )
    ax3.set_title("Average Expansion Size by Time Period and Body Multiplier")
    ax3.set_xlabel("Body Multiplier")
    ax3.set_ylabel("Time Period")
    ax4 = axes[1, 1]
    sns.heatmap(
        pivot_mitigation,
        annot=True,
        cmap="Oranges",
        fmt=".1f",
        ax=ax4,
        cbar_kws={"label": "Mitigation Rate (%)"},
    )
    ax4.set_title("Mitigation Rate by Time Period and Body Multiplier")
    ax4.set_xlabel("Body Multiplier")
    ax4.set_ylabel("Time Period")
    plt.tight_layout()
    if save_plots:
        # Ensure charts directory exists
        os.makedirs("charts", exist_ok=True)
        filename = f"{filename_prefix}_time_period_heatmap.png"
        plt.savefig(os.path.join("charts", filename), dpi=300, bbox_inches="tight")
        print(f"[INFO] Time period heatmap saved as {filename}")
    plt.close()


def create_time_period_summary_charts(
    time_period_results_df, filename_prefix: str, save_plots=True
):
    """Create summary charts for time period analysis."""

    if not PLOTTING_AVAILABLE:
        print("[WARNING] Plotting not available - skipping visualization")
        return

    if time_period_results_df.empty:
        print("[WARNING] No time period results to plot")
        return

    # Aggregate by time period (across all body multipliers)
    time_summary = (
        time_period_results_df.groupby("time_period")
        .agg(
            {
                "total_fvgs": "sum",
                "invalidation_rate": "mean",
                "avg_expansion_size": "mean",
                "mitigation_rate": "mean",
            }
        )
        .reset_index()
    )

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "FVG Performance Summary by Time Period", fontsize=16, fontweight="bold"
    )

    # Chart 1: Total FVGs by Time Period
    ax1 = axes[0, 0]
    ax1.bar(
        time_summary["time_period"],
        time_summary["total_fvgs"],
        color="skyblue",
        alpha=0.7,
    )
    ax1.set_title("Total FVGs by Time Period")
    ax1.set_xlabel("Time Period")
    ax1.set_ylabel("Total FVGs")
    ax1.tick_params(axis="x", rotation=45)

    # Chart 2: Invalidation Rate by Time Period
    ax2 = axes[0, 1]
    ax2.plot(
        time_summary["time_period"],
        time_summary["invalidation_rate"],
        marker="o",
        color="red",
        linewidth=2,
        markersize=8,
    )
    ax2.set_title("Average Invalidation Rate by Time Period")
    ax2.set_xlabel("Time Period")
    ax2.set_ylabel("Invalidation Rate (%)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # Chart 3: Average Expansion Size by Time Period
    ax3 = axes[1, 0]
    ax3.plot(
        time_summary["time_period"],
        time_summary["avg_expansion_size"],
        marker="s",
        color="green",
        linewidth=2,
        markersize=8,
    )
    ax3.set_title("Average Expansion Size by Time Period")
    ax3.set_xlabel("Time Period")
    ax3.set_ylabel("Average Expansion Size (points)")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)

    # Chart 4: Mitigation Rate by Time Period
    ax4 = axes[1, 1]
    ax4.plot(
        time_summary["time_period"],
        time_summary["mitigation_rate"],
        marker="^",
        color="orange",
        linewidth=2,
        markersize=8,
    )
    ax4.set_title("Average Mitigation Rate by Time Period")
    ax4.set_xlabel("Time Period")
    ax4.set_ylabel("Mitigation Rate (%)")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        # Ensure charts directory exists
        os.makedirs("charts", exist_ok=True)
        filename = f"{filename_prefix}_time_period_summary.png"
        plt.savefig(os.path.join("charts", filename), dpi=300, bbox_inches="tight")
        print(f"[INFO] Time period summary charts saved as {filename}")

    plt.close()
