#!/usr/bin/env python3
"""
FVG Analysis Script - Multi-Timeframe
Detects Fair Value Gaps and analyzes mitigation/expansion across timeframe pairs.
"""

import os
import hashlib
from datetime import datetime, time

# Use non-interactive backend to prevent GUI windows
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot

import numpy as np
import pandas as pd

# Configuration imports
from config import (
    PRIMARY_MARKET,
    min_fvg_size,
    min_fvg_size_3min,
    min_fvg_size_15min,
    min_expansion_size,
    min_expansion_size_3min,
    min_expansion_size_15min,
    get_min_fvg_sizes_range,
    period,
    custom_start,
    custom_end,
    min_search_bars_1min,
    min_search_bars_3min,
    min_search_bars_5min,
    min_search_bars_15min,
    min_search_bars_1min_mitigation_3min,
    min_search_bars_5min_mitigation,
    roll_days,
    mitigation_same_day,
    session_start_time,
    session_end_time,
    fvg_filter_start_time,
    fvg_filter_end_time,
    size_filtering_method,
    session_period_minutes,
    session_period_minutes_3min,
    session_period_minutes_15min,
    size_filtering_method_3min,
    size_filtering_method_15min,
    fvg_filter_end_time_3min,
    fvg_filter_end_time_15min,
)

# Utility imports
from utils import (
    # Data and contracts
    fetch_market_data,
    generate_nq_expirations,
    generate_es_expirations,
    # FVG detection and analysis
    detect_fvg,
    find_fvg_mitigations,
    calculate_expansion_after_mitigation,
    analyze_fvg_size_time_distribution,
    optimize_expansion_target,
    # Visualization and insights
    create_analysis_plots,
    create_time_based_heatmaps,
    create_mitigation_time_heatmap,
    create_interactive_heatmaps,
    create_interactive_mitigation_heatmap,
    create_optimizer_chart,
    save_fvg_size_time_insights,
)


def format_period_str(p):
    """Convert period like '5 years' to short format like '5y'."""
    replacements = [
        ("years", "y"), ("year", "y"), ("months", "m"), ("month", "m"),
        ("weeks", "w"), ("week", "w"), ("days", "d"), ("day", "d"),
        ("quarters", "q"), ("quarter", "q"),
    ]
    result = p.replace(" ", "")
    for long, short in replacements:
        result = result.replace(long, short)
    return result


def run_fvg_pipeline(
    label,
    df_detect,
    df_mitigate,
    df_expansion,
    fvg_size_threshold,
    exp_size_threshold,
    get_sizes_range_fn,
    search_bars_primary,
    search_bars_fallback,
    primary_label,
    fallback_label,
    fvg_cache_dir,
    interval_minutes=None,
    filtering_method=None,
    filter_end_time=None,
):
    """
    Single FVG pipeline: detect -> mitigate -> expand -> analyze -> visualize.

    Args:
        label: Pipeline identifier (e.g. "5min", "15min")
        df_detect: DataFrame for FVG detection (higher timeframe)
        df_mitigate: DataFrame for mitigation search (lower timeframe)
        df_expansion: DataFrame for expansion tracking
        fvg_size_threshold: Minimum FVG size for detection
        exp_size_threshold: Minimum expansion size for validity
        get_sizes_range_fn: Callable returning size threshold array for analysis
        search_bars_primary: Bars to search in mitigation data
        search_bars_fallback: Bars to search in detection data (fallback)
        primary_label: Label for primary mitigation method (e.g. "1min")
        fallback_label: Label for fallback mitigation method (e.g. "5min")
        fvg_cache_dir: Directory for cache files
        interval_minutes: Session period for time analysis (default: session_period_minutes)
        filtering_method: Size filtering method - "bins" or "cumulative" (default: size_filtering_method)
        filter_end_time: Session end time for FVG filtering/expansion (default: fvg_filter_end_time)
    """
    if interval_minutes is None:
        interval_minutes = session_period_minutes
    if filtering_method is None:
        filtering_method = size_filtering_method
    if filter_end_time is None:
        filter_end_time = fvg_filter_end_time
    print(f"\n[INFO] === {label.upper()} FVG DETECTION PIPELINE ===")
    print(f"[INFO] Parameters: min_fvg_size={fvg_size_threshold}, min_expansion_size={exp_size_threshold}")

    # Generate unique cache key
    param_str = (
        f"{PRIMARY_MARKET}_{label}_{period}_{custom_start}_{custom_end}_"
        f"{search_bars_primary}_{search_bars_fallback}_"
        f"{fvg_filter_start_time}_{filter_end_time}_"
        f"{session_start_time}_{session_end_time}_{roll_days}_"
        f"{mitigation_same_day}_{fvg_size_threshold}_{exp_size_threshold}"
    )
    cache_hash = hashlib.md5(param_str.encode()).hexdigest()
    cache_file = os.path.join(fvg_cache_dir, f"fvg_results_{label}_{cache_hash}.parquet")

    # Load from cache or compute
    df_fvgs = _load_or_compute_fvgs(
        cache_file, label, df_detect, df_mitigate,
        fvg_size_threshold, search_bars_primary, search_bars_fallback,
        primary_label, fallback_label, filter_end_time,
    )

    if df_fvgs.empty:
        print(f"[WARNING] No {label} FVGs found, skipping pipeline...")
        return

    # Compute expansion data if needed
    df_fvgs = _compute_expansions(df_fvgs, df_expansion, cache_file, label, exp_size_threshold, filter_end_time)

    # Compute RR trade simulations if needed (use 1min candles for precise stop walk)
    df_fvgs = _compute_rr_trades(df_fvgs, df_mitigate, cache_file, label, filter_end_time)

    # Print summary statistics
    _print_summary(df_fvgs, label)

    # Run size-time analysis and visualization
    _run_analysis_and_viz(df_fvgs, label, get_sizes_range_fn, exp_size_threshold, interval_minutes, filtering_method, filter_end_time)

    # Run RR analysis and save to store
    _run_rr_analysis(df_fvgs, label, interval_minutes, filter_end_time)


def _load_or_compute_fvgs(
    cache_file, label, df_detect, df_mitigate,
    fvg_size_threshold, search_bars_primary, search_bars_fallback,
    primary_label, fallback_label, filter_end_time,
):
    """Load FVG results from cache or compute fresh."""
    df_fvgs = None

    if os.path.exists(cache_file):
        print(f"[INFO] Loading cached {label} FVG results from {cache_file}")
        try:
            df_fvgs = pd.read_parquet(cache_file)
            if "is_mitigated" not in df_fvgs.columns or "first_open" not in df_fvgs.columns:
                print(f"[WARNING] Cached {label} data missing critical columns, regenerating...")
                df_fvgs = None
        except Exception as e:
            print(f"[WARNING] Error loading cached {label} data: {e}. Regenerating...")
            df_fvgs = None

    if df_fvgs is None:
        print(f"[INFO] Computing {label} FVG results using size-based detection")
        fvg_list = detect_fvg(df_detect, min_size_threshold=fvg_size_threshold)

        fvg_results = find_fvg_mitigations(
            df_mitigate, df_detect, fvg_list,
            fvg_filter_start_time, filter_end_time,
            search_bars_primary=search_bars_primary,
            search_bars_fallback=search_bars_fallback,
            primary_label=primary_label,
            fallback_label=fallback_label,
        )
        df_fvgs = pd.DataFrame(fvg_results)

        df_fvgs.to_parquet(cache_file, index=False)
        print(f"[INFO] Saved {label} FVG results to {cache_file}")

    return df_fvgs


def _compute_expansions(df_fvgs, df_expansion, cache_file, label, exp_size_threshold, filter_end_time):
    """Compute expansion data for mitigated FVGs if not already present."""
    expansion_cols = ["expansion_size", "expansion_time_seconds", "max_penetration_depth", "penetration_candle_count", "midpoint_crossing_count", "time_to_target_seconds", "time_to_invalidation_seconds", "risk_points", "achieved_rr"]
    missing_cols = [col for col in expansion_cols if col not in df_fvgs.columns]

    if not missing_cols and "is_expansion_valid" in df_fvgs.columns:
        return df_fvgs

    print(f"[INFO] Computing {label} expansion data...")

    for col in missing_cols:
        df_fvgs[col] = np.nan
    df_fvgs["is_expansion_valid"] = False

    mitigated = df_fvgs[df_fvgs["is_mitigated"]].copy()
    if not mitigated.empty:
        print(f"[INFO] Computing expansions for {len(mitigated)} mitigated {label} FVGs...")
        for idx, row in mitigated.iterrows():
            expansion_size, expansion_time, is_valid, _, pen_depth, pen_count, midpoint_count, ttt_secs, tti_secs, risk_pts, achieved_rr = (
                calculate_expansion_after_mitigation(
                    None, df_expansion, row, filter_end_time, debug=False,
                    expansion_size_threshold=exp_size_threshold,
                )
            )
            df_fvgs.loc[idx, "expansion_size"] = expansion_size
            df_fvgs.loc[idx, "expansion_time_seconds"] = (
                expansion_time.total_seconds() if expansion_time else np.nan
            )
            df_fvgs.loc[idx, "is_expansion_valid"] = is_valid
            df_fvgs.loc[idx, "max_penetration_depth"] = pen_depth
            df_fvgs.loc[idx, "penetration_candle_count"] = pen_count
            df_fvgs.loc[idx, "midpoint_crossing_count"] = midpoint_count
            df_fvgs.loc[idx, "time_to_target_seconds"] = ttt_secs if ttt_secs is not None else np.nan
            df_fvgs.loc[idx, "time_to_invalidation_seconds"] = tti_secs if tti_secs is not None else np.nan
            df_fvgs.loc[idx, "risk_points"] = risk_pts
            df_fvgs.loc[idx, "achieved_rr"] = achieved_rr

        df_fvgs.to_parquet(cache_file, index=False)
        print(f"[INFO] Updated {label} cache with expansion data")

    return df_fvgs


def _compute_rr_trades(df_fvgs, df_expansion, cache_file, label, filter_end_time):
    """Compute RR trade simulations for all 4 setups if not already present."""
    from utils.rr_analysis import compute_rr_for_fvgs
    return compute_rr_for_fvgs(df_fvgs, df_expansion, filter_end_time, cache_file=cache_file)


def _run_rr_analysis(df_fvgs, label, interval_minutes, filter_end_time):
    """Run RR aggregation (time x risk) and save to store."""
    from utils.rr_analysis import aggregate_rr_cells, save_rr_dataset

    print(f"\n[INFO] Running {label} RR expectancy analysis (time x risk)...")

    cells = aggregate_rr_cells(
        df_fvgs,
        fvg_filter_start_time=fvg_filter_start_time,
        fvg_filter_end_time=filter_end_time,
        interval_minutes=interval_minutes,
        min_samples=5,
    )

    if not cells:
        print(f"[WARNING] No RR cells generated for {label}")
        return

    try:
        save_rr_dataset(
            cells=cells,
            ticker=PRIMARY_MARKET,
            timeframe_label=label,
            data_period=period,
            session_period_minutes=interval_minutes,
        )
    except Exception as e:
        print(f"[WARNING] RR store write failed (non-fatal): {e}")


def _print_summary(df_fvgs, label):
    """Print summary statistics for a pipeline run."""
    mitigated_df = df_fvgs[df_fvgs["is_mitigated"]].copy()
    total = len(df_fvgs)
    mitigated = len(mitigated_df)
    mit_pct = (mitigated / total * 100) if total > 0 else 0
    valid_exp = mitigated_df["is_expansion_valid"].sum() if not mitigated_df.empty else 0
    invalidated = mitigated - valid_exp
    inv_pct = (invalidated / mitigated * 100) if mitigated > 0 else 0

    print(f"\n[INFO] === {label.upper()} ANALYSIS SUMMARY ===")
    print(f"[INFO] Total FVGs detected: {total:,}")
    print(f"[INFO] Mitigated FVGs: {mitigated:,} ({mit_pct:.1f}%)")
    print(f"[INFO] Valid expansions: {valid_exp:,}")
    print(f"[INFO] Invalidated FVGs: {invalidated:,} ({inv_pct:.1f}%)")


def _run_analysis_and_viz(df_fvgs, label, get_sizes_range_fn, exp_size_threshold, interval_minutes, filtering_method, filter_end_time):
    """Run size-time analysis, generate visualizations, and export results."""
    print(f"\n[INFO] Running {label} size-based time period analysis...")

    size_thresholds = get_sizes_range_fn()
    print(f"[INFO] Testing {len(size_thresholds)} size thresholds: {size_thresholds}")
    print(f"[INFO] Using {interval_minutes}-minute session periods for analysis")

    size_time_results_df = analyze_fvg_size_time_distribution(
        df_fvgs,
        size_thresholds,
        fvg_filter_start_time,
        filter_end_time,
        interval_minutes=interval_minutes,
        size_filtering_method=filtering_method,
    )

    if size_time_results_df.empty:
        print(f"[WARNING] No valid {label} size-time analysis results generated")
        return

    # Drop incomplete time periods (shorter than interval_minutes)
    if "time_period" in size_time_results_df.columns:
        def _period_is_full(tp):
            try:
                start, end = str(tp).split("-")
                sh, sm = map(int, start.strip().split(":"))
                eh, em = map(int, end.strip().split(":"))
                return (eh * 60 + em) - (sh * 60 + sm) >= interval_minutes
            except (ValueError, IndexError):
                return True
        full_mask = size_time_results_df["time_period"].apply(_period_is_full)
        dropped = (~full_mask).sum()
        if dropped > 0:
            dropped_periods = size_time_results_df.loc[~full_mask, "time_period"].unique()
            print(f"[INFO] Dropping {dropped} rows from incomplete periods: {list(dropped_periods)}")
            size_time_results_df = size_time_results_df[full_mask].reset_index(drop=True)

    print(f"[INFO] Creating {label} visualizations...")
    period_str = format_period_str(period)
    filename_prefix = (
        f"{PRIMARY_MARKET.lower()}_fvg_{label}_{period_str}_{interval_minutes}min_"
        f"size_time_analysis_{size_thresholds[0]:.2f}to{size_thresholds[-1]:.2f}_"
        f"exp{exp_size_threshold}"
    )

    # Generate static PNG charts and heatmaps
    create_analysis_plots(size_time_results_df, filename_prefix, save_plots=True)
    create_time_based_heatmaps(size_time_results_df, filename_prefix, save_plots=True)
    create_mitigation_time_heatmap(size_time_results_df, filename_prefix, save_plots=True)

    # Generate interactive Plotly HTML charts
    create_interactive_heatmaps(size_time_results_df, filename_prefix)
    create_interactive_mitigation_heatmap(size_time_results_df, filename_prefix)

    # Aggregate expansion optimizer across all cells
    all_expansion = df_fvgs.loc[
        df_fvgs["is_mitigated"] & df_fvgs["expansion_size"].notna(),
        "expansion_size",
    ]
    if len(all_expansion) > 0:
        agg_opt = optimize_expansion_target(all_expansion)
        print(f"[INFO] Aggregate optimal target: {agg_opt['optimal_target']:.1f} pts "
              f"(EV={agg_opt['optimal_ev']:.2f}, hit_rate={agg_opt['optimal_hit_rate']:.1%})")
        create_optimizer_chart(agg_opt["curve_df"], label, filename_prefix)

    # Save insights and export CSV
    save_fvg_size_time_insights(size_time_results_df, filename_prefix)
    csv_filename = f"csv/{filename_prefix}_results.csv"
    size_time_results_df.to_csv(csv_filename, index=False)
    print(f"[INFO] Results exported to {csv_filename}")

    try:
        from utils.heatmap_store import save_dataset
        save_dataset(
            df=size_time_results_df,
            ticker=PRIMARY_MARKET,
            timeframe_label=label,
            data_period=period,
            session_period_minutes=interval_minutes,
            size_filtering_method=filtering_method,
            size_range_start=float(size_thresholds[0]),
            size_range_end=float(size_thresholds[-1]),
            size_range_step=float(size_thresholds[1] - size_thresholds[0]) if len(size_thresholds) > 1 else 1.0,
            min_expansion_size=exp_size_threshold,
            source_csv_path=csv_filename,
        )
        print("[INFO] Heatmap data exported to store")
    except Exception as e:
        print(f"[WARNING] Heatmap store write failed (non-fatal): {e}")

    # Print top performing configurations
    print(f"\n[INFO] === {label.upper()} TOP PERFORMING SIZE THRESHOLDS ===")
    valid_results = size_time_results_df[size_time_results_df["total_fvgs"] > 0]

    if not valid_results.empty:
        best = valid_results.loc[valid_results["invalidation_rate"].idxmin()]
        print(f"Best invalidation rate: {best['invalidation_rate']:.1f}% for threshold {best['size_range']}, {best['time_period']}")

        best = valid_results.loc[valid_results["avg_expansion_size"].idxmax()]
        print(f"Best expansion size: {best['avg_expansion_size']:.2f} points for threshold {best['size_range']}, {best['time_period']}")

        best = valid_results.loc[valid_results["mitigation_rate"].idxmax()]
        print(f"Best mitigation rate: {best['mitigation_rate']:.1f}% for threshold {best['size_range']}, {best['time_period']}")


def main():
    """Main analysis function - runs FVG pipelines across timeframe pairs."""

    print(f"[INFO] Starting FVG analysis for {PRIMARY_MARKET}")
    print(f"[INFO] Session: {fvg_filter_start_time} to {fvg_filter_end_time}")
    print(f"[INFO] Period: {period}" + (f" (custom: {custom_start} to {custom_end})" if custom_start and custom_end else ""))

    # Prepare cache directories
    data_cache_dir = "data_cache"
    fvg_cache_dir = "fvg_cache"
    for cache_dir in [data_cache_dir, fvg_cache_dir]:
        os.makedirs(cache_dir, exist_ok=True)

    # Generate expiration dates
    current_year = datetime.now().year
    start_year = current_year - 2
    end_year = current_year + 1

    if PRIMARY_MARKET == "NQ":
        expiration_dates = generate_nq_expirations(start_year, end_year)
    elif PRIMARY_MARKET == "ES":
        expiration_dates = generate_es_expirations(start_year, end_year)
    else:
        raise ValueError(f"Unsupported market: {PRIMARY_MARKET}")

    # Fetch all timeframes
    print(f"[INFO] Fetching market data...")
    fetch_kwargs = dict(
        symbol=PRIMARY_MARKET,
        expiration_dates=expiration_dates,
        data_cache_dir=data_cache_dir,
        period=period,
        custom_start=custom_start,
        custom_end=custom_end,
        roll_days=roll_days,
    )

    # Allow API to run only one pipeline (e.g. FVG_TIMEFRAME=15min skips 5min)
    _run_tf = os.environ.get("FVG_TIMEFRAME", "all")

    # Fetch only the timeframes needed
    df_1min = fetch_market_data(timeframe="1min", min_search_bars=min_search_bars_1min, **fetch_kwargs)
    df_3min = None
    df_5min = None
    df_15min = None

    if _run_tf in ("all", "3min"):
        df_3min = fetch_market_data(timeframe="3min", min_search_bars=min_search_bars_3min, **fetch_kwargs)
    if _run_tf in ("all", "5min", "15min"):
        df_5min = fetch_market_data(timeframe="5min", min_search_bars=min_search_bars_5min, **fetch_kwargs)
    if _run_tf in ("all", "15min"):
        df_15min = fetch_market_data(timeframe="15min", min_search_bars=min_search_bars_15min, **fetch_kwargs)

    if df_1min.empty:
        print(f"[ERROR] Failed to fetch 1min market data")
        return

    loaded = [f"{len(df_1min)} 1min"]
    if df_3min is not None: loaded.append(f"{len(df_3min)} 3min")
    if df_5min is not None: loaded.append(f"{len(df_5min)} 5min")
    if df_15min is not None: loaded.append(f"{len(df_15min)} 15min")
    print(f"[INFO] Data loaded: {', '.join(loaded)} bars")

    # Reset indices for processing
    for df in [df_1min, df_3min, df_5min, df_15min]:
        if df is not None:
            df.reset_index(drop=False, inplace=True)

    # --- Pipeline 0: 3-min detect, 1-min mitigate, 3-min expansion ---
    if _run_tf in ("all", "3min") and df_3min is not None:
        run_fvg_pipeline(
            label="3min",
            df_detect=df_3min,
            df_mitigate=df_1min,
            df_expansion=df_3min,
            fvg_size_threshold=min_fvg_size_3min,
            exp_size_threshold=min_expansion_size_3min,
            get_sizes_range_fn=lambda: get_min_fvg_sizes_range(config_key="min_fvg_sizes_range_3min"),
            search_bars_primary=min_search_bars_1min_mitigation_3min,
            search_bars_fallback=min_search_bars_3min,
            primary_label="1min",
            fallback_label="3min",
            fvg_cache_dir=fvg_cache_dir,
            interval_minutes=session_period_minutes_3min,
            filtering_method=size_filtering_method_3min,
            filter_end_time=fvg_filter_end_time_3min or fvg_filter_end_time,
        )

    # --- Pipeline 1: 5-min detect, 1-min mitigate, 5-min expansion ---
    if _run_tf in ("all", "5min") and df_5min is not None:
        run_fvg_pipeline(
            label="5min",
            df_detect=df_5min,
            df_mitigate=df_1min,
            df_expansion=df_5min,
            fvg_size_threshold=min_fvg_size,
            exp_size_threshold=min_expansion_size,
            get_sizes_range_fn=get_min_fvg_sizes_range,
            search_bars_primary=min_search_bars_1min,
            search_bars_fallback=min_search_bars_5min,
            primary_label="1min",
            fallback_label="5min",
            fvg_cache_dir=fvg_cache_dir,
        )

    # Free 1-min data — no longer needed after Pipeline 0 & 1
    del df_1min
    if df_3min is not None:
        del df_3min

    # --- Pipeline 2: 15-min detect, 5-min mitigate, 5-min expansion ---
    if _run_tf in ("all", "15min") and df_5min is not None and df_15min is not None:
        run_fvg_pipeline(
            label="15min",
            df_detect=df_15min,
            df_mitigate=df_5min,
            df_expansion=df_5min,
            fvg_size_threshold=min_fvg_size_15min,
            exp_size_threshold=min_expansion_size_15min,
            get_sizes_range_fn=lambda: get_min_fvg_sizes_range(config_key="min_fvg_sizes_range_15min"),
            search_bars_primary=min_search_bars_5min_mitigation,
            search_bars_fallback=min_search_bars_15min,
            primary_label="5min",
            fallback_label="15min",
            fvg_cache_dir=fvg_cache_dir,
            interval_minutes=session_period_minutes_15min,
            filtering_method=size_filtering_method_15min,
            filter_end_time=fvg_filter_end_time_15min or fvg_filter_end_time,
        )

    print(f"\n[INFO] Analysis complete!")


if __name__ == "__main__":
    main()
