import os

import numpy as np
import pandas as pd


def find_optimal_fvg_size_time(
    results_df: pd.DataFrame,
    weight_expansion: float = 0.4,
    weight_invalidation: float = 0.6,
):
    """Find the optimal FVG size and time combination based on weighted scoring."""

    # Filter valid results
    valid_results = results_df[
        (results_df["total_fvgs"] > 0)
        & (results_df["avg_expansion_size"] > 0)
        & (results_df["invalidation_rate"] >= 0)
    ].copy()

    if len(valid_results) == 0:
        return None

    # Normalize metrics to 0-1 scale
    exp_range = (
        valid_results["avg_expansion_size"].max()
        - valid_results["avg_expansion_size"].min()
    )
    if exp_range > 0:
        valid_results["norm_expansion"] = (
            valid_results["avg_expansion_size"]
            - valid_results["avg_expansion_size"].min()
        ) / exp_range
    else:
        valid_results["norm_expansion"] = 0.5

    valid_results["norm_invalidation"] = 1 - (
        valid_results["invalidation_rate"] / 100
    )  # Invert so lower invalidation = higher score

    # Calculate weighted score
    valid_results["weighted_score"] = (
        weight_expansion * valid_results["norm_expansion"]
        + weight_invalidation * valid_results["norm_invalidation"]
    )

    # Find optimal
    if isinstance(valid_results, pd.DataFrame):
        optimal_idx = valid_results["weighted_score"].idxmax()
        optimal_row = valid_results.loc[optimal_idx]
    else:
        return None

    return {
        "optimal_time_period": optimal_row["time_period"],
        "optimal_min_fvg_size": optimal_row["min_fvg_size"],
        "optimal_size_range": optimal_row["size_range"],
        "total_fvgs": optimal_row["total_fvgs"],
        "invalidation_rate": optimal_row["invalidation_rate"],
        "avg_expansion_size": optimal_row["avg_expansion_size"],
        "expansion_efficiency": optimal_row["expansion_efficiency"],
        "mitigation_rate": optimal_row["mitigation_rate"],
        "weighted_score": optimal_row["weighted_score"],
        "weights_used": {
            "expansion": weight_expansion,
            "invalidation": weight_invalidation,
        },
    }


def save_insights_to_file(filename_prefix: str, insights_content: str):
    """Save insights and statistics to a text file."""
    insights_filename = f"{filename_prefix}_insights.txt"
    filepath = os.path.join("csv", insights_filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(insights_content)
        print(f"[INFO] Insights saved to {filepath}")
    except Exception as e:
        print(f"[WARNING] Failed to save insights file: {e}")


def create_filename_prefix(min_fvg_sizes: np.ndarray) -> str:
    """Create a filename prefix based on min_fvg_sizes range."""
    min_val = min_fvg_sizes[0]
    max_val = min_fvg_sizes[-1]
    step = min_fvg_sizes[1] - min_fvg_sizes[0] if len(min_fvg_sizes) > 1 else 0.25

    return f"fvg_analysis_{min_val:.2f}to{max_val:.2f}by{step:.2f}"


def save_body_multiplier_insights(results_df, filename_prefix: str):
    """Save insights and statistics for body multiplier analysis."""

    insights_output = []

    def add_insight(text=""):
        insights_output.append(text)

    add_insight("=" * 80)
    add_insight("FVG BODY MULTIPLIER ANALYSIS INSIGHTS")
    add_insight("=" * 80)

    # Filter valid results
    valid_results = results_df[results_df["total_fvgs"] > 0].copy()

    if valid_results.empty:
        add_insight("No valid results found for analysis.")
        return

    add_insight("\nSUMMARY STATISTICS")
    add_insight("=" * 50)

    # Overall statistics
    total_fvgs = valid_results["total_fvgs"].sum()
    total_mitigated = valid_results["mitigated_fvgs"].sum()
    total_valid_expansions = valid_results["valid_expansions"].sum()

    add_insight(f"Total FVGs analyzed: {total_fvgs:,}")
    add_insight(f"Total mitigated FVGs: {total_mitigated:,}")
    add_insight(f"Total valid expansions: {total_valid_expansions:,}")
    add_insight(f"Overall mitigation rate: {(total_mitigated / total_fvgs * 100):.1f}%")
    add_insight(
        f"Overall invalidation rate: {((total_mitigated - total_valid_expansions) / total_mitigated * 100):.1f}%"
    )

    # Best performing body multipliers
    add_insight(f"\nTOP PERFORMING BODY MULTIPLIERS")
    add_insight("=" * 50)

    # Best by mitigation rate
    best_mitigation = valid_results.loc[valid_results["mitigation_percentage"].idxmax()]
    add_insight(
        f"Best mitigation rate: {best_mitigation['mitigation_percentage']:.1f}% at body multiplier {best_mitigation['body_multiplier']:.1f}"
    )

    # Best by expansion size
    best_expansion = valid_results.loc[valid_results["avg_expansion"].idxmax()]
    add_insight(
        f"Best expansion size: {best_expansion['avg_expansion']:.2f} points at body multiplier {best_expansion['body_multiplier']:.1f}"
    )

    # Best by invalidation rate (lowest)
    best_invalidation = valid_results.loc[
        valid_results["invalidated_percentage"].idxmin()
    ]
    add_insight(
        f"Best invalidation rate: {best_invalidation['invalidated_percentage']:.1f}% at body multiplier {best_invalidation['body_multiplier']:.1f}"
    )

    # Sweet spot analysis
    add_insight(f"\nSWEET SPOT ANALYSIS")
    add_insight("=" * 50)

    # Define sweet spot criteria
    avg_expansion = valid_results["avg_expansion"].mean()
    sweet_spot = valid_results[
        (valid_results["invalidated_percentage"] < 25)
        & (valid_results["avg_expansion"] > avg_expansion)
        & (valid_results["mitigation_percentage"] > 50)
    ].copy()

    if len(sweet_spot) > 0:
        # Calculate a composite score
        sweet_spot["composite_score"] = (
            (sweet_spot["avg_expansion"] / sweet_spot["avg_expansion"].max()) * 0.4
            + (
                sweet_spot["mitigation_percentage"]
                / sweet_spot["mitigation_percentage"].max()
            )
            * 0.3
            + ((100 - sweet_spot["invalidated_percentage"]) / 100) * 0.3
        )

        best_sweet_spot = sweet_spot.loc[sweet_spot["composite_score"].idxmax()]
        add_insight(
            f"Recommended body multiplier: {best_sweet_spot['body_multiplier']:.1f}"
        )
        add_insight(f"  - Total FVGs: {best_sweet_spot['total_fvgs']:,}")
        add_insight(
            f"  - Mitigation rate: {best_sweet_spot['mitigation_percentage']:.1f}%"
        )
        add_insight(
            f"  - Average expansion: {best_sweet_spot['avg_expansion']:.2f} points"
        )
        add_insight(
            f"  - Invalidation rate: {best_sweet_spot['invalidated_percentage']:.1f}%"
        )
        add_insight(f"  - Composite score: {best_sweet_spot['composite_score']:.3f}")
    else:
        add_insight("No body multipliers meet the sweet spot criteria.")

    # Performance trends
    add_insight(f"\nPERFORMANCE TRENDS")
    add_insight("=" * 50)

    # Correlation analysis
    corr_expansion = valid_results["body_multiplier"].corr(
        valid_results["avg_expansion"]
    )
    corr_mitigation = valid_results["body_multiplier"].corr(
        valid_results["mitigation_percentage"]
    )
    corr_invalidation = valid_results["body_multiplier"].corr(
        valid_results["invalidated_percentage"]
    )

    add_insight(f"Correlation with body multiplier:")
    add_insight(f"  - Expansion size: {corr_expansion:.3f}")
    add_insight(f"  - Mitigation rate: {corr_mitigation:.3f}")
    add_insight(f"  - Invalidation rate: {corr_invalidation:.3f}")

    # Save insights
    insights_content = "\n".join(insights_output)
    save_insights_to_file(filename_prefix, insights_content)


def save_time_period_insights(time_period_results_df, filename_prefix: str):
    """Save insights and statistics for time period analysis."""

    insights_output = []

    def add_insight(text=""):
        insights_output.append(text)

    add_insight("=" * 80)
    add_insight("FVG TIME PERIOD ANALYSIS INSIGHTS")
    add_insight("=" * 80)

    if time_period_results_df.empty:
        add_insight("No time period results found for analysis.")
        return

    add_insight("\nTIME PERIOD PERFORMANCE SUMMARY")
    add_insight("=" * 50)

    # Aggregate by time period
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
        .round(2)
    )

    for time_period, stats in time_summary.iterrows():
        add_insight(
            f"{time_period}: {int(stats['total_fvgs'])} FVGs, "
            f"{stats['invalidation_rate']:.1f}% invalidation, "
            f"{stats['avg_expansion_size']:.1f} avg expansion, "
            f"{stats['mitigation_rate']:.1f}% mitigation"
        )

    # Best and worst time periods
    add_insight(f"\nBEST AND WORST TIME PERIODS")
    add_insight("=" * 50)

    best_expansion_period = time_summary.loc[
        time_summary["avg_expansion_size"].idxmax()
    ]
    worst_expansion_period = time_summary.loc[
        time_summary["avg_expansion_size"].idxmin()
    ]

    add_insight(
        f"Best expansion period: {best_expansion_period.name} ({best_expansion_period['avg_expansion_size']:.1f} points)"
    )
    add_insight(
        f"Worst expansion period: {worst_expansion_period.name} ({worst_expansion_period['avg_expansion_size']:.1f} points)"
    )

    best_invalidation_period = time_summary.loc[
        time_summary["invalidation_rate"].idxmin()
    ]
    worst_invalidation_period = time_summary.loc[
        time_summary["invalidation_rate"].idxmax()
    ]

    add_insight(
        f"Best invalidation period: {best_invalidation_period.name} ({best_invalidation_period['invalidation_rate']:.1f}%)"
    )
    add_insight(
        f"Worst invalidation period: {worst_invalidation_period.name} ({worst_invalidation_period['invalidation_rate']:.1f}%)"
    )

    # Body multiplier insights by time period
    add_insight(f"\nBODY MULTIPLIER PERFORMANCE BY TIME PERIOD")
    add_insight("=" * 50)

    for time_period in time_period_results_df["time_period"].unique():
        period_data = time_period_results_df[
            time_period_results_df["time_period"] == time_period
        ]
        if not period_data.empty:
            best_bm = period_data.loc[period_data["avg_expansion_size"].idxmax()]
            add_insight(
                f"{time_period}: Best body multiplier {best_bm['body_multiplier']:.1f} "
                f"({best_bm['avg_expansion_size']:.1f} expansion, {best_bm['invalidation_rate']:.1f}% invalidation)"
            )

    # Save insights
    insights_content = "\n".join(insights_output)
    save_insights_to_file(f"{filename_prefix}_time_period", insights_content)


def save_fvg_size_time_insights(size_time_results_df, filename_prefix: str):
    """Save insights and statistics for FVG size and time period analysis."""

    insights_output = []

    def add_insight(text=""):
        insights_output.append(text)

    add_insight("=" * 80)
    add_insight("FVG SIZE AND TIME PERIOD ANALYSIS INSIGHTS")
    add_insight("=" * 80)

    if size_time_results_df.empty:
        add_insight("No size and time period results found for analysis.")
        return

    # Filter valid results
    valid_results = size_time_results_df[size_time_results_df["total_fvgs"] > 0]

    if len(valid_results) > 0:
        add_insight(
            f"Minimum FVG size thresholds tested: {len(valid_results['size_range'].unique())} different levels"
        )
        add_insight(
            f"Time intervals analyzed: {len(valid_results['time_period'].unique())}"
        )
        add_insight(f"Total configurations tested: {len(valid_results)}")
        add_insight(
            f"Average invalidation rate: {valid_results['invalidation_rate'].mean():.2f}%"
        )
        add_insight(
            f"Average expansion size: {valid_results['avg_expansion_size'].mean():.2f} points"
        )

        # Best configurations
        best_inv_idx = valid_results["invalidation_rate"].idxmin()
        best_exp_idx = valid_results["avg_expansion_size"].idxmax()
        add_insight(
            f"Best invalidation rate: {valid_results.loc[best_inv_idx, 'invalidation_rate']:.2f}% for threshold {valid_results.loc[best_inv_idx, 'size_range']}, {valid_results.loc[best_inv_idx, 'time_period']}"
        )
        add_insight(
            f"Best expansion size: {valid_results.loc[best_exp_idx, 'avg_expansion_size']:.2f} points for threshold {valid_results.loc[best_exp_idx, 'size_range']}, {valid_results.loc[best_exp_idx, 'time_period']}"
        )

        # Time-based insights
        add_insight("\n" + "=" * 50)
        add_insight("TIME-BASED INSIGHTS")
        add_insight("=" * 50)

        # Calculate unique totals using smallest threshold
        # Handle both cumulative ("0.25") and bins ("0.25-1.25") formats
        def extract_min_threshold(size_range_str):
            """Extract the minimum threshold from size_range string."""
            if '-' in size_range_str:
                # Bins format: "0.25-1.25" -> return 0.25
                return float(size_range_str.split('-')[0])
            else:
                # Cumulative format: "0.25" -> return 0.25
                return float(size_range_str)
        
        thresholds = valid_results["size_range"].apply(extract_min_threshold)
        smallest_threshold = thresholds.min()
        
        # Find the size_range string that contains this smallest threshold
        smallest_threshold_mask = thresholds == smallest_threshold
        smallest_threshold_str = valid_results.loc[smallest_threshold_mask, "size_range"].iloc[0]
        
        unique_totals = valid_results[
            valid_results["size_range"] == smallest_threshold_str
        ].set_index("time_period")["total_fvgs"]

        time_stats = (
            valid_results.groupby("time_period")
            .agg(
                {
                    "invalidation_rate": "mean",
                    "avg_expansion_size": "mean",
                    "mitigation_rate": "mean",
                    "p75_mitigation_time": "mean",
                }
            )
            .round(2)
        )

        add_insight("Performance by time period (unique FVGs):")
        for time_period, stats in time_stats.iterrows():
            unique_fvgs = int(unique_totals.get(time_period, 0))
            add_insight(
                f"  {time_period}: {unique_fvgs} unique FVGs, {stats['invalidation_rate']:.1f}% avg invalidation, {stats['avg_expansion_size']:.1f} avg expansion, {stats['mitigation_rate']:.1f}% avg mitigation, {stats['p75_mitigation_time']:.1f} min avg p75 time"
            )

    add_insight("\nSize and time period analysis complete!")

    # Save insights
    insights_content = "\n".join(insights_output)
    save_insights_to_file(f"{filename_prefix}_size_time", insights_content)
