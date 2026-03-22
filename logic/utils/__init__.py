"""
Financial Content Automation - Utils Package

This package contains refactored utility modules organized by theme:

- time_utils: Time and timezone handling utilities
- contract_utils: Futures contract management utilities
- data_cache_utils: Data caching and fetching utilities
- fvg_detection: Fair Value Gap detection algorithms
- fvg_analysis: FVG analysis and processing functions
- visualization_utils: Chart and plot generation utilities
- statistical_analysis: Statistical analysis functions
- insights_utils: Insights generation and file output utilities
"""

from .contract_utils import (
    create_contract,
    generate_es_expirations,
    generate_nq_expirations,
    get_contract_for_date,
)
from .data_cache_utils import fetch_and_cache, fetch_market_data, get_filename
from .fvg_analysis import (
    analyze_fvg_size_time_distribution,
    analyze_fvgs_by_time_period,
    calculate_expansion_after_mitigation,
    calculate_fvg_size,
    find_fvg_mitigations,
    get_random_fvg_with_expansion,
    optimize_expansion_target,
    study_specific_fvg,
)
from .fvg_detection import (
    detect_fvg,
    detect_fvg_by_size_ranges,
    detect_fvg_with_size_threshold,
)
from .insights_utils import (
    create_filename_prefix,
    find_optimal_fvg_size_time,
    save_fvg_size_time_insights,
    save_insights_to_file,
    save_time_period_insights,
)
from .statistical_analysis import (
    calculate_combined_time_stats,
    calculate_day_of_week_stats,
    calculate_time_period_stats,
    crosscheck_invalidated_fvgs,
    crosscheck_valid_expansions,
    crosscheck_all_fvgs,
)

# Import commonly used functions for easier access
from .time_utils import (
    assign_time_period_to_fvgs,
    create_time_intervals,
    ensure_ny_timezone,
    format_timedelta_analysis,
)
from .interactive_plots import (
    create_interactive_heatmaps,
    create_interactive_mitigation_heatmap,
    create_optimizer_chart,
)
from .rr_analysis import (
    compute_rr_for_fvgs,
    aggregate_rr_cells,
    save_rr_dataset,
    load_rr_dataset,
    get_rr_manifest,
    SETUPS as RR_SETUPS,
    N_VALUES as RR_N_VALUES,
)
from .visualization_utils import (
    create_analysis_plots,
    create_expansion_percentile_charts,
    create_time_based_heatmaps,
    create_mitigation_time_heatmap,
    plot_fvg_expansion,
)

__all__ = [
    # time_utils
    "format_timedelta_analysis",
    "ensure_ny_timezone",
    "create_time_intervals",
    "assign_time_period_to_fvgs",
    # contract_utils
    "generate_nq_expirations",
    "generate_es_expirations",
    "create_contract",
    "get_contract_for_date",
    # data_cache_utils
    "fetch_and_cache",
    "fetch_market_data", 
    "get_filename",
    # fvg_detection
    "detect_fvg",
    "detect_fvg_by_size_ranges",
    "detect_fvg_with_size_threshold",
    # fvg_analysis
    "study_specific_fvg",
    "find_fvg_mitigations",
    "calculate_expansion_after_mitigation",
    "calculate_fvg_size",
    "get_random_fvg_with_expansion",
    "analyze_fvgs_by_time_period",
    "analyze_fvg_size_time_distribution",
    "optimize_expansion_target",
    # visualization_utils
    "plot_fvg_expansion",
    "create_time_based_heatmaps",
    "create_mitigation_time_heatmap",
    "create_analysis_plots",
    "create_expansion_percentile_charts",
    # interactive_plots
    "create_interactive_heatmaps",
    "create_interactive_mitigation_heatmap",
    "create_optimizer_chart",
    # statistical_analysis
    "calculate_day_of_week_stats",
    "calculate_time_period_stats",
    "calculate_combined_time_stats",
    "crosscheck_invalidated_fvgs",
    "crosscheck_valid_expansions",
    "crosscheck_all_fvgs",
    # insights_utils
    "find_optimal_fvg_size_time",
    "save_insights_to_file",
    "create_filename_prefix",
    "save_time_period_insights",
    "save_fvg_size_time_insights",
]
