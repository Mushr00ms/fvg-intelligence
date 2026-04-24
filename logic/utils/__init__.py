"""
Utilities exposed at the package level via lazy imports.

Keeping these imports lazy avoids pulling heavy analysis dependencies such as
NumPy/Pandas into lightweight runtime paths that only need contract helpers.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "aggregate_rr_cells": "rr_analysis",
    "analyze_fvg_size_time_distribution": "fvg_analysis",
    "analyze_fvgs_by_time_period": "fvg_analysis",
    "assign_time_period_to_fvgs": "time_utils",
    "calculate_combined_time_stats": "statistical_analysis",
    "calculate_day_of_week_stats": "statistical_analysis",
    "calculate_expansion_after_mitigation": "fvg_analysis",
    "calculate_fvg_size": "fvg_analysis",
    "calculate_time_period_stats": "statistical_analysis",
    "compute_rr_for_fvgs": "rr_analysis",
    "create_analysis_plots": "visualization_utils",
    "create_contract": "contract_utils",
    "create_expansion_percentile_charts": "visualization_utils",
    "create_filename_prefix": "insights_utils",
    "create_interactive_heatmaps": "interactive_plots",
    "create_interactive_mitigation_heatmap": "interactive_plots",
    "create_mitigation_time_heatmap": "visualization_utils",
    "create_optimizer_chart": "interactive_plots",
    "create_time_based_heatmaps": "visualization_utils",
    "create_time_intervals": "time_utils",
    "crosscheck_all_fvgs": "statistical_analysis",
    "crosscheck_invalidated_fvgs": "statistical_analysis",
    "crosscheck_valid_expansions": "statistical_analysis",
    "detect_fvg": "fvg_detection",
    "detect_fvg_by_size_ranges": "fvg_detection",
    "detect_fvg_with_size_threshold": "fvg_detection",
    "ensure_ny_timezone": "time_utils",
    "fetch_and_cache": "data_cache_utils",
    "fetch_market_data": "data_cache_utils",
    "find_fvg_mitigations": "fvg_analysis",
    "find_optimal_fvg_size_time": "insights_utils",
    "format_timedelta_analysis": "time_utils",
    "generate_es_expirations": "contract_utils",
    "generate_nq_expirations": "contract_utils",
    "get_contract_for_date": "contract_utils",
    "get_filename": "data_cache_utils",
    "get_random_fvg_with_expansion": "fvg_analysis",
    "get_rr_manifest": "rr_analysis",
    "load_rr_dataset": "rr_analysis",
    "N_VALUES": "rr_analysis",
    "optimize_expansion_target": "fvg_analysis",
    "plot_fvg_expansion": "visualization_utils",
    "RR_SETUPS": "rr_analysis",
    "save_fvg_size_time_insights": "insights_utils",
    "save_insights_to_file": "insights_utils",
    "save_rr_dataset": "rr_analysis",
    "save_time_period_insights": "insights_utils",
    "study_specific_fvg": "fvg_analysis",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
