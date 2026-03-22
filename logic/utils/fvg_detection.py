from typing import List, Optional, Tuple

import pandas as pd

try:
    # Try relative import first (when used as a module)
    from ..config import min_fvg_size
except ImportError:
    # Fall back to absolute import (when run from logic directory)
    from config import min_fvg_size


def detect_fvg(data, min_size_threshold=None):
    """
    Simplified FVG detection based purely on gap size, removing body multiplier logic.

    Args:
        data: DataFrame with OHLC data
        min_size_threshold: Minimum FVG size in points (uses config default if None)

    Returns:
        List of FVG tuples or None values
        Tuple format: (fvg_type, y0, y1, time_candle1, time_candle2, time_candle3, idx, middle_open, middle_low, middle_high, first_open)
    """
    if min_size_threshold is None:
        min_size_threshold = min_fvg_size

    fvg_list: list[
        Optional[Tuple[str, float, float, object, object, object, int, float, float, float, float]]
    ] = [None] * len(data)

    for i in range(2, len(data)):
        first_high = data["high"].iloc[i - 2]
        first_low = data["low"].iloc[i - 2]
        first_open = data["open"].iloc[i - 2]
        middle_open = data["open"].iloc[i - 1]
        middle_low = data["low"].iloc[i - 1]
        middle_high = data["high"].iloc[i - 1]
        third_low = data["low"].iloc[i]
        third_high = data["high"].iloc[i]

        time_candle1 = data["date"].iloc[i - 2]
        time_candle2 = data["date"].iloc[i - 1]
        time_candle3 = data["date"].iloc[i]

        # Check for bullish FVG (gap between first candle high and third candle low)
        if third_low > first_high:
            fvg_size = third_low - first_high
            if fvg_size >= min_size_threshold:
                fvg_list[i] = (
                    "bullish",
                    first_high,
                    third_low,
                    time_candle1,
                    time_candle2,
                    time_candle3,
                    i,
                    middle_open,
                    middle_low,
                    middle_high,
                    first_open,
                )
        # Check for bearish FVG (gap between third candle high and first candle low)
        elif third_high < first_low:
            fvg_size = first_low - third_high
            if fvg_size >= min_size_threshold:
                fvg_list[i] = (
                    "bearish",
                    third_high,
                    first_low,
                    time_candle1,
                    time_candle2,
                    time_candle3,
                    i,
                    middle_open,
                    middle_low,
                    middle_high,
                    first_open,
                )
    return fvg_list


def detect_fvg_by_size_ranges(data, size_thresholds=None):
    """
    Detect FVGs and categorize them by size ranges instead of body multipliers.
    This replaces the body multiplier approach with size-based categorization.

    Args:
        data: DataFrame with OHLC data  
        size_thresholds: List of minimum size thresholds to test
        
    Returns:
        dict: {size_threshold: [list of FVGs that meet this threshold]}
    """
    if size_thresholds is None:
        from config import get_min_fvg_sizes_range
        size_thresholds = get_min_fvg_sizes_range()

    print(f"[DEBUG] Detecting FVGs across {len(size_thresholds)} size thresholds: {size_thresholds}")

    # Detect all FVGs with smallest threshold first
    min_threshold = min(size_thresholds) if size_thresholds else min_fvg_size
    all_fvgs = detect_fvg(data, min_size_threshold=min_threshold)
    
    # Categorize FVGs by size thresholds
    result = {}
    for threshold in size_thresholds:
        fvg_list = [None] * len(data)
        
        for i, fvg in enumerate(all_fvgs):
            if fvg is not None:
                fvg_type, y0, y1 = fvg[0], fvg[1], fvg[2]
                fvg_size = abs(y1 - y0)
                
                # Include FVG if it meets this size threshold
                if fvg_size >= threshold:
                    fvg_list[i] = fvg
                    
        result[threshold] = fvg_list

    # Print summary statistics
    fvg_counts = {
        threshold: len([fvg for fvg in fvg_list if fvg is not None])
        for threshold, fvg_list in result.items()
    }
    total_fvgs = len([fvg for fvg in all_fvgs if fvg is not None])
    print(f"[DEBUG] Total FVGs detected: {total_fvgs}")
    print(f"[DEBUG] FVGs per size threshold: {fvg_counts}")

    return result


def detect_fvg_with_size_threshold(
    data: pd.DataFrame,
    min_size_threshold: float = 5.0,
) -> List[Optional[tuple]]:
    """
    FVG detection with custom size threshold (body multiplier logic removed).

    Args:
        data: DataFrame with OHLC data
        min_size_threshold: Minimum FVG size in points

    Returns:
        List of FVG tuples or None values
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")

    if len(data) < 3:
        return [None] * len(data)

    return detect_fvg(data, min_size_threshold=min_size_threshold)
