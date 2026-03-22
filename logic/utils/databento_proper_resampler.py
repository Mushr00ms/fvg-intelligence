"""
Databento Proper Resampler
Correctly handles OHLCV resampling following Databento conventions.
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DatabentoResampler:
    """Properly resample Databento OHLCV data"""
    
    @staticmethod
    def interpolate_ohlcv(
        df: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        interp_interval: str = "1min",
    ) -> pd.DataFrame:
        """
        Interpolate OHLCV records between start and end.
        
        Databento only sends an OHLCV record if a trade happens in that interval,
        so we need to fill missing intervals.
        
        Args:
            df: DataFrame with OHLCV data
            start: Start timestamp
            end: End timestamp  
            interp_interval: Interpolation interval (default "1min")
            
        Returns:
            DataFrame with interpolated records
        """
        if df.empty:
            return df
            
        def _interpolate_group(group):
            """Interpolate OHLCV records for each group"""
            # Reindex with a complete index using specified start/end times
            group = group.reindex(
                pd.date_range(
                    start=start,
                    end=end,
                    freq=interp_interval,
                    inclusive="left",
                ).rename(group.index.name),
            )
            
            # Forward fill close prices (may remain NaN if no prior data exists)
            group["close"] = group["close"].ffill()
            
            # For intervals with no trades, set open/high/low equal to the close and volume to 0
            group = group.fillna({
                **{col: group["close"] for col in ["open", "high", "low"]},
                "volume": 0,
            })
            
            # Ensure volume is integer
            group["volume"] = group["volume"].astype(int)
            
            # Drop unnecessary columns if they exist
            group = group.drop(columns=["rtype", "instrument_id", "publisher_id"], errors="ignore")
            
            return group
        
        # Check if we have symbol column for grouping
        if "symbol" in df.columns:
            # Group by symbol if multiple contracts
            df_interpolated = (
                df.groupby("symbol")
                .apply(_interpolate_group, include_groups=False)
                .reset_index("symbol")
                .sort_index()
            )
        else:
            # Single contract, no grouping needed
            df_interpolated = _interpolate_group(df)
        
        return df_interpolated
    
    @staticmethod
    def resample_ohlcv(
        df: pd.DataFrame,
        resample_interval: str,
        interpolate_first: bool = False,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Resample OHLCV bars to the specified interval.
        
        Args:
            df: DataFrame with OHLCV data
            resample_interval: Target interval (e.g., "5min", "15min", "1h")
            interpolate_first: Whether to interpolate missing 1-minute bars first
            start: Start timestamp for interpolation
            end: End timestamp for interpolation
            
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
        
        # Optionally interpolate first to ensure consistent data
        if interpolate_first:
            if start is None:
                start = df.index.min()
            if end is None:
                end = df.index.max()
            df = DatabentoResampler.interpolate_ohlcv(df, start, end, "1min")
        
        # Define aggregation rules for OHLCV
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        
        # Add average if present
        if "average" in df.columns:
            agg_dict["average"] = "mean"
        
        # Check if we have symbol column for grouping
        if "symbol" in df.columns:
            # Group by symbol if multiple contracts
            resampled_df = (
                df.groupby("symbol")
                .resample(resample_interval)
                .agg(agg_dict)
                .reset_index("symbol")
                .sort_index()
            )
        else:
            # Single contract, no grouping needed
            resampled_df = df.resample(resample_interval).agg(agg_dict)
        
        # Remove rows where all prices are NaN (no data in that period)
        price_cols = ["open", "high", "low", "close"]
        resampled_df = resampled_df.dropna(subset=price_cols, how="all")
        
        # Ensure volume is integer
        if "volume" in resampled_df.columns:
            resampled_df["volume"] = resampled_df["volume"].fillna(0).astype(int)
        
        return resampled_df
    
    @staticmethod
    def validate_resampled_data(
        original_df: pd.DataFrame,
        resampled_df: pd.DataFrame,
    ) -> dict:
        """
        Validate that resampled data maintains integrity.
        
        Args:
            original_df: Original DataFrame
            resampled_df: Resampled DataFrame
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "issues": [],
            "original_bars": len(original_df),
            "resampled_bars": len(resampled_df),
        }
        
        if original_df.empty or resampled_df.empty:
            results["valid"] = False
            results["issues"].append("Empty DataFrame")
            return results
        
        # Check date range consistency
        orig_start, orig_end = original_df.index.min(), original_df.index.max()
        resamp_start, resamp_end = resampled_df.index.min(), resampled_df.index.max()
        
        # Allow small differences due to rounding to period boundaries
        if resamp_start > orig_start + pd.Timedelta(hours=1):
            results["issues"].append(f"Start date mismatch: {orig_start} vs {resamp_start}")
        
        if resamp_end < orig_end - pd.Timedelta(hours=1):
            results["issues"].append(f"End date mismatch: {orig_end} vs {resamp_end}")
        
        # Check volume consistency (should be equal or very close)
        if "volume" in original_df.columns and "volume" in resampled_df.columns:
            orig_total_volume = original_df["volume"].sum()
            resamp_total_volume = resampled_df["volume"].sum()
            
            if orig_total_volume > 0:
                volume_diff_pct = abs(orig_total_volume - resamp_total_volume) / orig_total_volume
                if volume_diff_pct > 0.01:  # 1% tolerance
                    results["issues"].append(
                        f"Volume mismatch: {orig_total_volume} vs {resamp_total_volume} "
                        f"({volume_diff_pct:.2%} difference)"
                    )
        
        # Check price relationships (high >= low, etc.)
        invalid_highs = (resampled_df["high"] < resampled_df[["open", "close"]].max(axis=1)).sum()
        invalid_lows = (resampled_df["low"] > resampled_df[["open", "close"]].min(axis=1)).sum()
        
        if invalid_highs > 0:
            results["issues"].append(f"Invalid high prices: {invalid_highs} bars")
        
        if invalid_lows > 0:
            results["issues"].append(f"Invalid low prices: {invalid_lows} bars")
        
        # Check for NaN values in critical columns
        critical_cols = ["open", "high", "low", "close"]
        for col in critical_cols:
            if col in resampled_df.columns:
                nan_count = resampled_df[col].isna().sum()
                if nan_count > 0:
                    results["issues"].append(f"NaN values in {col}: {nan_count} bars")
        
        results["valid"] = len(results["issues"]) == 0
        return results
    
    @staticmethod
    def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean infinite values from DataFrame.
        
        Args:
            df: DataFrame potentially containing infinite values
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # Identify numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # Replace infinite values with NaN
        for col in numeric_cols:
            df_clean.loc[np.isinf(df_clean[col]), col] = np.nan
        
        # Forward fill NaN values in price columns
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].ffill()
        
        # Drop rows where all price columns are still NaN
        df_clean = df_clean.dropna(subset=price_cols, how="all")
        
        logger.info(f"Cleaned {len(df) - len(df_clean)} rows with infinite/invalid values")
        
        return df_clean