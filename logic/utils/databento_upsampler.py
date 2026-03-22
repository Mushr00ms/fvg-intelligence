"""
Databento Timeframe Upsampler
Generates higher timeframe data from 1-minute base data with memory-efficient streaming.
Uses proper Databento resampling conventions.
"""

import logging
from typing import Dict, Generator, Optional, List
import pandas as pd
import numpy as np
from datetime import timedelta
from .databento_proper_resampler import DatabentoResampler

logger = logging.getLogger(__name__)

class TimeframeUpsampler:
    """Upsamples 1-minute data to higher timeframes"""
    
    # Supported timeframes and their pandas resample rules
    TIMEFRAME_RULES = {
        '1min': None,  # Base timeframe, no resampling needed
        '3min': '3min',
        '5min': '5min',
        '15min': '15min',
        '30min': '30min',
        '60min': '60min',
        '1hour': '60min',
        '4hour': '4h',
        '1day': '1D',
        'daily': '1D'
    }
    
    def __init__(self, chunk_size: int = 100000):
        """
        Initialize the upsampler.
        
        Args:
            chunk_size: Number of 1-minute bars to process at once
        """
        self.chunk_size = chunk_size
    
    def upsample(self, data: pd.DataFrame, target_timeframe: str,
                 method: str = 'standard') -> pd.DataFrame:
        """
        Upsample 1-minute data to a higher timeframe.
        
        Args:
            data: DataFrame with 1-minute OHLCV data
            target_timeframe: Target timeframe (e.g., '5min', '1hour')
            method: Aggregation method ('standard' or 'volume_weighted')
            
        Returns:
            Upsampled DataFrame
        """
        if target_timeframe not in self.TIMEFRAME_RULES:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        
        rule = self.TIMEFRAME_RULES[target_timeframe]
        
        # No resampling needed for 1min
        if rule is None:
            return data.copy()
        
        logger.info(f"Upsampling {len(data)} bars from 1min to {target_timeframe}")
        
        # Clean any infinite values first
        data_clean = DatabentoResampler.clean_infinite_values(data)
        
        # Use proper Databento resampling
        return DatabentoResampler.resample_ohlcv(
            data_clean,
            rule,
            interpolate_first=False  # Don't interpolate for upsampling
        )
    
    def _standard_upsample(self, data: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Standard OHLCV aggregation.
        
        Args:
            data: 1-minute data
            rule: Pandas resample rule
            
        Returns:
            Upsampled DataFrame
        """
        # Define aggregation rules for OHLCV
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Add barCount if present
        if 'barCount' in data.columns:
            agg_rules['barCount'] = 'sum'
        
        # Resample
        resampled = data.resample(rule).agg(agg_rules)
        
        # Calculate average price (VWAP style if volume available)
        if 'volume' in resampled.columns:
            # Calculate typical price for each bar
            typical_price = (resampled['high'] + resampled['low'] + resampled['close']) / 3
            resampled['average'] = typical_price
        else:
            resampled['average'] = (resampled['open'] + resampled['high'] + 
                                   resampled['low'] + resampled['close']) / 4
        
        # Remove bars with no data (e.g., market closed periods)
        resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Preserve contract information if present
        if 'contract' in data.columns:
            # Use the most common contract in each period
            contract_mode = data.resample(rule)['contract'].agg(lambda x: x.mode()[0] if len(x) > 0 else None)
            resampled['contract'] = contract_mode
        
        return resampled
    
    def _volume_weighted_upsample(self, data: pd.DataFrame, rule: str) -> pd.DataFrame:
        """
        Volume-weighted aggregation for more accurate pricing.
        
        Args:
            data: 1-minute data with volume
            rule: Pandas resample rule
            
        Returns:
            Upsampled DataFrame with volume-weighted prices
        """
        if 'volume' not in data.columns:
            logger.warning("Volume not available, falling back to standard upsampling")
            return self._standard_upsample(data, rule)
        
        # Create a copy for calculations
        df = data.copy()
        
        # Calculate dollar volume for VWAP
        df['dollar_volume'] = df['close'] * df['volume']
        
        # Resample with custom aggregation
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'dollar_volume': 'sum'
        })
        
        # Calculate VWAP
        resampled['average'] = resampled['dollar_volume'] / resampled['volume']
        resampled['average'] = resampled['average'].fillna(resampled['close'])
        
        # Drop helper column
        resampled = resampled.drop('dollar_volume', axis=1)
        
        # Add barCount if needed
        if 'barCount' in data.columns:
            resampled['barCount'] = df.resample(rule)['barCount'].sum()
        
        # Remove empty bars
        resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])
        
        return resampled
    
    def stream_upsample(self, data_generator: Generator[pd.DataFrame, None, None],
                       target_timeframe: str) -> Generator[pd.DataFrame, None, None]:
        """
        Stream upsampling for memory-efficient processing of large datasets.
        
        Args:
            data_generator: Generator yielding 1-minute data chunks
            target_timeframe: Target timeframe
            
        Yields:
            Upsampled data chunks
        """
        if target_timeframe not in self.TIMEFRAME_RULES:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        
        rule = self.TIMEFRAME_RULES[target_timeframe]
        if rule is None:
            # No upsampling needed
            yield from data_generator
            return
        
        # Buffer to handle incomplete periods at chunk boundaries
        buffer = pd.DataFrame()
        
        for chunk in data_generator:
            # Combine with buffer
            if not buffer.empty:
                combined = pd.concat([buffer, chunk])
            else:
                combined = chunk
            
            # Determine complete periods
            period_freq = self._get_period_frequency(rule)
            last_complete_period = combined.index[-1].floor(period_freq)
            
            # Split into complete and incomplete data
            complete_data = combined[combined.index <= last_complete_period]
            buffer = combined[combined.index > last_complete_period]
            
            if not complete_data.empty:
                # Upsample complete data
                upsampled = self._standard_upsample(complete_data, rule)
                if not upsampled.empty:
                    yield upsampled
        
        # Process remaining buffer
        if not buffer.empty:
            upsampled = self._standard_upsample(buffer, rule)
            if not upsampled.empty:
                yield upsampled
    
    def _get_period_frequency(self, rule: str) -> str:
        """Convert resample rule to period frequency"""
        freq_map = {
            '5T': '5T',
            '15T': '15T',
            '30T': '30T',
            '60T': 'H',
            '4H': '4H',
            '1D': 'D'
        }
        return freq_map.get(rule, rule)
    
    def generate_all_timeframes(self, data: pd.DataFrame,
                               timeframes: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate multiple timeframes from 1-minute data.
        
        Args:
            data: 1-minute base data
            timeframes: List of target timeframes (None for all)
            
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        if timeframes is None:
            timeframes = ['5min', '15min', '60min', '4hour', '1day']
        
        results = {'1min': data.copy()}
        
        for timeframe in timeframes:
            if timeframe != '1min':
                try:
                    results[timeframe] = self.upsample(data, timeframe)
                    logger.info(f"Generated {timeframe}: {len(results[timeframe])} bars")
                except Exception as e:
                    logger.error(f"Failed to generate {timeframe}: {e}")
        
        return results
    
    def validate_upsampled_data(self, original: pd.DataFrame,
                               upsampled: pd.DataFrame, timeframe: str) -> Dict[str, any]:
        """
        Validate upsampled data for consistency.
        
        Args:
            original: Original 1-minute data
            upsampled: Upsampled data
            timeframe: Target timeframe
            
        Returns:
            Validation results
        """
        results = {
            'original_bars': len(original),
            'upsampled_bars': len(upsampled),
            'timeframe': timeframe,
            'issues': []
        }
        
        if original.empty or upsampled.empty:
            results['issues'].append("Empty data")
            return results
        
        # Check date range consistency
        orig_start, orig_end = original.index.min(), original.index.max()
        up_start, up_end = upsampled.index.min(), upsampled.index.max()
        
        if up_start > orig_start + timedelta(hours=1):
            results['issues'].append(f"Start date mismatch: {orig_start} vs {up_start}")
        
        if up_end < orig_end - timedelta(hours=1):
            results['issues'].append(f"End date mismatch: {orig_end} vs {up_end}")
        
        # Check volume consistency
        if 'volume' in original.columns and 'volume' in upsampled.columns:
            orig_total_volume = original['volume'].sum()
            up_total_volume = upsampled['volume'].sum()
            volume_diff = abs(orig_total_volume - up_total_volume)
            
            if volume_diff > orig_total_volume * 0.01:  # 1% tolerance
                results['issues'].append(f"Volume mismatch: {orig_total_volume} vs {up_total_volume}")
        
        # Check price relationships
        if not upsampled.empty:
            # High should be highest, low should be lowest
            invalid_highs = (upsampled['high'] < upsampled[['open', 'close']].max(axis=1)).sum()
            invalid_lows = (upsampled['low'] > upsampled[['open', 'close']].min(axis=1)).sum()
            
            if invalid_highs > 0:
                results['issues'].append(f"Invalid high prices: {invalid_highs} bars")
            
            if invalid_lows > 0:
                results['issues'].append(f"Invalid low prices: {invalid_lows} bars")
        
        # Expected bar count (approximate)
        rule = self.TIMEFRAME_RULES.get(timeframe)
        if rule:
            if rule == '5min':
                expected_ratio = 5
            elif rule == '15min':
                expected_ratio = 15
            elif rule == '30min':
                expected_ratio = 30
            elif rule == '60min':
                expected_ratio = 60
            elif rule == '4h':
                expected_ratio = 240
            elif rule == '1D':
                expected_ratio = 1440
            else:
                expected_ratio = None
            
            if expected_ratio:
                expected_bars = len(original) / expected_ratio
                actual_bars = len(upsampled)
                
                # Allow 20% deviation for market hours
                if abs(actual_bars - expected_bars) > expected_bars * 0.2:
                    results['issues'].append(
                        f"Unexpected bar count: expected ~{expected_bars:.0f}, got {actual_bars}"
                    )
        
        results['valid'] = len(results['issues']) == 0
        return results