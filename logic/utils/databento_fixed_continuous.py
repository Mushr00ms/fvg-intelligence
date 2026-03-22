"""
Fixed Databento Continuous Contract Builder
Properly handles Databento's continuous contract data format.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np

from .databento_loader import DatabentoLoader
from .databento_proper_resampler import DatabentoResampler

logger = logging.getLogger(__name__)

class FixedContinuousContractBuilder:
    """
    Properly builds continuous contracts from Databento data.
    
    Key insight: Databento files already contain continuous historical data
    for each contract symbol. We don't need to "roll" them - just load
    the appropriate file for the date range requested.
    """
    
    def __init__(self, data_directory: str = "databento_data"):
        """
        Initialize the continuous contract builder.
        
        Args:
            data_directory: Path to Databento data
        """
        # Convert to absolute path if relative
        import os
        if not os.path.isabs(data_directory):
            # Get the logic directory path (parent of utils)
            logic_dir = Path(__file__).parent.parent
            data_directory = str(logic_dir / data_directory)
            
        self.data_dir = Path(data_directory)
        self.loader = DatabentoLoader()
        
    def load_continuous_data(self, symbol: str, start_date: datetime, 
                            end_date: datetime) -> pd.DataFrame:
        """
        Load continuous contract data for a symbol and date range.
        
        For Databento, each file already contains continuous historical data.
        We just need to find the right file(s) and load the requested date range.
        
        Args:
            symbol: Market symbol (ES or NQ)
            start_date: Start date for the series
            end_date: End date for the series
            
        Returns:
            DataFrame with continuous contract data
        """
        logger.info(f"Loading continuous data for {symbol} from {start_date} to {end_date}")
        
        all_data = []
        
        # Find all relevant files for this symbol
        pattern = f"*.ohlcv-1m.{symbol}*.dbn.zst"
        files = list(self.data_dir.glob(pattern))
        
        # Filter out spread files (contain hyphen in contract spec)
        single_contract_files = []
        for file in files:
            # Extract contract part from filename
            parts = file.name.split('.')
            if len(parts) > 2:
                contract_spec = parts[2]
                # Skip spread files (e.g., NQM0-NQH1)
                if '-' not in contract_spec:
                    single_contract_files.append(file)
        
        logger.info(f"Found {len(single_contract_files)} single contract files for {symbol}")
        
        # Load data from each file that overlaps our date range
        for file_path in single_contract_files:
            try:
                # Parse date range from filename
                parts = file_path.name.split('-')
                if len(parts) >= 3:
                    file_start_str = parts[2]  # YYYYMMDD
                    file_end_str = parts[3].split('.')[0]  # YYYYMMDD
                    
                    file_start = datetime.strptime(file_start_str, '%Y%m%d')
                    file_end = datetime.strptime(file_end_str, '%Y%m%d')
                    
                    # Make sure dates are timezone-naive for comparison
                    compare_start = start_date.replace(tzinfo=None) if hasattr(start_date, 'tzinfo') else start_date
                    compare_end = end_date.replace(tzinfo=None) if hasattr(end_date, 'tzinfo') else end_date
                    
                    # Check if this file overlaps our requested range
                    if file_end >= compare_start and file_start <= compare_end:
                        logger.debug(f"Loading data from {file_path.name}")
                        
                        # Load data from this file
                        df = self.loader.read_dbn_file(file_path, start_date, end_date)
                        
                        if not df.empty:
                            # Add contract identifier from filename
                            contract_spec = file_path.name.split('.')[2]
                            df['contract'] = contract_spec
                            
                            all_data.append(df)
                        
            except (MemoryError, SystemExit, KeyboardInterrupt):
                raise
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            logger.warning(f"No data found for {symbol} in specified date range")
            return pd.DataFrame()
        
        # Combine all data — resolve overlapping contracts by keeping only
        # the front-month within its active window (before roll_days of expiry).
        if len(all_data) > 1 and 'contract' in all_data[0].columns:
            import re
            month_order = {'F':1,'G':2,'H':3,'J':4,'K':5,'M':6,'N':7,'Q':8,'U':9,'V':10,'X':11,'Z':12}


            def _contract_expiry(c, data_chunk=None):
                """Estimate expiry date from contract code like NQU5.
                Disambiguates decade using actual data timestamps."""
                m = re.search(r'([A-Z])(\d)', str(c))
                if not m:
                    return pd.Timestamp.max.tz_localize('America/New_York')
                month_code = m.group(1)
                year_digit = int(m.group(2))
                exp_month = month_order.get(month_code, 12)
                # Disambiguate decade from actual data timestamps
                if data_chunk is not None and not data_chunk.empty:
                    data_year = data_chunk.index[len(data_chunk)//2].year
                    decade_base = (data_year // 10) * 10
                else:
                    decade_base = 2020
                exp_year = decade_base + year_digit
                # If expiry is more than 5 years before data, bump decade
                if data_chunk is not None and not data_chunk.empty:
                    if exp_year < data_year - 5:
                        exp_year += 10
                # Third Friday approximation
                day = 15 + (4 - pd.Timestamp(exp_year, exp_month, 15).weekday()) % 7
                return pd.Timestamp(exp_year, exp_month, day, tz='America/New_York')

            # Sort contract DataFrames by expiry (front month first)
            all_data.sort(key=lambda d: _contract_expiry(d['contract'].iloc[0], d))

            # Assign each contract its active window: from previous contract's
            # roll date to this contract's roll date
            roll_days = 8
            filtered = []
            prev_roll = pd.Timestamp.min.tz_localize('America/New_York')
            for i, chunk in enumerate(all_data):
                contract = chunk['contract'].iloc[0]
                expiry = _contract_expiry(contract)
                my_roll = expiry - pd.Timedelta(days=roll_days)
                # This contract is active from prev_roll to my_roll
                active = chunk[(chunk.index >= prev_roll) & (chunk.index < my_roll)]
                if not active.empty:
                    filtered.append(active)
                    logger.debug(f"Contract {contract}: {len(active)} bars ({prev_roll.strftime('%Y-%m-%d')} to {my_roll.strftime('%Y-%m-%d')})")
                prev_roll = my_roll

            # Last contract: use all remaining data after last roll
            if all_data:
                last = all_data[-1]
                remaining = last[last.index >= prev_roll]
                if not remaining.empty:
                    filtered.append(remaining)
                    logger.debug(f"Contract {last['contract'].iloc[0]} (tail): {len(remaining)} bars")

            combined_df = pd.concat(filtered, ignore_index=False) if filtered else pd.DataFrame()
        else:
            combined_df = pd.concat(all_data, ignore_index=False)

        # Remove any remaining duplicates and sort
        if not combined_df.empty:
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df.sort_index(inplace=True)
        
        # Clean any infinite values
        combined_df = DatabentoResampler.clean_infinite_values(combined_df)
        
        # Add metadata - convert datetimes to strings for JSON serialization
        combined_df.attrs['symbol'] = symbol
        combined_df.attrs['start_date'] = pd.Timestamp(start_date).isoformat()
        combined_df.attrs['end_date'] = pd.Timestamp(end_date).isoformat()
        combined_df.attrs['num_files'] = len(all_data)
        
        logger.info(f"Loaded {len(combined_df)} bars from {len(all_data)} files")
        
        return combined_df
    
    def validate_continuous_series(self, df: pd.DataFrame) -> dict:
        """
        Validate the continuous series for issues.
        
        Args:
            df: Continuous series DataFrame
            
        Returns:
            Validation results
        """
        results = {
            'total_bars': len(df),
            'duplicates': df.index.duplicated().sum(),
            'gaps': [],
            'extreme_jumps': [],
            'infinite_values': 0,
            'negative_prices': False
        }
        
        if df.empty:
            return results
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            results['infinite_values'] += np.isinf(df[col]).sum()
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                if (df[col] < 0).any():
                    results['negative_prices'] = True
                    break
        
        # Check for large gaps in time
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            
            # Gaps larger than 1 day (accounting for weekends)
            large_gaps = time_diffs[time_diffs > pd.Timedelta(days=3)]
            for idx, gap in large_gaps.items():
                results['gaps'].append({
                    'at': idx,
                    'gap_days': gap.days
                })
        
        # Check for extreme price jumps (>10% in one bar)
        if 'close' in df.columns and len(df) > 1:
            returns = df['close'].pct_change()
            extreme_returns = returns[abs(returns) > 0.10]
            
            for idx, ret in extreme_returns.items():
                results['extreme_jumps'].append({
                    'at': idx,
                    'return': ret,
                    'from_contract': df.loc[idx, 'contract'] if 'contract' in df.columns else 'unknown'
                })
        
        return results