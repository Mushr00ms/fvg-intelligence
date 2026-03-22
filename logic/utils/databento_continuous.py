"""
Databento Continuous Contract Builder
Stitches individual futures contracts into continuous time series with roll handling.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from .databento_contract_mapper import DatabentoContractMapper
from .databento_loader import DatabentoLoader

logger = logging.getLogger(__name__)

class ContinuousContractBuilder:
    """Builds continuous contracts from individual futures contracts"""
    
    def __init__(self, data_directory: str = "databento_data", roll_days: int = 8):
        """
        Initialize the continuous contract builder.
        
        Args:
            data_directory: Path to Databento data
            roll_days: Days before expiration to roll to next contract
        """
        # Convert to absolute path if relative
        import os
        if not os.path.isabs(data_directory):
            # Get the logic directory path (parent of utils)
            logic_dir = Path(__file__).parent.parent
            data_directory = str(logic_dir / data_directory)
            
        self.roll_days = roll_days
        self.mapper = DatabentoContractMapper(data_directory)
        self.loader = DatabentoLoader()
        
    def build_continuous_series(self, symbol: str, start_date: datetime, 
                               end_date: datetime, adjustment_method: str = 'ratio') -> pd.DataFrame:
        """
        Build a continuous contract series from individual contracts.
        
        Args:
            symbol: Market symbol (ES or NQ)
            start_date: Start date for the series
            end_date: End date for the series
            adjustment_method: Method for adjusting price gaps ('ratio', 'difference', or 'none')
            
        Returns:
            DataFrame with continuous contract data
        """
        logger.info(f"Building continuous series for {symbol} from {start_date} to {end_date}")
        
        # Get contracts for the date range
        contracts = self.mapper.find_contracts_for_range(symbol, start_date, end_date, self.roll_days)
        
        if not contracts:
            logger.warning(f"No contracts found for {symbol} in specified date range")
            return pd.DataFrame()
        
        # Load and stitch contracts
        continuous_data = []
        adjustments = []
        previous_contract = None
        
        for i, contract_info in enumerate(contracts):
            logger.debug(f"Processing contract {contract_info['contract']} ({i+1}/{len(contracts)})")
            
            # Load contract data
            try:
                df = self.loader.read_dbn_file(
                    contract_info['file'],
                    contract_info['start'],
                    contract_info['end']
                )
                
                if df.empty:
                    logger.debug(f"No data loaded for contract {contract_info['contract']}")
                    continue
                
                # Add contract identifier
                df['contract'] = contract_info['contract']
                
                # Calculate adjustment factor if needed
                if previous_contract is not None and adjustment_method != 'none':
                    adjustment = self._calculate_adjustment(
                        previous_contract, df, adjustment_method
                    )
                    if adjustment is not None:
                        adjustments.append(adjustment)
                        df = self._apply_adjustments(df, adjustments, adjustment_method)
                
                continuous_data.append(df)
                previous_contract = df
                
            except Exception as e:
                logger.error(f"Failed to load contract {contract_info['contract']}: {e}")
                continue
        
        if not continuous_data:
            logger.warning("No data loaded for continuous contract")
            return pd.DataFrame()
        
        # Combine all contracts
        continuous_df = pd.concat(continuous_data, ignore_index=False)
        
        # Remove duplicates (keep first)
        continuous_df = continuous_df[~continuous_df.index.duplicated(keep='first')]
        
        # Sort by date
        continuous_df.sort_index(inplace=True)
        
        # Add metadata
        continuous_df.attrs['symbol'] = symbol
        continuous_df.attrs['roll_days'] = self.roll_days
        continuous_df.attrs['adjustment_method'] = adjustment_method
        continuous_df.attrs['contract_count'] = len(contracts)
        
        logger.info(f"Built continuous series with {len(continuous_df)} bars from {len(contracts)} contracts")
        
        return continuous_df
    
    def _calculate_adjustment(self, prev_df: pd.DataFrame, curr_df: pd.DataFrame,
                             method: str) -> Optional[float]:
        """
        Calculate adjustment factor between two contracts.
        
        Args:
            prev_df: Previous contract data
            curr_df: Current contract data
            method: Adjustment method ('ratio' or 'difference')
            
        Returns:
            Adjustment factor or None if no overlap
        """
        # Find overlap period
        overlap_start = max(prev_df.index.min(), curr_df.index.min())
        overlap_end = min(prev_df.index.max(), curr_df.index.max())
        
        if overlap_start > overlap_end:
            # No overlap
            logger.debug("No overlap between contracts for adjustment calculation")
            return None
        
        # Get overlapping data
        prev_overlap = prev_df.loc[overlap_start:overlap_end]
        curr_overlap = curr_df.loc[overlap_start:overlap_end]
        
        if prev_overlap.empty or curr_overlap.empty:
            return None
        
        # Calculate adjustment based on method
        if method == 'ratio':
            # Use ratio of closing prices
            prev_close = prev_overlap['close'].iloc[-1]
            curr_close = curr_overlap['close'].iloc[-1]
            if curr_close != 0:
                adjustment = prev_close / curr_close
            else:
                adjustment = 1.0
        elif method == 'difference':
            # Use difference of closing prices
            prev_close = prev_overlap['close'].iloc[-1]
            curr_close = curr_overlap['close'].iloc[-1]
            adjustment = prev_close - curr_close
        else:
            adjustment = None
        
        logger.debug(f"Calculated {method} adjustment: {adjustment}")
        return adjustment
    
    def _apply_adjustments(self, df: pd.DataFrame, adjustments: List[float],
                          method: str) -> pd.DataFrame:
        """
        Apply cumulative adjustments to contract data.
        
        Args:
            df: Contract data
            adjustments: List of adjustment factors
            method: Adjustment method
            
        Returns:
            Adjusted DataFrame
        """
        if not adjustments:
            return df
        
        df_adjusted = df.copy()
        
        if method == 'ratio':
            # Apply cumulative ratio adjustments
            cumulative_adjustment = np.prod(adjustments)
            for col in ['open', 'high', 'low', 'close', 'average']:
                if col in df_adjusted.columns:
                    df_adjusted[col] = df_adjusted[col] * cumulative_adjustment
        elif method == 'difference':
            # Apply cumulative difference adjustments
            cumulative_adjustment = sum(adjustments)
            for col in ['open', 'high', 'low', 'close', 'average']:
                if col in df_adjusted.columns:
                    df_adjusted[col] = df_adjusted[col] + cumulative_adjustment
        
        return df_adjusted
    
    def get_roll_dates(self, symbol: str, start_date: datetime,
                      end_date: datetime) -> List[Tuple[datetime, str, str]]:
        """
        Get roll dates and contract transitions for a date range.
        
        Args:
            symbol: Market symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List of tuples (roll_date, from_contract, to_contract)
        """
        contracts = self.mapper.find_contracts_for_range(symbol, start_date, end_date, self.roll_days)
        
        roll_dates = []
        for i in range(len(contracts) - 1):
            curr_contract = contracts[i]
            next_contract = contracts[i + 1]
            
            # Roll date is the end of current contract period
            roll_date = curr_contract['end']
            
            roll_dates.append((
                roll_date,
                curr_contract['contract'],
                next_contract['contract']
            ))
        
        return roll_dates
    
    def validate_continuous_series(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate the continuous series for issues.
        
        Args:
            df: Continuous contract DataFrame
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_bars': len(df),
            'date_range': (df.index.min(), df.index.max()) if not df.empty else (None, None),
            'contracts_used': df['contract'].nunique() if 'contract' in df.columns else 0,
            'gaps': [],
            'duplicates': 0,
            'negative_prices': False,
            'roll_points': []
        }
        
        if df.empty:
            return results
        
        # Check for duplicates
        results['duplicates'] = df.index.duplicated().sum()
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                if (df[col] < 0).any():
                    results['negative_prices'] = True
                    break
        
        # Find gaps in time series
        if len(df) > 1:
            time_diff = df.index.to_series().diff()
            expected_diff = pd.Timedelta(minutes=1)
            
            # Gaps larger than 2 hours (accounting for market closures)
            large_gaps = time_diff[time_diff > pd.Timedelta(hours=2)]
            results['gaps'] = [(idx, gap) for idx, gap in large_gaps.items()]
        
        # Find roll points (contract changes)
        if 'contract' in df.columns:
            contract_changes = df['contract'].ne(df['contract'].shift())
            roll_indices = df.index[contract_changes].tolist()
            if roll_indices and roll_indices[0] == df.index[0]:
                roll_indices = roll_indices[1:]  # Remove first index
            results['roll_points'] = roll_indices
        
        return results
    
    def export_continuous_series(self, df: pd.DataFrame, output_path: str,
                                format: str = 'parquet'):
        """
        Export continuous series to file.
        
        Args:
            df: Continuous contract DataFrame
            output_path: Output file path
            format: Export format ('parquet', 'csv', 'hdf')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'parquet':
            df.to_parquet(output_path, compression='snappy')
        elif format == 'csv':
            df.to_csv(output_path)
        elif format == 'hdf':
            df.to_hdf(output_path, key='continuous', mode='w', complevel=9)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported continuous series to {output_path}")