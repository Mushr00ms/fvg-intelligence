"""
Databento Data Loader
Efficiently loads and processes Databento DBN files with memory-optimized streaming.
"""

import logging
from pathlib import Path
from typing import Generator, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import databento as db

logger = logging.getLogger(__name__)

class DatabentoLoader:
    """Loads Databento DBN files and converts to pandas DataFrames"""
    
    # DBN format constants
    DBN_VERSION = 1
    OHLCV_SCHEMA_SIZE = 64  # Size of OHLCV-1m record in bytes
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize the loader.
        
        Args:
            chunk_size: Number of records to process at once
        """
        self.chunk_size = chunk_size
        
    def read_dbn_file(self, file_path: Path, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Read a DBN file and convert to DataFrame.
        
        Args:
            file_path: Path to the .dbn.zst file
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with OHLCV data
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.debug(f"Loading DBN file: {file_path.name}")
        
        try:
            # Use databento's native DBN reader
            dbn_store = db.DBNStore.from_file(str(file_path))
            
            # Convert to DataFrame
            df = dbn_store.to_df()

            if df.empty:
                logger.warning(f"No data in {file_path.name}")
                return pd.DataFrame()

            # Filter by symbol if file contains multiple contracts
            # Extract expected contract from filename: ...ohlcv-1m.CONTRACT.dbn.zst
            if 'symbol' in df.columns:
                import re
                fname_match = re.search(r'\.ohlcv-1m\.([A-Z0-9\-]+)\.dbn\.zst$', file_path.name)
                if fname_match:
                    expected_symbol = fname_match.group(1)
                    if expected_symbol in df['symbol'].values:
                        df = df[df['symbol'] == expected_symbol]
                        logger.debug(f"Filtered to symbol {expected_symbol}: {len(df)} bars")

            # Rename columns to match expected format
            column_map = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }

            # Select and rename columns
            available_cols = [col for col in column_map.keys() if col in df.columns]
            df = df[available_cols].rename(columns=column_map)
            
            # Add average and barCount columns
            df['average'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            df['barCount'] = 1
            
            # Ensure index is named 'date' for compatibility
            df.index.name = 'date'
            
            # Convert index to NY timezone
            df.index = df.index.tz_convert('America/New_York')
            
            # Apply date filters if specified
            if start_date:
                start_pd = pd.Timestamp(start_date).tz_localize('America/New_York') if not hasattr(start_date, 'tzinfo') or not start_date.tzinfo else pd.Timestamp(start_date).tz_convert('America/New_York')
                df = df[df.index >= start_pd]
            
            if end_date:
                end_pd = pd.Timestamp(end_date).tz_localize('America/New_York') if not hasattr(end_date, 'tzinfo') or not end_date.tzinfo else pd.Timestamp(end_date).tz_convert('America/New_York')
                df = df[df.index <= end_pd]
            
            logger.debug(f"Loaded {len(df)} records from {file_path.name}")
            return df
            
        except (MemoryError, SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return pd.DataFrame()
    
    def stream_dbn_file(self, file_path: Path,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Stream a DBN file in chunks for memory efficiency.
        For now, this just loads the entire file and yields it.
        Future optimization: implement true streaming with databento.
        
        Args:
            file_path: Path to the .dbn.zst file
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Yields:
            DataFrames with OHLCV data chunks
        """
        df = self.read_dbn_file(file_path, start_date, end_date)
        if not df.empty:
            # Yield in chunks
            for i in range(0, len(df), self.chunk_size):
                yield df.iloc[i:i+self.chunk_size]
    
    
    def load_multiple_files(self, file_paths: list,
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          deduplicate: bool = True) -> pd.DataFrame:
        """
        Load multiple DBN files and combine them.
        
        Args:
            file_paths: List of paths to DBN files
            start_date: Optional start date filter
            end_date: Optional end date filter
            deduplicate: Whether to remove duplicate timestamps
            
        Returns:
            Combined DataFrame
        """
        all_data = []
        
        for file_path in file_paths:
            try:
                df = self.read_dbn_file(Path(file_path), start_date, end_date)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=False)
        
        # Sort by timestamp
        combined_df.sort_index(inplace=True)
        
        # Remove duplicates if requested
        if deduplicate:
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        return combined_df
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data for quality issues.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_records': len(df),
            'has_nulls': df.isnull().any().any(),
            'null_counts': df.isnull().sum().to_dict(),
            'negative_prices': (df[['open', 'high', 'low', 'close']] < 0).any().any(),
            'zero_volume_bars': (df['volume'] == 0).sum(),
            'date_range': (df.index.min(), df.index.max()) if not df.empty else (None, None),
            'gaps_detected': False
        }
        
        # Check for time gaps
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            expected_diff = pd.Timedelta(minutes=1)
            gaps = time_diffs[time_diffs > expected_diff * 2]
            results['gaps_detected'] = len(gaps) > 0
            results['gap_count'] = len(gaps)
            if len(gaps) > 0:
                results['largest_gap'] = gaps.max()
        
        return results