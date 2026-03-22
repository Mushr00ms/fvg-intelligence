"""
Databento Main Data Interface
Primary entry point for all Databento data requests with caching and optimization.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
from tqdm import tqdm

from .databento_contract_mapper import DatabentoContractMapper
from .databento_loader import DatabentoLoader
from .databento_cache_manager import DatabentoCache, CacheTier
from .databento_fixed_continuous import FixedContinuousContractBuilder
from .databento_upsampler import TimeframeUpsampler

logger = logging.getLogger(__name__)

class DatabentoInterface:
    """Main interface for accessing Databento historical data"""
    
    def __init__(self, data_directory: str = "databento_data",
                 cache_directory: str = "databento_cache",
                 enable_cache: bool = True,
                 memory_limit_mb: int = 4096):
        """
        Initialize the Databento interface.
        
        Args:
            data_directory: Path to Databento data files
            cache_directory: Path to cache storage
            enable_cache: Whether to use caching
            memory_limit_mb: Memory limit in MB
        """
        # Convert to absolute paths if relative
        import os
        if not os.path.isabs(data_directory):
            # Get the logic directory path (parent of utils)
            logic_dir = Path(__file__).parent.parent
            data_directory = str(logic_dir / data_directory)
        if not os.path.isabs(cache_directory):
            # Get the logic directory path (parent of utils)
            logic_dir = Path(__file__).parent.parent
            cache_directory = str(logic_dir / cache_directory)
            
        self.data_dir = Path(data_directory)
        self.enable_cache = enable_cache
        self.memory_limit_mb = memory_limit_mb
        
        # Initialize components
        self.mapper = DatabentoContractMapper(data_directory)
        self.loader = DatabentoLoader()
        self.cache = DatabentoCache(cache_directory) if enable_cache else None
        self.continuous_builder = FixedContinuousContractBuilder(data_directory)
        self.upsampler = TimeframeUpsampler()
        
        # Statistics
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'data_loaded_mb': 0,
            'processing_time_seconds': 0
        }
    
    def _get_data_end_date(self):
        """Find the latest date available in the raw databento data files."""
        import re
        latest = None
        for f in self.data_dir.glob("*.dbn.zst"):
            m = re.search(r'-(\d{8})\.ohlcv', f.name)
            if m:
                d = pd.to_datetime(m.group(1)).tz_localize('America/New_York')
                if latest is None or d > latest:
                    latest = d
        if latest:
            logger.info(f"Actual data end date: {latest.strftime('%Y-%m-%d')}")
        return latest

    def fetch_market_data(self, symbol: str, timeframe: str,
                         expiration_dates: Optional[List[datetime]] = None,
                         data_cache_dir: Optional[str] = None,  # Ignored, for compatibility
                         period: str = "2 years",
                         custom_start: Optional[str] = None,
                         custom_end: Optional[str] = None,
                         roll_days: int = 8,
                         min_search_bars: int = 100,
                         debug: bool = True) -> pd.DataFrame:
        """
        Main entry point for fetching market data.
        
        Args:
            symbol: Market symbol (e.g., "NQ", "ES")
            timeframe: Timeframe string (e.g., "5min", "1min")
            expiration_dates: List of contract expiration dates (optional, not used)
            data_cache_dir: Cache directory (ignored, uses internal cache)
            period: Time period for analysis
            custom_start: Custom start date string
            custom_end: Custom end date string
            roll_days: Days before expiration to roll contracts
            min_search_bars: Minimum bars to search (not used)
            debug: Debug output flag
            
        Returns:
            DataFrame: Combined market data
        """
        start_time = time.time()
        self.stats['requests'] += 1
        
        # Find the actual latest date available in raw data
        actual_end = self._get_data_end_date()

        # Parse dates
        if custom_start and custom_end:
            start_date = pd.to_datetime(custom_start).tz_localize('America/New_York')
            end_date = pd.to_datetime(custom_end).tz_localize('America/New_York')
            if actual_end and end_date > actual_end:
                end_date = actual_end
        else:
            end_date = actual_end or pd.Timestamp.now(tz='America/New_York')
            if "years" in period:
                years = int(period.split()[0])
                start_date = end_date - pd.DateOffset(years=years)
            elif "months" in period:
                months = int(period.split()[0])
                start_date = end_date - pd.DateOffset(months=months)
            elif "days" in period:
                days = int(period.split()[0])
                start_date = end_date - pd.DateOffset(days=days)
            else:
                start_date = end_date - pd.DateOffset(years=2)
        
        if debug:
            logger.info(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}")
        
        # Convert timeframe format
        timeframe_map = {
            '1min': '1min',
            '3min': '3min',
            '5min': '5min',
            '15min': '15min',
            '60min': '60min',
            '1hour': '60min',
            '4hour': '4hour',
            '1day': '1day'
        }
        
        target_timeframe = timeframe_map.get(timeframe, '1min')
        
        # Check cache first
        if self.enable_cache and self.cache:
            cached_data = self.cache.get(symbol, target_timeframe, 
                                        start_date.to_pydatetime(), 
                                        end_date.to_pydatetime())
            if cached_data is not None:
                self.stats['cache_hits'] += 1
                if debug:
                    logger.info(f"Cache hit! Loaded {len(cached_data)} bars from cache")
                return cached_data
        
        # Load continuous contract data from 1-minute data
        continuous_data = self.continuous_builder.load_continuous_data(
            symbol, start_date.to_pydatetime(), end_date.to_pydatetime()
        )
        
        if continuous_data.empty:
            if debug:
                logger.warning(f"No data found for {symbol} in specified period")
            return pd.DataFrame()
        
        # Upsample if needed
        if target_timeframe != '1min':
            result_data = self.upsampler.upsample(continuous_data, target_timeframe)
        else:
            result_data = continuous_data
        
        # Cache the result
        if self.enable_cache and self.cache and not result_data.empty:
            self.cache.put(symbol, target_timeframe,
                          start_date.to_pydatetime(), end_date.to_pydatetime(),
                          result_data)
        
        # Update statistics
        self.stats['processing_time_seconds'] += time.time() - start_time
        self.stats['data_loaded_mb'] += result_data.memory_usage(deep=True).sum() / 1024**2
        
        if debug:
            logger.info(f"Loaded {len(result_data)} bars for {symbol} {timeframe}")
        
        return result_data
    
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime,
                timeframe: str = '1min', use_continuous: bool = True,
                roll_days: int = 8) -> pd.DataFrame:
        """
        Get market data with specified parameters.
        
        Args:
            symbol: Market symbol
            start_date: Start date
            end_date: End date
            timeframe: Data timeframe
            use_continuous: Whether to build continuous contract
            roll_days: Days before expiration to roll
            
        Returns:
            DataFrame with market data
        """
        logger.info(f"Getting {symbol} data from {start_date} to {end_date}")
        
        # Check cache
        if self.enable_cache and self.cache:
            cached_data = self.cache.get(symbol, timeframe, start_date, end_date)
            if cached_data is not None:
                logger.info("Data loaded from cache")
                return cached_data
        
        # Load data
        if use_continuous:
            # Load continuous contract data
            base_data = self.continuous_builder.load_continuous_data(
                symbol, start_date, end_date
            )
        else:
            # Load individual contracts
            contracts = self.mapper.find_contracts_for_range(
                symbol, start_date, end_date, roll_days
            )
            
            if not contracts:
                logger.warning(f"No contracts found for {symbol}")
                return pd.DataFrame()
            
            # Load and combine
            all_data = []
            for contract_info in contracts:
                df = self.loader.read_dbn_file(
                    contract_info['file'],
                    contract_info['start'],
                    contract_info['end']
                )
                if not df.empty:
                    all_data.append(df)
            
            if not all_data:
                return pd.DataFrame()
            
            base_data = pd.concat(all_data).sort_index()
            base_data = base_data[~base_data.index.duplicated(keep='first')]
        
        # Upsample if needed
        if timeframe != '1min':
            result_data = self.upsampler.upsample(base_data, timeframe)
        else:
            result_data = base_data
        
        # Cache result
        if self.enable_cache and self.cache and not result_data.empty:
            self.cache.put(symbol, timeframe, start_date, end_date, result_data)
        
        return result_data
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols"""
        return list(self.mapper.available_files.keys())
    
    def get_available_date_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Get the available date range for a symbol.
        
        Args:
            symbol: Market symbol
            
        Returns:
            Tuple of (start_date, end_date) or None
        """
        if symbol not in self.mapper.contract_metadata:
            return None
        
        all_starts = []
        all_ends = []
        
        for contract_code, file_infos in self.mapper.contract_metadata[symbol].items():
            for info in file_infos:
                all_starts.append(info['start'])
                all_ends.append(info['end'])
        
        if all_starts and all_ends:
            return (min(all_starts).to_pydatetime(), max(all_ends).to_pydatetime())
        
        return None
    
    def validate_data_availability(self, symbol: str, start_date: datetime,
                                  end_date: datetime) -> Dict[str, Any]:
        """
        Validate data availability for a request.
        
        Args:
            symbol: Market symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            Validation results
        """
        results = {
            'symbol': symbol,
            'requested_start': start_date,
            'requested_end': end_date,
            'available': False,
            'coverage_pct': 0,
            'gaps': [],
            'contracts_needed': 0
        }
        
        # Check symbol availability
        if symbol not in self.mapper.available_files:
            results['error'] = f"Symbol {symbol} not available"
            return results
        
        # Get available range
        available_range = self.get_available_date_range(symbol)
        if not available_range:
            results['error'] = "No data range available"
            return results
        
        results['available_start'] = available_range[0]
        results['available_end'] = available_range[1]
        
        # Check coverage
        contracts = self.mapper.find_contracts_for_range(symbol, start_date, end_date)
        results['contracts_needed'] = len(contracts)
        
        if contracts:
            results['available'] = True
            
            # Calculate coverage
            total_days = (end_date - start_date).days
            covered_days = 0
            
            for contract in contracts:
                contract_days = (contract['end'] - contract['start']).days
                covered_days += contract_days
            
            results['coverage_pct'] = (covered_days / total_days * 100) if total_days > 0 else 0
            
            # Find gaps
            for i in range(len(contracts) - 1):
                gap_start = contracts[i]['end']
                gap_end = contracts[i + 1]['start']
                if gap_end > gap_start:
                    results['gaps'].append((gap_start, gap_end))
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get interface statistics"""
        stats = self.stats.copy()
        
        # Add cache statistics if available
        if self.cache:
            cache_stats = self.cache.get_statistics()
            stats['cache'] = cache_stats
        
        # Calculate averages
        if stats['requests'] > 0:
            stats['avg_processing_time'] = stats['processing_time_seconds'] / stats['requests']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['requests']
        else:
            stats['avg_processing_time'] = 0
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def clear_cache(self, tier: Optional[CacheTier] = None):
        """Clear cache (optionally for specific tier)"""
        if self.cache:
            if tier:
                self.cache.clear_tier(tier)
                logger.info(f"Cleared cache tier {tier.value}")
            else:
                for cache_tier in CacheTier:
                    self.cache.clear_tier(cache_tier)
                logger.info("Cleared all cache tiers")
    
    def cleanup_cache(self, max_age_days: int = 30, max_size_gb: float = 50.0):
        """Run cache cleanup"""
        if self.cache:
            self.cache.cleanup(max_age_days, max_size_gb)