"""
Data Cache Utilities - Databento Implementation
Handles historical data fetching and caching using Databento.
"""

import os
import logging
from datetime import datetime
import pandas as pd
import pytz
from pathlib import Path

# Import Databento interface
from .databento_data_interface import DatabentoInterface

# Initialize logger
logger = logging.getLogger(__name__)

# Define cache directory (for compatibility)
cache_dir = "data_cache"
os.makedirs(cache_dir, exist_ok=True)

# Initialize Databento interface as singleton
_databento_interface = None

def get_databento_interface():
    """Get or create Databento interface singleton"""
    global _databento_interface
    if _databento_interface is None:
        _databento_interface = DatabentoInterface(
            data_directory="databento_data",
            cache_directory="databento_cache",
            enable_cache=True,
            memory_limit_mb=4096
        )
    return _databento_interface

def get_filename(contract, bar_size):
    """
    Legacy function for compatibility.
    Returns cache filename for a contract.
    """
    return os.path.join(
        cache_dir, f"{contract.localSymbol}_{bar_size.replace(' ', '')}.parquet"
    )

def fetch_and_cache(contract, bar_size, active_start, active_end, debug=True, ib=None):
    """
    Legacy function replaced with Databento implementation.
    
    Args:
        contract: Contract object (used for symbol extraction)
        bar_size: Bar size string (e.g., "1 min", "5 mins")
        active_start: Start date
        active_end: End date
        debug: Debug output flag
        ib: IB connection (ignored, no longer needed)
        
    Returns:
        DataFrame with market data
    """
    if debug:
        logger.info(f"[DATABENTO] Fetching data for contract (legacy interface)")
    
    # Extract symbol from contract
    symbol = None
    if hasattr(contract, 'symbol'):
        symbol = contract.symbol
    elif hasattr(contract, 'localSymbol'):
        # Extract base symbol from local symbol (e.g., "NQZ3" -> "NQ")
        local_symbol = contract.localSymbol
        if local_symbol.startswith('NQ'):
            symbol = 'NQ'
        elif local_symbol.startswith('ES'):
            symbol = 'ES'
    
    if not symbol:
        logger.error("Could not determine symbol from contract")
        return pd.DataFrame()
    
    # Convert bar size to timeframe
    timeframe_map = {
        "1 min": "1min",
        "5 mins": "5min",
        "15 mins": "15min",
        "60 mins": "60min",
        "1 hour": "60min",
        "4 hours": "4hour",
        "1 day": "1day"
    }
    
    timeframe = timeframe_map.get(bar_size, "1min")
    
    # Get Databento interface
    databento = get_databento_interface()
    
    # Fetch data using Databento
    try:
        df = databento.get_data(
            symbol=symbol,
            start_date=active_start,
            end_date=active_end,
            timeframe=timeframe,
            use_continuous=True,
            roll_days=8
        )
        
        if debug and not df.empty:
            logger.info(f"[DATABENTO] Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"[DATABENTO] Failed to fetch data: {e}")
        return pd.DataFrame()

def fetch_market_data(symbol, timeframe, expiration_dates, data_cache_dir=None, 
                     period="2 years", custom_start=None, custom_end=None, 
                     roll_days=8, min_search_bars=100, debug=True):
    """
    Main function for fetching market data - Databento implementation.
    
    Args:
        symbol: Market symbol (e.g., "NQ", "ES")
        timeframe: Timeframe string (e.g., "5min", "1min") 
        expiration_dates: List of contract expiration dates (not used with Databento)
        data_cache_dir: Cache directory (not used, kept for compatibility)
        period: Time period for analysis
        custom_start: Custom start date string
        custom_end: Custom end date string
        roll_days: Days before expiration to roll contracts
        min_search_bars: Minimum bars to search (not used)
        debug: Debug output flag
        
    Returns:
        DataFrame: Combined market data
    """
    if debug:
        logger.info(f"[DATABENTO] Fetching {symbol} {timeframe} data for period: {period}")
    
    # Get Databento interface
    databento = get_databento_interface()
    
    # Use Databento's fetch_market_data which maintains compatibility
    try:
        df = databento.fetch_market_data(
            symbol=symbol,
            timeframe=timeframe,
            expiration_dates=expiration_dates,
            data_cache_dir=data_cache_dir,
            period=period,
            custom_start=custom_start,
            custom_end=custom_end,
            roll_days=roll_days,
            min_search_bars=min_search_bars,
            debug=debug
        )
        
        if debug:
            if not df.empty:
                logger.info(f"[DATABENTO] Successfully loaded {len(df)} bars")
            else:
                logger.warning(f"[DATABENTO] No data retrieved for {symbol} {timeframe}")
        
        return df
        
    except Exception as e:
        logger.error(f"[DATABENTO] Error fetching market data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# Additional utility functions for compatibility
def clear_cache(symbol=None):
    """Clear cached data"""
    databento = get_databento_interface()
    if databento.cache:
        databento.clear_cache()
        logger.info("[DATABENTO] Cache cleared")

def get_cache_statistics():
    """Get cache statistics"""
    databento = get_databento_interface()
    return databento.get_statistics()

def validate_data_availability(symbol, start_date, end_date):
    """Validate if data is available for the requested period"""
    databento = get_databento_interface()
    return databento.validate_data_availability(symbol, start_date, end_date)

# Legacy compatibility - ensure timezone handling matches original
from .time_utils import ensure_ny_timezone

logger.info("[DATABENTO] Data cache utils initialized with Databento backend")