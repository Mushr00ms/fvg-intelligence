"""
Databento Intelligent Cache Manager
Implements tiered caching system for optimized data access and memory management.
"""

import os
import json
import logging
import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from enum import Enum

logger = logging.getLogger(__name__)

class CacheTier(Enum):
    """Cache tier definitions"""
    TIER1_2YR = "tier1_2yr"
    TIER2_5YR = "tier2_5yr"
    TIER3_10YR = "tier3_10yr"
    TIER4_15YR = "tier4_15yr"

class DatabentoCache:
    """Intelligent tiered cache manager for Databento data"""
    
    # Cache tier configurations
    CACHE_TIERS = {
        CacheTier.TIER1_2YR: {
            'name': '2yr',
            'days': 730,
            'format': 'parquet',
            'compression': 'snappy',
            'priority': 1
        },
        CacheTier.TIER2_5YR: {
            'name': '5yr',
            'days': 1825,
            'format': 'parquet',
            'compression': 'gzip',
            'priority': 2
        },
        CacheTier.TIER3_10YR: {
            'name': '10yr',
            'days': 3650,
            'format': 'parquet',
            'compression': 'brotli',
            'priority': 3
        },
        CacheTier.TIER4_15YR: {
            'name': '15yr',
            'days': 5475,
            'format': 'parquet',
            'compression': 'brotli',
            'priority': 4
        }
    }
    
    def __init__(self, cache_directory: str = "logic/databento_cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_directory: Base directory for cache storage
        """
        self.cache_dir = Path(cache_directory)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create tier directories
        for tier in CacheTier:
            tier_dir = self.cache_dir / tier.value
            tier_dir.mkdir(exist_ok=True)
        
        # Load cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'promotions': 0,
            'evictions': 0
        }
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {'entries': {}, 'access_counts': {}, 'last_cleanup': None}
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        try:
            # Convert any datetime objects to strings before saving
            import copy
            metadata_copy = copy.deepcopy(self.metadata)
            
            # Recursive function to convert datetime objects
            def convert_datetimes(obj):
                if isinstance(obj, dict):
                    return {k: convert_datetimes(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetimes(item) for item in obj]
                elif hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                elif isinstance(obj, (pd.Timestamp, datetime)):
                    return pd.Timestamp(obj).isoformat()
                else:
                    return obj
            
            # Convert all datetime objects recursively
            metadata_copy = convert_datetimes(metadata_copy)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_copy, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
            # Log the problematic data for debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _generate_cache_key(self, symbol: str, timeframe: str, 
                          start_date: datetime, end_date: datetime) -> str:
        """Generate unique cache key for data request"""
        # Normalize dates to day precision for consistent cache keys
        # This prevents cache misses due to microsecond differences
        start_normalized = pd.Timestamp(start_date).normalize()
        end_normalized = pd.Timestamp(end_date).normalize()
        
        # Use date strings without time component
        start_str = start_normalized.strftime('%Y-%m-%d')
        end_str = end_normalized.strftime('%Y-%m-%d')
            
        key_str = f"{symbol}_{timeframe}_{start_str}_{end_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _determine_tier(self, start_date: datetime, end_date: datetime) -> CacheTier:
        """Determine appropriate cache tier based on date range"""
        # Convert to pandas timestamps for consistent handling
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        days_range = (end - start).days
        
        for tier, config in self.CACHE_TIERS.items():
            if days_range <= config['days']:
                return tier
        
        # Default to largest tier
        return CacheTier.TIER4_15YR
    
    def _get_cache_path(self, cache_key: str, tier: CacheTier) -> Path:
        """Get the file path for a cache entry"""
        tier_dir = self.cache_dir / tier.value
        return tier_dir / f"{cache_key}.parquet"
    
    def get(self, symbol: str, timeframe: str,
            start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Retrieve data from cache if available.
        
        Args:
            symbol: Market symbol
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame if cache hit, None if miss
        """
        cache_key = self._generate_cache_key(symbol, timeframe, start_date, end_date)
        
        # Check if entry exists in metadata
        if cache_key not in self.metadata['entries']:
            self.stats['misses'] += 1
            return None
        
        entry = self.metadata['entries'][cache_key]
        tier = CacheTier(entry['tier'])
        cache_path = self._get_cache_path(cache_key, tier)
        
        if not cache_path.exists():
            # Cache entry exists in metadata but file is missing
            logger.warning(f"Cache file missing: {cache_path}")
            del self.metadata['entries'][cache_key]
            self._save_metadata()
            self.stats['misses'] += 1
            return None
        
        try:
            # Load cached data
            df = pd.read_parquet(cache_path)
            
            # Update access count and time
            self.metadata['access_counts'][cache_key] = \
                self.metadata['access_counts'].get(cache_key, 0) + 1
            entry['last_accessed'] = datetime.now().isoformat()
            self._save_metadata()
            
            self.stats['hits'] += 1
            logger.info(f"Cache hit for {symbol} {timeframe} from tier {tier.value}")
            
            # Consider promotion if frequently accessed
            self._consider_promotion(cache_key, entry, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load cache file {cache_path}: {e}")
            self.stats['misses'] += 1
            return None
    
    def put(self, symbol: str, timeframe: str,
            start_date: datetime, end_date: datetime,
            data: pd.DataFrame) -> bool:
        """
        Store data in cache.
        
        Args:
            symbol: Market symbol
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date
            data: DataFrame to cache
            
        Returns:
            True if successful, False otherwise
        """
        if data.empty:
            return False
        
        cache_key = self._generate_cache_key(symbol, timeframe, start_date, end_date)
        tier = self._determine_tier(start_date, end_date)
        cache_path = self._get_cache_path(cache_key, tier)
        
        try:
            # Save data with appropriate compression
            compression = self.CACHE_TIERS[tier]['compression']
            data.to_parquet(cache_path, compression=compression)
            
            # Update metadata - handle different datetime types
            if hasattr(start_date, 'isoformat'):
                start_date_str = start_date.isoformat()
            else:
                start_date_str = pd.Timestamp(start_date).isoformat()
                
            if hasattr(end_date, 'isoformat'):
                end_date_str = end_date.isoformat()
            else:
                end_date_str = pd.Timestamp(end_date).isoformat()
            
            # Create entry with all datetime objects already converted
            new_entry = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date_str,
                'end_date': end_date_str,
                'tier': tier.value,
                'created': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'size_bytes': int(cache_path.stat().st_size),  # Ensure it's an int
                'record_count': int(len(data))  # Ensure it's an int
            }
            
            # Store the entry
            self.metadata['entries'][cache_key] = new_entry
            self.metadata['access_counts'][cache_key] = 0
            
            self._save_metadata()
            
            logger.info(f"Cached {len(data)} records for {symbol} {timeframe} in tier {tier.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _consider_promotion(self, cache_key: str, entry: Dict, data: pd.DataFrame):
        """Consider promoting frequently accessed data to a faster tier"""
        access_count = self.metadata['access_counts'].get(cache_key, 0)
        current_tier = CacheTier(entry['tier'])
        
        # Promotion thresholds
        promotion_thresholds = {
            CacheTier.TIER4_15YR: 10,
            CacheTier.TIER3_10YR: 20,
            CacheTier.TIER2_5YR: 50
        }
        
        # Check if eligible for promotion
        if current_tier in promotion_thresholds:
            threshold = promotion_thresholds[current_tier]
            if access_count >= threshold:
                # Find target tier
                target_tier = None
                for tier in [CacheTier.TIER1_2YR, CacheTier.TIER2_5YR, CacheTier.TIER3_10YR]:
                    if self.CACHE_TIERS[tier]['priority'] < self.CACHE_TIERS[current_tier]['priority']:
                        # Check if data fits in this tier
                        days_range = (pd.Timestamp(entry['end_date']) - 
                                    pd.Timestamp(entry['start_date'])).days
                        if days_range <= self.CACHE_TIERS[tier]['days']:
                            target_tier = tier
                            break
                
                if target_tier:
                    self._promote_entry(cache_key, current_tier, target_tier, data)
    
    def _promote_entry(self, cache_key: str, from_tier: CacheTier, 
                      to_tier: CacheTier, data: pd.DataFrame):
        """Promote cache entry to a faster tier"""
        try:
            old_path = self._get_cache_path(cache_key, from_tier)
            new_path = self._get_cache_path(cache_key, to_tier)
            
            # Save with new compression
            compression = self.CACHE_TIERS[to_tier]['compression']
            data.to_parquet(new_path, compression=compression)
            
            # Remove old file
            if old_path.exists():
                old_path.unlink()
            
            # Update metadata
            self.metadata['entries'][cache_key]['tier'] = to_tier.value
            self._save_metadata()
            
            self.stats['promotions'] += 1
            logger.info(f"Promoted cache entry from {from_tier.value} to {to_tier.value}")
            
        except Exception as e:
            logger.error(f"Failed to promote cache entry: {e}")
    
    def cleanup(self, max_age_days: int = 30, max_size_gb: float = 50.0):
        """
        Clean up old or large cache entries.
        
        Args:
            max_age_days: Maximum age for cache entries
            max_size_gb: Maximum total cache size in GB
        """
        logger.info("Starting cache cleanup")
        
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=max_age_days)
        total_size = 0
        entries_to_remove = []
        
        # Sort entries by last access time
        sorted_entries = sorted(
            self.metadata['entries'].items(),
            key=lambda x: x[1].get('last_accessed', ''),
            reverse=True
        )
        
        for cache_key, entry in sorted_entries:
            tier = CacheTier(entry['tier'])
            cache_path = self._get_cache_path(cache_key, tier)
            
            if cache_path.exists():
                size_bytes = cache_path.stat().st_size
                total_size += size_bytes
                
                # Check age
                last_accessed = pd.Timestamp(entry.get('last_accessed', entry['created']))
                if last_accessed < cutoff_time:
                    entries_to_remove.append(cache_key)
                    continue
                
                # Check total size
                if total_size > max_size_gb * 1024**3:
                    entries_to_remove.append(cache_key)
        
        # Remove marked entries
        for cache_key in entries_to_remove:
            self._remove_entry(cache_key)
        
        self.metadata['last_cleanup'] = current_time.isoformat()
        self._save_metadata()
        
        logger.info(f"Cleanup complete: removed {len(entries_to_remove)} entries")
    
    def _remove_entry(self, cache_key: str):
        """Remove a cache entry"""
        if cache_key in self.metadata['entries']:
            entry = self.metadata['entries'][cache_key]
            tier = CacheTier(entry['tier'])
            cache_path = self._get_cache_path(cache_key, tier)
            
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    self.stats['evictions'] += 1
                except Exception as e:
                    logger.error(f"Failed to remove cache file: {e}")
            
            del self.metadata['entries'][cache_key]
            if cache_key in self.metadata['access_counts']:
                del self.metadata['access_counts'][cache_key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.stats.copy()
        
        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        stats['hit_rate'] = stats['hits'] / total_requests if total_requests > 0 else 0
        
        # Calculate cache size
        total_size = 0
        tier_sizes = {}
        
        for tier in CacheTier:
            tier_dir = self.cache_dir / tier.value
            tier_size = sum(f.stat().st_size for f in tier_dir.glob("*.parquet"))
            tier_sizes[tier.value] = tier_size
            total_size += tier_size
        
        stats['total_size_mb'] = total_size / (1024**2)
        stats['tier_sizes_mb'] = {k: v / (1024**2) for k, v in tier_sizes.items()}
        stats['entry_count'] = len(self.metadata['entries'])
        
        return stats
    
    def clear_tier(self, tier: CacheTier):
        """Clear all entries from a specific tier"""
        tier_dir = self.cache_dir / tier.value
        
        # Remove all files
        for file_path in tier_dir.glob("*.parquet"):
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
        
        # Update metadata
        entries_to_remove = [
            key for key, entry in self.metadata['entries'].items()
            if entry['tier'] == tier.value
        ]
        
        for key in entries_to_remove:
            del self.metadata['entries'][key]
            if key in self.metadata['access_counts']:
                del self.metadata['access_counts'][key]
        
        self._save_metadata()
        logger.info(f"Cleared tier {tier.value}: removed {len(entries_to_remove)} entries")