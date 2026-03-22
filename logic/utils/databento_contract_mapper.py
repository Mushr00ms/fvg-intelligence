"""
Databento Contract Mapper
Maps date ranges to appropriate Databento contract files and handles contract transitions.
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DatabentoContractMapper:
    """Maps date ranges to Databento contract files"""
    
    # Contract month codes
    MONTH_CODES = {
        'F': 1,  # January
        'G': 2,  # February
        'H': 3,  # March
        'J': 4,  # April
        'K': 5,  # May
        'M': 6,  # June
        'N': 7,  # July
        'Q': 8,  # August
        'U': 9,  # September
        'V': 10, # October
        'X': 11, # November
        'Z': 12  # December
    }
    
    # Reverse mapping
    MONTH_TO_CODE = {v: k for k, v in MONTH_CODES.items()}
    
    def __init__(self, data_directory: str = "logic/databento_data"):
        """
        Initialize the contract mapper.
        
        Args:
            data_directory: Path to the Databento data directory
        """
        self.data_dir = Path(data_directory)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {data_directory} does not exist")
        
        # Cache available files
        self.available_files = self._scan_available_files()
        self.contract_metadata = self._parse_contract_metadata()
        
    def _scan_available_files(self) -> Dict[str, List[Path]]:
        """Scan directory for available DBN files"""
        files = {}
        
        # Look for .dbn.zst files in the data directory
        # First check the base directory
        for file_path in self.data_dir.glob("*.dbn.zst"):
            filename = file_path.name
            
            # Extract symbol from filename (ES or NQ)
            if "ES" in filename:
                symbol = "ES"
            elif "NQ" in filename:
                symbol = "NQ"
            else:
                continue
            
            if symbol not in files:
                files[symbol] = []
            files[symbol].append(file_path)
        
        # Also check subdirectories for each symbol
        for symbol in ["ES", "NQ"]:
            symbol_dir = self.data_dir / symbol
            if symbol_dir.exists():
                for file_path in symbol_dir.glob("**/*.dbn.zst"):
                    if symbol not in files:
                        files[symbol] = []
                    files[symbol].append(file_path)
        
        # Sort files by name for each symbol
        for symbol in files:
            files[symbol].sort()
        
        logger.info(f"Found {sum(len(f) for f in files.values())} contract files")
        return files
    
    def _parse_contract_metadata(self) -> Dict[str, Dict]:
        """Parse contract information from filenames"""
        metadata = {}
        
        for symbol, file_list in self.available_files.items():
            metadata[symbol] = {}
            
            for file_path in file_list:
                filename = file_path.name
                
                # Parse contract codes from filename
                # Format: glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-1m.CONTRACT.dbn.zst
                # or: glbx-mdp3-YYYYMMDD-YYYYMMDD.ohlcv-1m.CONTRACT1-CONTRACT2.dbn.zst
                
                pattern = r'glbx-mdp3-(\d{8})-(\d{8})\.ohlcv-1m\.([A-Z0-9\-]+)\.dbn\.zst'
                match = re.match(pattern, filename)
                
                if match:
                    start_date_str = match.group(1)
                    end_date_str = match.group(2)
                    contract_spec = match.group(3)
                    
                    start_date = pd.to_datetime(start_date_str).tz_localize('UTC')
                    end_date = pd.to_datetime(end_date_str).tz_localize('UTC')
                    
                    # Parse contract(s)
                    if '-' in contract_spec and symbol in contract_spec:
                        # Split contract pair (e.g., ESM0-ESU0)
                        contracts = contract_spec.split('-')
                        for contract in contracts:
                            if symbol in contract:
                                contract_code = contract.replace(symbol, '')
                                if contract_code not in metadata[symbol]:
                                    metadata[symbol][contract_code] = []
                                metadata[symbol][contract_code].append({
                                    'file': file_path,
                                    'start': start_date,
                                    'end': end_date,
                                    'is_spread': '-' in contract_spec
                                })
                    elif symbol in contract_spec:
                        # Single contract
                        contract_code = contract_spec.replace(symbol, '')
                        if contract_code not in metadata[symbol]:
                            metadata[symbol][contract_code] = []
                        metadata[symbol][contract_code].append({
                            'file': file_path,
                            'start': start_date,
                            'end': end_date,
                            'is_spread': False
                        })
        
        return metadata
    
    def parse_contract_code(self, contract_code: str) -> Tuple[int, int]:
        """
        Parse contract code to year and month.
        
        Args:
            contract_code: Contract code like 'M0', 'Z1', 'H2'
            
        Returns:
            Tuple of (year, month)
        """
        if len(contract_code) != 2:
            raise ValueError(f"Invalid contract code: {contract_code}")
        
        month_char = contract_code[0]
        year_digit = contract_code[1]
        
        if month_char not in self.MONTH_CODES:
            raise ValueError(f"Invalid month code: {month_char}")
        
        month = self.MONTH_CODES[month_char]
        
        # Convert single digit year to full year
        # Disambiguate decade: check file date ranges to determine if 201x or 202x
        year_int = int(year_digit)
        # Default to 2020s decade; caller can override via context
        year = 2020 + year_int
        
        return year, month
    
    def get_contract_expiration(self, symbol: str, contract_code: str) -> datetime:
        """
        Get the expiration date for a contract.
        
        Args:
            symbol: Market symbol (ES or NQ)
            contract_code: Contract code like 'M0', 'Z1'
            
        Returns:
            Expiration datetime
        """
        year, month = self.parse_contract_code(contract_code)
        
        # Futures typically expire on the third Friday of the contract month
        # This is a simplified calculation
        first_day = datetime(year, month, 1)
        
        # Find the third Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(weeks=2)
        
        return third_friday
    
    def find_contracts_for_range(self, symbol: str, start_date: datetime, 
                                 end_date: datetime, roll_days: int = 8) -> List[Dict]:
        """
        Find all contracts needed to cover a date range.
        
        Args:
            symbol: Market symbol (ES or NQ)
            start_date: Start of date range
            end_date: End of date range
            roll_days: Days before expiration to roll to next contract
            
        Returns:
            List of contract info dictionaries
        """
        if symbol not in self.contract_metadata:
            raise ValueError(f"No data available for symbol {symbol}")
        
        contracts_needed = []
        seen_files = set()
        
        # Convert dates to pandas timestamps for comparison (ensure timezone-aware)
        if hasattr(start_date, 'tzinfo') and start_date.tzinfo:
            start_pd = pd.Timestamp(start_date)
        else:
            start_pd = pd.Timestamp(start_date).tz_localize('UTC')
            
        if hasattr(end_date, 'tzinfo') and end_date.tzinfo:
            end_pd = pd.Timestamp(end_date)
        else:
            end_pd = pd.Timestamp(end_date).tz_localize('UTC')
        
        # Iterate through all contracts for this symbol
        for contract_code, file_infos in self.contract_metadata[symbol].items():
            for file_info in file_infos:
                # Check if this file covers any part of our date range
                file_start = file_info['start']
                file_end = file_info['end']
                
                # Check for overlap
                if file_start <= end_pd and file_end >= start_pd:
                    # Skip spread files — their prices are differentials, not outright
                    if file_info.get('is_spread', False):
                        continue
                    file_path = file_info['file']
                    if file_path not in seen_files:
                        seen_files.add(file_path)
                        
                        # Calculate effective date range for this contract
                        effective_start = max(file_start, start_pd)
                        effective_end = min(file_end, end_pd)
                        
                        # For Databento historical files, we use the full data range
                        # These are not live contracts but historical data files
                        # The contract code in the filename is just an identifier
                        # Comment out expiration truncation that was incorrectly limiting data:
                        # if not file_info['is_spread']:
                        #     try:
                        #         expiration = self.get_contract_expiration(symbol, contract_code)
                        #         roll_date = pd.Timestamp(expiration - timedelta(days=roll_days))
                        #         effective_end = min(effective_end, roll_date)
                        #     except:
                        #         pass
                        
                        contracts_needed.append({
                            'symbol': symbol,
                            'contract': contract_code,
                            'file': file_path,
                            'start': effective_start,
                            'end': effective_end,
                            'is_spread': file_info['is_spread']
                        })
        
        # Sort by start date
        contracts_needed.sort(key=lambda x: x['start'])
        
        return contracts_needed
    
    def get_continuous_contract_files(self, symbol: str, start_date: datetime,
                                     end_date: datetime, roll_days: int = 8) -> List[Path]:
        """
        Get list of files for building a continuous contract.
        
        Args:
            symbol: Market symbol (ES or NQ)
            start_date: Start of date range
            end_date: End of date range
            roll_days: Days before expiration to roll
            
        Returns:
            Ordered list of file paths
        """
        contracts = self.find_contracts_for_range(symbol, start_date, end_date, roll_days)
        return [c['file'] for c in contracts]
    
    def get_file_for_date(self, symbol: str, target_date: datetime) -> Optional[Path]:
        """
        Get the appropriate file for a specific date.
        
        Args:
            symbol: Market symbol (ES or NQ)
            target_date: Date to find file for
            
        Returns:
            Path to the file or None if not found
        """
        contracts = self.find_contracts_for_range(symbol, target_date, target_date)
        if contracts:
            return contracts[0]['file']
        return None