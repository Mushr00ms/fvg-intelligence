"""
tick_data_feed.py — Load Databento trades data and provide bars + tick iterator.

Loads .trades.dbn.zst files (one per trading day) from Databento's GLBX.MDP3
dataset. Provides:
  - 5-minute OHLCV bars for FVG detection
  - 1-minute OHLCV bars for sub-5min fill resolution
  - Raw tick iterator for tick-accurate fill simulation

Reuses patterns from bot/backtest/enrich_volume.py (extraction, contract
resolution, loading).
"""

import os
import sys
import zipfile
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from logic.utils.contract_utils import generate_nq_expirations, get_contract_for_date
from bot.backtest.us_holidays import is_trading_day

# NQ month letter codes
_MONTH_LETTERS = {3: "H", 6: "M", 9: "U", 12: "Z"}

# Extract to Linux FS for fast I/O (10x vs /mnt/c/)
DEFAULT_EXTRACT_DIR = "/tmp/databento_trades_nq"

# RTH session for NQ
RTH_START = time(9, 30)
RTH_END = time(16, 0)

NY_TZ = "America/New_York"


def expiry_to_symbol(exp_date) -> str:
    """Convert expiration date to Databento NQ symbol (e.g. NQH5)."""
    d = exp_date.date() if hasattr(exp_date, "date") and callable(exp_date.date) else exp_date
    letter = _MONTH_LETTERS[d.month]
    return f"NQ{letter}{d.year % 10}"


def find_trades_zips(root: str) -> list[str]:
    """Find all Databento trades zip files in project root."""
    zips = []
    for fname in sorted(os.listdir(root)):
        if not (fname.startswith("GLBX") and fname.endswith(".zip")):
            continue
        path = os.path.join(root, fname)
        try:
            with zipfile.ZipFile(path) as zf:
                if any(".trades.dbn.zst" in i.filename for i in zf.infolist()):
                    zips.append(path)
        except Exception:
            continue
    return zips


def extract_trades(zip_paths: list[str], out_dir: str, verbose: bool = True):
    """Extract .trades.dbn.zst files from zip(s) to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            entries = [i for i in zf.infolist() if ".trades.dbn.zst" in i.filename]
            for info in entries:
                dest = os.path.join(out_dir, os.path.basename(info.filename))
                if os.path.exists(dest):
                    total += 1
                    continue
                with zf.open(info) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                total += 1
    if verbose:
        print(f"  {total} tick files ready in {out_dir}")


def load_day_ticks(date_str: str, extract_dir: str, symbol: str = None):
    """
    Load a single day's Databento trades into a DataFrame.

    Args:
        date_str: "YYYY-MM-DD" format
        extract_dir: directory containing extracted .dbn.zst files
        symbol: NQ contract symbol to filter (e.g. "NQH5"). If None, all NQ outrights.

    Returns:
        DataFrame with columns: ts_event, price, size, side, symbol
        Filtered to RTH (09:30-16:00 ET), sorted by ts_event.
        None if no data found.
    """
    import databento as db

    compact = date_str.replace("-", "")
    path = os.path.join(extract_dir, f"glbx-mdp3-{compact}.trades.dbn.zst")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    try:
        store = db.DBNStore.from_file(path)
        df = store.to_df()
    except Exception:
        return None

    if df.empty:
        return None

    # Filter to NQ outrights (exclude spreads with "-")
    nq_mask = df["symbol"].str.startswith("NQ") & ~df["symbol"].str.contains("-")
    df = df[nq_mask].copy()

    # Filter to specific contract if provided
    if symbol:
        # Try exact match first, then try all NQ outrights as fallback
        sym_df = df[df["symbol"] == symbol]
        if sym_df.empty:
            # Roll fallback: use any NQ outright with volume
            pass
        else:
            df = sym_df

    # Convert to ET and filter RTH
    if df.index.tz is not None:
        df = df.reset_index()
        ts_col = "ts_event"
        if ts_col not in df.columns:
            # DBN index is ts_event
            df = df.rename(columns={df.columns[0]: ts_col})
        df["ts_et"] = df[ts_col].dt.tz_convert(NY_TZ)
    else:
        df["ts_et"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(NY_TZ)

    rth_mask = (df["ts_et"].dt.time >= RTH_START) & (df["ts_et"].dt.time < RTH_END)
    df = df[rth_mask].sort_values("ts_et").reset_index(drop=True)

    return df


def resample_ticks_to_bars(ticks_df, freq="5min"):
    """Resample tick data to OHLCV bars at given frequency.

    Args:
        ticks_df: DataFrame with ts_et, price, size columns
        freq: "5min" or "1min"

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    df = ticks_df.set_index("ts_et")
    bars = df["price"].resample(freq, label="left", closed="left").agg(
        open="first", high="max", low="min", close="last",
    )
    bars["volume"] = df["size"].resample(freq, label="left", closed="left").sum()
    bars = bars.dropna(subset=["open"]).reset_index()
    bars = bars.rename(columns={"ts_et": "date"})
    return bars


class TickDataFeed:
    """
    Provides historical bar and tick data for replay.

    Two data sources:
      - IB 1-second bars (parquet) → resampled to 5min for FVG detection
        (identical to backtester → pixel-perfect FVGs)
      - Databento trades (dbn.zst) → tick-by-tick for fill simulation
        (real slippage, volume at price)

    Usage:
        feed = TickDataFeed(ib_data_dir="bot/data/",
                            zip_paths=[...], extract_dir="/tmp/databento_trades_nq")
        feed.prepare()

        for day in feed.trading_days("2025-01-01", "2025-12-31"):
            bars_5m = feed.get_5min_bars(day)   # From IB 1s bars
            ticks = feed.get_ticks(day)          # From Databento
    """

    def __init__(self, zip_paths=None, extract_dir=DEFAULT_EXTRACT_DIR,
                 ib_data_dir=None):
        self._zip_paths = zip_paths or []
        self._extract_dir = extract_dir
        self._ib_data_dir = ib_data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data")
        self._exp_dates = generate_nq_expirations(2019, 2027)
        self._tick_cache = {}  # date_str -> DataFrame (keep 1 day in memory)

    def prepare(self, verbose=True):
        """Extract zip files if needed."""
        if self._zip_paths:
            extract_trades(self._zip_paths, self._extract_dir, verbose=verbose)

    def trading_days(self, start_date: str, end_date: str) -> list[str]:
        """List available trading days with BOTH IB bars and Databento ticks."""
        available = []
        for fname in sorted(os.listdir(self._extract_dir)):
            if not fname.endswith(".trades.dbn.zst"):
                continue
            parts = fname.replace("glbx-mdp3-", "").replace(".trades.dbn.zst", "")
            date_str = f"{parts[:4]}-{parts[4:6]}-{parts[6:8]}"
            if date_str < start_date or date_str > end_date:
                continue
            if not is_trading_day(date_str.replace("-", "")):
                continue
            # Require IB 1s data too (for pixel-perfect FVG detection)
            ib_path = os.path.join(self._ib_data_dir,
                                   f"nq_1secs_{parts}.parquet")
            if not os.path.exists(ib_path):
                continue
            available.append(date_str)
        return available

    def _resolve_symbol(self, date_str: str) -> str:
        """Resolve front-month NQ symbol for a date."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        exp = get_contract_for_date(dt, self._exp_dates, roll_days=8)
        return expiry_to_symbol(exp)

    def _load_day(self, date_str: str) -> pd.DataFrame | None:
        """Load and cache a day's tick data."""
        if date_str in self._tick_cache:
            return self._tick_cache[date_str]

        symbol = self._resolve_symbol(date_str)
        df = load_day_ticks(date_str, self._extract_dir, symbol=symbol)

        # Clear cache (keep only 1 day to limit memory)
        self._tick_cache.clear()
        if df is not None:
            self._tick_cache[date_str] = df
        return df

    def get_ticks(self, date_str: str) -> pd.DataFrame | None:
        """Get Databento tick data for fill simulation.

        Columns: ts_et, price, size, side, symbol.
        """
        return self._load_day(date_str)

    def _load_ib_1s(self, date_str: str) -> pd.DataFrame | None:
        """Load IB 1-second bars for a day.

        Same data the backtester uses → identical FVG detection.
        """
        compact = date_str.replace("-", "")
        path = os.path.join(self._ib_data_dir, f"nq_1secs_{compact}.parquet")
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(NY_TZ)
        return df.sort_values('date').reset_index(drop=True)

    def get_5min_bars(self, date_str: str) -> pd.DataFrame | None:
        """Get 5-minute OHLCV bars from IB 1s data (same as backtester).

        Uses IB data for pixel-perfect FVG detection parity.
        Falls back to Databento tick resampling if IB data unavailable.
        """
        ib_df = self._load_ib_1s(date_str)
        if ib_df is not None:
            bars = ib_df.set_index('date').resample('5min', label='left', closed='left').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna(subset=['open']).reset_index()
            return bars
        # Fallback: Databento ticks
        df = self._load_day(date_str)
        if df is None:
            return None
        return resample_ticks_to_bars(df, "5min")

    def get_1min_bars(self, date_str: str) -> pd.DataFrame | None:
        """Get 1-minute OHLCV bars from IB 1s data."""
        ib_df = self._load_ib_1s(date_str)
        if ib_df is not None:
            bars = ib_df.set_index('date').resample('1min', label='left', closed='left').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna(subset=['open']).reset_index()
            return bars
        df = self._load_day(date_str)
        if df is None:
            return None
        return resample_ticks_to_bars(df, "1min")

    def get_symbol(self, date_str: str) -> str:
        """Get front-month NQ symbol for a date."""
        return self._resolve_symbol(date_str)
