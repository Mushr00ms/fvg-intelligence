"""
fetch_data.py — WSL-side launcher for IB historical data fetcher.

Spawns python.exe to run ib_data_fetcher.py on Windows, which connects
to IB TWS and fetches 1-second bars day-by-day into bot/data/.

Usage:
    python3 bot/backtest/fetch_data.py --start 20260102 --end 20260322
    python3 bot/backtest/fetch_data.py --ytd          # Jan 2 to today
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, date


# WSL path conversion (same pattern as risk-webapp)
_WSL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRIDGE_DIR = os.path.join(_WSL_DIR, "bridge")
_DATA_DIR = os.path.join(_WSL_DIR, "data")

_FETCHER_WSL = os.path.join(_BRIDGE_DIR, "ib_data_fetcher.py")

if _WSL_DIR.startswith('/mnt/'):
    _parts = _WSL_DIR.split('/')
    _drive = _parts[2].upper()
    _win_base = _drive + ':\\' + '\\'.join(_parts[3:])
    FETCHER_WIN = _win_base + '\\bridge\\ib_data_fetcher.py'
    DATA_DIR_WIN = _win_base + '\\data'
else:
    FETCHER_WIN = _FETCHER_WSL
    DATA_DIR_WIN = _DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Fetch NQ historical data from IB")
    parser.add_argument("--start", help="Start date YYYYMMDD (default: 20260102)")
    parser.add_argument("--end", help="End date YYYYMMDD (default: today)")
    parser.add_argument("--ytd", action="store_true", help="Fetch YTD 2026 (Jan 2 to today)")
    parser.add_argument("--bar-size", default="1 secs", help="Bar size (default: '1 secs')")
    parser.add_argument("--ib-port", type=int, default=7497, help="IB TWS port")
    parser.add_argument("--client-id", type=int, default=20, help="IB client ID")
    args = parser.parse_args()

    if args.ytd:
        start = "20260102"
        end = datetime.now().strftime("%Y%m%d")
    else:
        start = args.start or "20260102"
        end = args.end or datetime.now().strftime("%Y%m%d")

    os.makedirs(_DATA_DIR, exist_ok=True)

    print(f"Fetching NQ {args.bar_size} bars: {start} → {end}")
    print(f"Output: {_DATA_DIR}")
    print(f"Using IB port: {args.ib_port}")
    print()

    cmd = [
        'python.exe', FETCHER_WIN,
        '--start', start,
        '--end', end,
        '--out-dir', DATA_DIR_WIN,
        '--bar-size', args.bar_size,
        '--ib-port', str(args.ib_port),
        '--client-id', str(args.client_id),
    ]

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(
            cmd,
            timeout=7200,  # 2 hour max
        )
        sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        print("\nFetch timed out after 2 hours. Re-run to resume (cached days are skipped).")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: python.exe not found. Make sure Windows Python is installed and accessible from WSL.")
        sys.exit(1)


if __name__ == "__main__":
    main()
