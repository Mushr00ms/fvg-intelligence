"""
ib_data_fetcher.py — Fetch historical 1-second bars from IB TWS.

Runs on Windows Python (via python.exe from WSL) to bypass Hyper-V isolation.
Fetches NQ futures data day-by-day, handles contract rolls, writes parquet.

Usage from WSL:
    python.exe C:\\path\\to\\ib_data_fetcher.py \\
        --start 20260102 --end 20260322 \\
        --out-dir C:\\path\\to\\bot\\data \\
        --ib-port 7497

The fetcher handles:
    - NQ quarterly contract rolls (H/M/U/Z, 8-day roll before expiry)
    - IB pacing limits (max 60 requests / 10 min, auto-throttle)
    - Resume capability (skips days already fetched)
    - Parquet output with consistent schema
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, date

import pytz
_ET = pytz.timezone('US/Eastern')
_UTC = pytz.utc


def _to_utc_str(naive_et_dt):
    """Convert a naive ET datetime to IB UTC dash-format: 'yyyymmdd-HH:MM:SS'."""
    et_aware = _ET.localize(naive_et_dt)
    utc_dt = et_aware.astimezone(_UTC)
    return utc_dt.strftime('%Y%m%d-%H:%M:%S')

try:
    from ib_async import IB, Future, util
    util.startLoop()
except ImportError:
    try:
        from ib_insync import IB, Future, util
        util.startLoop()
    except ImportError:
        print(json.dumps({"error": "Neither ib_async nor ib_insync installed"}))
        sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print(json.dumps({"error": "pandas not installed"}))
    sys.exit(1)


# NQ contract months: H=March, M=June, U=September, Z=December
CONTRACT_MONTHS = [(3, 'H'), (6, 'M'), (9, 'U'), (12, 'Z')]
ROLL_DAYS_BEFORE_EXPIRY = 8


def get_third_friday(year, month):
    """Get the third Friday of a given month (NQ futures expiration)."""
    for day in range(15, 22):
        d = date(year, month, day)
        if d.weekday() == 4:  # Friday
            return d
    return None


def get_nq_contract_for_date(d):
    """
    Determine the correct NQ contract for a given date.
    Returns (expiry_year, expiry_month) for the front-month contract,
    accounting for 8-day roll before expiry.
    """
    if isinstance(d, datetime):
        d = d.date()

    # Generate expirations for current year +/- 1
    expirations = []
    for yr in range(d.year - 1, d.year + 2):
        for month, code in CONTRACT_MONTHS:
            exp = get_third_friday(yr, month)
            if exp:
                expirations.append(exp)
    expirations.sort()

    # Find the contract: use the next expiration, unless within roll window
    for i, exp in enumerate(expirations):
        roll_date = exp - timedelta(days=ROLL_DAYS_BEFORE_EXPIRY)
        if d < roll_date:
            return exp
        elif d < exp:
            # Within roll window — use NEXT contract
            if i + 1 < len(expirations):
                return expirations[i + 1]
            return exp

    return expirations[-1]


def fetch_day(ib, contract, trade_date, bar_size='1 secs'):
    """
    Fetch historical bars for a single trading day.

    For 1-second bars, IB limits duration to 1800 S (30 min) per request.
    RTH is 09:30-16:00 = 6.5 hours = 13 x 30-min chunks.
    We paginate backwards from 16:00 in 30-min windows.

    For larger bar sizes (1 min, 5 min), a single '1 D' request works.

    Returns:
        list of bar dicts or None on failure
    """
    all_records = []

    if bar_size in ('1 secs', '5 secs', '10 secs', '15 secs', '30 secs'):
        # Paginate in 30-minute chunks for sub-minute bars
        # RTH: 09:30 → 16:00 = 13 chunks of 30 min
        chunks = []
        end_hour, end_min = 16, 0
        for _ in range(14):  # slight over-fetch to be safe
            end_dt = datetime(trade_date.year, trade_date.month, trade_date.day,
                              end_hour, end_min, 0)
            chunks.append(end_dt)
            # Move back 30 minutes
            end_min -= 30
            if end_min < 0:
                end_min += 60
                end_hour -= 1
            if end_hour < 9 or (end_hour == 9 and end_min < 30):
                break

        # Fetch from earliest to latest (but IB needs endDateTime)
        # We fetched end times from 16:00 backwards; reverse to process chronologically
        chunks.reverse()

        # Track pacing errors via IB error callback
        pacing_error = [False]
        def on_error(reqId, errorCode, errorString, contract):
            if errorCode == 162:
                pacing_error[0] = True

        ib.errorEvent += on_error

        for ci, end_dt in enumerate(chunks):
            end_str = _to_utc_str(end_dt)
            bars = None

            for attempt in range(4):
                pacing_error[0] = False
                try:
                    bars = ib.reqHistoricalData(
                        contract,
                        endDateTime=end_str,
                        durationStr='1800 S',
                        barSizeSetting=bar_size,
                        whatToShow='TRADES',
                        useRTH=True,
                        formatDate=2,
                        timeout=60,
                    )
                except Exception as e:
                    log(f"    Exception on chunk {ci+1}: {e}")
                    bars = None

                # Check if we got a pacing error (IB returns empty + fires error 162)
                if pacing_error[0] or (bars is not None and len(bars) == 0):
                    wait = 20 * (attempt + 1)  # 20s, 40s, 60s, 80s
                    log(f"    Pacing violation on chunk {ci+1}/{len(chunks)}, retry {attempt+1}/4, waiting {wait}s...")
                    time.sleep(wait)
                    continue

                break  # Got data or genuine empty (non-RTH hours)

            if bars:
                for bar in bars:
                    all_records.append({
                        'date': str(bar.date),
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume,
                    })
                log(f"    chunk {ci+1}/{len(chunks)}: {len(bars)} bars")

            # IB pacing: max 6 requests per 2s, max 60 per 10min.
            # 11s between requests = safe margin for both rules.
            time.sleep(11)

        ib.errorEvent -= on_error

    else:
        # For 1-min bars and above, single request works
        end_str = _to_utc_str(datetime(trade_date.year, trade_date.month, trade_date.day, 16, 0, 0))
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr='1 D',
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=2,
                timeout=120,
            )
        except Exception as e:
            log(f"  Error fetching {trade_date}: {e}")
            return None

        if bars:
            for bar in bars:
                all_records.append({
                    'date': str(bar.date),
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                })

    if not all_records:
        return None

    # Deduplicate by timestamp (overlapping chunks)
    seen = set()
    deduped = []
    for r in all_records:
        if r['date'] not in seen:
            seen.add(r['date'])
            deduped.append(r)

    # Sort chronologically
    deduped.sort(key=lambda x: x['date'])
    return deduped


def is_trading_day(d):
    """Check if a date is a weekday (rough filter, doesn't handle holidays)."""
    return d.weekday() < 5


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch NQ 1-second bars from IB")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--out-dir", required=True, help="Output directory for parquet files")
    parser.add_argument("--bar-size", default="1 secs", help="Bar size (default: '1 secs')")
    parser.add_argument("--ib-host", default="127.0.0.1")
    parser.add_argument("--ib-port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=20)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y%m%d").date()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    log(f"Fetching NQ {args.bar_size} bars: {start} → {end}")
    log(f"Output: {out_dir}")

    # Connect to IB
    ib = IB()
    ib.connect(args.ib_host, args.ib_port, clientId=args.client_id, timeout=15)
    log(f"Connected to IB at {args.ib_host}:{args.ib_port}")

    # Initial cooldown to clear any prior pacing penalties
    log("Waiting 15s for pacing cooldown...")
    time.sleep(15)

    current_date = start
    days_fetched = 0
    days_skipped = 0
    current_contract = None
    current_expiry = None

    while current_date <= end:
        if not is_trading_day(current_date):
            current_date += timedelta(days=1)
            continue

        # Check if already fetched
        bar_label = args.bar_size.replace(' ', '')
        out_file = os.path.join(out_dir, f"nq_{bar_label}_{current_date.strftime('%Y%m%d')}.parquet")
        if os.path.exists(out_file):
            log(f"  {current_date} — already cached, skipping")
            days_skipped += 1
            current_date += timedelta(days=1)
            continue

        # Resolve contract if needed
        expiry = get_nq_contract_for_date(current_date)
        if expiry != current_expiry:
            current_expiry = expiry
            contract = Future(
                symbol='NQ',
                lastTradeDateOrContractMonth=expiry.strftime('%Y%m%d'),
                exchange='CME',
                currency='USD',
            )
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                log(f"  Failed to qualify NQ contract for expiry {expiry}")
                current_date += timedelta(days=1)
                continue
            current_contract = contract
            log(f"  Contract: NQ {expiry.strftime('%b %Y')} (conId={contract.conId})")

        # Fetch (pacing is handled inside fetch_day for chunked requests)
        log(f"  Fetching {current_date}...")
        records = fetch_day(ib, current_contract, current_date, args.bar_size)

        if records:
            df = pd.DataFrame(records)
            df.to_parquet(out_file, index=False)
            days_fetched += 1
            log(f"  {current_date} — {len(records)} bars → {os.path.basename(out_file)}")
        else:
            log(f"  {current_date} — no data (holiday?)")

        # Small pause between days (main pacing is per-chunk inside fetch_day)
        time.sleep(1)
        current_date += timedelta(days=1)

    ib.disconnect()
    log(f"\nDone. Fetched {days_fetched} days, skipped {days_skipped} cached.")


if __name__ == "__main__":
    main()
