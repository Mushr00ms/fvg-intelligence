"""
ib_data_fetcher.py — Fetch historical bars or tick data from IB TWS.

Runs on Windows Python (via python.exe from WSL) to bypass Hyper-V isolation.
Fetches NQ futures data day-by-day, handles contract rolls, writes parquet.

Usage from WSL:
    # 1-second bars (existing)
    python.exe C:\\path\\to\\ib_data_fetcher.py \\
        --start 20260102 --end 20260322 \\
        --out-dir C:\\path\\to\\bot\\data \\
        --ib-port 7497

    # Historical ticks for a specific time window
    python.exe C:\\path\\to\\ib_data_fetcher.py \\
        --start 20260326 --end 20260326 \\
        --out-dir C:\\path\\to\\bot\\data \\
        --ticks --time-range 14:00-14:10 --what-to-show TRADES

The fetcher handles:
    - NQ quarterly contract rolls (H/M/U/Z, 8-day roll before expiry)
    - IB pacing limits (max 60 requests / 10 min, auto-throttle)
    - Resume capability (skips days already fetched)
    - Parquet output with consistent schema
    - Tick-by-tick historical data via reqHistoricalTicks
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


def _et_to_utc(naive_et_dt):
    """Convert a naive ET datetime to a UTC-aware datetime."""
    return _ET.localize(naive_et_dt).astimezone(_UTC)


def _utc_to_et_str(utc_dt):
    """Format a UTC datetime as an ET string for display."""
    return utc_dt.astimezone(_ET).strftime('%H:%M:%S.%f')[:-3]


def fetch_ticks(ib, contract, trade_date, start_time, end_time, what_to_show='TRADES'):
    """
    Fetch historical ticks for a time window using reqHistoricalTicks.

    IB returns max 1000 ticks per request. We paginate forward using
    startDateTime, advancing past the last received tick each iteration.

    All times are converted to UTC for the IB API call and stored as UTC
    in output (consistent with bar data). An 'time_et' column is included
    for readability.

    Args:
        ib: connected IB instance
        contract: qualified Future contract
        trade_date: date object
        start_time: tuple (hour, minute) in ET, e.g. (14, 0)
        end_time: tuple (hour, minute) in ET, e.g. (14, 10)
        what_to_show: 'TRADES' or 'BID_ASK'

    Returns:
        list of tick dicts or None
    """
    all_ticks = []
    seen_keys = set()

    # Build boundaries in UTC (user thinks in ET, IB API wants UTC)
    start_utc = _et_to_utc(datetime(
        trade_date.year, trade_date.month, trade_date.day,
        start_time[0], start_time[1], 0
    ))
    end_utc = _et_to_utc(datetime(
        trade_date.year, trade_date.month, trade_date.day,
        end_time[0], end_time[1], 0
    ))

    cursor_utc = start_utc
    page = 0

    while cursor_utc < end_utc:
        page += 1
        # Convert cursor to IB UTC dash-format string
        cursor_str = cursor_utc.strftime('%Y%m%d-%H:%M:%S')
        log(f"    Page {page}: requesting from {cursor_str} UTC ({_utc_to_et_str(cursor_utc)} ET)")

        try:
            ticks = ib.reqHistoricalTicks(
                contract,
                startDateTime=cursor_str,
                endDateTime='',
                numberOfTicks=1000,
                whatToShow=what_to_show,
                useRth=True,
            )
        except Exception as e:
            log(f"    Exception on tick page {page}: {e}")
            time.sleep(15)
            continue

        if not ticks:
            log(f"    Page {page}: no ticks returned, done.")
            break

        new_count = 0
        last_tick_utc = None

        for tick in ticks:
            # IB returns tick.time as UTC datetime (may be naive)
            tick_utc = tick.time
            if tick_utc.tzinfo is None:
                tick_utc = _UTC.localize(tick_utc)

            # Stop if past our end boundary
            if tick_utc >= end_utc:
                log(f"    Page {page}: reached end boundary at {_utc_to_et_str(tick_utc)} ET")
                break

            if what_to_show == 'TRADES':
                key = (str(tick_utc), tick.price, tick.size)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_ticks.append({
                    'time_utc': tick_utc.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'time_et': tick_utc.astimezone(_ET).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'price': tick.price,
                    'size': tick.size,
                    'exchange': getattr(tick, 'exchange', ''),
                    'conditions': getattr(tick, 'specialConditions', ''),
                })
            else:  # BID_ASK
                key = (str(tick_utc), tick.priceBid, tick.priceAsk)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                all_ticks.append({
                    'time_utc': tick_utc.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'time_et': tick_utc.astimezone(_ET).strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'price_bid': tick.priceBid,
                    'price_ask': tick.priceAsk,
                    'size_bid': tick.sizeBid,
                    'size_ask': tick.sizeAsk,
                })

            new_count += 1
            last_tick_utc = tick_utc

        log(f"    Page {page}: {len(ticks)} ticks received, {new_count} new (total: {len(all_ticks)})")

        if last_tick_utc is None:
            break

        # Advance cursor 1 second past last tick to avoid re-fetching
        cursor_utc = last_tick_utc + timedelta(seconds=1)

        # Check if last tick was past our end boundary
        if last_tick_utc >= end_utc:
            break

        # If IB returned fewer than 1000, we've exhausted the range
        if len(ticks) < 1000:
            break

        # Pacing: reqHistoricalTicks has same limits
        time.sleep(11)

    if not all_ticks:
        return None

    all_ticks.sort(key=lambda x: x['time_utc'])
    return all_ticks


def is_trading_day(d):
    """Check if a date is a weekday (rough filter, doesn't handle holidays)."""
    return d.weekday() < 5


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fetch NQ historical data from IB")
    parser.add_argument("--start", required=True, help="Start date YYYYMMDD")
    parser.add_argument("--end", required=True, help="End date YYYYMMDD")
    parser.add_argument("--out-dir", required=True, help="Output directory for parquet files")
    parser.add_argument("--bar-size", default="1 secs", help="Bar size (default: '1 secs')")
    parser.add_argument("--ticks", action="store_true", help="Fetch tick data instead of bars")
    parser.add_argument("--time-range", help="Time range in ET for ticks, e.g. '14:00-14:10'")
    parser.add_argument("--what-to-show", default="TRADES", choices=["TRADES", "BID_ASK"],
                        help="Tick data type (default: TRADES)")
    parser.add_argument("--ib-host", default="127.0.0.1")
    parser.add_argument("--ib-port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=20)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y%m%d").date()
    end = datetime.strptime(args.end, "%Y%m%d").date()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Parse time range for tick mode
    tick_start_time = (9, 30)   # default: RTH open
    tick_end_time = (16, 0)     # default: RTH close
    if args.ticks and args.time_range:
        parts = args.time_range.split('-')
        sh, sm = parts[0].split(':')
        eh, em = parts[1].split(':')
        tick_start_time = (int(sh), int(sm))
        tick_end_time = (int(eh), int(em))

    mode_label = f"ticks ({args.what_to_show})" if args.ticks else f"{args.bar_size} bars"
    log(f"Fetching NQ {mode_label}: {start} → {end}")
    if args.ticks:
        log(f"Time range (ET): {tick_start_time[0]:02d}:{tick_start_time[1]:02d} - {tick_end_time[0]:02d}:{tick_end_time[1]:02d}")
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

        # Build output filename
        if args.ticks:
            time_label = f"{tick_start_time[0]:02d}{tick_start_time[1]:02d}-{tick_end_time[0]:02d}{tick_end_time[1]:02d}"
            out_file = os.path.join(
                out_dir,
                f"nq_ticks_{args.what_to_show.lower()}_{current_date.strftime('%Y%m%d')}_{time_label}.parquet"
            )
            # Always re-fetch ticks (investigative, not bulk caching)
        else:
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
            # Try exact expiry date first, then month-only format
            resolved = False
            for date_fmt in [expiry.strftime('%Y%m%d'), expiry.strftime('%Y%m')]:
                new_contract = Future(
                    symbol='NQ',
                    lastTradeDateOrContractMonth=date_fmt,
                    exchange='CME',
                    currency='USD',
                )
                qualified = ib.qualifyContracts(new_contract)
                if qualified and new_contract.conId > 0:
                    current_expiry = expiry
                    current_contract = new_contract
                    log(f"  Contract: NQ {expiry.strftime('%b %Y')} (conId={new_contract.conId}, fmt={date_fmt})")
                    resolved = True
                    break

            if not resolved:
                if current_contract is not None:
                    log(f"  Failed to qualify NQ {expiry.strftime('%b %Y')}, staying on current contract")
                else:
                    # No current contract — try the previous expiry as fallback
                    prev_expiry = get_nq_contract_for_date(current_date - timedelta(days=30))
                    for date_fmt in [prev_expiry.strftime('%Y%m%d'), prev_expiry.strftime('%Y%m')]:
                        fallback = Future(
                            symbol='NQ',
                            lastTradeDateOrContractMonth=date_fmt,
                            exchange='CME',
                            currency='USD',
                        )
                        fb_qualified = ib.qualifyContracts(fallback)
                        if fb_qualified and fallback.conId > 0:
                            current_expiry = prev_expiry
                            current_contract = fallback
                            log(f"  Fallback to NQ {prev_expiry.strftime('%b %Y')} (conId={fallback.conId})")
                            resolved = True
                            break
                    if not resolved:
                        log(f"  Failed to qualify any NQ contract, skipping {current_date}")
                        current_date += timedelta(days=1)
                        continue

        log(f"  Fetching {current_date}...")

        if args.ticks:
            records = fetch_ticks(
                ib, current_contract, current_date,
                tick_start_time, tick_end_time, args.what_to_show,
            )
        else:
            records = fetch_day(ib, current_contract, current_date, args.bar_size)

        if records:
            df = pd.DataFrame(records)
            df.to_parquet(out_file, index=False)
            days_fetched += 1
            label = "ticks" if args.ticks else "bars"
            log(f"  {current_date} — {len(records)} {label} → {os.path.basename(out_file)}")
        else:
            log(f"  {current_date} — no data (holiday?)")

        time.sleep(1)
        current_date += timedelta(days=1)

    ib.disconnect()
    log(f"\nDone. Fetched {days_fetched} days, skipped {days_skipped} cached.")


if __name__ == "__main__":
    main()
