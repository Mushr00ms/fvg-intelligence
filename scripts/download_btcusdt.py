"""Download BTCUSDT perpetual aggTrades from Binance (Jan 2020 → present).

Uses monthly archives for completed months, daily for the current partial month.
Downloads to Linux FS for speed, with parallel workers.
"""

import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from logic.utils.binance_downloader import BinanceAggTradesDownloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Use Linux FS — 10x faster than /mnt/c/ on WSL2
RAW_DIR = "/home/cr0wn/binance_data/raw"
WORKERS = 4

dl = BinanceAggTradesDownloader(symbol="BTCUSDT", raw_dir=RAW_DIR)

today = date.today()
last_complete_month_end = today.replace(day=1) - timedelta(days=1)

# Build list of monthly periods
monthly_periods = dl._month_range(date(2020, 1, 1), last_complete_month_end)

# Build list of daily periods for current partial month
current_month_start = today.replace(day=1)
yesterday = today - timedelta(days=1)
daily_periods = dl._day_range(current_month_start, yesterday) if current_month_start <= yesterday else []


def download_one(period_date, granularity):
    """Download, verify, and extract a single archive. Returns (date, ok, msg)."""
    zip_url, checksum_url = dl._build_urls(period_date, granularity)
    zip_name = zip_url.rsplit("/", 1)[-1]
    zip_path = dl.raw_dir / zip_name

    if zip_path.exists():
        csv_path = dl._csv_path_for_zip(zip_path)
        if csv_path and csv_path.exists():
            return (period_date, "skip", zip_name)
        try:
            dl._extract_csv(zip_path)
            return (period_date, "skip", zip_name)
        except Exception:
            pass

    ok = dl._download_file(zip_url, zip_path)
    if not ok:
        return (period_date, "fail", f"download failed: {zip_name}")

    if not dl._verify_checksum(zip_path, checksum_url):
        return (period_date, "fail", f"checksum failed: {zip_name}")

    try:
        dl._extract_csv(zip_path)
    except Exception as e:
        return (period_date, "fail", f"extract failed: {zip_name}: {e}")

    return (period_date, "ok", zip_name)


total = len(monthly_periods) + len(daily_periods)
done = 0
ok_count = 0
skip_count = 0
fail_count = 0

print(f"=== Downloading {len(monthly_periods)} monthly + {len(daily_periods)} daily files "
      f"with {WORKERS} workers → {RAW_DIR} ===\n")

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {}
    for p in monthly_periods:
        futures[pool.submit(download_one, p, "monthly")] = p
    for p in daily_periods:
        futures[pool.submit(download_one, p, "daily")] = p

    for fut in as_completed(futures):
        done += 1
        period_date, status, msg = fut.result()
        tag = period_date.strftime("%Y-%m")
        if status == "ok":
            ok_count += 1
            logger.info("[%d/%d] OK  %s", done, total, msg)
        elif status == "skip":
            skip_count += 1
            logger.info("[%d/%d] SKIP %s", done, total, msg)
        else:
            fail_count += 1
            logger.error("[%d/%d] FAIL %s", done, total, msg)

print(f"\n=== Done: {ok_count} downloaded, {skip_count} skipped, {fail_count} failed ===")
print(f"Data in: {RAW_DIR}")
