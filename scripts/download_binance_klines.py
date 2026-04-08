#!/usr/bin/env python3
"""Download Binance official USD-M futures klines (1m + 5m)."""

from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from datetime import date, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "data" / "binance_official"

def get_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def download_klines(symbol: str, interval: str, out_dir: Path, start_year=2020, end_year=None):
    """Download monthly kline ZIPs for given interval."""
    base = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}"
    out = out_dir / symbol / interval
    out.mkdir(parents=True, exist_ok=True)
    session = get_session()
    last_complete_month = date.today().replace(day=1) - timedelta(days=1)
    last_complete_tag = f"{last_complete_month.year}-{last_complete_month.month:02d}"
    if end_year is None:
        end_year = last_complete_month.year

    downloaded = 0
    skipped = 0
    failed = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            tag = f"{year}-{month:02d}"
            if tag > last_complete_tag:
                break
            fname = f"{symbol}-{interval}-{tag}.zip"
            url = f"{base}/{interval}/{fname}"
            dest = out / fname
            csv_name = fname.replace(".zip", ".csv")
            csv_dest = out / csv_name

            if csv_dest.exists():
                skipped += 1
                continue

            try:
                resp = session.get(url, stream=True, timeout=60)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()

                tmp = dest.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                os.replace(str(tmp), str(dest))

                # Verify checksum
                chk_resp = session.get(url + ".CHECKSUM", timeout=15)
                if chk_resp.status_code == 200:
                    expected = chk_resp.text.strip().split()[0].lower()
                    h = hashlib.sha256()
                    with open(dest, "rb") as f:
                        while True:
                            c = f.read(8192)
                            if not c: break
                            h.update(c)
                    actual = h.hexdigest()
                    if actual != expected:
                        print(f"  CHECKSUM FAIL: {fname}")
                        failed += 1
                        continue

                # Extract
                with zipfile.ZipFile(dest, "r") as zf:
                    zf.extractall(out)
                downloaded += 1
                print(f"  {fname} OK")

            except Exception as e:
                print(f"  {fname} FAILED: {e}")
                failed += 1

    return downloaded, skipped, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--intervals", default="1m,5m", help="Comma-separated list, e.g. 1m,5m")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    intervals = [item.strip() for item in args.intervals.split(",") if item.strip()]
    for interval in ["1m", "5m"]:
        if interval not in intervals:
            continue
        print(f"\n=== Downloading {interval} klines ===")
        d, s, f = download_klines(args.symbol, interval, out_dir, start_year=args.start_year, end_year=args.end_year)
        print(f"  {d} downloaded, {s} skipped, {f} failed")

        # Count CSVs
        csvs = list((out_dir / args.symbol / interval).glob("*.csv"))
        print(f"  {len(csvs)} CSVs available")


if __name__ == "__main__":
    main()
