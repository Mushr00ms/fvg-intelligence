#!/usr/bin/env python3
"""Download Binance official OHLCV klines for BTCUSDT (1m + 5m) and audit against our resampled data."""

import hashlib
import os
import sys
import zipfile
from datetime import date
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT"
OUT_DIR = Path("/home/cr0wn/binance_data/official_klines")

def get_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def download_klines(interval, start_year=2020, end_year=2025):
    """Download monthly kline ZIPs for given interval."""
    out = OUT_DIR / interval
    out.mkdir(parents=True, exist_ok=True)
    session = get_session()

    downloaded = 0
    skipped = 0
    failed = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == 2025 and month > 10:
                break
            tag = f"{year}-{month:02d}"
            fname = f"BTCUSDT-{interval}-{tag}.zip"
            url = f"{BASE}/{interval}/{fname}"
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
    for interval in ["1m", "5m"]:
        print(f"\n=== Downloading {interval} klines ===")
        d, s, f = download_klines(interval)
        print(f"  {d} downloaded, {s} skipped, {f} failed")

        # Count CSVs
        csvs = list((OUT_DIR / interval).glob("*.csv"))
        print(f"  {len(csvs)} CSVs available")


if __name__ == "__main__":
    main()
