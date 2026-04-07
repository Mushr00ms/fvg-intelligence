#!/usr/bin/env python3
"""Download Binance SOLUSDT USDⓈ-M futures klines (1m + 5m), Jan 2021 – Dec 2025."""

import hashlib
import os
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SYMBOL = "SOLUSDT"
BASE = f"https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}"
OUT_DIR = Path("/home/cr0wn/binance_data/sol_klines")
START_YEAR = 2021
END_YEAR = 2025


def get_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def download_klines(interval):
    out = OUT_DIR / interval
    out.mkdir(parents=True, exist_ok=True)
    session = get_session()
    downloaded = skipped = failed = 0

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            tag = f"{year}-{month:02d}"
            fname = f"{SYMBOL}-{interval}-{tag}.zip"
            url = f"{BASE}/{interval}/{fname}"
            dest = out / fname
            csv_dest = out / fname.replace(".zip", ".csv")

            if csv_dest.exists():
                skipped += 1
                continue

            try:
                resp = session.get(url, stream=True, timeout=60)
                if resp.status_code == 404:
                    print(f"  {fname} 404")
                    continue
                resp.raise_for_status()

                tmp = dest.with_suffix(".tmp")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                os.replace(str(tmp), str(dest))

                chk_resp = session.get(url + ".CHECKSUM", timeout=15)
                if chk_resp.status_code == 200:
                    expected = chk_resp.text.strip().split()[0].lower()
                    h = hashlib.sha256()
                    with open(dest, "rb") as f:
                        while True:
                            c = f.read(8192)
                            if not c:
                                break
                            h.update(c)
                    if h.hexdigest() != expected:
                        print(f"  CHECKSUM FAIL: {fname}")
                        failed += 1
                        continue

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
        csvs = list((OUT_DIR / interval).glob("*.csv"))
        print(f"  {len(csvs)} CSVs available")


if __name__ == "__main__":
    main()
