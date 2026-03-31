"""Resample BTCUSDT aggTrades → 1min, 5min, 15min, 1h, 4h candles with integrity checks.

Processes monthly/daily CSVs in parallel. Each CSV is loaded once, split by day,
then resampled to all 5 timeframes with integrity verification.
"""

import logging
import sys
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(process)d] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("/home/cr0wn/binance_data/raw")
OUT_DIR = Path("/home/cr0wn/binance_data/resampled")
TIMEFRAMES = ["1min", "5min", "15min", "1h", "4h"]
WORKERS = 4


def process_csv(csv_path_str: str) -> dict:
    """Process a single CSV file: load, split by day, resample to all timeframes.

    Returns summary dict with counts and any errors.
    """
    import pandas as pd
    from logic.utils.binance_resampler import BinanceAggTradesResampler, BinanceResamplerIntegrityChecker

    csv_path = Path(csv_path_str)
    rs = BinanceAggTradesResampler(raw_dir=str(RAW_DIR), output_dir=str(OUT_DIR))
    checker = BinanceResamplerIntegrityChecker()

    result = {
        "file": csv_path.name,
        "days": 0,
        "candles": {tf: 0 for tf in TIMEFRAMES},
        "integrity_ok": 0,
        "integrity_fail": 0,
        "errors": [],
    }

    try:
        # Load trades, split by day
        day_frames = {}
        for chunk in rs.load_aggtrades_chunked(csv_path):
            chunk["trade_date"] = chunk["timestamp"].dt.date
            for day, group in chunk.groupby("trade_date"):
                clean = group.drop(columns=["trade_date"])
                if day in day_frames:
                    day_frames[day] = pd.concat([day_frames[day], clean], ignore_index=True)
                else:
                    day_frames[day] = clean.reset_index(drop=True)

        for day_date, trades in sorted(day_frames.items()):
            if trades.empty:
                continue

            result["days"] += 1
            date_str = day_date.strftime("%Y%m%d")
            day_ok = True

            for tf in TIMEFRAMES:
                tf_tag = tf.replace("min", "m")
                out_path = OUT_DIR / tf_tag / f"btcusdt_{tf_tag}_{date_str}.parquet"

                if out_path.exists():
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)

                candles = rs.resample_to_ohlcv(trades, tf)
                if candles.empty:
                    continue

                result["candles"][tf] += len(candles)

                # Integrity checks (volume + price bounds + first/last)
                ir = checker.run_all_checks(trades, candles, tf, duration_hours=24.0)
                if ir.all_passed:
                    result["integrity_ok"] += 1
                else:
                    result["integrity_fail"] += 1
                    day_ok = False
                    result["errors"].append(f"{day_date} {tf}: {ir.summary}")

                # Write parquet
                candles.to_parquet(out_path)

            if not day_ok:
                logger.warning("%s: integrity issues on %s", csv_path.name, day_date)

    except Exception as e:
        result["errors"].append(f"{csv_path.name}: {e}")
        logger.error("FAILED %s: %s", csv_path.name, e)

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tf in TIMEFRAMES:
        tf_tag = tf.replace("min", "m")
        (OUT_DIR / tf_tag).mkdir(parents=True, exist_ok=True)

    # Gather all CSVs
    csvs = sorted(glob.glob(str(RAW_DIR / "BTCUSDT-aggTrades-*.csv")))
    print(f"=== Resampling {len(csvs)} CSVs → {TIMEFRAMES} with {WORKERS} workers ===")
    print(f"Output: {OUT_DIR}\n")

    t0 = time.time()
    total_days = 0
    total_candles = {tf: 0 for tf in TIMEFRAMES}
    total_ok = 0
    total_fail = 0
    all_errors = []

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_csv, csv): csv for csv in csvs}

        done = 0
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            total_days += r["days"]
            total_ok += r["integrity_ok"]
            total_fail += r["integrity_fail"]
            all_errors.extend(r["errors"])
            for tf in TIMEFRAMES:
                total_candles[tf] += r["candles"][tf]

            logger.info(
                "[%d/%d] %s — %d days, %s",
                done, len(csvs), r["file"], r["days"],
                "ALL OK" if not r["errors"] else f"{len(r['errors'])} issues",
            )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Completed in {elapsed/60:.1f} minutes")
    print(f"Days processed: {total_days}")
    print(f"Integrity: {total_ok} passed, {total_fail} failed")
    print(f"\nCandles generated:")
    for tf in TIMEFRAMES:
        tf_tag = tf.replace("min", "m")
        n_files = len(glob.glob(str(OUT_DIR / tf_tag / "*.parquet")))
        print(f"  {tf:>5s}: {total_candles[tf]:>10,} candles across {n_files} files")

    if all_errors:
        print(f"\n{len(all_errors)} errors:")
        for e in all_errors[:20]:
            print(f"  - {e}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")

    print(f"\nDisk usage:")
    os.system(f"du -sh {OUT_DIR}/*/")


if __name__ == "__main__":
    main()
