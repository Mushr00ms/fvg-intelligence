#!/usr/bin/env python3
"""Convert Binance SOLUSDT monthly kline CSVs → per-day parquets.

Input:  /home/cr0wn/binance_data/sol_klines/{1m,5m}/SOLUSDT-{1m,5m}-YYYY-MM.csv
Output: /home/cr0wn/binance_data/sol_resampled/{1m,5m,15m,1h}/solusdt_{tf}_YYYYMMDD.parquet

Binance kline CSV columns (no header in newer files, headered in older):
  open_time, open, high, low, close, volume, close_time, quote_volume,
  trades, taker_buy_base, taker_buy_quote, ignore
"""

import glob
import os
from pathlib import Path

import pandas as pd

IN_DIR = Path("/home/cr0wn/binance_data/sol_klines")
OUT_DIR = Path("/home/cr0wn/binance_data/sol_resampled")

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def load_monthly_csv(path: Path) -> pd.DataFrame:
    # Some Binance files have a header row, some don't. Detect.
    with open(path, "r") as f:
        first = f.readline().strip()
    has_header = first.startswith("open_time")
    df = pd.read_csv(
        path,
        header=0 if has_header else None,
        names=None if has_header else KLINE_COLS,
    )
    # open_time may be ms or us depending on month
    ot = df["open_time"].iloc[0]
    unit = "us" if ot > 1e15 else "ms"
    df["date"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def resample_higher(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df_1m.set_index("date")
    out = pd.DataFrame({
        "open": g["open"].resample(rule, label="left", closed="left").first(),
        "high": g["high"].resample(rule, label="left", closed="left").max(),
        "low":  g["low"].resample(rule, label="left", closed="left").min(),
        "close": g["close"].resample(rule, label="left", closed="left").last(),
        "volume": g["volume"].resample(rule, label="left", closed="left").sum(),
    }).dropna(subset=["open"]).reset_index()
    return out


def write_per_day(df: pd.DataFrame, tf_tag: str):
    out_sub = OUT_DIR / tf_tag
    out_sub.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["_day"] = df["date"].dt.strftime("%Y%m%d")
    written = 0
    for day, g in df.groupby("_day"):
        out_path = out_sub / f"solusdt_{tf_tag}_{day}.parquet"
        if out_path.exists():
            continue
        g.drop(columns=["_day"]).reset_index(drop=True).to_parquet(out_path)
        written += 1
    return written


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1m source → write 1m per-day, plus derive 15m and 1h
    for native in ["1m", "5m"]:
        csvs = sorted(glob.glob(str(IN_DIR / native / f"SOLUSDT-{native}-*.csv")))
        print(f"\n=== {native}: {len(csvs)} monthly CSVs ===")
        for csv in csvs:
            df = load_monthly_csv(Path(csv))
            n = write_per_day(df, native)
            print(f"  {Path(csv).name}: {len(df):,} bars, {n} new days")

            if native == "1m":
                # Derive 15m, 1h from 1m
                for rule, tag in [("15min", "15m"), ("1h", "1h")]:
                    higher = resample_higher(df, rule)
                    write_per_day(higher, tag)

    print("\nDone. Counts per timeframe:")
    for tf in ["1m", "5m", "15m", "1h"]:
        n = len(list((OUT_DIR / tf).glob("*.parquet")))
        print(f"  {tf:>4}: {n} day-parquets")


if __name__ == "__main__":
    main()
