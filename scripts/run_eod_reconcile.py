#!/usr/bin/env python3
"""
run_eod_reconcile.py — Standalone EOD reconciliation runner.

Downloads today's 1-second bars from IB, runs the backtester against the
active strategy, compares to live trades in the DB, and sends the
reconciliation report via Telegram.

Usage:
    python scripts/run_eod_reconcile.py              # Today
    python scripts/run_eod_reconcile.py 2026-03-28   # Specific date
    python scripts/run_eod_reconcile.py --skip-download  # Skip IB download (data must exist)
"""

import argparse
import asyncio
import os
import sys
import time as _time
from datetime import datetime, date as _date

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytz

from bot.bot_config import load_bot_config, default_config_path
from bot.bot_logging.bot_logger import BotLogger
from bot.db import TradeDB
from bot.alerts.telegram import TelegramAlerter
from bot.state.state_manager import StateManager
from bot.strategy.strategy_loader import StrategyLoader
from bot.backtest.backtester import load_1s_bars, run_backtest
from bot.backtest.eod_reconciler import (
    match_trades, format_telegram_report, build_backtest_config,
    build_weekly_summary, result_to_db_kwargs, ReconciliationResult,
    validate_fills, has_bad_fills,
)

ET = pytz.timezone("US/Eastern")


def download_1s_bars(config, date_str, data_dir):
    """Download 1-second bars from IB for the given date. Returns parquet path or None."""
    out_file = os.path.join(data_dir, f"nq_1secs_{date_str}.parquet")

    if os.path.exists(out_file):
        print(f"  [cached] {out_file}")
        return out_file

    import pandas as pd
    from ib_async import IB, Future
    from bot.bridge.ib_data_fetcher import get_nq_contract_for_date

    _ET = pytz.timezone("US/Eastern")
    _UTC = pytz.utc

    def _to_utc_str(naive_et_dt):
        return _ET.localize(naive_et_dt).astimezone(_UTC).strftime("%Y%m%d-%H:%M:%S")

    trade_date = _date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    exp_date = get_nq_contract_for_date(trade_date)
    contract = Future("NQ", exchange="CME",
                      lastTradeDateOrContractMonth=exp_date.strftime("%Y%m"))

    ib = IB()
    print(f"  Connecting to IB ({config.ib_host}:{config.ib_port}, clientId=20)...")
    ib.connect(config.ib_host, config.ib_port, clientId=20, timeout=15)

    try:
        ib.qualifyContracts(contract)
        print(f"  Contract: {contract.localSymbol or contract}")

        # Paginate in 30-min chunks (IB limit for 1-sec bars)
        all_records = []
        chunks = []
        end_hour, end_min = 16, 0
        for _ in range(14):
            end_dt = datetime(trade_date.year, trade_date.month,
                              trade_date.day, end_hour, end_min, 0)
            chunks.append(end_dt)
            end_min -= 30
            if end_min < 0:
                end_min += 60
                end_hour -= 1
            if end_hour < 9 or (end_hour == 9 and end_min < 30):
                break
        chunks.reverse()

        pacing_error = [False]

        def on_error(reqId, errorCode, errorString, contract):
            if errorCode == 162:
                pacing_error[0] = True
        ib.errorEvent += on_error

        for ci, end_dt in enumerate(chunks):
            end_str = _to_utc_str(end_dt)
            print(f"  Chunk {ci+1}/{len(chunks)}: ending {end_dt.strftime('%H:%M')} ET...",
                  end="", flush=True)
            bars = None
            for attempt in range(4):
                pacing_error[0] = False
                try:
                    bars = ib.reqHistoricalData(
                        contract, endDateTime=end_str,
                        durationStr="1800 S", barSizeSetting="1 secs",
                        whatToShow="TRADES", useRTH=True,
                        formatDate=2, timeout=60)
                except Exception:
                    bars = None
                if pacing_error[0] or (bars is not None and len(bars) == 0):
                    print(f" pacing({attempt+1})", end="", flush=True)
                    _time.sleep(20 * (attempt + 1))
                    continue
                break
            if bars:
                for bar in bars:
                    all_records.append({
                        "date": str(bar.date), "open": bar.open,
                        "high": bar.high, "low": bar.low,
                        "close": bar.close, "volume": bar.volume,
                    })
                print(f" {len(bars)} bars")
            else:
                print(" no bars")
            _time.sleep(11)  # IB pacing

        ib.errorEvent -= on_error

        if not all_records:
            print("  No bars downloaded!")
            return None

        # Deduplicate + sort
        seen = set()
        deduped = []
        for r in all_records:
            if r["date"] not in seen:
                seen.add(r["date"])
                deduped.append(r)
        deduped.sort(key=lambda x: x["date"])

        df = pd.DataFrame(deduped)
        df.to_parquet(out_file, index=False)
        print(f"  Saved {len(deduped)} bars -> {out_file}")
        return out_file

    finally:
        ib.disconnect()


def fetch_fill_ticks(config, date_str, live_trades):
    """Fetch tick data around each live trade's fill times for validation."""
    from bot.bridge.ib_data_fetcher import get_nq_contract_for_date
    from ib_async import IB, Future
    from dateutil import parser as dtparser

    trade_date = _date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    WINDOW_SECS = 10

    windows = []
    for t in live_trades:
        for fill_type, time_key in [("entry", "entry_time"), ("exit", "exit_time")]:
            ft = t.get(time_key)
            if not ft:
                continue
            windows.append((t["group_id"], fill_type, ft))

    if not windows:
        return {}

    _ET = pytz.timezone("US/Eastern")
    _UTC = pytz.utc

    exp_date = get_nq_contract_for_date(trade_date)
    contract = Future("NQ", exchange="CME",
                      lastTradeDateOrContractMonth=exp_date.strftime("%Y%m"))

    ib = IB()
    print(f"  Connecting to IB for tick validation (clientId=21)...")
    ib.connect(config.ib_host, config.ib_port, clientId=21, timeout=15)

    result = {}
    try:
        ib.qualifyContracts(contract)

        for gid, fill_type, fill_time_str in windows:
            fill_dt = dtparser.parse(fill_time_str)
            if fill_dt.tzinfo is None:
                fill_dt = _ET.localize(fill_dt)
            start_dt = fill_dt - __import__("datetime").timedelta(seconds=WINDOW_SECS)
            end_dt = fill_dt + __import__("datetime").timedelta(seconds=WINDOW_SECS)

            start_utc = start_dt.astimezone(_UTC).strftime("%Y%m%d-%H:%M:%S")
            end_utc = end_dt.astimezone(_UTC).strftime("%Y%m%d-%H:%M:%S")

            try:
                ticks = ib.reqHistoricalTicks(
                    contract, startDateTime=start_utc, endDateTime=end_utc,
                    numberOfTicks=1000, whatToShow="TRADES", useRth=True)
            except Exception:
                ticks = []

            filtered = [
                {"price": tk.price, "size": tk.size, "time_utc": str(tk.time)}
                for tk in ticks
            ]
            result[(gid, fill_type)] = filtered
            _time.sleep(2)

        return result
    finally:
        ib.disconnect()


async def run_reconciliation(target_date=None, skip_download=False):
    """Run full EOD reconciliation for the given date."""

    # --- Setup ---
    config = load_bot_config(default_config_path())
    log_dir = config.log_dir or os.path.join(ROOT, "bot", "logs")
    state_dir = config.state_dir or os.path.join(ROOT, "bot", "bot_state")
    strategy_dir = config.strategy_dir or os.path.join(ROOT, "logic", "strategies")
    data_dir = os.path.join(ROOT, "bot", "data")
    os.makedirs(data_dir, exist_ok=True)

    logger = BotLogger(log_dir)
    db = TradeDB()
    telegram = TelegramAlerter(config.telegram_bot_token, config.telegram_chat_id,
                               logger, db)
    state_mgr = StateManager(state_dir, logger)
    strategy = StrategyLoader(strategy_dir, logger)
    strategy.load()

    # Determine date
    if target_date:
        today = target_date  # "2026-03-31"
    else:
        today = datetime.now(ET).strftime("%Y-%m-%d")

    today_fmt = today.replace("-", "")

    print(f"\n{'='*50}")
    print(f"EOD Reconciliation — {today}")
    print(f"{'='*50}")

    # Load state to get start_balance
    daily_state = state_mgr.load()
    if daily_state and daily_state.date == today:
        start_balance = daily_state.start_balance
        kill_switch = daily_state.kill_switch_active
        print(f"State: balance=${start_balance:,.2f}, trades={daily_state.trade_count}, "
              f"kill_switch={kill_switch}")
    else:
        # Fallback: query DB for last known balance
        rows = db.query(
            "SELECT end_balance FROM daily_stats ORDER BY trade_date DESC LIMIT 1")
        start_balance = rows[0]["end_balance"] if rows else 100000
        kill_switch = False
        print(f"State: no state for today, using last balance=${start_balance:,.2f}")

    # --- Step 1: Download 1-second bars ---
    print(f"\n[1/5] Downloading 1-second bars...")
    if skip_download:
        data_file = os.path.join(data_dir, f"nq_1secs_{today_fmt}.parquet")
        if not os.path.exists(data_file):
            err = f"--skip-download but {data_file} not found"
            print(f"  ERROR: {err}")
            result = ReconciliationResult(date=today, live_count=0,
                                          backtest_count=0, matched_count=0, error=err)
            db.insert_reconciliation(**result_to_db_kwargs(result))
            if telegram.enabled:
                await telegram.alert_reconciliation(format_telegram_report(result))
            return
    else:
        data_file = None
        for attempt in range(3):
            try:
                data_file = download_1s_bars(config, today_fmt, data_dir)
                if data_file:
                    break
                print(f"  Retry {attempt+1}/3 (no bars)...")
                await asyncio.sleep(60)
            except Exception as e:
                print(f"  Retry {attempt+1}/3: {e}")
                await asyncio.sleep(60)

        if data_file is None:
            err = "1-second bar download failed after 3 attempts"
            print(f"  ERROR: {err}")
            result = ReconciliationResult(date=today, live_count=0,
                                          backtest_count=0, matched_count=0, error=err)
            db.insert_reconciliation(**result_to_db_kwargs(result))
            if telegram.enabled:
                await telegram.alert_reconciliation(format_telegram_report(result))
            logger.close()
            return

    # --- Step 2: Run backtester ---
    print(f"\n[2/5] Running backtester...")
    try:
        bt_config = build_backtest_config(config, strategy.strategy, start_balance)
        df = load_1s_bars(data_dir, start_date=today_fmt, end_date=today_fmt)
        bt_trades, _ = run_backtest(df, strategy.strategy, bt_config)
        print(f"  Backtest produced {len(bt_trades)} trades")
    except Exception as e:
        err = f"Backtest failed: {e}"
        print(f"  ERROR: {err}")
        result = ReconciliationResult(date=today, live_count=0,
                                      backtest_count=0, matched_count=0, error=err)
        db.insert_reconciliation(**result_to_db_kwargs(result))
        if telegram.enabled:
            await telegram.alert_reconciliation(format_telegram_report(result))
        logger.close()
        return

    # --- Step 3: Load live trades ---
    print(f"\n[3/5] Loading live trades from DB...")
    live_trades = db.get_trades(date=today, limit=999)
    live_trades = [t for t in live_trades if t.get("exit_reason")]
    print(f"  {len(live_trades)} closed live trades")

    # --- Step 4: Compare ---
    print(f"\n[4/5] Comparing live vs backtest...")
    result = match_trades(live_trades, bt_trades)
    result.date = today
    result.kill_switch_active = kill_switch

    print(f"  Matched: {result.matched_count}")
    print(f"  Divergences: {len(result.divergences)}")
    print(f"  Live P&L: ${result.live_net_pnl:+,.2f}")
    print(f"  Backtest P&L: ${result.backtest_net_pnl:+,.2f}")

    # --- Step 4b: Tick-validate fills ---
    fills_garbage = False
    if live_trades and not skip_download:
        try:
            print(f"  Validating fills against tick data...")
            ticks_by_window = fetch_fill_ticks(config, today_fmt, live_trades)
            if ticks_by_window:
                checks = validate_fills(live_trades, ticks_by_window)
                fills_garbage = has_bad_fills(checks)
                bad_count = sum(1 for c in checks if not c.valid)
                print(f"  Tick validation: {len(checks)} checked, {bad_count} bad")
        except Exception as e:
            print(f"  Tick validation skipped: {e}")

    # --- Step 5: Save + send ---
    print(f"\n[5/5] Saving results and sending Telegram report...")
    db.insert_reconciliation(**result_to_db_kwargs(result))

    # Weekly summary (Fridays only)
    balance = start_balance + (daily_state.realized_pnl if daily_state else 0)
    weekly_html = build_weekly_summary(db, today, balance)

    if telegram.enabled:
        msg = format_telegram_report(result, weekly_html, fills_garbage=fills_garbage)
        await telegram.alert_reconciliation(msg)
        print(f"  Telegram report sent!")
    else:
        print(f"  WARNING: Telegram not configured, printing report:\n")
        msg = format_telegram_report(result, weekly_html, fills_garbage=fills_garbage)
        # Strip HTML tags for console
        import re
        print(re.sub(r"<[^>]+>", "", msg))

    logger.log("eod_reconcile_done",
               date=today,
               live_trades=result.live_count,
               backtest_trades=result.backtest_count,
               matched=result.matched_count,
               divergences=len(result.divergences))

    print(f"\nDone.")
    logger.close()


def main():
    parser = argparse.ArgumentParser(description="Run EOD reconciliation standalone")
    parser.add_argument("date", nargs="?", default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip IB data download (parquet must exist)")
    args = parser.parse_args()

    asyncio.run(run_reconciliation(
        target_date=args.date,
        skip_download=args.skip_download,
    ))


if __name__ == "__main__":
    main()
