"""
send_crypto_daily_reconciliation.py - Manually render/send the crypto bot daily report.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from bot.alerts.telegram import TelegramAlerter
from bot.bot_logging.bot_logger import BotLogger
from crypto_bot.config import default_config_path, load_config
from crypto_bot.daily_report import build_daily_reconciliation_report
from crypto_bot.state_store import StateStore


def parse_args():
    parser = argparse.ArgumentParser(description="Send the crypto bot daily reconciliation to Telegram")
    parser.add_argument("--config", default=None, help="Path to crypto_bot_config.json")
    parser.add_argument("--print-only", action="store_true", help="Render the report but do not send it")
    return parser.parse_args()


async def main():
    args = parse_args()
    config_path = args.config or default_config_path()
    config = load_config(config_path if os.path.exists(config_path) else None)

    store = StateStore(config.state_dir)
    state = store.load()
    if state is None:
        raise SystemExit(f"No runtime state found in {config.state_dir}")

    report = build_daily_reconciliation_report(
        state,
        mode=config.execution_mode,
        report_tz=config.daily_reset_timezone,
    )
    print(report)

    if args.print_only:
        return

    logger = BotLogger(config.log_dir)
    telegram = TelegramAlerter(config.telegram_bot_token, config.telegram_chat_id, logger)
    if not telegram.enabled:
        raise SystemExit("Telegram is not configured. Set telegram_enabled/bot_token/chat_id in crypto_bot_config.json.")

    await telegram.alert_reconciliation(report)
    logger.log(
        "crypto_daily_reconciliation_manual_sent",
        date=state.day,
        symbol=state.symbol,
        strategy_id=state.strategy_id,
    )
    logger.close()


if __name__ == "__main__":
    asyncio.run(main())
