"""
main.py — CLI entry point for the FVG trading bot.

Usage:
    python -m bot.main                          # Paper + dry run (safe default)
    python -m bot.main --live                   # Live (port 7496)
    python -m bot.main --no-dry-run             # Paper, real orders
    python -m bot.main --config bot_config.json # Custom config file
"""

import argparse
import asyncio
import signal
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from bot.bot_config import load_bot_config, default_config_path
from bot.core.engine import BotEngine


def parse_args():
    parser = argparse.ArgumentParser(description="FVG Intelligence Trading Bot")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to bot_config.json (default: bot/bot_config.json)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live trading (port 7496). Default is paper (7497)."
    )
    parser.add_argument(
        "--no-dry-run", action="store_true",
        help="Disable dry run mode — actually place orders."
    )
    parser.add_argument(
        "--balance", type=float, default=None,
        help="Override starting balance (default: query from IB or $76,000)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config_path = args.config or default_config_path()
    config = load_bot_config(config_path if os.path.exists(config_path) else None)

    # Apply CLI overrides
    if args.live:
        config.paper_mode = False
        config.ib_port = 7496
    if args.no_dry_run:
        config.dry_run = False

    # Safety banner
    mode = "LIVE" if not config.paper_mode else "PAPER"
    dry = " (DRY RUN)" if config.dry_run else ""
    print(f"\n{'='*50}")
    print(f"  FVG Intelligence Bot — {mode}{dry}")
    print(f"  IB: {config.ib_host}:{config.ib_port}")
    print(f"  Strategy dir: {config.strategy_dir}")
    print(f"{'='*50}\n")

    if not config.paper_mode and not config.dry_run:
        print("  *** LIVE TRADING MODE — REAL MONEY AT RISK ***")
        print()

    # Run the engine
    engine = BotEngine(config)

    # Handle signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler(sig, frame):
        print("\nShutdown signal received...")
        engine._shutdown = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(engine.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
