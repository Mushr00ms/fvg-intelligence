"""
main.py — CLI entry point for the standalone crypto bot.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from crypto_bot.config import default_config_path, load_config
from crypto_bot.engine import CryptoBotEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone BTC Binance Futures bot")
    parser.add_argument("--config", default=None, help="Path to crypto_bot_config.json")
    parser.add_argument("--live", action="store_true", help="Run live instead of dry-run")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config or default_config_path()
    config = load_config(config_path if os.path.exists(config_path) else None)
    if args.live:
        config.execution_mode = "live"

    print("\n" + "=" * 56)
    print(f"  Crypto Bot — {config.execution_mode.upper()}")
    print(f"  Symbol:   {config.symbol}")
    print(f"  Strategy: {config.strategy_path}")
    print(f"  Leverage: {config.leverage}x")
    print(f"  Market:   {config.base_url}")
    print("=" * 56 + "\n")

    engine = CryptoBotEngine(config)
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
