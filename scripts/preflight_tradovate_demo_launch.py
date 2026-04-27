"""Preflight checks for NQ launch: IBKR data + Tradovate demo execution.

This script does not place orders. It validates the effective config that
`bot.main` will use, loads the active strategy, optionally checks the IB socket,
and can verify Tradovate credentials before the Telegram manager starts.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

import pytz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.backtest.us_holidays import US_MARKET_HOLIDAYS, is_trading_day
from bot.bot_config import default_config_path, load_bot_config
from bot.execution.broker_factory import create_broker_adapter
from bot.strategy.strategy_loader import StrategyLoader

NY_TZ = pytz.timezone("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an IBKR-data/Tradovate-demo NQ bot launch config."
    )
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help="Path to bot config JSON.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Rejected for this launch: IBKR must stay in paper-data mode on port 7497.",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Apply the same order-placement override as bot.main --no-dry-run.",
    )
    parser.add_argument(
        "--allow-dry-run",
        action="store_true",
        help="Permit dry_run=true. By default, demo launch requires demo order placement.",
    )
    parser.add_argument(
        "--check-ib-socket",
        action="store_true",
        help="Open a TCP connection to the configured IBKR API endpoint.",
    )
    parser.add_argument(
        "--check-secrets",
        action="store_true",
        help="Load Tradovate credentials through SecretStore. May use SSM/network.",
    )
    return parser.parse_args()


def _apply_main_overrides(config, args: argparse.Namespace) -> None:
    if args.live:
        config.paper_mode = False
        config.ib_port = 7496
    if args.no_dry_run:
        config.dry_run = False


def _check_ib_socket(config) -> None:
    with socket.create_connection((config.ib_host, config.ib_port), timeout=3):
        pass


def main() -> int:
    args = parse_args()
    config_path = args.config
    config = load_bot_config(config_path if os.path.exists(config_path) else None)
    _apply_main_overrides(config, args)

    errors: list[str] = []
    warnings: list[str] = []
    checks: list[str] = []

    if config.execution_backend != "ib_data_tradovate_exec":
        errors.append(
            "execution_backend must be 'ib_data_tradovate_exec' for IBKR data "
            f"+ Tradovate execution; got {config.execution_backend!r}."
        )
    else:
        checks.append("backend routes market data to IBKR and execution to Tradovate")

    if config.tradovate_environment != "demo":
        errors.append(
            f"tradovate_environment must be 'demo' for this launch; got "
            f"{config.tradovate_environment!r}."
        )
    else:
        checks.append("Tradovate environment is demo")

    if config.dry_run and not args.allow_dry_run:
        errors.append(
            "dry_run is true. For a Tradovate demo execution launch, set "
            "dry_run=false or pass --no-dry-run so orders reach the demo account."
        )
    elif config.dry_run:
        warnings.append("dry_run=true: no Tradovate demo orders will be placed")
    else:
        checks.append("order placement is enabled for the Tradovate demo account")

    if args.live:
        errors.append(
            "Do not pass --live for this launch. IBKR must run in paper mode on port 7497."
        )
    elif not config.paper_mode or config.ib_port != 7497:
        errors.append(
            f"IBKR data must run in paper mode on port 7497; got "
            f"paper_mode={config.paper_mode} ib_port={config.ib_port}."
        )
    else:
        checks.append("IBKR market data is configured for paper mode on port 7497")

    now_et = datetime.now(NY_TZ)
    today_key = now_et.strftime("%Y%m%d")
    if not is_trading_day(today_key):
        errors.append(
            f"{now_et.date().isoformat()} ET is not a trading day: "
            f"{US_MARKET_HOLIDAYS.get(today_key, 'weekend')}."
        )
    else:
        checks.append(f"{now_et.date().isoformat()} ET is a trading day")

    try:
        strategy = StrategyLoader(config.strategy_dir)
        strategy.load()
        checks.append(
            f"active strategy {strategy.strategy_id!r} loaded with "
            f"{strategy.cell_count} enabled lookup cells"
        )
    except Exception as exc:
        errors.append(f"active strategy failed to load: {exc}")

    try:
        adapter = create_broker_adapter(config)
        checks.append(f"broker adapter instantiated: {adapter.__class__.__name__}")
    except Exception as exc:
        errors.append(f"broker adapter failed to instantiate: {exc}")

    if args.check_ib_socket:
        try:
            _check_ib_socket(config)
            checks.append(f"IBKR API socket reachable at {config.ib_host}:{config.ib_port}")
        except OSError as exc:
            errors.append(
                f"IBKR API socket not reachable at {config.ib_host}:{config.ib_port}: {exc}"
            )

    if args.check_secrets:
        try:
            from bot.secret_store import SecretStore

            secrets = SecretStore(environment=config.tradovate_environment).load_tradovate()
            checks.append(
                "Tradovate credentials loaded for user "
                f"{secrets.username!r} in {config.tradovate_environment!r}"
            )
        except Exception as exc:
            errors.append(f"Tradovate credential check failed: {exc}")

    print("Tradovate demo launch preflight")
    print(f"  Config: {config_path}")
    print(f"  Backend: {config.execution_backend}")
    print(f"  Tradovate: {config.tradovate_environment}")
    print(f"  IBKR data endpoint: {config.ib_host}:{config.ib_port}")
    print(f"  Order mode: {'dry_run' if config.dry_run else 'demo order placement enabled'}")
    print()

    for check in checks:
        print(f"OK: {check}")
    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
