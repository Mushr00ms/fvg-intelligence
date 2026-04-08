"""
config.py — Configuration for the standalone crypto bot.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass
class CryptoBotConfig:
    execution_mode: str = "dry_run"  # dry_run | live
    symbol: str = "BTCUSDC"
    strategy_path: str = "logic/strategies/btc-5min-wf-train2024-prune2025-ev007-s200-mitonly-p15.json"
    state_dir: str = "crypto_bot/state"
    log_dir: str = "crypto_bot/logs"

    # Binance USD-M
    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://fapi.binance.com"
    ws_base_url: str = "wss://fstream.binance.com"
    recv_window: int = 5000
    leverage: int = 4
    margin_type: str = "CROSSED"
    position_mode: str = "HEDGE"
    user_stream_keepalive: int = 1800
    stream_reconnect_seconds: int = 5
    reconcile_interval_seconds: int = 60
    account_asset: str = ""

    # Strategy / execution
    min_fvg_bps: float = 5.0
    mitigation_window_5m: int = 90
    risk_per_trade: float = 0.006
    maker_fee: float = 0.0
    tp_fee: float = 0.0
    stop_fee: float = 0.0004
    max_concurrent: int = 4
    max_cumulative_risk_pct: float = 0.05
    max_daily_loss_pct: float = 0.10
    max_margin_usage_pct: float = 1.0
    min_liquidation_buffer_pct: float = 0.05
    allow_start_with_open_positions: bool = False
    resume_managed_positions: bool = True

    # Ops
    starting_balance: float = 50000.0
    save_interval_seconds: int = 15
    daily_reset_timezone: str = "America/New_York"
    market_timezone: str = "America/New_York"
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    def __post_init__(self):
        self.execution_mode = (self.execution_mode or "dry_run").lower()
        if self.execution_mode not in {"dry_run", "live"}:
            raise ValueError("execution_mode must be 'dry_run' or 'live'")
        self.margin_type = self.margin_type.upper()
        self.position_mode = self.position_mode.upper()
        if self.position_mode not in {"ONE_WAY", "HEDGE"}:
            raise ValueError("position_mode must be 'ONE_WAY' or 'HEDGE'")
        if self.leverage <= 0:
            raise ValueError("leverage must be > 0")
        if not (0 < self.risk_per_trade < 1):
            raise ValueError("risk_per_trade must be between 0 and 1")
        if not (0 < self.max_cumulative_risk_pct <= 1):
            raise ValueError("max_cumulative_risk_pct must be between 0 and 1")
        if not (0 < self.max_margin_usage_pct <= 1):
            raise ValueError("max_margin_usage_pct must be between 0 and 1")
        if not (0 <= self.min_liquidation_buffer_pct < 1):
            raise ValueError("min_liquidation_buffer_pct must be between 0 and 1")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        ZoneInfo(self.daily_reset_timezone)
        ZoneInfo(self.market_timezone)
        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        if not self.telegram_enabled:
            self.telegram_enabled = bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def dry_run(self) -> bool:
        return self.execution_mode != "live"


def load_config(config_path: str | None = None) -> CryptoBotConfig:
    """Load config from JSON file and optional env overrides."""
    kwargs = {}
    path = config_path or default_config_path()
    if path and os.path.exists(path):
        with open(path) as f:
            kwargs.update(json.load(f))

    env_map = {
        "CRYPTO_BOT_MODE": ("execution_mode", str),
        "CRYPTO_BOT_SYMBOL": ("symbol", str),
        "CRYPTO_BOT_BASE_URL": ("base_url", str),
        "CRYPTO_BOT_WS_BASE_URL": ("ws_base_url", str),
        "CRYPTO_BOT_RECV_WINDOW": ("recv_window", int),
        "CRYPTO_BOT_LEVERAGE": ("leverage", int),
        "CRYPTO_BOT_MARGIN_TYPE": ("margin_type", str),
        "CRYPTO_BOT_POSITION_MODE": ("position_mode", str),
        "CRYPTO_BOT_RISK_PER_TRADE": ("risk_per_trade", float),
        "CRYPTO_BOT_MAX_CONCURRENT": ("max_concurrent", int),
        "CRYPTO_BOT_MAX_CUMULATIVE_RISK_PCT": ("max_cumulative_risk_pct", float),
        "CRYPTO_BOT_MAX_DAILY_LOSS_PCT": ("max_daily_loss_pct", float),
        "CRYPTO_BOT_MAX_MARGIN_USAGE_PCT": ("max_margin_usage_pct", float),
        "CRYPTO_BOT_MIN_LIQUIDATION_BUFFER_PCT": ("min_liquidation_buffer_pct", float),
    }
    for env_key, (field_name, converter) in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            kwargs[field_name] = converter(value)

    return CryptoBotConfig(**kwargs)


def default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "crypto_bot_config.json")
