"""
bot_config.py — Bot configuration: IB connection, risk parameters, operational modes.

Loads from bot/bot_config.json with environment variable overrides.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BotConfig:
    """Complete bot configuration."""

    # IB Connection
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497             # Paper: 7497, Live: 7496
    ib_client_id: int = 1
    ib_timeout: int = 30            # Connection timeout in seconds

    # Contract
    ticker: str = "NQ"
    exchange: str = "CME"
    currency: str = "USD"

    # Strategy
    strategy_dir: str = ""          # Path to logic/strategies/ (auto-detected)
    min_fvg_size: float = 0.25      # Minimum FVG size in points

    # Risk Management — 3-tier Kelly-inspired sizing
    risk_per_trade: float = 0.01    # Default fallback (used if tiers disabled)
    risk_small_pct: float = 0.005   # 0.5% for 5-10pt, 10-15pt risk buckets
    risk_medium_pct: float = 0.015  # 1.5% for 15-20pt, 20-25pt, 25-30pt, 30-40pt
    risk_large_pct: float = 0.03    # 3.0% for 40-80pt risk buckets
    small_buckets: list = field(default_factory=lambda: ["5-10", "10-15"])
    large_buckets: list = field(default_factory=lambda: ["40-80"])
    use_risk_tiers: bool = True     # Enable 3-tier risk (False = uniform risk_per_trade)
    max_trade_loss_pct: float = 0.015  # 1.5% max single trade loss (slippage buffer)
    max_concurrent: int = 3         # Max open positions at any time
    max_daily_trades: int = 15      # Max trades per session
    kill_switch_pct: float = -0.10  # -10% daily loss = emergency halt (catastrophic only)
    dd_scale_tier1_pct: float = -0.02  # -2% daily → 50% size
    dd_scale_tier2_pct: float = -0.04  # -4% daily → 25% size
    max_cumulative_risk_pct: float = 0.05  # 5% max cumulative open risk
    point_value: float = 20.0       # NQ = $20/point
    tick_size: float = 0.25         # NQ tick size

    # Slippage model — BACKTESTER ONLY. In live trading, IB handles fills at
    # exact limit prices. Keep False for the live bot.
    use_slippage: bool = False      # Only enable in backtester (--slip flag)
    slippage_ticks: int = 1         # Number of ticks for slippage (backtester)

    # Session Times (ET)
    session_start: str = "09:30"
    session_end: str = "16:00"
    # Backtest parity defaults:
    # - no new entries at/after 15:45
    # - pending entries remain valid until session end
    # - open positions flatten at session end
    last_entry_time: str = "15:45"
    cancel_unfilled_time: str = "16:00"
    flatten_time: str = "16:00"

    # Operational Modes
    paper_mode: bool = True         # True = port 7497, False = port 7496
    dry_run: bool = True            # Log but don't place orders

    # Telegram Alerts
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_enabled: bool = False

    # State & Logging
    state_dir: str = ""             # Auto-detected: bot/bot_state/
    log_dir: str = ""               # Auto-detected: bot/logs/
    state_save_interval: int = 30   # Seconds between periodic state saves
    strategy_reload_interval: int = 60  # Seconds between strategy file checks

    # Partial Fill
    partial_fill_timeout: int = 300  # 5 minutes in seconds

    # Bridge (WSL → Windows)
    bridge_port: int = 9100          # TCP port for WSL↔Windows bridge
    auto_launch_bridge: bool = True  # Auto-launch bridge via python.exe

    # Margin Management — time-aware (IB intraday vs overnight)
    # Intraday maintenance applies during RTH: 09:30–16:00 ET (08:30–15:00 CT)
    # Overnight initial applies outside RTH (we're always flat before this)
    margin_intraday_maintenance: float = 22924.0   # NQ intraday maintenance margin
    margin_overnight_initial: float = 46373.0      # NQ overnight initial margin
    margin_fallback_per_contract: float = 22924.0  # Ultimate fallback (= intraday maintenance)
    margin_intraday_start: str = "09:30"           # ET — intraday margin begins
    margin_intraday_end: str = "16:00"             # ET — intraday margin ends (15 min before RTH close)
    margin_buffer_pct: float = 0.05                # 5% safety buffer above required margin
    margin_refresh_interval: int = 1800            # Re-fetch margin every 30 minutes
    margin_management_enabled: bool = True         # Enable intelligent margin priority system

    # Connection Recovery
    max_disconnect_minutes: int = 5  # Flatten if disconnected longer than this
    reconnect_interval: int = 10     # Seconds between reconnect attempts

    def __post_init__(self):
        """Apply defaults for auto-detected paths."""
        bot_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(bot_dir)

        # WSL2: auto-detect Windows host IP if ib_host is still localhost
        if self.ib_host == "127.0.0.1":
            detected = _detect_wsl_windows_host()
            if detected:
                self.ib_host = detected

        if not self.strategy_dir:
            self.strategy_dir = os.path.join(project_dir, "logic", "strategies")
        if not self.state_dir:
            self.state_dir = os.path.join(bot_dir, "bot_state")
        if not self.log_dir:
            self.log_dir = os.path.join(bot_dir, "logs")

        # Auto-set port based on paper_mode
        if self.paper_mode and self.ib_port == 7496:
            self.ib_port = 7497
        elif not self.paper_mode and self.ib_port == 7497:
            self.ib_port = 7496

        # Enable telegram if both token and chat_id are set
        if self.telegram_bot_token and self.telegram_chat_id:
            self.telegram_enabled = True

        # Create runtime directories
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


def _detect_wsl_windows_host():
    """Detect the Windows host IP from WSL2. Returns None if not in WSL."""
    # First try default gateway (most reliable for WSL2 mirrored networking)
    try:
        import subprocess
        result = subprocess.run(['ip', 'route', 'show', 'default'],
                                capture_output=True, text=True, timeout=3)
        for part in result.stdout.split():
            if '.' in part and part[0].isdigit():
                return part
    except Exception:
        pass
    # Fallback: resolv.conf nameserver
    try:
        with open('/etc/resolv.conf') as f:
            for line in f:
                if line.strip().startswith('nameserver'):
                    ip = line.strip().split()[-1]
                    if ip != '127.0.0.1':
                        return ip
    except FileNotFoundError:
        pass
    return None


# Environment variable overrides (BOT_ prefix)
_ENV_MAP = {
    "BOT_IB_HOST": ("ib_host", str),
    "BOT_IB_PORT": ("ib_port", int),
    "BOT_IB_CLIENT_ID": ("ib_client_id", int),
    "BOT_TICKER": ("ticker", str),
    "BOT_RISK_PER_TRADE": ("risk_per_trade", float),
    "BOT_RISK_SMALL_PCT": ("risk_small_pct", float),
    "BOT_RISK_MEDIUM_PCT": ("risk_medium_pct", float),
    "BOT_RISK_LARGE_PCT": ("risk_large_pct", float),
    "BOT_USE_RISK_TIERS": ("use_risk_tiers", lambda x: x.lower() in ("1", "true", "yes")),
    "BOT_MAX_CONCURRENT": ("max_concurrent", int),
    "BOT_MAX_DAILY_TRADES": ("max_daily_trades", int),
    "BOT_KILL_SWITCH_PCT": ("kill_switch_pct", float),
    "BOT_PAPER_MODE": ("paper_mode", lambda x: x.lower() in ("1", "true", "yes")),
    "BOT_DRY_RUN": ("dry_run", lambda x: x.lower() in ("1", "true", "yes")),
    "BOT_TELEGRAM_TOKEN": ("telegram_bot_token", str),
    "BOT_TELEGRAM_CHAT_ID": ("telegram_chat_id", str),
    "BOT_STRATEGY_DIR": ("strategy_dir", str),
    "BOT_STATE_DIR": ("state_dir", str),
    "BOT_LOG_DIR": ("log_dir", str),
    "BOT_MARGIN_INTRADAY": ("margin_intraday_maintenance", float),
    "BOT_MARGIN_OVERNIGHT": ("margin_overnight_initial", float),
    "BOT_MARGIN_FALLBACK": ("margin_fallback_per_contract", float),
    "BOT_MARGIN_BUFFER_PCT": ("margin_buffer_pct", float),
    "BOT_MARGIN_REFRESH": ("margin_refresh_interval", int),
    "BOT_MARGIN_ENABLED": ("margin_management_enabled", lambda x: x.lower() in ("1", "true", "yes")),
}


def load_bot_config(config_path=None):
    """
    Load bot configuration from JSON file + environment variable overrides.

    Priority: env vars > config file > dataclass defaults
    """
    kwargs = {}

    # 1. Load from JSON if provided
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            file_config = json.load(f)
        kwargs.update(file_config)

    # 2. Apply environment variable overrides
    for env_key, (field_name, converter) in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val is not None:
            kwargs[field_name] = converter(val)

    return BotConfig(**kwargs)


def default_config_path():
    """Return the default config file path: bot/bot_config.json"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_config.json")
