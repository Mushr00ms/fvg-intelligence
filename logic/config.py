import json
import os
import uuid
from datetime import *

import numpy as np
import pytz

# Note: Using Databento for historical data
# Keeping IB_AVAILABLE for backward compatibility
IB_AVAILABLE = False
IB = None
Future = None
r_id = str(uuid.uuid4())[:6]
print(f"""
---------------------------------------------------------------
[INFO - {str(uuid.uuid4())[:6]}] Loading Python Specific Env and historical data""")

# ============================================================================
# MASTER CONFIGURATION - Content Generation & Trading Analysis System
# ============================================================================

# --- Market Selection ---
# Primary market for analysis and content generation
PRIMARY_MARKET = os.environ.get("FVG_TICKER", "NQ")  # Options: 'ES', 'NQ' — overrideable via env var

# --- Time and Session Configuration ---
NY_TIMEZONE = pytz.timezone("America/New_York")

# Session configurations for different markets
SESSION_CONFIGS = {
    "ES": {
        "us_session": {"start": time(9, 30), "end": time(16, 0)},
        "asian_session": {"start": time(18, 0), "end": time(2, 0)},  # Next day
        "london_session": {"start": time(3, 0), "end": time(9, 0)},
        "extended_hours": {"start": time(6, 0), "end": time(17, 0)},
        "primary_session": "us_session",
    },
    "NQ": {
        "us_session": {"start": time(9, 30), "end": time(16, 0)},
        "asian_session": {"start": time(18, 0), "end": time(2, 0)},
        "london_session": {"start": time(3, 0), "end": time(9, 0)},
        "extended_hours": {"start": time(6, 0), "end": time(17, 0)},
        "primary_session": "us_session",
    },
}

# Get current session config based on primary market
CURRENT_SESSION_CONFIG = SESSION_CONFIGS.get(
    PRIMARY_MARKET, SESSION_CONFIGS[PRIMARY_MARKET]
)

# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================


def load_market_config(market_symbol):
    """Load market-specific configuration from JSON file in configs/ directory."""
    # Get the directory where this config.py file is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "configs", f"{market_symbol}_config.json")

    if not os.path.exists(config_file):
        return None

    try:
        with open(config_file, "r") as f:
            config = json.load(f)
            return config
    except Exception as e:
        print(f"[ERROR] Failed to load config file {config_file}: {e}")
        return None


def save_market_config(market_symbol, config):
    """Save market-specific configuration to JSON file in configs/ directory."""
    # Get the directory where this config.py file is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "configs", f"{market_symbol}_config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2, default=str)


_FVG_SIZES_RANGE_DEFAULTS = {
    "min_fvg_sizes_range": (0.25, 32.25, 2),
    "min_fvg_sizes_range_3min": (0.25, 20.25, 1.0),
    "min_fvg_sizes_range_15min": (10.0, 40.0, 2.0),
}


def get_min_fvg_sizes_range(market_symbol=None, config_key="min_fvg_sizes_range"):
    """Get the FVG sizes range for analysis based on market configuration."""
    # Allow env var override: FVG_SIZE_START/END/STEP specify the inclusive range
    if "FVG_SIZE_START" in os.environ:
        start = float(os.environ["FVG_SIZE_START"])
        end   = float(os.environ["FVG_SIZE_END"])
        step  = float(os.environ["FVG_SIZE_STEP"])
        return np.arange(start, end + step / 2, step)
    market = market_symbol if market_symbol else PRIMARY_MARKET
    config = load_market_config(market)
    if (
        config
        and "fvg_parameters" in config
        and config_key in config["fvg_parameters"]
    ):
        range_params = config["fvg_parameters"][config_key]
        return np.arange(
            range_params["start"], range_params["end"], range_params["step"]
        )
    fallback = _FVG_SIZES_RANGE_DEFAULTS.get(config_key, (0.25, 32.25, 2))
    return np.arange(*fallback)


def get_stats_filename_prefix():
    """Generate filename prefix that includes market identifier."""
    return f"{PRIMARY_MARKET.lower()}_fvg"


# --- Data Analysis Configuration ---
period = os.environ.get("FVG_PERIOD", "5 years")  # overrideable via env var
custom_start = None
custom_end = None

# Load market-specific configuration
MARKET_CONFIG = load_market_config(PRIMARY_MARKET)
if MARKET_CONFIG is None:
    print(
        f"[WARNING] No market-specific config found for {PRIMARY_MARKET}, using legacy defaults"
    )
    # Legacy fallback values
    lookback_period = 10
    min_search_bars_1min = 300
    min_search_bars_5min = 100
    min_search_bars_3min = 150
    min_search_bars_15min = 60
    min_search_bars_5min_mitigation = 180
    min_search_bars_1min_mitigation_3min = 300
    roll_days = 8
    mitigation_same_day = True
    min_expansion_size = 10.0
    min_expansion_size_3min = 10.0
    min_expansion_size_15min = 10.0
    min_fvg_size = 0.25
    min_fvg_size_3min = 0.25
    min_fvg_size_15min = 0.25
    # Use existing session configs
    CURRENT_SESSION_CONFIG = SESSION_CONFIGS.get(PRIMARY_MARKET, SESSION_CONFIGS["ES"])
else:
    print(f"[INFO - {str(uuid.uuid4())[:6]}] Loaded market-specific config for {PRIMARY_MARKET}")
    # Load from market-specific config
    lookback_period = MARKET_CONFIG["analysis_parameters"]["lookback_period"]
    min_search_bars_1min = MARKET_CONFIG["analysis_parameters"]["min_search_bars_1min"]
    min_search_bars_5min = MARKET_CONFIG["analysis_parameters"]["min_search_bars_5min"]
    roll_days = MARKET_CONFIG["analysis_parameters"]["roll_days"]
    mitigation_same_day = MARKET_CONFIG["analysis_parameters"]["mitigation_same_day"]

    # FVG parameters from market config
    min_expansion_size = MARKET_CONFIG["fvg_parameters"]["min_expansion_size"]
    min_fvg_size = MARKET_CONFIG["fvg_parameters"]["min_fvg_size"]
    # 3-min pipeline parameters
    min_fvg_size_3min = MARKET_CONFIG["fvg_parameters"].get("min_fvg_size_3min", min_fvg_size)
    min_expansion_size_3min = MARKET_CONFIG["fvg_parameters"].get("min_expansion_size_3min", min_expansion_size)
    min_search_bars_3min = MARKET_CONFIG["analysis_parameters"].get("min_search_bars_3min", 150)
    min_search_bars_1min_mitigation_3min = MARKET_CONFIG["analysis_parameters"].get("min_search_bars_1min_mitigation_3min", 300)

    # 15-min pipeline parameters
    min_fvg_size_15min = MARKET_CONFIG["fvg_parameters"].get("min_fvg_size_15min", min_fvg_size)
    min_expansion_size_15min = MARKET_CONFIG["fvg_parameters"].get("min_expansion_size_15min", min_expansion_size)
    min_search_bars_15min = MARKET_CONFIG["analysis_parameters"].get("min_search_bars_15min", 60)
    min_search_bars_5min_mitigation = MARKET_CONFIG["analysis_parameters"].get("min_search_bars_5min_mitigation", 180)

    # Analysis method parameters from market config
    session_period_minutes = MARKET_CONFIG["analysis_parameters"]["session_period_minutes"]
    session_period_minutes_3min = MARKET_CONFIG["analysis_parameters"].get("session_period_minutes_3min", 30)
    session_period_minutes_15min = MARKET_CONFIG["analysis_parameters"].get("session_period_minutes_15min", 60)
    size_filtering_method = MARKET_CONFIG["analysis_parameters"]["size_filtering_method"]
    size_filtering_method_3min = MARKET_CONFIG["analysis_parameters"].get("size_filtering_method_3min", size_filtering_method)
    size_filtering_method_15min = MARKET_CONFIG["analysis_parameters"].get("size_filtering_method_15min", size_filtering_method)

    # Allow env var overrides for on-demand generation from the dashboard
    if "FVG_SESSION_MINUTES" in os.environ:
        _spm = int(os.environ["FVG_SESSION_MINUTES"])
        session_period_minutes = _spm
        session_period_minutes_3min = _spm
        session_period_minutes_15min = _spm
    if "FVG_METHOD" in os.environ:
        _method = os.environ["FVG_METHOD"]
        size_filtering_method = _method
        size_filtering_method_3min = _method
        size_filtering_method_15min = _method
    if "FVG_MIN_EXP" in os.environ:
        _exp = float(os.environ["FVG_MIN_EXP"])
        min_expansion_size = _exp
        min_expansion_size_3min = _exp
        min_expansion_size_15min = _exp

    # 3-min filter end time
    _fvg_end_3min_str = MARKET_CONFIG["analysis_parameters"].get("fvg_filter_end_time_3min", None)
    if _fvg_end_3min_str:
        _h, _m = map(int, _fvg_end_3min_str.split(":"))
        fvg_filter_end_time_3min = time(_h, _m)
    else:
        fvg_filter_end_time_3min = None

    # 15-min filter end time (extends session for expansion tracking)
    _fvg_end_15min_str = MARKET_CONFIG["analysis_parameters"].get("fvg_filter_end_time_15min", None)
    if _fvg_end_15min_str:
        _h, _m = map(int, _fvg_end_15min_str.split(":"))
        fvg_filter_end_time_15min = time(_h, _m)
    else:
        fvg_filter_end_time_15min = None  # will fall back to fvg_filter_end_time

    # Convert session config from JSON format to time objects
    session_config = MARKET_CONFIG["session_config"]
    CURRENT_SESSION_CONFIG = {}
    for session_name, times in session_config.items():
        if session_name != "primary_session":
            start_time_str = times["start"]
            end_time_str = times["end"]
            start_hour, start_min = map(int, start_time_str.split(":"))
            end_hour, end_min = map(int, end_time_str.split(":"))
            CURRENT_SESSION_CONFIG[session_name] = {
                "start": time(start_hour, start_min),
                "end": time(end_hour, end_min),
            }
    CURRENT_SESSION_CONFIG["primary_session"] = session_config["primary_session"]

# Use primary session times
primary_session_name = CURRENT_SESSION_CONFIG["primary_session"]
primary_session = CURRENT_SESSION_CONFIG[primary_session_name]
session_start_time = primary_session["start"]
session_end_time = primary_session["end"]
fvg_filter_start_time = session_start_time
fvg_filter_end_time = session_end_time

# --- Logging Configuration ---
LOGGING_CONFIG = {
    "level": "INFO",
    "file": "logs/content_system.log",
    "max_file_size_mb": 50,
    "backup_count": 5,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# ============================================================================
# LEGACY CONFIGURATION (for backward compatibility with existing code)
# ============================================================================
ny_tz = NY_TIMEZONE

# ============================================================================
# INITIALIZATION
# ============================================================================

# Create necessary directories (relative to script location)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, "data_cache"), exist_ok=True)
fvg_cache_dir = os.path.join(script_dir, "fvg_cache")
os.makedirs(fvg_cache_dir, exist_ok=True)
os.makedirs(os.path.join(script_dir, "content"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "content", "templates"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "content", "generated"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "content", "posted"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "exports"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "charts"), exist_ok=True)

# No IB connection needed - using Databento
ib = None
print(f"""[INFO - {str(uuid.uuid4())[:6]}] Python data interface ready for {PRIMARY_MARKET} analysis
---------------------------------------------------------------""")

# Load market-specific config if available
MARKET_SPECIFIC_CONFIG = load_market_config(
    PRIMARY_MARKET
)  # Keep for backward compatibility

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib not available, some visualization functions may not work")
    plt = None
    mdates = None
