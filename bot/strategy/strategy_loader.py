"""
strategy_loader.py — Load the active trading strategy and build a lookup table.

Reads from logic/strategies/ (the strategy builder's output).
Supports hot-reload: checks file mtime every 60s and swaps the lookup atomically.
"""

import importlib.util
import os
import sys
import threading
import time

# Import strategy_store directly to avoid logic.utils.__init__ pulling in heavy deps
_STORE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "logic", "utils", "strategy_store.py")
)
_spec = importlib.util.spec_from_file_location("strategy_store", _STORE_PATH)
_strategy_store = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_strategy_store)

get_active_strategy = _strategy_store.get_active_strategy
get_strategy_manifest = _strategy_store.get_strategy_manifest
get_strategy_mtime = _strategy_store.get_strategy_mtime
validate_strategy = _strategy_store.validate_strategy
VALID_RR_TARGETS = _strategy_store.VALID_RR_TARGETS
VALID_SETUPS = _strategy_store.VALID_SETUPS


class StrategyLoader:
    """
    Loads the active strategy and provides O(1) cell lookup for the bot engine.

    Usage:
        loader = StrategyLoader(strategy_dir)
        loader.load()                       # Initial load (raises on failure)
        cell = loader.find_cell("10:30-11:00", 12.5)  # Returns cell config or None
        loader.check_reload()               # Call periodically for hot-reload
    """

    def __init__(self, strategy_dir, logger=None):
        self._strategy_dir = strategy_dir
        self._logger = logger
        self._strategy = None
        self._strategy_id = None
        self._lookup = {}           # {(time_period, risk_range): cell_config}
        self._reload_lock = threading.RLock()  # Protects _lookup during hot-reload
        self._last_mtime = None
        self._last_check = 0

    def load(self):
        """
        Load the active strategy. Raises RuntimeError if no strategy or validation fails.
        """
        strategy = get_active_strategy(self._strategy_dir)
        if strategy is None:
            raise RuntimeError(
                "No active strategy set. Create one in the dashboard "
                f"and set it as active. Strategy dir: {self._strategy_dir}"
            )

        errors = validate_strategy(strategy)
        if errors:
            raise RuntimeError(
                f"Strategy validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        self._strategy = strategy
        self._strategy_id = strategy["meta"]["id"]
        with self._reload_lock:
            self._lookup = self._build_lookup(strategy)
        self._last_mtime = get_strategy_mtime(self._strategy_id, self._strategy_dir)

        if self._logger:
            self._logger.log(
                "strategy_loaded",
                id=self._strategy_id,
                name=strategy["meta"].get("name", ""),
                cells=len(self._lookup),
                enabled=strategy["stats"]["enabled_cells"],
            )

    def check_reload(self):
        """
        Check if the strategy file has been modified and reload if so.
        Call this periodically (e.g. every 60s). Safe to call frequently.
        """
        if not self._strategy_id:
            return

        now = time.time()
        if now - self._last_check < 5:  # Don't check filesystem more than every 5s
            return
        self._last_check = now

        current_mtime = get_strategy_mtime(self._strategy_id, self._strategy_dir)
        if current_mtime and current_mtime != self._last_mtime:
            try:
                old_id = self._strategy_id
                self.load()
                if self._logger:
                    self._logger.log("strategy_reloaded", id=self._strategy_id)
            except Exception as e:
                if self._logger:
                    self._logger.log(
                        "strategy_reload_failed",
                        error=str(e),
                        keeping=old_id,
                    )
                # Keep the old strategy on reload failure

    def find_cell(self, time_period, risk_pts):
        """
        Find the matching strategy cell for a given time period and risk in points.

        Args:
            time_period: e.g. "10:30-11:00"
            risk_pts: actual risk in points (e.g. 12.5)

        Returns:
            dict with {setup, rr_target, median_risk, ev, win_rate, samples} or None
        """
        risk_range = self._risk_to_range(risk_pts)
        if not risk_range:
            return None
        with self._reload_lock:
            return self._lookup.get((time_period, risk_range))

    @property
    def strategy(self):
        return self._strategy

    @property
    def strategy_id(self):
        return self._strategy_id

    @property
    def cell_count(self):
        return len(self._lookup)

    def _build_lookup(self, strategy):
        """
        Build (time_period, risk_range) -> cell_config lookup from strategy cells.

        Only includes enabled cells. If same (time_period, risk_range) has multiple
        entries, keeps the one with highest EV.
        """
        lookup = {}
        for cell in strategy.get("cells", []):
            if not cell.get("enabled", True):
                continue

            key = (cell["time_period"], cell["risk_range"])
            config = {
                "setup": cell["setup"],
                "rr_target": cell["rr_target"],
                "median_risk": cell.get("median_risk", 0),
                "ev": cell.get("ev", 0),
                "win_rate": cell.get("win_rate", 0),
                "samples": cell.get("samples", 0),
            }

            # If duplicate key, keep highest EV
            if key in lookup:
                if config["ev"] > lookup[key]["ev"]:
                    lookup[key] = config
            else:
                lookup[key] = config

        return lookup

    @staticmethod
    def _risk_to_range(risk_pts):
        """
        Map a risk value in points to its risk range bucket string.
        Uses the standard bins: [5, 10, 15, 20, 25, 30, 40, 50, 200]
        """
        bins = [5, 10, 15, 20, 25, 30, 40, 50, 200]
        for i in range(len(bins) - 1):
            if bins[i] <= risk_pts < bins[i + 1]:
                return f"{bins[i]}-{bins[i + 1]}"
        return None
