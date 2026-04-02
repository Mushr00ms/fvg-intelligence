"""
strategy.py — Loader for the BTC leverage-aware strategy JSON.
"""

from __future__ import annotations

import json
from pathlib import Path


RISK_BINS = [1, 7, 10, 12, 14, 17, 20, 24, 31, 43, 994]


def risk_to_range(risk_bps: float) -> str | None:
    for i in range(len(RISK_BINS) - 1):
        if RISK_BINS[i] <= risk_bps < RISK_BINS[i + 1]:
            return f"{RISK_BINS[i]}-{RISK_BINS[i+1]}"
    return None


class BTCStrategyLoader:
    def __init__(self, strategy_path: str):
        self._path = Path(strategy_path)
        self.strategy = None
        self.strategy_id = ""
        self._lookup = {}

    def load(self):
        with open(self._path) as f:
            self.strategy = json.load(f)
        self.strategy_id = self.strategy["meta"]["id"]
        self._lookup = {}
        for cell in self.strategy.get("cells", []):
            if not cell.get("enabled", True):
                continue
            key = (cell["time_period"], cell["risk_range"], cell["setup"])
            self._lookup[key] = cell

    def find_cell(self, time_period: str, risk_bps: float, setup: str):
        risk_range = risk_to_range(risk_bps)
        if not risk_range:
            return None
        return self._lookup.get((time_period, risk_range, setup))
