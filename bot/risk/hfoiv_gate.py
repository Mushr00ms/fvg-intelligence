"""
hfoiv_gate.py — High-Frequency Order-Imbalance Volatility pre-entry filter.

Computes rolling volatility of 5-min aggressor imbalance and normalizes
by time-of-day bucket across recent sessions.  When imbalance volatility
is elevated (high percentile), reduces position size.

Signal source (both backtest and live):
    Databento CME tick data with native aggressor side classification.
    buy_vol  = sum(size where side == 'A')
    sell_vol = sum(size where side == 'B')
    imbalance = buy_vol - sell_vol

Rolling window starts from 08:30 ETH so the gate is active at RTH open.

Usage (backtester):
    gate = HFOIVGate(HFOIVConfig(enabled=True))
    for day in trading_days:
        gate.reset_day()
        for bar in imbalance_bars[day]:         # 08:30-16:00
            gate.update(bar_minutes, bar.imbalance)
        # At trade entry:
        mult, info = gate.get_size_multiplier(entry_minutes)
        contracts = max(1, floor(base_qty * dd_mult * mult))

Usage (live engine):
    Same gate object, fed by tick accumulator building 5-min imbalance bars.
"""

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Configuration ────────────────────────────────────────────────────────

@dataclass
class HFOIVConfig:
    """Configuration for the HFOIV gate.

    Stored in strategy meta as ``hfoiv_gate: {...}``.
    """
    enabled: bool = False
    rolling_bars: int = 12              # N bars for rolling std (12 × 5min = 60min)
    lookback_sessions: int = 60         # sessions of history for normalization
    bucket_minutes: int = 30            # time-of-day bucket width
    # Graduated thresholds: (percentile, size_multiplier).
    # Evaluated top-down — first match wins.
    thresholds: list = field(default_factory=lambda: [
        (90, 0.25),
        (80, 0.50),
        (70, 0.75),
    ])

    def __post_init__(self):
        """Validate configuration — this controls real money."""
        if self.rolling_bars < 1:
            raise ValueError(f"rolling_bars must be >= 1, got {self.rolling_bars}")
        if self.lookback_sessions < 1:
            raise ValueError(f"lookback_sessions must be >= 1, got {self.lookback_sessions}")
        if self.bucket_minutes < 1:
            raise ValueError(f"bucket_minutes must be >= 1, got {self.bucket_minutes}")
        if not self.thresholds:
            raise ValueError("thresholds must be non-empty")
        for pct, mult in self.thresholds:
            if not (0 <= pct <= 100):
                raise ValueError(f"threshold percentile must be 0-100, got {pct}")
            if mult > 1.0:
                raise ValueError(f"threshold multiplier must be <= 1.0 (gate can only "
                                 f"reduce size, never increase), got {mult}")
            if mult < 0:
                raise ValueError(f"threshold multiplier must be >= 0, got {mult}")


# ── Gate ─────────────────────────────────────────────────────────────────

class HFOIVGate:
    """Stateful HFOIV gate.  One instance per backtest run or live session.

    Maintains:
        _rolling         – intra-day deque of imbalance values (reset each day)
        _history          – cross-session normalization: {bucket: deque of HFOIV floats}
        _session_staged   – current session's HFOIV values (flushed to _history on reset_day)
        _session_count    – number of completed sessions flushed into history
    """

    def __init__(self, config: Optional[HFOIVConfig] = None):
        self.config = config or HFOIVConfig()
        self._rolling: deque = deque(maxlen=self.config.rolling_bars)

        # Cross-session history: {bucket_str: deque of HFOIV values}
        # Max size generous enough to hold lookback_sessions × bars_per_bucket
        _max = self.config.lookback_sessions * (self.config.bucket_minutes // 5 + 1)
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=_max))

        # Staging area for current session (flushed on next reset_day)
        self._session_staged: dict[str, list] = defaultdict(list)
        self._session_count: int = 0

    # ── Day lifecycle ────────────────────────────────────────────────

    def reset_day(self):
        """Call at the START of each new trading day.

        Flushes the previous session's staged HFOIV values into the
        normalization history, then clears intra-day state.
        """
        # Flush staged values from previous session into history
        if self._session_staged:
            for bucket, values in self._session_staged.items():
                self._history[bucket].extend(values)
            self._session_count += 1
            self._session_staged.clear()

        self._rolling.clear()

    # ── Feed ─────────────────────────────────────────────────────────

    def update(self, minutes_since_midnight: int, imbalance: float):
        """Feed a completed 5-min bar's aggressor imbalance value.

        Should be called for every 5-min bar from 08:30 ETH through 16:00,
        in chronological order within the day.
        """
        # Guard against NaN/Inf from corrupted data
        if not math.isfinite(imbalance):
            return

        self._rolling.append(imbalance)

        if len(self._rolling) < self.config.rolling_bars:
            return  # not enough bars for rolling std yet

        hfoiv = float(np.std(list(self._rolling)))

        # Guard against NaN from degenerate data
        if not math.isfinite(hfoiv):
            return

        bucket = self._time_bucket(minutes_since_midnight)
        self._session_staged[bucket].append(hfoiv)

    # ── Query ────────────────────────────────────────────────────────

    def get_size_multiplier(self, minutes_since_midnight: int) -> tuple:
        """Return (size_multiplier, info_dict) for the current HFOIV state.

        Returns 1.0 (no adjustment) when:
            - gate is disabled
            - rolling window not yet full
            - insufficient normalization history (< lookback_sessions)
            - HFOIV is NaN/Inf (data corruption)
        """
        if not self.config.enabled:
            return 1.0, {}

        if len(self._rolling) < self.config.rolling_bars:
            return 1.0, {"reason": "insufficient_bars",
                         "bars": len(self._rolling)}

        hfoiv = float(np.std(list(self._rolling)))

        # NaN/Inf guard — return safe default
        if not math.isfinite(hfoiv):
            return 1.0, {"reason": "nan_hfoiv"}

        bucket = self._time_bucket(minutes_since_midnight)

        history = self._history.get(bucket)
        if history is None or len(history) == 0 or \
                self._session_count < self.config.lookback_sessions:
            return 1.0, {"reason": "warmup",
                         "sessions": self._session_count,
                         "hfoiv": round(hfoiv, 2)}

        # Percentile rank: % of historical values <= current
        arr = np.array(history)
        percentile = float(np.sum(arr <= hfoiv) / len(arr) * 100)

        # NaN guard on percentile (e.g. all-NaN history after filtering)
        if not math.isfinite(percentile):
            return 1.0, {"reason": "nan_percentile", "hfoiv": round(hfoiv, 2)}

        # Graduated thresholds — sorted descending, first match wins
        mult = 1.0
        for pct_threshold, size_mult in sorted(self.config.thresholds,
                                               key=lambda t: t[0],
                                               reverse=True):
            if percentile >= pct_threshold:
                mult = size_mult
                break

        return mult, {
            "hfoiv": round(hfoiv, 2),
            "percentile": round(percentile, 1),
            "bucket": bucket,
            "mult": mult,
            "history_len": len(arr),
            "sessions": self._session_count,
        }

    # ── Internals ────────────────────────────────────────────────────

    def _time_bucket(self, minutes_since_midnight: int) -> str:
        """Map minutes-since-midnight to a bucket label like '10:00-10:30'."""
        bm = self.config.bucket_minutes
        start = (minutes_since_midnight // bm) * bm
        end = start + bm
        sh, sm = divmod(start, 60)
        eh, em = divmod(end, 60)
        return f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}"
