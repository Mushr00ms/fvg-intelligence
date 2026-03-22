"""
bot_logger.py — Structured JSONL logging for the trading bot.

Writes JSON lines to bot/logs/YYYY-MM-DD.jsonl and to console stdout.
Each line is a self-contained JSON object with timestamp and event type.
"""

import json
import os
import sys
from datetime import datetime

import pytz

NY_TZ = pytz.timezone("America/New_York")


class BotLogger:
    def __init__(self, log_dir, clock=None):
        self._log_dir = log_dir
        self._clock = clock
        os.makedirs(log_dir, exist_ok=True)
        self._current_date = None
        self._file = None
        self._console = sys.stdout

    def _now(self):
        """Get current time from injected clock or system clock."""
        if self._clock is not None:
            return self._clock.now()
        return datetime.now(NY_TZ)

    def _ensure_file(self):
        """Open/rotate the daily log file."""
        today = self._now().strftime("%Y-%m-%d")
        if today != self._current_date:
            if self._file:
                self._file.close()
            filepath = os.path.join(self._log_dir, f"{today}.jsonl")
            self._file = open(filepath, "a", buffering=1)  # line-buffered
            self._current_date = today

    def log(self, event, **kwargs):
        """
        Log a structured event.

        Args:
            event: Event type string (e.g. "fvg_detected", "order_placed")
            **kwargs: Arbitrary key-value pairs for the event payload
        """
        self._ensure_file()

        record = {
            "ts": self._now().isoformat(),
            "event": event,
            **kwargs,
        }

        line = json.dumps(record, default=str)

        # Write to file
        self._file.write(line + "\n")

        # Write to console (formatted)
        self._write_console(record)

    def _write_console(self, record):
        """Pretty-print a log record to console."""
        ts = record["ts"]
        event = record["event"]
        # Truncate timestamp to HH:MM:SS for console
        time_part = ts[11:19] if len(ts) > 19 else ts

        # Color coding by event severity
        color = _EVENT_COLORS.get(event, "\033[0m")
        reset = "\033[0m"

        # Build detail string from remaining keys
        details = {k: v for k, v in record.items() if k not in ("ts", "event")}
        detail_str = ""
        if details:
            parts = []
            for k, v in details.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
            detail_str = " " + " ".join(parts)

        print(
            f"{color}[{time_part}] {event.upper()}{reset}{detail_str}",
            file=self._console,
        )

    def close(self):
        """Close the log file."""
        if self._file:
            self._file.close()
            self._file = None


# Console color map (ANSI)
_EVENT_COLORS = {
    # Green — positive outcomes
    "bot_start": "\033[92m",
    "tp_filled": "\033[92m",
    "order_filled": "\033[92m",
    "connection_restored": "\033[92m",
    "strategy_loaded": "\033[92m",
    "strategy_reloaded": "\033[92m",
    # Yellow — informational
    "fvg_detected": "\033[93m",
    "mitigation": "\033[93m",
    "order_placed": "\033[93m",
    "partial_fill": "\033[93m",
    "setup_accepted": "\033[92m",
    "setup_evaluated": "\033[93m",
    "state_saved": "\033[90m",          # dim
    # Red — warnings / losses
    "sl_filled": "\033[91m",
    "kill_switch": "\033[91m",
    "connection_lost": "\033[91m",
    "order_rejected": "\033[91m",
    "flatten": "\033[91m",
    "setup_rejected": "\033[33m",
    # Cyan — system
    "bot_stop": "\033[96m",
    "daily_summary": "\033[96m",
    "reconciliation": "\033[96m",
    "eod_cancel": "\033[96m",
    "eod_flatten": "\033[96m",
    "eod_exit": "\033[33m",
}
