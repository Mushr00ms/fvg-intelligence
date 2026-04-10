"""
clock.py — Precision time source for the trading bot.

Problem: System clocks drift. WSL2 can drift by minutes. A trading bot
that cancels orders at 15:50 or flattens at 15:55 cannot tolerate even
30 seconds of drift.

Solution: NTP-synced clock with IB server time cross-validation.

Architecture:
    1. On startup, query multiple NTP servers to compute offset from local clock
    2. Optionally cross-check with IB reqCurrentTime()
    3. All bot code calls clock.now() instead of datetime.now()
    4. Periodic re-sync every 5 minutes
    5. Log warnings if drift exceeds threshold

Two time categories in the bot:
    - Market data timestamps (bar dates, FVG times): come FROM IB, always trusted
    - Session management (entry windows, EOD actions): use THIS clock
"""

import asyncio
import socket
import struct
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytz

NY_TZ = pytz.timezone("America/New_York")

# NTP pool servers (tried in order, first success wins)
NTP_SERVERS = [
    "time.google.com",
    "time.cloudflare.com",
    "pool.ntp.org",
    "time.windows.com",
    "time.nist.gov",
]

# NTP epoch offset: NTP uses 1900-01-01, Unix uses 1970-01-01
_NTP_EPOCH_OFFSET = 2208988800

# Thresholds
DRIFT_WARN_MS = 500      # Warn if local clock off by more than 500ms
DRIFT_CRITICAL_MS = 5000  # Critical if off by more than 5 seconds
SYNC_INTERVAL = 300       # Re-sync every 5 minutes


def _query_ntp(server, timeout=3.0):
    """
    Query a single NTP server and return the offset in seconds.

    Uses raw NTP protocol (RFC 5905) — no external dependency needed.
    Returns (offset_seconds, round_trip_ms) or raises on failure.
    """
    # NTP packet: 48 bytes, LI=0, VN=4, Mode=3 (client)
    packet = b'\x23' + 47 * b'\0'

    # Timestamps for offset calculation
    t1 = time.time()  # Client send time

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(packet, (server, 123))
        data, _ = sock.recvfrom(1024)
    finally:
        sock.close()

    t4 = time.time()  # Client receive time

    if len(data) < 48:
        raise ValueError("NTP response too short")

    # Extract transmit timestamp (bytes 40-47)
    # NTP timestamp: 32-bit seconds + 32-bit fraction since 1900-01-01
    ntp_seconds = struct.unpack('!I', data[40:44])[0]
    ntp_fraction = struct.unpack('!I', data[44:48])[0]
    t3 = ntp_seconds - _NTP_EPOCH_OFFSET + ntp_fraction / (2**32)

    # Extract receive timestamp (bytes 32-39)
    ntp_recv_seconds = struct.unpack('!I', data[32:36])[0]
    ntp_recv_fraction = struct.unpack('!I', data[36:40])[0]
    t2 = ntp_recv_seconds - _NTP_EPOCH_OFFSET + ntp_recv_fraction / (2**32)

    # NTP offset formula: ((t2 - t1) + (t3 - t4)) / 2
    offset = ((t2 - t1) + (t3 - t4)) / 2.0
    round_trip = (t4 - t1) * 1000  # ms

    return offset, round_trip


def _sync_ntp(servers=None):
    """
    Try NTP servers in order. Return best offset (lowest round-trip).
    Returns (offset_seconds, server_used, round_trip_ms) or None on total failure.
    """
    servers = servers or NTP_SERVERS
    best = None

    for server in servers:
        try:
            offset, rtt = _query_ntp(server, timeout=3.0)
            if best is None or rtt < best[2]:
                best = (offset, server, rtt)
        except (socket.error, socket.timeout, OSError, ValueError, struct.error):
            continue

    return best


class Clock:
    """
    Precision time source for the trading bot.

    Usage:
        clock = Clock(logger=bot_logger)
        clock.sync()                    # NTP sync on startup
        now = clock.now()               # Corrected datetime in NY timezone
        clock.validate_with_ib(ib)      # Cross-check with IB server time
    """

    def __init__(self, logger=None):
        self._logger = logger
        self._offset = 0.0              # Seconds to ADD to local time
        self._last_sync = 0.0
        self._sync_server = None
        self._sync_rtt_ms = 0.0
        self._ib_offset = None          # Offset vs IB server time
        self._sync_count = 0
        self._lock = None  # Set by engine if running in async context

    def sync(self):
        """
        Synchronize with NTP servers. Call on startup and periodically.
        Updates the internal offset that corrects all now() calls.
        """
        result = _sync_ntp()
        if result is None:
            if self._logger:
                self._logger.log(
                    "clock_sync_failed",
                    servers_tried=len(NTP_SERVERS),
                    using="system_clock",
                )
            # Fall back to system clock (offset = 0)
            return False

        offset, server, rtt = result
        self._offset = offset
        self._sync_server = server
        self._sync_rtt_ms = rtt
        self._last_sync = time.time()
        self._sync_count += 1

        drift_ms = abs(offset * 1000)

        if self._logger:
            level = "clock_sync"
            if drift_ms > DRIFT_CRITICAL_MS:
                level = "clock_drift_critical"
            elif drift_ms > DRIFT_WARN_MS:
                level = "clock_drift_warning"

            self._logger.log(
                level,
                offset_ms=round(offset * 1000, 1),
                server=server,
                rtt_ms=round(rtt, 1),
                sync_count=self._sync_count,
            )

        return True

    async def validate_with_broker(self, broker):
        """
        Cross-check our NTP-corrected time with the broker's server time.
        Logs a warning if they disagree by more than 1 second.

        Args:
            broker: BrokerAdapter instance (or legacy IBConnection).
        """
        if not broker.is_connected:
            return

        try:
            # BrokerAdapter path
            if hasattr(broker, 'get_server_time'):
                broker_time = await broker.get_server_time()
            # Legacy IBConnection path
            elif hasattr(broker, 'ib'):
                broker_time = await broker.ib.reqCurrentTimeAsync()
            else:
                return

            if broker_time is None:
                return

            broker_unix = broker_time.timestamp()
            our_unix = self.now_utc().timestamp()
            offset = our_unix - broker_unix

            self._ib_offset = offset

            if self._logger:
                if abs(offset) > 1.0:
                    self._logger.log(
                        "clock_broker_mismatch",
                        our_vs_broker_ms=round(offset * 1000, 1),
                        note="Bot clock disagrees with broker server by >1s",
                    )
                else:
                    self._logger.log(
                        "clock_broker_validated",
                        our_vs_broker_ms=round(offset * 1000, 1),
                    )
        except Exception as e:
            if self._logger:
                self._logger.log("clock_broker_check_error", error=str(e))

    # Backward compat alias
    validate_with_ib = validate_with_broker

    def now(self):
        """
        Current time in Eastern Time, corrected by NTP offset.
        This is the ONLY time source the bot should use for session decisions.
        """
        corrected_unix = time.time() + self._offset
        utc_dt = datetime.fromtimestamp(corrected_unix, tz=timezone.utc)
        return utc_dt.astimezone(NY_TZ)

    def now_utc(self):
        """Current time in UTC, corrected by NTP offset."""
        corrected_unix = time.time() + self._offset
        return datetime.fromtimestamp(corrected_unix, tz=timezone.utc)

    def today_str(self):
        """Today's date string in NY timezone: 'YYYY-MM-DD'."""
        return self.now().strftime("%Y-%m-%d")

    def check_resync(self):
        """Re-sync if SYNC_INTERVAL has passed since last sync."""
        if time.time() - self._last_sync > SYNC_INTERVAL:
            self.sync()

    @property
    def offset_ms(self):
        """Current NTP offset in milliseconds."""
        return round(self._offset * 1000, 1)

    @property
    def is_synced(self):
        """True if NTP sync has succeeded at least once."""
        return self._sync_count > 0

    @property
    def is_trusted(self):
        """True if the clock has had at least one successful NTP sync.

        When False, the bot should block new trade entries — session timing
        decisions (entry windows, EOD flatten) cannot be trusted because the
        system clock may have drifted arbitrarily (especially on WSL2).
        """
        return self._sync_count > 0

    @property
    def sync_age_seconds(self):
        """Seconds since last successful NTP sync."""
        if self._last_sync == 0:
            return float('inf')
        return time.time() - self._last_sync

    def __repr__(self):
        return (
            f"Clock(offset={self.offset_ms}ms, "
            f"server={self._sync_server}, "
            f"synced={self.is_synced})"
        )
