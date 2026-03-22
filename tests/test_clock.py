"""Tests for the NTP-synced precision clock."""

import time
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from bot.clock import Clock, _query_ntp, _sync_ntp, DRIFT_WARN_MS, NY_TZ


class TestClockBasics:
    """Tests for clock functionality without real NTP."""

    def test_now_returns_ny_timezone(self):
        clock = Clock()
        now = clock.now()
        assert now.tzinfo is not None
        assert "Eastern" in str(now.tzinfo) or "EDT" in str(now.tzinfo) or "EST" in str(now.tzinfo) or "US/Eastern" in str(now.tzinfo) or "America" in str(now.tzinfo)

    def test_now_utc_returns_utc(self):
        clock = Clock()
        now = clock.now_utc()
        assert now.tzinfo == timezone.utc

    def test_today_str_format(self):
        clock = Clock()
        today = clock.today_str()
        assert len(today) == 10  # YYYY-MM-DD
        assert today[4] == "-" and today[7] == "-"

    def test_offset_applied(self):
        """Manually set offset and verify it's applied."""
        clock = Clock()
        clock._offset = 60.0  # Pretend we're 60 seconds ahead
        now = clock.now()
        system_now = datetime.now(NY_TZ)
        diff = (now - system_now).total_seconds()
        assert 55 < diff < 65  # Should be ~60s ahead

    def test_zero_offset_by_default(self):
        clock = Clock()
        assert clock._offset == 0.0
        assert clock.offset_ms == 0.0

    def test_is_synced_false_before_sync(self):
        clock = Clock()
        assert clock.is_synced is False

    def test_sync_age_infinite_before_sync(self):
        clock = Clock()
        assert clock.sync_age_seconds == float('inf')

    def test_repr(self):
        clock = Clock()
        r = repr(clock)
        assert "Clock(" in r
        assert "offset=" in r


class TestClockWithMockedNTP:
    """Tests with mocked NTP responses."""

    @patch("bot.clock._sync_ntp")
    def test_sync_sets_offset(self, mock_sync):
        mock_sync.return_value = (0.05, "time.google.com", 15.2)
        clock = Clock()
        result = clock.sync()
        assert result is True
        assert clock._offset == 0.05
        assert clock.offset_ms == 50.0
        assert clock._sync_server == "time.google.com"
        assert clock.is_synced is True

    @patch("bot.clock._sync_ntp")
    def test_sync_failure_keeps_zero_offset(self, mock_sync):
        mock_sync.return_value = None
        clock = Clock()
        result = clock.sync()
        assert result is False
        assert clock._offset == 0.0
        assert clock.is_synced is False

    @patch("bot.clock._sync_ntp")
    def test_resync_updates_offset(self, mock_sync):
        mock_sync.return_value = (0.05, "time.google.com", 15.0)
        clock = Clock()
        clock.sync()
        assert clock._sync_count == 1

        mock_sync.return_value = (0.02, "time.cloudflare.com", 10.0)
        clock._last_sync = time.time() - 400  # Force resync
        clock.check_resync()
        assert clock._sync_count == 2
        assert clock._offset == 0.02

    @patch("bot.clock._sync_ntp")
    def test_drift_warning_logged(self, mock_sync):
        mock_sync.return_value = (0.8, "pool.ntp.org", 20.0)  # 800ms drift
        logger = MagicMock()
        clock = Clock(logger=logger)
        clock.sync()
        # Should log clock_drift_warning (>500ms)
        logger.log.assert_called()
        call_args = logger.log.call_args
        assert call_args[0][0] == "clock_drift_warning"

    @patch("bot.clock._sync_ntp")
    def test_critical_drift_logged(self, mock_sync):
        mock_sync.return_value = (6.0, "pool.ntp.org", 20.0)  # 6 seconds drift!
        logger = MagicMock()
        clock = Clock(logger=logger)
        clock.sync()
        call_args = logger.log.call_args
        assert call_args[0][0] == "clock_drift_critical"


def _can_reach_ntp():
    """Check if we can reach an NTP server (for skipping tests in offline envs)."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("time.google.com", 123))
        s.close()
        return True
    except (socket.error, OSError):
        return False


class TestNTPQuery:
    """Test real NTP queries (may fail in CI without network)."""

    @pytest.mark.skipif(
        not _can_reach_ntp(),
        reason="NTP server unreachable"
    )
    def test_real_ntp_query(self):
        offset, rtt = _query_ntp("time.google.com", timeout=5.0)
        assert isinstance(offset, float)
        assert isinstance(rtt, float)
        assert rtt > 0
        # Offset should be reasonable (< 10 seconds for most systems)
        assert abs(offset) < 10.0

    @pytest.mark.skipif(
        not _can_reach_ntp(),
        reason="NTP server unreachable"
    )
    def test_real_sync(self):
        result = _sync_ntp()
        if result is not None:
            offset, server, rtt = result
            assert isinstance(offset, float)
            assert isinstance(server, str)
            assert rtt > 0

    @pytest.mark.skipif(
        not _can_reach_ntp(),
        reason="NTP server unreachable"
    )
    def test_real_clock_sync(self):
        clock = Clock()
        synced = clock.sync()
        if synced:
            assert clock.is_synced
            assert abs(clock.offset_ms) < 10000  # < 10 seconds
            # Now() should be very close to system time (within offset)
            now = clock.now()
            sys_now = datetime.now(NY_TZ)
            diff_ms = abs((now - sys_now).total_seconds() * 1000)
            assert diff_ms < 10000


class TestClockIntegrationWithTimeGates:
    """Test that TimeGates uses the injected clock."""

    @patch("bot.clock._sync_ntp")
    def test_time_gates_uses_clock(self, mock_sync):
        mock_sync.return_value = (0.0, "test", 1.0)  # No offset
        clock = Clock()
        clock.sync()

        from bot.risk.time_gates import TimeGates

        class FakeConfig:
            session_start = "09:30"
            session_end = "16:00"
            last_entry_time = "15:45"
            cancel_unfilled_time = "15:50"
            flatten_time = "15:55"

        gates = TimeGates(FakeConfig(), clock=clock)
        # The now_et() should come from the clock
        now = gates.now_et()
        assert now.tzinfo is not None
