"""Tests for US market holiday calendar."""

from bot.backtest.us_holidays import is_trading_day, trading_days_in_range


class TestIsTradingDay:
    """Test trading day detection."""

    def test_weekday_no_holiday(self):
        assert is_trading_day("20260323") is True  # Monday

    def test_saturday(self):
        assert is_trading_day("20260321") is False  # Saturday

    def test_sunday(self):
        assert is_trading_day("20260322") is False  # Sunday

    def test_mlk_day_2026(self):
        assert is_trading_day("20260119") is False

    def test_good_friday_2026(self):
        assert is_trading_day("20260403") is False

    def test_thanksgiving_2026(self):
        assert is_trading_day("20261126") is False

    def test_christmas_2026(self):
        assert is_trading_day("20261225") is False

    def test_day_after_holiday_is_trading(self):
        """Day after Christmas 2026 (Sat) — skip to Monday Dec 28."""
        assert is_trading_day("20261228") is True

    def test_independence_day_observed_2026(self):
        """July 3, 2026 (Friday) — July 4 is Saturday, observed Friday."""
        assert is_trading_day("20260703") is False


class TestTradingDaysInRange:
    """Test date range iteration."""

    def test_one_week(self):
        """Mon-Fri week with no holidays = 5 trading days."""
        days = list(trading_days_in_range("20260302", "20260306"))
        assert len(days) == 5
        assert days[0] == "20260302"  # Monday
        assert days[-1] == "20260306"  # Friday

    def test_range_skips_weekend(self):
        """Fri-Mon range = 2 trading days."""
        days = list(trading_days_in_range("20260306", "20260309"))
        assert len(days) == 2
        assert "20260307" not in days  # Saturday
        assert "20260308" not in days  # Sunday

    def test_range_skips_holiday(self):
        """Week containing MLK Day (Jan 19, 2026 = Monday)."""
        days = list(trading_days_in_range("20260119", "20260123"))
        assert "20260119" not in days  # Holiday
        assert len(days) == 4  # Tue-Fri

    def test_single_day_trading(self):
        days = list(trading_days_in_range("20260323", "20260323"))
        assert days == ["20260323"]

    def test_single_day_holiday(self):
        days = list(trading_days_in_range("20261225", "20261225"))
        assert days == []
