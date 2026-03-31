"""Tests for Binance aggTrades downloader — URL construction, checksum, resume, atomic writes."""

import hashlib
import os
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from logic.utils.binance_downloader import BinanceAggTradesDownloader, DownloadReport


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def downloader(tmp_path):
    return BinanceAggTradesDownloader(symbol="BTCUSDT", raw_dir=str(tmp_path / "raw"))


@pytest.fixture
def raw_dir(downloader):
    return downloader.raw_dir


# ── URL Construction ──────────────────────────────────────────────────────

class TestBuildUrls:
    def test_monthly_url(self, downloader):
        zip_url, chk_url = downloader._build_urls(date(2024, 1, 1), "monthly")
        assert zip_url == (
            "https://data.binance.vision/data/futures/um/monthly/aggTrades/"
            "BTCUSDT/BTCUSDT-aggTrades-2024-01.zip"
        )
        assert chk_url == zip_url + ".CHECKSUM"

    def test_daily_url(self, downloader):
        zip_url, chk_url = downloader._build_urls(date(2024, 3, 15), "daily")
        assert zip_url == (
            "https://data.binance.vision/data/futures/um/daily/aggTrades/"
            "BTCUSDT/BTCUSDT-aggTrades-2024-03-15.zip"
        )
        assert chk_url == zip_url + ".CHECKSUM"

    def test_custom_symbol(self, tmp_path):
        dl = BinanceAggTradesDownloader(symbol="ETHUSDT", raw_dir=str(tmp_path / "raw"))
        zip_url, _ = dl._build_urls(date(2024, 6, 1), "monthly")
        assert "/ETHUSDT/" in zip_url
        assert "ETHUSDT-aggTrades-2024-06.zip" in zip_url


# ── Date Ranges ───────────────────────────────────────────────────────────

class TestDateRanges:
    def test_month_range(self, downloader):
        months = downloader._month_range(date(2024, 10, 5), date(2025, 2, 20))
        assert months == [
            date(2024, 10, 1), date(2024, 11, 1), date(2024, 12, 1),
            date(2025, 1, 1), date(2025, 2, 1),
        ]

    def test_month_range_single(self, downloader):
        months = downloader._month_range(date(2024, 6, 15), date(2024, 6, 20))
        assert months == [date(2024, 6, 1)]

    def test_day_range(self, downloader):
        days = downloader._day_range(date(2024, 1, 29), date(2024, 2, 2))
        assert days == [
            date(2024, 1, 29), date(2024, 1, 30), date(2024, 1, 31),
            date(2024, 2, 1), date(2024, 2, 2),
        ]


# ── Checksum Parsing ──────────────────────────────────────────────────────

class TestChecksumParsing:
    def test_standard_format(self, downloader):
        text = "d752f3a1b0c4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0  BTCUSDT-aggTrades-2024-01.zip\n"
        result = downloader._parse_checksum(text)
        assert result == "d752f3a1b0c4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0"

    def test_no_filename(self, downloader):
        text = "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234"
        result = downloader._parse_checksum(text)
        assert result == "abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234"

    def test_empty_string(self, downloader):
        assert downloader._parse_checksum("") is None
        assert downloader._parse_checksum("  ") is None

    def test_invalid_format(self, downloader):
        assert downloader._parse_checksum("short_hash  file.zip") is None


# ── SHA256 File Hash ──────────────────────────────────────────────────────

class TestSha256File:
    def test_known_hash(self, downloader, raw_dir):
        test_file = raw_dir / "test.bin"
        content = b"hello binance data"
        test_file.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        actual = downloader._sha256_file(test_file)
        assert actual == expected


# ── Resume (Skip Existing) ───────────────────────────────────────────────

class TestResume:
    def test_skip_existing_zip(self, downloader, raw_dir):
        """If the ZIP already exists, download_range should skip it."""
        # Create a fake ZIP with a CSV inside
        zip_path = raw_dir / "BTCUSDT-aggTrades-2024-01-15.zip"
        csv_name = "BTCUSDT-aggTrades-2024-01-15.csv"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(csv_name, "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n")

        with patch.object(downloader, "_download_file") as mock_dl:
            report = downloader.download_range(date(2024, 1, 15), date(2024, 1, 15), granularity="daily")

        mock_dl.assert_not_called()
        assert report.skipped == 1
        assert report.downloaded == 0


# ── Extract CSV ───────────────────────────────────────────────────────────

class TestExtractCsv:
    def test_extracts_single_csv(self, downloader, raw_dir):
        zip_path = raw_dir / "test.zip"
        csv_content = "agg_trade_id,price,quantity\n1,50000.0,0.5\n"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("test.csv", csv_content)

        csv_path = downloader._extract_csv(zip_path)
        assert csv_path.exists()
        assert csv_path.read_text() == csv_content

    def test_rejects_multi_csv_zip(self, downloader, raw_dir):
        zip_path = raw_dir / "multi.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.csv", "data")
            zf.writestr("b.csv", "data")

        with pytest.raises(ValueError, match="Expected 1 CSV"):
            downloader._extract_csv(zip_path)


# ── Atomic Write ──────────────────────────────────────────────────────────

class TestAtomicWrite:
    def test_no_tmp_remains_on_success(self, downloader, raw_dir):
        """After successful download, no .tmp file should remain."""
        dest = raw_dir / "test_download.zip"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Length": "5"}
        mock_response.iter_content.return_value = [b"hello"]
        mock_response.raise_for_status = MagicMock()

        with patch.object(downloader._session, "get", return_value=mock_response):
            result = downloader._download_file("https://example.com/test.zip", dest)

        assert result is True
        assert dest.exists()
        assert not dest.with_suffix(".zip.tmp").exists()

    def test_tmp_cleaned_on_failure(self, downloader, raw_dir):
        """On network error, .tmp should be cleaned up."""
        dest = raw_dir / "fail_download.zip"

        with patch.object(
            downloader._session, "get", side_effect=Exception("network error")
        ):
            result = downloader._download_file("https://example.com/fail.zip", dest)

        assert result is False
        assert not dest.exists()
        assert not dest.with_suffix(".zip.tmp").exists()


# ── Checksum Verification Flow ────────────────────────────────────────────

class TestChecksumVerification:
    def test_mismatch_deletes_zip(self, downloader, raw_dir):
        """A checksum mismatch should delete the corrupt ZIP."""
        zip_path = raw_dir / "corrupt.zip"
        zip_path.write_bytes(b"corrupt data")

        wrong_hash = "0" * 64 + "  corrupt.zip"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = wrong_hash
        mock_resp.raise_for_status = MagicMock()

        with patch.object(downloader._session, "get", return_value=mock_resp):
            result = downloader._verify_checksum(zip_path, "https://example.com/checksum")

        assert result is False
        assert not zip_path.exists()

    def test_match_keeps_zip(self, downloader, raw_dir):
        """Correct checksum should return True and keep the file."""
        zip_path = raw_dir / "good.zip"
        content = b"good data"
        zip_path.write_bytes(content)

        correct_hash = hashlib.sha256(content).hexdigest() + "  good.zip"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = correct_hash
        mock_resp.raise_for_status = MagicMock()

        with patch.object(downloader._session, "get", return_value=mock_resp):
            result = downloader._verify_checksum(zip_path, "https://example.com/checksum")

        assert result is True
        assert zip_path.exists()

    def test_404_checksum_passes(self, downloader, raw_dir):
        """If .CHECKSUM returns 404, skip verification (return True)."""
        zip_path = raw_dir / "nochecksum.zip"
        zip_path.write_bytes(b"data")

        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch.object(downloader._session, "get", return_value=mock_resp):
            result = downloader._verify_checksum(zip_path, "https://example.com/checksum")

        assert result is True
