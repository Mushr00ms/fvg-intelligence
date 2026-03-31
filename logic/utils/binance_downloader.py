"""
binance_downloader.py — Download Binance USD-M futures aggTrades from data.binance.vision.

Streams bulk ZIP archives with SHA256 checksum verification, atomic writes,
resume-by-existence, and configurable retry logic.

Usage:
    from logic.utils.binance_downloader import BinanceAggTradesDownloader

    dl = BinanceAggTradesDownloader(symbol="BTCUSDT")
    report = dl.download_range(date(2024, 1, 1), date(2024, 3, 31), granularity="monthly")
    print(report)
"""

import hashlib
import logging
import os
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class DownloadReport:
    """Summary of a download_range() run."""
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    checksum_failures: int = 0
    file_paths: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class BinanceAggTradesDownloader:
    """Download Binance USD-M futures aggTrades from data.binance.vision."""

    BASE_URL = "https://data.binance.vision/data/futures/um"

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        raw_dir: str = "logic/binance_data/raw",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        chunk_size: int = 8192,
    ):
        self.symbol = symbol.upper()
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size

        self._session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    # ── Public API ────────────────────────────────────────────────────────

    def download_range(
        self,
        start_date: date,
        end_date: date,
        granularity: str = "monthly",
    ) -> DownloadReport:
        """Download aggTrades archives for the given date range.

        Args:
            start_date: First date (inclusive).
            end_date:   Last date (inclusive).
            granularity: "monthly" for bulk historical, "daily" for recent/incremental.

        Returns:
            DownloadReport with counts and file paths.
        """
        report = DownloadReport()

        if granularity == "monthly":
            periods = self._month_range(start_date, end_date)
        elif granularity == "daily":
            periods = self._day_range(start_date, end_date)
        else:
            raise ValueError(f"granularity must be 'monthly' or 'daily', got '{granularity}'")

        total = len(periods)
        for i, period_date in enumerate(periods, 1):
            zip_url, checksum_url = self._build_urls(period_date, granularity)
            zip_name = zip_url.rsplit("/", 1)[-1]
            zip_path = self.raw_dir / zip_name

            logger.info("[%d/%d] %s", i, total, zip_name)

            # Resume: skip if ZIP already exists
            if zip_path.exists():
                logger.info("  Skipped (already exists)")
                report.skipped += 1
                csv_path = self._csv_path_for_zip(zip_path)
                if csv_path and csv_path.exists():
                    report.file_paths.append(csv_path)
                else:
                    try:
                        csv_path = self._extract_csv(zip_path)
                        report.file_paths.append(csv_path)
                    except Exception as e:
                        logger.warning("  Extract failed for existing ZIP: %s", e)
                continue

            # Download
            ok = self._download_file(zip_url, zip_path)
            if not ok:
                report.failed += 1
                report.errors.append(f"Download failed: {zip_url}")
                continue

            # Checksum
            if not self._verify_checksum(zip_path, checksum_url):
                report.checksum_failures += 1
                report.errors.append(f"Checksum mismatch: {zip_name}")
                continue

            # Extract
            try:
                csv_path = self._extract_csv(zip_path)
                report.downloaded += 1
                report.file_paths.append(csv_path)
                logger.info("  OK → %s", csv_path.name)
            except Exception as e:
                report.failed += 1
                report.errors.append(f"Extract failed: {zip_name}: {e}")

        logger.info(
            "Done: %d downloaded, %d skipped, %d failed, %d checksum failures",
            report.downloaded, report.skipped, report.failed, report.checksum_failures,
        )
        return report

    # ── URL Construction ──────────────────────────────────────────────────

    def _build_urls(self, period_date: date, granularity: str) -> tuple:
        """Return (zip_url, checksum_url) for the given date and granularity."""
        if granularity == "monthly":
            tag = period_date.strftime("%Y-%m")
            path = f"{self.BASE_URL}/monthly/aggTrades/{self.symbol}/{self.symbol}-aggTrades-{tag}.zip"
        else:
            tag = period_date.strftime("%Y-%m-%d")
            path = f"{self.BASE_URL}/daily/aggTrades/{self.symbol}/{self.symbol}-aggTrades-{tag}.zip"
        return path, path + ".CHECKSUM"

    # ── Download ──────────────────────────────────────────────────────────

    def _download_file(self, url: str, dest: Path) -> bool:
        """Stream-download a file with atomic write (.tmp → rename)."""
        tmp_path = dest.with_suffix(dest.suffix + ".tmp")
        try:
            resp = self._session.get(url, stream=True, timeout=120)
            if resp.status_code == 404:
                logger.warning("  Not found (404): %s", url)
                return False
            resp.raise_for_status()

            total_bytes = int(resp.headers.get("Content-Length", 0))
            downloaded = 0

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=self.chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes and downloaded % (self.chunk_size * 1024) < self.chunk_size:
                        pct = downloaded * 100 // total_bytes
                        logger.debug("  %d%%", pct)

            os.replace(str(tmp_path), str(dest))
            return True

        except Exception as e:
            logger.error("  Download error: %s", e)
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    # ── Checksum ──────────────────────────────────────────────────────────

    def _verify_checksum(self, zip_path: Path, checksum_url: str) -> bool:
        """Download .CHECKSUM file and verify SHA256 of the local ZIP."""
        try:
            resp = self._session.get(checksum_url, timeout=30)
            if resp.status_code == 404:
                logger.warning("  No checksum file available, skipping verification")
                return True
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("  Could not fetch checksum (%s), skipping verification", e)
            return True

        expected_hash = self._parse_checksum(resp.text)
        if not expected_hash:
            logger.warning("  Could not parse checksum file, skipping verification")
            return True

        actual_hash = self._sha256_file(zip_path)

        if actual_hash != expected_hash:
            logger.error(
                "  CHECKSUM MISMATCH for %s\n    expected: %s\n    actual:   %s",
                zip_path.name, expected_hash, actual_hash,
            )
            zip_path.unlink(missing_ok=True)
            return False

        logger.debug("  Checksum OK")
        return True

    @staticmethod
    def _parse_checksum(text: str) -> Optional[str]:
        """Parse Binance CHECKSUM format: '<sha256hex>  <filename>'."""
        text = text.strip()
        if not text:
            return None
        parts = text.split()
        if len(parts) >= 1 and len(parts[0]) == 64:
            return parts[0].lower()
        return None

    def _sha256_file(self, path: Path) -> str:
        """Compute SHA256 hex digest of a file using streaming reads."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    # ── Extract ───────────────────────────────────────────────────────────

    def _extract_csv(self, zip_path: Path) -> Path:
        """Extract the single CSV from a ZIP archive."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if len(csv_names) != 1:
                raise ValueError(
                    f"Expected 1 CSV in {zip_path.name}, found {len(csv_names)}: {csv_names}"
                )
            csv_name = csv_names[0]
            csv_path = self.raw_dir / csv_name
            if not csv_path.exists():
                zf.extract(csv_name, self.raw_dir)
            return csv_path

    def _csv_path_for_zip(self, zip_path: Path) -> Optional[Path]:
        """Infer the expected CSV filename from a ZIP path without opening it."""
        csv_name = zip_path.stem + ".csv"
        csv_path = self.raw_dir / csv_name
        return csv_path if csv_path.exists() else None

    # ── Date Iteration ────────────────────────────────────────────────────

    @staticmethod
    def _month_range(start: date, end: date) -> list:
        """Generate first-of-month dates covering [start, end]."""
        months = []
        current = start.replace(day=1)
        end_month = end.replace(day=1)
        while current <= end_month:
            months.append(current)
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return months

    @staticmethod
    def _day_range(start: date, end: date) -> list:
        """Generate daily dates covering [start, end]."""
        days = []
        current = start
        while current <= end:
            days.append(current)
            current += timedelta(days=1)
        return days
