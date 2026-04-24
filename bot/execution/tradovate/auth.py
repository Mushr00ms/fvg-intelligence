"""
auth.py — Tradovate authentication and token management.

Handles initial authentication via /auth/accesstokenrequest and
automatic token renewal before expiry.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# Environment base URLs
ENVIRONMENTS = {
    "demo": "https://demo.tradovateapi.com/v1",
    "live": "https://live.tradovateapi.com/v1",
}


@dataclass
class TokenInfo:
    """Authentication token state."""
    access_token: str
    md_access_token: str = ""       # Separate token for md.tradovateapi.com
    expiration_time: float = 0.0    # Unix timestamp when token expires
    user_id: int = 0
    name: str = ""

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expiration_time

    @property
    def seconds_until_expiry(self) -> float:
        return max(0.0, self.expiration_time - time.time())


@dataclass
class TradovateCredentials:
    """Credentials for Tradovate API authentication."""
    username: str
    password: str
    app_id: str
    app_version: str
    cid: int
    sec: str
    device_id: str
    environment: str = "demo"       # "demo" | "live"

    @property
    def base_url(self) -> str:
        return ENVIRONMENTS[self.environment]


class TradovateAuth:
    """Manages Tradovate API authentication and token lifecycle."""

    # Renew token when less than this many seconds remain
    RENEW_BUFFER_SECONDS = 300      # 5 minutes before expiry

    def __init__(self, credentials: TradovateCredentials):
        self._creds = credentials
        self._token: Optional[TokenInfo] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._renewal_task: Optional[asyncio.Task] = None

    @property
    def token(self) -> Optional[TokenInfo]:
        return self._token

    @property
    def access_token(self) -> str:
        if self._token is None:
            raise RuntimeError("Not authenticated — call authenticate() first")
        return self._token.access_token

    @property
    def is_authenticated(self) -> bool:
        return self._token is not None and not self._token.is_expired

    async def authenticate(self) -> TokenInfo:
        """Perform initial authentication. Returns TokenInfo on success."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = f"{self._creds.base_url}/auth/accesstokenrequest"
        body = {
            "name": self._creds.username,
            "password": self._creds.password,
            "appId": self._creds.app_id,
            "appVersion": self._creds.app_version,
            "cid": self._creds.cid,
            "sec": self._creds.sec,
            "deviceId": self._creds.device_id,
        }

        async with self._session.post(url, json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise AuthenticationError(
                    f"Tradovate auth failed (HTTP {resp.status}): {text}"
                )
            data = await resp.json()

        # Handle Tradovate's time penalty (p-ticket) for too many failed logins
        if "p-ticket" in data:
            wait_secs = data.get("p-time", 5)
            logger.warning(
                "Tradovate time penalty: waiting %ds before retry", wait_secs
            )
            await asyncio.sleep(wait_secs)
            return await self._renew_with_p_ticket(data["p-ticket"])

        if "accessToken" not in data:
            raise AuthenticationError(
                f"Tradovate auth response missing accessToken: {data}"
            )

        # Parse expiration — Tradovate returns expirationTime as ISO string
        # We convert to Unix timestamp
        expiration_str = data.get("expirationTime", "")
        if expiration_str:
            from datetime import datetime, timezone
            try:
                exp_dt = datetime.fromisoformat(expiration_str.replace("Z", "+00:00"))
                exp_unix = exp_dt.timestamp()
            except (ValueError, TypeError):
                # Fallback: 24 hours from now
                exp_unix = time.time() + 86400
        else:
            exp_unix = time.time() + 86400

        self._token = TokenInfo(
            access_token=data["accessToken"],
            md_access_token=data.get("mdAccessToken", ""),
            expiration_time=exp_unix,
            user_id=data.get("userId", 0),
            name=data.get("name", ""),
        )

        logger.info(
            "Tradovate authenticated: user=%s env=%s expires_in=%.0fs md_token=%s",
            self._token.name, self._creds.environment,
            self._token.seconds_until_expiry,
            "yes" if self._token.md_access_token else "no",
        )

        return self._token

    async def _renew_with_p_ticket(self, p_ticket: str) -> TokenInfo:
        """Retry authentication with a p-ticket after time penalty."""
        url = f"{self._creds.base_url}/auth/accesstokenrequest"
        body = {
            "name": self._creds.username,
            "password": self._creds.password,
            "appId": self._creds.app_id,
            "appVersion": self._creds.app_version,
            "cid": self._creds.cid,
            "sec": self._creds.sec,
            "deviceId": self._creds.device_id,
            "p-ticket": p_ticket,
        }

        async with self._session.post(url, json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise AuthenticationError(
                    f"Tradovate p-ticket auth failed (HTTP {resp.status}): {text}"
                )
            data = await resp.json()

        if "accessToken" not in data:
            raise AuthenticationError(
                f"Tradovate p-ticket auth response missing accessToken: {data}"
            )

        from datetime import datetime
        expiration_str = data.get("expirationTime", "")
        exp_unix = time.time() + 86400
        if expiration_str:
            try:
                exp_dt = datetime.fromisoformat(expiration_str.replace("Z", "+00:00"))
                exp_unix = exp_dt.timestamp()
            except (ValueError, TypeError):
                pass

        self._token = TokenInfo(
            access_token=data["accessToken"],
            md_access_token=data.get("mdAccessToken", ""),
            expiration_time=exp_unix,
            user_id=data.get("userId", 0),
            name=data.get("name", ""),
        )
        return self._token

    async def start_renewal_loop(self) -> None:
        """Start background task that renews token before expiry."""
        if self._renewal_task and not self._renewal_task.done():
            return
        self._renewal_task = asyncio.ensure_future(self._renewal_loop())

    async def renew_token(self) -> TokenInfo:
        """Renew the current access token without starting a new session.

        Uses /auth/renewAccessToken which extends the existing session.
        Falls back to full re-auth if renewal fails.
        """
        if self._session is None:
            self._session = aiohttp.ClientSession()

        url = f"{self._creds.base_url}/auth/renewAccessToken"
        headers = {
            "Authorization": f"Bearer {self._token.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        async with self._session.get(url, headers=headers) as resp:
            if resp.status != 200:
                logger.warning("Token renewal returned HTTP %d, falling back to re-auth", resp.status)
                return await self.authenticate()
            data = await resp.json()

        if data.get("errorText"):
            logger.warning("Token renewal error: %s, falling back to re-auth", data["errorText"])
            return await self.authenticate()

        new_token = data.get("accessToken", "")
        if not new_token:
            logger.warning("Token renewal returned no accessToken, falling back to re-auth")
            return await self.authenticate()

        expiration_str = data.get("expirationTime", "")
        if expiration_str:
            from datetime import datetime, timezone
            try:
                exp_dt = datetime.fromisoformat(expiration_str.replace("Z", "+00:00"))
                exp_unix = exp_dt.timestamp()
            except (ValueError, TypeError):
                exp_unix = time.time() + 86400
        else:
            exp_unix = time.time() + 86400

        self._token = TokenInfo(
            access_token=new_token,
            md_access_token=self._token.md_access_token,
            expiration_time=exp_unix,
            user_id=data.get("userId", self._token.user_id),
            name=data.get("name", self._token.name),
        )

        logger.info(
            "Tradovate token renewed: expires_in=%.0fs",
            self._token.seconds_until_expiry,
        )
        return self._token

    async def _renewal_loop(self) -> None:
        """Background loop that renews token before it expires."""
        while True:
            try:
                if self._token is None:
                    await asyncio.sleep(10)
                    continue

                wait = self._token.seconds_until_expiry - self.RENEW_BUFFER_SECONDS
                if wait > 0:
                    await asyncio.sleep(wait)

                logger.info("Renewing Tradovate token (expires in %.0fs)",
                            self._token.seconds_until_expiry)
                await self.renew_token()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Token renewal failed: %s", e)
                await asyncio.sleep(30)

    async def close(self) -> None:
        """Clean up resources."""
        if self._renewal_task and not self._renewal_task.done():
            self._renewal_task.cancel()
            try:
                await self._renewal_task
            except asyncio.CancelledError:
                pass
        if self._session and not self._session.closed:
            await self._session.close()


class AuthenticationError(Exception):
    """Raised when Tradovate authentication fails."""
