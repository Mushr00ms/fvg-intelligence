"""
secrets.py — Credential loading from AWS SSM Parameter Store.

All broker credentials (Tradovate, Binance, Telegram) are stored as
SecureString parameters in SSM, encrypted at rest with KMS.
The EC2 instance's IAM role grants ssm:GetParameter — no keys on disk.

Parameter naming convention:
    /fvg-bot/{environment}/tradovate/username
    /fvg-bot/{environment}/tradovate/password
    /fvg-bot/{environment}/tradovate/cid
    /fvg-bot/{environment}/tradovate/sec
    /fvg-bot/{environment}/tradovate/app_id
    /fvg-bot/{environment}/tradovate/device_id
    /fvg-bot/{environment}/telegram/bot_token
    /fvg-bot/{environment}/telegram/chat_id

Where {environment} is "demo", "live", or "paper".

Setup (one-time, from a machine with AWS credentials):

    aws ssm put-parameter --name "/fvg-bot/demo/tradovate/username" \\
        --value "myuser" --type SecureString --overwrite

    aws ssm put-parameter --name "/fvg-bot/demo/tradovate/password" \\
        --value "mypass" --type SecureString --overwrite

    aws ssm put-parameter --name "/fvg-bot/demo/tradovate/cid" \\
        --value "12345" --type SecureString --overwrite

    aws ssm put-parameter --name "/fvg-bot/demo/tradovate/sec" \\
        --value "api-secret-here" --type SecureString --overwrite

    aws ssm put-parameter --name "/fvg-bot/demo/tradovate/app_id" \\
        --value "MyBot" --type SecureString --overwrite

    aws ssm put-parameter --name "/fvg-bot/demo/tradovate/device_id" \\
        --value "device-uuid-here" --type SecureString --overwrite

EC2 instance IAM policy (minimum):

    {
        "Effect": "Allow",
        "Action": "ssm:GetParametersByPath",
        "Resource": "arn:aws:ssm:*:*:parameter/fvg-bot/*"
    }
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# SSM path prefix — override via BOT_SSM_PREFIX env var
DEFAULT_SSM_PREFIX = "/fvg-bot"


@dataclass(frozen=True)
class TradovateSecrets:
    """Tradovate credentials loaded from SSM."""
    username: str
    password: str
    cid: int
    sec: str
    app_id: str
    app_version: str = "1.0"
    device_id: str = ""


@dataclass(frozen=True)
class TelegramSecrets:
    """Telegram credentials loaded from SSM."""
    bot_token: str
    chat_id: str


class SecretStore:
    """Loads secrets from AWS SSM Parameter Store.

    Uses the EC2 instance's IAM role for authentication — no access keys
    needed on disk. Falls back to environment variables for local dev.
    """

    def __init__(self, environment: str = "demo", ssm_prefix: str = ""):
        self._env = environment
        self._prefix = ssm_prefix or os.environ.get(
            "BOT_SSM_PREFIX", DEFAULT_SSM_PREFIX
        )
        self._cache: Dict[str, str] = {}
        self._ssm_client = None

    def _get_ssm_client(self):
        """Lazy-init boto3 SSM client."""
        if self._ssm_client is None:
            import boto3
            region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            self._ssm_client = boto3.client("ssm", region_name=region)
        return self._ssm_client

    def _param_path(self, *parts: str) -> str:
        """Build SSM parameter path: /fvg-bot/{env}/tradovate/username"""
        return f"{self._prefix}/{self._env}/{'/'.join(parts)}"

    def _get_parameter(self, *path_parts: str) -> Optional[str]:
        """Fetch a single SSM parameter. Returns None if not found."""
        name = self._param_path(*path_parts)

        # Check cache
        if name in self._cache:
            return self._cache[name]

        try:
            client = self._get_ssm_client()
            resp = client.get_parameter(Name=name, WithDecryption=True)
            value = resp["Parameter"]["Value"]
            self._cache[name] = value
            return value
        except Exception as e:
            # ClientError: ParameterNotFound, AccessDeniedException, etc.
            logger.debug("SSM parameter %s not found: %s", name, e)
            return None

    def _get_all_parameters(self, service: str) -> Dict[str, str]:
        """Fetch all parameters under a service path in one call.

        E.g., service="tradovate" fetches all /fvg-bot/demo/tradovate/*
        """
        path = f"{self._prefix}/{self._env}/{service}"

        try:
            client = self._get_ssm_client()
            params: Dict[str, str] = {}
            paginator = client.get_paginator("get_parameters_by_path")

            for page in paginator.paginate(
                Path=path,
                WithDecryption=True,
                Recursive=False,
            ):
                for param in page.get("Parameters", []):
                    # Extract key name from full path
                    # /fvg-bot/demo/tradovate/username → username
                    key = param["Name"].rsplit("/", 1)[-1]
                    params[key] = param["Value"]
                    self._cache[param["Name"]] = param["Value"]

            return params

        except Exception as e:
            logger.warning("Failed to fetch SSM parameters at %s: %s", path, e)
            return {}

    def load_tradovate(self) -> TradovateSecrets:
        """Load Tradovate credentials from SSM.

        Falls back to environment variables for local development:
            TRADOVATE_USERNAME, TRADOVATE_PASSWORD, TRADOVATE_CID,
            TRADOVATE_SEC, TRADOVATE_APP_ID, TRADOVATE_DEVICE_ID
        """
        params = self._get_all_parameters("tradovate")

        # Env var fallback for local dev (never on prod EC2)
        def _get(key: str, env_key: str, default: str = "") -> str:
            return params.get(key) or os.environ.get(env_key, default)

        username = _get("username", "TRADOVATE_USERNAME")
        password = _get("password", "TRADOVATE_PASSWORD")
        cid_str = _get("cid", "TRADOVATE_CID", "0")
        sec = _get("sec", "TRADOVATE_SEC")
        app_id = _get("app_id", "TRADOVATE_APP_ID")
        app_version = _get("app_version", "TRADOVATE_APP_VERSION", "1.0")
        device_id = _get("device_id", "TRADOVATE_DEVICE_ID")

        if not username or not password:
            raise SecretLoadError(
                "Tradovate credentials not found in SSM "
                f"({self._prefix}/{self._env}/tradovate/*) "
                "or environment variables (TRADOVATE_USERNAME, etc.)"
            )

        try:
            cid = int(cid_str)
        except (ValueError, TypeError):
            raise SecretLoadError(f"Invalid Tradovate CID: {cid_str!r}")

        logger.info(
            "Loaded Tradovate credentials: user=%s env=%s source=%s",
            username, self._env,
            "ssm" if params else "env",
        )

        return TradovateSecrets(
            username=username,
            password=password,
            cid=cid,
            sec=sec,
            app_id=app_id,
            app_version=app_version,
            device_id=device_id,
        )

    def load_telegram(self) -> Optional[TelegramSecrets]:
        """Load Telegram credentials from SSM. Returns None if not configured."""
        params = self._get_all_parameters("telegram")

        bot_token = params.get("bot_token") or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = params.get("chat_id") or os.environ.get("TELEGRAM_CHAT_ID", "")

        if not bot_token or not chat_id:
            return None

        return TelegramSecrets(bot_token=bot_token, chat_id=chat_id)


class SecretLoadError(Exception):
    """Raised when required secrets cannot be loaded."""
