"""
secret_store.py -- Credential loading with tiered backends.

Lookup order (first non-empty wins):
    1. AWS SSM Parameter Store   -- production (EC2 IAM role, KMS-encrypted)
    2. systemd credentials       -- dev/WSL2 (LoadCredential, not in /proc)
    3. Environment variables     -- CI / one-off testing only

SSM parameter naming:
    /fvg-bot/{environment}/tradovate/{key}
    /fvg-bot/{environment}/telegram/{key}

systemd credential naming (files under $CREDENTIALS_DIRECTORY):
    tradovate-{key}     (e.g. tradovate-username, tradovate-sec)
    telegram-{key}      (e.g. telegram-bot_token)

Provision systemd credentials:
    bash ops/wsl/setup_credentials.sh          # interactive
    systemctl --user restart nq-bot-manager    # picks up new creds

SSM setup (one-time, from a machine with AWS credentials):

    aws ssm put-parameter --name "/fvg-bot/demo/tradovate/username" \\
        --value "myuser" --type SecureString --overwrite

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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

CREDENTIALS_DIR = os.environ.get("CREDENTIALS_DIRECTORY", "")

# SSM path prefix -- override via BOT_SSM_PREFIX env var
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
    """Loads secrets with tiered fallback: SSM -> systemd creds -> env vars."""

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

        if name in self._cache:
            return self._cache[name]

        try:
            client = self._get_ssm_client()
            resp = client.get_parameter(Name=name, WithDecryption=True)
            value = resp["Parameter"]["Value"]
            self._cache[name] = value
            return value
        except Exception as e:
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
                    key = param["Name"].rsplit("/", 1)[-1]
                    params[key] = param["Value"]
                    self._cache[param["Name"]] = param["Value"]

            return params

        except Exception as e:
            logger.warning("Failed to fetch SSM parameters at %s: %s", path, e)
            return {}

    @staticmethod
    def _read_credential(service: str, key: str) -> str:
        """Read a systemd credential file (e.g. tradovate-username).

        Returns empty string if $CREDENTIALS_DIRECTORY is unset or file
        does not exist.
        """
        if not CREDENTIALS_DIR:
            return ""
        path = Path(CREDENTIALS_DIR) / f"{service}-{key}"
        try:
            return path.read_text().strip()
        except (OSError, ValueError):
            return ""

    def load_tradovate(self) -> TradovateSecrets:
        """Load Tradovate credentials.

        Lookup order: SSM -> systemd credentials -> env vars.
        """
        params = self._get_all_parameters("tradovate")

        def _get(key: str, env_key: str, default: str = "") -> str:
            return (
                params.get(key)
                or self._read_credential("tradovate", key)
                or os.environ.get(env_key, default)
            )

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
                f"({self._prefix}/{self._env}/tradovate/*), "
                "systemd credentials, or environment variables"
            )

        try:
            cid = int(cid_str)
        except (ValueError, TypeError):
            raise SecretLoadError(f"Invalid Tradovate CID: {cid_str!r}")

        source = "ssm" if params else ("systemd" if CREDENTIALS_DIR else "env")
        logger.info(
            "Loaded Tradovate credentials: user=%s env=%s source=%s",
            username,
            self._env,
            source,
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
        """Load Telegram credentials. Returns None if not configured."""
        params = self._get_all_parameters("telegram")

        bot_token = (
            params.get("bot_token")
            or self._read_credential("telegram", "bot_token")
            or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        )
        chat_id = (
            params.get("chat_id")
            or self._read_credential("telegram", "chat_id")
            or os.environ.get("TELEGRAM_CHAT_ID", "")
        )

        if not bot_token or not chat_id:
            return None

        return TelegramSecrets(bot_token=bot_token, chat_id=chat_id)


class SecretLoadError(Exception):
    """Raised when required secrets cannot be loaded."""
