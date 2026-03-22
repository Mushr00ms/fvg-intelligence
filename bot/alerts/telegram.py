"""
telegram.py — Telegram Bot API alerts for critical trading events.

Sends messages via HTTP POST to the Telegram Bot API.
Fire-and-forget: errors are logged but don't block the event loop.
"""

import asyncio
import json
from urllib.request import Request, urlopen
from urllib.error import URLError


TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlerter:
    """
    Send alerts to Telegram via the Bot API.

    Setup:
    1. Create a bot via @BotFather on Telegram
    2. Get the bot token
    3. Get your chat_id (message @userinfobot)
    4. Set bot_token and chat_id in bot_config.json
    """

    def __init__(self, bot_token, chat_id, logger=None):
        self._token = bot_token
        self._chat_id = chat_id
        self._logger = logger
        self._enabled = bool(bot_token and chat_id)

    @property
    def enabled(self):
        return self._enabled

    def send_sync(self, message):
        """
        Send a message synchronously. Use for critical alerts where
        delivery confirmation matters (e.g., kill switch).
        """
        if not self._enabled:
            return False

        url = TELEGRAM_API.format(token=self._token)
        payload = json.dumps({
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "HTML",
        }).encode("utf-8")

        try:
            req = Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except (URLError, OSError) as e:
            if self._logger:
                self._logger.log("telegram_error", error=str(e))
            return False

    async def send(self, message):
        """
        Send a message asynchronously (runs in thread pool to avoid blocking).
        Fire-and-forget: errors logged but don't propagate.
        """
        if not self._enabled:
            return

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self.send_sync, message)
        except Exception as e:
            if self._logger:
                self._logger.log("telegram_error", error=str(e))

    async def alert_kill_switch(self, reason, daily_pnl, balance):
        """Send kill switch alert."""
        msg = (
            "<b>KILL SWITCH ACTIVATED</b>\n\n"
            f"Reason: {reason}\n"
            f"Daily P&L: ${daily_pnl:,.0f}\n"
            f"Balance: ${balance:,.0f}"
        )
        await self.send(msg)

    async def alert_connection_lost(self, downtime_seconds):
        """Send connection lost alert."""
        msg = (
            "<b>CONNECTION LOST</b>\n\n"
            f"Downtime: {downtime_seconds:.0f}s\n"
            "Attempting reconnect..."
        )
        await self.send(msg)

    async def alert_order_rejected(self, group_id, reason):
        """Send order rejection alert."""
        msg = (
            "<b>ORDER REJECTED</b>\n\n"
            f"Group: {group_id}\n"
            f"Reason: {reason}"
        )
        await self.send(msg)

    async def alert_daily_summary(self, daily_state):
        """Send end-of-day summary."""
        pnl = daily_state.realized_pnl
        pnl_pct = daily_state.daily_pnl_pct * 100
        emoji = "+" if pnl >= 0 else ""
        msg = (
            "<b>DAILY SUMMARY</b>\n\n"
            f"Date: {daily_state.date}\n"
            f"Trades: {daily_state.trade_count}\n"
            f"P&L: {emoji}${pnl:,.0f} ({emoji}{pnl_pct:.1f}%)\n"
            f"Balance: ${daily_state.start_balance + pnl:,.0f}"
        )
        await self.send(msg)

    async def alert_bot_start(self, config, strategy_id):
        """Send bot startup notification."""
        mode = "PAPER" if config.paper_mode else "LIVE"
        dry = " (DRY RUN)" if config.dry_run else ""
        msg = (
            f"<b>BOT STARTED</b> [{mode}{dry}]\n\n"
            f"Strategy: {strategy_id}\n"
            f"Port: {config.ib_port}"
        )
        await self.send(msg)
