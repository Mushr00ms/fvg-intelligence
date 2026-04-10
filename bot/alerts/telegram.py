"""
telegram.py — Telegram Bot API alerts for critical trading events.

Sends messages via HTTP POST to the Telegram Bot API.
Fire-and-forget: errors are logged but don't block the event loop.
"""

import asyncio
import json
from urllib.request import Request, urlopen
from urllib.error import URLError


TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramAlerter:
    """
    Send alerts to Telegram via the Bot API.

    Setup:
    1. Create a bot via @BotFather on Telegram
    2. Get the bot token
    3. Get your chat_id (message @userinfobot)
    4. Set bot_token and chat_id in bot_config.json
    """

    def __init__(self, bot_token, chat_id, logger=None, db=None):
        self._token = bot_token
        self._chat_id = chat_id
        self._logger = logger
        self._db = db  # TradeDB for alert queue (optional)
        self._enabled = bool(bot_token and chat_id)

    @property
    def enabled(self):
        return self._enabled

    def _api_call(self, method, payload, timeout=10):
        """Call the Telegram Bot API and return the decoded result payload."""
        if not self._enabled:
            return None
        url = TELEGRAM_API.format(token=self._token, method=method)
        body = json.dumps(payload).encode("utf-8")
        try:
            req = Request(url, data=body, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            if not raw.get("ok"):
                description = raw.get("description", "telegram api call failed")
                if "message is not modified" in description.lower():
                    return {"ok": True, "unchanged": True}
                raise RuntimeError(description)
            return raw.get("result")
        except (URLError, OSError) as e:
            if self._logger:
                self._logger.log("telegram_error", error=str(e))
            return None
        except Exception as e:
            if self._logger:
                self._logger.log("telegram_error", error=str(e))
            return None

    def send_sync(self, message, reply_markup=None, disable_notification=False):
        """
        Send a message synchronously. Use for critical alerts where
        delivery confirmation matters (e.g., kill switch).
        """
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_notification": disable_notification,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        result = self._api_call("sendMessage", payload)
        return bool(result)

    def send_message_sync(self, chat_id, message, reply_markup=None, disable_notification=False):
        """Send a message to an explicit Telegram chat."""
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_notification": disable_notification,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return self._api_call("sendMessage", payload)

    def edit_message_text_sync(self, chat_id, message_id, message, reply_markup=None):
        """Edit an existing bot-authored message."""
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": message,
            "parse_mode": "HTML",
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return self._api_call("editMessageText", payload)

    def answer_callback_query_sync(self, callback_query_id, text=None, show_alert=False):
        """Acknowledge an inline keyboard callback."""
        payload = {
            "callback_query_id": callback_query_id,
            "show_alert": show_alert,
        }
        if text:
            payload["text"] = text
        return self._api_call("answerCallbackQuery", payload)

    def get_updates_sync(self, offset=None, timeout=0, allowed_updates=None):
        """Poll Telegram updates synchronously for manager command handling."""
        payload = {
            "timeout": timeout,
        }
        if offset is not None:
            payload["offset"] = offset
        if allowed_updates:
            payload["allowed_updates"] = allowed_updates
        result = self._api_call("getUpdates", payload, timeout=max(timeout + 5, 10))
        return result or []

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

    async def send_queued(self, event_type, message):
        """Send with DB-backed queue for retry on failure.

        Falls back to fire-and-forget if DB is not configured.
        """
        if self._db:
            self._db.queue_alert(event_type, message)

        success = await self._try_send(message)
        if success and self._db:
            # Mark the most recent unsent alert as sent
            unsent = self._db.get_unsent_alerts(limit=1)
            if unsent:
                self._db.mark_alert_sent(unsent[0]['id'])
        return success

    async def _try_send(self, message):
        """Attempt to send, return True on success."""
        if not self._enabled:
            return False
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self.send_sync, message)
            return result
        except Exception as e:
            if self._logger:
                self._logger.log("telegram_error", error=str(e))
            return False

    async def retry_unsent(self):
        """Retry delivery of queued but unsent alerts. Call periodically from engine."""
        if not self._db or not self._enabled:
            return 0

        unsent = self._db.get_unsent_alerts(max_attempts=3, limit=10)
        sent_count = 0
        for alert in unsent:
            success = await self._try_send(alert['message'])
            if success:
                self._db.mark_alert_sent(alert['id'])
                sent_count += 1
            else:
                self._db.mark_alert_failed(alert['id'], "send_failed")
        return sent_count

    async def alert_kill_switch(self, reason, daily_pnl, balance):
        """Send kill switch alert (queued for guaranteed delivery)."""
        msg = (
            "<b>KILL SWITCH ACTIVATED</b>\n\n"
            f"Reason: {reason}\n"
            f"Daily P&L: ${daily_pnl:,.0f}\n"
            f"Balance: ${balance:,.0f}"
        )
        await self.send_queued("kill_switch", msg)

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
            f"Trades: {daily_state.filled_trade_count}\n"
            f"P&L: {emoji}${pnl:,.0f} ({emoji}{pnl_pct:.1f}%)\n"
            f"Balance: ${daily_state.start_balance + pnl:,.0f}"
        )
        await self.send(msg)

    async def alert_reconciliation(self, report_html):
        """Send EOD reconciliation report (queued for guaranteed delivery)."""
        return await self.send_queued("reconciliation", report_html)

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
