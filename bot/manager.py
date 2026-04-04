"""
manager.py - Telegram-driven process manager for the NQ Interactive Brokers bot.

Usage:
    python -m bot.manager
    python -m bot.manager --live --no-dry-run
    python -m bot.manager --config path\\to\\bot_config.json --live --no-dry-run
"""

import argparse
import html
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from bot.alerts.telegram import TelegramAlerter
from bot.bot_config import default_config_path, load_bot_config


ALLOWED_UPDATE_TYPES = [
    "message",
    "channel_post",
    "edited_message",
    "edited_channel_post",
    "callback_query",
]
IGNORED_FORWARD_EVENTS = {"telegram_error", "alert_retry_error"}
PANEL_ACTIONS = {"start", "stop", "status", "logs", "panel"}


def _format_duration(seconds):
    seconds = max(int(seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _extract_message(update):
    for key in ALLOWED_UPDATE_TYPES:
        if key in update:
            return update[key]
    return None


def _parse_command(text):
    if not text:
        return "", []
    parts = text.strip().split()
    if not parts:
        return "", []
    command = parts[0].lower().lstrip("/")
    if "@" in command:
        command = command.split("@", 1)[0]
    return command, parts[1:]


def _render_log_record(record):
    ts = str(record.get("ts", ""))
    event = str(record.get("event", "unknown"))
    details = []
    for key in sorted(record):
        if key in {"ts", "event"}:
            continue
        details.append(f"{key}={record[key]}")
    body = f"{event} {' '.join(details)}".strip()
    if ts:
        return f"{ts} {body}"
    return body


def _forwardable_error(record):
    event = str(record.get("event", ""))
    if not event or event in IGNORED_FORWARD_EVENTS:
        return False
    return event == "bot_error" or event.endswith("_error")


@dataclass
class ProcessStatus:
    running: bool
    pid: int | None
    exit_code: int | None
    started_at: float | None
    stopped_at: float | None


class JsonLogReader:
    """Read structured JSONL bot logs for status, tail, and error forwarding."""

    def __init__(self, log_dir):
        self._log_dir = Path(log_dir)
        self._offsets = {}

    def start_at_end(self):
        for path in self._base_log_files():
            try:
                self._offsets[str(path)] = path.stat().st_size
            except OSError:
                continue

    def recent_records(self, limit=20):
        recent = deque(maxlen=limit)
        for path in self._recent_log_files():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for raw in fh:
                        record = self._parse_record(raw)
                        if record is not None:
                            recent.append(record)
            except OSError:
                continue
        return list(recent)

    def read_new_records(self):
        new_records = []
        for path in self._base_log_files():
            path_key = str(path)
            try:
                size = path.stat().st_size
            except OSError:
                self._offsets.pop(path_key, None)
                continue

            start = self._offsets.get(path_key, 0)
            if start > size:
                start = 0

            try:
                with path.open("r", encoding="utf-8") as fh:
                    fh.seek(start)
                    for raw in fh:
                        record = self._parse_record(raw)
                        if record is not None:
                            new_records.append(record)
                    self._offsets[path_key] = fh.tell()
            except OSError:
                continue
        return new_records

    def _recent_log_files(self):
        return sorted(self._base_log_files(), key=lambda p: p.name)

    def _base_log_files(self):
        if not self._log_dir.exists():
            return []
        return sorted(self._log_dir.glob("*.jsonl"))

    @staticmethod
    def _parse_record(raw_line):
        raw_line = raw_line.strip()
        if not raw_line:
            return None
        try:
            return json.loads(raw_line)
        except json.JSONDecodeError:
            return {"event": "raw_log_line", "line": raw_line}


class ManagedBotProcess:
    """Own the child `python -m bot.main` process for the NQ IB bot."""

    def __init__(self, config_path, bot_live, bot_dry_run, cwd):
        self._config_path = config_path
        self._bot_live = bot_live
        self._bot_dry_run = bot_dry_run
        self._cwd = cwd
        self._process = None
        self._started_at = None
        self._stopped_at = None
        self._stop_requested = False
        self._exit_notified = False

    def build_command(self):
        cmd = [sys.executable, "-m", "bot.main"]
        if self._config_path:
            cmd.extend(["--config", self._config_path])
        if self._bot_live:
            cmd.append("--live")
        if not self._bot_dry_run:
            cmd.append("--no-dry-run")
        return cmd

    def start(self):
        if self.is_running():
            return False, "already running"

        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self._process = subprocess.Popen(
            self.build_command(),
            cwd=self._cwd,
            creationflags=creationflags,
        )
        self._started_at = time.time()
        self._stopped_at = None
        self._stop_requested = False
        self._exit_notified = False
        return True, None

    def stop(self, timeout_seconds):
        if not self.is_running():
            return False, "not running"

        self._stop_requested = True
        self._send_graceful_stop()

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if not self.is_running():
                break
            time.sleep(0.25)

        if self.is_running():
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

        self._stopped_at = time.time()
        return True, None

    def is_running(self):
        return self._process is not None and self._process.poll() is None

    def status(self):
        exit_code = None
        if self._process is not None:
            exit_code = self._process.poll()
            if exit_code is not None and self._stopped_at is None:
                self._stopped_at = time.time()
        return ProcessStatus(
            running=self.is_running(),
            pid=self._process.pid if self._process is not None else None,
            exit_code=exit_code,
            started_at=self._started_at,
            stopped_at=self._stopped_at,
        )

    def unexpected_exit(self):
        if self._process is None or self.is_running():
            return None
        if self._stop_requested or self._exit_notified:
            return None

        exit_code = self._process.poll()
        self._stopped_at = self._stopped_at or time.time()
        if exit_code == 0:
            self._exit_notified = True
            return None

        self._exit_notified = True
        return exit_code

    def _send_graceful_stop(self):
        if not self.is_running():
            return
        if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
            try:
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
                return
            except (OSError, ValueError):
                pass
        try:
            self._process.send_signal(signal.SIGTERM)
        except (OSError, ValueError):
            pass


class TelegramBotManager:
    """Telegram command loop supervising the NQ IB bot child process."""

    def __init__(self, config, config_path, bot_live, bot_dry_run):
        self._config = config
        self._config_path = config_path
        self._poll_seconds = max(1, int(config.telegram_manager_poll_seconds))
        self._stop_timeout = max(1, int(config.telegram_manager_stop_timeout))
        self._default_log_lines = max(1, int(config.telegram_manager_log_lines))
        self._authorized_chat_id = str(config.telegram_chat_id)
        self._telegram = TelegramAlerter(
            config.telegram_bot_token,
            config.telegram_chat_id,
        )
        self._bot_process = ManagedBotProcess(
            config_path=config_path,
            bot_live=bot_live,
            bot_dry_run=bot_dry_run,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        self._log_reader = JsonLogReader(config.log_dir)
        self._offset = None
        self._shutdown = False
        self._bot_live = bot_live
        self._bot_dry_run = bot_dry_run
        self._panel_chat_id = self._authorized_chat_id
        self._panel_message_id = None

    def run(self):
        if not self._telegram.enabled:
            raise RuntimeError("Telegram manager requires telegram_bot_token and telegram_chat_id")

        self._prime_updates_offset()
        self._log_reader.start_at_end()
        self._publish_control_panel()

        try:
            while not self._shutdown:
                self._poll_updates()
                self._forward_new_errors()
                self._notify_unexpected_exit()
                time.sleep(self._poll_seconds)
        finally:
            if self._bot_process.is_running():
                self._bot_process.stop(self._stop_timeout)

    def shutdown(self):
        self._shutdown = True

    def _prime_updates_offset(self):
        updates = self._telegram.get_updates_sync(
            offset=None,
            timeout=0,
            allowed_updates=ALLOWED_UPDATE_TYPES,
        )
        if updates:
            self._offset = updates[-1]["update_id"] + 1

    def _poll_updates(self):
        updates = self._telegram.get_updates_sync(
            offset=self._offset,
            timeout=0,
            allowed_updates=ALLOWED_UPDATE_TYPES,
        )
        for update in updates:
            self._offset = update["update_id"] + 1
            self._handle_update(update)

    def _handle_update(self, update):
        if "callback_query" in update:
            self._handle_callback_query(update["callback_query"])
            return

        message = _extract_message(update)
        if not message:
            return

        if message.get("from", {}).get("is_bot"):
            return

        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != self._authorized_chat_id:
            return

        command, args = _parse_command(message.get("text") or message.get("caption") or "")
        if not command:
            return

        if command in {"help", "commands"}:
            self._send_message(self._help_message())
            return
        if command in {"panel", "menu"}:
            self._publish_control_panel(force_new=True)
            return
        if command in {"status", "health"}:
            self._send_message(self._status_message())
            return
        if command in {"start", "startbot", "start_bot", "run"}:
            self._handle_start()
            return
        if command in {"stop", "stopbot", "stop_bot", "halt"}:
            self._handle_stop()
            return
        if command in {"logs", "tail"}:
            self._handle_logs(args)
            return

        self._send_message(
            "<b>Unknown command</b>\nUse <code>/help</code> to see the supported manager commands."
        )

    def _handle_callback_query(self, callback):
        message = callback.get("message") or {}
        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != self._authorized_chat_id:
            self._telegram.answer_callback_query_sync(
                callback.get("id"),
                text="Unauthorized chat",
                show_alert=True,
            )
            return

        data = str(callback.get("data", "")).strip().lower()
        if data not in PANEL_ACTIONS:
            self._telegram.answer_callback_query_sync(
                callback.get("id"),
                text="Unknown action",
            )
            return

        self._panel_chat_id = chat_id
        self._panel_message_id = message.get("message_id")

        if data == "start":
            self._handle_start(panel_callback_id=callback.get("id"))
            return
        if data == "stop":
            self._handle_stop(panel_callback_id=callback.get("id"))
            return
        if data == "logs":
            self._handle_logs([], callback_query_id=callback.get("id"))
            return
        if data in {"status", "panel"}:
            self._refresh_control_panel()
            self._telegram.answer_callback_query_sync(
                callback.get("id"),
                text="Panel refreshed",
            )
            return

    def _handle_start(self, panel_callback_id=None):
        started, reason = self._bot_process.start()
        if not started:
            self._refresh_control_panel()
            if panel_callback_id:
                self._telegram.answer_callback_query_sync(
                    panel_callback_id,
                    text="NQ bot already running",
                )
            else:
                self._send_message(
                    f"<b>NQ bot already running</b>\n{html.escape(self._status_line())}"
                )
            return
        self._refresh_control_panel()
        if panel_callback_id:
            self._telegram.answer_callback_query_sync(
                panel_callback_id,
                text="NQ bot start requested",
            )
            return
        self._send_message(
            "<b>NQ IB bot start requested</b>\n"
            f"Mode: {html.escape(self._mode_label())}\n"
            f"Command: <code>{html.escape(' '.join(self._bot_process.build_command()))}</code>"
        )

    def _handle_stop(self, panel_callback_id=None):
        stopped, reason = self._bot_process.stop(self._stop_timeout)
        if not stopped:
            self._refresh_control_panel()
            if panel_callback_id:
                self._telegram.answer_callback_query_sync(
                    panel_callback_id,
                    text="NQ bot is not running",
                )
            else:
                self._send_message("<b>NQ bot is not running</b>")
            return
        self._refresh_control_panel()
        if panel_callback_id:
            self._telegram.answer_callback_query_sync(
                panel_callback_id,
                text="NQ bot stopped",
            )
            return
        self._send_message("<b>NQ IB bot stopped</b>")

    def _handle_logs(self, args, callback_query_id=None):
        limit = self._default_log_lines
        if args:
            try:
                limit = max(1, min(int(args[0]), 50))
            except ValueError:
                pass

        records = self._log_reader.recent_records(limit=limit)
        if not records:
            if callback_query_id:
                self._telegram.answer_callback_query_sync(
                    callback_query_id,
                    text="No bot logs available yet",
                )
            self._send_message("<b>No bot logs available yet</b>")
            return

        rendered = "\n".join(_render_log_record(r) for r in records[-limit:])
        if callback_query_id:
            self._telegram.answer_callback_query_sync(
                callback_query_id,
                text=f"Sending last {limit} log lines",
            )
        self._send_preformatted("Recent NQ bot logs", rendered)

    def _forward_new_errors(self):
        for record in self._log_reader.read_new_records():
            if not _forwardable_error(record):
                continue
            rendered = _render_log_record(record)
            self._send_preformatted("NQ bot error", rendered)
            self._refresh_control_panel()

    def _notify_unexpected_exit(self):
        exit_code = self._bot_process.unexpected_exit()
        if exit_code is None:
            return
        self._refresh_control_panel()
        self._send_message(
            "<b>NQ IB bot exited unexpectedly</b>\n"
            f"Exit code: <code>{exit_code}</code>"
        )

    def _send_message(self, message):
        self._telegram.send_sync(message)

    def _send_preformatted(self, title, body):
        safe_body = html.escape(body)
        max_body = 3400
        if len(safe_body) > max_body:
            safe_body = safe_body[-max_body:]
            safe_body = "...\n" + safe_body
        self._send_message(f"<b>{html.escape(title)}</b>\n<pre>{safe_body}</pre>")

    def _publish_control_panel(self, force_new=False):
        panel_text = self._manager_online_message()
        panel_markup = self._panel_markup()

        if not force_new and self._panel_message_id is not None:
            edited = self._telegram.edit_message_text_sync(
                self._panel_chat_id,
                self._panel_message_id,
                panel_text,
                reply_markup=panel_markup,
            )
            if edited is not None:
                return

        sent = self._telegram.send_message_sync(
            self._panel_chat_id,
            panel_text,
            reply_markup=panel_markup,
            disable_notification=True,
        )
        if sent:
            self._panel_message_id = sent.get("message_id")

    def _refresh_control_panel(self):
        self._publish_control_panel(force_new=False)

    def _panel_markup(self):
        return {
            "inline_keyboard": [
                [
                    {"text": "Start", "callback_data": "start"},
                    {"text": "Stop", "callback_data": "stop"},
                ],
                [
                    {"text": "Status", "callback_data": "status"},
                    {"text": f"Logs ({self._default_log_lines})", "callback_data": "logs"},
                ],
            ]
        }

    def _help_message(self):
        return (
            "<b>NQ IB bot manager</b>\n"
            "<code>/panel</code> repost the button panel\n"
            "<code>/status</code> current process state\n"
            "<code>/start</code> start the managed NQ bot\n"
            "<code>/stop</code> stop the managed NQ bot\n"
            f"<code>/logs [{self._default_log_lines}]</code> recent structured bot logs"
        )

    def _status_message(self):
        last_record = self._log_reader.recent_records(limit=1)
        last_line = _render_log_record(last_record[-1]) if last_record else "no bot logs yet"
        return (
            "<b>NQ IB bot status</b>\n"
            f"{html.escape(self._status_line())}\n"
            f"Mode: {html.escape(self._mode_label())}\n"
            f"Last log: <code>{html.escape(last_line)}</code>"
        )

    def _status_line(self):
        status = self._bot_process.status()
        if status.running:
            uptime = _format_duration(time.time() - status.started_at) if status.started_at else "unknown"
            return f"RUNNING pid={status.pid} uptime={uptime}"
        if status.pid is None:
            return "STOPPED pid=none"
        if status.exit_code is None:
            return f"STOPPED pid={status.pid}"
        return f"STOPPED pid={status.pid} exit={status.exit_code}"

    def _mode_label(self):
        market_mode = "LIVE" if self._bot_live else "PAPER"
        order_mode = "REAL ORDERS" if not self._bot_dry_run else "DRY RUN"
        return f"{market_mode} / {order_mode}"

    def _manager_online_message(self):
        return (
            "<b>NQ IB manager panel</b>\n"
            f"State: <code>{html.escape(self._status_line())}</code>\n"
            f"Launch mode: <code>{html.escape(self._mode_label())}</code>\n"
            f"Logs button: last <code>{self._default_log_lines}</code> lines\n"
            "Fallback commands: <code>/panel</code> <code>/status</code> <code>/start</code> "
            f"<code>/stop</code> <code>/logs {self._default_log_lines}</code>"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Telegram manager for the NQ IB bot")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to bot_config.json (default: bot/bot_config.json)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Start the managed bot in live mode (port 7496)."
    )
    parser.add_argument(
        "--no-dry-run", action="store_true",
        help="Start the managed bot with real order placement enabled."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config or default_config_path()
    config = load_bot_config(config_path if os.path.exists(config_path) else None)

    manager = TelegramBotManager(
        config=config,
        config_path=config_path if os.path.exists(config_path) else None,
        bot_live=args.live or not config.paper_mode,
        bot_dry_run=not args.no_dry_run if args.no_dry_run else config.dry_run,
    )

    def _signal_handler(sig, frame):
        manager.shutdown()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _signal_handler)

    manager.run()


if __name__ == "__main__":
    main()
