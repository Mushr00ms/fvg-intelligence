import json

from bot.bot_config import BotConfig
from bot.manager import (
    JsonLogReader,
    ManagedBotProcess,
    ProcessStatus,
    TelegramBotManager,
    _forwardable_error,
    _parse_command,
)


class _FakeTelegram:
    def __init__(self):
        self.enabled = True
        self.messages = []
        self.panel_messages = []
        self.edits = []
        self.callbacks = []

    def send_sync(self, message, reply_markup=None, disable_notification=False):
        self.messages.append(message)
        return True

    def send_message_sync(self, chat_id, message, reply_markup=None, disable_notification=False):
        self.panel_messages.append((chat_id, message, reply_markup, disable_notification))
        return {"message_id": 987}

    def edit_message_text_sync(self, chat_id, message_id, message, reply_markup=None):
        self.edits.append((chat_id, message_id, message, reply_markup))
        return {"message_id": message_id}

    def answer_callback_query_sync(self, callback_query_id, text=None, show_alert=False):
        self.callbacks.append((callback_query_id, text, show_alert))
        return {"ok": True}

    def get_updates_sync(self, offset=None, timeout=0, allowed_updates=None):
        return []


class _FakeProcess:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return True, None

    def stop(self, timeout_seconds):
        self.stopped = True
        return True, None

    def build_command(self):
        return ["python", "-m", "bot.main", "--live", "--no-dry-run"]

    def status(self):
        return ProcessStatus(
            running=self.started and not self.stopped,
            pid=1234 if self.started else None,
            exit_code=0 if self.stopped else None,
            started_at=100.0 if self.started else None,
            stopped_at=200.0 if self.stopped else None,
        )

    def unexpected_exit(self):
        return None

    def is_running(self):
        return self.started and not self.stopped


class _FakeLogReader:
    def recent_records(self, limit=20):
        return [{"ts": "2026-04-06T09:30:00-04:00", "event": "bot_start", "mode": "LIVE"}]

    def read_new_records(self):
        return []

    def start_at_end(self):
        return None


def test_parse_command_strips_prefix_and_bot_suffix():
    command, args = _parse_command("/Start_Bot@prod_manager 25")
    assert command == "start_bot"
    assert args == ["25"]


def test_managed_process_builds_live_real_order_command(tmp_path):
    process = ManagedBotProcess(
        config_path=str(tmp_path / "bot_config.json"),
        bot_live=True,
        bot_dry_run=False,
        cwd=str(tmp_path),
    )

    cmd = process.build_command()

    assert cmd[1:3] == ["-m", "bot.main"]
    assert "--config" in cmd
    assert "--live" in cmd
    assert "--no-dry-run" in cmd


def test_json_log_reader_recent_records_and_incremental_tail(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_file = log_dir / "2026-04-06.jsonl"

    log_file.write_text(
        json.dumps({"ts": "2026-04-06T09:30:00-04:00", "event": "bot_start"}) + "\n"
        + json.dumps({"ts": "2026-04-06T09:31:00-04:00", "event": "startup_complete"}) + "\n",
        encoding="utf-8",
    )

    reader = JsonLogReader(log_dir)
    recent = reader.recent_records(limit=1)
    assert recent == [{"ts": "2026-04-06T09:31:00-04:00", "event": "startup_complete"}]

    reader.start_at_end()
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": "2026-04-06T09:32:00-04:00", "event": "bot_error", "error": "boom"}) + "\n")

    new_records = reader.read_new_records()
    assert new_records == [{
        "ts": "2026-04-06T09:32:00-04:00",
        "event": "bot_error",
        "error": "boom",
    }]


def test_forwardable_error_skips_telegram_failures():
    assert _forwardable_error({"event": "bot_error", "error": "crash"}) is True
    assert _forwardable_error({"event": "eod_reconcile_error", "error": "failed"}) is True
    assert _forwardable_error({"event": "telegram_error", "error": "network"}) is False


def test_manager_handles_status_start_stop_and_logs(tmp_path):
    cfg = BotConfig(
        telegram_bot_token="token",
        telegram_chat_id="123",
        strategy_dir=str(tmp_path / "strategies"),
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
    )
    manager = TelegramBotManager(
        config=cfg,
        config_path=str(tmp_path / "bot_config.json"),
        bot_live=True,
        bot_dry_run=False,
    )
    manager._telegram = _FakeTelegram()
    manager._bot_process = _FakeProcess()
    manager._log_reader = _FakeLogReader()
    manager._publish_control_panel()

    manager._handle_update({
        "update_id": 1,
        "message": {"chat": {"id": 123}, "text": "/status", "from": {"is_bot": False}},
    })
    manager._handle_update({
        "update_id": 2,
        "message": {"chat": {"id": 123}, "text": "/start", "from": {"is_bot": False}},
    })
    manager._handle_update({
        "update_id": 3,
        "message": {"chat": {"id": 123}, "text": "/logs 5", "from": {"is_bot": False}},
    })
    manager._handle_update({
        "update_id": 4,
        "message": {"chat": {"id": 123}, "text": "/stop", "from": {"is_bot": False}},
    })

    assert "NQ IB bot status" in manager._telegram.messages[0]
    assert "start requested" in manager._telegram.messages[1]
    assert "Recent NQ bot logs" in manager._telegram.messages[2]
    assert "NQ IB bot stopped" in manager._telegram.messages[3]
    assert manager._telegram.panel_messages[0][0] == "123"
    assert "NQ IB manager panel" in manager._telegram.panel_messages[0][1]


def test_manager_handles_panel_callbacks(tmp_path):
    cfg = BotConfig(
        telegram_bot_token="token",
        telegram_chat_id="123",
        strategy_dir=str(tmp_path / "strategies"),
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
    )
    manager = TelegramBotManager(
        config=cfg,
        config_path=str(tmp_path / "bot_config.json"),
        bot_live=True,
        bot_dry_run=False,
    )
    manager._telegram = _FakeTelegram()
    manager._bot_process = _FakeProcess()
    manager._log_reader = _FakeLogReader()
    manager._panel_message_id = 987

    manager._handle_update({
        "update_id": 10,
        "callback_query": {
            "id": "cb1",
            "data": "start",
            "message": {"message_id": 987, "chat": {"id": 123}},
        },
    })
    manager._handle_update({
        "update_id": 11,
        "callback_query": {
            "id": "cb2",
            "data": "logs",
            "message": {"message_id": 987, "chat": {"id": 123}},
        },
    })

    assert manager._bot_process.started is True
    assert any("Recent NQ bot logs" in msg for msg in manager._telegram.messages)
    assert ("cb1", "NQ bot start requested", False) in manager._telegram.callbacks
    assert ("cb2", "Sending last 20 log lines", False) in manager._telegram.callbacks
