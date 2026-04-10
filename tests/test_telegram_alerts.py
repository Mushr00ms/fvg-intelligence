import asyncio
import os
import tempfile

from bot.alerts.telegram import TelegramAlerter
from bot.db import TradeDB


def _make_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db = TradeDB(tmp.name)
    return db, tmp.name


def test_send_queued_returns_true_and_marks_sent(monkeypatch):
    db, path = _make_db()
    try:
        alerter = TelegramAlerter("token", "chat", db=db)

        async def _ok(_message):
            return True

        monkeypatch.setattr(alerter, "_try_send", _ok)

        sent = asyncio.run(alerter.send_queued("reconciliation", "hello"))

        assert sent is True
        unsent = db.get_unsent_alerts(limit=10)
        assert unsent == []
        sent_rows = db.query("SELECT sent FROM alerts")
        assert sent_rows[0]["sent"] == 1
    finally:
        os.unlink(path)


def test_alert_reconciliation_returns_false_when_delivery_fails(monkeypatch):
    db, path = _make_db()
    try:
        alerter = TelegramAlerter("token", "chat", db=db)

        async def _fail(_message):
            return False

        monkeypatch.setattr(alerter, "_try_send", _fail)

        sent = asyncio.run(alerter.alert_reconciliation("report"))

        assert sent is False
        unsent = db.get_unsent_alerts(limit=10)
        assert len(unsent) == 1
        assert unsent[0]["event_type"] == "reconciliation"
    finally:
        os.unlink(path)
