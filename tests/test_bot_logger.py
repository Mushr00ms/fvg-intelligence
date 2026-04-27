from bot.bot_logging.bot_logger import BotLogger


def test_console_log_marks_system_events(tmp_path, capsys):
    logger = BotLogger(str(tmp_path), execution_backend="ib_data_tradovate_exec")

    logger.log("waiting_for_session", seconds=4526)
    logger.close()

    out = capsys.readouterr().out
    assert "[SYS] WAITING_FOR_SESSION" in out


def test_console_log_marks_tradovate_connected(tmp_path, capsys):
    logger = BotLogger(str(tmp_path), execution_backend="ib_data_tradovate_exec")

    logger.log(
        "tradovate_connected",
        environment="demo",
        account="DEMO123",
        order_ws="connected",
        market_data_ws="connected",
    )
    logger.close()

    out = capsys.readouterr().out
    assert "[TV] TRADOVATE_CONNECTED" in out
    assert "environment=demo" in out
    assert "order_ws=connected" in out


def test_console_log_marks_ibkr_data_events(tmp_path, capsys):
    logger = BotLogger(str(tmp_path), execution_backend="ib_data_tradovate_exec")

    logger.log("connection_lost", time="2026-04-27T08:15:00-04:00")
    logger.close()

    out = capsys.readouterr().out
    assert "[IBKR] CONNECTION_LOST" in out
