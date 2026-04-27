"""Tests for BotConfig defaults that must match backtest semantics."""

from bot.bot_config import BotConfig, load_bot_config


def test_session_defaults_match_backtest(tmp_path):
    cfg = BotConfig(
        strategy_dir=str(tmp_path / "strategies"),
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
    )

    assert cfg.session_start == "09:30"
    assert cfg.session_end == "16:00"
    assert cfg.last_entry_time == "15:45"
    assert cfg.cancel_unfilled_time == "15:45"  # cancel at entry cutoff
    assert cfg.flatten_time == "16:00"


def test_tradovate_launch_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("BOT_EXECUTION_BACKEND", "ib_data_tradovate_exec")
    monkeypatch.setenv("BOT_TRADOVATE_ENVIRONMENT", "DEMO")
    monkeypatch.setenv("BOT_TRADOVATE_ACCOUNT_SPEC", "DEMO12345")
    monkeypatch.setenv("BOT_STRATEGY_DIR", str(tmp_path / "strategies"))
    monkeypatch.setenv("BOT_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("BOT_LOG_DIR", str(tmp_path / "logs"))

    cfg = load_bot_config()

    assert cfg.execution_backend == "ib_data_tradovate_exec"
    assert cfg.tradovate_environment == "demo"
    assert cfg.tradovate_account_spec == "DEMO12345"
