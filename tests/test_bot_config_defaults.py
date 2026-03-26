"""Tests for BotConfig defaults that must match backtest semantics."""

from bot.bot_config import BotConfig


def test_session_defaults_match_backtest(tmp_path):
    cfg = BotConfig(
        strategy_dir=str(tmp_path / "strategies"),
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
    )

    assert cfg.session_start == "09:30"
    assert cfg.session_end == "16:00"
    assert cfg.last_entry_time == "15:45"
    assert cfg.cancel_unfilled_time == "16:00"
    assert cfg.flatten_time == "16:00"
