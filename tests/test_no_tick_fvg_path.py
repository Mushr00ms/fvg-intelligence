"""Regression test: tick-path FVG detection must NOT exist in BotEngine.

The tick path was removed because it could synthesize FVGs from intra-bar
OHLC values that diverged from the canonical, post-close 5min bar — breaking
live/backtest parity (see EOD reconciliation 2026-04-07). FVG detection is
now driven exclusively by `_on_5min_update` consuming IB's settled bar.

If anyone reintroduces the tick path under any name, this test fails.
"""
from bot.core import engine as engine_mod


def test_engine_has_no_tick_fvg_path():
    forbidden_attrs = (
        "_on_tick_bar_complete",
        "_tick_detected_bars",
        "_tick_bar_builder",
    )
    for name in forbidden_attrs:
        assert not hasattr(engine_mod.BotEngine, name), (
            f"BotEngine.{name} reintroduced — tick FVG path breaks live/BT parity"
        )

    src = open(engine_mod.__file__).read()
    assert "detect_from_tick_bar" not in src, (
        "engine.py must not call fvg_mgr.detect_from_tick_bar — canonical bar only"
    )
    assert "fvg_detected_tick" not in src, (
        "engine.py must not emit fvg_detected_tick events"
    )
    assert "TickBarBuilder" not in src, (
        "engine.py must not import TickBarBuilder"
    )
