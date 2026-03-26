import sys
import types

import pandas as pd
import pytest

sys.modules.setdefault("numba", types.SimpleNamespace(njit=lambda *args, **kwargs: (lambda fn: fn)))

from bot.backtest.backtester import _summarize_trade_exit, walk_trade_1s


def _make_1s_bars(rows):
    return pd.DataFrame(
        {
            "date": pd.to_datetime([row[0] for row in rows]),
            "open": [row[1] for row in rows],
            "high": [row[2] for row in rows],
            "low": [row[3] for row in rows],
            "close": [row[4] for row in rows],
        }
    )


def test_runner_trail_ratchets_stop_from_breakeven():
    df = _make_1s_bars(
        [
            ("2026-03-22 09:30:01", 100.0, 110.0, 100.0, 109.0),
            ("2026-03-22 09:30:02", 109.0, 113.0, 108.5, 112.0),
            ("2026-03-22 09:30:03", 112.0, 112.0, 107.0, 107.5),
        ]
    )

    result = walk_trade_1s(
        df,
        pd.Timestamp("2026-03-22 09:30:00"),
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        side="BUY",
        tp_mode="runner-trail",
    )

    assert result == ("2026-03-22 09:30:03", 108.0, "TRAIL", True, 3.0)


def test_split_mode_uses_fixed_tp_for_single_contract():
    summary = _summarize_trade_exit(
        side="BUY",
        entry_price=100.0,
        target_price=110.0,
        contracts=1,
        tp_mode="split",
        walk_result=("2026-03-22 09:30:01", 110.0, "TP"),
    )

    assert summary["exit_reason"] == "TP"
    assert summary["pnl_pts"] == 10.0
    assert summary["tp_touched"] is True
    assert summary["tp_exit_contracts"] == 0
    assert summary["runner_contracts"] == 0


def test_split_mode_blends_tp_contracts_with_runner_contract():
    summary = _summarize_trade_exit(
        side="BUY",
        entry_price=100.0,
        target_price=110.0,
        contracts=3,
        tp_mode="split",
        walk_result=("2026-03-22 09:30:03", 108.0, "TRAIL", True, 3.0),
    )

    assert summary["exit_reason"] == "SPLIT"
    assert summary["exit_price"] == 108.0
    assert summary["runner_exit_reason"] == "TRAIL"
    assert summary["runner_exit_price"] == 108.0
    assert summary["tp_exit_contracts"] == 2
    assert summary["runner_contracts"] == 1
    assert summary["pnl_pts"] == pytest.approx((10.0 * 2 + 8.0) / 3)
    assert summary["excursion_pts"] == 3.0
