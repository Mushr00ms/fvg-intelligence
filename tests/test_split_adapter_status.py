from types import SimpleNamespace

from bot.execution.split_adapter import SplitAdapter


def test_split_adapter_reports_per_leg_connection_status():
    data = SimpleNamespace(is_connected=False, disconnect_seconds=7.0)
    exec_ = SimpleNamespace(is_connected=True, disconnect_seconds=0.0)
    adapter = SplitAdapter(data_adapter=data, exec_adapter=exec_)

    status = adapter.connection_status()

    assert status["data"] == {
        "broker": "IBKR",
        "connected": False,
        "disconnect_seconds": 7.0,
    }
    assert status["execution"] == {
        "broker": "Tradovate",
        "connected": True,
        "disconnect_seconds": 0.0,
    }
