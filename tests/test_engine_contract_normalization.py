"""Tests for contract normalization in BotEngine startup."""

import asyncio
from types import SimpleNamespace

from bot.core.engine import BotEngine
from bot.execution.broker_adapter import ContractInfo


class _MockBroker:
    def __init__(self):
        self.calls = []

    async def _get_ib_contract(self, contract):
        self.calls.append(contract)
        return SimpleNamespace(includeExpired=False, conId=int(contract.broker_contract_id))


def _make_engine(execution_backend="ib", ib_conn=None, broker=None):
    engine = BotEngine.__new__(BotEngine)
    engine.config = SimpleNamespace(execution_backend=execution_backend)
    engine.ib_conn = object() if ib_conn is None else ib_conn
    engine.broker = broker or _MockBroker()
    return engine


def test_normalize_runtime_contract_converts_contract_info_for_ib():
    engine = _make_engine()
    contract = ContractInfo(
        symbol="NQ",
        broker_contract_id="12345",
        expiry="202606",
        exchange="CME",
    )

    native = asyncio.run(engine._normalize_runtime_contract(contract))

    assert native.conId == 12345
    assert native.includeExpired is False
    assert engine.broker.calls == [contract]


def test_normalize_runtime_contract_leaves_native_ib_contract_unchanged():
    engine = _make_engine()
    native_contract = SimpleNamespace(includeExpired=False, conId=12345)

    result = asyncio.run(engine._normalize_runtime_contract(native_contract))

    assert result is native_contract
    assert engine.broker.calls == []


def test_normalize_runtime_contract_skips_non_ib_backends():
    broker = _MockBroker()
    engine = _make_engine(execution_backend="tradovate", broker=broker, ib_conn=None)
    contract = ContractInfo(
        symbol="NQ",
        broker_contract_id="12345",
        expiry="202606",
        exchange="CME",
    )

    result = asyncio.run(engine._normalize_runtime_contract(contract))

    assert result is contract
    assert broker.calls == []
