"""
broker_factory.py — Create the appropriate BrokerAdapter based on config.

Dispatches on config.execution_backend: "ib", "tradovate", "binance_um".
"""

from __future__ import annotations

from bot.execution.broker_adapter import BrokerAdapter


def create_broker_adapter(config, logger=None, clock=None) -> BrokerAdapter:
    """Factory: instantiate the correct broker adapter from config.

    Args:
        config: BotConfig with execution_backend and broker-specific fields.
        logger: Bot logger instance.
        clock: NTP-corrected clock instance.

    Returns:
        BrokerAdapter implementation.
    """
    backend = config.execution_backend

    if backend == "ib":
        # Lazy import to avoid pulling in ib_async when not needed
        from bot.execution.ib_connection import IBConnection
        # For now, return the IBConnection directly — the full IBAdapter
        # wrapping will come in a subsequent step. Engine still uses
        # IBConnection for IB mode.
        raise NotImplementedError(
            "IBAdapter not yet implemented — use ib_connection directly for IB mode"
        )

    elif backend == "tradovate":
        from bot.execution.tradovate.tradovate_adapter import TradovateAdapter
        return TradovateAdapter(config, bot_logger=logger, clock=clock)

    else:
        raise ValueError(
            f"Unknown execution_backend: {backend!r}. "
            f"Supported: 'ib', 'tradovate'"
        )
