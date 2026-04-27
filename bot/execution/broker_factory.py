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
        from bot.execution.ib_adapter import IBAdapter
        return IBAdapter(config, bot_logger=logger, clock=clock)

    elif backend == "tradovate":
        from bot.execution.tradovate.tradovate_adapter import TradovateAdapter
        return TradovateAdapter(config, bot_logger=logger, clock=clock)

    elif backend == "ib_data_tradovate_exec":
        from bot.execution.ib_adapter import IBAdapter
        from bot.execution.tradovate.tradovate_adapter import TradovateAdapter
        from bot.execution.split_adapter import SplitAdapter
        data = IBAdapter(config, bot_logger=logger, clock=clock)
        exec_ = TradovateAdapter(config, bot_logger=logger, clock=clock, execution_only=True)
        return SplitAdapter(data_adapter=data, exec_adapter=exec_, bot_logger=logger)

    else:
        raise ValueError(
            f"Unknown execution_backend: {backend!r}. "
            f"Supported: 'ib', 'tradovate', 'ib_data_tradovate_exec'"
        )
