"""
ib_connection.py — IB TWS connection wrapper with auto-reconnect.

Uses ib_async (maintained fork of ib_insync) for async TWS communication.
"""

import asyncio
from datetime import datetime, timedelta

import pytz

NY_TZ = pytz.timezone("America/New_York")


class IBConnection:
    """
    Wrapper around ib_async.IB() providing:
    - Connect/disconnect
    - Auto-reconnect loop on disconnect
    - Connection health monitoring
    - Disconnect duration tracking
    """

    def __init__(self, host, port, client_id, logger=None, clock=None):
        self._host = host
        self._port = port
        self._client_id = client_id
        self._logger = logger
        self._clock = clock
        self._connected = False
        self._disconnect_time = None
        self._reconnect_task = None
        self._ib = None
        self._reconnect_interval = 10  # seconds
        self._reconnect_callbacks = []
        self._initial_connect_done = False

    def _now(self):
        if self._clock is not None:
            return self._clock.now()
        return datetime.now(NY_TZ)

    async def connect(self):
        """Connect to TWS. Registers disconnect callback."""
        from ib_async import IB

        if self._ib is None:
            self._ib = IB()

        self._ib.disconnectedEvent += self._on_disconnected
        self._ib.connectedEvent += self._on_connected

        await self._ib.connectAsync(
            self._host, self._port, clientId=self._client_id,
            timeout=30,
        )
        self._connected = True
        self._disconnect_time = None
        self._initial_connect_done = True

        if self._logger:
            self._logger.log(
                "bot_start",
                host=self._host,
                port=self._port,
                client_id=self._client_id,
            )

    async def disconnect(self):
        """Graceful disconnect from TWS."""
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        if self._ib and self._ib.isConnected():
            self._ib.disconnect()
        self._connected = False

    def _on_disconnected(self):
        """Called by ib_async on unexpected disconnect."""
        self._connected = False
        self._disconnect_time = self._now()

        if self._logger:
            self._logger.log("connection_lost", time=self._disconnect_time.isoformat())

        # Start reconnect loop
        if self._reconnect_task is None or self._reconnect_task.done():
            loop = asyncio.get_event_loop()
            self._reconnect_task = loop.create_task(self._reconnect_loop())

    def on_reconnect(self, callback):
        """Register an async callback to fire on reconnect (not initial connect)."""
        self._reconnect_callbacks.append(callback)

    def _on_connected(self):
        """Called by ib_async on successful connection."""
        self._connected = True
        was_reconnect = self._disconnect_time is not None

        if was_reconnect and self._logger:
            duration = (self._now() - self._disconnect_time).total_seconds()
            self._logger.log(
                "connection_restored",
                downtime_seconds=round(duration, 1),
            )

        self._disconnect_time = None

        # Fire reconnect callbacks (only on REconnect, not initial connect)
        if was_reconnect and self._initial_connect_done:
            for cb in self._reconnect_callbacks:
                asyncio.ensure_future(cb())

    async def _reconnect_loop(self):
        """Try reconnecting every N seconds until successful."""
        while not self._connected:
            try:
                await asyncio.sleep(self._reconnect_interval)
                if not self._connected:
                    if self._logger:
                        self._logger.log("reconnect_attempt")
                    await self._ib.connectAsync(
                        self._host, self._port, clientId=self._client_id,
                        timeout=15,
                    )
            except asyncio.CancelledError:
                return
            except Exception as e:
                if self._logger:
                    self._logger.log("reconnect_failed", error=str(e))

    @property
    def is_connected(self):
        return self._connected and self._ib is not None and self._ib.isConnected()

    @property
    def disconnect_duration(self):
        """Time since disconnect. None if connected."""
        if self._disconnect_time is None:
            return None
        return self._now() - self._disconnect_time

    @property
    def disconnect_seconds(self):
        """Seconds since disconnect, or 0 if connected."""
        d = self.disconnect_duration
        return d.total_seconds() if d else 0

    @property
    def ib(self):
        """Access the underlying ib_async.IB instance."""
        return self._ib
