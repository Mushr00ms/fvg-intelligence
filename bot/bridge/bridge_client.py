"""
bridge_client.py — WSL-side client that connects to the Windows IB bridge.

Handles:
- Launching the bridge process via python.exe (if not already running)
- WSL path conversion (same pattern as risk-webapp/ib_client.py)
- TCP connection to the bridge
- Sending commands and receiving events
- Reconnection if bridge goes down
"""

import asyncio
import json
import os
import subprocess
import time


# WSL path conversion (from risk-webapp/ib_client.py pattern)
_WSL_DIR = os.path.dirname(os.path.abspath(__file__))
if _WSL_DIR.startswith('/mnt/'):
    _parts = _WSL_DIR.split('/')
    _drive = _parts[2].upper()
    BRIDGE_SCRIPT_WIN = _drive + ':\\' + '\\'.join(_parts[3:]) + '\\ib_bridge.py'
else:
    BRIDGE_SCRIPT_WIN = os.path.join(_WSL_DIR, 'ib_bridge.py')


class BridgeClient:
    """
    Async TCP client that connects to the IB bridge running on Windows.

    Usage:
        client = BridgeClient(config, logger)
        await client.start()                    # Launch bridge + connect
        await client.send({"action": "status"}) # Send command
        msg = await client.recv()               # Receive event
        client.on_event = callback              # Set event handler
    """

    @staticmethod
    def _detect_windows_host():
        """Detect the Windows host IP from WSL2. Falls back to 127.0.0.1."""
        try:
            with open('/etc/resolv.conf') as f:
                for line in f:
                    if line.strip().startswith('nameserver'):
                        ip = line.strip().split()[-1]
                        if ip != '127.0.0.1':
                            return ip
        except FileNotFoundError:
            pass
        # Fallback: try default gateway
        try:
            import subprocess
            result = subprocess.run(['ip', 'route', 'show', 'default'],
                                    capture_output=True, text=True, timeout=3)
            for part in result.stdout.split():
                if '.' in part and part[0].isdigit():
                    return part
        except Exception:
            pass
        return "127.0.0.1"

    def __init__(self, config, logger=None):
        self._config = config
        self._logger = logger
        self._reader = None
        self._writer = None
        self._bridge_process = None
        self._tcp_host = self._detect_windows_host()
        self._tcp_port = getattr(config, 'bridge_port', 9100)
        self._connected = False
        self._event_handlers = {}       # event_name -> [callbacks]
        self._response_queue = asyncio.Queue()
        self._recv_task = None
        self._heartbeat_task = None
        self._on_disconnect_callback = None  # Engine sets this to pause trading
        self._heartbeat_interval = getattr(config, 'bridge_heartbeat_interval', 30)
        self._heartbeat_timeout = getattr(config, 'bridge_heartbeat_timeout', 5)

    async def start(self, launch_bridge=True):
        """
        Start the bridge and connect.

        Args:
            launch_bridge: If True, launch python.exe ib_bridge.py first.
                          If False, assume bridge is already running.
        """
        if launch_bridge:
            self._launch_bridge()
            await asyncio.sleep(2)  # Give bridge time to start TCP server

        # Connect to bridge TCP
        await self._connect()

        # Start heartbeat monitor
        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

    async def _connect(self, retries=5, delay=2):
        """Connect to the bridge TCP server."""
        for attempt in range(retries):
            try:
                self._reader, self._writer = await asyncio.open_connection(
                    self._tcp_host, self._tcp_port
                )
                self._connected = True

                # Start background receiver
                self._recv_task = asyncio.ensure_future(self._recv_loop())

                if self._logger:
                    self._logger.log("bridge_connected", port=self._tcp_port)
                return

            except (ConnectionRefusedError, OSError) as e:
                if attempt < retries - 1:
                    if self._logger:
                        self._logger.log(
                            "bridge_connect_retry",
                            attempt=attempt + 1,
                            error=str(e),
                        )
                    await asyncio.sleep(delay)
                else:
                    raise ConnectionError(
                        f"Cannot connect to IB bridge at {self._tcp_host}:{self._tcp_port} "
                        f"after {retries} attempts. Is the bridge running?"
                    )

    def _launch_bridge(self):
        """Launch the IB bridge as a Windows Python subprocess."""
        cmd = [
            'python.exe',
            BRIDGE_SCRIPT_WIN,
            '--port', str(self._tcp_port),
            '--ib-host', self._config.ib_host,
            '--ib-port', str(self._config.ib_port),
            '--client-id', str(self._config.ib_client_id),
        ]

        if self._logger:
            self._logger.log("bridge_launching", cmd=' '.join(cmd))

        # Launch detached — bridge runs independently
        self._bridge_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

        if self._logger:
            self._logger.log("bridge_launched", pid=self._bridge_process.pid)

    async def send(self, msg):
        """Send a JSON command to the bridge."""
        if not self._connected or self._writer is None:
            raise ConnectionError("Not connected to bridge")

        line = json.dumps(msg, default=str) + "\n"
        self._writer.write(line.encode())
        await self._writer.drain()

    async def send_and_wait(self, msg, event_type, timeout=10):
        """Send a command and wait for a specific response event."""
        future = asyncio.get_event_loop().create_future()

        def handler(data):
            if not future.done():
                future.set_result(data)

        self.on(event_type, handler, once=True)
        await self.send(msg)

        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            if self._logger:
                self._logger.log("bridge_timeout", action=msg.get("action"), event=event_type)
            return None

    def on(self, event_type, callback, once=False):
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append((callback, once))

    def off(self, event_type, callback=None):
        """Remove event handler(s)."""
        if callback:
            self._event_handlers[event_type] = [
                (cb, o) for cb, o in self._event_handlers.get(event_type, [])
                if cb != callback
            ]
        else:
            self._event_handlers.pop(event_type, None)

    def on_disconnect(self, callback):
        """Register a callback for bridge disconnection (engine uses this to pause trading)."""
        self._on_disconnect_callback = callback

    async def _heartbeat_loop(self):
        """Periodic heartbeat: ping bridge, reconnect if unresponsive."""
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                if not self._connected:
                    # Already disconnected — attempt reconnect
                    await self._attempt_reconnect()
                    continue

                # Send ping, expect pong
                try:
                    result = await self.send_and_wait(
                        {"action": "ping"}, "pong", timeout=self._heartbeat_timeout
                    )
                    if result is None:
                        # Pong timeout — bridge is unresponsive
                        if self._logger:
                            self._logger.log("bridge_heartbeat_timeout")
                        self._connected = False
                        if self._on_disconnect_callback:
                            cb_result = self._on_disconnect_callback()
                            if asyncio.iscoroutine(cb_result):
                                await cb_result
                        await self._attempt_reconnect()
                except (ConnectionError, OSError):
                    self._connected = False
                    await self._attempt_reconnect()
        except asyncio.CancelledError:
            pass

    async def _attempt_reconnect(self):
        """Try to reconnect to the bridge with backoff."""
        if self._logger:
            self._logger.log("bridge_reconnecting")
        try:
            # Cancel old recv task
            if self._recv_task and not self._recv_task.done():
                self._recv_task.cancel()
            await self._connect(retries=3, delay=3)
            if self._logger:
                self._logger.log("bridge_reconnected")
        except ConnectionError:
            if self._logger:
                self._logger.log("bridge_reconnect_failed")

    async def _recv_loop(self):
        """Background task: read JSON events from bridge and dispatch."""
        try:
            while self._connected and self._reader:
                line = await self._reader.readline()
                if not line:
                    self._connected = False
                    if self._logger:
                        self._logger.log("bridge_disconnected")
                    break

                try:
                    msg = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue

                event_type = msg.get("event", "unknown")
                self._dispatch(event_type, msg)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._logger:
                self._logger.log("bridge_recv_error", error=str(e))
            self._connected = False

    def _dispatch(self, event_type, msg):
        """Dispatch an event to registered handlers."""
        handlers = self._event_handlers.get(event_type, [])
        remaining = []
        for callback, once in handlers:
            try:
                result = callback(msg)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as e:
                if self._logger:
                    self._logger.log("bridge_handler_error", event=event_type, error=str(e))
            if not once:
                remaining.append((callback, once))
        self._event_handlers[event_type] = remaining

    @property
    def is_connected(self):
        return self._connected

    async def stop(self):
        """Disconnect from bridge and optionally kill the process."""
        self._connected = False

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

        if self._writer:
            try:
                await self.send({"action": "disconnect"})
            except Exception:
                pass
            self._writer.close()

        if self._bridge_process:
            self._bridge_process.terminate()
            self._bridge_process = None
