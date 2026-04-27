"""
ws_client.py — Tradovate WebSocket client.

Handles the Tradovate text-frame protocol:
  'o' = open (connection established)
  'h' = heartbeat
  'a' = data array (JSON)
  'c' = close

Requests follow: url\\nid\\nquery\\nbody
Responses contain {s: status, i: request_id, d: data}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class TradovateWebSocket:
    """WebSocket client for Tradovate's text-frame protocol.

    Two instances are used per adapter:
    - Order/account WebSocket (demo/live URL)
    - Market data WebSocket (md URL)
    """

    # Heartbeat interval — Tradovate expects a heartbeat (empty array "[]")
    # every ~2.5 seconds to keep the connection alive.
    HEARTBEAT_INTERVAL = 2.5

    # Reconnect parameters
    RECONNECT_MIN_DELAY = 1.0
    RECONNECT_MAX_DELAY = 30.0

    def __init__(self, name: str = "ws", token_getter: Optional[Callable[[], str]] = None):
        self._name = name
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._url: Optional[str] = None
        self._access_token: Optional[str] = None
        self._token_getter: Optional[Callable[[], str]] = token_getter
        self._token_refresher: Optional[Callable] = None

        # Request tracking
        self._next_id = 1
        self._pending_requests: Dict[int, asyncio.Future] = {}

        # Subscription listeners: list of (filter_fn, callback)
        self._listeners: List[_Listener] = []

        # Heartbeat
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_message_time: float = 0.0

        # Receive loop
        self._receive_task: Optional[asyncio.Task] = None

        # Reconnection
        self._reconnect_callbacks: List[Callable] = []
        self._is_connected = False
        self._disconnect_time: float = 0.0
        self._should_run = False
        self._last_close_code: Optional[int] = None
        self._last_close_reason: str = ""
        self._last_disconnect_reason: str = ""

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def disconnect_seconds(self) -> float:
        if self._is_connected:
            return 0.0
        if self._disconnect_time > 0:
            return time.time() - self._disconnect_time
        return 0.0

    async def connect(self, url: str, access_token: str) -> None:
        """Connect to the Tradovate WebSocket and authenticate."""
        self._url = url
        self._access_token = access_token
        self._should_run = True

        if self._session is None:
            self._session = aiohttp.ClientSession()

        self._ws = await self._session.ws_connect(url)
        logger.info("[%s] WebSocket connected to %s", self._name, url)

        # Wait for 'o' (open) frame
        msg = await self._ws.receive()
        if msg.type == aiohttp.WSMsgType.TEXT and msg.data.startswith("o"):
            logger.debug("[%s] Received open frame", self._name)
        else:
            raise ConnectionError(
                f"Expected 'o' frame, got: {msg.type} {msg.data[:50]}"
            )

        # Start heartbeat
        self._last_message_time = time.time()
        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

        # Authenticate — read response directly (receive loop not yet running)
        auth_id = self._allocate_id()
        auth_msg = f"authorize\n{auth_id}\n\n{access_token}"
        await self._ws.send_str(auth_msg)

        auth_response = await self._read_auth_response(auth_id, timeout=10.0)
        status = auth_response.get("s", 0)
        if status != 200:
            raise ConnectionError(
                f"Tradovate WS auth failed (status={status}): {auth_response}"
            )

        self._is_connected = True
        logger.info("[%s] WebSocket authenticated", self._name)

        # Start receive loop
        self._receive_task = asyncio.ensure_future(self._receive_loop())

    async def disconnect(self) -> None:
        """Gracefully close the WebSocket."""
        self._should_run = False
        self._is_connected = False

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()

        if self._ws and not self._ws.closed:
            await self._ws.close()

        # Cancel pending requests
        for fut in self._pending_requests.values():
            if not fut.done():
                fut.cancel()
        self._pending_requests.clear()
        self._listeners.clear()

        if self._session and not self._session.closed:
            await self._session.close()

    def on_reconnect(self, callback: Callable) -> None:
        """Register a callback to fire after reconnection."""
        self._reconnect_callbacks.append(callback)

    # ── Request / Response ──────────────────────────────────────────────

    def _allocate_id(self) -> int:
        req_id = self._next_id
        self._next_id += 1
        return req_id

    async def request(
        self,
        url: str,
        body: Optional[Dict] = None,
        query: str = "",
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Send a request over WebSocket and wait for the correlated response.

        Format: url\\nid\\nquery\\nbody
        """
        if not self._is_connected:
            raise ConnectionError(f"[{self._name}] Not connected")

        req_id = self._allocate_id()
        body_str = json.dumps(body) if body else ""
        message = f"{url}\n{req_id}\n{query}\n{body_str}"

        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[req_id] = fut

        await self._ws.send_str(message)
        return await self._wait_for_response(req_id, timeout)

    async def _wait_for_response(
        self, req_id: int, timeout: float = 10.0
    ) -> Dict[str, Any]:
        """Wait for a response matching the given request ID."""
        fut = self._pending_requests.get(req_id)
        if fut is None:
            fut = asyncio.get_event_loop().create_future()
            self._pending_requests[req_id] = fut

        try:
            result = await asyncio.wait_for(fut, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"[{self._name}] Request {req_id} timed out after {timeout}s"
            )
        finally:
            self._pending_requests.pop(req_id, None)

    # ── Subscriptions ───────────────────────────────────────────────────

    def add_listener(
        self,
        filter_fn: Callable[[Dict], bool],
        callback: Callable[[Dict], Any],
    ) -> Callable[[], None]:
        """Add a listener for incoming data. Returns an unsubscribe function."""
        listener = _Listener(filter_fn=filter_fn, callback=callback)
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    async def subscribe(
        self,
        url: str,
        body: Optional[Dict] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
        callback: Optional[Callable[[Dict], Any]] = None,
    ) -> _Subscription:
        """Subscribe to a WebSocket endpoint with a filter and callback.

        Returns a Subscription object with an unsubscribe() method.
        """
        response = await self.request(url, body)

        unsub_listener = None
        if filter_fn and callback:
            unsub_listener = self.add_listener(filter_fn, callback)

        return _Subscription(
            ws=self,
            subscribe_url=url,
            response=response,
            unsub_listener=unsub_listener,
        )

    async def _read_auth_response(self, auth_id: int, timeout: float = 10.0) -> Dict:
        """Read WS frames directly until the auth response arrives.

        Called during connect() before the receive loop starts.
        """
        import asyncio as _aio
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            try:
                msg = await _aio.wait_for(self._ws.receive(), timeout=remaining)
            except _aio.TimeoutError:
                break
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            data = msg.data
            if not data:
                continue
            if data[0] == "c":
                self._handle_text_frame(data)
                raise ConnectionError(
                    f"[{self._name}] Closed during auth: "
                    f"code={self._last_close_code} reason={self._last_close_reason or 'unknown'}"
                )
            if data[0] != "a":
                continue
            try:
                items = json.loads(data[1:])
            except json.JSONDecodeError:
                continue
            for item in items:
                if item.get("i") == auth_id:
                    return item
        raise TimeoutError(f"[{self._name}] Auth response timed out after {timeout}s")

    # ── Receive Loop ────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """Main loop processing incoming WebSocket messages."""
        disconnect_reason = "unknown"
        try:
            while self._should_run:
                if self._ws is None or self._ws.closed:
                    disconnect_reason = "ws_closed_externally"
                    break

                msg = await self._ws.receive()
                self._last_message_time = time.time()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    keep_running = self._handle_text_frame(msg.data)
                    if not keep_running:
                        disconnect_reason = (
                            f"close_frame code={self._last_close_code} "
                            f"reason={self._last_close_reason or 'none'}"
                        )
                        break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    disconnect_reason = "ws_msg_closed"
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    disconnect_reason = f"ws_error: {msg.data}"
                    break

        except asyncio.CancelledError:
            return
        except Exception as e:
            disconnect_reason = f"exception: {e}"

        # Connection lost — cancel heartbeat before reconnecting
        self._is_connected = False
        self._disconnect_time = time.time()
        self._last_disconnect_reason = disconnect_reason
        logger.warning("[%s] Disconnected: %s", self._name, disconnect_reason)

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

        if self._should_run:
            asyncio.ensure_future(self._reconnect_loop())

    def _handle_text_frame(self, data: str) -> bool:
        """Parse Tradovate text frame by prefix character."""
        if not data:
            return True

        prefix = data[0]

        if prefix == "h":
            return True

        if prefix == "a":
            # Data array
            try:
                items = json.loads(data[1:])
            except json.JSONDecodeError:
                logger.warning("[%s] Failed to parse data frame: %s", self._name, data[:100])
                return True

            for item in items:
                self._dispatch_item(item)
            return True

        if prefix == "c":
            code, reason = self._parse_close_frame(data)
            self._last_close_code = code
            self._last_close_reason = reason
            if code is None and not reason:
                logger.info("[%s] Received close frame", self._name)
            else:
                logger.warning(
                    "[%s] Received close frame: code=%s reason=%s",
                    self._name,
                    code if code is not None else "unknown",
                    reason or "unknown",
                )
            self._is_connected = False
            return False

        # Unknown prefix
        logger.debug("[%s] Unknown frame prefix '%s': %s", self._name, prefix, data[:100])
        return True

    def _parse_close_frame(self, data: str) -> tuple[Optional[int], str]:
        """Parse Tradovate SockJS-style close frames: c[code, "reason"]."""
        payload = data[1:].strip()
        if not payload:
            return None, ""

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None, payload

        if not isinstance(parsed, list):
            return None, str(parsed)

        code = parsed[0] if parsed and isinstance(parsed[0], int) else None
        reason = str(parsed[1]) if len(parsed) > 1 and parsed[1] is not None else ""
        return code, reason

    def _dispatch_item(self, item: Dict) -> None:
        """Route an incoming data item to pending requests and listeners."""
        # Check if this is a response to a pending request
        req_id = item.get("i")
        if req_id is not None and req_id in self._pending_requests:
            fut = self._pending_requests.get(req_id)
            if fut and not fut.done():
                fut.set_result(item)

        # Dispatch to all matching listeners
        for listener in list(self._listeners):
            try:
                if listener.filter_fn(item):
                    listener.callback(item)
            except Exception as e:
                logger.error("[%s] Listener error: %s", self._name, e)

    # ── Heartbeat ───────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Send heartbeat frames to keep the connection alive."""
        try:
            while self._should_run:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                if not self._ws or self._ws.closed:
                    break
                elapsed = time.time() - self._last_message_time
                if elapsed >= self.HEARTBEAT_INTERVAL:
                    try:
                        await self._ws.send_str("[]")
                    except Exception as e:
                        logger.warning("[%s] Heartbeat send failed: %s", self._name, e)
                        try:
                            if self._ws and not self._ws.closed:
                                await self._ws.close()
                        except Exception:
                            pass
                        break
        except asyncio.CancelledError:
            return

    # ── Reconnection ────────────────────────────────────────────────────

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        delay = self.RECONNECT_MIN_DELAY

        while self._should_run and not self._is_connected:
            logger.info("[%s] Reconnecting in %.1fs...", self._name, delay)
            await asyncio.sleep(delay)

            try:
                # Clean up previous failed attempt
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                if self._ws and not self._ws.closed:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass

                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession()

                self._ws = await self._session.ws_connect(self._url)

                # Wait for open frame
                msg = await self._ws.receive()
                if not (msg.type == aiohttp.WSMsgType.TEXT and msg.data.startswith("o")):
                    raise ConnectionError("No open frame")

                # Re-authenticate with fresh token if available
                self._last_message_time = time.time()
                self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

                token = self._token_getter() if self._token_getter else self._access_token
                auth_id = self._allocate_id()
                await self._ws.send_str(f"authorize\n{auth_id}\n\n{token}")
                auth_resp = await self._read_auth_response(auth_id, timeout=10.0)
                if auth_resp.get("s") != 200:
                    # 401 = expired/invalid token — force refresh before next retry
                    if auth_resp.get("s") == 401 and self._token_refresher:
                        try:
                            result = self._token_refresher()
                            if asyncio.iscoroutine(result):
                                await result
                            logger.info("[%s] Token refreshed after 401, retrying immediately", self._name)
                            delay = self.RECONNECT_MIN_DELAY
                            continue
                        except Exception as refresh_err:
                            logger.warning("[%s] Token refresh failed: %s", self._name, refresh_err)
                    raise ConnectionError(f"Auth failed: {auth_resp}")

                self._is_connected = True
                self._receive_task = asyncio.ensure_future(self._receive_loop())

                logger.info("[%s] Reconnected successfully", self._name)

                # Fire reconnect callbacks
                for cb in self._reconnect_callbacks:
                    try:
                        result = cb()
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error("[%s] Reconnect callback error: %s", self._name, e)

                return  # Success

            except Exception as e:
                logger.warning("[%s] Reconnect failed: %s", self._name, e)
                delay = min(delay * 2, self.RECONNECT_MAX_DELAY)


# ── Internal Types ──────────────────────────────────────────────────────

class _Listener:
    """A WebSocket data listener with a filter function."""
    __slots__ = ("filter_fn", "callback")

    def __init__(self, filter_fn: Callable, callback: Callable):
        self.filter_fn = filter_fn
        self.callback = callback


class _Subscription:
    """Represents an active WebSocket subscription."""

    def __init__(
        self,
        ws: TradovateWebSocket,
        subscribe_url: str,
        response: Dict,
        unsub_listener: Optional[Callable] = None,
    ):
        self._ws = ws
        self._subscribe_url = subscribe_url
        self.response = response
        self._unsub_listener = unsub_listener

    async def unsubscribe(self, unsub_url: Optional[str] = None) -> None:
        """Cancel this subscription."""
        if self._unsub_listener:
            self._unsub_listener()

        if unsub_url:
            try:
                await self._ws.request(unsub_url)
            except Exception:
                pass  # Best effort
