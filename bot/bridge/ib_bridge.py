"""
ib_bridge.py — Persistent IB Gateway bridge running on Windows Python.

WSL2 cannot directly reach IB TWS on Windows due to Hyper-V network isolation.
This script runs on Windows (via python.exe) and provides a TCP socket server
that the WSL-side bot connects to.

Protocol: newline-delimited JSON messages over TCP.

Launched from WSL:
    python.exe C:\\path\\to\\ib_bridge.py --port 9100 --ib-port 7497

Architecture:
    WSL Bot  <--TCP:9100-->  ib_bridge.py (Windows)  <--IB API-->  TWS/Gateway
"""

import asyncio
import json
import signal
import sys
import time
import traceback

# Try ib_async first, fall back to ib_insync
try:
    from ib_async import IB, Future, util, LimitOrder, StopOrder, MarketOrder
    util.startLoop()
except ImportError:
    try:
        from ib_insync import IB, Future, util, LimitOrder, StopOrder, MarketOrder
        util.startLoop()
    except ImportError:
        print(json.dumps({"error": "Neither ib_async nor ib_insync installed on Windows Python"}),
              flush=True)
        sys.exit(1)


class IBBridge:
    """
    Persistent bridge between WSL bot and IB TWS.

    Connects to TWS, subscribes to bar data, and relays events to
    the WSL bot over a TCP socket. Also accepts commands (place_order,
    cancel_order, flatten, etc.) from the bot.
    """

    def __init__(self, ib_host, ib_port, ib_client_id, tcp_port):
        self.ib_host = ib_host
        self.ib_port = ib_port
        self.ib_client_id = ib_client_id
        self.tcp_port = tcp_port
        self.ib = IB()
        self.writer = None
        self._running = True
        self._bars_5min = None
        self._bars_1min = None
        self._contract = None

    async def run(self):
        """Main entry: start TCP server, connect to IB, serve forever."""
        server = await asyncio.start_server(
            self._handle_client, '0.0.0.0', self.tcp_port
        )
        self._log("bridge_started", tcp_port=self.tcp_port)

        async with server:
            await server.serve_forever()

    async def _handle_client(self, reader, writer):
        """Handle a single bot connection."""
        self.writer = writer
        addr = writer.get_extra_info('peername')
        self._log("client_connected", addr=str(addr))

        try:
            # Connect to IB
            await self._connect_ib()

            # Process commands from bot
            while self._running:
                line = await reader.readline()
                if not line:
                    break
                await self._process_command(line.decode().strip())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._send({"event": "error", "error": str(e), "trace": traceback.format_exc()})
        finally:
            self._log("client_disconnected")
            await self._cleanup()
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            self.writer = None

    async def _connect_ib(self):
        """Connect to IB TWS/Gateway."""
        try:
            await self.ib.connectAsync(
                self.ib_host, self.ib_port,
                clientId=self.ib_client_id, timeout=15
            )
            self.ib.disconnectedEvent += self._on_ib_disconnect
            self._send({"event": "connected", "host": self.ib_host, "port": self.ib_port})
        except Exception as e:
            self._send({"event": "connect_failed", "error": str(e)})
            raise

    def _on_ib_disconnect(self):
        """IB disconnected callback."""
        self._send({"event": "disconnected"})

    async def _process_command(self, line):
        """Parse and execute a JSON command from the bot."""
        if not line:
            return

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            self._send({"event": "error", "error": f"Invalid JSON: {line}"})
            return

        action = cmd.get("action")

        try:
            if action == "ping":
                self._send({"event": "pong", "ts": time.time()})

            elif action == "status":
                self._send({
                    "event": "status",
                    "connected": self.ib.isConnected(),
                    "ts": time.time(),
                })

            elif action == "get_time":
                if self.ib.isConnected():
                    ib_time = self.ib.reqCurrentTime()
                    self._send({"event": "server_time", "time": str(ib_time)})

            elif action == "get_balance":
                await self._cmd_get_balance(cmd)

            elif action == "resolve_contract":
                await self._cmd_resolve_contract(cmd)

            elif action == "subscribe_bars":
                await self._cmd_subscribe_bars(cmd)

            elif action == "place_bracket":
                await self._cmd_place_bracket(cmd)

            elif action == "cancel_order":
                await self._cmd_cancel_order(cmd)

            elif action == "flatten":
                await self._cmd_flatten(cmd)

            elif action == "get_positions":
                await self._cmd_get_positions()

            elif action == "get_open_orders":
                await self._cmd_get_open_orders()

            elif action == "modify_order_qty":
                await self._cmd_modify_order_qty(cmd)

            elif action == "disconnect":
                self._running = False

            else:
                self._send({"event": "error", "error": f"Unknown action: {action}"})

        except Exception as e:
            self._send({
                "event": "command_error",
                "action": action,
                "error": str(e),
                "trace": traceback.format_exc(),
            })

    # -------------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------------

    async def _cmd_get_balance(self, cmd):
        account = cmd.get("account", "")
        values = self.ib.accountValues()
        balance = None
        for av in values:
            if av.tag == "NetLiquidation" and av.currency == "USD":
                if not account or av.account == account:
                    balance = float(av.value)
                    break
        self._send({"event": "balance", "balance": balance})

    async def _cmd_resolve_contract(self, cmd):
        contract = Future(
            symbol=cmd.get("symbol", "NQ"),
            lastTradeDateOrContractMonth=cmd.get("expiry", ""),
            exchange=cmd.get("exchange", "CME"),
            currency=cmd.get("currency", "USD"),
        )
        qualified = await self.ib.qualifyContractsAsync(contract)
        if qualified:
            self._contract = contract
            self._send({
                "event": "contract_resolved",
                "symbol": contract.symbol,
                "conId": contract.conId,
                "localSymbol": contract.localSymbol,
                "expiry": cmd.get("expiry", ""),
            })
        else:
            self._send({"event": "contract_error", "error": "Failed to qualify contract"})

    async def _cmd_subscribe_bars(self, cmd):
        if not self._contract:
            self._send({"event": "error", "error": "No contract resolved"})
            return

        # 5-minute bars
        self._bars_5min = self.ib.reqHistoricalData(
            self._contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="5 mins",
            whatToShow="TRADES",
            useRTH=True,
            keepUpToDate=True,
        )
        self._bars_5min.updateEvent += self._on_5min_update

        # 1-minute bars
        self._bars_1min = self.ib.reqHistoricalData(
            self._contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="1 min",
            whatToShow="TRADES",
            useRTH=True,
            keepUpToDate=True,
        )
        self._bars_1min.updateEvent += self._on_1min_update

        self._send({"event": "bars_subscribed"})

    async def _cmd_place_bracket(self, cmd):
        if not self._contract:
            self._send({"event": "error", "error": "No contract resolved"})
            return

        side = cmd["side"]
        qty = cmd["qty"]
        entry_price = cmd["entry_price"]
        tp_price = cmd["tp_price"]
        sl_price = cmd["sl_price"]
        group_id = cmd.get("group_id", "")
        reverse_side = "SELL" if side == "BUY" else "BUY"

        # Create bracket
        parent = LimitOrder(action=side, totalQuantity=qty, lmtPrice=entry_price, transmit=False)
        tp = LimitOrder(action=reverse_side, totalQuantity=qty, lmtPrice=tp_price,
                        parentId=parent.orderId, transmit=False)
        sl = StopOrder(action=reverse_side, totalQuantity=qty, auxPrice=sl_price,
                       parentId=parent.orderId, transmit=True)

        # Place orders
        entry_trade = self.ib.placeOrder(self._contract, parent)
        tp_trade = self.ib.placeOrder(self._contract, tp)
        sl_trade = self.ib.placeOrder(self._contract, sl)

        # Register callbacks
        entry_trade.filledEvent += lambda t: self._on_fill(t, group_id, "entry")
        entry_trade.statusEvent += lambda t: self._on_order_status(t, group_id, "entry")
        tp_trade.filledEvent += lambda t: self._on_fill(t, group_id, "tp")
        sl_trade.filledEvent += lambda t: self._on_fill(t, group_id, "sl")

        self._send({
            "event": "bracket_placed",
            "group_id": group_id,
            "entry_order_id": parent.orderId,
            "tp_order_id": tp.orderId,
            "sl_order_id": sl.orderId,
        })

    async def _cmd_cancel_order(self, cmd):
        order_id = cmd.get("order_id")
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                self._send({"event": "order_cancelled", "order_id": order_id})
                return
        self._send({"event": "order_not_found", "order_id": order_id})

    async def _cmd_flatten(self, cmd):
        positions = self.ib.positions()
        closed = 0
        for pos in positions:
            if pos.contract.symbol == self._contract.symbol and pos.position != 0:
                side = "SELL" if pos.position > 0 else "BUY"
                qty = abs(pos.position)
                order = MarketOrder(action=side, totalQuantity=qty)
                self.ib.placeOrder(pos.contract, order)
                closed += 1
        self._send({"event": "flatten_done", "positions_closed": closed})

    async def _cmd_get_positions(self):
        positions = self.ib.positions()
        result = []
        for pos in positions:
            result.append({
                "symbol": pos.contract.symbol,
                "position": pos.position,
                "avgCost": pos.avgCost,
                "account": pos.account,
            })
        self._send({"event": "positions", "data": result})

    async def _cmd_get_open_orders(self):
        orders = self.ib.openOrders()
        result = []
        for order in orders:
            result.append({
                "orderId": order.orderId,
                "action": order.action,
                "totalQuantity": order.totalQuantity,
                "orderType": order.orderType,
                "lmtPrice": getattr(order, 'lmtPrice', None),
                "auxPrice": getattr(order, 'auxPrice', None),
                "parentId": order.parentId,
            })
        self._send({"event": "open_orders", "data": result})

    async def _cmd_modify_order_qty(self, cmd):
        order_id = cmd.get("order_id")
        new_qty = cmd.get("new_qty")
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                trade.order.totalQuantity = new_qty
                self.ib.placeOrder(self._contract, trade.order)
                self._send({"event": "order_modified", "order_id": order_id, "new_qty": new_qty})
                return
        self._send({"event": "order_not_found", "order_id": order_id})

    # -------------------------------------------------------------------------
    # Event callbacks → forwarded to bot via TCP
    # -------------------------------------------------------------------------

    def _on_5min_update(self, bars, hasNewBar):
        if hasNewBar and len(bars) >= 1:
            bar = bars[-1]
            self._send({
                "event": "bar_5min",
                "open": bar.open, "high": bar.high,
                "low": bar.low, "close": bar.close,
                "date": str(bar.date), "volume": bar.volume,
            })

    def _on_1min_update(self, bars, hasNewBar):
        if hasNewBar and len(bars) >= 1:
            bar = bars[-1]
            self._send({
                "event": "bar_1min",
                "open": bar.open, "high": bar.high,
                "low": bar.low, "close": bar.close,
                "date": str(bar.date), "volume": bar.volume,
            })

    def _on_fill(self, trade, group_id, order_type):
        status = trade.orderStatus
        self._send({
            "event": "fill",
            "group_id": group_id,
            "order_type": order_type,
            "order_id": trade.order.orderId,
            "status": status.status,
            "filled": status.filled,
            "remaining": status.remaining,
            "avgFillPrice": status.avgFillPrice,
        })

    def _on_order_status(self, trade, group_id, order_type):
        status = trade.orderStatus
        self._send({
            "event": "order_status",
            "group_id": group_id,
            "order_type": order_type,
            "order_id": trade.order.orderId,
            "status": status.status,
            "filled": status.filled,
            "remaining": status.remaining,
        })

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _send(self, msg):
        """Send a JSON message to the bot via TCP."""
        if self.writer and not self.writer.is_closing():
            line = json.dumps(msg, default=str) + "\n"
            self.writer.write(line.encode())

    def _log(self, event, **kwargs):
        """Print structured log to stderr (visible in WSL process output)."""
        msg = {"ts": time.time(), "bridge_event": event, **kwargs}
        print(json.dumps(msg, default=str), file=sys.stderr, flush=True)

    async def _cleanup(self):
        """Clean up IB subscriptions and connection."""
        if self._bars_5min:
            self.ib.cancelHistoricalData(self._bars_5min)
        if self._bars_1min:
            self.ib.cancelHistoricalData(self._bars_1min)
        if self.ib.isConnected():
            self.ib.disconnect()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="IB Bridge for FVG Trading Bot")
    parser.add_argument("--port", type=int, default=9100, help="TCP port for bot connection")
    parser.add_argument("--ib-host", default="127.0.0.1", help="IB TWS host")
    parser.add_argument("--ib-port", type=int, default=7497, help="IB TWS port")
    parser.add_argument("--client-id", type=int, default=1, help="IB client ID")
    args = parser.parse_args()

    bridge = IBBridge(args.ib_host, args.ib_port, args.client_id, args.port)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(bridge.run())
    except KeyboardInterrupt:
        print("Bridge shutting down...", file=sys.stderr, flush=True)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
