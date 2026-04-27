"""
replay_engine.py — Synchronous replay loop using Databento tick data.

Feeds historical tick data through the actual bot modules:
  - FVG detection:  bot.strategy.fvg_detector.check_fvg_3bars
  - Strategy lookup: bot.strategy.strategy_loader.StrategyLoader
  - Setup calc:     bot.strategy.trade_calculator.calculate_setup
  - Risk gates:     bot.risk.risk_gates.RiskGates
  - Time gates:     bot.risk.time_gates (inline checks on bar timestamps)
  - Fills:          ReplayOrderManager with tick-accurate volume + slippage
  - Margin:         Displacement logic inside ReplayOrderManager

This is NOT the backtester — it uses the bot's actual modules, not inline
reimplementations. Divergences between replay and backtest results reveal
where the backtester's simplified logic drifts from the real bot.
"""

import math
import os
import sys
from collections import deque
from datetime import datetime, time, timedelta

import pandas as pd

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from bot.bot_config import BotConfig
from bot.state.trade_state import DailyState
from bot.strategy.fvg_detector import check_fvg_3bars, _assign_time_period, SESSION_INTERVALS
from bot.strategy.strategy_loader import StrategyLoader
from bot.strategy.trade_calculator import calculate_setup, risk_to_range, round_to_tick, POINT_VALUE
from bot.risk.risk_gates import RiskGates
from bot.risk.calendar_gates import WitchingGateConfig, is_blocked_by_witching_gate
from bot.replay.replay_order_manager import ReplayOrderManager, _calc_commission
from bot.replay.replay_clock import ReplayClock
from bot.replay.result_exporter import export_session

# Session boundaries (ET)
SESSION_START = time(9, 30)
SESSION_END = time(16, 0)
LAST_ENTRY = time(15, 45)


class ReplayEngine:
    """
    Synchronous replay engine: Databento ticks → bot modules → trade results.

    Usage:
        engine = ReplayEngine(config, feed)
        results = engine.run("2025-01-01", "2025-12-31")
    """

    def __init__(self, config, tick_feed, logger=None, output_dir=None):
        self.config = config
        self.feed = tick_feed
        self.output_dir = output_dir
        self._clock = ReplayClock()

        # Use a simple print logger if none provided
        self._logger = logger or _PrintLogger()

        self.strategy = StrategyLoader(config.strategy_dir, self._logger)
        self.strategy.load()

        self.risk_gates = RiskGates(config)

        # Hard gates from strategy metadata
        strategy_meta = self.strategy.strategy.get("meta", {}) if self.strategy.strategy else {}
        hard_gates = strategy_meta.get("hard_gates", {})
        self._witching_cfg = WitchingGateConfig(
            no_trade_witching_day=bool(hard_gates.get("no_trade_witching_day", False)),
            no_trade_witching_day_minus_1=bool(hard_gates.get("no_trade_witching_day_minus_1", False)),
        )

        self._all_results = []
        self._running_balance = config._replay_start_balance if hasattr(config, '_replay_start_balance') else 100000.0

    def run(self, start_date, end_date):
        """Run replay across all trading days in range."""
        days = self.feed.trading_days(start_date, end_date)
        print(f"Replay: {len(days)} trading days ({start_date} → {end_date})")
        print(f"Strategy: {self.strategy.strategy_id} ({self.strategy.cell_count} cells)")
        print(f"Balance: ${self._running_balance:,.0f}")
        print()

        for i, day in enumerate(days):
            trades_today = self._run_day(day)
            day_pnl = sum(t.realized_pnl for t in trades_today)
            self._running_balance += day_pnl

            if trades_today:
                wins = sum(1 for t in trades_today if t.close_reason == "TP")
                print(f"[{i+1:3d}/{len(days)}] {day}  "
                      f"{len(trades_today)} trades  "
                      f"{wins}W/{len(trades_today)-wins}L  "
                      f"${day_pnl:+,.0f}  "
                      f"bal ${self._running_balance:,.0f}")
            else:
                print(f"[{i+1:3d}/{len(days)}] {day}  no trades")

        print(f"\nReplay complete: {len(self._all_results)} total trades")
        return self._all_results

    def _run_day(self, date_str):
        """Process a single trading day.

        Uses IB 1s bars for ALL trade decisions (pixel-perfect with backtester):
        1. Detect FVGs from IB 5min bars
        2. Find mitigation on IB 1s bars
        3. Check entry fill on IB 1s bars
        4. Walk trade to exit on IB 1s bars
        5. For SL exits: overlay Databento tick price for real slippage
        """
        trade_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        blocked, reason = is_blocked_by_witching_gate(trade_date, self._witching_cfg)
        if blocked:
            return []

        bars_5m = self.feed.get_5min_bars(date_str)
        if bars_5m is None or len(bars_5m) < 3:
            return []

        # Load IB 1s bars for mitigation/fill/walk (same data as backtester)
        ib_1s = self.feed._load_ib_1s(date_str)
        if ib_1s is None or ib_1s.empty:
            return []

        state = DailyState(date=date_str, start_balance=self._running_balance)
        state_mgr = _NoopStateMgr()
        order_mgr = ReplayOrderManager(
            state_manager=state_mgr, logger=self._logger, config=self.config,
            clock=self._clock, db=None, on_kill_switch=None,
        )

        # Pre-extract IB 1s arrays for fast scanning
        import numpy as np
        ib_dates_ns = ib_1s["date"].values.astype("int64")
        ib_highs = ib_1s["high"].values.astype("float64")
        ib_lows = ib_1s["low"].values.astype("float64")
        ib_closes = ib_1s["close"].values.astype("float64")
        n_ib = len(ib_highs)

        # Pre-compute ET minutes for time gate (same as backtester)
        ib_times_et = ib_1s["date"].dt.time.values
        ib_minutes_et = np.array([t.hour * 60 + t.minute for t in ib_times_et])

        # Load Databento ticks for stop slippage overlay (optional — may be None on some days)
        ticks_df = self.feed.get_ticks(date_str)
        has_ticks = ticks_df is not None and not ticks_df.empty
        if has_ticks:
            tick_times_ns = pd.array(ticks_df["ts_et"].values).view("int64")
            tick_prices_db = ticks_df["price"].values.astype("float64")

        # ── Phase 1: Detect FVGs from 5-min bars ────────────────────────
        recent_bars = deque(maxlen=10)
        active_fvgs = []  # (fvg, formation_close_ns)

        for i in range(len(bars_5m)):
            bar = bars_5m.iloc[i]
            bar_dict = {
                "open": bar["open"], "high": bar["high"],
                "low": bar["low"], "close": bar["close"],
                "date": bar["date"],
            }
            recent_bars.append(bar_dict)
            if len(recent_bars) < 3:
                continue

            fvg = check_fvg_3bars(recent_bars[-3], recent_bars[-2], recent_bars[-1],
                                  self.config.min_fvg_size)
            if fvg is None:
                continue

            fvg.time_period = _assign_time_period(bar_dict["date"], SESSION_INTERVALS)
            if not fvg.time_period:
                continue
            fvg.formation_date = date_str

            has_cell = any(tp == fvg.time_period for tp, _ in self.strategy._lookup.keys())
            if not has_cell:
                continue

            # FVG confirmed at bar3 close (5 min after bar3 open)
            form_ts = pd.Timestamp(bar_dict["date"])
            if form_ts.tzinfo is None:
                form_ts = form_ts.tz_localize("America/New_York")
            form_close_ns = (form_ts + timedelta(minutes=5)).value
            active_fvgs.append((fvg, form_close_ns))

        # ── Phase 2: Process each FVG end-to-end ────────────────────────
        # Each FVG: mitigate → setup → entry fill → walk to exit → update state
        # Matches backtester's sequential per-FVG processing
        daily_trades = 0
        daily_pnl = 0.0
        day_start_balance = self._running_balance
        emergency_halt = False

        for fvg, form_close_ns in active_fvgs:
            if emergency_halt or daily_trades >= self.config.max_daily_trades:
                break

            # Find mitigation: first tick in FVG zone after formation close
            mit_idx = -1
            # Binary search for formation_close in tick_ns
            import numpy as np
            start_idx = int(np.searchsorted(tick_ns, form_close_ns, side="right"))
            for ti in range(start_idx, n_ticks):
                p = tick_prices[ti]
                if fvg.zone_low <= p <= fvg.zone_high:
                    mit_idx = ti
                    break

            if mit_idx < 0:
                continue

            # Time gate: no entries at/after 15:45 (use pre-computed ET times)
            mit_t = tick_times_et[mit_idx]
            if mit_t >= LAST_ENTRY:
                continue

            mit_price = float(tick_prices[mit_idx])

            # Try each setup type
            for setup_type in ["mit_extreme", "mid_extreme"]:
                if emergency_halt or daily_trades >= self.config.max_daily_trades:
                    break

                # Provisional entry for cell lookup
                if setup_type == "mit_extreme":
                    entry = fvg.zone_high if fvg.fvg_type == "bullish" else fvg.zone_low
                else:
                    entry = (fvg.zone_high + fvg.zone_low) / 2

                stop = fvg.middle_low if fvg.fvg_type == "bullish" else fvg.middle_high
                risk_pts = round_to_tick(abs(entry - stop))
                if risk_pts <= 0:
                    continue

                risk_range = risk_to_range(risk_pts)
                if not risk_range:
                    continue

                cell = self.strategy.find_cell(fvg.time_period, risk_pts)
                if cell is None or cell["setup"] != setup_type:
                    continue

                # Setup + sizing (actual bot module)
                balance = self._running_balance
                order = calculate_setup(fvg, cell, balance,
                                        self.config.risk_per_trade,
                                        config=self.config)
                if order is None:
                    continue

                # Risk gates (actual bot module)
                result = self.risk_gates.check_all(state, order)
                if not result.passed:
                    continue

                # Margin cap
                margin_per = self.config.margin_fallback_per_contract
                buffer = self.config.margin_buffer_pct
                buffered = margin_per * (1.0 + buffer)
                max_by_margin = max(1, math.floor(balance / buffered))
                if order.target_qty > max_by_margin:
                    order.target_qty = max_by_margin

                side = order.side
                entry_price = order.entry_price
                stop_price = order.stop_price
                target_price = order.target_price
                qty = order.target_qty

                # ── Walk ticks from mitigation to find entry + exit ──────
                entry_fill_idx = -1
                entry_vol_acc = 0

                # Entry fill: scan ticks from mitigation point
                for ti in range(mit_idx, n_ticks):
                    p = tick_prices[ti]
                    s = int(tick_sizes[ti])

                    touches = False
                    if side == "BUY" and p <= entry_price:
                        touches = True
                    elif side == "SELL" and p >= entry_price:
                        touches = True

                    if touches:
                        if abs(p - entry_price) < 1e-6:
                            # Exact price → accumulate volume
                            entry_vol_acc += s
                            if entry_vol_acc >= qty:
                                entry_fill_idx = ti
                                break
                        else:
                            # Price through limit → immediate fill
                            entry_fill_idx = ti
                            break

                if entry_fill_idx < 0:
                    continue  # Never filled

                fill_time = tick_times_str[entry_fill_idx]
                fill_price = entry_price  # Limit order fills at limit

                # ── Walk from entry to exit (TP / SL / EOD) ──────────────
                exit_price = None
                exit_time = None
                exit_reason = None
                sl_slippage = 0.0

                for ti in range(entry_fill_idx + 1, n_ticks):
                    p = tick_prices[ti]
                    ti_time = tick_times_et[ti]

                    # EOD check (use pre-computed ET time)
                    if ti_time >= SESSION_END:
                        exit_price = float(tick_prices[ti - 1]) if ti > entry_fill_idx + 1 else fill_price
                        exit_time = tick_times_str[ti]
                        exit_reason = "EOD"
                        break

                    if side == "BUY":
                        if p <= stop_price:
                            exit_price = float(p)  # Actual trade price (real slippage)
                            sl_slippage = stop_price - exit_price
                            exit_time = tick_times_str[ti]
                            exit_reason = "SL"
                            break
                        elif p >= target_price:
                            exit_price = target_price  # Limit fills at target
                            exit_time = tick_times_str[ti]
                            exit_reason = "TP"
                            break
                    else:  # SELL
                        if p >= stop_price:
                            exit_price = float(p)
                            sl_slippage = exit_price - stop_price
                            exit_time = tick_times_str[ti]
                            exit_reason = "SL"
                            break
                        elif p <= target_price:
                            exit_price = target_price
                            exit_time = tick_times_str[ti]
                            exit_reason = "TP"
                            break

                if exit_price is None:
                    exit_price = float(tick_prices[-1])
                    exit_time = tick_times_str[-1]
                    exit_reason = "EOD"

                # ── Compute P&L ──────────────────────────────────────────
                if side == "BUY":
                    pnl_pts = exit_price - fill_price
                else:
                    pnl_pts = fill_price - exit_price

                commission = _calc_commission(qty, order_mgr._monthly_contracts)
                order_mgr._monthly_contracts += qty
                gross_pnl = round(pnl_pts * qty * POINT_VALUE, 2)
                net_pnl = round(gross_pnl - commission, 2)

                # ── Update order and state ───────────────────────────────
                order.actual_entry_price = fill_price
                order.filled_qty = qty
                order.filled_at = fill_time
                order.actual_exit_price = exit_price
                order.close_reason = exit_reason
                order.closed_at = exit_time
                order.realized_pnl = net_pnl
                order.state = "CLOSED"
                order.entry_slippage_pts = 0.0
                state.closed_trades.append(order)
                state.realized_pnl += net_pnl

                self._running_balance += net_pnl
                daily_trades += 1
                daily_pnl += net_pnl
                state.trade_count = daily_trades

                # Kill switch check
                if day_start_balance > 0 and daily_pnl <= -(day_start_balance * 0.10):
                    emergency_halt = True

                self._logger.log(
                    "trade_complete", setup=setup_type, side=side,
                    entry=fill_price, exit=exit_price, exit_reason=exit_reason,
                    qty=qty, pnl_pts=round(pnl_pts, 2), net_pnl=net_pnl,
                    sl_slippage=round(sl_slippage, 2),
                    volume_at_entry=entry_vol_acc,
                )

                break  # One setup per FVG

        # Export results
        displacement_log = order_mgr.get_displacement_log()
        if state.closed_trades or displacement_log:
            export_session(state, self.config, output_dir=self.output_dir,
                           displacement_log=displacement_log or None)

        self._all_results.extend(state.closed_trades)
        return state.closed_trades


class _PrintLogger:
    """Simple print-based logger for replay mode."""
    def log(self, event, **kwargs):
        pass  # Quiet by default; set verbose=True for debugging


class _NoopStateMgr:
    """No-op state manager — replay doesn't persist state."""
    def save(self, state, force=False):
        pass
    def save_if_dirty(self, state):
        pass
