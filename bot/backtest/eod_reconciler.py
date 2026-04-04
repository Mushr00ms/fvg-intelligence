"""
eod_reconciler.py — End-of-day reconciliation: compare live trades vs backtest.

Called by the engine after _eod_cleanup() at 16:00 ET. Downloads today's
1-second bars from IB, runs the backtester on the same strategy, and
produces a divergence report sent via Telegram.

Pure functions — no async, no I/O. The engine handles orchestration.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from bot.backtest.us_holidays import US_MARKET_HOLIDAYS


# ── Weekly expectations (from 115-week backtest, 2024-2026, $100k base) ──

WEEKLY_EXPECTATIONS = {
    "trades": {"median": 20, "p25": 16, "p75": 24, "p5": 11},
    "win_rate": {"median": 42, "p25": 33, "p75": 47, "p5": 23},
    "pnl_100k": {"median": 2000, "p25": -1400, "p75": 6500, "p5": -7400, "worst": -13800},
}

# Tolerances for trade comparison
ENTRY_PRICE_TOL = 0.50   # points
PNL_PTS_TOL = 2.0        # points


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class Divergence:
    """A single divergence between live and backtest."""
    severity: str           # MISSED_LIVE, MISSED_BACKTEST, EXIT_MISMATCH, PRICE_DRIFT
    cell_key: str           # "10:30-11:00 | 15-20 | mit_extreme | SELL"
    live_detail: str        # Summary of live trade (or "N/A")
    backtest_detail: str    # Summary of backtest trade (or "N/A")


@dataclass
class ReconciliationResult:
    """Full reconciliation result for one trading day."""
    date: str
    live_count: int
    backtest_count: int
    matched_count: int
    divergences: list = field(default_factory=list)
    live_net_pnl: float = 0.0
    backtest_net_pnl: float = 0.0
    corrected_net_pnl: float = 0.0  # backtest prices × live sizing
    kill_switch_active: bool = False
    hfoiv_active: bool = False
    error: Optional[str] = None
    # Daily summary fields (set by engine before formatting)
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    balance: float = 0.0
    filled_trades: int = 0


# ── Trade normalization ──────────────────────────────────────────────────

def _cell_key(time_period, risk_range, setup, side):
    """Build a human-readable cell key for display."""
    return f"{time_period} | {risk_range} | {setup} | {side}"


def _normalize_live(trade):
    """Normalize a live trade dict (from DB) into comparison dict."""
    return {
        "time_period": trade["time_period"],
        "risk_range": trade["risk_range"],
        "setup": trade["setup"],
        "side": trade["side"],
        "entry_price": trade.get("actual_entry_price") or trade["entry_price"],
        "stop_price": trade["stop_price"],
        "target_price": trade["target_price"],
        "contracts": trade["contracts"],
        "exit_reason": trade.get("exit_reason", ""),
        "pnl_pts": trade.get("pnl_pts", 0),
        "net_pnl": trade.get("net_pnl", 0),
        "entry_time": trade.get("entry_time", ""),
        "n_value": trade.get("n_value", 0),
        "source": "live",
    }


def _normalize_backtest(trade):
    """Normalize a backtest Trade object into comparison dict."""
    return {
        "time_period": trade.time_period,
        "risk_range": trade.risk_range,
        "setup": trade.setup,
        "side": trade.side,
        "entry_price": trade.entry_price,
        "stop_price": trade.stop_price,
        "target_price": trade.target_price,
        "contracts": trade.contracts,
        "exit_reason": trade.exit_reason,
        "pnl_pts": trade.pnl_pts,
        "net_pnl": trade.pnl_dollars,
        "entry_time": trade.entry_time,
        "n_value": trade.n_value,
        "source": "backtest",
    }


# ── Matching ─────────────────────────────────────────────────────────────

def _group_key(t):
    return (t["time_period"], t["risk_range"], t["setup"], t["side"])


def _match_within_cell(live_group, bt_group):
    """
    Match trades within a single cell group by entry price proximity.
    Returns (matched_pairs, unmatched_live, unmatched_bt).
    """
    matched = []
    used_bt = set()
    unmatched_live = []

    for lt in live_group:
        best_idx = None
        best_diff = float("inf")
        for i, bt in enumerate(bt_group):
            if i in used_bt:
                continue
            diff = abs(lt["entry_price"] - bt["entry_price"])
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        if best_idx is not None and best_diff <= ENTRY_PRICE_TOL:
            matched.append((lt, bt_group[best_idx]))
            used_bt.add(best_idx)
        else:
            unmatched_live.append(lt)

    unmatched_bt = [bt for i, bt in enumerate(bt_group) if i not in used_bt]
    return matched, unmatched_live, unmatched_bt


def _trade_summary(t):
    """One-line summary of a normalized trade."""
    exit_r = t["exit_reason"] or "open"
    return f"{exit_r} @ {t['entry_price']:.2f}, {t['contracts']}ct, {t['pnl_pts']:+.1f}pts"


def _compare_matched(live, bt, cell_key, hfoiv_active=False):
    """Compare a matched pair and return any divergences."""
    divs = []

    # Exit reason mismatch — most important
    if live["exit_reason"] != bt["exit_reason"]:
        divs.append(Divergence(
            severity="EXIT_MISMATCH",
            cell_key=cell_key,
            live_detail=f"{live['exit_reason']} → {live['pnl_pts']:+.1f}pts",
            backtest_detail=f"{bt['exit_reason']} → {bt['pnl_pts']:+.1f}pts",
        ))
        return divs  # no point checking price drift if exit differs

    # Contract count mismatch
    if live["contracts"] != bt["contracts"]:
        live_detail = f"{live['contracts']}ct"
        bt_detail = f"{bt['contracts']}ct"
        # HFOIV reduces live sizing — backtester lacks volatility history,
        # so fewer contracts on live side is expected, not a real divergence.
        if hfoiv_active and live["contracts"] < bt["contracts"]:
            live_detail += " ⬇ HFOIV"
            severity = "HFOIV_EXPECTED"
        else:
            severity = "EXIT_MISMATCH"
        divs.append(Divergence(
            severity=severity,
            cell_key=cell_key,
            live_detail=live_detail,
            backtest_detail=bt_detail,
        ))

    # Price drift
    entry_diff = abs(live["entry_price"] - bt["entry_price"])
    pnl_diff = abs(live["pnl_pts"] - bt["pnl_pts"])
    if entry_diff > ENTRY_PRICE_TOL or pnl_diff > PNL_PTS_TOL:
        divs.append(Divergence(
            severity="PRICE_DRIFT",
            cell_key=cell_key,
            live_detail=f"entry={live['entry_price']:.2f} pnl={live['pnl_pts']:+.1f}pts",
            backtest_detail=f"entry={bt['entry_price']:.2f} pnl={bt['pnl_pts']:+.1f}pts",
        ))

    return divs


def match_trades(live_trades, backtest_trades, hfoiv_active=False):
    """
    Match live DB trades to backtest Trade objects and find divergences.

    Args:
        live_trades: List of dicts from TradeDB.get_trades(date=today)
        backtest_trades: List of backtester.Trade objects
        hfoiv_active: Whether the HFOIV gate was active on the live bot

    Returns:
        ReconciliationResult with matched trades and divergences
    """
    # Normalize
    live_norm = [_normalize_live(t) for t in live_trades if t.get("exit_reason")]
    bt_norm = [_normalize_backtest(t) for t in backtest_trades]

    # Group by cell key
    live_groups = defaultdict(list)
    bt_groups = defaultdict(list)
    for t in live_norm:
        live_groups[_group_key(t)].append(t)
    for t in bt_norm:
        bt_groups[_group_key(t)].append(t)

    # Sort each group by entry time
    for g in live_groups.values():
        g.sort(key=lambda t: t["entry_time"])
    for g in bt_groups.values():
        g.sort(key=lambda t: t["entry_time"])

    all_keys = set(live_groups.keys()) | set(bt_groups.keys())
    divergences = []
    matched_count = 0
    corrected_pnl = 0.0  # backtest prices × live sizing
    POINT_VALUE = 20.0    # NQ $20/point

    for key in sorted(all_keys):
        lg = live_groups.get(key, [])
        bg = bt_groups.get(key, [])
        ck = _cell_key(*key)

        matched, unmatched_live, unmatched_bt = _match_within_cell(lg, bg)
        matched_count += len(matched)

        # Check matched pairs for divergences
        for lt, bt in matched:
            divergences.extend(_compare_matched(lt, bt, ck, hfoiv_active))
            # Corrected P&L: use backtest pnl_pts (real price action)
            # but live contract count (HFOIV-adjusted sizing is correct)
            corrected_pnl += bt["pnl_pts"] * POINT_VALUE * lt["contracts"]

        # Unmatched live trades — use live P&L as-is (backtest has no data)
        for lt in unmatched_live:
            divergences.append(Divergence(
                severity="MISSED_LIVE",
                cell_key=ck,
                live_detail=_trade_summary(lt),
                backtest_detail="N/A — not taken by backtester",
            ))
            corrected_pnl += lt["net_pnl"]

        # Unmatched backtest trades — not taken live, don't add to corrected
        for bt in unmatched_bt:
            divergences.append(Divergence(
                severity="MISSED_BACKTEST",
                cell_key=ck,
                live_detail="N/A — not taken by live bot",
                backtest_detail=_trade_summary(bt),
            ))

    live_net = sum(t["net_pnl"] for t in live_norm)
    bt_net = sum(t["net_pnl"] for t in bt_norm)

    return ReconciliationResult(
        date=live_trades[0]["trade_date"] if live_trades else "",
        live_count=len(live_norm),
        backtest_count=len(bt_norm),
        matched_count=matched_count,
        divergences=divergences,
        live_net_pnl=round(live_net, 2),
        backtest_net_pnl=round(bt_net, 2),
        corrected_net_pnl=round(corrected_pnl, 2),
    )


# ── Telegram report formatting ──────────────────────────────────────────

def format_telegram_report(result, weekly_html=None, fills_garbage=False):
    """Format the reconciliation result as an HTML Telegram message."""
    # Daily summary line — included in all report variants
    pnl = result.daily_pnl
    pnl_pct = result.daily_pnl_pct
    sign = "+" if pnl >= 0 else ""
    daily_line = f"💰 <b>{sign}${pnl:,.0f}</b> ({sign}{pnl_pct:.1f}%)  ·  {result.filled_trades} trades  ·  ${result.balance:,.0f}"

    if result.error:
        return (
            "🔴 <b>RECONCILIATION SKIPPED</b>\n\n"
            f"📅 {result.date}\n"
            f"{daily_line}\n\n"
            f"Reason: {result.error}"
        )

    if result.live_count == 0 and result.backtest_count == 0:
        msg = (
            "⚪ <b>RECONCILIATION</b>\n\n"
            f"📅 {result.date}\n"
            f"{daily_line}\n\n"
            "No trades on either side."
        )
        if weekly_html:
            msg += f"\n\n{weekly_html}"
        return msg

    # Divergence counts by type
    severe = [d for d in result.divergences if d.severity in ("MISSED_LIVE", "MISSED_BACKTEST", "EXIT_MISMATCH")]
    drifts = [d for d in result.divergences if d.severity == "PRICE_DRIFT"]
    hfoiv = [d for d in result.divergences if d.severity == "HFOIV_EXPECTED"]

    # Header — color-coded by severity
    if not severe:
        header = "🟢 <b>RECONCILIATION — CLEAN</b>"
    elif len(severe) <= 2:
        header = "🟡 <b>RECONCILIATION — MINOR</b>"
    else:
        header = "🔴 <b>RECONCILIATION — CHECK</b>"

    lines = [
        header,
        f"📅 {result.date}",
        f"{daily_line}\n",
        f"📊 Live {result.live_count}  ·  Backtest {result.backtest_count}  ·  Matched {result.matched_count}",
    ]

    if severe:
        lines.append(f"\n⚠️ <b>{len(severe)} divergence{'s' if len(severe) != 1 else ''}</b>\n")
        for d in severe:
            icon = {"MISSED_BACKTEST": "👻", "MISSED_LIVE": "❓", "EXIT_MISMATCH": "🔀"}.get(d.severity, "·")
            lines.append(f"{icon} <b>{d.cell_key}</b>")
            if d.severity == "MISSED_BACKTEST":
                lines.append(f"     BT took: {d.backtest_detail}")
                lines.append(f"     Live: skipped")
            elif d.severity == "MISSED_LIVE":
                lines.append(f"     Live took: {d.live_detail}")
                lines.append(f"     BT: skipped")
            else:
                lines.append(f"     Live: {d.live_detail}")
                lines.append(f"     BT:   {d.backtest_detail}")
    elif result.matched_count > 0:
        lines.append(f"\n✅ All {result.matched_count} trades match")

    if drifts:
        lines.append(f"\n📐 {len(drifts)} price drift{'s' if len(drifts) != 1 else ''} (minor)")

    if hfoiv:
        for d in hfoiv:
            lines.append(f"\n⬇ {d.cell_key}: {d.live_detail} (BT {d.backtest_detail})")

    # P&L bar
    if fills_garbage and result.matched_count > 0:
        # Paper fills unreliable — use backtest prices × live sizing
        true_pnl = result.corrected_net_pnl
        pnl_icon = "📈" if true_pnl >= 0 else "📉"
        lines.append(f"\n{pnl_icon} <b>P&L</b> (paper fills unreliable)")
        pnl_warn = "  ⚠️" if round(result.live_net_pnl) != round(true_pnl) else ""
        lines.append(f"     Paper:     ${result.live_net_pnl:>+,.0f}{pnl_warn}")
        lines.append(f"     Corrected: ${true_pnl:>+,.0f}")
    else:
        delta = result.live_net_pnl - result.backtest_net_pnl
        live_icon = "📈" if result.live_net_pnl >= 0 else "📉"
        lines.append(f"\n{live_icon} <b>P&L</b>")
        lines.append(f"     Live:     ${result.live_net_pnl:>+,.0f}")
        lines.append(f"     Backtest: ${result.backtest_net_pnl:>+,.0f}")
        lines.append(f"     Delta:    ${delta:>+,.0f}")

    if result.kill_switch_active:
        lines.append("\n🛑 Kill switch fired — backtest-only trades expected")

    if weekly_html:
        lines.append(f"\n{'─' * 28}\n{weekly_html}")

    return "\n".join(lines)


# ── Weekly health check (Fridays, or Thursday before holiday Friday) ───

def _is_weekly_summary_day(today):
    """
    Return True if the weekly summary should be sent today.

    Triggers on Friday, OR on Thursday if Friday is a market holiday.
    """
    if today.weekday() == 4:  # Friday
        return True
    if today.weekday() == 3:  # Thursday — check if Friday is a holiday
        friday = today + timedelta(days=1)
        friday_key = friday.strftime("%Y%m%d")
        return friday_key in US_MARKET_HOLIDAYS
    return False


def build_weekly_summary(db, today_str, account_balance):
    """
    Build weekly health check HTML.

    Sent on Friday, or on Thursday if Friday is a market holiday.

    Args:
        db: TradeDB instance
        today_str: "YYYY-MM-DD" date string
        account_balance: Current account balance for scaling expectations

    Returns:
        HTML string or None if not a weekly summary day.
    """
    today = datetime.strptime(today_str, "%Y-%m-%d")
    if not _is_weekly_summary_day(today):
        return None

    # Find Monday of this week
    monday = today - timedelta(days=today.weekday())
    monday_str = monday.strftime("%Y-%m-%d")
    end_str = today_str
    end_day = today.strftime("%a")  # "Thu" or "Fri"

    # Query daily_stats for the week
    rows = db.query(
        "SELECT * FROM daily_stats WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        [monday_str, end_str],
    )

    if not rows:
        return (
            "<b>WEEKLY HEALTH CHECK</b>\n"
            f"Mon {monday_str} - {end_day} {end_str}\n"
            "No daily stats recorded this week."
        )

    total_trades = sum(r["total_trades"] for r in rows)
    total_wins = sum(r["wins"] for r in rows)
    total_net = sum(r["net_pnl"] for r in rows)
    win_rate = round(total_wins / total_trades * 100, 1) if total_trades else 0

    # Scale expectations to actual account balance
    scale = account_balance / 100_000
    exp = WEEKLY_EXPECTATIONS
    pnl_p25 = exp["pnl_100k"]["p25"] * scale
    pnl_p75 = exp["pnl_100k"]["p75"] * scale
    pnl_p5 = exp["pnl_100k"]["p5"] * scale

    lines = [
        "<b>WEEKLY HEALTH CHECK</b>",
        f"Mon {monday_str} - {end_day} {end_str}\n",
    ]

    # Trades
    t_ok = exp["trades"]["p25"] <= total_trades <= exp["trades"]["p75"]
    t_alarm = total_trades < exp["trades"]["p5"]
    if t_alarm:
        lines.append(f"Trades: {total_trades} !! (alarm &lt;{exp['trades']['p5']})")
        lines.append("  Check: connection, contract roll, strategy filter")
    elif not t_ok:
        lines.append(f"Trades: {total_trades} (normal: {exp['trades']['p25']}-{exp['trades']['p75']})")
    else:
        lines.append(f"Trades: {total_trades} (normal: {exp['trades']['p25']}-{exp['trades']['p75']})")

    # Win rate
    wr_ok = exp["win_rate"]["p25"] <= win_rate <= exp["win_rate"]["p75"]
    wr_alarm = win_rate < exp["win_rate"]["p5"]
    if wr_alarm:
        lines.append(f"Win rate: {win_rate:.0f}% !! (alarm &lt;{exp['win_rate']['p5']}%)")
        lines.append("  Check: fills, exit logic, slippage")
    elif not wr_ok:
        lines.append(f"Win rate: {win_rate:.0f}% (normal: {exp['win_rate']['p25']}-{exp['win_rate']['p75']}%)")
    else:
        lines.append(f"Win rate: {win_rate:.0f}% (normal: {exp['win_rate']['p25']}-{exp['win_rate']['p75']}%)")

    # Net P&L
    pnl_ok = pnl_p25 <= total_net <= pnl_p75
    pnl_alarm = total_net < pnl_p5
    if pnl_alarm:
        lines.append(f"Net P&L: ${total_net:+,.0f} !! (alarm &lt;${pnl_p5:,.0f})")
        lines.append("  Check: slippage, SL clusters, sizing")
    elif not pnl_ok:
        lines.append(f"Net P&L: ${total_net:+,.0f} (normal: ${pnl_p25:,.0f} to ${pnl_p75:,.0f})")
    else:
        lines.append(f"Net P&L: ${total_net:+,.0f} (normal: ${pnl_p25:,.0f} to ${pnl_p75:,.0f})")

    # Overall verdict
    all_ok = t_ok and wr_ok and pnl_ok
    any_alarm = t_alarm or wr_alarm or pnl_alarm
    if all_ok:
        lines.append("\nAll metrics within expected range.")
    elif any_alarm:
        lines.append("\nOne or more metrics outside alarm threshold — investigate.")

    # Reminder (always included)
    lines.append(
        "\nReminder:"
        "\n- 1 in 3 weeks will be red — normal"
        "\n- Up to 4 consecutive red weeks seen in backtest"
        "\n- Edge comes from winners ~1.7x larger than losers"
    )

    return "\n".join(lines)


# ── Backtest config builder ──────────────────────────────────────────────

def build_backtest_config(bot_config, strategy_dict, start_balance):
    """
    Build backtester config dict from BotConfig + loaded strategy.

    Maps the bot's runtime parameters to the backtester's config dictionary
    format, ensuring the backtest uses the exact same rules the live bot used.
    """
    meta = strategy_dict.get("meta", {})

    config = {
        "balance": start_balance,
        "risk_pct": bot_config.risk_per_trade,
        "max_concurrent": bot_config.max_concurrent,
        "max_daily_trades": bot_config.max_daily_trades,
        "min_fvg_size": bot_config.min_fvg_size,
        "slip": False,          # Live bot uses limit orders
        "mit_entry_ticks": 0,   # Live bot uses limit orders
        "tp_mode": "fixed",
        "risk_tiers": bot_config.use_risk_tiers,
        "margin_per_contract": bot_config.margin_intraday_initial,
        "hfoiv": {
            "enabled": False,  # Default off — overridden if bot uses it
        },
    }

    # Mirror risk tier settings from strategy meta if available
    if bot_config.use_risk_tiers and "risk_rules" in meta:
        config["risk_tiers"] = True

    return config


# ── Tick-based fill validation ────────────────────────────────────────────

@dataclass
class FillCheck:
    """Result of tick-validating a single fill price."""
    group_id: str
    fill_type: str          # "entry" or "exit"
    fill_price: float
    fill_time: str
    nearest_tick_price: float
    tick_distance_pts: float
    volume_at_price: int    # total volume traded at fill_price in window
    valid: bool             # True if fill price actually traded


def validate_fills(live_trades, ticks_by_window):
    """
    Validate live trade fill prices against actual tick data.

    Args:
        live_trades: List of DB trade dicts (closed only)
        ticks_by_window: dict mapping (group_id, "entry"|"exit") -> list of tick dicts
                         Each tick: {"price": float, "size": int, "time_utc": str}

    Returns:
        list of FillCheck objects (only for fills that could be checked)
    """
    checks = []
    for t in live_trades:
        gid = t["group_id"]
        for fill_type, price_key, time_key in [
            ("entry", "actual_entry_price", "entry_time"),
            ("exit", "actual_exit_price", "exit_time"),
        ]:
            fill_price = t.get(price_key)
            fill_time = t.get(time_key)
            if not fill_price or not fill_time:
                continue

            window_key = (gid, fill_type)
            ticks = ticks_by_window.get(window_key, [])
            if not ticks:
                continue

            # Check if fill price actually traded
            volume_at = sum(tk["size"] for tk in ticks if tk["price"] == fill_price)
            nearest = min(ticks, key=lambda tk: abs(tk["price"] - fill_price))
            dist = abs(nearest["price"] - fill_price)

            checks.append(FillCheck(
                group_id=gid,
                fill_type=fill_type,
                fill_price=fill_price,
                fill_time=fill_time,
                nearest_tick_price=nearest["price"],
                tick_distance_pts=round(dist, 2),
                volume_at_price=volume_at,
                valid=volume_at > 0,
            ))

    return checks


def has_bad_fills(checks):
    """Return True if any fills failed tick validation."""
    return any(not c.valid for c in checks)


# ── Serialization for DB ─────────────────────────────────────────────────

def result_to_db_kwargs(result):
    """Convert ReconciliationResult to kwargs for insert_reconciliation()."""
    divs_json = json.dumps([
        {
            "severity": d.severity,
            "cell_key": d.cell_key,
            "live_detail": d.live_detail,
            "backtest_detail": d.backtest_detail,
        }
        for d in result.divergences
    ])
    return {
        "trade_date": result.date,
        "live_count": result.live_count,
        "backtest_count": result.backtest_count,
        "matched_count": result.matched_count,
        "divergence_count": len(result.divergences),
        "live_net_pnl": result.live_net_pnl,
        "backtest_net_pnl": result.backtest_net_pnl,
        "divergences_json": divs_json,
        "error": result.error,
    }
