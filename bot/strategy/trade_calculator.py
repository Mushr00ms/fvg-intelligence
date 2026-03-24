"""
trade_calculator.py — Calculate trade entry, stop, target, and position sizing.

All prices are rounded to NQ tick size (0.25 points).
"""

import math

from bot.state.trade_state import OrderGroup, _new_id

# NQ contract specs (also available from config, but kept here for clarity)
TICK_SIZE = 0.25
POINT_VALUE = 20.0


def round_to_tick(price, tick=TICK_SIZE):
    """Round a price to the nearest tick."""
    return round(price / tick) * tick


def get_risk_pct_for_bucket(risk_range, config=None):
    """
    Return the risk % for a given risk bucket using 3-tier Kelly-inspired sizing.

    Args:
        risk_range: str like '5-10', '15-20', '40-80'
        config: BotConfig (or None for default 1%)

    Returns:
        float risk fraction (e.g. 0.005, 0.015, 0.03)
    """
    if config is None or not getattr(config, 'use_risk_tiers', False):
        return getattr(config, 'risk_per_trade', 0.01) if config else 0.01

    if risk_range in getattr(config, 'large_buckets', ['40-80']):
        return config.risk_large_pct
    elif risk_range in getattr(config, 'small_buckets', ['5-10', '10-15']):
        return config.risk_small_pct
    else:
        return config.risk_medium_pct


def risk_to_range(risk_pts):
    """Map risk in points to risk range bucket string."""
    bins = [5, 10, 15, 20, 25, 30, 40, 80]
    for i in range(len(bins) - 1):
        if bins[i] <= risk_pts < bins[i + 1]:
            return f"{bins[i]}-{bins[i + 1]}"
    return None


def calculate_setup(fvg, cell_config, balance, risk_pct=0.01, config=None):
    """
    Calculate a complete trade setup from an FVG and its matching strategy cell.

    Args:
        fvg: FVGRecord (the mitigated FVG)
        cell_config: dict from StrategyLoader.find_cell() with:
            setup, rr_target, median_risk, ev, win_rate, samples
        balance: current account balance in dollars
        risk_pct: fallback risk fraction (used if config is None or tiers disabled)
        config: BotConfig for risk tiers and slippage settings

    Returns:
        OrderGroup if valid setup, None if risk is 0 or outside range.
    """
    setup = cell_config["setup"]
    n_value = cell_config["rr_target"]

    # Determine side
    side = "BUY" if fvg.fvg_type == "bullish" else "SELL"

    # Calculate entry price
    if setup == "mit_extreme":
        if fvg.fvg_type == "bullish":
            entry = round_to_tick(fvg.zone_high)
        else:
            entry = round_to_tick(fvg.zone_low)
    elif setup == "mid_extreme":
        midpoint = (fvg.zone_high + fvg.zone_low) / 2
        entry = round_to_tick(midpoint)
    else:
        return None

    # Slippage: entry 1 tick deeper into the zone
    use_slip = getattr(config, 'use_slippage', False) if config else False
    slip_ticks = getattr(config, 'slippage_ticks', 1) if config else 1
    if use_slip:
        slip_amount = TICK_SIZE * slip_ticks
        if side == "BUY":
            entry = round_to_tick(entry - slip_amount)
        else:
            entry = round_to_tick(entry + slip_amount)

    # Calculate stop price (middle candle extreme) — no slippage on stop
    if fvg.fvg_type == "bullish":
        stop = round_to_tick(fvg.middle_low)
    else:
        stop = round_to_tick(fvg.middle_high)

    # Calculate risk
    risk_pts = round_to_tick(abs(entry - stop))
    if risk_pts <= 0:
        return None

    # Validate risk falls in the cell's risk range
    risk_range = cell_config.get("_risk_range", "")
    if risk_range:
        try:
            lo, hi = map(float, risk_range.split("-"))
            if not (lo <= risk_pts < hi):
                return None
        except (ValueError, AttributeError):
            pass

    # Calculate target price
    target_distance = round_to_tick(n_value * risk_pts)
    if fvg.fvg_type == "bullish":
        target = round_to_tick(entry + target_distance)
    else:
        target = round_to_tick(entry - target_distance)

    # TP trigger: price must trade 1 tick past the target to fill the limit order.
    # The limit order sits at `target`; `tp_trigger` is the actual fill condition.
    # The OrderManager uses tp_trigger for monitoring, fills at target.
    if use_slip:
        slip_amount = TICK_SIZE * slip_ticks
        if side == "BUY":
            tp_trigger = round_to_tick(target + slip_amount)
        else:
            tp_trigger = round_to_tick(target - slip_amount)
    else:
        tp_trigger = target

    # Position sizing — 3-tier risk
    bucket = risk_to_range(risk_pts) or risk_range
    actual_risk_pct = get_risk_pct_for_bucket(bucket, config) if config else risk_pct
    qty = calculate_position_size(balance, actual_risk_pct, risk_pts)

    return OrderGroup(
        group_id=_new_id(),
        fvg_id=fvg.fvg_id,
        setup=setup,
        side=side,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        risk_pts=risk_pts,
        n_value=n_value,
        target_qty=qty,
        risk_pct=actual_risk_pct,
    )


def calculate_position_size(balance, risk_pct, risk_pts, point_value=POINT_VALUE):
    """
    Calculate number of contracts based on account risk.

    contracts = floor(balance * risk_pct / (risk_pts * point_value))
    Minimum: 1 contract.

    Args:
        balance: account balance in dollars
        risk_pct: fraction of balance to risk (e.g. 0.01 = 1%)
        risk_pts: risk distance in points
        point_value: dollar value per point (NQ = $20)

    Returns:
        Number of contracts (int, >= 1)
    """
    if risk_pts <= 0 or balance <= 0:
        return 1

    risk_budget = balance * risk_pct
    risk_per_contract = risk_pts * point_value
    contracts = math.floor(risk_budget / risk_per_contract)

    return max(1, contracts)
