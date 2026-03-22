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


def calculate_setup(fvg, cell_config, balance, risk_pct=0.01):
    """
    Calculate a complete trade setup from an FVG and its matching strategy cell.

    Args:
        fvg: FVGRecord (the mitigated FVG)
        cell_config: dict from StrategyLoader.find_cell() with:
            setup, rr_target, median_risk, ev, win_rate, samples
        balance: current account balance in dollars
        risk_pct: fraction of balance to risk per trade (default 0.01 = 1%)

    Returns:
        OrderGroup if valid setup, None if risk is 0 or outside range.
    """
    setup = cell_config["setup"]
    n_value = cell_config["rr_target"]

    # Calculate entry price
    if setup == "mit_extreme":
        # Entry at FVG zone edge (mitigation level)
        if fvg.fvg_type == "bullish":
            entry = round_to_tick(fvg.zone_high)
        else:  # bearish
            entry = round_to_tick(fvg.zone_low)
    elif setup == "mid_extreme":
        # Entry at FVG midpoint
        midpoint = (fvg.zone_high + fvg.zone_low) / 2
        entry = round_to_tick(midpoint)
    else:
        return None

    # Calculate stop price (middle candle extreme)
    if fvg.fvg_type == "bullish":
        stop = round_to_tick(fvg.middle_low)
    else:  # bearish
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
    else:  # bearish
        target = round_to_tick(entry - target_distance)

    # Determine side
    side = "BUY" if fvg.fvg_type == "bullish" else "SELL"

    # Position sizing
    qty = calculate_position_size(balance, risk_pct, risk_pts)

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
