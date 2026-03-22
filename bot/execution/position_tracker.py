"""
position_tracker.py — Track open positions and calculate P&L.
"""

from bot.state.trade_state import CLOSE_TP, CLOSE_SL

POINT_VALUE = 20.0  # NQ


def calculate_pnl(order_group, fill_price, point_value=POINT_VALUE):
    """
    Calculate realized P&L for a closing fill.

    Bullish (BUY): pnl = (fill_price - entry_price) * qty * point_value
    Bearish (SELL): pnl = (entry_price - fill_price) * qty * point_value

    Args:
        order_group: OrderGroup
        fill_price: the fill price of TP or SL
        point_value: dollar value per point

    Returns:
        P&L in dollars (positive = profit, negative = loss)
    """
    qty = order_group.filled_qty or order_group.target_qty

    if order_group.side == "BUY":
        pnl = (fill_price - order_group.entry_price) * qty * point_value
    else:  # SELL
        pnl = (order_group.entry_price - fill_price) * qty * point_value

    return round(pnl, 2)


def determine_close_reason(order_group, fill_price):
    """Determine if a fill was TP or SL based on fill price."""
    if order_group.side == "BUY":
        if fill_price >= order_group.target_price - 0.25:
            return CLOSE_TP
        return CLOSE_SL
    else:
        if fill_price <= order_group.target_price + 0.25:
            return CLOSE_TP
        return CLOSE_SL


async def get_account_balance(ib_connection):
    """Query IB for current NetLiquidation value."""
    if not ib_connection.is_connected:
        return None

    ib = ib_connection.ib
    account_values = ib.accountValues()
    for av in account_values:
        if av.tag == "NetLiquidation" and av.currency == "USD":
            return float(av.value)
    return None


async def get_ib_positions(ib_connection):
    """Query IB for current open positions."""
    if not ib_connection.is_connected:
        return []
    return ib_connection.ib.positions()


async def get_ib_open_orders(ib_connection):
    """Query IB for current open orders."""
    if not ib_connection.is_connected:
        return []
    return ib_connection.ib.openOrders()
