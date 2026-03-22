import bisect
from datetime import datetime, timedelta

def generate_nq_expirations(start_year, end_year):
    """Generate NQ (E-mini Nasdaq) expiration dates - quarterly contracts."""
    expirations = []
    month_codes = [3, 6, 9, 12]  # March, June, September, December
    for year in range(start_year, end_year + 1):
        for month in month_codes:
            for day in range(15, 22):
                try:
                    d = datetime(year, month, day)
                    if d.weekday() == 4:  # Friday
                        expirations.append(d)
                        break
                except ValueError:
                    pass
    return expirations


def generate_es_expirations(start_year, end_year):
    """Generate ES (E-mini S&P 500) expiration dates - quarterly contracts."""
    expirations = []
    month_codes = [3, 6, 9, 12]  # March, June, September, December
    for year in range(start_year, end_year + 1):
        for month in month_codes:
            for day in range(15, 22):
                try:
                    d = datetime(year, month, day)
                    if d.weekday() == 4:  # Friday
                        expirations.append(d)
                        break
                except ValueError:
                    pass
    return expirations


def create_contract(exp_date, market_symbol="NQ", ib=None):
    """Create a futures contract for the specified market and expiration date."""
    exp_str = exp_date.strftime("%Y%m%d")

    # Market-specific contract creation
    if market_symbol == "NQ":
        symbol = "NQ"
    elif market_symbol == "ES":
        symbol = "ES"
    else:
        raise ValueError(
            f"Unsupported market symbol: {market_symbol}. Only NQ and ES are supported."
        )

    contract = Future(
        symbol=symbol,
        lastTradeDateOrContractMonth=exp_str,
        exchange="CME",
        currency="USD",
    )
    # Compare date to date, not datetime to date
    exp_date_as_date = exp_date.date() if hasattr(exp_date, "date") else exp_date
    if exp_date_as_date < datetime.now().date():
        contract.includeExpired = True
    if ib is not None:
        ib.qualifyContracts(contract)
    return contract


def get_contract_for_date(d, exp_dates, roll_days=8):
    if d.tzinfo is not None:
        d = d.replace(tzinfo=None)
    # Convert datetime to date for comparison with date objects
    if hasattr(d, "date"):
        d = d.date()
    # Convert exp_dates to date objects for consistent comparison
    exp_dates_as_dates = []
    for exp_date in exp_dates:
        if hasattr(exp_date, "date"):
            exp_dates_as_dates.append(exp_date.date())
        else:
            exp_dates_as_dates.append(exp_date)
    exp_dates_as_dates = sorted(exp_dates_as_dates)
    idx = bisect.bisect_right(exp_dates_as_dates, d)
    if idx == len(exp_dates):
        return exp_dates[-1]
    e = exp_dates[idx]
    roll_date = e - timedelta(days=roll_days)
    # Convert roll_date to date for comparison with d (which is a date)
    if hasattr(roll_date, "date"):
        roll_date = roll_date.date()
    if d >= roll_date:
        if idx + 1 < len(exp_dates):
            return exp_dates[idx + 1]
        else:
            return e
    else:
        return e
