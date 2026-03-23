"""Historical backtesting engine for the rolling covered call strategy."""

import numpy as np
import pandas as pd

from . import config
from .black_scholes import bs_call_price
from .data_fetcher import get_stock_data, compute_historical_volatility


def prepare_volatility_series(prices, window):
    """Return a fully populated annualized volatility series."""
    hist_vol = compute_historical_volatility(prices, window=window)
    hist_vol = hist_vol.where(hist_vol > 0)

    if hist_vol.dropna().empty:
        log_returns = np.log(prices / prices.shift(1)).dropna()
        fallback = log_returns.std() * np.sqrt(config.TRADING_DAYS) if not log_returns.empty else 0.20
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = 0.20
        return pd.Series(fallback, index=prices.index, dtype=float)

    return hist_vol.bfill().ffill()


def run_backtest(ticker=None, start=None, end=None,
                 strike_offset=None, expiry_days=None, vol_window=None):
    """Run a rolling covered call backtest.

    Strategy:
        1. Buy stock on day 0.
        2. Sell OTM call (strike = S * (1 + strike_offset)) expiring in expiry_days.
        3. At each expiry: settle the call, sell a new one.
        4. Track portfolio value daily for covered call vs stock-only.

    Returns a DataFrame with daily equity curves and metadata.
    """
    ticker = ticker or config.DEFAULT_TICKER
    start = start or config.BACKTEST_START
    end = end or config.BACKTEST_END
    strike_offset = strike_offset if strike_offset is not None else config.DEFAULT_STRIKE_OFFSET
    expiry_days = expiry_days or config.DEFAULT_EXPIRY_DAYS
    vol_window = vol_window or config.DEFAULT_VOL_WINDOW

    if expiry_days <= 0:
        raise ValueError("expiry_days must be positive")

    # Fetch data
    df = get_stock_data(ticker, start, end)
    prices = df["Close"].copy()
    prices = prices.dropna()

    # Estimate daily option values with a fully populated volatility series.
    hist_vol = prepare_volatility_series(prices, window=vol_window)

    # Initialize tracking
    n = len(prices)
    dates = prices.index
    price_arr = prices.values.flatten()

    # Stock-only equity: normalized to start at 1.0
    stock_equity = price_arr / price_arr[0]

    # Covered call equity with the short call marked to market each day.
    cc_equity = np.ones(n)
    cash = 0.0
    current_strike = None
    days_to_expiry = 0
    roll_dates = []

    for i in range(n):
        S = price_arr[i]
        sigma = float(hist_vol.iloc[i])

        # Roll: sell new call at start or at expiry
        if current_strike is None or days_to_expiry <= 0:
            # Settle previous call if exists
            if current_strike is not None:
                cash -= max(S - current_strike, 0.0)

            # Sell new call
            K = S * (1 + strike_offset)
            T = expiry_days / config.TRADING_DAYS
            premium = bs_call_price(S, K, T, config.RISK_FREE_RATE, sigma)
            cash += premium
            current_strike = K
            days_to_expiry = expiry_days
            roll_dates.append(dates[i])

        remaining_T = days_to_expiry / config.TRADING_DAYS
        call_value = bs_call_price(S, current_strike, remaining_T, config.RISK_FREE_RATE, sigma)

        days_to_expiry -= 1

        # Portfolio value = long stock + cash from premiums/settlements - short call liability.
        cc_equity[i] = (S + cash - call_value) / price_arr[0]

    results = pd.DataFrame({
        "date": dates,
        "price": price_arr,
        "stock_equity": stock_equity,
        "cc_equity": cc_equity,
    }).set_index("date")

    # Compute daily returns
    results["stock_return"] = results["stock_equity"].pct_change()
    results["cc_return"] = results["cc_equity"].pct_change()

    return results, roll_dates


def compute_monthly_returns(backtest_df, column):
    """Resample equity curve to monthly returns."""
    equity = backtest_df[column]
    monthly = equity.resample("ME").last()
    return monthly.pct_change().dropna()
