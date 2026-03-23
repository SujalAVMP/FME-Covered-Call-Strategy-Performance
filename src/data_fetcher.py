"""Data acquisition: stock prices, option chains, historical volatility."""

import numpy as np
import pandas as pd
import yfinance as yf

from . import config


def get_stock_data(ticker=None, start=None, end=None):
    """Download price history via yfinance with automatic split/dividend adjustment."""
    ticker = ticker or config.DEFAULT_TICKER
    start = start or config.BACKTEST_START
    end = end or config.BACKTEST_END
    # yfinance treats `end` as exclusive, so shift by one day to make the
    # user-facing end date behave as an inclusive bound.
    inclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=inclusive_end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} from {start} to {end}")
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_option_chain(ticker=None, expiry_date=None):
    """Fetch live option chain (calls and puts) for a given expiry."""
    ticker = ticker or config.DEFAULT_TICKER
    tk = yf.Ticker(ticker)
    try:
        chain = tk.option_chain(expiry_date)
        return chain.calls, chain.puts
    except Exception:
        return None, None


def get_available_expiries(ticker=None):
    """Return list of available option expiry dates."""
    ticker = ticker or config.DEFAULT_TICKER
    tk = yf.Ticker(ticker)
    try:
        return list(tk.options)
    except Exception:
        return []


def compute_historical_volatility(prices, window=None):
    """Rolling annualized historical volatility from a price Series."""
    window = window or config.DEFAULT_VOL_WINDOW
    log_returns = np.log(prices / prices.shift(1))
    rolling_std = log_returns.rolling(window=window).std()
    return rolling_std * np.sqrt(config.TRADING_DAYS)


def compute_annualized_drift(prices, window=None):
    """Rolling annualized drift (mean log return) from a price Series."""
    window = window or config.DEFAULT_VOL_WINDOW
    log_returns = np.log(prices / prices.shift(1))
    rolling_mean = log_returns.rolling(window=window).mean()
    return rolling_mean * config.TRADING_DAYS
