"""Beta and Jensen-alpha computation versus the Nifty 50 benchmark."""

import time

import numpy as np
import pandas as pd

from . import config
from .data_fetcher import get_stock_data
from .universe import BENCHMARK_TICKER, sector_of


def compute_beta(stock_prices, benchmark_prices, rf=None):
    """OLS regression of daily log returns: stock on benchmark.

    Returns beta, annualized Jensen alpha on excess log returns, and R^2.
    """
    rf = config.RISK_FREE_RATE if rf is None else rf
    stock = pd.Series(stock_prices).dropna()
    bench = pd.Series(benchmark_prices).dropna()

    # Align on shared dates
    aligned = pd.concat([stock, bench], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {"beta": np.nan, "alpha_jensen": np.nan, "r_squared": np.nan, "n_obs": len(aligned)}

    s = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]

    r_s = np.log(s / s.shift(1)).dropna()
    r_b = np.log(b / b.shift(1)).dropna()

    common = r_s.index.intersection(r_b.index)
    r_s = r_s.loc[common]
    r_b = r_b.loc[common]

    var_b = np.var(r_b, ddof=1)
    if var_b == 0:
        return {"beta": np.nan, "alpha_jensen": np.nan, "r_squared": np.nan, "n_obs": len(common)}

    cov_sb = np.cov(r_s, r_b, ddof=1)[0, 1]
    beta = cov_sb / var_b

    rf_daily = np.log((1 + rf) ** (1 / config.TRADING_DAYS))
    r_s_excess = r_s - rf_daily
    r_b_excess = r_b - rf_daily

    # Jensen alpha (daily) annualized on excess returns
    alpha_daily = r_s_excess.mean() - beta * r_b_excess.mean()
    alpha_annual = alpha_daily * config.TRADING_DAYS

    # R-squared
    correlation = np.corrcoef(r_s, r_b)[0, 1]
    r_squared = correlation ** 2

    return {
        "beta": float(beta),
        "alpha_jensen": float(alpha_annual),
        "r_squared": float(r_squared),
        "n_obs": int(len(common)),
    }


def compute_beta_table(tickers, start=None, end=None, rf=None, sleep_seconds=0.3):
    """Compute beta for each ticker against the Nifty 50 benchmark.

    Returns a DataFrame with columns [ticker, sector, beta, alpha_jensen, r_squared, n_obs].
    Tickers that fail to download are skipped with a printed warning.
    """
    start = start or config.BACKTEST_START
    end = end or config.BACKTEST_END
    rf = config.RISK_FREE_RATE if rf is None else rf

    # Download benchmark once
    benchmark_df = get_stock_data(BENCHMARK_TICKER, start, end)
    benchmark_prices = benchmark_df["Close"].dropna()

    rows = []
    for ticker in tickers:
        try:
            stock_df = get_stock_data(ticker, start, end)
            stock_prices = stock_df["Close"].dropna()
            stats = compute_beta(stock_prices, benchmark_prices, rf=rf)
            stats["ticker"] = ticker
            stats["sector"] = sector_of(ticker)
            rows.append(stats)
        except Exception as exc:
            print(f"[beta_analysis] Skipping {ticker}: {exc}")
        time.sleep(sleep_seconds)

    if not rows:
        return pd.DataFrame(columns=["ticker", "sector", "beta", "alpha_jensen", "r_squared", "n_obs"])

    df = pd.DataFrame(rows)
    return df[["ticker", "sector", "beta", "alpha_jensen", "r_squared", "n_obs"]]
