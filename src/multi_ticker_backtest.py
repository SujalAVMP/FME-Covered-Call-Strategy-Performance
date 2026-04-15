"""Cross-sectional orchestrator: runs the rolling covered-call backtest over a basket
of tickers, then aggregates results for sector / tenure / beta analysis."""

import time

import numpy as np
import pandas as pd

from . import config
from .backtest import run_backtest
from .risk_metrics import (annualized_return, annualized_volatility,
                            max_drawdown, sharpe_ratio, win_rate)
from .universe import sector_of


def run_universe_backtest(tickers, start=None, end=None,
                          strike_offset=None, expiry_days=None, vol_window=None,
                          sleep_seconds=0.3):
    """Run run_backtest() for each ticker. Returns {ticker: results_df}.

    Tickers that fail (e.g., missing data, yfinance error) are skipped with
    a warning rather than crashing the whole loop.
    """
    results = {}
    for ticker in tickers:
        try:
            df, _ = run_backtest(
                ticker=ticker,
                start=start,
                end=end,
                strike_offset=strike_offset,
                expiry_days=expiry_days,
                vol_window=vol_window,
            )
            results[ticker] = df
        except Exception as exc:
            print(f"[multi_ticker_backtest] Skipping {ticker}: {exc}")
        time.sleep(sleep_seconds)
    return results


def _total_return(equity_series):
    """Total return from a normalized equity curve (starts at 1.0)."""
    eq = equity_series.dropna()
    if len(eq) == 0:
        return np.nan
    return float(eq.iloc[-1] / eq.iloc[0] - 1)


def compute_alpha_table(results_dict):
    """For each ticker, compute headline metrics and the covered-call alpha.

    Returns a DataFrame with columns:
        ticker, sector, cc_total_return, stock_total_return, alpha_cc,
        cc_sharpe, stock_sharpe, cc_vol, stock_vol, cc_max_dd, stock_max_dd, win
    """
    rows = []
    for ticker, df in results_dict.items():
        if df is None or df.empty:
            continue

        cc_rets = df["cc_return"].dropna()
        stock_rets = df["stock_return"].dropna()
        cc_eq = df["cc_equity"]
        stock_eq = df["stock_equity"]

        cc_total = _total_return(cc_eq)
        stock_total = _total_return(stock_eq)
        alpha_cc = cc_total - stock_total

        rows.append({
            "ticker": ticker,
            "sector": sector_of(ticker),
            "cc_total_return": cc_total,
            "stock_total_return": stock_total,
            "alpha_cc": alpha_cc,
            "cc_sharpe": sharpe_ratio(cc_rets),
            "stock_sharpe": sharpe_ratio(stock_rets),
            "cc_vol": annualized_volatility(cc_rets),
            "stock_vol": annualized_volatility(stock_rets),
            "cc_max_dd": max_drawdown(cc_eq),
            "stock_max_dd": max_drawdown(stock_eq),
            "cc_win_rate": win_rate(cc_rets),
            "win": alpha_cc > 0,
        })

    return pd.DataFrame(rows)


def slice_by_tenure(results_dict, buckets):
    """Compute per-(ticker, period) returns given a list of (label, start, end) buckets.

    `buckets` is the output of one of the slicers in tenure_slicers.py.

    Returns a long-format DataFrame:
        ticker, sector, period_label, period_start, period_end,
        cc_return, stock_return, alpha_cc
    """
    rows = []
    for ticker, df in results_dict.items():
        if df is None or df.empty:
            continue

        for label, start, end in buckets:
            window = df.loc[(df.index >= start) & (df.index <= end)]
            if len(window) < 2:
                continue

            cc_eq = window["cc_equity"]
            stock_eq = window["stock_equity"]
            cc_period = float(cc_eq.iloc[-1] / cc_eq.iloc[0] - 1)
            stock_period = float(stock_eq.iloc[-1] / stock_eq.iloc[0] - 1)

            rows.append({
                "ticker": ticker,
                "sector": sector_of(ticker),
                "period_label": label,
                "period_start": start,
                "period_end": end,
                "cc_return": cc_period,
                "stock_return": stock_period,
                "alpha_cc": cc_period - stock_period,
            })

    return pd.DataFrame(rows)
