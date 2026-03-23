"""Risk-adjusted performance metrics for strategy comparison."""

import numpy as np
import pandas as pd

from . import config


def annualized_return(returns_series):
    """Geometric annualized return from daily returns."""
    total = (1 + returns_series).prod()
    n_years = len(returns_series) / config.TRADING_DAYS
    if n_years <= 0 or total <= 0:
        return 0.0
    return total ** (1 / n_years) - 1


def annualized_volatility(returns_series):
    """Annualized volatility from daily returns."""
    return returns_series.std() * np.sqrt(config.TRADING_DAYS)


def sharpe_ratio(returns_series, rf=None):
    """Annualized Sharpe ratio."""
    rf = rf if rf is not None else config.RISK_FREE_RATE
    ann_ret = annualized_return(returns_series)
    ann_vol = annualized_volatility(returns_series)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - rf) / ann_vol


def sortino_ratio(returns_series, rf=None):
    """Annualized Sortino ratio (uses downside deviation only)."""
    rf = rf if rf is not None else config.RISK_FREE_RATE
    ann_ret = annualized_return(returns_series)
    daily_rf = (1 + rf) ** (1 / config.TRADING_DAYS) - 1
    downside = returns_series[returns_series < daily_rf] - daily_rf
    downside_std = np.sqrt(np.mean(downside**2)) * np.sqrt(config.TRADING_DAYS)
    if downside_std == 0:
        return 0.0
    return (ann_ret - rf) / downside_std


def max_drawdown(equity_curve):
    """Maximum drawdown from an equity curve."""
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


def calmar_ratio(returns_series, equity_curve):
    """Calmar ratio = annualized return / |max drawdown|."""
    ann_ret = annualized_return(returns_series)
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return ann_ret / mdd


def value_at_risk(returns_series, confidence=0.95):
    """Historical Value-at-Risk at given confidence level."""
    returns = returns_series.dropna()
    if returns.empty:
        return 0.0
    var = -np.percentile(returns, (1 - confidence) * 100)
    return max(float(var), 0.0)


def conditional_var(returns_series, confidence=0.95):
    """Conditional VaR (Expected Shortfall) at given confidence level."""
    returns = returns_series.dropna()
    if returns.empty:
        return 0.0
    threshold = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return 0.0
    return max(float(-tail.mean()), 0.0)


def win_rate(returns_series):
    """Fraction of positive return periods."""
    return (returns_series > 0).mean()


def summary_table(cc_returns, stock_returns, cc_equity, stock_equity, rf=None):
    """Side-by-side comparison table of all risk metrics."""
    rf = rf if rf is not None else config.RISK_FREE_RATE

    def _metrics(rets, eq, name):
        return {
            "Strategy": name,
            "Annualized Return": f"{annualized_return(rets):.2%}",
            "Annualized Volatility": f"{annualized_volatility(rets):.2%}",
            "Sharpe Ratio": f"{sharpe_ratio(rets, rf):.3f}",
            "Sortino Ratio": f"{sortino_ratio(rets, rf):.3f}",
            "Max Drawdown": f"{max_drawdown(eq):.2%}",
            "Calmar Ratio": f"{calmar_ratio(rets, eq):.3f}",
            "VaR (95%)": f"{value_at_risk(rets):.4f}",
            "CVaR (95%)": f"{conditional_var(rets):.4f}",
            "Win Rate": f"{win_rate(rets):.2%}",
        }

    cc_row = _metrics(cc_returns, cc_equity, "Covered Call")
    stock_row = _metrics(stock_returns, stock_equity, "Stock Only")
    return pd.DataFrame([cc_row, stock_row]).set_index("Strategy")
