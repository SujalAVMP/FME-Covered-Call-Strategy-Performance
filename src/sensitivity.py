"""
Sensitivity analysis — sweep strike offsets, expiry periods, and volatility regimes
to see how covered call performance varies with these parameters.
"""

import numpy as np
import pandas as pd

from . import config
from .black_scholes import bs_call_price
from .monte_carlo import simulate_gbm_paths, covered_call_mc_returns, stock_only_mc_returns


def _annualized_sharpe(returns, r, days):
    """Approximate annualized Sharpe ratio from a horizon-return distribution."""
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0 or days <= 0:
        return 0.0

    std = returns.std()
    if std <= 0:
        return 0.0

    horizon_rf = (1 + r) ** (days / config.TRADING_DAYS) - 1
    return ((returns.mean() - horizon_rf) / std) * np.sqrt(config.TRADING_DAYS / days)


def strike_sensitivity(S_0, sigma, r=None, mu=None, expiry_days=30, strike_offsets=None, num_paths=5000):
    """
    For a fixed expiry, sweep strike prices from ITM to OTM.
    Returns a DataFrame with premium, max profit, breakeven, and MC-based stats.
    """
    r = config.RISK_FREE_RATE if r is None else r
    mu = r if mu is None else mu
    if strike_offsets is None:
        strike_offsets = config.STRIKE_OFFSETS

    T = expiry_days / config.TRADING_DAYS
    paths = simulate_gbm_paths(S_0, mu, sigma, T, num_steps=expiry_days, num_paths=num_paths)

    rows = []
    for offset in strike_offsets:
        K = S_0 * (1 + offset)
        premium = bs_call_price(S_0, K, T, r, sigma)
        cc_rets = covered_call_mc_returns(paths, K, premium, S_0)
        stock_rets = stock_only_mc_returns(paths, S_0)

        rows.append({
            "strike_offset": offset,
            "strike": K,
            "premium": premium,
            "max_profit": (K - S_0) + premium,
            "breakeven": S_0 - premium,
            "mean_return_cc": np.mean(cc_rets),
            "std_cc": np.std(cc_rets),
            "sharpe_cc": _annualized_sharpe(cc_rets, r, expiry_days),
            "mean_return_stock": np.mean(stock_rets),
            "std_stock": np.std(stock_rets),
            "prob_cc_wins": np.mean(cc_rets > stock_rets),
        })

    return pd.DataFrame(rows)


def expiry_sensitivity(S_0, sigma, r=None, mu=None, strike_offset=0.05, expiry_range=None, num_paths=5000):
    """
    For a fixed strike offset, sweep expiry from short to long dated.
    """
    r = config.RISK_FREE_RATE if r is None else r
    mu = r if mu is None else mu
    if expiry_range is None:
        expiry_range = config.EXPIRY_DAYS_RANGE

    rows = []
    for days in expiry_range:
        T = days / config.TRADING_DAYS
        K = S_0 * (1 + strike_offset)
        premium = bs_call_price(S_0, K, T, r, sigma)

        paths = simulate_gbm_paths(S_0, mu, sigma, T, num_steps=days, num_paths=num_paths)
        cc_rets = covered_call_mc_returns(paths, K, premium, S_0)

        # annualize the premium yield
        premium_yield_annual = (premium / S_0) * (config.TRADING_DAYS / days)

        rows.append({
            "expiry_days": days,
            "premium": premium,
            "premium_pct": premium / S_0,
            "premium_yield_annual": premium_yield_annual,
            "mean_return_cc": np.mean(cc_rets),
            "std_cc": np.std(cc_rets),
            "sharpe_cc": _annualized_sharpe(cc_rets, r, days),
        })

    return pd.DataFrame(rows)


def moneyness_expiry_heatmap(S_0, sigma, r=None, mu=None, strike_offsets=None, expiry_range=None, num_paths=3000):
    """
    2D grid: rows = strike offsets, cols = expiry days.
    Cell value = Sharpe ratio of the covered call from MC simulation.
    Returns a 2D numpy array + labels.
    """
    r = config.RISK_FREE_RATE if r is None else r
    mu = r if mu is None else mu
    if strike_offsets is None:
        strike_offsets = config.STRIKE_OFFSETS
    if expiry_range is None:
        expiry_range = config.EXPIRY_DAYS_RANGE

    heatmap = np.zeros((len(strike_offsets), len(expiry_range)))

    for i, offset in enumerate(strike_offsets):
        for j, days in enumerate(expiry_range):
            T = days / config.TRADING_DAYS
            K = S_0 * (1 + offset)
            premium = bs_call_price(S_0, K, T, r, sigma)
            paths = simulate_gbm_paths(S_0, mu, sigma, T, num_steps=days, num_paths=num_paths, seed=42+i*10+j)
            cc_rets = covered_call_mc_returns(paths, K, premium, S_0)
            heatmap[i, j] = _annualized_sharpe(cc_rets, r, days)

    return heatmap, strike_offsets, expiry_range


def volatility_regime_analysis(backtest_df, vol_series):
    """
    Split the backtest into high-vol and low-vol regimes based on median vol.
    Returns separate metrics for each regime.
    """
    median_vol = vol_series.median()

    # align indices
    common = backtest_df.index.intersection(vol_series.index)
    bt = backtest_df.loc[common]
    vol = vol_series.loc[common]

    high_vol_mask = vol > median_vol
    low_vol_mask = ~high_vol_mask

    result = {}
    for label, mask in [("high_vol", high_vol_mask), ("low_vol", low_vol_mask)]:
        cc_rets = bt.loc[mask, "cc_return"].dropna()
        stock_rets = bt.loc[mask, "stock_return"].dropna()
        result[label] = {
            "n_days": int(mask.sum()),
            "cc_mean_daily": cc_rets.mean(),
            "stock_mean_daily": stock_rets.mean(),
            "cc_std_daily": cc_rets.std(),
            "stock_std_daily": stock_rets.std(),
            "cc_outperforms": cc_rets.mean() > stock_rets.mean(),
        }

    result["median_vol"] = median_vol
    return result
