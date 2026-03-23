"""Monte Carlo simulation of stock paths and covered call strategy returns."""

import numpy as np
from scipy import stats

from . import config
from .black_scholes import bs_call_price


def simulate_gbm_paths(S_0, mu, sigma, T, num_steps=None, num_paths=None, seed=None):
    """Simulate stock price paths using Geometric Brownian Motion.

    Returns array of shape (num_steps + 1, num_paths).
    """
    num_steps = num_steps or config.MC_NUM_STEPS
    num_paths = num_paths or config.MC_NUM_PATHS
    seed = seed if seed is not None else config.MC_SEED

    if T < 0:
        raise ValueError("T must be non-negative")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if num_paths <= 0:
        raise ValueError("num_paths must be positive")

    rng = np.random.default_rng(seed)
    dt = T / num_steps
    Z = rng.standard_normal((num_steps, num_paths))

    # GBM: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_increments, axis=0)
    log_paths = np.vstack([np.zeros(num_paths), log_paths])  # prepend S_0 row

    return S_0 * np.exp(log_paths)


def covered_call_mc_returns(paths, K, C_0, S_0):
    """Compute covered call returns for each simulated path.

    Terminal value = min(S_T, K) + C_0
    Return = (terminal_value - S_0) / S_0
    """
    S_T = paths[-1, :]  # terminal prices
    terminal_value = np.minimum(S_T, K) + C_0
    return (terminal_value - S_0) / S_0


def covered_call_equity_paths(paths, K, C_0, r, sigma, T):
    """Mark a covered call portfolio to market along each simulated path."""
    num_steps = paths.shape[0] - 1
    dt = T / num_steps if num_steps > 0 else 0.0
    equity_paths = np.zeros_like(paths, dtype=float)

    for step in range(paths.shape[0]):
        remaining_T = max(T - step * dt, 0.0)
        call_value = bs_call_price(paths[step, :], K, remaining_T, r, sigma)
        equity_paths[step, :] = paths[step, :] + C_0 - call_value

    return equity_paths


def stock_only_mc_returns(paths, S_0):
    """Compute stock-only returns for each simulated path."""
    S_T = paths[-1, :]
    return (S_T - S_0) / S_0


def mc_summary(returns):
    """Compute summary statistics for a return distribution."""
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "VaR_5pct": 0.0,
            "CVaR_5pct": 0.0,
            "prob_loss": 0.0,
            "prob_positive": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    std = np.std(returns)
    var_threshold = np.percentile(returns, 5)
    tail = returns[returns <= var_threshold]

    return {
        "mean": np.mean(returns),
        "median": np.median(returns),
        "std": std,
        "skewness": float(stats.skew(returns)) if std > 0 else 0.0,
        "kurtosis": float(stats.kurtosis(returns)) if std > 0 else 0.0,
        "VaR_5pct": max(float(-var_threshold), 0.0),
        "CVaR_5pct": max(float(-tail.mean()), 0.0) if tail.size else 0.0,
        "prob_loss": float(np.mean(returns < 0)),
        "prob_positive": float(np.mean(returns > 0)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
    }
