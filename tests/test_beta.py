"""Tests for src/beta_analysis.py."""

import numpy as np
import pandas as pd

from src import beta_analysis


def _date_index(n):
    return pd.date_range("2024-01-01", periods=n, freq="B")


def test_compute_beta_recovers_known_value():
    """If stock_return = 1.5 * benchmark_return + noise, beta should be ~1.5."""
    rng = np.random.default_rng(42)
    n = 500
    bench_returns = rng.normal(0, 0.01, size=n)
    stock_returns = 1.5 * bench_returns + rng.normal(0, 0.001, size=n)

    bench_prices = pd.Series(np.exp(np.cumsum(bench_returns)) * 100, index=_date_index(n))
    stock_prices = pd.Series(np.exp(np.cumsum(stock_returns)) * 100, index=_date_index(n))

    result = beta_analysis.compute_beta(stock_prices, bench_prices)
    assert abs(result["beta"] - 1.5) < 0.05
    assert result["r_squared"] > 0.95
    assert result["n_obs"] > 400


def test_compute_beta_handles_zero_variance():
    n = 100
    bench_prices = pd.Series([100.0] * n, index=_date_index(n))
    stock_prices = pd.Series(np.linspace(100, 110, n), index=_date_index(n))
    result = beta_analysis.compute_beta(stock_prices, bench_prices)
    assert np.isnan(result["beta"])


def test_compute_beta_handles_short_series():
    bench_prices = pd.Series([100.0, 101.0], index=_date_index(2))
    stock_prices = pd.Series([100.0, 101.0], index=_date_index(2))
    result = beta_analysis.compute_beta(stock_prices, bench_prices)
    assert np.isnan(result["beta"])
    assert result["n_obs"] == 2
