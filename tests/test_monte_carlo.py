"""Regression tests for Monte Carlo portfolio paths."""

import numpy as np

from src.monte_carlo import covered_call_equity_paths


def test_covered_call_equity_paths_match_terminal_payoff():
    paths = np.array([
        [100.0, 100.0, 100.0],
        [102.0, 98.0, 101.0],
        [110.0, 90.0, 104.0],
    ])

    K = 105.0
    premium = 4.0
    equity_paths = covered_call_equity_paths(
        paths, K, premium, r=0.05, sigma=0.20, T=2 / 252
    )

    expected_terminal_value = np.minimum(paths[-1], K) + premium
    np.testing.assert_allclose(equity_paths[-1], expected_terminal_value)
