"""Covered call and stock-only payoff/profit functions (vectorized)."""

import numpy as np


def covered_call_payoff(S_T, K):
    """Payoff of covered call at expiry: min(S_T, K)."""
    return np.minimum(S_T, K)


def covered_call_profit(S_T, S_0, K, C_0):
    """Profit of covered call: min(S_T, K) + C_0 - S_0."""
    return np.minimum(S_T, K) + C_0 - S_0


def stock_only_payoff(S_T):
    """Payoff of holding stock: S_T."""
    return np.asarray(S_T, dtype=float)


def stock_only_profit(S_T, S_0):
    """Profit of holding stock: S_T - S_0."""
    return np.asarray(S_T, dtype=float) - S_0


def short_call_payoff(S_T, K):
    """Payoff from the short call leg: -max(S_T - K, 0)."""
    return -np.maximum(S_T - K, 0)


def breakeven_price(S_0, C_0):
    """Breakeven price for covered call = S_0 - C_0."""
    return S_0 - C_0


def max_profit(S_0, K, C_0):
    """Maximum profit for covered call = (K - S_0) + C_0."""
    return (K - S_0) + C_0


def max_loss(S_0, C_0):
    """Maximum loss for covered call (stock goes to 0) = S_0 - C_0."""
    return S_0 - C_0
