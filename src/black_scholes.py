"""Black-Scholes pricing, Greeks, and implied volatility."""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes European call price."""
    if T <= 0:
        return np.maximum(S - K, 0.0)
    if sigma <= 0:
        return np.maximum(S - K * np.exp(-r * T), 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes European put price via put-call parity."""
    if T <= 0:
        return np.maximum(K - S, 0.0)
    if sigma <= 0:
        return np.maximum(K * np.exp(-r * T) - S, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma):
    """Compute Greeks for a long call option."""
    if T <= 0:
        delta = 1.0 if S > K else 0.0
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    if sigma <= 0:
        exercise_boundary = K * np.exp(-r * T)
        delta = 1.0 if S > exercise_boundary else 0.0
        theta = -r * K * np.exp(-r * T) if S > exercise_boundary else 0.0
        return {"delta": delta, "gamma": 0.0, "theta": theta, "vega": 0.0}
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def covered_call_greeks(S, K, T, r, sigma):
    """Greeks for covered call = long stock + short call."""
    call_greeks = bs_greeks(S, K, T, r, sigma)
    return {
        "delta": 1.0 - call_greeks["delta"],      # Reduced directional exposure
        "gamma": -call_greeks["gamma"],             # Negative gamma (hurt by large moves)
        "theta": -call_greeks["theta"],             # Positive theta (time decay benefits)
        "vega": -call_greeks["vega"],               # Negative vega (hurt by vol increase)
    }


def implied_volatility(market_price, S, K, T, r, option_type="call"):
    """Compute implied volatility using Brent's method."""
    if T <= 0:
        return 0.0
    price_func = bs_call_price if option_type == "call" else bs_put_price

    def objective(sigma):
        return price_func(S, K, T, r, sigma) - market_price

    try:
        return brentq(objective, 1e-6, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan
