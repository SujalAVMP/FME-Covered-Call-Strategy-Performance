"""Sanity checks for payoff and Black-Scholes modules."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.payoff import (covered_call_payoff, covered_call_profit,
                        stock_only_profit, breakeven_price, max_profit, max_loss)
from src.black_scholes import bs_call_price, bs_put_price, bs_greeks, covered_call_greeks


def test_covered_call_payoff():
    # if S_T < K, payoff = S_T
    assert covered_call_payoff(80, 100) == 80
    # if S_T > K, payoff = K
    assert covered_call_payoff(120, 100) == 100
    # if S_T == K, payoff = K
    assert covered_call_payoff(100, 100) == 100

def test_covered_call_payoff_vectorized():
    S_T = np.array([70, 90, 100, 110, 130])
    K = 100
    expected = np.array([70, 90, 100, 100, 100])
    np.testing.assert_array_equal(covered_call_payoff(S_T, K), expected)

def test_covered_call_profit():
    # S_0=100, K=105, C_0=3
    # if S_T=110 (above K): profit = 105 + 3 - 100 = 8
    assert abs(covered_call_profit(110, 100, 105, 3) - 8.0) < 1e-10
    # if S_T=90 (below K): profit = 90 + 3 - 100 = -7
    assert abs(covered_call_profit(90, 100, 105, 3) - (-7.0)) < 1e-10

def test_breakeven():
    assert breakeven_price(100, 5) == 95

def test_max_profit():
    # K=105, S_0=100, C_0=3 -> max profit = 5 + 3 = 8
    assert max_profit(100, 105, 3) == 8

def test_max_loss():
    assert max_loss(100, 5) == 95

def test_bs_call_put_parity():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    C = bs_call_price(S, K, T, r, sigma)
    P = bs_put_price(S, K, T, r, sigma)
    # put-call parity: C - P = S - K*exp(-rT)
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8, f"Put-call parity violated: {lhs} != {rhs}"

def test_bs_call_known_value():
    # S=100, K=100, T=1, r=0.05, sigma=0.2
    # known BS call price ~ 10.4506
    C = bs_call_price(100, 100, 1.0, 0.05, 0.2)
    assert abs(C - 10.4506) < 0.01, f"BS call price {C} not close to 10.4506"

def test_bs_greeks_delta_range():
    g = bs_greeks(100, 100, 1.0, 0.05, 0.2)
    assert 0 < g["delta"] < 1, f"Call delta should be in (0,1), got {g['delta']}"

def test_bs_zero_volatility_limit():
    call = bs_call_price(100, 100, 1.0, 0.05, 0.0)
    put = bs_put_price(100, 100, 1.0, 0.05, 0.0)
    greeks = bs_greeks(100, 100, 1.0, 0.05, 0.0)

    assert abs(call - (100 - 100 * np.exp(-0.05))) < 1e-10
    assert put == 0.0
    assert greeks["gamma"] == 0.0
    assert greeks["vega"] == 0.0

def test_covered_call_delta():
    g = covered_call_greeks(100, 100, 1.0, 0.05, 0.2)
    # covered call delta = 1 - call_delta, should be in (0, 1)
    assert 0 < g["delta"] < 1
    # should be less than 1 (reduced exposure)
    assert g["delta"] < 1

def test_covered_call_theta_positive():
    g = covered_call_greeks(100, 100, 0.1, 0.05, 0.2)
    # covered call should have positive theta (benefits from time decay)
    assert g["theta"] > 0, f"Covered call theta should be positive, got {g['theta']}"

def test_covered_call_gamma_negative():
    g = covered_call_greeks(100, 100, 0.5, 0.05, 0.2)
    assert g["gamma"] < 0, f"Covered call gamma should be negative, got {g['gamma']}"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed+failed} tests")
