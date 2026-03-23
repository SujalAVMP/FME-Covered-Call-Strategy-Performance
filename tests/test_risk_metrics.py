"""Regression tests for risk metric edge cases."""

import pandas as pd

from src.risk_metrics import conditional_var, value_at_risk
from src.sensitivity import strike_sensitivity


def test_value_at_risk_is_not_negative_when_all_returns_are_positive():
    returns = pd.Series([0.01, 0.02, 0.03, 0.04])
    assert value_at_risk(returns, confidence=0.95) == 0.0
    assert conditional_var(returns, confidence=0.95) == 0.0


def test_strike_sensitivity_accepts_zero_rate_without_overwriting_it():
    result = strike_sensitivity(
        100,
        0.2,
        r=0.0,
        mu=0.0,
        expiry_days=30,
        strike_offsets=[0.0],
        num_paths=200,
    )

    premium = float(result.loc[0, "premium"])
    assert premium < 3.0
