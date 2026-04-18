"""Tests for live option-chain validation helpers."""

import pandas as pd
import pytest

from src import market_validation
from src.option_time import business_days_to_expiry


def test_validate_bs_against_market_uses_business_day_year_fraction(monkeypatch):
    today = pd.Timestamp("2025-01-02")
    expiry = "2025-01-31"

    calls = pd.DataFrame({
        "strike": [95.0, 100.0, 105.0],
        "bid": [8.0, 5.0, 2.5],
        "ask": [8.4, 5.4, 2.9],
        "lastPrice": [8.2, 5.2, 2.7],
    })
    prices = pd.DataFrame(
        {"Close": [99.0, 100.0]},
        index=pd.to_datetime(["2024-12-31", "2025-01-02"]),
    )
    hist_vol = pd.Series([0.2, 0.2], index=prices.index)

    monkeypatch.setattr(market_validation, "get_available_expiries", lambda ticker=None: [expiry])
    monkeypatch.setattr(market_validation, "get_option_chain", lambda ticker, expiry_date: (calls, None))
    monkeypatch.setattr(market_validation, "get_stock_data", lambda *args, **kwargs: prices)
    monkeypatch.setattr(
        market_validation,
        "compute_historical_volatility",
        lambda prices, window=None: hist_vol,
    )

    result = market_validation.validate_bs_against_market(
        ticker="RELIANCE.NS",
        today=today,
    )

    expected_days = business_days_to_expiry(today, expiry)
    assert result["days_to_expiry"].iloc[0] == expected_days
    assert result.attrs["T"] == pytest.approx(expected_days / 252)
