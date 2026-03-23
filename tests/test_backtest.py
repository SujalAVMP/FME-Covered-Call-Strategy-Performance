"""Regression tests for the rolling covered call backtest."""

import numpy as np
import pandas as pd
import pytest

from src import backtest


def _close_frame(prices):
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"Close": prices}, index=dates)


def test_backtest_marks_short_call_to_market(monkeypatch):
    price_frame = _close_frame([100.0, 100.0, 100.0, 100.0])
    sigma_series = pd.Series(0.20, index=price_frame.index)

    monkeypatch.setattr(backtest, "get_stock_data", lambda *args, **kwargs: price_frame)
    monkeypatch.setattr(
        backtest,
        "compute_historical_volatility",
        lambda prices, window=None: sigma_series,
    )

    results, roll_dates = backtest.run_backtest(expiry_days=2, strike_offset=0.05)

    assert roll_dates[0] == price_frame.index[0]
    assert results["stock_equity"].iloc[0] == pytest.approx(1.0)
    assert results["cc_equity"].iloc[0] == pytest.approx(1.0)
    assert results["cc_equity"].iloc[1] > results["stock_equity"].iloc[1]


def test_backtest_opens_position_on_day_zero_with_initial_nan_vol(monkeypatch):
    price_frame = _close_frame([100.0, 101.0, 102.0, 103.0, 104.0])
    sigma_series = pd.Series([np.nan, np.nan, 0.22, 0.22, 0.22], index=price_frame.index)

    monkeypatch.setattr(backtest, "get_stock_data", lambda *args, **kwargs: price_frame)
    monkeypatch.setattr(
        backtest,
        "compute_historical_volatility",
        lambda prices, window=None: sigma_series,
    )

    results, roll_dates = backtest.run_backtest(expiry_days=3, strike_offset=0.05)

    assert roll_dates[0] == price_frame.index[0]
    assert results["cc_equity"].iloc[0] == pytest.approx(1.0)
    assert results["cc_equity"].notna().all()


def test_prepare_volatility_series_falls_back_when_window_is_too_short(monkeypatch):
    price_frame = _close_frame([100.0, 101.0, 99.0, 102.0, 103.0])
    sigma_series = pd.Series(np.nan, index=price_frame.index)

    monkeypatch.setattr(
        backtest,
        "compute_historical_volatility",
        lambda prices, window=None: sigma_series,
    )

    prepared = backtest.prepare_volatility_series(price_frame["Close"], window=30)

    assert prepared.notna().all()
    assert (prepared > 0).all()
