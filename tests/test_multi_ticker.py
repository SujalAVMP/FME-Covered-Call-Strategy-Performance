"""Tests for src/multi_ticker_backtest.py and src/tenure_slicers.py."""

import numpy as np
import pandas as pd
import pytest

from src import multi_ticker_backtest, tenure_slicers


def _make_results_dict():
    """Build a synthetic {ticker: backtest_df} dict for two tickers."""
    dates = pd.date_range("2021-01-01", "2023-12-31", freq="B")
    n = len(dates)

    # Reliance: stock grows linearly, CC slightly less but smoother
    stock_eq_1 = np.linspace(1.0, 1.5, n)
    cc_eq_1 = np.linspace(1.0, 1.4, n)
    df1 = pd.DataFrame({
        "price": stock_eq_1 * 100,
        "stock_equity": stock_eq_1,
        "cc_equity": cc_eq_1,
        "stock_return": pd.Series(stock_eq_1).pct_change().values,
        "cc_return": pd.Series(cc_eq_1).pct_change().values,
    }, index=dates)

    # TCS: stock flat, CC wins thanks to premium income
    stock_eq_2 = np.ones(n)
    cc_eq_2 = np.linspace(1.0, 1.15, n)
    df2 = pd.DataFrame({
        "price": stock_eq_2 * 100,
        "stock_equity": stock_eq_2,
        "cc_equity": cc_eq_2,
        "stock_return": pd.Series(stock_eq_2).pct_change().values,
        "cc_return": pd.Series(cc_eq_2).pct_change().values,
    }, index=dates)

    return {"RELIANCE.NS": df1, "TCS.NS": df2}


def test_compute_alpha_table_columns_and_values():
    results = _make_results_dict()
    df = multi_ticker_backtest.compute_alpha_table(results)

    expected_cols = {"ticker", "sector", "cc_total_return", "stock_total_return",
                     "alpha_cc", "cc_sharpe", "stock_sharpe", "win"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 2

    # Reliance: CC underperforms (alpha < 0)
    rel = df[df["ticker"] == "RELIANCE.NS"].iloc[0]
    assert rel["alpha_cc"] < 0
    assert rel["win"] is False or rel["win"] == False  # noqa: E712

    # TCS: flat stock + premium → CC wins
    tcs = df[df["ticker"] == "TCS.NS"].iloc[0]
    assert tcs["alpha_cc"] > 0
    assert tcs["win"] is True or tcs["win"] == True  # noqa: E712


def test_slice_by_calendar_year_buckets():
    df = list(_make_results_dict().values())[0]
    buckets = tenure_slicers.slice_by_calendar_year(df)
    labels = [b[0] for b in buckets]
    assert labels == ["2021", "2022", "2023"]


def test_slice_by_market_regime_classification():
    # Build a synthetic Nifty: 2021 +20% (Bull), 2022 -10% (Bear), 2023 +5% (Sideways)
    dates_2021 = pd.date_range("2021-01-01", "2021-12-31", freq="B")
    dates_2022 = pd.date_range("2022-01-01", "2022-12-31", freq="B")
    dates_2023 = pd.date_range("2023-01-01", "2023-12-31", freq="B")

    nifty_2021 = np.linspace(100, 120, len(dates_2021))
    nifty_2022 = np.linspace(120, 108, len(dates_2022))
    nifty_2023 = np.linspace(108, 113.4, len(dates_2023))

    nifty = pd.Series(
        np.concatenate([nifty_2021, nifty_2022, nifty_2023]),
        index=dates_2021.append(dates_2022).append(dates_2023),
    )

    df = list(_make_results_dict().values())[0]
    buckets = tenure_slicers.slice_by_market_regime(df, nifty)

    assert any("Bull" in label for label, _, _ in buckets)
    assert any("Bear" in label for label, _, _ in buckets)
    assert any("Sideways" in label for label, _, _ in buckets)


def test_slice_by_tenure_alpha_calculation():
    results = _make_results_dict()
    any_df = next(iter(results.values()))
    buckets = tenure_slicers.slice_by_calendar_year(any_df)
    sliced = multi_ticker_backtest.slice_by_tenure(results, buckets)

    assert not sliced.empty
    assert {"ticker", "sector", "period_label", "cc_return",
            "stock_return", "alpha_cc"}.issubset(sliced.columns)
    # Each ticker should appear in each year bucket
    assert sliced["ticker"].nunique() == 2
    assert sliced["period_label"].nunique() == 3


def test_universe_backtest_handles_failed_ticker(monkeypatch):
    """If one ticker raises, the others should still complete."""
    def fake_run_backtest(ticker=None, **kwargs):
        if ticker == "BROKEN.NS":
            raise ValueError("simulated download failure")
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        return pd.DataFrame({
            "price": np.ones(10) * 100,
            "stock_equity": np.ones(10),
            "cc_equity": np.linspace(1.0, 1.05, 10),
            "stock_return": [np.nan] + [0.0] * 9,
            "cc_return": [np.nan] + [0.005] * 9,
        }, index=dates), []

    monkeypatch.setattr(multi_ticker_backtest, "run_backtest", fake_run_backtest)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    results = multi_ticker_backtest.run_universe_backtest(
        ["WORKING.NS", "BROKEN.NS", "ALSO_WORKING.NS"]
    )
    assert "WORKING.NS" in results
    assert "ALSO_WORKING.NS" in results
    assert "BROKEN.NS" not in results
