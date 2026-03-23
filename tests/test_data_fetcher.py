"""Regression tests for data download parameter handling."""

import pandas as pd

from src import data_fetcher


def test_get_stock_data_treats_end_date_as_inclusive(monkeypatch):
    captured = {}

    def fake_download(ticker, start=None, end=None, auto_adjust=None, progress=None):
        captured["ticker"] = ticker
        captured["start"] = start
        captured["end"] = end
        captured["auto_adjust"] = auto_adjust
        captured["progress"] = progress
        return pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2025-01-31"]),
        )

    monkeypatch.setattr(data_fetcher.yf, "download", fake_download)

    data_fetcher.get_stock_data("RELIANCE.NS", "2025-01-01", "2025-01-31")

    assert captured["ticker"] == "RELIANCE.NS"
    assert captured["start"] == "2025-01-01"
    assert captured["end"] == "2025-02-01"
    assert captured["auto_adjust"] is True
    assert captured["progress"] is False
