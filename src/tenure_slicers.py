"""Time-period slicers for cross-sectional tenure analysis.

Each slicer takes a backtest DataFrame (DatetimeIndex) and returns a list of
(label, start_date, end_date) tuples that downstream code can group on.
"""

import numpy as np
import pandas as pd

# Regime classification thresholds (annual Nifty return)
REGIME_BULL_THRESHOLD = 0.12   # > +12% → Bull
REGIME_BEAR_THRESHOLD = -0.05  # < -5% → Bear
# Otherwise → Sideways


def slice_by_calendar_year(df):
    """One bucket per calendar year present in the index."""
    if df.empty:
        return []
    years = sorted(set(df.index.year))
    buckets = []
    for year in years:
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31)
        buckets.append((str(year), start, end))
    return buckets


def classify_regime(annual_return):
    """Classify a single annual return into Bull / Sideways / Bear."""
    if annual_return > REGIME_BULL_THRESHOLD:
        return "Bull"
    if annual_return < REGIME_BEAR_THRESHOLD:
        return "Bear"
    return "Sideways"


def slice_by_market_regime(df, nifty_prices):
    """Bucket each calendar year by Nifty's annual return regime.

    Returns labels of the form "2022 (Bear)" so the user can see both the year
    and the regime in plots.
    """
    if df.empty or nifty_prices is None or len(nifty_prices) == 0:
        return []

    nifty = pd.Series(nifty_prices).dropna()

    years = sorted(set(df.index.year))
    buckets = []
    for year in years:
        start = pd.Timestamp(year=year, month=1, day=1)
        end = pd.Timestamp(year=year, month=12, day=31)
        nifty_window = nifty.loc[(nifty.index >= start) & (nifty.index <= end)]
        if len(nifty_window) < 2:
            continue
        annual_return = float(nifty_window.iloc[-1] / nifty_window.iloc[0] - 1)
        regime = classify_regime(annual_return)
        label = f"{year} ({regime})"
        buckets.append((label, start, end))
    return buckets


def slice_by_rolling_window(df, window_months=6, step_months=3):
    """Rolling sub-periods. Default: 6-month windows stepped every 3 months."""
    if df.empty:
        return []

    first = df.index.min()
    last = df.index.max()

    buckets = []
    start = pd.Timestamp(first)
    while start <= last:
        end = start + pd.DateOffset(months=window_months) - pd.Timedelta(days=1)
        if end > last:
            end = pd.Timestamp(last)
        label = f"{start.strftime('%Y-%m')} → {end.strftime('%Y-%m')}"
        buckets.append((label, start, end))
        if end >= last:
            break
        start = start + pd.DateOffset(months=step_months)
    return buckets
