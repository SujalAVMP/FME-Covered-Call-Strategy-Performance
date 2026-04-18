"""Shared time-to-expiry helpers for option pricing."""

import numpy as np
import pandas as pd

from . import config


def trading_days_to_year_fraction(trading_days):
    """Convert a trading-day count into a year fraction."""
    trading_days = int(trading_days)
    if trading_days < 0:
        raise ValueError("trading_days must be non-negative")
    return trading_days / config.TRADING_DAYS


def business_days_to_expiry(current_date, expiry_date):
    """Approximate remaining business days between two dates.

    We clamp the result to at least 1 so same-day expiry snapshots still
    produce a finite time value rather than a hard expiry.
    """
    start = pd.Timestamp(current_date).normalize()
    end = pd.Timestamp(expiry_date).normalize()
    if end < start:
        raise ValueError("expiry_date must not be earlier than current_date")
    return max(int(np.busday_count(start.date(), end.date())), 1)
