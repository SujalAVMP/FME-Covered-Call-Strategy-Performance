"""Compare Black-Scholes prices to a live option chain when supported.

This is only a snapshot diagnostic. It does not validate the historical
backtest, and it depends on the upstream data provider exposing a live option
chain for the requested ticker.
"""

import numpy as np
import pandas as pd

from . import config
from .black_scholes import bs_call_price
from .data_fetcher import (compute_historical_volatility, get_available_expiries,
                            get_option_chain, get_stock_data)
from .option_time import business_days_to_expiry, trading_days_to_year_fraction


def validate_bs_against_market(ticker=None, vol_window=None, moneyness_band=0.10, today=None):
    """Compare BS prices to market prices for the nearest supported expiry.

    Returns a DataFrame with columns:
        strike, market_price, bs_price, abs_error, pct_error, days_to_expiry
    Plus a `meta` attribute holding spot, sigma, T, expiry_date.
    """
    ticker = ticker or config.DEFAULT_TICKER
    vol_window = vol_window or config.DEFAULT_VOL_WINDOW

    expiries = get_available_expiries(ticker)
    if not expiries:
        raise RuntimeError(
            f"No option expiries available for {ticker}; "
            "the current data provider may not expose live option chains for this symbol."
        )

    # Pick the first expiry that's at least 7 days out
    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    chosen_expiry = None
    for exp in expiries:
        exp_ts = pd.Timestamp(exp)
        if (exp_ts - today).days >= 7:
            chosen_expiry = exp
            break
    if chosen_expiry is None:
        chosen_expiry = expiries[0]

    calls, _ = get_option_chain(ticker, chosen_expiry)
    if calls is None or calls.empty:
        raise RuntimeError(
            f"Empty option chain for {ticker} {chosen_expiry}; "
            "live market validation is unavailable for this symbol right now."
        )

    # Spot price + realized vol from recent history
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    stock_df = get_stock_data(ticker, start_date, end_date)
    prices = stock_df["Close"].dropna()
    spot = float(prices.iloc[-1])

    hist_vol = compute_historical_volatility(prices, window=vol_window)
    sigma = float(hist_vol.dropna().iloc[-1])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.20  # fallback

    # Use the same trading-day year fraction convention as the backtest.
    days_to_expiry = business_days_to_expiry(today, chosen_expiry)
    T = trading_days_to_year_fraction(days_to_expiry)

    # Filter to within ±moneyness_band of spot
    lower = spot * (1 - moneyness_band)
    upper = spot * (1 + moneyness_band)
    filtered = calls[(calls["strike"] >= lower) & (calls["strike"] <= upper)].copy()
    if filtered.empty:
        raise RuntimeError(
            f"No option strikes within ±{moneyness_band:.0%} of spot {spot:.2f}"
        )

    # Use mid price if bid/ask available, else lastPrice
    if "bid" in filtered.columns and "ask" in filtered.columns:
        mid = (filtered["bid"] + filtered["ask"]) / 2
        market_prices = mid.where(mid > 0, filtered["lastPrice"])
    else:
        market_prices = filtered["lastPrice"]

    bs_prices = filtered["strike"].apply(
        lambda K: bs_call_price(spot, K, T, config.RISK_FREE_RATE, sigma)
    )

    out = pd.DataFrame({
        "strike": filtered["strike"].values,
        "market_price": market_prices.values,
        "bs_price": bs_prices.values,
    })
    out["abs_error"] = (out["bs_price"] - out["market_price"]).abs()
    out["pct_error"] = out["abs_error"] / out["market_price"].replace(0, np.nan)
    out["days_to_expiry"] = days_to_expiry

    # Drop rows where market price is missing/zero (illiquid strikes)
    out = out[out["market_price"] > 0].reset_index(drop=True)

    out.attrs["spot"] = spot
    out.attrs["sigma"] = sigma
    out.attrs["T"] = T
    out.attrs["expiry"] = str(chosen_expiry)
    out.attrs["ticker"] = ticker
    return out
