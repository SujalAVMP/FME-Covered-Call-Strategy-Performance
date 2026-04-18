"""
Covered Call Strategy Analyzer — Streamlit Web App
ES418 Group 10

Run with:  streamlit run app/streamlit_app.py
"""

import sys
import os
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# make sure we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import config
from src.data_fetcher import get_stock_data
from src.black_scholes import bs_call_price, covered_call_greeks
from src.payoff import (covered_call_profit, stock_only_profit,
                        breakeven_price, max_profit, max_loss)
from src.monte_carlo import (simulate_gbm_paths, covered_call_mc_returns,
                             stock_only_mc_returns, covered_call_equity_paths,
                             mc_summary)
from src.backtest import run_backtest, compute_monthly_returns, prepare_volatility_series
from src.risk_metrics import summary_table
from src.sensitivity import strike_sensitivity, moneyness_expiry_heatmap
from src.visualizations import (plotly_payoff, plotly_equity_curves,
                                plotly_return_dist, plotly_mc_fan,
                                plotly_beta_vs_alpha, plotly_sector_tenure_heatmap,
                                plotly_alpha_by_regime, plotly_rolling_alpha_timeseries,
                                plotly_bs_vs_market)
from src.universe import SECTOR_UNIVERSE, BENCHMARK_TICKER, tickers_in_sectors
from src import multi_ticker_backtest, beta_analysis, tenure_slicers, market_validation


st.set_page_config(page_title="Covered Call Analyzer", layout="wide")
st.title("Covered Call Strategy Performance Analyzer")
st.caption("ES418: Financial Modelling and Engineering — Group 10")

# ─── Sidebar: User inputs ─────────────────────────────
st.sidebar.header("Parameters")

preset_ticker_labels = list(config.PRESET_TICKERS.keys())
default_preset = next(
    (
        label
        for label, symbol in config.PRESET_TICKERS.items()
        if symbol == config.DEFAULT_TICKER
    ),
    "Custom",
)
preset_options = preset_ticker_labels + ["Custom"]

selected_ticker_label = st.sidebar.selectbox(
    "Preset Stock",
    preset_options,
    index=preset_options.index(default_preset),
)

if selected_ticker_label == "Custom":
    ticker = st.sidebar.text_input("Custom Stock Ticker", value=config.DEFAULT_TICKER)
else:
    ticker = config.PRESET_TICKERS[selected_ticker_label]
    st.sidebar.caption(f"Using ticker: `{ticker}`")

start_date = st.sidebar.date_input("Backtest Start",
                                    value=datetime.date.fromisoformat(config.BACKTEST_START))
end_date = st.sidebar.date_input("Backtest End",
                                  value=datetime.date.fromisoformat(config.BACKTEST_END))

strike_type = st.sidebar.selectbox("Strike Price", ["ATM", "2% OTM", "5% OTM", "10% OTM", "5% ITM"])
strike_map = {"5% ITM": -0.05, "ATM": 0.0, "2% OTM": 0.02, "5% OTM": 0.05, "10% OTM": 0.10}
strike_offset = strike_map[strike_type]

expiry_days = st.sidebar.slider(
    "Days to Expiry (per roll)",
    7,
    90,
    config.DEFAULT_EXPIRY_DAYS,
)
mc_paths = st.sidebar.slider("Monte Carlo Paths", 1000, 20000, 5000, step=1000)

run = st.sidebar.button("Run Analysis", type="primary")

# Persist "have we run yet" across reruns triggered by other buttons (e.g. tab 5).
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "xs_run" not in st.session_state:
    st.session_state.xs_run = False
if run:
    st.session_state.has_run = True
    # New parameters → invalidate previous cross-sectional output.
    st.session_state.xs_run = False

if not st.session_state.has_run:
    st.info("Set your parameters in the sidebar and click **Run Analysis**.")
    st.stop()

# ─── Fetch data ────────────────────────────────────────
with st.spinner("Fetching stock data..."):
    try:
        df = get_stock_data(ticker, str(start_date), str(end_date))
    except Exception as e:
        st.error(f"Could not fetch data for {ticker}: {e}")
        st.stop()

prices = df["Close"]
S_0 = float(prices.iloc[-1])
hist_vol = prepare_volatility_series(prices, window=config.DEFAULT_VOL_WINDOW)
sigma = float(hist_vol.iloc[-1])
mu_real = float(np.log(prices.iloc[-1] / prices.iloc[0]) / (len(prices) / config.TRADING_DAYS))

K = S_0 * (1 + strike_offset)
T = expiry_days / config.TRADING_DAYS
C_0 = bs_call_price(S_0, K, T, config.RISK_FREE_RATE, sigma)

# ─── Tabs ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Payoff Analysis", "Historical Backtest", "Monte Carlo",
    "Sensitivity & Greeks", "Cross-Sectional Analysis"
])

# ═══ Tab 1: Payoff ═══════════════════════════════════
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_payoff = plotly_payoff(S_0, K, C_0)
        st.plotly_chart(fig_payoff, use_container_width=True)
    with col2:
        st.subheader("Key Figures")
        st.metric("Current Price", f"{S_0:,.2f}")
        st.metric("Strike Price", f"{K:,.2f}")
        st.metric("Call Premium", f"{C_0:,.2f}")
        st.metric("Breakeven", f"{breakeven_price(S_0, C_0):,.2f}")
        st.metric("Max Profit", f"{max_profit(S_0, K, C_0):,.2f}")
        st.metric("Max Loss (stock→0)", f"{max_loss(S_0, C_0):,.2f}")

# ═══ Tab 2: Historical Backtest ══════════════════════
with tab2:
    with st.spinner("Running backtest..."):
        bt_df, roll_dates = run_backtest(ticker, str(start_date), str(end_date),
                                          strike_offset=strike_offset,
                                          expiry_days=expiry_days)

    fig_eq = plotly_equity_curves(bt_df)
    st.plotly_chart(fig_eq, use_container_width=True)

    # metrics table
    cc_rets = bt_df["cc_return"].dropna()
    stock_rets = bt_df["stock_return"].dropna()
    metrics = summary_table(cc_rets, stock_rets, bt_df["cc_equity"], bt_df["stock_equity"])
    st.subheader("Performance Metrics")
    st.dataframe(metrics, use_container_width=True)

    # monthly return distributions
    cc_monthly = compute_monthly_returns(bt_df, "cc_equity")
    stock_monthly = compute_monthly_returns(bt_df, "stock_equity")
    if len(cc_monthly) > 2:
        fig_dist = plotly_return_dist(
            cc_monthly.values,
            stock_monthly.values,
            title="Distribution of Monthly Returns",
            x_label="Monthly Return",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ═══ Tab 3: Monte Carlo ═════════════════════════════
with tab3:
    with st.spinner(f"Simulating {mc_paths} paths..."):
        paths = simulate_gbm_paths(S_0, mu_real, sigma, T,
                                    num_steps=expiry_days, num_paths=mc_paths)

        cc_mc_rets = covered_call_mc_returns(paths, K, C_0, S_0)
        stock_mc_rets = stock_only_mc_returns(paths, S_0)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Covered Call")
        cc_stats = mc_summary(cc_mc_rets)
        for k, v in cc_stats.items():
            st.write(f"**{k}**: {v:.4f}")

    with col2:
        st.subheader("Stock Only")
        stock_stats = mc_summary(stock_mc_rets)
        for k, v in stock_stats.items():
            st.write(f"**{k}**: {v:.4f}")

    # fan charts side by side
    col1, col2 = st.columns(2)

    # build equity paths for visualization
    stock_equity_paths = paths / S_0
    cc_equity_paths = covered_call_equity_paths(
        paths, K, C_0, config.RISK_FREE_RATE, sigma, T
    ) / S_0

    with col1:
        fig_fan_stock = plotly_mc_fan(stock_equity_paths, title="Stock-Only Equity Paths")
        st.plotly_chart(fig_fan_stock, use_container_width=True)

    with col2:
        fig_fan_cc = plotly_mc_fan(cc_equity_paths, title="Covered Call Equity Paths")
        st.plotly_chart(fig_fan_cc, use_container_width=True)

    # probability of CC outperforming
    prob_win = np.mean(cc_mc_rets > stock_mc_rets)
    st.metric("Probability Covered Call Outperforms Stock", f"{prob_win:.1%}")

# ═══ Tab 4: Sensitivity & Greeks ═════════════════════
with tab4:
    st.subheader("Strike Sensitivity")
    with st.spinner("Running strike sweep..."):
        ss_df = strike_sensitivity(S_0, sigma, mu=mu_real, expiry_days=expiry_days, num_paths=3000)
    st.dataframe(ss_df.round(4), use_container_width=True)

    st.subheader("Moneyness × Expiry Heatmap (Sharpe Ratio)")
    with st.spinner("Computing heatmap (this takes a moment)..."):
        hmap, offsets, expiries = moneyness_expiry_heatmap(S_0, sigma, mu=mu_real, num_paths=2000)

    import plotly.express as px
    hmap_df = pd.DataFrame(hmap,
                            index=[f"{o:+.0%}" for o in offsets],
                            columns=[str(d) for d in expiries])
    fig_hmap = px.imshow(hmap_df, text_auto=".2f", color_continuous_scale="RdYlGn",
                          labels=dict(x="Days to Expiry", y="Strike Offset", color="Sharpe"))
    fig_hmap.update_layout(title="Sharpe Ratio Heatmap")
    st.plotly_chart(fig_hmap, use_container_width=True)

    # Greeks
    st.subheader("Position Greeks vs Stock Price")
    S_range = np.linspace(0.7 * K, 1.3 * K, 100)
    greeks_data = [covered_call_greeks(s, K, T, config.RISK_FREE_RATE, sigma) for s in S_range]

    col1, col2, col3 = st.columns(3)
    with col1:
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(x=S_range, y=[g["delta"] for g in greeks_data], mode='lines'))
        fig_d.update_layout(title="Delta", xaxis_title="Stock Price", template="plotly_white")
        st.plotly_chart(fig_d, use_container_width=True)
    with col2:
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=S_range, y=[g["gamma"] for g in greeks_data],
                                    mode='lines', line=dict(color='orange')))
        fig_g.update_layout(title="Gamma", xaxis_title="Stock Price", template="plotly_white")
        st.plotly_chart(fig_g, use_container_width=True)
    with col3:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=S_range, y=[g["theta"] for g in greeks_data],
                                    mode='lines', line=dict(color='green')))
        fig_t.update_layout(title="Theta", xaxis_title="Stock Price", template="plotly_white")
        st.plotly_chart(fig_t, use_container_width=True)


# ═══ Tab 5: Cross-Sectional Analysis ══════════════════
with tab5:
    st.header("Cross-Sectional Analysis")
    st.caption(
        "Run the rolling covered-call backtest across multiple stocks and sectors, "
        "compute beta vs Nifty 50, and find the regimes where the strategy actually beats stock-only."
    )

    all_sectors = list(SECTOR_UNIVERSE.keys())
    selected_sectors = st.multiselect(
        "Sectors to include",
        options=all_sectors,
        default=all_sectors,
        help="Each sector contributes 2-3 large-cap NSE stocks.",
    )

    slicer_choice = st.radio(
        "Tenure slicing",
        options=["Calendar Year", "Market Regime", "Rolling 6mo"],
        horizontal=True,
        help=(
            "Calendar Year: one bucket per year. "
            "Market Regime: classifies each year as Bull / Sideways / Bear from Nifty's annual return. "
            "Rolling 6mo: 6-month windows stepped every 3 months."
        ),
    )

    run_xs_btn = st.button("Run cross-sectional analysis", type="primary", key="run_xs")
    if run_xs_btn:
        st.session_state.xs_run = True

    if not st.session_state.xs_run:
        st.info(
            "Click **Run cross-sectional analysis** to download data for the selected universe "
            "and run the backtest. This is slower than the single-ticker analysis above (~30s)."
        )
    else:
        tickers = tickers_in_sectors(selected_sectors)
        if not tickers:
            st.error("Please select at least one sector.")
            st.stop()

        @st.cache_data(show_spinner=False)
        def _cached_universe_backtest(tickers_tuple, start, end, strike_offset, expiry_days):
            return multi_ticker_backtest.run_universe_backtest(
                list(tickers_tuple), start=start, end=end,
                strike_offset=strike_offset, expiry_days=expiry_days,
            )

        @st.cache_data(show_spinner=False)
        def _cached_beta_table(tickers_tuple, start, end):
            return beta_analysis.compute_beta_table(list(tickers_tuple), start=start, end=end)

        @st.cache_data(show_spinner=False)
        def _cached_nifty(start, end):
            from src.data_fetcher import get_stock_data
            return get_stock_data(BENCHMARK_TICKER, start, end)["Close"]

        with st.spinner(f"Running backtest on {len(tickers)} tickers..."):
            results = _cached_universe_backtest(
                tuple(tickers), str(start_date), str(end_date),
                strike_offset, expiry_days,
            )
        if not results:
            st.error("All tickers failed. Check your network connection.")
            st.stop()
        st.success(f"Completed {len(results)}/{len(tickers)} tickers.")

        with st.spinner("Computing beta vs Nifty 50..."):
            beta_table = _cached_beta_table(tuple(tickers), str(start_date), str(end_date))

        alpha_table = multi_ticker_backtest.compute_alpha_table(results)

        # Combine alpha + beta tables
        combined = alpha_table.merge(
            beta_table[["ticker", "beta", "r_squared"]],
            on="ticker", how="left"
        )

        # ── Display tables ──
        st.subheader("Per-Ticker Summary")
        display_cols = ["ticker", "sector", "beta", "cc_total_return",
                        "stock_total_return", "alpha_cc", "cc_sharpe", "stock_sharpe", "win"]
        display_df = combined[display_cols].copy()
        for c in ["cc_total_return", "stock_total_return", "alpha_cc"]:
            display_df[c] = display_df[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
        for c in ["beta", "cc_sharpe", "stock_sharpe"]:
            display_df[c] = display_df[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        st.dataframe(display_df, use_container_width=True)

        n_wins = int(combined["win"].sum())
        st.metric("Stocks where Covered Call beat Stock-Only",
                  f"{n_wins} / {len(combined)}")

        # ── Beta vs Alpha scatter ──
        st.subheader("Beta vs Covered Call Alpha")
        st.caption(
            "The headline chart for the professor's question. A negative slope means "
            "low-beta stocks favor the covered call (premium income dominates capped upside)."
        )
        fig_ba = plotly_beta_vs_alpha(combined)
        st.plotly_chart(fig_ba, use_container_width=True)

        # ── Tenure analysis ──
        st.subheader("Tenure Analysis")

        if slicer_choice == "Calendar Year":
            buckets = tenure_slicers.slice_by_calendar_year(next(iter(results.values())))
        elif slicer_choice == "Market Regime":
            with st.spinner("Downloading Nifty for regime classification..."):
                nifty_close = _cached_nifty(str(start_date), str(end_date))
            buckets = tenure_slicers.slice_by_market_regime(
                next(iter(results.values())), nifty_close
            )
        else:  # Rolling 6mo
            buckets = tenure_slicers.slice_by_rolling_window(
                next(iter(results.values())), window_months=6, step_months=3
            )

        slice_df = multi_ticker_backtest.slice_by_tenure(results, buckets)

        if slice_df.empty:
            st.warning("No data in the selected windows.")
        else:
            fig_hm = plotly_sector_tenure_heatmap(
                slice_df, title=f"Sector × {slicer_choice}: Mean Covered Call Alpha"
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            if slicer_choice == "Market Regime":
                fig_reg = plotly_alpha_by_regime(slice_df)
                st.plotly_chart(fig_reg, use_container_width=True)
            elif slicer_choice == "Rolling 6mo":
                fig_roll = plotly_rolling_alpha_timeseries(slice_df)
                st.plotly_chart(fig_roll, use_container_width=True)

        # ── BS validation ──
        with st.expander("Black-Scholes pricing validation (current option chain)", expanded=False):
            st.caption(
                "Compares BS theoretical prices against a live option chain for the selected "
                "ticker when the upstream data provider exposes one. This is only a snapshot "
                "diagnostic and does not validate the historical backtest."
            )
            try:
                with st.spinner(f"Fetching current option chain for {ticker}..."):
                    val_df = market_validation.validate_bs_against_market(ticker)
                fig_val = plotly_bs_vs_market(val_df)
                st.plotly_chart(fig_val, use_container_width=True)
                mean_pct_err = val_df["pct_error"].mean()
                st.metric("Mean absolute pricing error",
                          f"{mean_pct_err:.1%}" if pd.notna(mean_pct_err) else "—")
                st.dataframe(val_df.round(2), use_container_width=True)
            except Exception as exc:
                st.warning(f"Validation skipped: {exc}")
