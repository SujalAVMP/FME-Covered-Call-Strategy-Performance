"""
All the charts we need for the report and the website.
matplotlib for static PNGs (report), plotly for interactive (Streamlit).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .payoff import (covered_call_payoff, covered_call_profit,
                     stock_only_profit, short_call_payoff)


# ──────────────────────────────────────────────────────
#  1. Payoff Diagram (matplotlib — for report)
# ──────────────────────────────────────────────────────
def plot_payoff_diagram(S_0, K, save_path=None):
    S_range = np.linspace(0.5 * S_0, 1.5 * S_0, 500)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S_range, S_range, '--', label="Long Stock", alpha=0.7)
    ax.plot(S_range, short_call_payoff(S_range, K), '--', label="Short Call", alpha=0.7)
    ax.plot(S_range, covered_call_payoff(S_range, K), 'k-', linewidth=2, label="Covered Call")
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(K, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel("Stock Price at Expiry ($S_T$)")
    ax.set_ylabel("Payoff")
    ax.set_title("Covered Call — Payoff at Expiry")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────
#  2. Profit/Loss Diagram
# ──────────────────────────────────────────────────────
def plot_profit_diagram(S_0, K, C_0, save_path=None):
    S_range = np.linspace(0.5 * S_0, 1.5 * S_0, 500)
    cc_profit = covered_call_profit(S_range, S_0, K, C_0)
    stock_profit = stock_only_profit(S_range, S_0)

    breakeven = S_0 - C_0
    max_prof = (K - S_0) + C_0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(S_range, stock_profit, '--', label="Stock Only", alpha=0.7)
    ax.plot(S_range, cc_profit, 'k-', linewidth=2, label="Covered Call")
    ax.axhline(0, color='gray', linewidth=0.5)

    # annotate key points
    ax.axvline(breakeven, color='red', linestyle=':', alpha=0.6)
    ax.annotate(f"Breakeven\n₹{breakeven:.0f}", xy=(breakeven, 0),
                xytext=(breakeven - 0.08*S_0, max_prof*0.5),
                arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=9)
    ax.axhline(max_prof, color='green', linestyle=':', alpha=0.4)
    ax.annotate(f"Max Profit = ₹{max_prof:.1f}", xy=(K*1.1, max_prof),
                fontsize=9, color='green')

    ax.set_xlabel("Stock Price at Expiry ($S_T$)")
    ax.set_ylabel("Profit / Loss")
    ax.set_title("Covered Call — Profit & Loss at Expiry")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────
#  3. Historical Equity Curves
# ──────────────────────────────────────────────────────
def plot_equity_curves(backtest_df, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(backtest_df.index, backtest_df["stock_equity"], label="Stock Only", alpha=0.8)
    ax.plot(backtest_df.index, backtest_df["cc_equity"], label="Covered Call", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (log scale, normalized to 1)")
    ax.set_title("Equity Curves — Covered Call vs Buy-and-Hold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────
#  4. Drawdown Comparison
# ──────────────────────────────────────────────────────
def plot_drawdown(backtest_df, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))

    for col, label in [("stock_equity", "Stock Only"), ("cc_equity", "Covered Call")]:
        eq = backtest_df[col]
        peak = eq.cummax()
        dd = (eq - peak) / peak
        ax.fill_between(dd.index, dd.values, alpha=0.3, label=label)

    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.set_title("Underwater Chart — Drawdown from Peak")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────
#  5. Return Distribution (histogram + KDE)
# ──────────────────────────────────────────────────────
def plot_return_distribution(cc_returns, stock_returns, save_path=None,
                             title="Return Distribution", x_label="Return"):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(cc_returns.min(), stock_returns.min()),
                       max(cc_returns.max(), stock_returns.max()), 40)
    ax.hist(stock_returns, bins=bins, alpha=0.4, label="Stock Only", density=True)
    ax.hist(cc_returns, bins=bins, alpha=0.4, label="Covered Call", density=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────
#  6. Risk-Return Scatter
# ──────────────────────────────────────────────────────
def plot_risk_return_scatter(points, save_path=None):
    """
    points: list of dicts with keys 'label', 'return', 'volatility'
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for p in points:
        ax.scatter(p["volatility"], p["return"], s=100, zorder=5)
        ax.annotate(p["label"], (p["volatility"], p["return"]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Risk–Return Tradeoff")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────
#  7. Moneyness–Expiry Heatmap
# ──────────────────────────────────────────────────────
def plot_heatmap(data, strike_offsets, expiry_range, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    xlabels = [str(d) for d in expiry_range]
    ylabels = [f"{o:+.0%}" for o in strike_offsets]
    sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                xticklabels=xlabels, yticklabels=ylabels, ax=ax)
    ax.set_xlabel("Days to Expiry")
    ax.set_ylabel("Strike Offset (moneyness)")
    ax.set_title("Covered Call Sharpe Ratio — Strike vs Expiry")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────
#  8. Rolling Sharpe Ratio
# ──────────────────────────────────────────────────────
def plot_rolling_sharpe(backtest_df, window=60, rf=0.07, save_path=None):
    daily_rf = (1 + rf) ** (1/252) - 1

    fig, ax = plt.subplots(figsize=(10, 4))
    for col, label in [("stock_return", "Stock Only"), ("cc_return", "Covered Call")]:
        excess = backtest_df[col] - daily_rf
        rolling_mean = excess.rolling(window).mean()
        rolling_std = backtest_df[col].rolling(window).std()
        rolling_sr = (rolling_mean / rolling_std) * np.sqrt(252)
        ax.plot(rolling_sr.index, rolling_sr.values, label=label, alpha=0.8)

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Sharpe Ratio")
    ax.set_title(f"{window}-Day Rolling Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────
#  9. Greeks vs Stock Price
# ──────────────────────────────────────────────────────
def plot_greeks(K, T, r, sigma, save_path=None):
    from .black_scholes import covered_call_greeks

    S_range = np.linspace(0.7 * K, 1.3 * K, 200)
    deltas, gammas, thetas = [], [], []
    for S in S_range:
        g = covered_call_greeks(S, K, T, r, sigma)
        deltas.append(g["delta"])
        gammas.append(g["gamma"])
        thetas.append(g["theta"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(S_range, deltas)
    axes[0].set_title("Delta (Covered Call)")
    axes[0].set_xlabel("Stock Price")
    axes[0].axvline(K, color='gray', linestyle=':', alpha=0.5)

    axes[1].plot(S_range, gammas, color='orange')
    axes[1].set_title("Gamma (Covered Call)")
    axes[1].set_xlabel("Stock Price")
    axes[1].axvline(K, color='gray', linestyle=':', alpha=0.5)

    axes[2].plot(S_range, thetas, color='green')
    axes[2].set_title("Theta (Covered Call)")
    axes[2].set_xlabel("Stock Price")
    axes[2].axvline(K, color='gray', linestyle=':', alpha=0.5)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.suptitle("Covered Call Position Greeks", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────
#  10. Monte Carlo Fan Chart
# ──────────────────────────────────────────────────────
def plot_mc_fan_chart(paths_cc, paths_stock, save_path=None):
    """
    paths_cc and paths_stock: arrays of shape (num_steps+1, num_paths)
    representing equity value over time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    time_axis = np.arange(paths_cc.shape[0])

    for ax, paths, title in [(axes[0], paths_stock, "Stock Only"),
                              (axes[1], paths_cc, "Covered Call")]:
        p5  = np.percentile(paths, 5, axis=1)
        p25 = np.percentile(paths, 25, axis=1)
        p50 = np.percentile(paths, 50, axis=1)
        p75 = np.percentile(paths, 75, axis=1)
        p95 = np.percentile(paths, 95, axis=1)

        ax.fill_between(time_axis, p5, p95, alpha=0.15, color='blue', label='5th–95th')
        ax.fill_between(time_axis, p25, p75, alpha=0.3, color='blue', label='25th–75th')
        ax.plot(time_axis, p50, 'b-', linewidth=1.5, label='Median')
        ax.set_title(title)
        ax.set_xlabel("Trading Days")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Portfolio Value")
    plt.suptitle("Monte Carlo Simulation — Covered Call vs Stock", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ──────────────────────────────────────────────────────
#  Plotly versions for the Streamlit app
# ──────────────────────────────────────────────────────

def plotly_payoff(S_0, K, C_0):
    S_range = np.linspace(0.5 * S_0, 1.5 * S_0, 500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=stock_only_profit(S_range, S_0),
                             mode='lines', name='Stock Only', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=S_range, y=covered_call_profit(S_range, S_0, K, C_0),
                             mode='lines', name='Covered Call', line=dict(width=2.5)))
    fig.add_hline(y=0, line_color='gray', line_width=0.5)
    fig.update_layout(title="Profit/Loss at Expiry",
                      xaxis_title="Stock Price at Expiry (S_T)",
                      yaxis_title="Profit / Loss",
                      template="plotly_white")
    return fig


def plotly_equity_curves(backtest_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["stock_equity"],
                             mode='lines', name='Stock Only'))
    fig.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df["cc_equity"],
                             mode='lines', name='Covered Call'))
    fig.update_layout(title="Equity Curves — Covered Call vs Buy-and-Hold",
                      yaxis_title="Normalized Portfolio Value",
                      yaxis_type="log", template="plotly_white")
    return fig


def plotly_return_dist(cc_returns, stock_returns,
                       title="Return Distribution", x_label="Return"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=stock_returns, name="Stock Only",
                               opacity=0.5, nbinsx=40, histnorm='probability density'))
    fig.add_trace(go.Histogram(x=cc_returns, name="Covered Call",
                               opacity=0.5, nbinsx=40, histnorm='probability density'))
    fig.update_layout(barmode='overlay', title=title,
                      xaxis_title=x_label, yaxis_title="Density",
                      template="plotly_white")
    return fig


# ──────────────────────────────────────────────────────
#  11. Cross-sectional: Beta vs Covered Call Alpha
# ──────────────────────────────────────────────────────
def plot_beta_vs_alpha(combined_df, save_path=None):
    """Scatter of beta (x) vs covered-call alpha (y), colored by sector.

    `combined_df` must have columns [ticker, sector, beta, alpha_cc].
    """
    df = combined_df.dropna(subset=["beta", "alpha_cc"]).copy()
    fig, ax = plt.subplots(figsize=(9, 6))

    sectors = sorted(df["sector"].unique())
    cmap = plt.get_cmap("tab10")
    for i, sector in enumerate(sectors):
        sub = df[df["sector"] == sector]
        ax.scatter(sub["beta"], sub["alpha_cc"], s=120, alpha=0.8,
                   color=cmap(i % 10), label=sector, edgecolor="black", linewidth=0.5)
        for _, row in sub.iterrows():
            ax.annotate(row["ticker"].replace(".NS", ""),
                        (row["beta"], row["alpha_cc"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

    # OLS trend line
    if len(df) >= 2:
        slope, intercept = np.polyfit(df["beta"], df["alpha_cc"], 1)
        x_line = np.linspace(df["beta"].min(), df["beta"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "k--", alpha=0.6,
                label=f"OLS: slope={slope:+.3f}")

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Beta vs Nifty 50")
    ax.set_ylabel("Covered Call Alpha (CC Total Return − Stock Total Return)")
    ax.set_title("Where the Covered Call Wins: Beta vs Alpha")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────
#  12. Cross-sectional: Sector × Tenure Heatmap
# ──────────────────────────────────────────────────────
def plot_sector_tenure_heatmap(slice_df, save_path=None, title=None):
    """Heatmap of mean alpha_cc by (sector, period_label).

    `slice_df` is the long-format output of multi_ticker_backtest.slice_by_tenure.
    """
    if slice_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    pivot = (slice_df.groupby(["sector", "period_label"])["alpha_cc"]
             .mean()
             .unstack("period_label"))

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(pivot.columns) + 4), 5))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0,
                cbar_kws={"label": "Mean Covered Call Alpha"}, ax=ax)
    ax.set_xlabel("Period")
    ax.set_ylabel("Sector")
    ax.set_title(title or "Sector × Tenure: Covered Call Alpha")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────
#  13. Cross-sectional: Alpha by Market Regime
# ──────────────────────────────────────────────────────
def plot_alpha_by_regime(slice_df, save_path=None):
    """Bar chart of mean alpha_cc by regime, with std error bars across stocks."""
    if slice_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    # Extract regime label from "2022 (Bear)" etc.
    df = slice_df.copy()
    df["regime"] = df["period_label"].str.extract(r"\((\w+)\)")
    df = df.dropna(subset=["regime"])
    if df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No regime labels found", ha="center", va="center")
        return fig

    regime_order = ["Bull", "Sideways", "Bear"]
    grouped = df.groupby("regime")["alpha_cc"].agg(["mean", "std", "count"])
    grouped = grouped.reindex([r for r in regime_order if r in grouped.index])

    colors = {"Bull": "#2ca02c", "Sideways": "#7f7f7f", "Bear": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(grouped.index, grouped["mean"],
                  yerr=grouped["std"].fillna(0), capsize=8,
                  color=[colors.get(r, "gray") for r in grouped.index],
                  edgecolor="black")
    for bar, n in zip(bars, grouped["count"]):
        height = bar.get_height()
        ax.annotate(f"n={int(n)}", (bar.get_x() + bar.get_width() / 2, height),
                    ha="center", va="bottom" if height >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Covered Call Alpha")
    ax.set_title("Covered Call Performance by Market Regime")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────
#  14. Cross-sectional: Rolling Alpha Time Series
# ──────────────────────────────────────────────────────
def plot_rolling_alpha_timeseries(slice_df, save_path=None):
    """Time series of sector-mean alpha across rolling windows."""
    if slice_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    df = slice_df.copy()
    grouped = df.groupby(["sector", "period_label", "period_start"])["alpha_cc"].mean().reset_index()
    grouped = grouped.sort_values("period_start")

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("tab10")
    for i, (sector, sub) in enumerate(grouped.groupby("sector")):
        ax.plot(sub["period_start"], sub["alpha_cc"], "-o", label=sector,
                alpha=0.85, color=cmap(i % 10), markersize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Window Start")
    ax.set_ylabel("Sector Mean Covered Call Alpha")
    ax.set_title("Rolling-Window Covered Call Alpha by Sector")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────
#  15. Validation: BS price vs Market price
# ──────────────────────────────────────────────────────
def plot_bs_vs_market(validation_df, save_path=None):
    """Scatter of BS theoretical vs market price for current option chain."""
    df = validation_df
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(df["market_price"], df["bs_price"], s=80, alpha=0.7,
               edgecolor="black", linewidth=0.5)
    lo = min(df["market_price"].min(), df["bs_price"].min()) * 0.95
    hi = max(df["market_price"].max(), df["bs_price"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel("Market Price")
    ax.set_ylabel("Black-Scholes Price")
    title_meta = ""
    if "ticker" in df.attrs:
        title_meta = f" — {df.attrs['ticker']} ({df.attrs.get('expiry', '')})"
    ax.set_title(f"BS vs Market Call Prices{title_meta}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    residuals = df["bs_price"] - df["market_price"]
    ax.bar(df["strike"], residuals, color=["green" if r > 0 else "red" for r in residuals],
           alpha=0.7, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Strike")
    ax.set_ylabel("BS − Market")
    ax.set_title("Pricing Residuals")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────
#  Plotly versions of cross-sectional plots (Streamlit)
# ──────────────────────────────────────────────────────
def plotly_beta_vs_alpha(combined_df):
    import plotly.express as px
    df = combined_df.dropna(subset=["beta", "alpha_cc"]).copy()
    df["ticker_short"] = df["ticker"].str.replace(".NS", "", regex=False)
    fig = px.scatter(df, x="beta", y="alpha_cc", color="sector", text="ticker_short",
                     hover_data=["ticker", "cc_total_return", "stock_total_return"],
                     title="Beta vs Covered Call Alpha")
    if len(df) >= 2:
        slope, intercept = np.polyfit(df["beta"], df["alpha_cc"], 1)
        x_line = np.linspace(df["beta"].min(), df["beta"].max(), 100)
        fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept,
                                 mode="lines", name=f"OLS slope={slope:+.3f}",
                                 line=dict(dash="dash", color="black")))
    fig.update_traces(textposition="top center", marker=dict(size=12, line=dict(width=1, color="black")))
    fig.update_layout(xaxis_title="Beta vs Nifty 50",
                      yaxis_title="Covered Call Alpha",
                      yaxis_tickformat=".0%",
                      template="plotly_white")
    fig.add_hline(y=0, line_color="gray", line_width=0.8)
    return fig


def plotly_sector_tenure_heatmap(slice_df, title="Sector × Tenure: Covered Call Alpha"):
    import plotly.express as px
    if slice_df.empty:
        return go.Figure()
    pivot = (slice_df.groupby(["sector", "period_label"])["alpha_cc"]
             .mean()
             .unstack("period_label"))
    fig = px.imshow(pivot, text_auto=".1%", color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    labels=dict(color="Alpha"))
    fig.update_layout(title=title, xaxis_title="Period", yaxis_title="Sector",
                      template="plotly_white")
    return fig


def plotly_alpha_by_regime(slice_df):
    if slice_df.empty:
        return go.Figure()
    df = slice_df.copy()
    df["regime"] = df["period_label"].str.extract(r"\((\w+)\)")
    df = df.dropna(subset=["regime"])
    if df.empty:
        return go.Figure()
    regime_order = ["Bull", "Sideways", "Bear"]
    grouped = df.groupby("regime")["alpha_cc"].agg(["mean", "std", "count"])
    grouped = grouped.reindex([r for r in regime_order if r in grouped.index]).reset_index()

    colors = {"Bull": "#2ca02c", "Sideways": "#7f7f7f", "Bear": "#d62728"}
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["regime"],
        y=grouped["mean"],
        error_y=dict(type="data", array=grouped["std"].fillna(0)),
        marker_color=[colors.get(r, "gray") for r in grouped["regime"]],
        text=[f"n={int(n)}" for n in grouped["count"]],
        textposition="outside",
    ))
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.update_layout(title="Covered Call Performance by Market Regime",
                      yaxis_title="Mean Covered Call Alpha",
                      yaxis_tickformat=".0%",
                      template="plotly_white")
    return fig


def plotly_rolling_alpha_timeseries(slice_df):
    if slice_df.empty:
        return go.Figure()
    df = slice_df.copy()
    grouped = df.groupby(["sector", "period_label", "period_start"])["alpha_cc"].mean().reset_index()
    grouped = grouped.sort_values("period_start")

    fig = go.Figure()
    for sector, sub in grouped.groupby("sector"):
        fig.add_trace(go.Scatter(x=sub["period_start"], y=sub["alpha_cc"],
                                 mode="lines+markers", name=sector))
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.update_layout(title="Rolling-Window Covered Call Alpha by Sector",
                      xaxis_title="Window Start",
                      yaxis_title="Sector Mean Alpha",
                      yaxis_tickformat=".0%",
                      template="plotly_white")
    return fig


def plotly_bs_vs_market(validation_df):
    df = validation_df
    fig = make_subplots(rows=1, cols=2, subplot_titles=("BS vs Market", "Residuals"))

    fig.add_trace(go.Scatter(x=df["market_price"], y=df["bs_price"],
                             mode="markers", marker=dict(size=10),
                             name="Strikes"), row=1, col=1)
    lo = min(df["market_price"].min(), df["bs_price"].min()) * 0.95
    hi = max(df["market_price"].max(), df["bs_price"].max()) * 1.05
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(dash="dash", color="black"),
                             name="y = x"), row=1, col=1)

    residuals = df["bs_price"] - df["market_price"]
    colors = ["green" if r > 0 else "red" for r in residuals]
    fig.add_trace(go.Bar(x=df["strike"], y=residuals, marker_color=colors,
                         name="BS − Market"), row=1, col=2)

    fig.update_xaxes(title_text="Market Price", row=1, col=1)
    fig.update_yaxes(title_text="Black-Scholes Price", row=1, col=1)
    fig.update_xaxes(title_text="Strike", row=1, col=2)
    fig.update_yaxes(title_text="BS − Market", row=1, col=2)
    fig.update_layout(template="plotly_white", showlegend=False,
                      title=f"BS vs Market — {df.attrs.get('ticker', '')} {df.attrs.get('expiry', '')}")
    return fig


def plotly_mc_fan(paths, title="Monte Carlo Simulation"):
    t = np.arange(paths.shape[0])
    p5  = np.percentile(paths, 5, axis=1)
    p25 = np.percentile(paths, 25, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p95 = np.percentile(paths, 95, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=p95, mode='lines', line=dict(width=0),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=t, y=p5, fill='tonexty', mode='lines',
                             line=dict(width=0), name='5th–95th', fillcolor='rgba(0,100,255,0.1)'))
    fig.add_trace(go.Scatter(x=t, y=p75, mode='lines', line=dict(width=0),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=t, y=p25, fill='tonexty', mode='lines',
                             line=dict(width=0), name='25th–75th', fillcolor='rgba(0,100,255,0.25)'))
    fig.add_trace(go.Scatter(x=t, y=p50, mode='lines', name='Median',
                             line=dict(color='blue', width=2)))
    fig.update_layout(title=title, xaxis_title="Trading Days",
                      yaxis_title="Portfolio Value", template="plotly_white")
    return fig
