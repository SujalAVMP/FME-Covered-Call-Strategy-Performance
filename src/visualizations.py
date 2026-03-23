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
