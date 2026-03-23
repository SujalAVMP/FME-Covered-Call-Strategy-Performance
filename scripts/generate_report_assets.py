"""Generate the figures and metrics table used by the project report."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.backtest import compute_monthly_returns, prepare_volatility_series, run_backtest
from src.black_scholes import bs_call_price
from src.data_fetcher import get_stock_data
from src.monte_carlo import (covered_call_equity_paths, covered_call_mc_returns,
                             simulate_gbm_paths, stock_only_mc_returns)
from src.risk_metrics import summary_table
from src.sensitivity import moneyness_expiry_heatmap
from src.visualizations import (plot_drawdown, plot_equity_curves, plot_greeks,
                                plot_heatmap, plot_mc_fan_chart,
                                plot_payoff_diagram, plot_profit_diagram,
                                plot_return_distribution)


def save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def latex_escape(text):
    """Escape LaTeX special characters in generated table content."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = str(text)
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def write_metrics_table(metrics_df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_rows = [
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "Calmar Ratio",
        "VaR (95%)",
        "CVaR (95%)",
        "Win Rate",
    ]

    lines = [
        r"\begin{tabular}{lcc}",
        r"    \toprule",
        r"    \textbf{Metric} & \textbf{Covered Call} & \textbf{Stock Only} \\",
        r"    \midrule",
    ]

    for metric in metric_rows:
        cc_value = latex_escape(metrics_df.loc["Covered Call", metric])
        stock_value = latex_escape(metrics_df.loc["Stock Only", metric])
        lines.append(f"    {latex_escape(metric)} & {cc_value} & {stock_value} \\\\")

    lines.extend([r"    \bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    figure_dir = ROOT / "output" / "figures"
    table_dir = ROOT / "output" / "tables"

    data = get_stock_data()
    prices = data["Close"].dropna()
    hist_vol = prepare_volatility_series(prices, window=config.DEFAULT_VOL_WINDOW)
    sigma = float(hist_vol.iloc[-1])

    S_0 = float(prices.iloc[-1])
    K = S_0 * (1 + config.DEFAULT_STRIKE_OFFSET)
    T = config.DEFAULT_EXPIRY_DAYS / config.TRADING_DAYS
    premium = bs_call_price(S_0, K, T, config.RISK_FREE_RATE, sigma)

    save_figure(
        plot_payoff_diagram(S_0, K),
        figure_dir / "payoff_diagram.png",
    )
    save_figure(
        plot_profit_diagram(S_0, K, premium),
        figure_dir / "profit_diagram.png",
    )

    backtest_df, _ = run_backtest()
    save_figure(
        plot_equity_curves(backtest_df),
        figure_dir / "equity_curves.png",
    )
    save_figure(
        plot_drawdown(backtest_df),
        figure_dir / "drawdown.png",
    )

    metrics = summary_table(
        backtest_df["cc_return"].dropna(),
        backtest_df["stock_return"].dropna(),
        backtest_df["cc_equity"],
        backtest_df["stock_equity"],
    )
    write_metrics_table(metrics, table_dir / "performance_metrics.tex")

    mu_real = float(np.log(prices.iloc[-1] / prices.iloc[0]) / (len(prices) / config.TRADING_DAYS))
    paths = simulate_gbm_paths(
        S_0,
        mu_real,
        sigma,
        T,
        num_steps=config.DEFAULT_EXPIRY_DAYS,
        num_paths=config.MC_NUM_PATHS,
    )
    cc_mc_returns = covered_call_mc_returns(paths, K, premium, S_0)
    stock_mc_returns = stock_only_mc_returns(paths, S_0)

    stock_equity_paths = paths / S_0
    cc_equity_paths = covered_call_equity_paths(
        paths,
        K,
        premium,
        config.RISK_FREE_RATE,
        sigma,
        T,
    ) / S_0

    save_figure(
        plot_mc_fan_chart(cc_equity_paths, stock_equity_paths),
        figure_dir / "mc_fan_chart.png",
    )
    save_figure(
        plot_return_distribution(
            cc_mc_returns,
            stock_mc_returns,
            title="Terminal Return Distribution",
            x_label="Terminal Return",
        ),
        figure_dir / "return_distribution.png",
    )

    heatmap, strike_offsets, expiry_range = moneyness_expiry_heatmap(
        S_0, sigma, mu=mu_real, num_paths=2000
    )
    save_figure(
        plot_heatmap(heatmap, strike_offsets, expiry_range),
        figure_dir / "heatmap.png",
    )
    save_figure(
        plot_greeks(K, T, config.RISK_FREE_RATE, sigma),
        figure_dir / "greeks.png",
    )

    cc_monthly = compute_monthly_returns(backtest_df, "cc_equity")
    stock_monthly = compute_monthly_returns(backtest_df, "stock_equity")

    print(f"Generated {len(list(figure_dir.glob('*.png')))} figures in {figure_dir}")
    print(f"Wrote metrics table to {table_dir / 'performance_metrics.tex'}")
    print(f"Monthly return samples: covered call={len(cc_monthly)}, stock={len(stock_monthly)}")


if __name__ == "__main__":
    main()
