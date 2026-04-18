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
                                plot_return_distribution,
                                plot_beta_vs_alpha, plot_sector_tenure_heatmap,
                                plot_alpha_by_regime, plot_rolling_alpha_timeseries,
                                plot_bs_vs_market)
from src import beta_analysis, market_validation, multi_ticker_backtest, tenure_slicers
from src.universe import BENCHMARK_TICKER, all_tickers


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


def write_cross_sectional_table(combined_df, output_path):
    """Write a LaTeX longtable of per-ticker cross-sectional results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = combined_df.dropna(subset=["beta", "alpha_cc"]).copy()
    df = df.sort_values("beta")

    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"    \toprule",
        r"    \textbf{Ticker} & \textbf{Sector} & \textbf{Beta} & "
        r"\textbf{CC Return} & \textbf{Stock Return} & \textbf{Alpha} & \textbf{Win?} \\",
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        ticker = latex_escape(row["ticker"].replace(".NS", ""))
        sector = latex_escape(row["sector"])
        beta = f"{row['beta']:.2f}"
        cc_ret = f"{row['cc_total_return']*100:.1f}\\%"
        stock_ret = f"{row['stock_total_return']*100:.1f}\\%"
        alpha = f"{row['alpha_cc']*100:+.1f}\\%"
        win = "Yes" if row["win"] else "No"
        lines.append(
            f"    {ticker} & {sector} & {beta} & {cc_ret} & {stock_ret} & {alpha} & {win} \\\\"
        )
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

    # ─── Cross-sectional analysis ──────────────────────────
    print("\n[cross-sectional] Running multi-ticker backtest...")
    universe = all_tickers()
    results_dict = multi_ticker_backtest.run_universe_backtest(universe)
    print(f"[cross-sectional] {len(results_dict)}/{len(universe)} tickers succeeded")

    if results_dict:
        alpha_table = multi_ticker_backtest.compute_alpha_table(results_dict)
        beta_table = beta_analysis.compute_beta_table(universe)
        combined = alpha_table.merge(
            beta_table[["ticker", "beta", "alpha_jensen", "r_squared"]],
            on="ticker", how="left"
        )

        save_figure(
            plot_beta_vs_alpha(combined),
            figure_dir / "beta_vs_alpha.png",
        )

        # Calendar year heatmap
        any_df = next(iter(results_dict.values()))
        year_buckets = tenure_slicers.slice_by_calendar_year(any_df)
        slice_year = multi_ticker_backtest.slice_by_tenure(results_dict, year_buckets)
        save_figure(
            plot_sector_tenure_heatmap(slice_year, title="Sector × Calendar Year: Covered Call Alpha"),
            figure_dir / "sector_tenure_heatmap_year.png",
        )

        # Regime heatmap + bar
        try:
            nifty_close = get_stock_data(BENCHMARK_TICKER)["Close"]
            regime_buckets = tenure_slicers.slice_by_market_regime(any_df, nifty_close)
            slice_regime = multi_ticker_backtest.slice_by_tenure(results_dict, regime_buckets)
            save_figure(
                plot_sector_tenure_heatmap(slice_regime, title="Sector × Market Regime: Covered Call Alpha"),
                figure_dir / "sector_tenure_heatmap_regime.png",
            )
            save_figure(
                plot_alpha_by_regime(slice_regime),
                figure_dir / "alpha_by_regime.png",
            )
        except Exception as exc:
            print(f"[cross-sectional] Regime analysis failed: {exc}")

        # Rolling 6-month
        rolling_buckets = tenure_slicers.slice_by_rolling_window(any_df)
        slice_rolling = multi_ticker_backtest.slice_by_tenure(results_dict, rolling_buckets)
        save_figure(
            plot_sector_tenure_heatmap(slice_rolling, title="Sector × Rolling 6mo: Covered Call Alpha"),
            figure_dir / "sector_tenure_heatmap_rolling.png",
        )
        save_figure(
            plot_rolling_alpha_timeseries(slice_rolling),
            figure_dir / "rolling_alpha_timeseries.png",
        )

        # Cross-sectional table
        write_cross_sectional_table(combined, table_dir / "cross_sectional_alpha.tex")

    # ─── BS vs market validation ───────────────────────────
    try:
        print("\n[validation] Fetching current option chain for validation...")
        val_df = market_validation.validate_bs_against_market(config.DEFAULT_TICKER)
        save_figure(
            plot_bs_vs_market(val_df),
            figure_dir / "bs_vs_market_validation.png",
        )
        mean_pct_err = float(val_df["pct_error"].mean())
        print(f"[validation] Mean pct error: {mean_pct_err:.1%}")
    except Exception as exc:
        print(f"[validation] Skipped: {exc}")

    print(f"\nGenerated {len(list(figure_dir.glob('*.png')))} figures in {figure_dir}")
    print(f"Wrote metrics table to {table_dir / 'performance_metrics.tex'}")
    print(f"Monthly return samples: covered call={len(cc_monthly)}, stock={len(stock_monthly)}")


if __name__ == "__main__":
    main()
