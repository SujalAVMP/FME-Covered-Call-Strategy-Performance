# Covered Call Strategy Performance

ES418 Financial Modelling and Engineering, Group 10.

This project studies the performance of a rolling covered call strategy and compares it with a stock-only buy-and-hold benchmark. The repository includes:

- Python modules for pricing, backtesting, risk analysis, and sensitivity analysis
- A Streamlit app for interactive exploration
- A LaTeX report with generated figures and tables

The default backtest uses approximately monthly 21-trading-day rolls, a 30-day realized-volatility estimate, and a fixed initial volatility fallback only before enough history exists to estimate realized volatility without look-ahead.

## Project Structure

```text
covered_call_project/
├── app/                    # Streamlit app
├── output/                 # Generated figures and report tables
├── report/                 # LaTeX report source
├── scripts/                # Utility scripts
├── src/                    # Core analysis code
├── tests/                  # Automated tests
├── pyproject.toml          # uv project configuration
└── uv.lock                 # Locked dependency versions
```

## Prerequisites

- Python 3.12 or newer
- `uv` installed: https://docs.astral.sh/uv/

## Setup

From the project root:

```bash
uv sync --all-groups
```

This creates a local virtual environment and installs both runtime and test dependencies.

## Run the Streamlit App

```bash
uv run streamlit run app/streamlit_app.py
```

The sidebar includes a preset stock dropdown for common NSE tickers, along with a `Custom` option if you want to enter any valid Yahoo Finance ticker manually.

## Run the Tests

```bash
uv run python scripts/run_tests.py
```

## Regenerate Report Assets

The report uses figures and a metrics table generated from the Python code.

```bash
uv run python scripts/generate_report_assets.py
```

This writes PNG figures to `output/figures/` and the performance table to `output/tables/performance_metrics.tex`.

The Black-Scholes market-validation plot is generated only when the upstream data provider exposes a live option chain for the selected ticker.
