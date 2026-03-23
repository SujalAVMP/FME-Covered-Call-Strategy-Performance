"""Central configuration and default parameters."""

# Default stock
DEFAULT_TICKER = "RELIANCE.NS"
PRESET_TICKERS = {
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
    "State Bank of India (SBIN.NS)": "SBIN.NS",
    "ITC (ITC.NS)": "ITC.NS",
    "Larsen & Toubro (LT.NS)": "LT.NS",
    "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
    "Hindustan Unilever (HINDUNILVR.NS)": "HINDUNILVR.NS",
}

# Model risk-free rate assumption used in pricing and performance metrics
RISK_FREE_RATE = 0.07

# Backtest date range
BACKTEST_START = "2020-01-01"
BACKTEST_END = "2025-12-31"

# Rolling covered call parameters
DEFAULT_STRIKE_OFFSET = 0.05   # 5% OTM by default
DEFAULT_EXPIRY_DAYS = 30       # Monthly rolls
DEFAULT_VOL_WINDOW = 30        # 30-day rolling historical vol

# Monte Carlo settings
MC_NUM_PATHS = 10_000
MC_NUM_STEPS = 252             # Trading days in a year
MC_SEED = 42

# Sensitivity sweep ranges
STRIKE_OFFSETS = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15]
EXPIRY_DAYS_RANGE = [7, 14, 21, 30, 45, 60, 90]

# Trading days per year (for annualization)
TRADING_DAYS = 252
