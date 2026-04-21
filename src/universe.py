"""Multi-ticker universe and sector mapping for cross-sectional analysis."""

# Sector → list of NSE tickers. 16 stocks across 7 sectors.
SECTOR_UNIVERSE = {
    "Energy":   ["RELIANCE.NS", "ONGC.NS"],
    "IT":       ["TCS.NS", "INFY.NS", "WIPRO.NS"],
    "Banking":  ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"],
    "FMCG":     ["HINDUNILVR.NS", "ITC.NS"],
    "Pharma":   ["SUNPHARMA.NS", "DRREDDY.NS"],
    "Auto":     ["MARUTI.NS", "M&M.NS"],
    "Metals":   ["TATASTEEL.NS", "HINDALCO.NS"],
    "US Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
}

# Benchmark for beta calculations
BENCHMARK_TICKER = "^NSEI"  # Nifty 50 index


def all_tickers():
    """Flat list of every ticker in the universe."""
    return [t for tickers in SECTOR_UNIVERSE.values() for t in tickers]


def sector_of(ticker):
    """Return the sector name for a given ticker, or 'Unknown' if missing."""
    for sector, tickers in SECTOR_UNIVERSE.items():
        if ticker in tickers:
            return sector
    return "Unknown"


def tickers_in_sectors(sectors):
    """Return all tickers that belong to any of the given sectors."""
    return [t for s in sectors if s in SECTOR_UNIVERSE for t in SECTOR_UNIVERSE[s]]
