"""Generate realistic sample data for development and testing.

This script creates synthetic historical price data that follows realistic
market patterns. Use this when actual Yahoo Finance data is unavailable
(e.g., in restricted network environments).

The sample data uses known approximate CAGRs and volatilities for each ticker
to produce plausible time series. It is NOT real market data and should not
be used for actual financial analysis.

Usage:
    python seed_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Approximate financial characteristics for each ticker
# (start_price, annual_return, annual_volatility, start_date)
# Based on approximate real-world performance 2015-2025
TICKER_PROFILES = {
    # Pittsburgh HQ Index constituents
    "PNC":  (88,   0.12, 0.25, "2015-01-02"),
    "PPG":  (115,  0.08, 0.22, "2015-01-02"),
    "ANSS": (85,   0.22, 0.28, "2015-01-02"),
    "WAB":  (42,   0.15, 0.30, "2015-01-02"),
    "HWM":  (18,   0.35, 0.35, "2020-04-01"),
    "DKS":  (44,   0.18, 0.35, "2015-01-02"),
    "COHR": (40,   0.12, 0.40, "2015-01-02"),
    "WCC":  (38,   0.15, 0.35, "2015-01-02"),
    "VTRS": (16,  -0.05, 0.28, "2020-11-16"),
    "X":    (27,   0.02, 0.45, "2015-01-02"),
    "AA":   (28,   0.05, 0.40, "2016-11-01"),
    "DUOL": (102,  0.30, 0.45, "2021-07-28"),
    "AUR":  (10,  -0.10, 0.55, "2021-11-04"),
    "CNX":  (18,   0.10, 0.35, "2015-01-02"),
    "FNB":  (13,   0.08, 0.28, "2015-01-02"),
    "AEO":  (14,   0.05, 0.35, "2015-01-02"),
    "KMT":  (35,   0.02, 0.30, "2015-01-02"),
    "KOP":  (35,   0.08, 0.30, "2015-01-02"),

    # Pittsburgh Operations Index additions
    "GOOGL": (27,  0.20, 0.25, "2015-01-02"),
    "UBER":  (42, 0.15, 0.40, "2019-05-10"),
    "AMZN":  (15,  0.25, 0.28, "2015-01-02"),
    "MSFT":  (46,  0.28, 0.25, "2015-01-02"),
    "RTX":   (48,  0.10, 0.25, "2015-01-02"),
    "ETN":   (65,  0.18, 0.25, "2015-01-02"),
    "BKNG":  (1100, 0.12, 0.28, "2015-01-02"),
    "SHW":   (125, 0.18, 0.22, "2015-01-02"),

    # Benchmarks
    "SPY":  (206, 0.12, 0.16, "2015-01-02"),
    "VT":   (58,  0.08, 0.15, "2015-01-02"),
    "VTI":  (107, 0.12, 0.16, "2015-01-02"),
    "IWM":  (119, 0.07, 0.20, "2015-01-02"),
    "XLI":  (55,  0.10, 0.18, "2015-01-02"),
    "XLF":  (24,  0.10, 0.20, "2015-01-02"),
    "XLB":  (48,  0.08, 0.18, "2015-01-02"),
}

# Shared market shocks to inject correlation between stocks
# (date_range, magnitude) — these simulate real events
MARKET_EVENTS = [
    # 2018 Q4 selloff
    ("2018-10-01", "2018-12-24", -0.15),
    # COVID crash
    ("2020-02-20", "2020-03-23", -0.30),
    # COVID recovery
    ("2020-03-24", "2020-08-31", 0.40),
    # 2022 bear market
    ("2022-01-03", "2022-10-12", -0.20),
    # 2023 recovery
    ("2023-01-01", "2023-07-31", 0.18),
    # 2024 rally
    ("2024-01-01", "2024-12-31", 0.20),
]


def generate_price_series(
    start_price: float,
    annual_return: float,
    annual_volatility: float,
    start_date: str,
    end_date: str = "2026-02-14",
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV price series.

    Uses geometric Brownian motion with drift, plus shared market events
    for realistic cross-stock correlation.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # Generate business day date range
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)

    if n == 0:
        return pd.DataFrame()

    # Daily parameters
    daily_return = annual_return / 252
    daily_vol = annual_volatility / np.sqrt(252)

    # Generate random daily returns (GBM)
    daily_returns = rng.normal(daily_return, daily_vol, n)

    # Apply market events (shared shocks for correlation)
    for event_start, event_end, magnitude in MARKET_EVENTS:
        event_start_ts = pd.Timestamp(event_start)
        event_end_ts = pd.Timestamp(event_end)

        mask = (dates >= event_start_ts) & (dates <= event_end_ts)
        event_days = mask.sum()
        if event_days > 0:
            # Distribute the event magnitude across event days
            # Add some randomness to the distribution
            event_daily = magnitude / event_days
            # Company-specific sensitivity (beta-like, 0.5 to 1.5)
            beta = 0.5 + rng.random() * 1.0
            daily_returns[mask] += event_daily * beta

    # Build price series from returns
    cumulative = np.cumprod(1 + daily_returns)
    close_prices = start_price * cumulative

    # Generate OHLV from close
    daily_range = daily_vol * close_prices
    high = close_prices + np.abs(rng.normal(0, 1, n)) * daily_range * 0.5
    low = close_prices - np.abs(rng.normal(0, 1, n)) * daily_range * 0.5
    open_prices = close_prices * (1 + rng.normal(0, daily_vol * 0.3, n))

    # Ensure high >= close >= low
    high = np.maximum(high, np.maximum(close_prices, open_prices))
    low = np.minimum(low, np.minimum(close_prices, open_prices))
    low = np.maximum(low, 0.01)  # No negative prices

    # Volume: roughly correlated with volatility
    base_volume = rng.lognormal(15, 0.5, n)  # ~3M shares average
    volume = (base_volume * (1 + np.abs(daily_returns) * 10)).astype(int)

    df = pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close_prices,
        "Volume": volume,
    }, index=dates)

    df.index.name = "Date"
    return df


def main():
    cache_dir = Path(__file__).parent / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating sample data in {cache_dir}/")

    for i, (ticker, (start_price, annual_ret, annual_vol, start_date)) in enumerate(TICKER_PROFILES.items()):
        df = generate_price_series(
            start_price=start_price,
            annual_return=annual_ret,
            annual_volatility=annual_vol,
            start_date=start_date,
            seed=42 + i,  # Deterministic but different per ticker
        )

        path = cache_dir / f"{ticker}.parquet"
        df.to_parquet(path)
        print(f"  {ticker}: {len(df)} days, ${df['Close'].iloc[0]:.2f} -> ${df['Close'].iloc[-1]:.2f} ({start_date} to {df.index[-1].date()})")

    print(f"\nDone. {len(TICKER_PROFILES)} tickers seeded.")
    print("Run the pipeline with: python -c \"from src.pipeline import Pipeline; ...\"")


if __name__ == "__main__":
    main()
