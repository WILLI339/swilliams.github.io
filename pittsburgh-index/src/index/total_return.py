"""Total return calculations.

When using yfinance with auto_adjust=True, the Close prices already reflect
total return (splits + dividends reinvested). This module provides utilities
for computing total return metrics and comparing price-only vs total return.
"""

import logging

import pandas as pd

from ..data.fetcher import fetch_dividends, fetch_splits

logger = logging.getLogger(__name__)


def compute_total_return(series: pd.Series) -> float:
    """Compute total return (cumulative) for a price series.

    Args:
        series: Price series (assumed to be total-return-adjusted).

    Returns:
        Total return as a decimal (e.g., 1.5 means 150% total return).
    """
    if series.empty or len(series) < 2:
        return 0.0
    return (series.iloc[-1] / series.iloc[0]) - 1


def compute_annualized_return(series: pd.Series) -> float:
    """Compute annualized return (CAGR) for a price series.

    Args:
        series: Price series.

    Returns:
        Annualized return as a decimal.
    """
    if series.empty or len(series) < 2:
        return 0.0

    total_return = series.iloc[-1] / series.iloc[0]
    years = (series.index[-1] - series.index[0]).days / 365.25

    if years <= 0:
        return 0.0

    return total_return ** (1 / years) - 1


def get_dividend_summary(tickers: list[str]) -> pd.DataFrame:
    """Get a summary of dividend history for multiple tickers.

    Returns:
        DataFrame with columns: ticker, total_dividends, avg_annual_dividend,
        latest_dividend, latest_dividend_date.
    """
    rows = []
    for ticker in tickers:
        divs = fetch_dividends(ticker)
        if divs.empty:
            rows.append({
                "ticker": ticker,
                "total_dividends": 0,
                "dividend_count": 0,
                "avg_annual_dividend": 0,
                "latest_dividend": None,
                "latest_dividend_date": None,
            })
            continue

        years = (divs.index[-1] - divs.index[0]).days / 365.25
        avg_annual = divs.sum() / years if years > 0 else 0

        rows.append({
            "ticker": ticker,
            "total_dividends": divs.sum(),
            "dividend_count": len(divs),
            "avg_annual_dividend": avg_annual,
            "latest_dividend": divs.iloc[-1],
            "latest_dividend_date": divs.index[-1],
        })

    return pd.DataFrame(rows)


def get_split_summary(tickers: list[str]) -> pd.DataFrame:
    """Get a summary of stock split history for multiple tickers.

    Returns:
        DataFrame with columns: ticker, split_count, splits (list of
        date:ratio pairs).
    """
    rows = []
    for ticker in tickers:
        splits = fetch_splits(ticker)
        if splits.empty:
            rows.append({
                "ticker": ticker,
                "split_count": 0,
                "splits": [],
            })
            continue

        split_list = [
            {"date": d.strftime("%Y-%m-%d"), "ratio": r}
            for d, r in splits.items()
        ]

        rows.append({
            "ticker": ticker,
            "split_count": len(splits),
            "splits": split_list,
        })

    return pd.DataFrame(rows)
