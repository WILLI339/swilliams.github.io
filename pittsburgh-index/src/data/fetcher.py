"""Fetch stock price data from Yahoo Finance with retry logic."""

import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_ticker_history(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> pd.DataFrame:
    """Download historical OHLCV data for a single ticker.

    Uses yfinance with auto_adjust=True so that all prices are adjusted for
    splits and dividends (i.e., the Close column represents total-return-adjusted
    prices).

    Args:
        ticker: Stock ticker symbol.
        start: Start date (YYYY-MM-DD). If None, fetches maximum history.
        end: End date (YYYY-MM-DD). If None, fetches through today.
        max_retries: Number of retry attempts on failure.
        backoff_seconds: Base delay between retries (doubles each attempt).

    Returns:
        DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume.
        Returns empty DataFrame if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            kwargs = {"auto_adjust": True, "progress": False}
            if start:
                kwargs["start"] = start
            else:
                kwargs["period"] = "max"
            if end:
                kwargs["end"] = end

            data = yf.download(ticker, **kwargs)

            if data.empty:
                logger.warning("No data returned for %s", ticker)
                return pd.DataFrame()

            # yf.download may return MultiIndex columns for single ticker;
            # flatten if needed
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Ensure index is DatetimeIndex with no timezone
            data.index = pd.to_datetime(data.index).tz_localize(None)
            data.index.name = "Date"

            logger.info(
                "Fetched %d rows for %s (%s to %s)",
                len(data), ticker,
                data.index.min().strftime("%Y-%m-%d"),
                data.index.max().strftime("%Y-%m-%d"),
            )
            return data

        except Exception as e:
            wait = backoff_seconds * (2 ** attempt)
            logger.warning(
                "Attempt %d/%d failed for %s: %s. Retrying in %.1fs",
                attempt + 1, max_retries, ticker, e, wait,
            )
            time.sleep(wait)

    logger.error("All %d attempts failed for %s", max_retries, ticker)
    return pd.DataFrame()


def fetch_ticker_info(ticker: str) -> dict:
    """Fetch metadata for a ticker (shares outstanding, market cap, etc.)."""
    try:
        info = yf.Ticker(ticker).info
        return info
    except Exception as e:
        logger.warning("Failed to fetch info for %s: %s", ticker, e)
        return {}


def fetch_dividends(ticker: str) -> pd.Series:
    """Fetch dividend history for a ticker."""
    try:
        dividends = yf.Ticker(ticker).dividends
        if not dividends.empty:
            dividends.index = pd.to_datetime(dividends.index).tz_localize(None)
        return dividends
    except Exception as e:
        logger.warning("Failed to fetch dividends for %s: %s", ticker, e)
        return pd.Series(dtype=float)


def fetch_splits(ticker: str) -> pd.Series:
    """Fetch stock split history for a ticker."""
    try:
        splits = yf.Ticker(ticker).splits
        if not splits.empty:
            splits.index = pd.to_datetime(splits.index).tz_localize(None)
        return splits
    except Exception as e:
        logger.warning("Failed to fetch splits for %s: %s", ticker, e)
        return pd.Series(dtype=float)
