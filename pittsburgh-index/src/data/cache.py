"""Cache management for historical price data.

Stores one Parquet file per ticker. On subsequent runs, only fetches data
after the last cached date to avoid redundant API calls.
"""

import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .fetcher import fetch_ticker_history

logger = logging.getLogger(__name__)


def get_cache_path(ticker: str, cache_dir: Path) -> Path:
    """Return the Parquet file path for a given ticker."""
    return cache_dir / f"{ticker}.parquet"


def load_cached(ticker: str, cache_dir: Path) -> pd.DataFrame | None:
    """Load cached data for a ticker, or None if no cache exists."""
    path = get_cache_path(ticker, cache_dir)
    if path.exists():
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        logger.debug("Loaded %d cached rows for %s", len(df), ticker)
        return df
    return None


def save_cache(df: pd.DataFrame, ticker: str, cache_dir: Path) -> None:
    """Save a DataFrame to the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = get_cache_path(ticker, cache_dir)
    df.to_parquet(path)
    logger.debug("Saved %d rows to cache for %s", len(df), ticker)


def fetch_with_cache(
    ticker: str,
    cache_dir: Path,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> pd.DataFrame:
    """Fetch data for a ticker, using cache to avoid re-downloading history.

    On first call: downloads full history and caches it.
    On subsequent calls: loads cache, fetches only new data since last cached
    date, merges, and updates the cache.

    Args:
        ticker: Stock ticker symbol.
        cache_dir: Directory for Parquet cache files.
        max_retries: Retries for API calls.
        backoff_seconds: Base backoff delay.

    Returns:
        DataFrame with full historical data (cached + new).
    """
    cached = load_cached(ticker, cache_dir)

    if cached is not None and not cached.empty:
        last_date = cached.index.max()
        start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        today = pd.Timestamp.now().normalize()

        # If cache is already up to date (last trading day), skip fetch
        if last_date >= today - timedelta(days=3):
            logger.info("Cache for %s is recent (last: %s), skipping fetch", ticker, last_date.date())
            return cached

        logger.info("Fetching new data for %s from %s", ticker, start)
        new_data = fetch_ticker_history(
            ticker, start=start,
            max_retries=max_retries, backoff_seconds=backoff_seconds,
        )

        if new_data.empty:
            logger.info("No new data for %s since %s", ticker, last_date.date())
            return cached

        combined = pd.concat([cached, new_data])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        save_cache(combined, ticker, cache_dir)
        logger.info(
            "Updated cache for %s: %d -> %d rows",
            ticker, len(cached), len(combined),
        )
        return combined

    else:
        logger.info("No cache for %s, fetching full history", ticker)
        data = fetch_ticker_history(
            ticker,
            max_retries=max_retries, backoff_seconds=backoff_seconds,
        )
        if not data.empty:
            save_cache(data, ticker, cache_dir)
        return data


def fetch_all_with_cache(
    tickers: list[str],
    cache_dir: Path,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Fetch data for multiple tickers with caching.

    Returns:
        Dict mapping ticker -> DataFrame.
    """
    results = {}
    for ticker in tickers:
        df = fetch_with_cache(ticker, cache_dir, max_retries, backoff_seconds)
        if not df.empty:
            results[ticker] = df
        else:
            logger.warning("No data available for %s, skipping", ticker)
    return results


def check_cache_freshness(
    tickers: list[str],
    cache_dir: Path,
    stale_days: int = 7,
) -> dict[str, str]:
    """Check how fresh cached data is for each ticker.

    Returns:
        Dict mapping ticker -> status string ("fresh", "stale", "missing").
    """
    status = {}
    cutoff = pd.Timestamp.now().normalize() - timedelta(days=stale_days)

    for ticker in tickers:
        cached = load_cached(ticker, cache_dir)
        if cached is None or cached.empty:
            status[ticker] = "missing"
        elif cached.index.max() < cutoff:
            status[ticker] = "stale"
        else:
            status[ticker] = "fresh"

    return status
