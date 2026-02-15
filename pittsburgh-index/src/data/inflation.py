"""Inflation data fetching and CPI-based adjustment.

Uses the FRED API to fetch CPI-U (Consumer Price Index for All Urban Consumers).
Falls back to a bundled CSV or cached Parquet if FRED is unavailable.
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"


def fetch_cpi_from_fred_api(api_key: str) -> pd.Series:
    """Fetch CPI-U series from FRED using the fredapi library.

    Args:
        api_key: FRED API key.

    Returns:
        Monthly CPI-U series with DatetimeIndex.
    """
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        cpi = fred.get_series("CPIAUCSL")
        cpi.index = pd.to_datetime(cpi.index).tz_localize(None)
        cpi.name = "CPI"
        return cpi
    except ImportError:
        logger.warning("fredapi not installed, falling back to CSV download")
        raise
    except Exception as e:
        logger.error("FRED API error: %s", e)
        raise


def fetch_cpi_from_csv() -> pd.Series:
    """Fetch CPI-U from FRED's public CSV endpoint (no API key needed)."""
    try:
        df = pd.read_csv(FRED_CPI_URL, parse_dates=["DATE"], index_col="DATE")
        cpi = df["CPIAUCSL"]
        cpi.index.name = "Date"
        cpi.name = "CPI"
        return cpi
    except Exception as e:
        logger.error("Failed to fetch CPI CSV from FRED: %s", e)
        raise


def fetch_cpi(cache_dir: Path) -> pd.Series:
    """Fetch CPI data, using cache to avoid redundant downloads.

    Tries in order:
    1. Load from cache if fresh
    2. FRED API (if FRED_API_KEY env var is set)
    3. FRED public CSV endpoint (no key needed)

    Args:
        cache_dir: Directory for the CPI cache file.

    Returns:
        Monthly CPI-U series.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "cpi.parquet"

    # Try cache first
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cpi = cached.iloc[:, 0]
        cpi.index = pd.to_datetime(cpi.index)
        last_date = cpi.index.max()

        # CPI is monthly; if we have data within the last 45 days, it's fresh
        if (pd.Timestamp.now() - last_date).days < 45:
            logger.info("Using cached CPI data (last: %s)", last_date.date())
            return cpi

    # Fetch fresh data
    cpi = None
    api_key = os.environ.get("FRED_API_KEY")

    if api_key:
        try:
            cpi = fetch_cpi_from_fred_api(api_key)
        except Exception:
            pass

    if cpi is None:
        try:
            cpi = fetch_cpi_from_csv()
        except Exception:
            if cache_path.exists():
                logger.warning("Using stale cached CPI data")
                cached = pd.read_parquet(cache_path)
                return cached.iloc[:, 0]
            raise RuntimeError("Cannot fetch CPI data and no cache exists")

    # Save to cache
    cpi.to_frame().to_parquet(cache_path)
    logger.info("Cached CPI data: %d months", len(cpi))
    return cpi


def interpolate_cpi_daily(cpi_monthly: pd.Series) -> pd.Series:
    """Interpolate monthly CPI to daily frequency using forward-fill.

    Args:
        cpi_monthly: Monthly CPI series.

    Returns:
        Daily CPI series (forward-filled from monthly values).
    """
    daily_index = pd.date_range(
        start=cpi_monthly.index.min(),
        end=cpi_monthly.index.max(),
        freq="D",
    )
    daily = cpi_monthly.reindex(daily_index).ffill()
    daily.name = "CPI"
    return daily


def adjust_for_inflation(
    series: pd.Series | pd.DataFrame,
    cpi_daily: pd.Series,
    base_date: str | pd.Timestamp,
) -> pd.Series | pd.DataFrame:
    """Convert nominal values to real (inflation-adjusted) values.

    Real value = Nominal value * (CPI at base_date / CPI at date)

    Args:
        series: Nominal price series or DataFrame of series.
        cpi_daily: Daily CPI values (from interpolate_cpi_daily).
        base_date: Reference date for "constant dollars".

    Returns:
        Inflation-adjusted series in base_date dollars.
    """
    base_date = pd.Timestamp(base_date)
    if base_date not in cpi_daily.index:
        # Find nearest available date
        base_date = cpi_daily.index[cpi_daily.index.get_indexer([base_date], method="nearest")[0]]

    base_cpi = cpi_daily.loc[base_date]

    # Align CPI with the series index
    aligned_cpi = cpi_daily.reindex(series.index, method="ffill")

    deflator = base_cpi / aligned_cpi

    if isinstance(series, pd.DataFrame):
        return series.multiply(deflator, axis=0)
    else:
        return series * deflator
