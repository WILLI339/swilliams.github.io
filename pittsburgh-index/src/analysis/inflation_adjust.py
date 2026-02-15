"""Inflation adjustment wrapper for the analysis pipeline.

Provides a clean interface to apply inflation adjustment to comparison
DataFrames and individual series.
"""

import logging
from pathlib import Path

import pandas as pd

from ..data.inflation import fetch_cpi, interpolate_cpi_daily, adjust_for_inflation

logger = logging.getLogger(__name__)


class InflationAdjuster:
    """Manages CPI data loading and provides inflation adjustment methods."""

    def __init__(self, cache_dir: Path):
        """Initialize with CPI data.

        Args:
            cache_dir: Directory for CPI cache file.
        """
        self._cpi_monthly = fetch_cpi(cache_dir)
        self._cpi_daily = interpolate_cpi_daily(self._cpi_monthly)

    @property
    def cpi_daily(self) -> pd.Series:
        """Daily CPI series."""
        return self._cpi_daily

    @property
    def latest_cpi_date(self) -> pd.Timestamp:
        """Most recent date with CPI data."""
        return self._cpi_monthly.index.max()

    def adjust_series(
        self,
        series: pd.Series,
        base_date: str | pd.Timestamp,
    ) -> pd.Series:
        """Adjust a single series for inflation.

        Args:
            series: Nominal price series.
            base_date: Reference date for constant dollars.

        Returns:
            Inflation-adjusted series.
        """
        adjusted = adjust_for_inflation(series, self._cpi_daily, base_date)
        if isinstance(adjusted, pd.Series) and series.name:
            adjusted.name = f"{series.name} (Real)"
        return adjusted

    def adjust_dataframe(
        self,
        df: pd.DataFrame,
        base_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Adjust all columns in a DataFrame for inflation.

        Args:
            df: DataFrame with nominal price series as columns.
            base_date: Reference date for constant dollars.

        Returns:
            Inflation-adjusted DataFrame with "(Real)" suffix on column names.
        """
        adjusted = adjust_for_inflation(df, self._cpi_daily, base_date)
        adjusted.columns = [f"{col} (Real)" for col in adjusted.columns]
        return adjusted
