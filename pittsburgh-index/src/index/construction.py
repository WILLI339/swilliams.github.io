"""Build index time series from constituent data and weights.

The index is constructed as a daily return-weighted series, normalized to a
base value (default 100) on the base date. This follows the same methodology
used by major indexes like the S&P 500.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from .constituents import IndexDefinition
from .weighting import compute_weights

logger = logging.getLogger(__name__)


def _get_rebalance_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
) -> list[pd.Timestamp]:
    """Generate rebalance dates for the given frequency.

    Args:
        start: First rebalance date.
        end: Last possible rebalance date.
        frequency: "quarterly" or "annual".

    Returns:
        List of rebalance dates.
    """
    if frequency == "annual":
        dates = pd.date_range(start=start, end=end, freq="YS")
    else:  # quarterly
        dates = pd.date_range(start=start, end=end, freq="QS")
    return [d for d in dates if d >= start]


def build_price_index(
    index_def: IndexDefinition,
    price_data: dict[str, pd.DataFrame],
    shares_outstanding: dict[str, float] | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.Series:
    """Build the price index time series.

    Uses adjusted close prices (which account for splits and dividends when
    auto_adjust=True from yfinance). The resulting series represents total
    return performance.

    Algorithm:
    1. Determine the trading date range from base_date to end_date.
    2. Generate rebalance dates per the index's frequency.
    3. On each trading day, compute the weighted return from constituents.
    4. Chain daily returns to build the index level series.

    Args:
        index_def: Index definition with constituents and config.
        price_data: Dict of ticker -> DataFrame with 'Close' column.
        shares_outstanding: Optional shares outstanding for market-cap weighting.
        end_date: End date for the index. Defaults to the latest available data.

    Returns:
        Series with DatetimeIndex and index level values, starting at base_value.
    """
    base_date = index_def.base_date

    if end_date is None:
        # Use the latest date across all constituent data
        latest_dates = [df.index.max() for df in price_data.values() if not df.empty]
        if not latest_dates:
            raise ValueError("No price data available for any constituent")
        end_date = max(latest_dates)

    # Build a common date index from all constituent data
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df.index)
    trading_days = sorted([d for d in all_dates if base_date <= d <= end_date])

    if not trading_days:
        raise ValueError(f"No trading days between {base_date} and {end_date}")

    # Build a close price matrix: rows = trading days, cols = tickers
    all_tickers = index_def.all_tickers
    close_matrix = pd.DataFrame(index=trading_days, columns=all_tickers, dtype=float)

    for ticker in all_tickers:
        if ticker in price_data:
            df = price_data[ticker]
            close_col = df["Close"].reindex(trading_days)
            close_matrix[ticker] = close_col

    # Forward-fill prices (for days a stock didn't trade, carry forward)
    close_matrix = close_matrix.ffill()

    # Compute daily returns
    returns_matrix = close_matrix.pct_change()

    # Get rebalance dates
    rebalance_dates = _get_rebalance_dates(
        base_date, end_date, index_def.rebalance_frequency,
    )

    # Build the index
    index_values = [index_def.base_value]
    current_weights = {}

    for i, date in enumerate(trading_days):
        # Rebalance weights on rebalance dates or on the first day
        if i == 0 or date in rebalance_dates:
            current_weights = compute_weights(
                index_def, price_data, date, shares_outstanding,
            )
            if not current_weights:
                logger.warning("No weights computed for %s, using previous", date)

        if i == 0:
            continue

        # Compute weighted return for this day
        daily_return = 0.0
        weight_sum = 0.0

        for ticker, weight in current_weights.items():
            ret = returns_matrix.loc[date, ticker]
            if pd.notna(ret):
                daily_return += weight * ret
                weight_sum += weight

        # Normalize if not all weights contributed (missing data)
        if weight_sum > 0 and weight_sum < 0.99:
            daily_return = daily_return / weight_sum

        new_value = index_values[-1] * (1 + daily_return)
        index_values.append(new_value)

    index_series = pd.Series(
        index_values,
        index=pd.DatetimeIndex(trading_days),
        name=index_def.name,
    )

    logger.info(
        "Built index '%s': %s to %s, %d days, %.2f -> %.2f",
        index_def.name,
        trading_days[0].date(),
        trading_days[-1].date(),
        len(trading_days),
        index_values[0],
        index_values[-1],
    )

    return index_series


def build_all_indexes(
    indexes: dict[str, IndexDefinition],
    price_data: dict[str, pd.DataFrame],
    shares_outstanding: dict[str, float] | None = None,
) -> dict[str, pd.Series]:
    """Build time series for all defined indexes.

    Args:
        indexes: Dict of index key -> IndexDefinition.
        price_data: Dict of ticker -> DataFrame.
        shares_outstanding: Optional shares outstanding data.

    Returns:
        Dict of index key -> index level Series.
    """
    results = {}
    for key, index_def in indexes.items():
        try:
            series = build_price_index(index_def, price_data, shares_outstanding)
            results[key] = series
        except Exception as e:
            logger.error("Failed to build index '%s': %s", key, e)
    return results


def normalize_to_base(
    series: pd.Series,
    base_date: pd.Timestamp | str,
    base_value: float = 100,
) -> pd.Series:
    """Normalize a price series so it equals base_value on the base_date.

    Useful for comparing benchmark ETFs against the Pittsburgh indexes on
    the same scale.

    Args:
        series: Price or index level series.
        base_date: Date to set as base_value.
        base_value: Value at base_date (default 100).

    Returns:
        Normalized series.
    """
    base_date = pd.Timestamp(base_date)

    # Find the nearest available date
    available = series.index[series.index >= base_date]
    if available.empty:
        raise ValueError(f"No data on or after {base_date}")
    actual_base_date = available[0]

    base_price = series.loc[actual_base_date]
    if base_price == 0:
        raise ValueError(f"Base price is zero on {actual_base_date}")

    normalized = series * (base_value / base_price)
    normalized.name = series.name
    return normalized
