"""Compare Pittsburgh indexes against benchmark indexes.

Normalizes all series to a common base date and value for fair comparison.
"""

import logging

import pandas as pd

from ..index.construction import normalize_to_base

logger = logging.getLogger(__name__)


def build_comparison_table(
    indexes: dict[str, pd.Series],
    benchmarks: dict[str, pd.Series],
    base_date: str | pd.Timestamp,
    base_value: float = 100,
) -> pd.DataFrame:
    """Build a single DataFrame with all indexes and benchmarks normalized.

    All series are normalized to start at base_value on base_date, enabling
    direct visual and numerical comparison.

    Args:
        indexes: Dict of index name -> index level Series.
        benchmarks: Dict of benchmark name -> price Series.
        base_date: Common base date for normalization.
        base_value: Common base value (default 100).

    Returns:
        DataFrame with DatetimeIndex and one column per series.
    """
    all_series = {}

    for name, series in {**indexes, **benchmarks}.items():
        try:
            normalized = normalize_to_base(series, base_date, base_value)
            all_series[name] = normalized
        except (ValueError, KeyError) as e:
            logger.warning("Could not normalize '%s': %s", name, e)

    if not all_series:
        raise ValueError("No series could be normalized")

    df = pd.DataFrame(all_series)
    df = df.sort_index()

    # Forward-fill missing dates (different trading calendars)
    df = df.ffill()

    logger.info(
        "Built comparison table: %d series, %d days (%s to %s)",
        len(df.columns), len(df),
        df.index.min().date(), df.index.max().date(),
    )

    return df


def compute_relative_performance(
    comparison_df: pd.DataFrame,
    target_col: str,
    benchmark_col: str,
) -> pd.Series:
    """Compute relative performance (target / benchmark) over time.

    A value > 1 means the target has outperformed the benchmark since base date.

    Args:
        comparison_df: Normalized comparison DataFrame.
        target_col: Column name for the target index.
        benchmark_col: Column name for the benchmark.

    Returns:
        Series of relative performance ratios.
    """
    relative = comparison_df[target_col] / comparison_df[benchmark_col]
    relative.name = f"{target_col} vs {benchmark_col}"
    return relative


def compute_rolling_correlation(
    comparison_df: pd.DataFrame,
    col_a: str,
    col_b: str,
    window: int = 252,
) -> pd.Series:
    """Compute rolling correlation between two series.

    Args:
        comparison_df: Comparison DataFrame.
        col_a: First series column name.
        col_b: Second series column name.
        window: Rolling window in trading days (default 252 = ~1 year).

    Returns:
        Rolling correlation Series.
    """
    returns_a = comparison_df[col_a].pct_change()
    returns_b = comparison_df[col_b].pct_change()
    corr = returns_a.rolling(window).corr(returns_b)
    corr.name = f"Correlation({col_a}, {col_b})"
    return corr


def slice_to_period(
    df: pd.DataFrame,
    years: int | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Slice a comparison DataFrame to a specific lookback period.

    Args:
        df: Full comparison DataFrame.
        years: Number of years to look back. None for full history.
        end_date: End date. Defaults to the last available date.

    Returns:
        Sliced DataFrame.
    """
    if end_date is None:
        end_date = df.index.max()

    if years is None:
        return df

    start_date = end_date - pd.DateOffset(years=years)
    sliced = df[df.index >= start_date]
    return sliced
