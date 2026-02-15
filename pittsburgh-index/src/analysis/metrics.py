"""Performance metrics for index analysis.

Computes standard financial performance metrics: CAGR, volatility, Sharpe
ratio, maximum drawdown, and more.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def cagr(series: pd.Series) -> float:
    """Compound Annual Growth Rate.

    Args:
        series: Price/index level series.

    Returns:
        CAGR as a decimal (e.g., 0.10 = 10% annual return).
    """
    if series.empty or len(series) < 2:
        return 0.0

    total_return = series.iloc[-1] / series.iloc[0]
    years = (series.index[-1] - series.index[0]).days / 365.25

    if years <= 0:
        return 0.0

    return total_return ** (1 / years) - 1


def annualized_volatility(series: pd.Series) -> float:
    """Annualized volatility (standard deviation of daily returns).

    Args:
        series: Price/index level series.

    Returns:
        Annualized volatility as a decimal.
    """
    daily_returns = series.pct_change().dropna()
    if daily_returns.empty:
        return 0.0
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def sharpe_ratio(
    series: pd.Series,
    risk_free_rate: float = 0.04,
) -> float:
    """Annualized Sharpe ratio.

    Args:
        series: Price/index level series.
        risk_free_rate: Annual risk-free rate (default 4%).

    Returns:
        Sharpe ratio.
    """
    ann_return = cagr(series)
    ann_vol = annualized_volatility(series)

    if ann_vol == 0:
        return 0.0

    return (ann_return - risk_free_rate) / ann_vol


def max_drawdown(series: pd.Series) -> float:
    """Maximum drawdown (largest peak-to-trough decline).

    Args:
        series: Price/index level series.

    Returns:
        Maximum drawdown as a negative decimal (e.g., -0.30 = -30%).
    """
    if series.empty:
        return 0.0

    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return drawdown.min()


def drawdown_series(series: pd.Series) -> pd.Series:
    """Compute the drawdown series (peak-to-trough at each point).

    Args:
        series: Price/index level series.

    Returns:
        Series of drawdown values (all <= 0).
    """
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    dd.name = f"{series.name} Drawdown"
    return dd


def rolling_returns(
    series: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Compute rolling N-day returns.

    Args:
        series: Price/index level series.
        window: Rolling window in trading days (default 252 = ~1 year).

    Returns:
        Series of rolling returns.
    """
    rolling = series / series.shift(window) - 1
    rolling.name = f"{series.name} {window}d Rolling Return"
    return rolling


def summary_metrics(
    series: pd.Series,
    risk_free_rate: float = 0.04,
) -> dict:
    """Compute a summary of key performance metrics.

    Args:
        series: Price/index level series.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        Dict with metric name -> value.
    """
    return {
        "name": series.name,
        "start_date": series.index[0].strftime("%Y-%m-%d") if not series.empty else None,
        "end_date": series.index[-1].strftime("%Y-%m-%d") if not series.empty else None,
        "start_value": series.iloc[0] if not series.empty else None,
        "end_value": series.iloc[-1] if not series.empty else None,
        "total_return": (series.iloc[-1] / series.iloc[0] - 1) if len(series) >= 2 else 0,
        "cagr": cagr(series),
        "annualized_volatility": annualized_volatility(series),
        "sharpe_ratio": sharpe_ratio(series, risk_free_rate),
        "max_drawdown": max_drawdown(series),
    }


def comparison_summary(
    comparison_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """Compute summary metrics for all series in a comparison DataFrame.

    Args:
        comparison_df: DataFrame with one column per series.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        DataFrame with metrics as rows and series as columns.
    """
    metrics = {}
    for col in comparison_df.columns:
        series = comparison_df[col].dropna()
        if not series.empty:
            metrics[col] = summary_metrics(series, risk_free_rate)

    return pd.DataFrame(metrics).T
