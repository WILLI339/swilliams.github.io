"""Market-cap and equal weighting schemes for index construction."""

import logging
from pathlib import Path

import pandas as pd

from .constituents import IndexDefinition
from ..data.fetcher import fetch_ticker_info

logger = logging.getLogger(__name__)


def compute_equal_weights(tickers: list[str]) -> dict[str, float]:
    """Compute equal weights for a list of tickers.

    Args:
        tickers: List of active ticker symbols.

    Returns:
        Dict mapping ticker -> weight (all equal, summing to 1.0).
    """
    n = len(tickers)
    if n == 0:
        return {}
    weight = 1.0 / n
    return {t: weight for t in tickers}


def compute_market_cap_weights(
    tickers: list[str],
    price_data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    shares_outstanding: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute market-cap weights for a list of tickers on a given date.

    If shares_outstanding data is not provided, falls back to equal weighting.

    Args:
        tickers: List of active ticker symbols.
        price_data: Dict of ticker -> OHLCV DataFrame.
        date: Date for which to compute weights.
        shares_outstanding: Dict of ticker -> shares outstanding. If None,
            will attempt to fetch from yfinance.

    Returns:
        Dict mapping ticker -> weight (summing to 1.0).
    """
    if not tickers:
        return {}

    market_caps = {}
    for ticker in tickers:
        if ticker not in price_data:
            continue

        df = price_data[ticker]
        # Find the closest available date
        available = df.index[df.index <= date]
        if available.empty:
            continue

        price = df.loc[available[-1], "Close"]

        shares = None
        if shares_outstanding and ticker in shares_outstanding:
            shares = shares_outstanding[ticker]

        if shares is None:
            # Try to fetch from yfinance
            info = fetch_ticker_info(ticker)
            shares = info.get("sharesOutstanding")

        if shares and shares > 0:
            market_caps[ticker] = price * shares
        else:
            logger.warning(
                "No shares outstanding for %s, will use equal weight fallback",
                ticker,
            )

    if not market_caps:
        logger.warning("No market cap data available, falling back to equal weights")
        return compute_equal_weights(tickers)

    # If some tickers are missing market cap, use equal weights for all
    if len(market_caps) < len(tickers):
        missing = set(tickers) - set(market_caps)
        logger.warning(
            "Missing market cap for %s, falling back to equal weights",
            ", ".join(missing),
        )
        return compute_equal_weights(tickers)

    total = sum(market_caps.values())
    return {t: mc / total for t, mc in market_caps.items()}


def compute_weights(
    index_def: IndexDefinition,
    price_data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    shares_outstanding: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute weights based on the index's configured weighting scheme.

    Args:
        index_def: The index definition with weighting scheme.
        price_data: Dict of ticker -> OHLCV DataFrame.
        date: Date for weight computation.
        shares_outstanding: Optional pre-fetched shares outstanding data.

    Returns:
        Dict mapping ticker -> weight.
    """
    active_tickers = index_def.active_tickers_on(date)
    # Filter to tickers we actually have data for
    active_tickers = [t for t in active_tickers if t in price_data]

    if not active_tickers:
        return {}

    if index_def.weighting == "equal":
        return compute_equal_weights(active_tickers)
    elif index_def.weighting == "market_cap":
        return compute_market_cap_weights(
            active_tickers, price_data, date, shares_outstanding,
        )
    else:
        logger.warning(
            "Unknown weighting scheme '%s', falling back to equal",
            index_def.weighting,
        )
        return compute_equal_weights(active_tickers)


def fetch_shares_outstanding(tickers: list[str]) -> dict[str, float]:
    """Fetch current shares outstanding for multiple tickers.

    Returns:
        Dict mapping ticker -> shares outstanding.
    """
    shares = {}
    for ticker in tickers:
        info = fetch_ticker_info(ticker)
        so = info.get("sharesOutstanding")
        if so:
            shares[ticker] = so
            logger.debug("%s: %,.0f shares outstanding", ticker, so)
        else:
            logger.warning("No shares outstanding data for %s", ticker)
    return shares
