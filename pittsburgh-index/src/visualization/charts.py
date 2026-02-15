"""Chart generation functions.

Each function produces one chart type. All styling is controlled by the
BrandStyle instance — no colors or fonts are hardcoded here.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

from .style import BrandStyle
from ..analysis.metrics import drawdown_series, rolling_returns, summary_metrics

logger = logging.getLogger(__name__)


def _format_axes(ax: plt.Axes, style: BrandStyle) -> None:
    """Apply common axis formatting."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter(
        style.chart_config.get("date_format", "%b %Y")
    ))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.legend(loc=style.chart_config.get("legend_location", "upper left"))
    plt.tight_layout()


def chart_index_vs_benchmarks(
    index_series: pd.Series,
    index_key: str,
    benchmarks: dict[str, pd.Series],
    benchmark_configs: list[dict],
    style: BrandStyle,
    title_suffix: str = "",
) -> plt.Figure:
    """Chart a Pittsburgh index against benchmark indexes.

    Args:
        index_series: The Pittsburgh index time series (normalized).
        index_key: Key name of the index (e.g., "pgh_hq_index").
        benchmarks: Dict of benchmark name -> normalized Series.
        benchmark_configs: List of benchmark config dicts (with ticker, name).
        style: BrandStyle instance.
        title_suffix: Optional suffix for the chart title.

    Returns:
        matplotlib Figure.
    """
    title = f"{index_series.name} vs Benchmarks"
    if title_suffix:
        title += f" ({title_suffix})"

    fig, ax = style.create_figure(title)

    # Plot the Pittsburgh index
    ax.plot(
        index_series.index, index_series.values,
        color=style.get_index_color(index_key),
        label=index_series.name,
        **style.get_index_line_style(index_key),
    )

    # Plot benchmarks
    bm_style = style.get_benchmark_line_style()
    ticker_to_name = {bm["ticker"]: bm["name"] for bm in benchmark_configs}

    for ticker, series in benchmarks.items():
        label = ticker_to_name.get(ticker, ticker)
        ax.plot(
            series.index, series.values,
            color=style.get_benchmark_color(ticker),
            label=label,
            **bm_style,
        )

    ax.set_ylabel("Index Level (Base = 100)")
    ax.axhline(y=100, color=style.colors.get("grid", "#E0E0E0"), linewidth=0.5)
    _format_axes(ax, style)

    return fig


def chart_indexes_head_to_head(
    indexes: dict[str, pd.Series],
    index_keys: list[str],
    style: BrandStyle,
    title: str = "Pittsburgh HQ vs Operations Index",
) -> plt.Figure:
    """Chart two Pittsburgh indexes against each other.

    Args:
        indexes: Dict of index key -> normalized Series.
        index_keys: List of index keys to include.
        style: BrandStyle instance.
        title: Chart title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = style.create_figure(title)

    for key in index_keys:
        if key not in indexes:
            continue
        series = indexes[key]
        ax.plot(
            series.index, series.values,
            color=style.get_index_color(key),
            label=series.name,
            **style.get_index_line_style(key),
        )

    ax.set_ylabel("Index Level (Base = 100)")
    ax.axhline(y=100, color=style.colors.get("grid", "#E0E0E0"), linewidth=0.5)
    _format_axes(ax, style)

    return fig


def chart_drawdowns(
    comparison_df: pd.DataFrame,
    index_keys: list[str],
    style: BrandStyle,
    include_benchmarks: list[str] | None = None,
) -> plt.Figure:
    """Chart drawdowns for indexes and optionally benchmarks.

    Args:
        comparison_df: Normalized comparison DataFrame.
        index_keys: Pittsburgh index column names.
        style: BrandStyle instance.
        include_benchmarks: Optional list of benchmark column names to include.

    Returns:
        matplotlib Figure.
    """
    fig, ax = style.create_figure("Drawdowns")

    columns = list(index_keys)
    if include_benchmarks:
        columns.extend(include_benchmarks)

    for col in columns:
        if col not in comparison_df.columns:
            continue

        dd = drawdown_series(comparison_df[col])

        if col in index_keys:
            color = style.get_index_color(col)
            line_kw = style.get_index_line_style(col)
        else:
            color = style.get_benchmark_color(col)
            line_kw = style.get_benchmark_line_style()

        ax.fill_between(
            dd.index, dd.values, 0,
            alpha=0.3, color=color, label=col,
        )
        ax.plot(dd.index, dd.values, color=color, **line_kw)

    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    _format_axes(ax, style)

    return fig


def chart_rolling_returns(
    comparison_df: pd.DataFrame,
    index_keys: list[str],
    style: BrandStyle,
    window: int = 252,
    include_benchmarks: list[str] | None = None,
) -> plt.Figure:
    """Chart rolling N-day returns.

    Args:
        comparison_df: Normalized comparison DataFrame.
        index_keys: Pittsburgh index column names.
        style: BrandStyle instance.
        window: Rolling window in trading days.
        include_benchmarks: Optional benchmark columns to include.

    Returns:
        matplotlib Figure.
    """
    period_label = f"{window // 252}Y" if window >= 252 else f"{window}D"
    fig, ax = style.create_figure(f"Rolling {period_label} Returns")

    columns = list(index_keys)
    if include_benchmarks:
        columns.extend(include_benchmarks)

    for col in columns:
        if col not in comparison_df.columns:
            continue

        rr = rolling_returns(comparison_df[col], window)

        if col in index_keys:
            color = style.get_index_color(col)
            line_kw = style.get_index_line_style(col)
        else:
            color = style.get_benchmark_color(col)
            line_kw = style.get_benchmark_line_style()

        ax.plot(rr.index, rr.values, color=color, label=col, **line_kw)

    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.axhline(y=0, color=style.colors.get("grid", "#E0E0E0"), linewidth=1)
    _format_axes(ax, style)

    return fig


def chart_sector_breakdown(
    constituents: list,
    index_name: str,
    style: BrandStyle,
) -> plt.Figure:
    """Chart sector composition of an index as a horizontal bar chart.

    Args:
        constituents: List of Constituent objects.
        index_name: Name of the index for the title.
        style: BrandStyle instance.

    Returns:
        matplotlib Figure.
    """
    sector_counts = {}
    for c in constituents:
        if c.removed is None:  # Only active constituents
            sector_counts[c.sector] = sector_counts.get(c.sector, 0) + 1

    if not sector_counts:
        fig, ax = style.create_figure(f"{index_name} Sector Breakdown")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    sectors = sorted(sector_counts.keys())
    counts = [sector_counts[s] for s in sectors]

    fig, ax = style.create_figure(f"{index_name}: Sector Breakdown")

    colors = plt.cm.Set2(np.linspace(0, 1, len(sectors)))
    ax.barh(sectors, counts, color=colors)
    ax.set_xlabel("Number of Constituents")

    for i, count in enumerate(counts):
        ax.text(count + 0.1, i, str(count), va="center")

    plt.tight_layout()
    return fig


def chart_performance_table(
    metrics_df: pd.DataFrame,
    style: BrandStyle,
) -> plt.Figure:
    """Render a performance metrics summary as a table chart.

    Args:
        metrics_df: DataFrame from comparison_summary() with metrics as columns.
        style: BrandStyle instance.

    Returns:
        matplotlib Figure.
    """
    fig, ax = style.create_figure("Performance Summary")
    ax.axis("off")

    # Select and format key columns
    display_cols = ["cagr", "annualized_volatility", "sharpe_ratio", "max_drawdown", "total_return"]
    available = [c for c in display_cols if c in metrics_df.columns]

    if not available:
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    display_df = metrics_df[available].copy()

    # Format percentages
    for col in ["cagr", "annualized_volatility", "max_drawdown", "total_return"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    if "sharpe_ratio" in display_df.columns:
        display_df["sharpe_ratio"] = display_df["sharpe_ratio"].map(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )

    # Rename columns for display
    col_labels = {
        "cagr": "CAGR",
        "annualized_volatility": "Volatility",
        "sharpe_ratio": "Sharpe",
        "max_drawdown": "Max DD",
        "total_return": "Total Return",
    }
    display_df.columns = [col_labels.get(c, c) for c in display_df.columns]

    table = ax.table(
        cellText=display_df.values,
        rowLabels=display_df.index,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(style.fonts.get("tick_size", 10))
    table.scale(1, 1.5)

    # Style header row
    for j in range(len(display_df.columns)):
        table[0, j].set_facecolor(style.colors.get("grid", "#E0E0E0"))
        table[0, j].set_text_props(fontweight="bold")

    plt.tight_layout()
    return fig
