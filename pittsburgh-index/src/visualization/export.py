"""Export charts to files with proper sizing and format."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from .style import BrandStyle

logger = logging.getLogger(__name__)


def save_chart(
    fig: plt.Figure,
    filename: str,
    output_dir: Path,
    style: BrandStyle,
    formats: list[str] | None = None,
) -> list[Path]:
    """Save a chart figure to disk.

    Args:
        fig: matplotlib Figure to save.
        filename: Base filename without extension (e.g., "hq_vs_benchmarks").
        output_dir: Directory to save to.
        style: BrandStyle for DPI and format settings.
        formats: List of formats to save. Defaults to the config setting.
            Supported: "png", "svg", "both".

    Returns:
        List of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = style.get_dpi()

    if formats is None:
        fmt = style.get_output_format()
        if fmt == "both":
            formats = ["png", "svg"]
        else:
            formats = [fmt]

    saved = []
    for fmt in formats:
        path = output_dir / f"{filename}.{fmt}"
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        logger.info("Saved chart: %s", path)
        saved.append(path)

    plt.close(fig)
    return saved


def save_all_charts(
    charts: dict[str, plt.Figure],
    output_dir: Path,
    style: BrandStyle,
) -> dict[str, list[Path]]:
    """Save multiple charts.

    Args:
        charts: Dict of filename -> Figure.
        output_dir: Directory to save to.
        style: BrandStyle for DPI and format settings.

    Returns:
        Dict of filename -> list of saved paths.
    """
    results = {}
    for filename, fig in charts.items():
        paths = save_chart(fig, filename, output_dir, style)
        results[filename] = paths
    return results
