"""Load branding configuration and apply matplotlib styling.

All visual parameters are read from config/branding.yaml. No colors, fonts,
or sizes are hardcoded in chart code.
"""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml

logger = logging.getLogger(__name__)


class BrandStyle:
    """Manages branding configuration and matplotlib style application."""

    def __init__(self, config_path: Path):
        """Load branding config from YAML.

        Args:
            config_path: Path to branding.yaml.
        """
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        self.config = raw.get("branding", {})
        self.fonts = self.config.get("fonts", {})
        self.colors = self.config.get("colors", {})
        self.line_styles = self.config.get("line_styles", {})
        self.chart_config = self.config.get("chart", {})
        self.watermark_config = self.config.get("watermark", {})
        self.logo_config = self.config.get("logo", {})
        self.title_prefix = self.config.get("title_prefix", "Pittsburgh Stock Index")

    def apply_matplotlib_defaults(self) -> None:
        """Set matplotlib rcParams based on branding config."""
        mpl.use("Agg")  # Non-interactive backend

        params = {
            "font.family": self.fonts.get("family", "sans-serif"),
            "font.size": self.fonts.get("tick_size", 10),
            "axes.titlesize": self.fonts.get("title_size", 16),
            "axes.labelsize": self.fonts.get("label_size", 12),
            "legend.fontsize": self.fonts.get("legend_size", 10),
            "xtick.labelsize": self.fonts.get("tick_size", 10),
            "ytick.labelsize": self.fonts.get("tick_size", 10),
            "figure.figsize": (
                self.chart_config.get("figure_width", 12),
                self.chart_config.get("figure_height", 7),
            ),
            "figure.dpi": self.chart_config.get("dpi", 150),
            "axes.facecolor": self.colors.get("plot_background", "#FAFAFA"),
            "figure.facecolor": self.colors.get("background", "#FFFFFF"),
            "axes.grid": True,
            "grid.alpha": self.chart_config.get("grid_alpha", 0.3),
            "grid.color": self.colors.get("grid", "#E0E0E0"),
            "text.color": self.colors.get("text", "#333333"),
            "axes.labelcolor": self.colors.get("text", "#333333"),
            "xtick.color": self.colors.get("text", "#333333"),
            "ytick.color": self.colors.get("text", "#333333"),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
        mpl.rcParams.update(params)

    def get_index_color(self, index_key: str) -> str:
        """Get the color for a Pittsburgh index."""
        return self.colors.get(index_key, "#333333")

    def get_benchmark_color(self, ticker: str) -> str:
        """Get the color for a benchmark ticker."""
        bm_colors = self.colors.get("benchmarks", {})
        return bm_colors.get(ticker, "#999999")

    def get_index_line_style(self, index_key: str) -> dict:
        """Get line style kwargs for a Pittsburgh index."""
        style = self.line_styles.get(index_key, {})
        return {
            "linewidth": style.get("linewidth", 2.5),
            "linestyle": style.get("linestyle", "-"),
            "alpha": style.get("alpha", 1.0),
        }

    def get_benchmark_line_style(self) -> dict:
        """Get line style kwargs for benchmark series."""
        style = self.line_styles.get("benchmarks", {})
        return {
            "linewidth": style.get("linewidth", 1.5),
            "linestyle": style.get("linestyle", "--"),
            "alpha": style.get("alpha", 0.7),
        }

    def get_figure_size(self) -> tuple[float, float]:
        """Get figure size as (width, height) tuple."""
        return (
            self.chart_config.get("figure_width", 12),
            self.chart_config.get("figure_height", 7),
        )

    def get_dpi(self) -> int:
        """Get chart DPI."""
        return self.chart_config.get("dpi", 150)

    def get_output_format(self) -> str:
        """Get chart output format."""
        return self.chart_config.get("format", "png")

    def apply_watermark(self, ax: mpl.axes.Axes) -> None:
        """Apply watermark to an axes if configured."""
        text = self.watermark_config.get("text")
        if text:
            ax.text(
                0.5, 0.5, text,
                transform=ax.transAxes,
                fontsize=self.watermark_config.get("fontsize", 36),
                alpha=self.watermark_config.get("alpha", 0.15),
                ha="center", va="center",
                rotation=self.watermark_config.get("rotation", 30),
                color=self.colors.get("text", "#333333"),
            )

    def create_figure(self, title: str = "") -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Create a styled figure and axes.

        Args:
            title: Chart title (will be prefixed with title_prefix if set).

        Returns:
            Tuple of (Figure, Axes).
        """
        fig, ax = plt.subplots(figsize=self.get_figure_size())

        if title:
            full_title = f"{self.title_prefix}: {title}" if self.title_prefix else title
            ax.set_title(full_title, pad=15, fontweight="bold")

        self.apply_watermark(ax)
        return fig, ax
