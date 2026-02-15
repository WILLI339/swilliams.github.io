"""Pipeline orchestrator.

Runs the full data pipeline: fetch -> construct -> analyze -> visualize.
Each stage can also be run independently via the CLI.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from .data.cache import fetch_all_with_cache, check_cache_freshness
from .index.constituents import (
    load_index_definitions,
    load_benchmarks,
    get_all_tickers,
)
from .index.construction import build_all_indexes, normalize_to_base
from .index.weighting import fetch_shares_outstanding
from .analysis.compare import build_comparison_table
from .analysis.metrics import comparison_summary
from .analysis.inflation_adjust import InflationAdjuster
from .visualization.style import BrandStyle
from .visualization.charts import (
    chart_index_vs_benchmarks,
    chart_indexes_head_to_head,
    chart_drawdowns,
    chart_rolling_returns,
    chart_sector_breakdown,
    chart_performance_table,
)
from .visualization.export import save_all_charts

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the full Pittsburgh Stock Index pipeline."""

    def __init__(self, project_dir: Path):
        """Initialize the pipeline.

        Args:
            project_dir: Root directory of the pittsburgh-index project.
        """
        self.project_dir = project_dir
        self.config_dir = project_dir / "config"

        # Load settings
        settings_path = self.config_dir / "settings.yaml"
        with open(settings_path) as f:
            self.settings = yaml.safe_load(f)

        data_settings = self.settings.get("data", {})
        self.cache_dir = project_dir / data_settings.get("cache_dir", "data/raw")
        self.processed_dir = project_dir / data_settings.get("processed_dir", "data/processed")
        self.inflation_dir = project_dir / data_settings.get("inflation_dir", "data/inflation")
        self.output_dir = project_dir / data_settings.get("output_dir", "output/charts")

        self.max_retries = data_settings.get("max_retries", 3)
        self.backoff = data_settings.get("retry_backoff_seconds", 2)

        # These get populated during pipeline stages
        self.indexes = None
        self.benchmarks = None
        self.price_data = None
        self.index_series = None
        self.comparison_df = None

    def load_config(self) -> None:
        """Load index and benchmark definitions from config files."""
        logger.info("Loading configuration...")
        self.indexes = load_index_definitions(self.config_dir / "indexes.yaml")
        self.benchmarks = load_benchmarks(self.config_dir / "benchmarks.yaml")
        logger.info(
            "Loaded %d indexes and %d benchmarks",
            len(self.indexes), len(self.benchmarks),
        )

    def fetch_data(self) -> dict[str, pd.DataFrame]:
        """Fetch/update price data for all tickers with caching.

        Returns:
            Dict of ticker -> DataFrame.
        """
        if self.indexes is None:
            self.load_config()

        all_tickers = get_all_tickers(self.indexes, self.benchmarks)
        logger.info("Fetching data for %d tickers...", len(all_tickers))

        # Check freshness first
        freshness = check_cache_freshness(
            all_tickers, self.cache_dir,
            self.settings.get("data", {}).get("stale_warning_days", 7),
        )
        stale = [t for t, s in freshness.items() if s == "stale"]
        missing = [t for t, s in freshness.items() if s == "missing"]
        if stale:
            logger.info("Stale cache for: %s", ", ".join(stale))
        if missing:
            logger.info("No cache for: %s", ", ".join(missing))

        self.price_data = fetch_all_with_cache(
            all_tickers, self.cache_dir, self.max_retries, self.backoff,
        )

        logger.info("Fetched data for %d/%d tickers", len(self.price_data), len(all_tickers))
        return self.price_data

    def build_indexes(self) -> dict[str, pd.Series]:
        """Construct index time series from constituent data.

        Returns:
            Dict of index key -> index level Series.
        """
        if self.price_data is None:
            self.fetch_data()

        logger.info("Building indexes...")

        # Fetch shares outstanding for market-cap weighting
        all_tickers = []
        for idx_def in self.indexes.values():
            if idx_def.weighting == "market_cap":
                all_tickers.extend(idx_def.all_tickers)
        all_tickers = list(set(all_tickers))

        shares = {}
        if all_tickers:
            logger.info("Fetching shares outstanding for %d tickers...", len(all_tickers))
            shares = fetch_shares_outstanding(all_tickers)

        self.index_series = build_all_indexes(self.indexes, self.price_data, shares)

        # Save processed index data
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for key, series in self.index_series.items():
            path = self.processed_dir / f"{key}.parquet"
            series.to_frame().to_parquet(path)
            logger.info("Saved processed index: %s", path)

        return self.index_series

    def build_comparison(self, base_date: str | None = None) -> pd.DataFrame:
        """Build comparison table of all indexes and benchmarks.

        Args:
            base_date: Override base date for normalization. Defaults to
                the earliest index base date.

        Returns:
            Normalized comparison DataFrame.
        """
        if self.index_series is None:
            self.build_indexes()

        if base_date is None:
            base_date = min(idx.base_date for idx in self.indexes.values())

        # Normalize benchmark ETFs
        benchmark_series = {}
        for bm in self.benchmarks:
            ticker = bm["ticker"]
            if ticker in self.price_data:
                close = self.price_data[ticker]["Close"]
                close.name = bm["name"]
                benchmark_series[bm["name"]] = close

        # Build named index series dict
        named_indexes = {}
        for key, series in self.index_series.items():
            named_indexes[series.name] = series

        self.comparison_df = build_comparison_table(
            named_indexes, benchmark_series, base_date,
        )

        # Save comparison data
        path = self.processed_dir / "comparison.parquet"
        self.comparison_df.to_parquet(path)
        logger.info("Saved comparison table: %s", path)

        return self.comparison_df

    def generate_charts(
        self,
        include_inflation: bool = False,
    ) -> dict[str, list[Path]]:
        """Generate all charts and save to output directory.

        Args:
            include_inflation: If True, also generate inflation-adjusted charts.

        Returns:
            Dict of chart name -> list of saved file paths.
        """
        if self.comparison_df is None:
            self.build_comparison()

        logger.info("Generating charts...")
        style = BrandStyle(self.config_dir / "branding.yaml")
        style.apply_matplotlib_defaults()

        charts = {}
        index_keys = list(self.indexes.keys())
        benchmark_names = [bm["name"] for bm in self.benchmarks]
        base_date = min(idx.base_date for idx in self.indexes.values())

        # 1. Each index vs benchmarks
        for key in index_keys:
            if key not in self.index_series:
                continue

            idx_def = self.indexes[key]
            idx_series = self.comparison_df.get(idx_def.name)
            if idx_series is None:
                continue

            bm_series = {
                bm["name"]: self.comparison_df[bm["name"]]
                for bm in self.benchmarks
                if bm["name"] in self.comparison_df.columns
            }

            fig = chart_index_vs_benchmarks(
                idx_series, key, bm_series, self.benchmarks, style,
            )
            charts[f"{key}_vs_benchmarks"] = fig

        # 2. Head-to-head: HQ vs Operations
        if len(index_keys) >= 2:
            named_indexes = {}
            for key in index_keys:
                if key in self.index_series:
                    name = self.indexes[key].name
                    if name in self.comparison_df.columns:
                        named_indexes[key] = self.comparison_df[name]
                        named_indexes[key].name = name

            fig = chart_indexes_head_to_head(named_indexes, index_keys, style)
            charts["hq_vs_ops"] = fig

        # 3. Drawdowns
        fig = chart_drawdowns(
            self.comparison_df,
            [self.indexes[k].name for k in index_keys if k in self.indexes],
            style,
            include_benchmarks=["S&P 500"],
        )
        charts["drawdowns"] = fig

        # 4. Rolling 12-month returns
        fig = chart_rolling_returns(
            self.comparison_df,
            [self.indexes[k].name for k in index_keys if k in self.indexes],
            style,
            window=252,
            include_benchmarks=["S&P 500"],
        )
        charts["rolling_12m_returns"] = fig

        # 5. Sector breakdown for each index
        for key in index_keys:
            if key not in self.indexes:
                continue
            idx_def = self.indexes[key]
            active = idx_def.active_constituents_on(pd.Timestamp.now())
            fig = chart_sector_breakdown(active, idx_def.name, style)
            charts[f"{key}_sectors"] = fig

        # 6. Performance summary table
        metrics = comparison_summary(self.comparison_df)
        fig = chart_performance_table(metrics, style)
        charts["performance_summary"] = fig

        # 7. Inflation-adjusted charts
        if include_inflation:
            try:
                adjuster = InflationAdjuster(self.inflation_dir)
                real_df = adjuster.adjust_dataframe(self.comparison_df, base_date)

                # Re-normalize real values to base=100
                for col in real_df.columns:
                    first_valid = real_df[col].first_valid_index()
                    if first_valid is not None:
                        base_val = real_df.loc[first_valid, col]
                        if base_val != 0:
                            real_df[col] = real_df[col] * (100 / base_val)

                # Real vs nominal for HQ index
                for key in index_keys:
                    if key not in self.indexes:
                        continue
                    name = self.indexes[key].name
                    real_name = f"{name} (Real)"
                    if name in self.comparison_df.columns and real_name in real_df.columns:
                        import matplotlib.pyplot as plt
                        fig, ax = style.create_figure(f"{name}: Nominal vs Real")
                        ax.plot(
                            self.comparison_df.index,
                            self.comparison_df[name].values,
                            color=style.get_index_color(key),
                            label=f"{name} (Nominal)",
                            **style.get_index_line_style(key),
                        )
                        ax.plot(
                            real_df.index, real_df[real_name].values,
                            color=style.get_index_color(key),
                            label=real_name,
                            linestyle=":",
                            linewidth=2.0,
                            alpha=0.8,
                        )
                        ax.set_ylabel("Index Level (Base = 100)")
                        ax.legend()
                        plt.tight_layout()
                        charts[f"{key}_inflation_adjusted"] = fig

            except Exception as e:
                logger.error("Failed to generate inflation-adjusted charts: %s", e)

        # Save all charts
        saved = save_all_charts(charts, self.output_dir, style)
        logger.info("Generated %d charts", len(saved))
        return saved

    def run(self, include_inflation: bool = False) -> dict[str, list[Path]]:
        """Run the full pipeline end-to-end.

        Args:
            include_inflation: Include inflation-adjusted charts.

        Returns:
            Dict of chart name -> saved file paths.
        """
        logger.info("Running full pipeline...")
        self.load_config()
        self.fetch_data()
        self.build_indexes()
        self.build_comparison()
        return self.generate_charts(include_inflation)
