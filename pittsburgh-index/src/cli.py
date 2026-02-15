"""Command-line interface for the Pittsburgh Stock Index pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import Pipeline


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="pittsburgh-index",
        description="Pittsburgh Stock Index — data pipeline and chart generator",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Project root directory (default: auto-detected)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    # fetch
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch/update price data for all tickers (uses cache)",
    )

    # build
    build_parser = subparsers.add_parser(
        "build",
        help="Fetch data and build index time series",
    )

    # compare
    compare_parser = subparsers.add_parser(
        "compare",
        help="Build comparison table and compute metrics",
    )

    # charts
    charts_parser = subparsers.add_parser(
        "charts",
        help="Run full pipeline and generate charts",
    )
    charts_parser.add_argument(
        "--inflation",
        action="store_true",
        help="Include inflation-adjusted charts",
    )

    # status
    status_parser = subparsers.add_parser(
        "status",
        help="Check cache freshness for all tickers",
    )

    # run (full pipeline)
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full pipeline (fetch + build + compare + charts)",
    )
    run_parser.add_argument(
        "--inflation",
        action="store_true",
        help="Include inflation-adjusted charts",
    )

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return 0

    pipeline = Pipeline(args.project_dir)

    try:
        if args.command == "fetch":
            pipeline.load_config()
            pipeline.fetch_data()
            print("Data fetch complete.")

        elif args.command == "build":
            pipeline.load_config()
            pipeline.fetch_data()
            index_series = pipeline.build_indexes()
            for key, series in index_series.items():
                print(f"  {series.name}: {series.iloc[0]:.2f} -> {series.iloc[-1]:.2f}")

        elif args.command == "compare":
            pipeline.load_config()
            pipeline.fetch_data()
            pipeline.build_indexes()
            comparison = pipeline.build_comparison()
            print(f"Comparison table: {len(comparison)} days, {len(comparison.columns)} series")

            from .analysis.metrics import comparison_summary
            metrics = comparison_summary(comparison)
            print("\nPerformance Summary:")
            print(metrics[["cagr", "annualized_volatility", "sharpe_ratio", "max_drawdown"]].to_string(
                float_format=lambda x: f"{x:.2%}" if abs(x) < 100 else f"{x:.2f}"
            ))

        elif args.command == "charts":
            saved = pipeline.run(include_inflation=args.inflation)
            print(f"\nGenerated {len(saved)} charts:")
            for name, paths in saved.items():
                for p in paths:
                    print(f"  {p}")

        elif args.command == "status":
            pipeline.load_config()
            from .data.cache import check_cache_freshness
            from .index.constituents import get_all_tickers
            tickers = get_all_tickers(pipeline.indexes, pipeline.benchmarks)
            freshness = check_cache_freshness(tickers, pipeline.cache_dir)

            for status_val in ["missing", "stale", "fresh"]:
                tickers_with_status = [t for t, s in freshness.items() if s == status_val]
                if tickers_with_status:
                    print(f"\n{status_val.upper()}: {', '.join(sorted(tickers_with_status))}")

        elif args.command == "run":
            saved = pipeline.run(include_inflation=args.inflation)
            print(f"\nFull pipeline complete. Generated {len(saved)} charts:")
            for name, paths in saved.items():
                for p in paths:
                    print(f"  {p}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    except Exception as e:
        logging.getLogger(__name__).error("Pipeline failed: %s", e, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
