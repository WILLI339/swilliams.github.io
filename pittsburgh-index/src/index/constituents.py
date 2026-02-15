"""Load and validate index constituent definitions from YAML config."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Constituent:
    """A single index constituent (stock)."""
    ticker: str
    name: str
    sector: str
    added: pd.Timestamp
    removed: pd.Timestamp | None = None
    notes: str = ""

    def is_active_on(self, date: pd.Timestamp) -> bool:
        """Check if this constituent is active (in the index) on a given date."""
        if date < self.added:
            return False
        if self.removed is not None and date >= self.removed:
            return False
        return True


@dataclass
class IndexDefinition:
    """Complete definition of a Pittsburgh index."""
    name: str
    description: str
    base_date: pd.Timestamp
    base_value: float
    weighting: str
    rebalance_frequency: str
    inclusion_criteria: dict
    constituents: list[Constituent] = field(default_factory=list)

    @property
    def all_tickers(self) -> list[str]:
        """All unique tickers across all constituents (active and historical)."""
        return list({c.ticker for c in self.constituents})

    def active_constituents_on(self, date: pd.Timestamp) -> list[Constituent]:
        """Return constituents that are active on a given date."""
        return [c for c in self.constituents if c.is_active_on(date)]

    def active_tickers_on(self, date: pd.Timestamp) -> list[str]:
        """Return tickers of active constituents on a given date."""
        return [c.ticker for c in self.active_constituents_on(date)]


def _parse_constituent(entry: dict) -> Constituent:
    """Parse a single constituent entry from YAML."""
    return Constituent(
        ticker=entry["ticker"],
        name=entry["name"],
        sector=entry["sector"],
        added=pd.Timestamp(entry["added"]),
        removed=pd.Timestamp(entry["removed"]) if entry.get("removed") else None,
        notes=entry.get("notes", ""),
    )


def load_index_definitions(config_path: Path) -> dict[str, IndexDefinition]:
    """Load all index definitions from the indexes.yaml config file.

    Handles the `include_hq_index` flag for the Operations Index, which
    includes all HQ Index constituents automatically.

    Args:
        config_path: Path to indexes.yaml.

    Returns:
        Dict mapping index key (e.g., "pgh_hq_index") to IndexDefinition.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    indexes = {}

    for key, idx_config in config.items():
        constituents = [_parse_constituent(c) for c in idx_config.get("constituents", [])]

        indexes[key] = IndexDefinition(
            name=idx_config["name"],
            description=idx_config["description"],
            base_date=pd.Timestamp(idx_config["base_date"]),
            base_value=idx_config.get("base_value", 100),
            weighting=idx_config.get("weighting", "market_cap"),
            rebalance_frequency=idx_config.get("rebalance_frequency", "quarterly"),
            inclusion_criteria=idx_config.get("inclusion_criteria", {}),
            constituents=constituents,
        )

    # Handle include_hq_index: merge HQ constituents into ops index
    if "pgh_ops_index" in indexes and "pgh_hq_index" in indexes:
        ops_config = config.get("pgh_ops_index", {})
        if ops_config.get("include_hq_index", False):
            hq_tickers = {c.ticker for c in indexes["pgh_hq_index"].constituents}
            ops_tickers = {c.ticker for c in indexes["pgh_ops_index"].constituents}

            for c in indexes["pgh_hq_index"].constituents:
                if c.ticker not in ops_tickers:
                    indexes["pgh_ops_index"].constituents.append(c)

            logger.info(
                "Merged %d HQ constituents into Operations Index",
                len(hq_tickers - ops_tickers),
            )

    for key, idx in indexes.items():
        logger.info(
            "Loaded index '%s' with %d constituents",
            idx.name, len(idx.constituents),
        )

    return indexes


def load_benchmarks(config_path: Path) -> list[dict]:
    """Load benchmark definitions from benchmarks.yaml.

    Returns:
        List of benchmark dicts with ticker, name, description, category.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("benchmarks", [])


def get_all_tickers(
    indexes: dict[str, IndexDefinition],
    benchmarks: list[dict],
) -> list[str]:
    """Get a deduplicated list of all tickers needed (constituents + benchmarks)."""
    tickers = set()
    for idx in indexes.values():
        tickers.update(idx.all_tickers)
    for bm in benchmarks:
        tickers.add(bm["ticker"])
    return sorted(tickers)
