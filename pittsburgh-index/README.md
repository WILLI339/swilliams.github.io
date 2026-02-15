# Pittsburgh Stock Index

Custom stock market indexes tracking publicly traded companies with ties to Pittsburgh, PA.

## Indexes

### Pittsburgh HQ Index
Companies headquartered in the Pittsburgh Metropolitan Statistical Area (MSA):
- Must maintain principal executive offices in the Pittsburgh MSA (Allegheny, Westmoreland, Washington, Butler, Beaver, Fayette, Armstrong counties)
- Minimum $500M market cap
- Minimum 100,000 average daily trading volume
- Currently 18 constituents including PNC, PPG, Ansys, Wabtec, Dick's Sporting Goods, Duolingo, US Steel, and others

### Pittsburgh Operations Index
Companies with significant operational presence in the Pittsburgh MSA:
- Includes all HQ Index companies, plus
- Companies with a major office, R&D center, or facility (500+ employees) in the MSA
- Same market cap and volume minimums
- Adds companies like Google (Bakery Square R&D), Amazon, Microsoft, Uber, and others

## Benchmarks

Both indexes are compared against:
- **S&P 500** (SPY) — broad US large-cap
- **Total World Stock** (VT) — global equities
- **Total US Market** (VTI) — full US market
- **Russell 2000** (IWM) — small-cap (size-matched comparison)
- **Industrials Sector** (XLI) — sector-matched
- **Financials Sector** (XLF) — sector-matched
- **Materials Sector** (XLB) — sector-matched

## Methodology

- **Weighting**: Market-cap weighted (with equal-weight fallback)
- **Rebalancing**: Quarterly
- **Base date**: January 2, 2015 (base value = 100)
- **Prices**: Adjusted close (accounts for splits and dividends = total return)
- **Inflation**: Optional CPI-U adjustment via FRED data

## Setup

```bash
cd pittsburgh-index
pip install -e ".[inflation,dev]"
```

For inflation-adjusted charts, get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html and set:
```bash
export FRED_API_KEY=your_key_here
```

## Usage

```bash
# Check data freshness
python -m pittsburgh_index status

# Fetch/update data only
python -m pittsburgh_index fetch

# Build indexes and show values
python -m pittsburgh_index build

# Show comparison metrics
python -m pittsburgh_index compare

# Generate all charts
python -m pittsburgh_index charts

# Generate charts with inflation adjustment
python -m pittsburgh_index charts --inflation

# Full pipeline (fetch + build + compare + charts)
python -m pittsburgh_index run --inflation
```

## Configuration

All configuration is in YAML files under `config/`:

| File | Purpose |
|---|---|
| `indexes.yaml` | Index constituents, inclusion criteria, weighting |
| `benchmarks.yaml` | Benchmark ETFs for comparison |
| `branding.yaml` | Colors, fonts, chart sizes, watermark, logo |
| `settings.yaml` | Cache paths, retry logic, chart defaults |

### Updating Constituents

Edit `config/indexes.yaml` to add/remove companies. Each constituent has:
- `ticker`: Stock symbol
- `added`: Date entered the index
- `removed`: Date left the index (null = still active)
- `notes`: Context (e.g., IPO date, spin-off details)

### Changing Visual Style

Edit `config/branding.yaml`. No code changes needed. Controls:
- Colors for each index and benchmark
- Line widths, styles, opacity
- Font family and sizes
- Chart dimensions and DPI
- Watermark and logo

## Project Structure

```
pittsburgh-index/
├── config/           # YAML configuration
├── src/
│   ├── data/         # Fetching and caching (yfinance, FRED)
│   ├── index/        # Index construction (constituents, weighting)
│   ├── analysis/     # Comparison, metrics, inflation adjustment
│   └── visualization/  # Chart generation and export
├── data/             # Cached data (gitignored)
├── output/charts/    # Generated images (gitignored)
└── tests/
```
