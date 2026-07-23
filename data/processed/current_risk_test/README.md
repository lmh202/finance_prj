# Current risk-engine test set

This directory extends the original 2021–2023 held-out evaluation with
untouched 2024–present market data and a broader 40-symbol universe.

## Universe

- 21 original research symbols;
- 6 symbols in the current saved portfolio;
- 20 stable broad-market, sector, bond, real-estate, and commodity ETFs.

The groups overlap. Stable ETF tickers are used instead of today's index
constituents to avoid reconstructing earlier years with current constituents.

## Files

| File | Purpose |
|---|---|
| `universe.csv` | Symbols, group membership, and ETF roles |
| `ohlc.parquet` | Adjusted OHLC downloaded from Yahoo Finance |
| `risk_backtest_panel.parquet` | Daily HAR forecasts, VaR/ES/bands, realized outcomes, breach labels, and maturity flags |
| `risk_backtest_summary.csv` | Descriptive risk results by universe group |
| `current_validation_summary.csv` | Block-aware validation against naïve volatility |
| `risk_level_calibration.csv` | Forecast and realized volatility by risk-level decile |
| `current_validation_report.md` | Human-readable validation result |
| `rss_headlines.json` | Deduplicated raw RSS archive with publication and first-fetch timestamps |
| `rss_event_observations.parquet` | Prospective RSS snapshots joined to the latest pre-event risk forecast |
| `manifest.json` | Build timestamp, sources, coverage, failures, and caveats |

## RSS interpretation

RSS has no historical archive. The first collection is a saturated feed-window
baseline and has `eligible_for_rate_analysis = false`. It must not be treated as
a historical news-rate observation.

Subsequent runs append incremental observations. Stories count as fresh only
when first fetched within 72 hours of their publication timestamp. If any feed
fails during a run, that snapshot is not eligible for arrival-rate analysis.
Forward 5/20-day returns remain `pending` until enough market sessions have
elapsed and become `mature` on a later refresh.

## Refresh

From the repository root:

```powershell
python scripts/build_current_risk_testset.py
python scripts/validate_current_risk_testset.py
```

Use `--no-rss` on the builder when only market-data refresh is wanted.

## Statistical caution

Daily 5- and 20-day outcomes overlap, and symbols share market shocks. Use the
date-level moving-block intervals in `current_validation_summary.csv`, not IID
binomial tests based on the raw number of rows.
