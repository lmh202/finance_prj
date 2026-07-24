# Risk Engine volatility optimisation

Primary target: 5-session realised volatility; secondary target: 20-session.
All reported model choices are nested, annual walk-forward, and embargoed.

## Leaderboard

| h | model | mean QLIKE | gain vs HAR | positive years | worst year |
|---:|---|---:|---:|---:|---:|
| 5 | linear_gamma | 0.64859 | +7.79% | 5/6 | -0.81% |
| 5 | xgb_price | 0.64954 | +6.79% | 4/6 | -5.44% |
| 5 | har_news_linear_deployable | 0.68139 | +4.80% | 5/6 | -1.02% |
| 5 | har_news_linear_research | 0.68139 | +4.80% | 5/6 | -1.02% |
| 5 | ewma | 0.70470 | -1.73% | 2/6 | -12.08% |
| 5 | current_frozen_har | 0.72288 | -0.73% | 3/6 | -5.73% |
| 5 | har | 0.72299 | +0.00% | 0/6 | +0.00% |
| 5 | ridge_har | 0.72299 | -0.00% | 1/6 | -0.00% |
| 5 | har_news_xgb_deployable | 0.72387 | -0.05% | 1/6 | -0.27% |
| 5 | har_news_xgb_research | 0.72387 | -0.05% | 1/6 | -0.27% |
| 5 | naive_rv22 | 0.80326 | -14.77% | 0/6 | -28.41% |
| 20 | current_frozen_har | 0.47732 | +4.63% | 4/6 | -0.84% |
| 20 | har_news_linear_deployable | 0.49550 | +1.68% | 4/6 | -10.82% |
| 20 | har_news_linear_research | 0.49550 | +1.68% | 4/6 | -10.82% |
| 20 | ridge_har | 0.51990 | +0.00% | 4/6 | -0.01% |
| 20 | har | 0.51994 | +0.00% | 0/6 | +0.00% |
| 20 | har_news_xgb_deployable | 0.52041 | -0.13% | 3/6 | -0.76% |
| 20 | har_news_xgb_research | 0.52041 | -0.13% | 3/6 | -0.76% |
| 20 | linear_gamma | 0.55091 | -3.99% | 2/6 | -24.06% |
| 20 | xgb_price | 0.56682 | -10.89% | 2/6 | -40.48% |
| 20 | ewma | 0.57142 | -22.35% | 0/6 | -33.95% |
| 20 | naive_rv22 | 0.73796 | -55.57% | 0/6 | -72.43% |

### Legacy GARCH controls (reference only)

These controls use the earlier 2021–2023 fixed test split, so they are not mixed into the nested promotion ranking.

| h | model | QLIKE | log-RMSE |
|---:|---|---:|---:|
| 5 | GARCH(1,1) | 0.46319 | 0.51630 |
| 5 | EGARCH(1,1,1) | 0.46401 | 0.51090 |
| 20 | GARCH(1,1) | 0.23623 | 0.33327 |
| 20 | EGARCH(1,1,1) | 0.22236 | 0.32466 |

## Promotion gates

- **price_5d `linear_gamma` vs `current_frozen_har`**: gain +10.44%, DM p=0.0148, bootstrap [+3.07%, +18.92%] — **HOLD**
  - Failed: var95_calibrated, band_calibrated, external_price_not_worse
- **news_5d `har_news_linear_deployable` vs `current_frozen_har`**: gain +5.24%, DM p=0.00109, bootstrap [+2.38%, +8.83%] — **HOLD**
  - Failed: rss_shadow_60_mature
- **model_20d `har_news_linear_deployable` vs `current_frozen_har`**: gain -3.89%, DM p=0.135, bootstrap [-8.43%, -0.61%] — **HOLD**
  - Failed: aggregate_not_worse_1pct, significantly_better_if_promoted, tail_not_worse, rss_shadow_60_mature

## External 2024+ price generalisation

- 5d `all`: +1.66% QLIKE gain (40 symbols, 42,507 rows)
- 5d `original_research`: +13.30% QLIKE gain (21 symbols, 22,470 rows)
- 5d `external_generalization`: -13.08% QLIKE gain (19 symbols, 20,037 rows)
- 20d `all`: -4.44% QLIKE gain (40 symbols, 41,907 rows)
- 20d `original_research`: -3.51% QLIKE gain (21 symbols, 22,155 rows)
- 20d `external_generalization`: -5.42% QLIKE gain (19 symbols, 19,752 rows)

## Positive news features

A feature is listed only when removing it worsened inner-fold QLIKE in
at least 60% of folds with positive median contribution.

- 5d `deployable` `log_count`: 6/6 folds, median drop-column gain +0.177%
- 5d `research` `log_count`: 6/6 folds, median drop-column gain +0.177%
- 20d `deployable` `has_news`: 5/6 folds, median drop-column gain +0.168%
- 20d `research` `has_news`: 5/6 folds, median drop-column gain +0.168%

## Guardrails

- `coverage_active` is excluded because its upper endpoint is non-causal.
- Missing/stale RSS never blocks the price forecast; the news multiplier is 1.
- 2021-2023 is confirmation evidence, not a pristine untouched test.
- The FNSPID-derived news model remains research/non-commercial.