# Current risk-engine validation

Untouched evaluation period: 2024 through the latest mature outcome.
Confidence intervals use moving blocks of trading dates and preserve
the cross-section within each date.

## Results

| Scope | Horizon | HAR QLIKE | Naïve QLIKE | HAR gain | DM p | VaR-95 breaches (95% CI) | Band coverage (95% CI) | ES ratio | Risk-level ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 5d | 0.583 | 0.635 | 8.3% | 0.00345 | 4.6% [3.6%, 6.0%] | 95.4% [94.4%, 96.1%] | 0.91 | 0.322 |
| all | 20d | 0.287 | 0.407 | 29.4% | 1.64e-05 | 4.4% [2.7%, 6.2%] | 95.4% [93.7%, 96.6%] | 0.85 | 0.327 |
| original_research | 5d | 0.666 | 0.719 | 7.4% | 0.0274 | 5.4% [4.3%, 6.7%] | 93.7% [92.6%, 94.6%] | 0.92 | 0.284 |
| original_research | 20d | 0.291 | 0.460 | 36.7% | 2.22e-16 | 5.3% [3.4%, 7.5%] | 93.0% [90.3%, 95.1%] | 0.86 | 0.290 |
| external_generalization | 5d | 0.491 | 0.542 | 9.5% | 0.00626 | 3.7% [2.6%, 5.3%] | 97.3% [96.2%, 98.2%] | 0.91 | 0.200 |
| external_generalization | 20d | 0.283 | 0.348 | 18.6% | 0.183 | 3.3% [1.6%, 5.5%] | 98.0% [96.8%, 98.9%] | 0.80 | 0.201 |
| current_portfolio | 5d | 0.618 | 0.626 | 1.3% | 0.829 | 4.1% [2.7%, 5.7%] | 95.6% [94.3%, 96.8%] | 0.93 | 0.092 |
| current_portfolio | 20d | 0.416 | 0.473 | 12.0% | 0.509 | 4.0% [2.0%, 6.4%] | 95.4% [93.2%, 97.2%] | 0.82 | 0.066 |

## Current conclusion

- Across all symbols, HAR improves QLIKE over naïve volatility by 8.3% at 5 days and 29.4% at 20 days (DM p=0.00345 and 1.64e-05).
- Pooled VaR-95 breach rates are 4.6% and 4.4%; both block intervals contain the 5% target.
- The original research assets' 5-day band covers 93.7%, below target, while external assets cover 97.3%, indicating that the same band is conservative outside the development universe.
- For the current portfolio, HAR point-forecast improvements are not statistically distinguishable from naïve volatility (DM p=0.829 and 0.509), although VaR/band calibration remains close to target.
- Risk level ranks future volatility well in the broad pooled panel,
  but should be interpreted cautiously for the six-symbol current portfolio.

## Interpretation rules

- HAR is better than naïve when QLIKE is lower, the relative gain is
  positive, and the date-aggregated DM statistic is negative.
- A calibrated VaR-95 process should have a breach interval containing 5%.
- A calibrated 95% band should have a coverage interval containing 95%.
- ES ratio should be near 1; below 1 means the predicted tail is conservative.
- Risk-level correlation should be positive and risk-decile realized
  volatility should generally increase monotonically.

Daily outcomes overlap. The block-aware intervals in this report should
be used instead of IID binomial tests on the raw row count.