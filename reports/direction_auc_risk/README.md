# Direction AUC: price, news, and HAR risk

## Question

Does the risk engine add out-of-sample information about whether the price will
be higher in 5 or 20 trading days?

## Protocol

- Classifier: the same deterministic `HistGradientBoostingClassifier` used by
  `scripts/train_ablation.py`.
- Training data: observations dated through 2020-12-02.
- Embargo: 30 calendar days before the test seam, preventing forward-return
  labels from crossing into 2021.
- Test data: 2021-01-01 through 2023-11-29, 11,712 observations.
- Risk features:
  - horizon-specific HAR forecast volatility (`risk_sigma_5d`/`20d`);
  - expanding historical percentile of the forecast (`risk_level_5d`/`20d`).
- Uncertainty: paired 20-trading-day moving-block bootstrap, preserving the
  cross-section within each resampled date.

The HAR coefficients were estimated before the direction test period. VaR, ES,
and interval width are not added separately because they are fixed multiples of
the same volatility forecast.

## Results

| Horizon | Risk only | Price | Price + news | Price + risk | Price + news + risk |
|---|---:|---:|---:|---:|---:|
| 5 days | 0.5049 | **0.5212** | 0.5154 | **0.5215** | 0.5200 |
| 20 days | 0.4894 | **0.5384** | 0.5393 | 0.5312 | 0.5324 |

Lift over price-only:

| Horizon | Addition | ΔAUC | 95% block-bootstrap interval |
|---|---|---:|---:|
| 5 days | news | -0.0058 | [-0.0158, +0.0066] |
| 5 days | risk | +0.0004 | [-0.0102, +0.0116] |
| 5 days | news + risk | -0.0012 | [-0.0127, +0.0119] |
| 20 days | news | +0.0009 | [-0.0074, +0.0110] |
| 20 days | risk | -0.0072 | [-0.0198, +0.0104] |
| 20 days | news + risk | -0.0060 | [-0.0194, +0.0122] |

## Conclusion

Risk does not add a statistically distinguishable direction signal. At five
days its measured lift is effectively zero; at 20 days the point estimate is
negative. All lift intervals include zero.

This is consistent with the risk-engine research: volatility magnitude is
predictable, but direction is not. The risk forecast is also strongly redundant
with the existing price-volatility feature (`corr(risk_sigma_5d, vol_20d) =
0.798`). It should therefore remain a position-sizing/risk-control input rather
than a long/short direction feature.
