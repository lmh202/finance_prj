# Multi-task MLP Decision-Layer Exploration

## Decision

- Validation-selected family: **price_risk**
- Network: **small [64, 32]**
- Alpha calibration: **1.00**
- Worth continuing under all predeclared checks: **False**
- Formal checkpoint replaced: **no**

The MLP directly predicts the next-20-session risk-adjusted return relative to
SPY. HAR-X + News risk is fixed and enters both the target normalization and
the portfolio optimiser.

## Feature-family comparison

| family | config | alpha_scale | validation_cer_gain | test_rank_ic | test_cer_gain | test_positive_years | bootstrap_low | q10_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| price_risk | small | 1.0 | 0.1745 | 0.0481 | -0.0542 | 2 | -0.2061 | 0.1131 |
| deployable_news | medium | 0.02 | 0.0036 | 0.006 | -0.0016 | 1 | -0.0107 | 0.0967 |
| research_news | small | 1.0 | 0.0746 | 0.0476 | -0.011 | 2 | -0.1468 | 0.1043 |

## News increment

```json
{
  "deployable_news": {
    "validation_cer_gain_vs_price_risk_mlp": -0.17091871233089886,
    "test_cer_gain_vs_price_risk_mlp": 0.052543477399001934,
    "test_rank_ic_gain_vs_price_risk_mlp": -0.042092574734811955
  },
  "research_news": {
    "validation_cer_gain_vs_price_risk_mlp": -0.09986991576182552,
    "test_cer_gain_vs_price_risk_mlp": 0.04318035408236698,
    "test_rank_ic_gain_vs_price_risk_mlp": -0.000445998071359692
  }
}
```

## Continuation checks

```json
{
  "validation_gain_positive": true,
  "test_cer_gain_positive": false,
  "test_rank_ic_positive": true,
  "test_positive_years_at_least_2": true,
  "bootstrap_lower_bound_positive": false,
  "q10_coverage_between_7_and_13pct": true
}
```

## External price-risk diagnostic

Only the price-risk family can be reconstructed on 2024-2026 because no
historical RSS feature archive exists. It reaches CER 0.1847 versus 0.0925 for
risk-only, but annual turnover rises from 0.24 to 8.83 and maximum drawdown
worsens from -9.9% to -21.4%. Combined with its negative 2021-2023 CER gain,
this is regime-dependent performance rather than reliable transfer.

## Material findings

1. The price-risk MLP strongly overfits the two-year validation regime:
   validation CER gain is +0.1745, while the locked test gain is -0.0542.
2. Deployable news produces the safest MLP, but its test CER gain is still
   -0.0016 and its test rank IC is only 0.0060.
3. The full 40-field research-news block does not improve test rank IC over
   price-risk inputs (-0.0004 incremental IC), indicating that the sample is
   too small for unconstrained feature interaction learning.
4. The conditional q10 head is reasonably calibrated across all three families
   (9.7%-11.3% realised coverage), so distributional downside learning is more
   promising than direct alpha prediction.

## Interpretation

Model architecture, feature family, and alpha calibration are selected only
from 2019-2020 walk-forward validation. The 2021-2023 result is a locked
diagnostic, but not a pristine blind test because those years were already
observed by earlier experiments. Rich news cannot be evaluated on 2024-2026
because no historical RSS feature archive exists.
