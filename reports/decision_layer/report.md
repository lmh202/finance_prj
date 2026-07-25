# Final Decision Layer Validation

## Decision

- Selected return model: **xgb_medium**
- Validation-selected return-signal scale: **0.030**
- Evaluated decision strategy: **xgb_medium_scaled**
- Formal production mode: **risk_only**
- Promotion checks passed: **9/11**

The return model predicts the next-20-session stock return relative to SPY.
The formal HAR-X + News five-session volatility forecast controls position
sizing and the portfolio covariance.  DeepSeek is not allowed to alter the
numeric decision; it is an explanation-only layer.

## Evaluation protocol

- Pretraining history: 2013-2017
- Walk-forward model selection and signal calibration: 2018-2020
- Locked historical test: 2021-2023
- External transfer test: 2024-2026, including previously unseen symbols
- Leakage control: 20 trading sessions between training labels and each fold
- Portfolio stress: 10, 25, and 50 bps one-way transaction costs

## Predictive results

| period | model | rank_ic_mean | top_bottom_20d_spread | r2 |
| --- | --- | --- | --- | --- |
| validation | elastic_net_a0005 | 0.0506 | 0.0162 | 0.0014 |
| validation | elastic_net_a002 | 0.0618 | 0.0185 | 0.004 |
| validation | xgb_shallow | 0.0641 | 0.0173 | -0.0052 |
| validation | xgb_medium | 0.0674 | 0.0122 | -0.0155 |
| validation | quantile_hgb | 0.0604 | 0.0115 | -0.0138 |
| validation_deployed_signal | xgb_medium_scaled | 0.0674 | 0.0122 | -0.0101 |
| locked_test | xgb_medium | 0.01 | 0.011 | -0.003 |
| external | xgb_medium | 0.0379 | 0.02 | -0.0009 |

## 2018-2020 validation portfolio results (25 bps)

| strategy | certainty_equivalent | sharpe | max_drawdown | annual_turnover | total_transaction_cost |
| --- | --- | --- | --- | --- | --- |
| elastic_net_a0005 | -0.2179 | 0.5644 | -0.3766 | 4.5628 | 0.0342 |
| elastic_net_a002 | -0.2292 | 0.5284 | -0.379 | 4.1751 | 0.0313 |
| xgb_shallow | -0.1277 | 0.746 | -0.338 | 5.4185 | 0.0406 |
| xgb_medium | -0.0604 | 0.9189 | -0.3159 | 6.381 | 0.0479 |
| quantile_hgb | -0.2222 | 0.5601 | -0.4066 | 7.1921 | 0.0539 |
| equal_weight | 0.0463 | 1.0708 | -0.2952 | 0.0 | 0.0 |
| risk_only | 0.0396 | 0.8858 | -0.2396 | 0.6989 | 0.0052 |
| momentum_rule | -0.1249 | 0.678 | -0.4405 | 7.5767 | 0.0568 |
| xgb_medium_scaled | 0.0552 | 0.9421 | -0.2303 | 0.7207 | 0.0054 |

## 2021-2023 locked historical test (25 bps)

| strategy | certainty_equivalent | sharpe | max_drawdown | annual_turnover | total_transaction_cost |
| --- | --- | --- | --- | --- | --- |
| equal_weight | -0.0455 | 0.6292 | -0.3709 | 0.0 | 0.0 |
| risk_only | 0.0336 | 0.6987 | -0.2404 | 0.8087 | 0.0059 |
| momentum_rule | -0.0412 | 0.4095 | -0.3606 | 7.4319 | 0.054 |
| xgb_medium_scaled | 0.0455 | 0.7717 | -0.2434 | 0.9495 | 0.0069 |

## 2024-2026 external test (25 bps)

| strategy | certainty_equivalent | sharpe | max_drawdown | annual_turnover | total_transaction_cost |
| --- | --- | --- | --- | --- | --- |
| equal_weight | 0.1699 | 1.529 | -0.1766 | 0.0 | 0.0 |
| risk_only | 0.0925 | 1.3644 | -0.0995 | 0.2418 | 0.0015 |
| momentum_rule | 0.2142 | 1.67 | -0.2016 | 11.4629 | 0.0705 |
| xgb_medium_scaled | 0.0864 | 1.2808 | -0.0984 | 0.2612 | 0.0016 |

## Promotion gates

```json
{
  "promoted": false,
  "production_mode": "risk_only",
  "selected_model": "xgb_medium",
  "selected_strategy": "xgb_medium_scaled",
  "return_signal_scale": 0.03,
  "checks": {
    "locked_test_cer_above_risk_only": true,
    "locked_test_sharpe_above_risk_only": true,
    "positive_test_years_at_least_2_of_3": true,
    "bootstrap_utility_gain_lower_bound_positive": false,
    "max_drawdown_not_worse_by_more_than_2pp": true,
    "es_not_worse_by_more_than_10pct": true,
    "external_cer_not_below_risk_only": false,
    "external_unseen_cer_not_below_risk_only": true,
    "max_position_constraint": true,
    "max_change_constraint": true,
    "minimum_trade_constraint": true
  },
  "statistics": {
    "bootstrap_utility_gain_95": [
      -0.008003754054736219,
      0.011338541442004066,
      0.03352288752798527
    ],
    "positive_test_years": 2,
    "locked_test_cer_gain": 0.011900458702099728,
    "locked_test_sharpe_gain": 0.07307043381206701,
    "external_cer_gain": -0.0061137147316027984,
    "external_unseen_cer_gain": 0.0002018682316877185
  }
}
```

## Interpretation

Average direction accuracy is not the promotion target.  A candidate is useful
only when its ranking survives transaction costs and improves realised
risk-adjusted utility over the risk-only allocation.  If the ML gates fail,
the checkpoint retains the selected model for research while the backend uses
the deterministic risk-only optimiser.  The selected shrinkage shows that a
small return tilt may be useful, but it remains disabled until both statistical
uncertainty and external transfer gates pass.
