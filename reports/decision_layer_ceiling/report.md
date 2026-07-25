# Fixed-Risk Decision-Layer Ceiling

## Locked specification

- Risk input: formal HAR-X + News five-session OOF risk
- Return model: **quantile_hgb**
- Signal: daily cross-sectional rank, amplitude **0.0025**
- Rebalance interval: **5 sessions**
- Turnover-penalty multiplier: **2.0**
- Diagnostic gates passed: **6/11**
- Automatic promotion: **disabled**

All model and optimiser choices were made on the 2018-2020 expanding
walk-forward validation period.  The 2021-2023 and 2024-2026 results were
computed only after the specification was locked.  They are labelled reused
diagnostics because prior experiments had already exposed those periods.

## Predictive ranking

| period | model | rank_ic_mean | top_bottom_20d_spread |
| --- | --- | --- | --- |
| validation | elastic_net_a0005 | 0.0506 | 0.0162 |
| validation | elastic_net_a002 | 0.0618 | 0.0185 |
| validation | xgb_shallow | 0.0641 | 0.0173 |
| validation | xgb_medium | 0.0674 | 0.0122 |
| validation | quantile_hgb | 0.0604 | 0.0115 |
| validation | xgb_rank_d2 | 0.0848 | 0.021 |
| validation | xgb_rank_d3 | 0.0869 | 0.0158 |
| locked_test_reused | quantile_hgb | 0.0097 | 0.0047 |
| external_reused_price_risk_only | quantile_hgb | 0.0496 | 0.0229 |

## Portfolio results at 25 bps

| period | strategy | certainty_equivalent | sharpe | max_drawdown | annual_turnover |
| --- | --- | --- | --- | --- | --- |
| validation | equal_weight | 0.0463 | 1.0708 | -0.2952 | 0.0 |
| validation | risk_only | 0.032 | 0.8585 | -0.2427 | 0.3559 |
| validation | quantile_hgb_rank_signal | 0.0554 | 0.9395 | -0.207 | 0.369 |
| locked_test_reused | risk_only | 0.0185 | 0.6101 | -0.2305 | 0.3353 |
| locked_test_reused | quantile_hgb_rank_signal | 0.017 | 0.6034 | -0.2522 | 0.4123 |
| external_reused | risk_only | 0.0916 | 1.216 | -0.1279 | 0.2007 |
| external_reused | quantile_hgb_rank_signal | 0.0828 | 1.1053 | -0.1191 | 0.1972 |

## Fair validation-tuned risk-only comparison

The risk-only optimiser was also tuned independently on validation. Its locked
configuration uses a 10-session rebalance and a 2.0 turnover-penalty
multiplier.

| Period | Candidate CER | Tuned risk-only CER | Candidate gain |
| --- | ---: | ---: | ---: |
| 2018-2020 validation | 0.0554 | 0.0477 | +0.0077 |
| 2021-2023 reused test | 0.0170 | 0.0201 | -0.0031 |
| 2024-2026 reused external | 0.0828 | 0.0875 | -0.0047 |

## Material problems found

1. The selected return model's daily rank IC falls from 0.0604 in validation
   to 0.0097 in 2021-2023. The return signal is therefore not stable enough.
2. Aggregate validation utility is fragile across calendar years. Most of the
   apparent benefit comes from particular regimes rather than a consistent
   annual improvement.
3. The candidate underperforms both the like-for-like risk-only comparator and
   the independently validation-tuned risk-only comparator on both reused
   holdouts.
4. The external panel has no historical RSS archive, so it cannot measure the
   complete live HAR-X + News decision path.

## Diagnostic gates

```json
{
  "candidate_passes_diagnostic_gates": false,
  "automatic_promotion_allowed": false,
  "reason_automatic_promotion_disabled": "The 2021-2026 holdouts were observed by earlier experiments and are no longer pristine blind tests.",
  "selected_model": "quantile_hgb",
  "selected_spec": {
    "name": "quantile_hgb",
    "kind": "quantile_hgb",
    "max_depth": 3,
    "learning_rate": 0.04,
    "max_iter": 300,
    "min_samples_leaf": 40,
    "l2_regularization": 2.0
  },
  "signal_transform": "daily_cross_sectional_rank_to_minus1_plus1",
  "alpha_strength": 0.0025,
  "rebalance_sessions": 5,
  "turnover_penalty_multiplier": 2.0,
  "fixed_risk_input": "formal HAR-X + News five-session OOF sigma for 2018-2023",
  "checks": {
    "validation_cer_above_risk_only": true,
    "locked_test_cer_above_risk_only": false,
    "locked_test_sharpe_above_risk_only": false,
    "positive_test_years_at_least_2_of_3": true,
    "bootstrap_lower_bound_positive": false,
    "locked_test_drawdown_not_worse_by_2pp": false,
    "external_cer_not_below_risk_only": false,
    "external_unseen_cer_not_below_risk_only": true,
    "max_position_constraint": true,
    "max_change_constraint": true,
    "minimum_trade_constraint": true
  },
  "statistics": {
    "bootstrap_utility_gain_95": [
      -0.02076311276492698,
      -0.0018754210050726076,
      0.015449875037556878
    ],
    "positive_test_years": 2,
    "locked_test_cer_gain": -0.0015489746192672016,
    "locked_test_sharpe_gain": -0.006707121807169147,
    "external_cer_gain": -0.008813573236070127,
    "external_unseen_cer_gain": 0.005668203967929122
  },
  "known_limitations": [
    "The 2024-2026 external risk panel is price-only because no historical RSS archive exists; it does not test the full live news path.",
    "The optimizer is fully invested and does not yet enforce sector, commodity, or cash constraints."
  ]
}
```

## Interpretation

This experiment estimates the practical ceiling of the current decision
architecture with the risk engine frozen.  A positive result does not authorize
deployment without new prospective data, because both historical holdouts have
already been observed.  The 2024-2026 panel also lacks historical RSS and
therefore tests price-risk transfer rather than the complete live news path.
