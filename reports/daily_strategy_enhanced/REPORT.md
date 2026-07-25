# Enhanced Daily Strategy — XGBoost Residual Alpha

## Selected design

The current rule score remains the prior. `xgb_26` predicts the
unexplained part of the future five-session beta-adjusted cross-sectional
return rank. The selected blend is `eta=0.75` with a
residual cap of `0.50`. HAR-X + News risk and
Portfolio Health remain external position controls.

## Diagnostic 2021–2023 at 25 bps

| strategy | cagr | sharpe | certainty_equivalent | max_drawdown | average_cash_weight |
| --- | --- | --- | --- | --- | --- |
| daily_strategy_rule | 0.1081 | 0.8066 | 0.0541 | -0.2071 | 0.3154 |
| daily_strategy_xgb_residual | 0.1056 | 0.7805 | 0.0504 | -0.2330 | 0.3627 |

## External 2024–2026 at 25 bps

| group | strategy | cagr | sharpe | certainty_equivalent | max_drawdown | cer_gain_vs_rule |
| --- | --- | --- | --- | --- | --- | --- |
| all | rule | 0.2676 | 1.5855 | 0.1753 | -0.1591 |  |
| all | xgb_residual | 0.2431 | 1.2553 | 0.1299 | -0.1670 | -0.0454 |
| seen | rule | 0.3618 | 1.7670 | 0.2239 | -0.1774 |  |
| seen | xgb_residual | 0.3424 | 1.5661 | 0.1936 | -0.1770 | -0.0303 |
| unseen | rule | 0.0731 | 0.8017 | 0.0487 | -0.1196 |  |
| unseen | xgb_residual | 0.0872 | 0.8138 | 0.0533 | -0.1247 | 0.0045 |

## Promotion

- Validation candidate passed: `False`.
- Diagnostic significance passed: `False`.
- External generalisation passed: `False`.
- Status: `experimental_only`.

The model is only loaded by the backend when every gate passes. Otherwise the
existing rule Daily Strategy remains the production prior.

## Upper-bound conclusion

The searched XGBoost residual-alpha family did not produce a robust
improvement. The selected ceiling candidate loses CER and Sharpe in both the
2021-2023 diagnostic period and the full 2024-2026 external universe. A
validation-defined conservative challenger was also evaluated separately and
did not repair the degradation. Adding the available FNSPID sentiment fields
changed validation Rank IC only marginally. The rule strategy therefore
remains production, and the checkpoint stays available for research only.
