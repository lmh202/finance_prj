# Gated News Decision Layer

## Decision architecture

Daily Strategy supplies the prior direction. The learned model can add at most
5% only when recent news exists and the estimated
Q advantage is at least 0.50. HAR-X risk and Portfolio
Health are deliberately absent from the learned state: risk determines the
covariance, risky gross exposure and cash; Health changes the volatility budget
and risk aversion. No-news rows use the exact strategy-only path.

## Selection protocol

The model was selected with three expanding validation folds (2018, 2019,
2020). The final experimental artifact was refitted through 2020 and evaluated
on 2021-2023. That latter range is diagnostic, not a fresh blind test, because
earlier decision experiments already inspected it.

Selected model: `mlp_32x16_a001`. Selected allocation:
`risk6_vol15_sig10`.

### Validation

| validation_year | certainty_equivalent | sharpe | max_drawdown | cer_gain_vs_strategy | sharpe_gain_vs_strategy | news_applied_share |
| --- | --- | --- | --- | --- | --- | --- |
| 2018.0000 | 0.0094 | 0.6657 | -0.2060 | -0.0146 | -0.0688 | 0.0503 |
| 2019.0000 | 0.1689 | 1.7755 | -0.0883 | -0.0011 | -0.0200 | 0.0455 |
| 2020.0000 | -0.2243 | 0.0952 | -0.2594 | 0.1061 | 0.2171 | 0.0506 |

### Diagnostic test at 25 bps

| strategy | cagr | sharpe | certainty_equivalent | max_drawdown | average_turnover | average_cash_weight | news_applied_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strategy_risk_control | 0.1081 | 0.8066 | 0.0541 | -0.2071 | 0.0148 | 0.3154 | 0.0000 |
| gated_news | 0.1110 | 0.8207 | 0.0561 | -0.2088 | 0.0146 | 0.3163 | 0.0589 |
| same_model_news_unavailable | 0.1081 | 0.8066 | 0.0541 | -0.2071 | 0.0148 | 0.3154 | 0.0000 |
| same_model_sentiment_zero | 0.1122 | 0.8334 | 0.0579 | -0.2098 | 0.0146 | 0.3152 | 0.0615 |

## Engineering checks

- No-news exact fallback: `True`.
- Doubling every HAR-X sigma lowers average risky gross:
  `True`.
- Validation-only numerical gates passed:
  `False`.
- Full numerical/statistical promotion gates passed:
  `False`.
- Direct-news residual status: `experimental_only`.

## Deployment decision

The production-safe path is `strategy_external_harx_news_risk`: Daily Strategy
sets direction while the formal HAR-X + News estimate and Portfolio Health
control stock weights, risky gross exposure, and cash. Thus news still affects
the output through estimated risk. The direct 5% news residual is disabled
because it failed cross-year and statistical gates. A fresh live archive with
at least 60 mature five-session decisions is required before reconsidering it.
