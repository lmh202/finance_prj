# Risk Engine Model Benchmark

## Executive conclusion

Four non-trivial candidates were compared on the same leakage-safe,
stock-equal, annual walk-forward evaluation for future five-session realised
volatility:

1. HAR-X;
2. HAR-X + News;
3. XGBoost Gamma;
4. Residual MLP.

XGBoost Gamma achieved the lowest out-of-fold QLIKE, but HAR-X + News remains
the strongest production specification. It delivers a statistically
significant news increment, the best calibration ratio among the challengers,
and a much smaller deployment gap than the tree or neural candidates. The
Residual MLP improved average QLIKE but was not statistically stable and should
remain a research model.

## Evaluation design

- Target: future five-session daily realised volatility.
- Universe: 21 stocks used by the FNSPID risk-engine study.
- Outer test years: 2018–2023.
- Observations: 27,027 stock-date forecasts.
- Split: expanding annual walk-forward with a five-session embargo.
- Weighting: every stock receives equal weight in QLIKE.
- Primary metric: QLIKE; lower is better.
- Supporting metrics: log-RMSE, log-volatility R², calibration ratio,
  high-volatility QLIKE, positive-stock share, and Newey–West DM tests.
- No naïve forecast is counted as one of the four candidate models.

## Candidate specifications

### HAR-X

An interpretable pooled log-volatility model combining multi-horizon realised
volatility, Parkinson range volatility, and absolute returns. It is the
reference model for incremental gains.

### HAR-X + News

The formal risk-engine specification. A regularised Gamma variance-ratio term
uses causal news attention (`log_count`) to modify the HAR-X forecast. A valid
no-news sample is represented by a zero count and remains in the same model.

### XGBoost Gamma

A nonlinear gradient-boosted tree model trained with a Gamma objective on the
engineered price feature set. It captures threshold effects and feature
interactions but requires additional calibration and external generalisation
work before deployment.

### Residual MLP

A neural model with 68 inputs: 34 price features and 34 deployable news/state
features. The network contains 128- and 64-unit hidden layers with LayerNorm,
GELU activations, 15% dropout, AdamW optimisation, and 17,537 trainable
parameters. It predicts a bounded log-volatility residual around a fold-local
HAR forecast. The architecture is fixed; only the epoch count is selected from
the final mature year inside each outer training window.

## Out-of-fold results

| Model | QLIKE | Gain vs HAR-X | Log-RMSE | Log R² | Realised / forecast | High-vol QLIKE | Positive stocks | DM p vs HAR-X |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost Gamma | 0.657 | 10.31% | 0.568 | 0.291 | 0.892 | 1.354 | 95.2% | 0.0010 |
| HAR-X + News | 0.694 | 5.24% | 0.540 | 0.360 | 0.981 | 1.681 | 85.7% | 0.0011 |
| Residual MLP | 0.710 | 3.10% | 0.576 | 0.271 | 0.913 | 1.562 | 66.7% | 0.8394 |
| HAR-X | 0.733 | reference | 0.519 | 0.408 | 1.057 | 1.999 | reference | reference |

QLIKE and log-RMSE reward different behaviour. HAR-X has the lowest log-RMSE
but performs poorly in high-volatility observations. XGBoost reduces high-vol
QLIKE most aggressively, while HAR-X + News produces the calibration ratio
closest to one.

## Stability across years

HAR-X + News improved on HAR-X in five of six outer years. XGBoost also
improved in five of six years, but both XGBoost and the MLP worsened in 2023.
The MLP's main failure was 2021, when its QLIKE was 25.8% worse than HAR-X.
Its average 3.1% improvement is therefore not statistically distinguishable
from the reference (`p=0.839`).

This result does not show that neural networks are unsuitable for volatility
forecasting. It shows that this 21-stock, engineered-feature panel does not yet
support a stable neural advantage. A sequence model would require broader
cross-sectional data, longer live-news history, and a separate tuning budget.

## Why HAR-X + News remains the formal output

The historical news increment passes the risk-engine gates:

- 5.24% stock-equal OOF QLIKE improvement;
- 95% moving-block bootstrap interval of +2.38% to +8.83%;
- DM test `p=0.001087`;
- improvement in five of six years and 85.7% of stocks;
- 13.56% QLIKE improvement in the high-volatility regime;
- VaR-95 breach rate of 4.32%;
- 95% risk-band coverage of 96.34%;
- ES ratio of 0.983.

XGBoost Gamma is the point-forecast challenger, not the production replacement:
its current benchmark is price-only and does not satisfy the product
requirement that the formal output contain a validated positive news
contribution. The MLP includes news inputs but does not provide a significant
increment.

## Follow-up: can news improve XGBoost?

Two leakage-controlled architectures were tested after the initial benchmark.
Both retain every zero-news observation.

### Direct Price + News XGBoost

News fields were appended directly to the 34 price features. News family,
direct/ratio target, HAR blend, calibration, and tree count were selected inside
each outer training window.

- Price-only XGBoost QLIKE: 0.6571.
- Direct Price + News XGBoost QLIKE: 0.6855.
- Relative QLIKE change: **-4.32%**.
- News-active change: **-2.73%**.
- High-volatility change: **-7.93%**.
- DM test: `p=0.108`.

Direct feature concatenation therefore diluted the strong price signal. News
accounted for only 1.67% of final-tree total gain.

### Regularised news residual ratio

A second specification kept the XGBoost price forecast fixed and allowed news
to explain only the remaining variance ratio:

`sigma = sigma_xgb_price × GammaRatio(news features)`

| News family | Candidate QLIKE | Gain vs price-only XGBoost | DM p | 95% block-bootstrap gain | Positive years | Positive stocks | High-vol gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention | 0.6486 | +1.30% | 0.00137 | +0.51% to +2.67% | 4/6 | 81.0% | -4.17% |
| Log count only | 0.6531 | +0.61% | 0.00027 | +0.27% to +1.39% | 5/6 | 71.4% | -2.58% |
| All deployable news | 0.6518 | +0.81% | 0.05235 | -0.10% to +2.44% | 4/6 | 71.4% | -4.58% |
| Attention + FinBERT | 0.6627 | -0.84% | 0.64580 | -3.12% to +1.89% | 4/6 | 57.1% | -8.12% |

The attention family contains article frequency, recent counts, time since the
last article, and causal coverage/silence states. It improves average QLIKE and
passes the aggregate statistical tests, even after a conservative four-family
multiple-test adjustment. However, it worsens QLIKE in the realized
high-volatility regime by 4.17%, so it fails the Risk Engine tail-robustness
gate. Adding FinBERT sentiment fields does not improve the XGBoost residual
model in the current data.

The result is therefore a useful research ceiling, not a promotion candidate.
News can improve XGBoost's average prediction when added as a constrained
residual component, but the improvement is smaller than the formal HAR-X +
News contribution and is not robust in the regime that matters most for risk.

## Recommendation

- Keep **HAR-X + News** as the formal five-session Risk Engine.
- Keep **XGBoost Gamma** as the primary research challenger.
- Retain **XGBoost + Attention residual ratio** as the best news-aware XGBoost
  research specification, but do not promote it until high-volatility QLIKE is
  no worse than the price-only XGBoost reference.
- Do not promote the **Residual MLP** from the current experiment.
- Accumulate at least 60 mature live RSS forecasts.
- Add unseen individual equities, rather than relying mainly on ETF-based
  external symbols.
- Re-test an XGBoost model with a deployable news contract and re-calibrate
  VaR/ES before considering replacement.
