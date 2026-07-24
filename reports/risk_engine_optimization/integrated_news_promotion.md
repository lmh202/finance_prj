# HAR-News integrated risk promotion

## Product definition

The formal primary output is future five-session realised volatility:

`Price Features + News Features -> Predicted Risk`

It does not predict price direction. The former “optional overlay” wording is
retired. The deployed model is one multiplicative Gamma risk function:

`sigma_5d = HAR(price features) * sqrt(GammaRatio(log_count))`

An observed no-news session is represented by `log_count=0`; it is not replaced
with sentiment-neutral data or dropped.

## Why this specification

Against the same price HAR base, adding causal news attention produced:

- stock-equal OOF QLIKE gain: **5.24%**
- 95% moving-block bootstrap: **+2.38% to +8.83%**
- DM test: **p=0.001087**
- positive outer years: **5/6**
- positive stocks: **85.71%**
- high-volatility QLIKE gain: **13.56%**
- VaR-95 breach rate: **4.32%**
- 95% risk-band coverage: **96.34%**
- ES ratio: **0.983**

`log_count` was selected in all 6 outer folds. Research-only FNSPID fields did
not improve the deployable ceiling, so the formal news contract remains small
and reproducible from RSS.

## Rejected joint alternative

Putting news directly into the already-strong 34-feature price Gamma model
improved aggregate QLIKE by only about 0.08% for `log_count`, and broader news
sets were unstable. Those results are retained in
`joint_model_diagnostic.csv`. The promoted HAR-News Gamma structure is the one
with statistically significant news increment; the result is not manufactured
by forcing coefficients into a stronger price-only model.

## Horizon policy

- **5 sessions:** formal HAR-News integrated model.
- **20 sessions:** auxiliary price-only HAR-X diagnostic. News worsened OOF
  QLIKE by 3.89%, so it is not presented as a news-driven primary output.

The previous official checkpoint is backed up as
`data/processed/risk_model.pre_har_news_integrated.json`.
