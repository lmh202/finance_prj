# Direct XGBoost news-feature experiment

## Question

Can deployable news features improve the existing price-only XGBoost forecast of future five-session realised volatility?

## Result

- Price-only XGBoost QLIKE: **0.657140**
- Price + News XGBoost QLIKE: **0.685524**
- Relative QLIKE gain: **-4.32%**
- DM test: **p=0.107522**
- 95% moving-block bootstrap gain: **[-7.65%, -1.09%]**
- Positive outer years: **3/6**
- Positive stocks: **0.0%**
- Median stock gain: **-2.17%**
- News-active rows: **-2.73%**
- Zero-news rows: **-4.87%**
- High-volatility rows: **-7.93%**
- News share of final-tree total gain: **1.7%**

## Outer-year gains

| Test year | QLIKE gain vs price-only XGBoost |
|---:|---:|
| 2018 | +0.52% |
| 2019 | +0.23% |
| 2020 | -9.31% |
| 2021 | +0.13% |
| 2022 | -10.08% |
| 2023 | -0.27% |

## Leakage controls

- The price-tree configuration is inherited from the original inner search for each outer year.
- News family, target form, HAR blend, calibration, and tree count are selected inside the outer training window.
- The final inner validation year is truncated at the outer embargo cutoff.
- Every zero-news observation remains in training and scoring.
- The outer test year is not used for feature-family selection.

## Selected news families

- `all_deployable`: 2/6 outer folds
- `attention_finbert`: 3/6 outer folds
- `log_count`: 1/6 outer folds

## Interpretation

The experiment does not establish a stable positive news increment for direct XGBoost. The result should not replace the formal HAR-X + News model.

## Residual news-ratio follow-up

Direct concatenation is not the only possible architecture. A second experiment
kept the price-only XGBoost forecast fixed and used a regularised Gamma news
model to estimate only its residual variance ratio:

`sigma = sigma_xgb_price × GammaRatio(news features)`

| News family | QLIKE | Gain vs price-only XGBoost | DM p | 95% block-bootstrap gain | Positive years | Positive stocks | High-vol gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention | 0.6486 | +1.30% | 0.00137 | +0.51% to +2.67% | 4/6 | 81.0% | -4.17% |
| Log count | 0.6531 | +0.61% | 0.00027 | +0.27% to +1.39% | 5/6 | 71.4% | -2.58% |
| All deployable | 0.6518 | +0.81% | 0.05235 | -0.10% to +2.44% | 4/6 | 71.4% | -4.58% |
| Attention + FinBERT | 0.6627 | -0.84% | 0.64580 | -3.12% to +1.89% | 4/6 | 57.1% | -8.12% |

The attention specification is the best average result. It uses news frequency,
recent counts, time since the last article, and causal coverage/silence states.
Its aggregate gain remains significant under a conservative correction for the
four predefined families. However, its high-volatility QLIKE worsens by 4.17%,
so it does not pass the Risk Engine promotion gate.

FinBERT sentiment fields do not improve the XGBoost residual model in this
experiment. The positive news contribution comes from attention and recency,
not sentiment polarity.

## Final conclusion

News can improve XGBoost when it is introduced as a constrained residual
component rather than concatenated directly with all price inputs. The best
gain is +1.30%, materially smaller than the formal HAR-X + News gain of +5.24%,
and it is not robust in high-volatility observations. Keep it as a research
challenger; do not replace the formal model.
