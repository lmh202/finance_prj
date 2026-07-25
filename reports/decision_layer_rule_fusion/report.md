# Rule-Fusion Decision Layer Backtest

## Bottom line

The fixed rule-fusion candidate **DID NOT PASS** the historical deployment gates.
News contribution validated: **false**.

This result evaluates the actual rule structure, not the previous ML optimiser.
The 50/30/20 weights were not retuned on the locked test.

## Protocol

- Validation: **2018-2020**
- Locked historical test: **2021-2023**
- Rebalance interval: **5 sessions**
- Primary one-way transaction cost: **25 bps**
- Initial cash buffer: **5%**
- Directional inputs at close t first affect returns on t+1
- Risk input: out-of-fold HAR-X + News five-session percentile
- News input: next-session FNSPID/FinBERT features with causal five-session decay
- Health input: trailing two-year portfolio health, recalculated through t

## Portfolio performance

| period | strategy | cagr | sharpe | certainty_equivalent | max_drawdown | annual_turnover | average_cash_weight |
| --- | --- | --- | --- | --- | --- | --- | --- |
| locked_test | equal_weight | 0.1271 | 0.5922 | -0.0472 | -0.3735 | 0.0000 | 0.0447 |
| locked_test | fusion_full | 0.1261 | 0.5783 | -0.0594 | -0.3606 | 0.9074 | 0.0247 |
| locked_test | fusion_full_no_risk | 0.1334 | 0.6037 | -0.0517 | -0.3418 | 1.0463 | 0.0252 |
| locked_test | fusion_no_news | 0.1489 | 0.6520 | -0.0406 | -0.3528 | 1.0918 | 0.0237 |
| locked_test | strategy_only | 0.1477 | 0.6490 | -0.0409 | -0.3240 | 1.3907 | 0.0253 |
| locked_test | strategy_risk | 0.1484 | 0.6577 | -0.0345 | -0.3443 | 1.2357 | 0.0250 |
| validation | equal_weight | 0.2824 | 0.9448 | -0.0043 | -0.2987 | 0.0000 | 0.0369 |
| validation | fusion_full | 0.2038 | 0.7679 | -0.0416 | -0.2852 | 0.7733 | 0.0258 |
| validation | fusion_full_no_risk | 0.2312 | 0.8470 | -0.0152 | -0.2728 | 0.8390 | 0.0321 |
| validation | fusion_no_news | 0.1637 | 0.6474 | -0.0893 | -0.2927 | 0.9731 | 0.0309 |
| validation | strategy_only | 0.1924 | 0.7109 | -0.0847 | -0.2932 | 1.4342 | 0.0246 |
| validation | strategy_risk | 0.1973 | 0.7139 | -0.0962 | -0.2830 | 1.1697 | 0.0251 |

## Signal diagnostics

| period | strategy | direction_accuracy_20d | auc_20d | rank_ic_5d | rank_ic_20d | top_bottom_spread_20d |
| --- | --- | --- | --- | --- | --- | --- |
| locked_test | fusion_full | 0.5071 | 0.4902 | 0.0147 | 0.0266 | -0.0067 |
| locked_test | fusion_full_no_risk | 0.5077 | 0.4836 | 0.0136 | 0.0232 | -0.0057 |
| locked_test | fusion_no_news | 0.5116 | 0.5018 | 0.0158 | 0.0296 | -0.0056 |
| locked_test | strategy_only | 0.5190 | 0.5087 | 0.0142 | 0.0254 | -0.0069 |
| locked_test | strategy_risk | 0.5171 | 0.5128 | 0.0142 | 0.0282 | -0.0059 |
| validation | fusion_full | 0.5515 | 0.5139 | 0.0266 | 0.0397 | -0.0144 |
| validation | fusion_full_no_risk | 0.5523 | 0.5090 | 0.0266 | 0.0372 | -0.0136 |
| validation | fusion_no_news | 0.5525 | 0.5174 | 0.0304 | 0.0354 | -0.0121 |
| validation | strategy_only | 0.5452 | 0.5251 | 0.0300 | 0.0338 | -0.0097 |
| validation | strategy_risk | 0.5454 | 0.5301 | 0.0292 | 0.0333 | -0.0119 |

## News contribution: full fusion minus no-news fusion

```json
{
  "validation": {
    "mean_daily_return_gain": 0.0001200084469101449,
    "newey_west_lag": 5,
    "newey_west_t": 1.453264472081794,
    "newey_west_p": 0.14615033757881785,
    "block_length": 20,
    "bootstrap_samples": 2000,
    "cer_gain_ci_low": -0.007604124063422468,
    "cer_gain_ci_high": 0.1349812788470646,
    "probability_cer_gain_positive": 0.944
  },
  "locked_test": {
    "mean_daily_return_gain": -8.035701619299869e-05,
    "newey_west_lag": 5,
    "newey_west_t": -1.2970348142902284,
    "newey_west_p": 0.1946192083445546,
    "block_length": 20,
    "bootstrap_samples": 2000,
    "cer_gain_ci_low": -0.046444476266686,
    "cer_gain_ci_high": 0.008446292627889691,
    "probability_cer_gain_positive": 0.0955
  }
}
```

## Locked-test initial-cash sensitivity

| initial_cash_weight | strategy | cagr | sharpe | certainty_equivalent | max_drawdown |
| --- | --- | --- | --- | --- | --- |
| 0.0000 | equal_weight | 0.1330 | 0.5966 | -0.0583 | -0.3875 |
| 0.0000 | strategy_only | 0.1445 | 0.6375 | -0.0447 | -0.3337 |
| 0.0000 | fusion_no_news | 0.1537 | 0.6711 | -0.0333 | -0.3494 |
| 0.0000 | fusion_full | 0.1257 | 0.5788 | -0.0577 | -0.3647 |
| 0.0500 | equal_weight | 0.1271 | 0.5922 | -0.0472 | -0.3735 |
| 0.0500 | strategy_only | 0.1477 | 0.6490 | -0.0409 | -0.3240 |
| 0.0500 | fusion_no_news | 0.1489 | 0.6520 | -0.0406 | -0.3528 |
| 0.0500 | fusion_full | 0.1261 | 0.5783 | -0.0594 | -0.3606 |
| 0.1000 | equal_weight | 0.1210 | 0.5877 | -0.0367 | -0.3590 |
| 0.1000 | strategy_only | 0.1441 | 0.6373 | -0.0440 | -0.3269 |
| 0.1000 | fusion_no_news | 0.1476 | 0.6483 | -0.0414 | -0.3534 |
| 0.1000 | fusion_full | 0.1329 | 0.6101 | -0.0440 | -0.3382 |

## Deployment checks

```json
{
  "news_contribution_validated": false,
  "fusion_deployment_validated": false,
  "checks": {
    "validation_cer_above_no_news": true,
    "locked_test_cer_above_no_news": false,
    "locked_test_sharpe_above_no_news": false,
    "locked_test_cer_above_strategy_only": false,
    "locked_test_cer_above_equal_weight": false,
    "locked_test_news_nw_p_below_005": false,
    "locked_test_news_bootstrap_ci_positive": false,
    "locked_test_drawdown_not_worse_than_no_news_2pp": true
  },
  "primary_transaction_cost_bps": 25.0
}
```

## Data coverage

| year | rows | symbols | news_day_share | unique_stories |
| --- | --- | --- | --- | --- |
| 2018.0000 | 5271.0000 | 21.0000 | 0.6054 | 12682.0000 |
| 2019.0000 | 5291.0000 | 21.0000 | 0.6791 | 15135.0000 |
| 2020.0000 | 4437.0000 | 21.0000 | 0.6054 | 10709.0000 |
| 2021.0000 | 4032.0000 | 16.0000 | 0.4931 | 6021.0000 |
| 2022.0000 | 4016.0000 | 16.0000 | 0.7311 | 19178.0000 |
| 2023.0000 | 3664.0000 | 16.0000 | 0.8881 | 36018.0000 |

## Interpretation limits

- The historical dataset contains deduplicated story counts, FinBERT
  sentiment, and event-family shares, but not the live per-article importance
  field. Event-family weights are therefore used as a documented importance
  proxy.
- Historical article timestamps are represented by their causal next-trading
  session. The five-step decay uses trading sessions rather than exact hours.
- The full news-enabled fusion cannot be tested on the 2024-2026 external
  panel because that panel has no archived RSS/FinBERT news channel.
- Portfolio results assume a 5% initial cash buffer because independent
  increase/reduce recommendations require a funding convention. Buys cannot
  exceed available cash plus sale proceeds, and leverage is prohibited.
- This is an evaluation of a deterministic allocation rule, not evidence that
  any individual recommendation will be profitable.
