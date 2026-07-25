# Residual Contextual-Bandit Exploration

## Bottom line

Worth continuing: **false**.
News helped on the reused diagnostic period: **false**.
Promoted: **false**.

The model is a full-information contextual bandit, not PPO/DQN. Daily
Strategy is the prior action; the learned Ridge/MLP Q-policy supplies a
residual action, HAR-X controls action conservatism, and a deterministic
safety layer enforces cash/no-leverage execution.

## Time protocol

- Train: **2018-2019**
- Configuration selection: **2020**
- Diagnostic: **2021-2023**
- Diagnostic status: previously observed, therefore not a fresh blind test
- Rebalance: every **5 sessions**
- Primary one-way transaction cost: **25 bps**

## Selected full policy

```json
{
  "model": "mlp_64x32_a001",
  "model_kind": "mlp",
  "policy": {
    "news_enabled": true,
    "model": "mlp_64x32_a001",
    "kind": "mlp",
    "alpha": 0.001,
    "hidden": "64x32",
    "residual_weight": 0.75,
    "q_margin": 0.0,
    "policy_maximum_change": 0.04,
    "total_return": 0.442398171358237,
    "cagr": 0.44031130014356745,
    "benchmark_cagr": 0.18252915820840965,
    "annual_return": 0.441747822971794,
    "annual_volatility": 0.38890836190684214,
    "sharpe": 1.1358660965937495,
    "sortino": 1.2919267300205726,
    "certainty_equivalent": -0.012001318911395886,
    "max_drawdown": -0.30879523203320347,
    "daily_es95": -0.06354187161614132,
    "average_one_way_turnover": 0.0064291093014333415,
    "annual_turnover": 1.6201355439612022,
    "total_transaction_cost": 0.004066411633156588,
    "optimizer_success_rate": 1.0,
    "maximum_weight": 0.20892256180883273,
    "maximum_change": 0.03608396276798659,
    "minimum_active_trade": 0.0,
    "n_days": 253,
    "average_cash_weight": 0.030089931222183612,
    "average_health_score": 61.801216591402145,
    "learned_increase_share": 0.23323170731707318,
    "learned_reduce_share": 0.10060975609756098,
    "learned_hold_share": 0.6661585365853658
  }
}
```

## Diagnostic portfolio performance

| strategy | cagr | sharpe | certainty_equivalent | max_drawdown | annual_turnover | average_cash_weight |
| --- | --- | --- | --- | --- | --- | --- |
| bandit_full_news_zeroed | 0.1397 | 0.8243 | 0.0517 | -0.2448 | 1.0338 | 0.0205 |
| strategy_risk | 0.1484 | 0.6577 | -0.0345 | -0.3443 | 1.2357 | 0.0250 |
| strategy_only | 0.1477 | 0.6490 | -0.0409 | -0.3240 | 1.3907 | 0.0253 |
| equal_weight | 0.1271 | 0.5922 | -0.0472 | -0.3735 | 0.0000 | 0.0447 |
| bandit_full | 0.1576 | 0.6497 | -0.0636 | -0.3801 | 1.9983 | 0.0173 |
| bandit_no_news | 0.1375 | 0.5403 | -0.1851 | -0.5252 | 0.6712 | 0.0234 |

## Paired statistical tests

```json
{
  "full_vs_strategy_risk": {
    "mean_daily_return_gain": 6.065688931566083e-05,
    "newey_west_lag": 5,
    "newey_west_t": 0.2582894388525888,
    "newey_west_p": 0.7961835363838154,
    "block_length": 20,
    "bootstrap_samples": 2000,
    "cer_gain_ci_low": -0.15717700874821788,
    "cer_gain_ci_high": 0.10299219120919459,
    "probability_cer_gain_positive": 0.3065
  },
  "full_vs_same_model_news_zeroed": {
    "mean_daily_return_gain": 0.00016533070671041834,
    "newey_west_lag": 5,
    "newey_west_t": 0.49585328628147046,
    "newey_west_p": 0.6199979263277453,
    "block_length": 20,
    "bootstrap_samples": 2000,
    "cer_gain_ci_low": -0.28745187503013253,
    "cer_gain_ci_high": 0.06844053078707206,
    "probability_cer_gain_positive": 0.0975
  }
}
```

## Checks

```json
{
  "diagnostic_cer_above_strategy_risk": false,
  "diagnostic_sharpe_above_strategy_risk": false,
  "diagnostic_cer_above_same_model_news_zeroed": false,
  "diagnostic_cer_above_retrained_no_news": true,
  "full_vs_strategy_bootstrap_ci_positive": false,
  "news_bootstrap_ci_positive": false
}
```

## Interpretation

- `bandit_full_news_zeroed` uses the exact same fitted model but zeros all
  news inputs at inference, isolating the operational effect of news.
- `bandit_no_news` is independently trained without news and estimates the
  no-news model ceiling.
- The candidate cannot be promoted even if diagnostic metrics improve,
  because 2021-2023 has already influenced project decisions and the
  2024-2026 panel has no archived news input.
- Counterfactual rewards use next-five-session excess return minus formal
  risk and transaction-cost penalties. New actions first earn t+1 returns.
