# Adaptive Fusion Design

> Historical design record. The 75/25 candidate was not promoted and has since
> been replaced as the preferred research candidate by the gross-only systemic
> overlay documented in `fusion_generalization_report.md`.

## Decision

The best product-specific pre-2024 research candidate was a hierarchical
guardrail:

1. form a stable target from 75% neutral allocation and 25% Daily Strategy
   rank;
2. estimate a causal portfolio stress score;
3. vary the base volatility ceiling from 25% in confirmed calm/bullish
   conditions to 15% under stress or a non-bullish market;
4. project the target onto that volatility ceiling and the production
   position/trade constraints.

Risk is therefore a binding constraint when necessary, not a permanent vote
against every Strategy position.

The subsequent locked 2024-2026 forward evaluation **did not validate the
candidate for promotion**. It continued to lower volatility and drawdown
relative to Daily Strategy, but delivered lower CAGR, Sharpe, and Calmar.
The method should remain a research reference rather than replace production.

## Why the current objective loses return

The production optimiser minimises

```text
-expected return
+ 0.5 * risk aversion * portfolio variance
+ turnover cost
```

at every rebalance. With a measured Strategy IC of only 0.02 and effective
risk aversion near 7.8 when Health is 60, the covariance term dominates even
when total portfolio risk is already acceptable. Raising the base volatility
target alone cannot remove this relative-weight effect.

The proposed hierarchy instead solves

```text
minimise distance(weights, 75% neutral + 25% Strategy target) + turnover

subject to:
    predicted annual volatility <= dynamic risk budget
    sum(weights) <= 100%
    0 <= stock weight <= 20%
    weekly stock-weight change <= 5pp
    active trade >= 1pp
```

## Causal stress and risk budget

The stress score uses only information available at the rebalance date:

```text
45% expanding percentile of median HAR-X sigma
25% expanding percentile of SPY 20-day volatility
15% expanding percentile of average asset correlation
15% breadth of assets above their own 80th-percentile HAR-X sigma
```

The score has fast attack and slower release. Its percentile score is mapped
continuously from calm at 50 to full stress at 80.

Relaxation additionally requires all three bullish conditions:

- SPY price above its 50-day moving average;
- SPY 50-day moving average above its 200-day moving average;
- positive SPY 20-day momentum.

This prevents a secular bear from being labelled safe merely because
volatility has normalised. It does not use a realised drawdown threshold; that
rule was rejected after failing the 2000-2013 external test.

The base volatility ceiling is

```text
25% in confirmed bullish/calm conditions
15% under full stress or outside the bullish regime
```

Portfolio Health then applies the existing budget factor. At Health=60 these
become effective ceilings of 21.5% and 12.9%.

## 21-asset results

All results use the production Daily Strategy inputs, real HAR-X estimates,
five-session rebalancing, 25 bps transaction cost, and the existing 20%
position/5pp change/1pp minimum-trade controls.

### 2019-2023 validation

| Policy | CAGR | Sharpe | Volatility | Max drawdown | Calmar |
|---|---:|---:|---:|---:|---:|
| Daily Strategy | 28.09% | 1.072 | 26.4% | -38.9% | 0.72 |
| Current Fusion | 13.73% | 0.997 | 13.9% | -20.5% | 0.67 |
| **75/25 guardrail candidate** | **19.60%** | **1.193** | 16.1% | **-24.6%** | **0.80** |

Relative to current Fusion, the candidate gains 5.87 percentage points of
annual return and 0.196 Sharpe, while giving back 4.1 percentage points of
maximum-drawdown protection. Relative to naked Daily Strategy, it retains
14.3 percentage points of drawdown protection and has higher Sharpe.

The paired 20-day block bootstrap for candidate minus current Fusion gives
Delta Sharpe +0.196, but its 95% interval remains wide and crosses zero:
[-0.382, +0.703]. The sample therefore supports an experimental candidate,
not automatic promotion.

### 2014-2023 full panel

| Policy | CAGR | Sharpe | Volatility | Max drawdown | Calmar |
|---|---:|---:|---:|---:|---:|
| Daily Strategy | 24.79% | 1.088 | 22.8% | -38.9% | 0.64 |
| Current Fusion | 13.89% | 1.080 | 12.8% | -20.5% | 0.68 |
| **75/25 guardrail candidate** | **19.28%** | **1.219** | 15.5% | **-24.6%** | **0.78** |

The candidate's risk constraint binds on 35.5% of rebalances. Average risky
gross exposure is 83.75%, demonstrating that the return improvement comes from
better separation of target construction and risk control, not simply from
holding more gross exposure.

## Crisis behaviour on the 21 assets

| Episode | Daily Strategy | Current Fusion | 75/25 guardrail |
|---|---:|---:|---:|
| 2015-2016 adjustment | -9.79% | +0.59% | -9.04% |
| 2018 Q4 | -22.38% | -8.33% | -14.93% |
| COVID crash | -27.00% | -16.12% | -17.59% |
| 2022 bear market | -36.85% | -19.45% | -23.38% |

The candidate preserves substantial protection in the three major shocks, but
does not match the current Fusion in the 2015-2016 adjustment. That episode is
the clearest remaining failure mode.

## Rolling-window stability

Candidate versus current Fusion on the 21 assets:

| Window | Sharpe win rate | Median Delta Sharpe | CAGR win rate | Median Delta CAGR |
|---|---:|---:|---:|---:|
| 1 year | 53.8% | +0.021 | 67.5% | +2.96pp |
| 2 years | 61.0% | +0.058 | 77.1% | +3.37pp |
| 3 years | 75.3% | +0.092 | 92.5% | +4.62pp |
| 5 years | 94.2% | +0.083 | 100.0% | +5.96pp |

Every three- and five-year window has a worse maximum drawdown than current
Fusion, with a median cost of about 3pp. That is the price paid for the higher
return and should be explicit in the product objective.

## External limitation

On the nine-sector-ETF panel from 2000-2013:

| Policy | CAGR | Sharpe | Volatility | Max drawdown |
|---|---:|---:|---:|---:|
| Current Fusion | 7.13% | 0.636 | 11.9% | -34.2% |
| 75/25 guardrail candidate | 5.92% | 0.506 | 13.0% | -34.3% |

The candidate retains drawdown protection but loses risk-adjusted return. This
means the 21-asset improvement is not evidence of a universally superior
allocation rule. The Daily Strategy rank and the neutral anchor behave
differently across universes; signal quality remains a real constraint.

## Locked forward result: 2024-01-02 to 2026-07-24

No candidate parameter was re-tuned for this period. The same 21 assets,
five-session rebalancing, 25 bps cost, 75/25 target, and dynamic guardrail were
used.

| Policy | CAGR | Sharpe | Volatility | Max drawdown | Calmar |
|---|---:|---:|---:|---:|---:|
| Daily Strategy | **42.56%** | **1.551** | 24.9% | -24.1% | **1.77** |
| 75/25 guardrail candidate | 25.63% | 1.326 | **18.5%** | **-19.1%** | 1.34 |

The candidate reduced annual volatility by 6.4 percentage points and maximum
drawdown by 4.9 percentage points, but gave up 16.9 percentage points of CAGR.
Candidate-minus-Strategy Sharpe was -0.225, with a paired 20-day block
bootstrap interval of [-0.562, +0.077] and only 6.4% bootstrap probability of
a positive difference.

This forward result confirms that the guardrail buys protection, but rejects
the stronger claim that it improves risk-adjusted performance over Daily
Strategy in the current market sample.

## Rejected alternatives

- Raising only the volatility target: increases return modestly but leaves
  covariance preference active at all times.
- Lowering risk aversion continuously: improves the 21-asset validation result
  but loses on the long-history ETF panel.
- Drawdown-aware re-risking: fails significantly in 2000-2013 because it keeps
  adding exposure during prolonged bear markets.
- A 25-40% Strategy sleeve over the current risk portfolio: produces high
  21-asset returns but significantly lowers Sharpe on the ETF panel.
- A 100% Strategy-first guardrail: improves 21-asset return but fails strongly
  on the ETF panel because the Strategy target itself is not transferable.

## Promotion recommendation

Do not promote the 75/25 guardrail. The required new locked period is now
available and did not pass the risk-adjusted-performance criterion. Any next
candidate should require:

1. a new locked 21-asset period or mature paper-trading archive;
2. paired Sharpe/CER confidence intervals that exclude zero;
3. maximum drawdown no worse than current Fusion by more than 5pp;
4. preserved protection in stress episodes;
5. no material increase in realised transaction costs;
6. explicit metadata showing stress, volatility budget, constraint binding,
   neutral weight, Strategy tilt, and final weight.
