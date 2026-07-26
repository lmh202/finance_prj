# Fusion Generalization Diagnosis and Retuning

> Superseded for shadow evaluation by
> `reports/regime_gated_fusion_report.md`, which adds a shock-confirmed,
> strict gross-only V-recovery state. This report remains the baseline
> diagnosis for the earlier fast-release candidate.

## Outcome

The old 75/25 guardrail did not generalize because Risk changed two things at
once:

1. it held too much cash during otherwise rewarded volatility; and
2. it rewrote Daily Strategy's cross-sectional stock weights.

The retuned Fusion preserves **100% of Daily Strategy's relative weights**.
Risk is allowed to change only the portfolio's uniform gross exposure. Gross
exposure falls when a causal systemic-stress score makes the Strategy
portfolio's predicted volatility exceed its current risk budget.

This restores most of the lost forward return while retaining meaningful
historical crisis protection.

## Failure decomposition: 2024-01-02 to 2026-07-24

| Construction | CAGR | Sharpe | Volatility | Max drawdown |
|---|---:|---:|---:|---:|
| Daily Strategy | 42.56% | 1.551 | 24.9% | -24.1% |
| Old 75/25 guardrail | 25.63% | 1.326 | 18.5% | -19.1% |
| Daily relative weights + old Fusion gross | 34.20% | 1.465 | 21.7% | -20.9% |
| Old Fusion relative weights + 100% gross | 32.21% | 1.422 | 21.2% | -21.5% |

Neither cash nor relative-weight distortion alone explains the failure. Both
channels made material contributions.

The old stress score also gave median individual HAR-X volatility too much
influence. High idiosyncratic volatility during a rising technology market was
treated like a systemic crisis. Its slow stress release then kept gross
exposure near 41%-73% during the April-May 2025 rebound.

## Retuned method

### Target portfolio

Daily Strategy supplies all relative stock weights:

```text
relative_weight_i = rank(Daily Strategy score_i)
```

Risk cannot reorder, favour, or suppress individual stocks.

### Systemic stress

Only common portfolio risk controls gross exposure:

```text
45% expanding percentile of SPY 20-day volatility
30% expanding percentile of average cross-asset correlation
25% breadth of assets above their own 80th-percentile HAR-X sigma
```

Median individual sigma is removed from the stress switch. HAR-X per-stock
sigma still enters the covariance matrix and predicted portfolio volatility.

Stress has:

- 75% fast attack when risk rises;
- 70% release when risk falls;
- no automatic penalty merely because the market is outside a bullish regime.

The 70% and immediate-release variants produce nearly identical forward
results. The 70% version retains better historical and external drawdown
protection and is selected.

### Uniform gross overlay

```text
base volatility budget = 30% in calm conditions
                       -> 15% under full systemic stress

effective budget = base budget * Portfolio Health factor

gross = clip(effective budget / predicted Strategy volatility, 35%, 100%)
final weight_i = gross * relative_weight_i
```

The existing maximum position, 5pp weekly change, 1pp minimum trade, and 25
bps transaction-cost assumptions remain in force.

## Primary forward result

### 2024-01-02 to 2026-07-24

| Metric | Daily Strategy | Old Fusion | **Retuned Fusion** |
|---|---:|---:|---:|
| Total return | **146.78%** | 78.83% | **136.40%** |
| CAGR | **42.56%** | 25.63% | **40.17%** |
| Sharpe | **1.551** | 1.326 | **1.536** |
| Annual volatility | 24.9% | **18.5%** | **23.8%** |
| Maximum drawdown | -24.1% | **-19.1%** | **-23.9%** |
| Calmar | **1.77** | 1.34 | **1.68** |

Relative to the old Fusion, the retuned method recovers:

- 14.54 percentage points of CAGR;
- 0.210 Sharpe;
- 57.6 percentage points of cumulative return.

Relative to Daily Strategy, it gives up 2.39 percentage points of CAGR while
lowering annual volatility by 1.1 percentage points. No systemic crisis occurs
in most of this period, so drawdown protection is only 0.2 percentage points.

Paired 20-day block bootstrap for retuned Fusion minus Daily Strategy Sharpe:

```text
Delta Sharpe = -0.015
95% interval = [-0.197, +0.103]
P(Delta > 0) = 44.3%
```

The two Sharpe ratios are not statistically distinguishable.

### Calendar results

| Year | Policy | Return | Sharpe | Volatility | Max drawdown |
|---:|---|---:|---:|---:|---:|
| 2024 | Daily Strategy | 32.73% | 1.400 | 22.0% | -16.5% |
| 2024 | Retuned Fusion | **34.04%** | **1.444** | 22.0% | **-16.4%** |
| 2025 | Daily Strategy | **35.82%** | **1.434** | 23.4% | -23.2% |
| 2025 | Retuned Fusion | 27.19% | 1.245 | **21.3%** | **-23.0%** |
| 2026 through July 24 | Daily Strategy | 36.90% | 1.952 | 31.5% | -15.9% |
| 2026 through July 24 | Retuned Fusion | **38.66%** | **2.080** | **30.5%** | **-15.4%** |

The remaining forward weakness is concentrated in the sharp April-May 2025
rebound. Faster release recovers part, but not all, of the missed upside.

## Historical 21-asset check

### 2014-2023

| Metric | Daily Strategy | Retuned Fusion |
|---|---:|---:|
| CAGR | **24.79%** | 20.21% |
| Sharpe | 1.088 | **1.130** |
| Annual volatility | 22.8% | **17.7%** |
| Maximum drawdown | -38.9% | **-32.6%** |
| Calmar | **0.64** | 0.62 |

### Crisis returns

| Episode | Daily Strategy | Retuned Fusion |
|---|---:|---:|
| 2015-2016 adjustment | **-9.79%** | -11.43% |
| 2018 Q4 | -22.38% | **-17.69%** |
| COVID crash | -27.00% | **-17.14%** |
| 2022 bear market | -36.85% | **-30.77%** |

The candidate protects the three major shocks, but still fails the smaller
2015-2016 adjustment.

## Long-history external ETF check

### 2000-2023

| Metric | Daily Strategy | Retuned Fusion |
|---|---:|---:|
| CAGR | **6.25%** | 6.16% |
| Sharpe | 0.430 | **0.486** |
| Annual volatility | 17.8% | **14.4%** |
| Maximum drawdown | -51.7% | **-38.3%** |
| Calmar | 0.12 | **0.16** |

This is the strongest generalization evidence: almost all long-run return is
retained while maximum drawdown improves by 13.4 percentage points.

## Interpretation and status

The retuned method fixes the architectural generalization failure:

- volatility is no longer interpreted as a bearish stock-selection signal;
- Risk no longer destroys Strategy's relative portfolio;
- gross exposure averages 97.8% in 2024-2026 because systemic stress is rare;
- Risk still cuts gross exposure during broad volatility/correlation shocks.

However, this is not a fresh promotion test. The 2024-2026 period was inspected
while diagnosing and tuning release speed. The 2025-current subsection is
reported separately but is not pristine because its aggregate performance had
already been observed.

The candidate should remain `experimental_only` until it accumulates a new
paper-trading archive or an unseen future period. It is materially better than
the old Fusion and is the preferred next shadow candidate, but should not be
described as statistically proven to outperform Daily Strategy.
