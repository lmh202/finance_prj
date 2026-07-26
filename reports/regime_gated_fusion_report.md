# Shock-Confirmed V-Recovery Fusion

## Outcome

The selected candidate passes the requested point-estimate acceptance check on
all three relevant samples:

- the 21-stock 2019-2023 validation period;
- the 21-stock 2024-2026 diagnostic period; and
- the independent nine-sector-ETF 2000-2023 history.

Relative to the current systemic gross-only Fusion, the candidate improves
CAGR, Sharpe, and Calmar in all three full periods. It does not alter Daily
Strategy's relative stock weights, and it leaves the existing crisis-entry
logic unchanged.

The candidate remains **experimental/shadow-only**. The 2024-2026 observations
were inspected while designing the recovery rule, and the bootstrap intervals
still include zero.

## Selected method

### Unchanged normal and crisis behaviour

Daily Strategy continues to determine the complete relative stock portfolio.
The existing Fusion still controls normal and crisis gross exposure:

```text
systemic stress =
    45% SPY volatility percentile
  + 30% cross-asset correlation percentile
  + 25% high-risk breadth

base volatility budget = 30% calm -> 15% full stress
gross = clip(effective budget / predicted Strategy volatility, 35%, 100%)
```

The existing fast attack and 70% stress release remain unchanged.

An additional general bull-market override was tested and rejected. In
rewarded high-volatility bull observations, the current Fusion already held
97.3% average gross historically and 100% average gross in 2024-2026. A broad
bull override therefore added almost no useful exposure and degraded the
forward result.

### Shock-confirmed V-recovery state

A recovery is eligible only after a confirmed abrupt shock:

```text
five- or ten-session SPY decline >= 8%
and systemic stress >= 80% (or emergency state)
```

Eligibility lasts for at most 16 rebalances. Recovery activates only when all
of the following causal conditions hold:

1. the emergency state has ended;
2. SPY's five-session return is at least +3%;
3. the rebound is strong relative to its own expanding history and at least
   50%-80% of assets participate;
4. SPY has recovered at least five percentage points from the shock trough;
5. the 200-day trend has not deteriorated by more than 0.5% over 20 sessions.

When activated, the risky gross target is raised toward 90% for two
rebalances. Recovery is cancelled immediately by a renewed emergency, a
five-percentage-point systemic-stress acceleration, or a five-session SPY loss
worse than 2%.

The 90% target is projected onto the common gross interval that satisfies the
existing maximum position and five-percentage-point position-change limits.

### Strict gross-only implementation

At every date:

```text
candidate_weight_i =
    current_fusion_relative_weight_i * candidate_gross
```

The maximum measured relative-weight deviation from current Fusion is below
`1e-16`. This removes a projection artifact found in the earlier research:
changing gross could otherwise leave different relative weights months later
and make recovery timing look like stock-selection alpha.

## Primary acceptance results

### 21 stocks: 2019-2023 validation

| Metric | Daily Strategy | Current Fusion | New Fusion |
|---|---:|---:|---:|
| Total return | 244.13% | 159.64% | **167.07%** |
| CAGR | 28.09% | 21.06% | **21.75%** |
| Sharpe | 1.072 | 1.115 | **1.139** |
| Annual volatility | 26.37% | **18.73%** | 18.85% |
| Maximum drawdown | -38.92% | -32.61% | **-31.79%** |
| Calmar | **0.722** | 0.646 | **0.684** |

New Fusion versus current Fusion:

- CAGR: **+0.69 percentage points**
- Sharpe: **+0.024**
- maximum drawdown: **+0.82 percentage points**
- Calmar: **+0.038**

The 2014-2018 design-period result is exactly unchanged because no eligible
recovery event fired in that period.

### 21 stocks: 2024-01-02 to 2026-07-24

| Metric | Daily Strategy | Current Fusion | New Fusion |
|---|---:|---:|---:|
| Total return | **146.78%** | 136.40% | **141.54%** |
| CAGR | **42.56%** | 40.17% | **41.36%** |
| Sharpe | 1.551 | 1.536 | **1.570** |
| Annual volatility | 24.86% | **23.85%** | 23.89% |
| Maximum drawdown | -24.05% | **-23.86%** | **-23.86%** |
| Calmar | **1.769** | 1.684 | **1.733** |

New Fusion versus current Fusion:

- total return: **+5.13 percentage points**
- CAGR: **+1.19 percentage points**
- Sharpe: **+0.033**
- maximum drawdown: unchanged
- Calmar: **+0.050**

Compared with Daily Strategy, New Fusion now has slightly higher Sharpe,
0.97 percentage points lower volatility, and 0.19 percentage points less
drawdown, while giving up 1.20 percentage points of CAGR.

### Calendar attribution

| Period | Policy | Return | Sharpe | Maximum drawdown |
|---|---|---:|---:|---:|
| 2024 | Daily Strategy | 32.73% | 1.400 | -16.47% |
| 2024 | New Fusion | **34.04%** | **1.444** | **-16.41%** |
| 2025 | Daily Strategy | **35.82%** | **1.434** | -23.16% |
| 2025 | Current Fusion | 27.19% | 1.245 | **-23.04%** |
| 2025 | New Fusion | **29.95%** | **1.340** | **-23.04%** |
| 2026 through July 24 | Daily Strategy | 36.90% | 1.952 | -15.92% |
| 2026 through July 24 | New Fusion | **38.66%** | **2.080** | **-15.42%** |

New Fusion is exactly identical to current Fusion in 2024 and 2026. All
forward improvement comes from the intended 2025 recovery event rather than a
lasting relative-weight difference.

## Event attribution

The selected rule is sparse:

| Dataset | Recovery date | Baseline gross | 90% target | Feasible gross |
|---|---:|---:|---:|---:|
| 21 stocks historical | 2020-05-21 | 42.8% | 90.0% | 84.7% |
| 21 stocks historical | 2022-03-18 | 54.1% | 90.0% | 90.0% |
| 21 stocks forward | 2025-04-29 | 52.5% | 90.0% | 90.0% |
| Sector ETFs external | 2011-09-16 | 57.2% | 90.0% | 64.7% |
| Sector ETFs external | 2020-05-11 | 50.3% | 90.0% | 63.0% |

The feasibility projection explains why some events do not immediately reach
90%: preserving identical relative weights while limiting every position move
to five percentage points is the binding constraint.

During the explicitly measured 2025 V-rebound window (April 8 to June 30):

| Policy | Return | Sharpe | Maximum drawdown |
|---|---:|---:|---:|
| Daily Strategy | **30.50%** | 4.502 | -4.38% |
| Current Fusion | 21.09% | 4.806 | **-1.95%** |
| New Fusion | **23.72%** | **5.195** | **-1.95%** |

The new state recovers 2.63 percentage points of the missed rebound without
giving back drawdown protection in that window.

## Crisis protection

The recovery rule activates only after the emergency has ended, so crisis
entry and crash protection are unchanged:

| Episode | Daily Strategy MDD | Current Fusion MDD | New Fusion MDD |
|---|---:|---:|---:|
| 2018 Q4 | -24.01% | **-19.51%** | **-19.51%** |
| COVID crash | -30.30% | **-20.04%** | **-20.04%** |
| 2022 bear market | -38.65% | -32.21% | **-31.38%** |

The 2015-2016 adjustment remains a failure on episode return: New Fusion is
identical to current Fusion and trails Daily Strategy. The new recovery rule
does not claim to solve that slower, smaller adjustment.

## External ETF check: 2000-2023

| Metric | Daily Strategy | Current Fusion | New Fusion |
|---|---:|---:|---:|
| CAGR | 6.25% | 6.16% | **6.22%** |
| Sharpe | 0.430 | 0.486 | **0.490** |
| Annual volatility | 17.79% | **14.45%** | 14.46% |
| Maximum drawdown | -51.65% | **-38.29%** | **-38.29%** |
| Calmar | 0.121 | 0.161 | **0.162** |

The 2000-2013 ETF subsection is modestly worse than current Fusion, while the
2014-2023 subsection is better. The full external history is positive but
small; it is supporting evidence, not proof.

## Uncertainty and constraints

Paired 20-session moving-block bootstrap, New Fusion minus current Fusion:

| Period | Delta Sharpe | 95% interval | P(positive) |
|---|---:|---:|---:|
| 21 stocks, 2019-2023 | +0.024 | [-0.005, +0.067] | 92.3% |
| 21 stocks, 2024-2026 | +0.033 | [0.000, +0.105] | 84.1% |
| Sector ETFs, 2000-2023 | +0.003 | [-0.002, +0.013] | 75.1% |

The intervals are not conventional proof of outperformance. The rule changes
only a handful of observations, so many bootstrap samples contain little or
none of the treatment.

Constraint audit:

- maximum position change is at most 5pp on every dataset;
- maximum stock/ETF weight remains below its configured cap;
- relative-weight deviation is numerically zero;
- 25 bps transaction cost is charged on all turnover.

The proportional recovery basket creates some sub-1pp component trades. A live
implementation therefore needs portfolio-basket execution semantics; applying
the existing one-percent minimum separately to every stock would break exact
uniform scaling.

## Status

`shock_v_recovery_90_uniform` is the selected research candidate. It should
replace the prior fast-release candidate in shadow evaluation, but it should
not be promoted to live allocation until:

1. a portfolio-basket execution path is defined;
2. its causal state is persisted across live decisions; and
3. it accumulates a genuinely unseen paper-trading period.

