# Fusion Forward Evaluation: 2024 to Current

> This report evaluates the original 75/25 guardrail. It is retained as the
> failure baseline and has been superseded by
> `fusion_generalization_report.md`.

- Evaluation: **2024-01-02 to 2026-07-24**
- Assets: **21**
- Rebalance frequency: **5 trading sessions**
- Transaction cost: **25 bps**
- New Fusion: **75% neutral core + 25% Daily Strategy tilt, projected onto a causal dynamic volatility guardrail**
- Status: **locked forward evaluation; no parameters re-tuned**

## Overall results

| Metric | Daily Strategy | New Fusion |
|---|---:|---:|
| Total return | +146.78% | +78.83% |
| CAGR | +42.56% | +25.63% |
| Sharpe | 1.551 | 1.326 |
| Annual volatility | 24.9% | 18.5% |
| Maximum drawdown | -24.1% | -19.1% |
| Calmar | 1.77 | 1.34 |

## Calendar breakdown

| Year | Policy | Return | Sharpe | Volatility | Max drawdown |
|---:|---|---:|---:|---:|---:|
| 2024 | Daily Strategy | +32.73% | 1.400 | 22.0% | -16.5% |
| 2024 | New Fusion | +19.23% | 1.039 | 18.6% | -14.5% |
| 2025 | Daily Strategy | +35.82% | 1.434 | 23.4% | -23.2% |
| 2025 | New Fusion | +24.68% | 1.403 | 16.9% | -18.2% |
| 2026 | Daily Strategy | +36.90% | 1.952 | 31.5% | -15.9% |
| 2026 | New Fusion | +20.30% | 1.685 | 21.1% | -12.3% |

## Paired uncertainty

- New Fusion minus Daily Strategy Sharpe: **-0.225**
- 95% moving-block interval: **[-0.562, +0.077]**
- Bootstrap probability of a positive difference: **6.4%**

## Scope

The risk forecast uses the supported no-news historical path because a complete look-ahead-safe 2024-current news archive was not supplied to this evaluation. Live news can still affect the production risk estimate.
