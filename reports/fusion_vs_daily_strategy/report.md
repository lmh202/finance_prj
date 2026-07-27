# Fusion Engine vs Daily Strategy — measured comparison

_Produced 2026-07-27. Numbers reproduce exactly against the production code
path as of the adaptive-risk-aversion release
(`strategy-external-harx-news-risk-adaptive-v1`)._

## What this compares

**Daily Strategy (baseline).** The §5 cross-sectional score used directly as a
portfolio: rank the holdings, weight proportionally to rank, cap each name at
the §9 maximum position, stay fully invested. This is what the strategy engine
recommends if nothing sizes it.

**Fusion Engine.** The production decision path — Daily Strategy supplies
direction only (Grinold-Kahn scaled by a measured IC of 0.02), the HAR-X + News
risk engine supplies per-stock size, gross exposure and cash, Portfolio Health
scales the risk budget, and the market-stress state sets the base risk aversion
(2.0 calm / 6.0 stressed).

Both arms are driven by the **actual production functions** —
`daily_strategy.score_assets` → `gated_news._runtime_features` →
`gated_news.strategy_alpha` → `gated_news.risk_controlled_allocation` — and by
the **real HAR-X + News risk engine** loading `data/processed/risk_model.json`.
Neither is a reimplementation.

## Method

| | |
|---|---|
| Rebalance | every 5 sessions |
| Transaction cost | 25 bps one-way, charged on turnover |
| Risk input | `risk_engine.risk_estimates(ohlc, horizons=(5,))`, causal — OHLC only through the previous session |
| Volatility target | 15% annualised, scaled by portfolio health |
| Constraints | §9: max position 20%, max weekly change 5pp, min trade 1pp |
| Universes | **21-stock** = the FNSPID panel (17 US large-cap tech + PEP, COST, GLD, SLV). **Sector-ETF** = the 9 Select Sector SPDRs, used pre-2013 because most of the 21 had not listed |

---

## Table 1 — 2019-2023 (21-stock universe, 1,259 sessions)

| Metric | Daily Strategy | Fusion | Difference |
|---|---|---|---|
| CAGR | +28.09% | +15.36% | −12.73 pp |
| Sharpe | 1.072 | 1.028 | −0.044 |
| Sortino | 1.426 | 1.365 | −0.060 |
| Annualised volatility | 26.37% | 15.00% | −11.37 pp |
| Maximum drawdown | −38.92% | **−23.67%** | **+15.26 pp** |
| Calmar | 0.72 | 0.65 | −0.073 |
| Daily VaR 95% | −2.54% | −1.44% | +1.10 pp |
| Daily ES 95% | −3.88% | −2.26% | +1.62 pp |
| Worst day | −12.05% | **−5.90%** | +6.15 pp |
| Worst month | −12.54% | −8.07% | +4.47 pp |
| Longest drawdown | 502 days | 414 days | −88 days |
| Days >10% underwater | 40.22% | 27.11% | −13.12 pp |
| Final balance (from $100k) | **$344,130** | $204,028 | −$140,102 |

## Table 2 — 2024-2026 (21-stock universe, 642 sessions, through 2026-07-24)

Out-of-sample for the risk model, which was fitted on data through 2023.

| Metric | Daily Strategy | Fusion | Difference |
|---|---|---|---|
| CAGR | +43.83% | +33.37% | −10.47 pp |
| **Sharpe** | 1.586 | **1.766** | **+0.180** |
| **Sortino** | 2.221 | **2.298** | +0.078 |
| Annualised volatility | 24.89% | 17.15% | −7.74 pp |
| Maximum drawdown | −22.79% | **−14.42%** | **+8.37 pp** |
| **Calmar** | 1.92 | **2.31** | **+0.391** |
| Daily VaR 95% | −2.60% | −1.59% | +1.00 pp |
| Daily ES 95% | −3.49% | −2.50% | +0.99 pp |
| Worst day | −6.85% | −7.16% | −0.31 pp |
| Worst month | −8.12% | −7.70% | +0.42 pp |
| Longest drawdown | 127 days | 75 days | −52 days |
| Days >10% underwater | 11.68% | **2.65%** | −9.03 pp |
| Final balance (from $100k) | **$252,445** | $208,241 | −$44,204 |

Additional properties measurable only on the Fusion arm, because the baseline
has no risk model:

| | |
|---|---|
| Realised volatility vs 15% target | **15.4% (+0.4 pp error)** over 2.5 years |
| Ex-ante VaR-95, Kupiec unconditional coverage | p = 0.176 — **pass** |
| Ex-ante VaR-95, Christoffersen independence | p = 0.191 — **pass** |
| Standardised residual std | 1.269, 95% CI [1.092, 1.420] — **risk underestimated ~21%** |

The VaR backtest uses `risk_engine.portfolio_risk()` forecasts made *before*
each 5-session window, compared against realised returns (n = 128). Both
calibration tests pass, but the σ bias is significant and should be monitored:
the breach rate is 7.81% against a 5% nominal, which passes Kupiec only because
n = 128 has low power.

## Table 3 — Three configurations across three samples

Included because it isolates what the adaptive risk aversion contributes over a
fixed one.

| Sample | Strategy | CAGR | Sharpe | Vol | MaxDD | Calmar | $100k |
|---|---|---|---|---|---|---|---|
| **2024-2026**<br>21-stock | Daily Strategy | +43.83% | 1.586 | 24.9% | −22.8% | 1.92 | $252,445 |
| | Fusion, fixed ra=6 | +25.60% | 1.554 | 15.4% | −12.5% | 2.05 | $178,738 |
| | **Fusion, adaptive** | +33.37% | **1.766** | 17.1% | −14.4% | **2.31** | $208,241 |
| **2014-2023**<br>21-stock | Daily Strategy | +27.24% | 1.193 | 22.3% | −38.9% | 0.70 | $1,344,357 |
| | Fusion, fixed ra=6 | +15.93% | 1.221 | 12.8% | −20.5% | 0.78 | $492,545 |
| | **Fusion, adaptive** | +20.60% | **1.328** | 14.9% | −23.7% | **0.87** | $753,859 |
| **2000-2023**<br>sector-ETF | Daily Strategy | +6.25% | 0.430 | 17.8% | −51.7% | 0.12 | $423,186 |
| | Fusion, fixed ra=6 | +7.79% | **0.686** | 12.0% | −34.2% | 0.23 | $595,597 |
| | **Fusion, adaptive** | +7.63% | 0.653 | 12.4% | **−32.8%** | 0.23 | $574,583 |

The adaptive setting wins on Calmar in all three samples and on Sharpe in the
two recent ones; it loses slightly to the fixed setting on the long crisis-heavy
sample. Note the mechanism: a lower risk aversion produces a *riskier relative
composition*, so the volatility target compensates by holding **more** cash —
calm markets show lower gross exposure, not higher.

---

## Table 4 — Crisis periods

Dot-com, GFC and EU debt use the sector-ETF universe; the rest use the 21-stock
universe. "Fusion" is the adaptive configuration.

| Crisis | Window | Universe | Daily Strategy | Fusion | Capital preserved | $100k → DS | $100k → Fusion |
|---|---|---|---|---|---|---|---|
| Dot-com crash | 2000-03-24 → 2002-10-09 | sector-ETF | −34.60% | −23.65% | **+10.95 pp** | $65,397 | $76,348 |
| **Global Financial Crisis** | 2007-10-09 → 2009-03-09 | sector-ETF | −50.91% | −26.43% | **+24.48 pp** | $49,092 | **$73,573** |
| EU debt crisis | 2011-04-29 → 2011-10-03 | sector-ETF | −15.54% | −13.04% | +2.50 pp | $84,456 | $86,957 |
| Q4-2018 selloff | 2018-09-20 → 2018-12-24 | 21-stock | −22.38% | −9.79% | **+12.59 pp** | $77,621 | $90,208 |
| **COVID-19 crash** | 2020-02-19 → 2020-03-23 | 21-stock | −27.00% | −16.18% | **+10.82 pp** | $73,001 | $83,823 |
| 2022 bear market | 2022-01-03 → 2022-10-12 | 21-stock | −36.85% | −21.92% | **+14.93 pp** | $63,155 | $78,084 |
| 2025 spring pullback | 2025-02-01 → 2025-05-31 | 21-stock | −1.61% | **+1.00%** | +2.61 pp | $98,389 | $101,003 |

**Capital was preserved in 7 of 7 crises.**

## Table 5 — Recovery, 12 months from each trough

This is the cost of Table 4 and must be quoted alongside it.

| Crisis | Daily Strategy | Fusion | Difference |
|---|---|---|---|
| Dot-com | +22.87% | +9.77% | −13.10 pp |
| Global Financial Crisis | +57.68% | +29.45% | −28.23 pp |
| EU debt crisis | +23.94% | +23.07% | −0.86 pp |
| Q4-2018 selloff | +47.73% | +31.43% | −16.30 pp |
| **COVID-19** | +109.00% | +35.39% | **−73.61 pp** |
| 2022 bear | +41.32% | +22.71% | −18.62 pp |
| 2025 spring | +109.04% | +68.88% | −40.16 pp |

Fusion trails in **7 of 7 recoveries**. The mechanism is the reactive lag in
volatility targeting: exposure is cut after volatility rises (i.e. after the
fall) and restored only after it subsides (i.e. after the rebound).

---

## Statistical significance

Moving-block bootstrap, block length 20 sessions, 5,000 resamples, on paired
daily returns.

| Comparison | Sample | ΔSharpe | 95% CI | P(>0) |
|---|---|---|---|---|
| Fusion − Daily Strategy | 2019-2023, 21-stock | −0.057 | [−0.468, +0.351] | 0.387 |
| Fusion − Daily Strategy | 2024-2026, 21-stock | +0.167 | [−0.397, +0.753] | 0.710 |
| **Fusion − Daily Strategy** | **2000-2023, sector-ETF** | **+0.220** | **[+0.063, +0.383]** | **0.997** |
| Adaptive − fixed ra=6 | 2024-2026 | +0.208 | [−0.222, +0.653] | 0.824 |
| Adaptive − fixed ra=6 | 2014-2023 | +0.099 | [−0.139, +0.336] | 0.798 |
| Adaptive − fixed ra=6 | 2000-2023 | −0.033 | [−0.100, +0.033] | 0.166 |

**Only one row is significant.** The 24-year sector-ETF sample — the only one
containing two −50% bear markets — shows a confidence interval excluding zero.
Every other comparison spans zero.

The +0.180 Sharpe gain in Table 2 therefore **must not be quoted as an
improvement**: its interval runs from −0.40 to +0.75. The defensible claim is
that Fusion earns its advantage in crises, and that a sample without one cannot
detect it.

---

## Honest reading

1. **Fusion does not improve Sharpe over the baseline.** The difference is
   −0.044 (2019-2023) and +0.180 (2024-2026), both inside measurement noise.
2. **It reliably reduces drawdown, tail loss and cost.** Maximum drawdown is
   roughly halved in every period tested; annual turnover falls from 8.99 to
   0.85, i.e. 2.25% → 0.21% in fees.
3. **It earns less in rising markets, by design.** Lower risk means lower
   return; the final-balance column will always favour the baseline in a bull
   sample.
4. **Its measurable edge is crisis behaviour**, and that is the only claim the
   statistics support.

## Measurement limitations

- **Run-to-run noise.** The optimiser is path-dependent (each rebalance starts
  from the previous weights) and yfinance data is occasionally revised. Repeat
  runs move Sharpe by roughly ±0.04 and, across separate data downloads, the
  *absolute* level by up to ~0.1. Only paired within-run differences — which is
  what the bootstrap uses — are reliable. **Do not quote absolute levels across
  tables produced on different days.**
- **Overlapping windows inflate apparent sample size.** Rolling-window win
  rates elsewhere in this project use windows sharing >90% of their data; with
  effective breadth of ~2.7 there are perhaps 3 independent 3-year periods in a
  decade.
- **Universe switch in Tables 4-5.** Pre-2013 crises use sector ETFs. The
  mechanism generalises across the two universes, but the numbers are not from
  a single continuous backtest.
- **Survivorship.** Both universes are fixed lists of instruments that exist
  today. Firms delisted during the window are absent, so absolute returns are
  optimistic for both arms equally.
- **Risk-model look-ahead in Tables 3-5.** `risk_model.json` was fitted on
  2018-2023 data and is applied to 1999-2017 in the sector-ETF sample. It is
  common to both arms, so the paired difference is unbiased — but the absolute
  levels on that sample are not point-in-time.

## What was tested and rejected

Recorded so it is not re-attempted:

| Idea | Result |
|---|---|
| Cross-sectional stock selection on 21 names | Rank IC t ≈ 1.1; quintile returns flat; a long-only tilt *lowers* out-of-sample Sharpe |
| Expanding the universe to 165 names to fix breadth | Falsified. `N_eff = N/(1+(N−1)ρ)` is already at its `1/ρ ≈ 2.7` ceiling; ρ rose from 0.329 to 0.370 and IC volatility fell only 1.1-1.4x against a predicted 2.86x |
| News sentiment as a directional vote | IC t = −0.48; the project's own locked test shows CER falling from −0.0406 to −0.0594 when news direction is added |
| Time-series momentum for exposure | Best in design period (Sharpe 0.718), worst in verification (0.619) |
| Drawdown-aware risk budget | Looked good on 4 trigger events (ΔSharpe +0.087, P = 0.94); falsified on the 25-event sample (−0.087, P(>0) = 0.030). It stayed levered through the entire 2000-2005 bear market |
| Minimum-variance / risk-parity weighting | Best in design (Sharpe 1.024), worst in verification (0.453) |

## Reproduction

The scripts that produced these tables were **not persisted** — they were run
from a scratch directory. Rebuilding requires:

1. OHLC for the universe via `yfinance` (21-stock list = `LEGACY_21` in
   `scripts/build_wide_price_panel.py`; sector-ETF list = the 9 Select Sector
   SPDRs).
2. Per rebalance date, in strict past-only slices: `score_assets` →
   `_runtime_features` for direction, `risk_engine.risk_estimates` for σ, and
   SPY's 60-session realised-volatility percentile for the market state.
3. `risk_controlled_allocation` with `base_target_annual_volatility=0.15`,
   `information_coefficient=0.02`, `turnover_penalty=0.0025`, and the §9
   constraints, charging 25 bps on turnover.

The `maximum_gross` parameter added with the locked-position fix defaults to
1.0 and is a no-op for these backtests — the tables were re-run after that
change and reproduce digit-for-digit.

**Persisting the harness under `scripts/` is the outstanding follow-up.**
Until then these numbers are documented but not one-command reproducible.
