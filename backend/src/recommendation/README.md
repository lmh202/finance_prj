# Final Decision Layer

## Production rule fusion

`fusion.py` is the default daily recommendation path. It produces one
explainable result per held asset:

```text
strategy (direction) + news (direction) + portfolio health (quality)
    -> raw directional score
    -> HAR-X risk attenuation
    -> AURORA score, outlook, confidence, action, and bounded position change
```

The initial directional weights are 50% strategy, 30% news, and 20% health.
HAR-X risk is not a bearish vote: it multiplies the raw score by
`1 - 0.45 * risk_percentile / 100`, so it can only move a recommendation
toward neutral. The live adapter removes the legacy strategy score's explicit
low-volatility rank before fusion to avoid treating volatility as direction.

News is importance/relevance/recency weighted and duplicate stories are
collapsed. With one or two current articles, news receives at most 10% and the
unused weight moves to strategy. With no current news, its weight is zero and
the asset still receives a recommendation. Strong strategy/news conflict
forces Hold; extreme volatility caps positive exposure; stale and unavailable
inputs are explicit in the output.

The current Health engine is portfolio-level. Its score is therefore a shared
quality input for all held assets and the API labels it with
`health_scope=portfolio`.

## Experimental numeric optimiser

The daily decision path combines two numerical inputs:

1. a candidate model that estimates each stock's next-20-session return
   relative to SPY; and
2. the formal HAR-X + News five-session volatility estimate.

The return model supplies cross-sectional alpha only after it passes the
promotion gates. The risk estimate always supplies per-stock volatility and
therefore affects the covariance matrix, target weights, and proposed trades.
If the return model fails promotion, production falls back to the same
deterministic optimiser with expected alpha set to zero.

The optimiser is long-only and fully invested. It applies the shared maximum
position, maximum weekly change, and minimum trade constraints, and includes
transaction costs in its objective.

DeepSeek is an explanation-only layer. It receives an already-fixed numeric
decision and cannot add, remove, reverse, or resize trades. Missing credentials,
network errors, malformed JSON, or schema violations use a deterministic
template instead.

## Files

- `fusion.py`: production rulebook, news aggregation, risk attenuation,
  conflict/staleness gates, and explainable per-asset results.
- `decision.py`: online feature parity, artifact loading, constrained
  optimisation, and the first fallback recommendation path.
- `llm_client.py`: validated DeepSeek JSON explanations, caching, and
  deterministic fallback.
- `engine.py`: legacy event-reaction and final fallback paths.

## Offline artifacts

- `data/processed/decision_model/decision_model.json`
- `data/processed/decision_model/return_model.joblib`
- `reports/decision_layer/report.md`

Set `DEEPSEEK_API_KEY` only in the runtime environment. Never commit it.
# Gated-news candidate

`gated_news.py` implements the current candidate decision architecture:

1. Daily Strategy supplies the prior direction.
2. A confidence-gated model may add a small news residual.
3. HAR-X + News risk is used externally for covariance, risky gross exposure,
   cash, and per-stock sizing.
4. Portfolio Health changes only the risk budget and risk aversion.
5. No recent news produces the exact strategy-only path.

The candidate checkpoint lives in
`data/processed/decision_model_candidate_gated_news`. The daily endpoint only
loads it when `metadata.json` records `promotion_status: promoted`; an
`experimental_only` checkpoint never applies a direct news residual.

Until promotion, the production path is Daily Strategy plus external
HAR-X + News position control. News therefore still changes the formal output
through estimated volatility, covariance, risky gross exposure, and cash, but
it does not cast an unvalidated directional vote.

## Alpha scaling (why the tilts are small)

`risk_controlled_allocation` is a mean-variance optimiser, so the conversion
from Daily Strategy's direction rank to an expected return decides how hard it
tilts. That conversion now uses Grinold-Kahn:

```text
alpha = IC * sigma_horizon * z(direction)      # strategy_alpha()
```

`STRATEGY_INFORMATION_COEFFICIENT = 0.02` is the assumed forward information
coefficient. It replaced a hardcoded `expected = direction * 0.010`, which
implied an IC near **0.14** — about ten times any measured value.

Measurements behind the constant:

| panel | Daily Strategy direction, rank IC vs 5d/20d forward return |
|---|---|
| 21-symbol FNSPID | +0.010 / +0.029, t ≈ 0.8 / 1.2 |
| best cheap reformulation (21) | +0.029 / +0.049, but quintile returns are flat and a long-only tilt *lowers* out-of-sample Sharpe |
| 165-symbol wide panel | −0.010 / −0.009; sector-neutral +0.005 |

The 21-symbol positive IC is a sector artifact: the panel is 17/21 tech, and
neutralising sector removes it. Effective breadth is the binding limit —
average pairwise return correlation is 0.33 (21 names) and 0.37 (165 names),
so `N_eff = N/(1+(N-1)rho)` sits at its `1/rho` ceiling of ~2.7 either way.
Adding names does not buy independent bets.

The setting is deliberately asymmetric. Backtested on the 21-symbol universe
at 25 bps, five-session rebalancing:

| assumed IC | Sharpe (signal works) | Sharpe (signal shuffled) | turnover | cost/yr |
|---|---|---|---|---|
| 0.02 | 1.076 | 1.025 | 0.64 | 0.16% |
| 0.05 | 1.298 | 1.033 | 3.04 | 0.76% |
| 0.14 (old) | 1.324 | 0.775 | 7.69 | 1.92% |

Being too conservative costs ~0.25 Sharpe when the signal is real. Being too
aggressive costs ~0.25 Sharpe **and** 1.9%/yr in fees when it is not — and the
shuffled-signal turnover reaches 22x/yr. Given the direction signal is not
statistically distinguishable from noise, the conservative side is the
defensible one. Raise the constant only with a walk-forward IC measurement
that supports it.

## Adaptive risk aversion (the market-stress state)

Risk aversion decides how far the mean-variance solution sits from minimum
variance. It is no longer fixed: `recommend_strategy_risk_control` picks it
from a causal market-stress state.

```text
rv   = SPY.pct_change().rolling(60).std() * sqrt(252)
pct  = rv.rolling(504, min_periods=252).rank(pct=True)   # current obs included
stressed = pct[-1] >= 0.75

CALM_RISK_AVERSION     = 2.0    # calm  -> further from minimum variance
STRESSED_RISK_AVERSION = 6.0    # stressed / unknown -> toward minimum variance
```

`src/recommendation/market_stress.py` owns the rule; the caller in
`gated_news.py` maps the state to the constant. **The rolling-rank expression
is pinned** — it is what was backtested, and it includes the current
observation in its own window.

**Fail closed.** `unknown` uses the stressed setting, so a missing or broken
benchmark fetch can never silently widen the risk budget. That also means
omitting `benchmark_close` reproduces the previous fixed-6.0 behaviour exactly,
which is what keeps the research scripts and the existing tests unchanged.

**What the state actually changes is composition, not exposure.** A lower risk
aversion moves the relative portfolio away from minimum variance, so its
predicted volatility rises — and the volatility target then compensates by
holding *more* cash. Do not expect calm markets to show a higher gross weight;
the distinguishing quantity is `predicted_annual_volatility`.

Measured on the production code path with the real HAR-X + News risk engine
(5-session rebalance, 25 bps one-way), adaptive vs the Daily Strategy baseline:

| Sample | Sharpe | Calmar | Max drawdown |
|---|---|---|---|
| 2024-2026, 21 stocks | 1.766 vs 1.586 | 2.31 vs 1.92 | −14.4% vs −22.8% |
| 2014-2023, 21 stocks | 1.328 vs 1.193 | 0.87 vs 0.70 | −23.7% vs −38.9% |
| 2000-2023, sector ETFs | 0.653 vs 0.430 | 0.23 vs 0.12 | −32.8% vs −51.7% |

Capital was preserved in 7 of 7 crises. **The significance caveat matters:**
against the Daily Strategy the gain is significant on the 24-year sample
(moving-block bootstrap ΔSharpe +0.220, 95% CI [+0.063, +0.383]); against a
*fixed* risk aversion of 6.0 nothing is significant on any sample. The
configuration is defensible, not proven superior to the fixed one.

The benchmark series comes from `routers/_common.load_benchmark_close()` — a
separate 5-year fetch, memoised for six hours. The two-year frame the other
engines share is too short: the signal needs ~312 sessions minimum and 564 for
a full reference window.

Known behaviour: the backtest rebalanced every five sessions while the endpoint
recomputes per request, so a percentile hovering near 0.75 could churn trades.
The turnover penalty, `max_change=0.05` and `min_trade=0.01` damp this, and
60-session realised volatility moves slowly. **Do not add hysteresis** — it
would deviate from the validated configuration. `volatility_percentile` is
recorded in `decision_meta` so boundary-hovering is observable after the fact.
