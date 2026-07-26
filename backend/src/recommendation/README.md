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
