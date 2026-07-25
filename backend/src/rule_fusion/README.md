# Rule Fusion Engine — four signals, four ordered steps, one decision

**Mission:** turn the daily strategy signal, news sentiment, the portfolio
health score and the volatility percentile into a single **per-holding**
decision a user can act on — and make every part of that decision traceable
back to the input that caused it.

The defining constraint: **the four inputs are never averaged into one score.**
Each acts on exactly one output dimension, in a fixed order, and each step
records what it saw and what it did.

| Step | Input | May change | May NOT change |
|---|---|---|---|
| 1 | Daily strategy (Buy/Sell/Neutral) | direction, base confidence | — |
| 2 | News sentiment (Pos/Neu/Neg) | confidence; direction **only** via a gated critical-event override | — |
| 3 | Health score 0–100 | confidence | direction |
| 4 | Volatility percentile 0–100 | position size | direction, confidence |

These are not comments — they are checked. `scripts/fusion_selfcheck.py`
restates each rule as an invariant and runs it over the whole scenario grid
(~4,000 cases). It exits non-zero if the engine stops obeying itself.

## Relationship to `src/recommendation/fusion.py`

There is a second, older fusion module in Developer 4's folder. It is a
**different design, not a duplicate**, and both can coexist:

| | `src/recommendation/fusion.py` | `src/rule_fusion/` (this engine) |
|---|---|---|
| Direction | blended vote: strategy **+ news + health** | strategy alone (Step 1); news may override, gated |
| Health | votes on direction | confidence only |
| Risk / volatility | pulls score toward neutral, cuts confidence, limits size | position size only |
| Combination | weighted numeric score (`STRATEGY_LABEL_SCORES`, `fuse_scores`) | staged ledger, nothing averaged |
| Served over HTTP | no — offline, via `scripts/backtest_rule_fusion.py` | yes — `GET /fusion/decisions` |

This engine is the strict-separation variant: it exists precisely because the
blended score cannot answer "which input caused this?". If the two ever need
to be reconciled, that is a Developer 4 decision — this folder does not touch
`src/recommendation/`.

## Public functions

```python
decide(inputs: FusionInputs) -> FusionDecision      # pure — no I/O, deterministic
fuse(strategy_recs, events, health, risk_levels, history, weights) -> List[FusionDecision]
rank(decisions) -> List[FusionDecision]             # most actionable first
```

`decide()` performs no data loading, so the entire rule table is exercisable
offline and one case at a time through `POST /fusion/simulate`. `fuse()` takes
every input by value — the router does the loading, same contract as
`risk_engine.risk_estimates`.

## The rules, in numbers

```
STEP 1  direction = {BUY: BUY, SELL: SELL, HOLD: NEUTRAL}[raw_signal]
        confidence = 0.20                             if NEUTRAL
                   = 0.45 + 0.20 * conviction         otherwise  (0.45 .. 0.65)
        conviction = clip((|score| - BUY_THRESHOLD) / (3.5 - BUY_THRESHOLD), 0, 1)

STEP 2  strength = clip(|sentiment| / 0.6, 0, 1) * (0.5 + 0.5 * importance/100)
        agrees    -> +0.15 * strength
        disagrees -> -0.20 * strength      asymmetric: contradiction is news,
        neutral   -> -0.05 * strength      confirmation mostly is not
        no story  ->  0.00

        CRITICAL OVERRIDE — all three gates must hold:
          a keyword from critical_events.json matches title+summary
          importance >= 60
          the symbol is genuinely named by the story (enforced in adapters)
        then |sentiment| >= 0.35 -> direction := sign(sentiment)
                                    confidence := min(0.75, 0.45 + 0.25*strength)
             |sentiment| <  0.35 -> direction := NEUTRAL, confidence := 0.15
                                    (a decisive event with an unclear sign is a
                                     reason to stand aside, not to guess)

STEP 3  health >= 70   BUY +0.10   SELL -0.05   NEUTRAL 0
        40 .. 70       BUY  0.00   SELL  0.00   NEUTRAL 0
        health <  40   BUY -0.15   SELL +0.10   NEUTRAL 0
        unknown        no-op, recorded

STEP 4  vol pct  <25 -> 1.25x   <50 -> 1.00x   <75 -> 0.75x   <90 -> 0.50x   else 0.25x
        NEW_BUY / ADD -> weight_points = 2.0 * multiplier
        TRIM          -> trim_fraction = min(0.75, 0.33 * multiplier)
        CLOSE         -> trim_fraction = 1.0
```

Confidence is clipped to [0,1] after every step.

**Action vocabulary** (holdings-aware, derived from the *fused* direction —
deliberately not `daily_strategy._final_action`, which maps the *pre-fusion*
raw signal):

| direction | held | confidence | action |
|---|---|---|---|
| any | any | < 0.35 | `HOLD` — not convinced enough to trade |
| BUY | no | >= 0.35 | `NEW_BUY` — initiate a position |
| BUY | yes | >= 0.35 | `ADD` — increase the position |
| SELL | yes | 0.35–0.70 | `TRIM` — partial sell-off |
| SELL | yes | >= 0.70 | `CLOSE` — fully liquidate |
| SELL | no | any | `HOLD` — nothing to sell, stay out |
| NEUTRAL | any | any | `HOLD` |

**Risk** is reported, not blended: `risk.level` is the volatility percentile
**verbatim**, and `risk.band` starts from it (<25 low, <60 moderate, <85
elevated, else high) then escalates **one notch per named driver** — critical
news, strategy/news disagreement, fragile health, or a position above
`CONSTRAINTS["max_stock_weight_pct"]`. Every driver is listed by name.

### Two interactions worth knowing before you tune anything

- **Step 3 can lift confidence above the Step 2 override cap.** The 0.75 cap
  applies to the *rebase*, not as a global ceiling, so a maximal override plus
  a supportive health tier lands near 0.78. That is intended — health is
  allowed to adjust confidence after an override.
- **A maximal override alone reaches `CLOSE`.** `CRITICAL_OVERRIDE_CAP` (0.75)
  sits above `CLOSE_CONFIDENCE` (0.70), so one sufficiently loud, decisive,
  relevant critical story can recommend a full exit without corroboration from
  any other channel. The gates are what keep that honest. To require two
  channels for a full exit, set `CRITICAL_OVERRIDE_CAP` below
  `CLOSE_CONFIDENCE` — the self-check still passes either way.

## Inputs / dependencies

- `daily_strategy.recommend_signals(...)` → `raw_signal` (BUY/SELL/HOLD) and
  the absolute −3.5..+3.5 confluence score. Chosen over `AssetSignal.action`
  because that score is a *cross-sectional percentile across your own
  holdings* — relative, not absolute (a one-stock portfolio always scores 100).
  `BUY_THRESHOLD` and the indicator weights are imported from that engine
  rather than re-hardcoded, so the two cannot drift.
- `news_intelligence.essential_news(...)` → `NewsEvent`. The
  positive/neutral/negative label is re-derived in `adapters.py` at the same
  ±0.15 cut `analyzer._sentiment_keywords` uses, because `analyzer` computes
  the label and then discards it and `NewsEvent` (frozen) carries only the
  float.
- `portfolio_health.compute_health(...)` → `HealthReport.score`. **Guarded:**
  that function returns `score=0.0` with an empty `metrics` dict when history
  is too short; passed through, it would read as a catastrophic portfolio, so
  the adapter maps it to `None` (unknown) and Step 3 becomes a recorded no-op.
- `risk_engine.risk_estimates(...)` → `RiskEstimate.risk_level` (0–100).
  **Degrades:** if the artifact `data/processed/risk_model.json` is missing or
  `risk_level` is NaN, `adapters.volatility_view` computes the same quantity
  (percentile of current 20-session annualized vol against the symbol's own
  history) from close prices, and reports `volatility_source` so the trace
  never hides which path ran. This is why `/fusion` has no `no_model` marker.

`critical_events.json` is this engine's own copy of the four-bucket taxonomy.
`news_intelligence/rules.json` is Developer 3's, uses different category names,
and is tuned for a different job — it is not edited or read here.

### On news appearing twice

`risk_engine.risk_level` already has news attention baked into σ̂ (its HAR-News
model). That is **not** a double count in this engine: news moves
**confidence** in Step 2, volatility moves **size** in Step 4. They act on
different output dimensions, so neither can amplify the other.

## Consumers

- `routers/fusion.py` → `GET /fusion/decisions`, `POST /fusion/simulate`,
  `GET /fusion/rules` (the live rule table + taxonomy, so a UI never needs a
  second copy of the thresholds).

## Validation

`python scripts/fusion_selfcheck.py` — prints a 15-row truth table, then
checks five invariants over the full scenario grid:

```
INV-1  health never changes direction                     3696 cases
INV-2  volatility changes size only                       3696 cases
INV-3  only a gated critical event overrides direction    1152 cases
INV-4  confidence ledger telescopes to the answer         4032 cases
INV-5  risk.level is the percentile verbatim              4032 cases
```

INV-4 is the one that makes the output auditable: every `Adjustment` carries
`confidence_before`/`delta`/`confidence_after`, and the deltas sum exactly to
the final confidence — including the Step 2 rebase, which is recorded as the
delta it implies. A step that may not move confidence still emits a row with
`delta 0.0`, so the invariant is visible in the API response rather than
something a reader has to take on trust.

## Rules

- Read-only w.r.t. the shared kernel; `src/interfaces.py` is untouched and all
  types here are declared locally.
- Reads other engines only through their public functions and constants —
  never their private helpers, never their folders.
- No training, no fitting, no network at request time. `decide()` is pure.
