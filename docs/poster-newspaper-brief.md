# AURORA Poster — Newspaper Edition

**Content brief + first-draft copy, for the teammate doing layout.**
Written 2026-07-26, revised 2026-07-27. Format: A1 portrait, printed.
Deliverable is content only — no HTML, no layout decisions locked.

---

## Summary — the whole argument, in one page

This document is long. If you read only this section, you have the position.

The through-line: **a newspaper is not an egalitarian structure, and neither is
AURORA.** Reporters file copy, wire services supply raw material, an editor
decides what runs. That hierarchy is the same shape as our architecture, where
each input controls exactly one dimension of the output and no dimension has
two owners. Every layout and copy decision below exists to protect that one
claim from being flattened into "three engines vote and we average them."

| Part | My position in one line |
|---|---|
| **1. Design** | The metaphor fits, but three equal columns read as three equal votes — the exact opposite of our claim. Fix with identical `DECIDES:` stamps, News styled as a wire service, and the fusion story in a band below rather than inside the columns. |
| **2. Layout** | Design for three reading distances. A reader who only gets the headlines and the ownership table at 1.5 m must still leave understanding the system. |
| **3. Form** | Body paragraphs are the *texture* layer, not the information layer. Information lives in headlines, stamps, tables and captions. The ownership table is the single most important object on the poster. Five figures, no more; the input→output flow diagram is the one that cannot be cut. |
| **4. Voice** | Use the newspaper's own headline → deck → body ladder; it solves "professional vs. accessible" natively. Gloss each term once, give numbers a baseline, and write captions as findings rather than descriptions. |
| **5. Models** | Front four chosen as *bait* — questions we want asked: HAR-X, FHS, FinBERT, and GARCH(1,1) as a control test. XGBoost is deliberately demoted to the second row and labelled a beaten challenger, because all three of its uses are `experimental_only`. "Twelve validated, one promoted" is a stronger claim than "we used fifteen models". |
| **6 & 8. Traps** | Print nothing that isn't traceable to code, a report, or an engine README. Three specific landmines: the retired 50/30/20 blend, the bare `{strategy: 1.0, …}` weights, and any `experimental_only` artifact implied to be live. |
| **7. Copy** | ~700 words of draft prose, block by block, with word budgets. Six values need a live run to fill; everything else is already verified. |

**The strongest single asset we have is not a model — it is the information
coefficient study.** We measured our own direction signal, found it was not
statistically distinguishable from noise, traced the one positive result to a
sector artifact, and then set the system's alpha scaling to a tenth of what the
old hardcoded value implied. That is why the architecture looks the way it does:
the direction signal is weak, so it is never allowed to size a position. It is
the editorial pull quote for exactly this reason — it converts what could read
as a limitation into the design rationale, with numbers behind it.

**The most common way this poster fails** is a teammate writing copy from
`backend/src/recommendation/README.md`, which still presents the retired
50/30/20 blend as production. It is wrong. CLAUDE.md and
`docs/AURORA-system-architecture.md` are right.

---

## Part 0 — How to use this document

- **Parts 1–6 are reasoning.** Read once, then ignore. They exist so that when
  you disagree with a copy choice you know what it was protecting.
- **Part 7 is the draft copy.** That is the thing to lay out. Every block has a
  word budget; going over is what kills poster readability.
- **Part 8 is the fact-check list.** Some numbers are placeholders that must be
  filled from a real run. They are marked `[FILL: …]`. Do not print any of them
  as-is.

Rule for the whole poster: **anything printed must be traceable to code, a
report in `reports/`, or an engine README.** Nothing invented. If a number
can't be sourced, cut the sentence.

---

## Part 1 — Design direction, and the one risk that matters

The newspaper metaphor fits AURORA better than a conventional research poster,
for a reason that isn't just aesthetic: **a newspaper is not an egalitarian
structure.** Reporters file copy, wire services supply raw material, and an
editor decides what runs and how much space it gets. That hierarchy is
isomorphic to AURORA's actual architecture, where the inputs are explicitly
*not* peers.

**The risk.** Three equal columns read as three equal votes. That is the exact
opposite of the system's central claim — Daily Strategy owns direction and
100% of relative weights, Health scales only the volatility budget, and News
never votes on direction at all. If the layout implies averaging, the poster
argues against the project.

**The fix, in three moves:**

1. Each desk column carries an identical, boxed **`DECIDES:` stamp** — same
   size, same position, one line. A reader at 1.5 m sees three different
   jurisdictions without reading a word of body copy.
2. **News is styled as a wire service, not a third reporter** — narrower
   column, smaller type, timestamps, telegraph voice. A visible rule or arrow
   runs from the wire column *into the Risk box*, not into the editor. That one
   line renders "news reaches the decision only through the risk forecast"
   without a sentence of explanation.
3. The full fusion story goes in a **full-width editorial band below** the
   three columns, not inside them. Putting it in the columns triplicates it and
   makes the columns unequal in length.

So the answer to "in the columns or below?" is **both, with a division of
labour**: stamps in the columns make the parallel honest at a glance; the band
below carries the mechanism.

---

## Part 2 — Layout map (A1 portrait, 594 × 841 mm)

```
┌──────────────────────────────────────────────────────┐
│ VOL. I · NO. 1      THE AURORA TIMES        26 JUL   │  masthead + double rule
│ "All the risk that's fit to print"                   │
├──────────────────────────────────────────────────────┤
│ LEAD STORY — full width headline                     │
│ deck + 4-sentence standfirst        ┌──────────────┐ │
│                                     │ MARKET       │ │
│                                     │ WEATHER      │ │
│                                     │ (regime)     │ │
├═════════════════════════════════════└──────────────┘═┤ ← the fold
│  MARKETS DESK    │  THE WIRE  ░░░  │  HEALTH DESK    │
│  Daily Strategy  │  News Intel     │  Portfolio      │
│                  │  (narrower,     │  Health         │
│  [FIG 5]         │   agate type)   │  [FIG — gauge]  │
│  body copy       │  body copy      │  body copy      │
│  ┌────────────┐  │  ┌───────────┐  │  ┌───────────┐  │
│  │DECIDES:    │  │  │DECIDES:   │  │  │DECIDES:   │  │
│  │direction + │  │  │nothing —  │  │  │volatility │  │
│  │all weights │  │  │feeds risk─┼──┼──│budget only│  │
│  └────────────┘  │  └───────────┘  │  └───────────┘  │
├──────────────────────────────────────┼───────────────┤
│ FROM THE EDITOR'S DESK               │ ↓ THE RISK    │
│ = the Fusion layer                   │   PAGE        │
│ [OWNERSHIP TABLE — the key table]    │ HAR-X + FHS   │
│ [pull quote]                         │ [FIG 2] [FIG 3]│
├──────────────────────────────────────────────────────┤
│ THE MARKET TABLE (agate) — one row per holding       │
├──────────────────────────────────────────────────────┤
│ ML INVENTORY (2 cols)  │  NOT YET FIT TO PRINT       │
├──────────────────────────────────────────────────────┤
│ bylines · stack · repo QR                            │
└──────────────────────────────────────────────────────┘
```

Reading distance is designed in three tiers:

| Tier | Distance | Carries |
|---|---|---|
| 1 | 3 m | Masthead, lead headline, the four ML names |
| 2 | 1.5 m | Desk headlines, `DECIDES:` stamps, ownership table, figure captions |
| 3 | 0.5 m | Body copy, formulas, agate market table |

A reader who only ever gets Tier 1 + Tier 2 must still leave understanding the
system. That is the test for whether the copy is correctly distributed.

---

## Part 3 — Form guide: what becomes a paragraph, table, list, or figure

The trap in a newspaper poster: newspapers *look* like solid paragraphs, but
poster readers do not read paragraphs. So —

> **Body paragraphs are the texture layer, not the information layer.**
> They make it read as a newspaper. The information load lives in headlines,
> stamps, tables and figure captions.

Assignment per block. **One primary form per block, plus at most one
supporting element.** Mixing forms inside a block is what makes a poster look
busy.

| Block | Primary form | Support | Why |
|---|---|---|---|
| Lead story | 4-sentence paragraph + one oversized number | — | The only place narrative earns its space |
| Market Weather | Icon + 3 label lines | Colour band | State information; sentences waste it |
| Each desk | Short paragraph (75–90 w) | 1 figure + `DECIDES:` stamp | Paragraph = texture, figure = information |
| **Editor's Desk** | **Table** | Flow figure + pull quote | See below |
| Risk Page | 3 formula lines | 2 small figures | Formulas are legitimate and high-status in a financial paper |
| Per-holding output | Agate table | — | `explain_allocation()` emits one record per holding; it is natively a table |
| ML inventory | Two-column table | — | Comparability is the whole point |
| Footer | List | QR | Credits, stack, limitations |

**The single most important object on the poster is the ownership table in the
editorial band.** It is the architecture claim in four rows. Make it large
enough to read at 1.5 m. Do not delete the third column ("Never does") — it
carries more information than the second, because it states what competing
designs fail to guarantee.

### Figures — five, in priority order

| # | Figure | Source | Why it earns space |
|---|---|---|---|
| 1 | **Four inputs → four output dimensions** flow diagram. Four lines, each landing on exactly one output. The News line does **not** reach the decision directly — it terminates in the Risk box. | Draw by hand from the ownership table | The one irreplaceable figure. If only one figure survives, this is it. |
| 2 | **HAR-X forecast vs realised volatility**, time series | `reports/risk_engine_optimization/` | The only promoted model must show evidence |
| 3 | **FHS return distribution with left tail marked (VaR/ES)**, with a normal curve overlaid for contrast | `risk_measures.py` output | Renders "we do not assume normality" in one image |
| 4 | **Four-model volatility benchmark**: HAR-X / HAR-X+News / XGBoost Gamma / Residual MLP | `reports/risk_engine_presentation/` — produced by `scripts/benchmark_risk_models_for_presentation.py` | Already built for presentation. Shows a tree model and a neural net were both tried and both lost. |
| 5 | **Regime colour band over a portfolio equity curve** | `daily_strategy.engine.backtest()` | Gives the Markets Desk column something to show |

Figure 4 is stronger than a two-bar ablation, because it is a *contest*, not a
comparison. Caption it as a result, not a description.

Cap the poster at five figures. A1 does not hold six.

---

## Part 4 — Writing rules: professional and readable at the same time

Use the newspaper's own three-layer structure — **headline → deck → body**.
This is not a compromise; it is the native form.

- **Headline** states the conclusion in plain words: *"News doesn't vote."*
- **Deck** gives one accessible sentence: *"Bad news never makes AURORA sell.
  It only makes AURORA size the position smaller."*
- **Body** is where HAR-X, FHS and SLSQP are allowed to appear.

Four hard rules:

1. **Numbers carry a baseline, not a parameter name.** Write "cut the
   volatility budget by 40%", never `risk_aversion=2.3`.
2. **Each term is glossed exactly once**, in 3–6 words, at first appearance:
   HAR-X (volatility forecast), FHS (tail simulation), FinBERT (financial-text
   sentiment). Never gloss the same term twice — that is how a poster runs out
   of room.
3. **Figure captions state findings, not contents.** Write "HAR-X turned up two
   days before the March volatility spike", not "predicted vs realised
   volatility". Captions have the highest read-rate of any text on a poster;
   spending them on description is waste.
4. **Keep the limitations box.** It is the highest-credibility block on the
   poster, not the lowest. A team that publishes what didn't work is read as a
   team that knows what did.

No code blocks. One function signature is an asset; a code listing is a
liability.

---

## Part 5 — The ML inventory

### 5.1 The front four

Selected on the criterion "what will judges ask, and where do we have a strong
answer" — i.e. these are chosen as **bait**, questions we want to be asked.

| # | Method | Why it is in the front four |
|---|---|---|
| 1 | **HAR-X + News** | The only model that actually runs in production |
| 2 | **Filtered Historical Simulation** | Answers the tail-risk question most teams cannot |
| 3 | **FinBERT** | The NLP leg, and its answer routes straight into the architecture claim |
| 4 | **GARCH(1,1) control test** | The most methodologically sophisticated thing in the repo |

**Rehearsed answers** — we built these collectively, so any of us can take any
of these questions. Worth rehearsing the exact numbers anyway:

> **"What model is actually running in production?"**
> HAR-X. HAR is the standard realised-volatility model in econometrics; we
> added news attention as an exogenous term. It improves five-session
> out-of-sample QLIKE by 5.24% over the price-only base, with a 95% moving-block
> bootstrap interval of +2.38% to +8.83% and Diebold-Mariano p = 0.0011. It is
> the only artifact we ever promoted.

> **"How do you compute tail risk — do you assume normality?"**
> No. We use Filtered Historical Simulation: empirical quantiles of
> standardised returns, fat-tailed and left-skewed, calibrated on data through
> 2020 only. Backtested with Kupiec and Christoffersen — VaR-95 breach rate
> 4.32% against a 5% target, 95% band coverage 96.34%, ES ratio 0.983.

> **"How do you use NLP?"**
> A local FinBERT — BERT-base fine-tuned on financial text, three-class. The
> weights sit on disk and run fully offline; we call no external API for it.
> But the important part is what it is *not* allowed to do: news never casts a
> directional vote. It only enters through the volatility forecast, so it
> changes position size, never buy-versus-sell.

> **"How do you know the news signal is real and not coincidence?"**
> We were worried "more news means more volatility" was just volatility
> clustering. So we ran GARCH(1,1) as a control — partial out the GARCH
> conditional forecast first, then check whether the news signal survives. It
> did, with FDR correction for multiple comparisons.

**Why XGBoost is deliberately *not* in the front four.** All three of its uses
in this repo are `experimental_only`. Putting it up front invites "so is it
live?", to which the honest answer is no — a net loss in a headline slot. But
it cannot be absent either, or we get asked "did you try gradient boosting?"
So it leads the second row, with its status stated:

> **XGBoost** — challenger, beaten three times (direction residual, Gamma
> variance, decision layer); stopped by the promotion gate.

That converts a weakness into evidence that the promotion gate is real.

### 5.2 Full inventory

Excludes rule-based components and the external-API LLM (DeepSeek), which is
explanation-only and cannot alter a decision.

**A. Production — runs on every request**

| Method | Location | What it does |
|---|---|---|
| **HAR-X** (HAR + exogenous news attention) | `risk_engine/engine.py` | Log realised volatility on 5/22/66-day HAR blocks + Parkinson range estimator + news attention. `σ̂ = σ̂_price · √news_ratio`. **The only promoted artifact in the project.** |
| **Filtered Historical Simulation** | `risk_engine` | Empirical standardised-return quantiles (fat-tailed, left-skewed, calibrated ≤2020) → VaR / ES with no Gaussian assumption |
| **EWMA covariance** | `risk_engine` | HAR σ̂ diagonal + EWMA correlation matrix → portfolio-level VaR / ES and diversification ratio |
| **FinBERT** (`ProsusAI/finbert`) | `news_intelligence/finbert_sentiment.py` | BERT-base encoder, three-class softmax; score = P(pos) − P(neg). Local inference, weights cached to disk, fully offline after first load. Also used offline in batch to build the historical sentiment table. |
| **SLSQP constrained optimisation** | `recommendation/gated_news.py` | Two-stage mean-variance: solve relative weights, then gross exposure and cash. Long-only, position caps, weekly-change caps, transaction costs in the objective. |

**B. Offline research and validation — all `experimental_only`**

| Method | Location | Role |
|---|---|---|
| **XGBoost** | `daily_strategy/learned.py`, `optimize_risk_engine.py`, `decision_layer_core.py` | Three independent candidates: direction residual alpha (40+ features), Gamma variance model, decision layer |
| **Multi-task MLP** (PyTorch, custom) | `explore_mlp_decision.py` | Shared encoder (Linear → LayerNorm → GELU → Dropout) with **three heads**: mean, q10 lower quantile, cross-sectional rank (tanh). Target: 20-session return relative to SPY ÷ ex-ante risk scale |
| **Residual MLP** (PyTorch, custom) | `benchmark_risk_models_for_presentation.py` | Predicts a bounded log-volatility residual around a fold-local HAR forecast — the network corrects the statistical model rather than replacing it |
| **Quantile regression** | the q10 head above | Lower-quantile estimation as an auxiliary task |
| **MLPRegressor** (sklearn) | `explore_residual_bandit.py`, `train_gated_news_decision.py` | Bandit Q-model and gated-news decision model |
| **Contextual bandit / residual Q-learning** | `explore_residual_bandit.py` | Daily Strategy is the prior action; the learned policy estimates only the residual correction |
| **HistGradientBoosting** | `decision_layer_core.py` | Decision-layer tree candidate |
| **ElasticNet** (L1+L2) | `decision_layer_core.py` | Decision-layer linear baseline |
| **Ridge** | `optimize_risk_engine.py`, `explore_residual_bandit.py` | HAR coefficient fitting; bandit Q-model |
| **Gamma GLM** (log link) | `optimize_risk_engine.py` | Variance is a positive, right-skewed target — Gamma fits it better than OLS |
| **GARCH(1,1) / EGARCH(1,1,1)** | `garch_control_check.py` | **Not a candidate model — a control.** Proves the news→volatility signal is not residual volatility clustering |

**C. Validation methodology** — this row is where most competing posters have
nothing, so give it real space:

```
Walk-forward / expanding-window cross-validation with embargo
· Diebold-Mariano test (p = 0.0011)
· Kupiec / Christoffersen VaR backtests
· Paired moving-block bootstrap (95% CI +2.38% to +8.83%)
· FDR correction for multiple comparisons
· Ablation: price-only vs price+news
· Time-locked out-of-sample windows (2021–23, 2024–26 opened only once)
```

**One-line version, if a compact list is needed:**

```
HAR-X, Filtered Historical Simulation, EWMA Covariance, FinBERT (BERT Transformer),
Multi-task MLP, Residual MLP, Quantile Regression, Contextual Bandit, XGBoost,
HistGradientBoosting, ElasticNet, Ridge, Gamma GLM, GARCH(1,1)/EGARCH, SLSQP Optimisation
```

Prefer the two-column split over this flat line. Fifteen names in a row looks
impressive until someone asks how many are live; the split turns that question
into the intended answer: **we validated twelve and promoted one.**

**Do not claim Random Forest or SHAP/TreeSHAP.** Neither exists in this
codebase. Random Forest appears in `daily_strategy/README.md` only as an early
plan that was never built. The interpretability counterpart we *do* have is
structural, and is stronger: `explain_allocation()` emits one auditable record
per holding, so the explanation is generated by the decision path itself rather
than reconstructed after the fact.

---

## Part 6 — Recurring cross-check

Every claim on the poster should survive this question: *does the live code do
this today?* Three specific traps:

- The 50/30/20 strategy/news/health blend is **retired**. It is described as
  production in `recommendation/README.md`, which is stale on that point. No
  router calls it. **It must not appear on the poster.**
- `fusion.explain_allocation()` **explains, it does not decide.** Its
  `component_weights` are literally `{strategy: 1.0, news: 0.0, health: 0.0}`.
  If that figure is printed, it needs the caption "*direction* weights only —
  news and health enter through other dimensions", or it reads as "news is
  worthless".
- Only `risk_model.json` is promoted. Every other artifact on disk is
  `experimental_only` and does not run at request time.

---

## Part 7 — FIRST DRAFT COPY

Word budgets are binding. Total prose ≈ 700 words, which is correct for A1.
`[FILL: …]` marks a value that must come from a real run.

---

### ▸ MASTHEAD

```
VOL. I · NO. 1                                        26 JULY 2026

              T H E   A U R O R A   T I M E S
              All the risk that's fit to print

PORTFOLIO INTELLIGENCE · GROUP CAPSTONE EDITION · FIVE ENGINES, ONE DECISION
```

---

### ▸ LEAD STORY *(90 words)*

**Kicker:** `EVERY MORNING, ONE DECISION`

**Headline:**

> # NEWS DOESN'T VOTE

**Deck:**

> Bad headlines never make AURORA sell. They only make it size the position
> smaller — and that distinction is the whole architecture.

**Standfirst** *(4 sentences, 75 words):*

> AURORA turns a portfolio, a price history and a live news feed into one
> daily recommendation. Five engines contribute, but they never take a vote:
> each one controls exactly one dimension of the answer, and no dimension has
> two owners. Daily Strategy decides direction. A HAR-X volatility model
> decides size. Portfolio Health decides how much risk the whole book may
> carry — and the news desk is never allowed near the direction.

**Oversized figure to set beside it:** `1` — *"engines that may set direction.
Out of five."*

---

### ▸ MARKET WEATHER *(box, 25 words)*

```
┌─────────────────────────────┐
│  MARKET WEATHER             │
│  ───────────────────────    │
│  [icon]                     │
│  TODAY:  [FILL: regime]     │
│  20d vol: [FILL: x.xx%]     │
│  Confidence: [FILL: 0.xx]   │
│                             │
│  Four regimes: bullish ·    │
│  bearish · high-volatility  │
│  · sideways                 │
└─────────────────────────────┘
```

Fill from `GET /strategy/regime`. Use one icon per regime — sun, storm, wind,
flat cloud.

---

### ▸ COLUMN 1 — MARKETS DESK *(85 words)*

**Standing head:** `MARKETS DESK`
**Sub-head:** `Daily Strategy Engine · By the AURORA staff`

**Headline:**

> ## Which Way, and How Much of Each

**Body:**

> The Markets Desk reads price history alone: momentum over 20 and 60 days,
> price against its 50- and 200-day averages, rolling volatility, RSI,
> drawdown and beta. It classifies the market into one of four regimes and
> ranks every holding cross-sectionally. That ranking is the *only* source of
> direction in the entire system, and it fixes 100% of the relative weights
> between stocks. What it never gets to decide is how much money stands behind
> that opinion.

**Figure:** FIG 5 — regime colour band over equity curve.
**Caption:** `[FILL: one-sentence finding from the backtest]`

**Stamp:**

```
▪ DECIDES: direction + all relative weights
```

---

### ▸ COLUMN 2 — THE WIRE *(narrower column, agate type, 85 words)*

**Standing head:** `THE WIRE`
**Sub-head:** `News Intelligence Engine · By the AURORA staff`

**Headline:**

> ## Copy Files, But Never to the Front Page

**Body:**

> The wire runs continuously: RSS feeds in, duplicates collapsed by URL and
> title similarity, at most five essential events surfaced per day and mapped
> to holdings. Sentiment is scored locally by FinBERT — a BERT encoder
> fine-tuned on financial text — which runs offline from cached weights and
> calls no external service. The wire files into exactly one place: the
> volatility forecast. A broken feed is reported as broken, never silently
> treated as a calm market.

**Visual:** a running ticker strip of `[FILL: 3–4 real headlines from
data/news_raw.json, with source and timestamp]`. Use genuine collected items
with real attribution — do not compose plausible-looking headlines.

**Stamp:**

```
▪ DECIDES: nothing — files to the Risk Page ──┐
```

*(rule continues into the Risk Page box)*

---

### ▸ COLUMN 3 — HEALTH DESK *(85 words)*

**Standing head:** `HEALTH DESK`
**Sub-head:** `Portfolio Health Engine · By the AURORA staff`

**Headline:**

> ## How Much Risk Is This Book Allowed?

**Body:**

> The Health Desk scores the portfolio 0–100 from annualised return,
> volatility, Sharpe and Sortino ratios, maximum drawdown, beta, and
> concentration by asset and by sector. The score was calibrated against
> rolling historical windows across many randomly generated portfolios, testing
> whether low-health books really did suffer worse forward drawdowns. Its one
> job in the daily decision is to set the volatility budget: a healthy
> portfolio is permitted more risk, an unhealthy one less. It never expresses a
> view on any stock.

**Figure:** health gauge + what-if delta.
**Caption:** `[FILL: e.g. "Health 68 → 73 if today's recommendation is
accepted"]`

**Stamp:**

```
▪ DECIDES: the volatility budget only
```

---

### ▸ EDITORIAL BAND — FROM THE EDITOR'S DESK *(130 words + table)*

**Standing head:** `FROM THE EDITOR'S DESK`
**Sub-head:** `The Fusion Layer`

**Headline:**

> ## How Three Desks Become One Decision — Without a Vote

**Body:**

> Most systems would average the three desks. AURORA does not average
> anything. Each input is assigned exactly one dimension of the output, and no
> dimension has two owners — so every number in the final recommendation has a
> single, traceable author. Direction comes from Strategy. Size, gross exposure
> and cash come from the risk model. The volatility budget comes from Health.
> The optimiser that assembles them is a long-only mean-variance solve under
> position caps, weekly-turnover caps and transaction costs. Afterwards, and
> only afterwards, an explanation layer renders one auditable record per
> holding — it reports the decision, it never revises it. If the explanation
> fails, the recommendation still stands.

**THE OWNERSHIP TABLE** *(the single most important object on the poster)*

| Input | Controls | Never does |
|---|---|---|
| **Daily Strategy** | Direction + all relative stock weights | — |
| **HAR-X + News risk** | Per-stock size, gross exposure, cash, covariance | Vote on direction |
| **Portfolio Health** | Volatility budget / risk aversion | Vote on direction |
| **News** | Reaches the decision **only** through the risk forecast | Vote on direction |

**PULL QUOTE** *(set large, in the band)*

> ### "The direction signal is not statistically distinguishable from noise. So we built a system that doesn't need it to be."

**HONESTY BOX — "What we measured before we trusted ourselves"** *(60 words)*

> We measured our own direction signal's information coefficient before letting
> it size anything. Across a 165-symbol panel it was −0.010 / −0.009; the
> positive value on our 21-symbol panel turned out to be a sector artifact —
> 17 of 21 names were tech, and neutralising sector removed it. Effective
> breadth is capped near 2.7 independent bets by an average pairwise
> correlation of 0.33. So we set the assumed IC to **0.02**, replacing a
> hardcoded value that implied **0.14** — roughly ten times anything we could
> measure.
>
> | Assumed IC | Sharpe (signal real) | Sharpe (signal shuffled) | Turnover | Cost/yr |
> |---|---|---|---|---|
> | **0.02** (chosen) | 1.076 | 1.025 | 0.64 | 0.16% |
> | 0.05 | 1.298 | 1.033 | 3.04 | 0.76% |
> | 0.14 (old) | 1.324 | 0.775 | 7.69 | 1.92% |
>
> Being too cautious costs about 0.25 Sharpe when the signal is real. Being too
> aggressive costs about 0.25 Sharpe **and** 1.9% a year in fees when it is
> not. We took the defensible side.

---

### ▸ SIDEBAR — THE RISK PAGE *(70 words + formulas)*

**Standing head:** `THE RISK PAGE`
**Sub-head:** `HAR-X + Filtered Historical Simulation`

**Headline:**

> ## The Only Model We Ever Promoted

**Body:**

> Everything the wire files arrives here. HAR-X forecasts volatility from log
> realised volatility over 5, 22 and 66 sessions, a Parkinson range estimator,
> and news attention. Filtered Historical Simulation then converts that
> forecast into downside risk using empirical quantiles of standardised
> returns — fat-tailed and left-skewed, calibrated on data through 2020 only.
> No normal distribution is assumed anywhere.

**Formulas** *(set in mono, 3 lines):*

```
σ̂_daily = σ̂_price · √(news_ratio)        news_ratio = exp(β · scaled log news count)
VaR_α    = σ̂_h · Q_α(z)                   Q_α = empirical quantile of standardised returns
ES_α     = σ̂_h · mean(z | z ≤ Q_α)        fat-tailed, left-skewed — no Gaussian assumption
```

**Evidence strip** *(set as four stat tiles):*

```
+5.24%      p = 0.0011      4.32%              96.34%
QLIKE       Diebold-        VaR-95 breach      95% band
gain vs     Mariano         (5% target)        coverage
price-only
```

Also worth one line: **news improved the five-session forecast but not the
twenty-session one, so we only ship it at five.** That asymmetry is a good
detail to be asked about.

**Figures:** FIG 2 (forecast vs realised), FIG 3 (FHS tail vs normal).

---

### ▸ THE MARKET TABLE *(agate, full width)*

Standing head: `THE MARKET TABLE — today's recommendation, holding by holding`

One row per holding, from `GET /recommendation/daily` → `fusion_results`.
Column spec:

| SYMBOL | ACTION | CURR. WT | TARGET WT | Δ | RISK LEVEL | σ̂ 5d | VaR-95 | NEWS | REASON |
|---|---|---|---|---|---|---|---|---|---|

Set in genuine agate — 7–8 pt at A1 is authentic and legible up close. This
table is the proof that the explanation layer is real and per-holding, so do
not summarise it or truncate to three rows. `[FILL: full table from a live
run]`

Footnote below the table, 1 line:

> Every row above was generated by the decision path itself, not reconstructed
> afterwards.

---

### ▸ ML INVENTORY *(two columns, footer band)*

Standing head: `THE MODELS — twelve validated, one promoted`

Left column `IN PRODUCTION` and right column `TESTED, NOT PROMOTED`, using the
tables in Part 5.2. Above them, the front four set large enough to read at 3 m:

```
HAR-X + News  ·  Filtered Historical Simulation  ·  FinBERT  ·  GARCH(1,1) Control
```

---

### ▸ NOT YET FIT TO PRINT *(boxed, 55 words)*

> **Known limits.** Our direction signal is weak, and we designed around that
> rather than hiding it. The XGBoost residual, the gated-news directional
> model, the multi-task MLP and the decision-layer candidates all failed
> promotion and are inert in production. The Next.js client cannot yet render
> the per-holding explanations. News is only validated at the five-session
> horizon.

---

### ▸ FOOTER

```
STAFF    [FILL: all names, alphabetical, no titles]
         Every engine on this page was built by the whole team, together.
STACK    FastAPI · PyTorch · XGBoost · scikit-learn · statsmodels/arch · scipy
         Streamlit · Next.js 16 · yfinance · feedparser · transformers
TESTS    48 passing · ~8s          RESEARCH  ~43 offline scripts · 22 reports
REPO     [QR code]
```

Set the staff box the way a masthead does — one block, names alphabetical, no
role labels. The second line matters: it is the only place the poster states
how the team worked, and it is a claim worth making explicitly.

---

## Part 8 — Fact-check list before printing

**Must be filled from a live run** (start the backend, hit the endpoints):

- [ ] Market Weather: regime, 20-day volatility, confidence — `GET /strategy/regime`
- [ ] Market Table: all rows — `GET /recommendation/daily` → `fusion_results`
- [ ] Health gauge value and what-if delta — `GET /health/report`
- [ ] Wire ticker headlines — real entries from `data/news_raw.json`, with real
      source attribution and timestamps. **Do not write plausible-sounding
      headlines.** If the store is thin, run
      `python backend/src/news_intelligence/collector.py` a few times first.
- [ ] Markets Desk figure caption — a real finding from `backtest()`
- [ ] Staff box names

**Already verified against code or reports — safe to print as written:**

- [x] HAR-X validation: +5.24% QLIKE, CI +2.38% to +8.83%, DM p=0.0011, 5/6
      years, 85.7% of stocks, +13.56% in high volatility
- [x] FHS backtest: VaR-95 breach 4.32%, band coverage 96.34%, ES ratio 0.983
- [x] IC study: 0.02 chosen vs 0.14 implied; the three-row Sharpe/turnover
      table; 165-symbol IC −0.010/−0.009; correlation 0.33/0.37; N_eff ≈ 2.7
- [x] Four regimes: bullish / bearish / high_volatility / sideways
- [x] `risk_model.json` is the only promoted artifact
- [x] FinBERT is `ProsusAI/finbert`, three-class, local, offline after first load
- [x] No Random Forest and no SHAP anywhere in the codebase

**Do not print:**

- The 50/30/20 blend (retired; `recommendation/README.md` is stale on this)
- `{strategy: 1.0, news: 0.0, health: 0.0}` without the "direction weights
  only" caption
- Any claim that we fine-tuned BERT — we integrated pre-trained weights and
  built the offline fallback around them
- Any suggestion that an `experimental_only` artifact is live

**A note on attribution.** We do not work by dividing engines between people —
we build each piece together, in one pass, then iterate. That is a deliberate
choice: it keeps engineering quality and the team's mental model of the system
consistent, and it means any of us can explain any part of the project. The
poster should reflect that rather than contradict it, so there are **no
per-desk bylines** — the desks are signed "By the AURORA staff" and every name
appears once in the staff box, which is standard practice for collectively
written newspaper copy.

One thing to be ready for: the per-developer ownership model is **historical**.
`CLAUDE.md` was updated on 2026-07-27 to say so, but each
`backend/src/<engine>/README.md` still opens with "Developer 1 — Portfolio
Intelligence Engine" and carries a "Split note" paragraph assigning files to a
person. A judge who opens the repo will see that and may ask who owns what. The
honest and better answer is the one above: those headers are a legacy of the
initial split, and the work has been collective since. Worth rewriting the five
README headers before the repo is shown, so the code and the poster tell the
same story.
