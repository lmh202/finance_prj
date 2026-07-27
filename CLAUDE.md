# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AURORA — a "portfolio intelligence copilot" built as a four-person group
capstone that has since grown a fifth engine and an offline research pipeline.

Three documents, in order of currency:

- `docs/AURORA-system-architecture.md` — **the current architecture picture**
  (verified 2026-07-26). Read this before non-trivial work.
- `backend/src/interfaces.py` — the frozen cross-engine contract.
- `Architecture.md` — the original product spec. Its formula appendices
  (§4 Health Score, §5 regime/percentile scoring, §6 news classification,
  §7 market-volatility factor) are still cited by engine code. Its top-level
  system diagram is obsolete.

The engine folders are hard module boundaries (see "Module boundaries" below).
They are **not** per-person assignments: the team builds each piece together
and iterates, so no engine belongs to one developer and any member can explain
any part. The `Developer N —` headers still sitting in each engine's
`README.md` are a leftover of the original split — don't treat them as current,
and don't attribute an engine to a person in anything user-facing.

## Commands

```bash
# Install all deps
pip install -r requirements.txt          # backend + Streamlit frontend
cd frontendjs && npm install             # Next.js frontend

# Start everything (one PowerShell command from repo root)
.\scripts\dev.ps1                        # backend (port 8000) + Streamlit (port 8501)

# Or run processes by hand:
uvicorn main:app --app-dir backend --reload --port 8000   # terminal 1: backend
streamlit run frontend/app.py                             # terminal 2: Streamlit UI
cd frontendjs && npm run dev                              # terminal 3: Next.js UI (port 3000)

# Standalone RSS collector (grows data/news_raw.json — run regularly)
python backend/src/news_intelligence/collector.py

# API docs (when backend is running)
# http://localhost:8000/docs
```

### Tests

`tests/` is a pytest suite (48 tests, ~8s). Run it from the repo root — the
test files insert `backend/` and `scripts/` onto `sys.path` themselves, so no
install step or `conftest.py` path hacking is needed.

```bash
python -m pytest tests/ -q                                  # whole suite
python -m pytest tests/test_gated_news_decision.py -q        # one file
python -m pytest tests/test_rule_fusion.py::test_strong_strategy_news_conflict_forces_hold -q   # one test
python -m pytest tests/ -q -k "risk_engine"                  # by keyword
```

Naming is historical, not structural: `test_rule_fusion*.py` covers
`src/recommendation/fusion.py` and `scripts/backtest_rule_fusion.py` — the
deleted `rule_fusion/` engine is gone and has no tests.

Sklearn/XGBoost `InconsistentVersionWarning`s on load are expected (artifacts
were pickled under older versions) and are not failures.

There is **no linter and no CI config** in the Python half of this repo — don't
assume `ruff`/`black`/etc. exist unless you add them. The `frontendjs/` Next.js
app **does** have `npm run lint` and `npm run typecheck`.

### Sandbox testing (safe mutation tests)

Set `AURORA_DATA_DIR` to a temporary directory so tests don't touch the
user's live `data/portfolio.csv`:

```powershell
$env:AURORA_DATA_DIR = "$pwd\tests\sandbox_data"
```

The backend reads all runtime state from that directory — portfolio,
settings, ticker cache, news cache. The Streamlit frontend honors
`AURORA_API_URL` (default `http://localhost:8000`).

## Environment variables

| Variable | Set by | Purpose |
|---|---|---|
| `AURORA_DATA_DIR` | you (optional) | Override `data/` root (default: `<repo>/data`). Use a sandbox dir for mutation tests so live state isn't touched. |
| `AURORA_API_URL` | `scripts/dev.ps1` | Backend URL the Streamlit frontend calls (default: `http://localhost:8000`). |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | you (optional) | Explanation-only LLM text in `recommendation/llm_client.py`, used on the fallback decision path. Missing key → deterministic template. See `.env.example`. |
| `NEXT_PUBLIC_BACKEND_URL` | `.env.local` (tracked) | Backend URL the Next.js frontend calls (default: `http://localhost:8000`). |

`anthropic>=0.40` is declared in `backend/requirements.txt` for a planned live
LLM classification pass, but **no code imports it** — there is no
`ANTHROPIC_API_KEY` code path today. Live news sentiment comes from local
FinBERT (see below).

## Architecture: FastAPI backend + Streamlit frontend (HTTP-only split)

The repo is split into processes that talk **only over HTTP**. No
`streamlit` import anywhere under `backend/`; no `src` import anywhere under
`frontend/`.

| Process | Port | Tech |
|---|---|---|
| Backend API | 8000 | FastAPI (uvicorn) |
| Streamlit UI | 8501 | Streamlit |
| Next.js UI | 3000 | Next.js 16 (Turbopack) |

```
backend/                     FastAPI service
  main.py                    entry point — uvicorn main:app --app-dir backend
  serialize.py               dataclass/DataFrame → JSON helpers (df_records, df_split, as_dict)
  routers/                   8 routers: one per contract engine (health, strategy, news,
                              recommendation, risk), plus shared-kernel routers
                              (market, portfolio) and analysis (see below)
    _common.py               shared request plumbing: load holdings → bail markers (409/502);
                              also refresh_news_store() — see "daily decision path" below
  src/
    interfaces.py            THE FROZEN CONTRACT
    config.py                DATA_DIR resolution (AURORA_DATA_DIR override)
    data_loader.py           shared kernel: NASDAQ symbol universe + prices + history
    portfolio.py             shared kernel: holdings CSV persistence + valuation
    portfolio_health/        compute_health, what_if_health  (+ backs the Performance page)
    daily_strategy/          classify_regime, score_assets, backtest (walk-forward ML)
    news_intelligence/       fetch_headlines, essential_news, sentiment_features
    recommendation/          the production decision + its explanation layer
    risk_engine/             added after the contract froze — risk_estimate(s), portfolio_risk
                              (HAR-X + News)
    analysis/                Auxiliary engine (NOT a contract engine) — calendar alignment,
                              constant-mix portfolio construction, risk stats. Ported from
                              frontendjs' old src/lib/metrics.ts; used ONLY by
                              routers/analysis.py's POST /analysis/explore, which serves the
                              frontendjs /portfolio analytics tab — NOT the Streamlit frontend.

frontend/                    Streamlit UI
  app.py                     Home — portfolio builder
  api_client.py              typed HTTP wrappers over the backend API (the ONLY data-access path)
  views/                     one view per engine page
  pages/                     6-line routing shims (1–6) — NEVER edit these (edit views/ instead)

scripts/                     ~43 offline research scripts + dev.ps1 (start backend + frontend)
reports/                     offline study write-ups (17 dirs + 5 standalone reports)
tests/                       pytest suite
data/                        gitignored runtime state + committed fixtures
  processed/                 trained artifacts + candidates (see promotion gate below)
```

### The five engines and their contract functions

`backend/src/interfaces.py` defines every cross-engine dataclass
(`HealthReport`, `RegimeState`, `AssetSignal`, `NewsEvent`, `ReactionRisk`,
`ProposedTrade`, `Recommendation`) and the exact function signatures below.
It is **frozen** — changing it requires whole-team agreement.
`RiskEstimate` / `PortfolioRisk` live in `risk_engine/engine.py` instead, since
that engine postdates the frozen contract.

| Engine | Functions | Technique |
|---|---|---|
| `portfolio_health/` | `compute_health()`, `what_if_health()` | Quant formulas + validation study |
| `daily_strategy/` | `classify_regime()`, `score_assets()`, `backtest()` | Rule-based (production) + scikit-learn walk-forward ML candidate |
| `news_intelligence/` | `fetch_headlines()`, `essential_news()`, `sentiment_features()` | Keyword rules + FinBERT (live), FinBERT/VADER (batch) |
| `recommendation/` | `reaction_risk()`, `recommend_daily()`, `recommend_event()`, `apply_constraints()` | Constrained optimisation + explanation layer |
| `risk_engine/` | `risk_estimate()`, `risk_estimates()`, `portfolio_risk()`, `model_available()` | HAR-X volatility + Filtered Historical Simulation |

Each `backend/src/<engine>/` folder has its own `README.md` with the mission,
contract, and definition-of-done — check it before working inside that folder.
Read past their `Developer N —` headers and their "Split note" paragraphs; the
mission/contract/DoD content is current, the per-person framing is not.

### The daily decision path (the thing that spans the most files)

`GET /recommendation/daily` in `routers/recommendation.py` is the single
user-facing decision. Each input controls exactly one dimension:

| Input | Controls | Never does |
|---|---|---|
| Daily Strategy | direction + all relative stock weights | — |
| HAR-X + News risk | per-stock size, gross exposure, cash, covariance | vote on direction |
| Portfolio Health | the volatility budget / risk aversion | vote on direction |
| Market stress (SPY 60d realised-vol percentile) | the base risk aversion only | vote on direction |
| News | reaches the decision **only** through the risk forecast | cast a directional vote |

Three properties of this are easy to break:

1. **News must be collected before risk is estimated.** `risk_engine` derives
   its causal news-attention input from `data/news_raw.json`. Refreshing the
   store after the risk call makes the model score the *previous* request's
   headlines, and produces no overlay at all on a cold start. The daily
   endpoint gets this by calling `essential_news()` first (it collects
   internally); `/risk/*` calls `routers/_common.refresh_news_store()`.
   The collector throttles per feed, so it's free when warm.
2. **`fusion.explain_allocation()` explains, it does not decide.**
   `gated_news.recommend_strategy_risk_control()` produces the numbers; fusion
   renders one `AssetFusionResult` per holding from that already-fixed
   decision. It is wrapped in its own try/except: an explanation failure must
   empty `fusion_results` and set `explanation_meta.fusion_error`, never
   discard a valid recommendation.
3. **The market-stress signal needs its own, longer history.** It ranks a
   60-session realised volatility over 504 observations of itself, so it needs
   ~564 sessions — the two-year frame from `load_holdings_history()` is not
   enough and would silently produce a *different* signal. Use
   `routers/_common.load_benchmark_close()` (5y, memoised 6h), never
   `history["SPY"]`. An unknown state falls back to the stressed setting, so a
   broken fetch narrows the risk budget rather than widening it.

Fallback chain: gated-news / strategy-risk-control → `decision.recommend_portfolio`
(+ DeepSeek text) → `engine.recommend_daily` (legacy signal path).

Retired but still on disk: `fusion.recommend_portfolio()` / `fuse_scores()`
(the 50/30/20 blended score). No router calls it; only
`scripts/backtest_rule_fusion.py` and `tests/test_rule_fusion.py` do. Don't
wire it back in.

### The promotion gate

Nothing under `scripts/`, `reports/`, or `data/processed/` runs as part of the
live app. Engines load a trained artifact only after checking its
`metadata.json` for `promotion_status: promoted`; anything marked
`experimental_only` is ignored at request time and the deterministic path stays
production. **Never assume an artifact is live just because it exists on disk.**

Currently promoted: `risk_model.json` only — and it's the exception that isn't
gated, because it is `risk_engine`'s only inference path with no deterministic
fallback. The strategy XGBoost residual, the decision-model candidates, and the
gated-news directional residual are all `experimental_only`.

### Module boundaries

These survive from the original one-branch-per-developer split, but they are
architectural rules, not social ones — they hold regardless of who is editing.

1. Treat each `backend/src/<engine>/` folder as encapsulated — reach into
   another engine only through its public contract functions, never through
   private helpers.
2. `backend/src/interfaces.py`, `backend/src/data_loader.py`, and
   `backend/src/portfolio.py` are the **shared kernel**: read-only from
   inside an engine folder. Changes to them are cross-cutting — flag them
   explicitly rather than making them silently.
3. `frontend/pages/*.py` are routing shims only — real page logic lives in
   each engine's view under `frontend/views/`.
4. New pip dependency → add it to the correct `requirements.txt`
   (`backend/requirements.txt` or `frontend/requirements.txt`) and say so.
5. API keys come from env vars or the gitignored `.streamlit/secrets.toml` —
   never hardcode or commit one.

### Router and view layer

Each engine exposes its contract functions through a thin FastAPI router
(`backend/routers/<engine>.py`) that calls the engine, converts results to
JSON via `serialize.py`, and returns them. The frontend consumes these
endpoints through `frontend/api_client.py` — a set of typed wrapper functions
that translate JSON responses back to DataFrames/dicts.

Two marker details signal expected states across the HTTP boundary:
- `empty_portfolio` (409) — no holdings yet, show onboarding UI
- `no_history` (502) — market data unavailable, show a retry message

The frontend's `api_client.py` raises `ApiUnavailable` when the backend is
down and `ApiMarker` for these two expected conditions.

### How news reaches the ML model (cross-engine data flow)

News flows on **two timescales**:

- **Slow / training**: `sentiment_features()` produces a long-format,
  look-ahead-safe table (`date, symbol, sentiment, news_count, has_news`)
  scored by a local model (FinBERT/VADER). It's an *optional* input to
  `score_assets(history, holdings, sentiment=None)` — days with no news get
  `sentiment=0, has_news=0`. The required ablation is price-only vs.
  price+news, validated with walk-forward splits only (never a random shuffle).
- **Fast / decision-time**: `essential_news()` refreshes `data/news_raw.json`,
  which is the news-attention input to `risk_engine`'s HAR-X model — that is
  how live news changes the formal recommendation. It also supplies context and
  `as_of` timestamps to `fusion.explain_allocation()`. A direct directional
  news vote exists in `gated_news.py` but is `experimental_only`.

Live sentiment scoring is local FinBERT (`ProsusAI/finbert`, weights cached to
`<DATA_DIR>/models/finbert/`) with a keyword-rule fallback when the model can't
load — see `news_intelligence/finbert_sentiment.py` and `analyzer.py`.
Degraded store qualities (`missing_store` / `stale_store` / `invalid_store`)
are deliberately **not** treated as an observed zero-news state: they surface
as `news_applied=False` plus a quality string, so a broken RSS feed is visible
instead of resembling a calm market.

### Shared kernel data shapes (used by every engine)

- `holdings` — `pd.DataFrame` from `backend.src.portfolio.load_portfolio()`:
  columns `symbol, name, shares, buy_price`.
- `prices` — `Dict[str, float]` from `backend.src.data_loader.get_latest_prices()`:
  latest close per symbol; a symbol may be **missing** — always handle that.
- `history` — `pd.DataFrame` from `backend.src.data_loader.get_history()`:
  daily adjusted close, `DatetimeIndex`, one column per symbol; columns may
  be **missing** for some symbols.
- `weights` — `Dict[str, float]` derived from `portfolio.build_view()`:
  current weight in percent per symbol (cash excluded).

`backend/src/data_loader.py` also owns the ticker universe: the full NASDAQ
Trader symbol directory (NASDAQ + NYSE/NYSE American/Arca/Cboe/IEX), cached
to `data/tickers.csv` for 7 days, with a small hardcoded `FALLBACK_UNIVERSE`
if the network and cache both fail. `to_yahoo_symbol()` translates directory
symbols (`BRK.B`) to yfinance symbols (`BRK-B`).

### Data directory

`data/` holds gitignored runtime state (`portfolio.csv`, `settings.json`,
`tickers.csv`, `news_raw.json` cache) alongside committed fixtures
(`sample_portfolio.csv`, `tickers.csv` seed). `data/processed/` holds trained
artifacts and candidates; the historical sentiment feature table is cached
there too (built once, read many times — don't rebuild it from raw corpora on
every run).

## Separate projects in this repo

### `frontendjs/` — Aurora (Next.js portfolio analytics)

A **tracked, separate** Next.js 16 App Router app with its own `CLAUDE.md`.
It sits outside the AURORA Python system's module boundaries (no engine folder,
no contract functions) but it is no longer fully independent either:
it used to run its own PostgreSQL database and Next.js API routes, but that
was ripped out — it now has **no database and no server-side API routes of
its own**. Every data operation goes through `frontendjs/src/lib/api-client.ts`,
a typed fetch client that calls this repo's FastAPI backend directly at
`NEXT_PUBLIC_BACKEND_URL` (default `http://localhost:8000`).
`backend/main.py` sets CORS to allow `http://localhost:3000` specifically
for this.

The app has two kinds of pages:
- `/` — a clean home/search entry point; picking a symbol routes to
  `/portfolio?add=SYMBOL`.
- Engine pages ported from `frontend/`'s Streamlit views: `/portfolio`
  (builder + analytics via `/portfolio*` plus `/analysis/explore` in shares
  mode), `/health`, `/strategy`, `/news`, `/react`, `/performance` and
  `/risk`. These read the same `data/portfolio.csv` the Streamlit frontend
  edits — the `empty_portfolio` / `no_history` markers are surfaced through
  `ApiMarkerError` in the api-client.

Known gap: `/react` calls `/recommendation/daily` but its `DailyRecommendation`
type has no `fusion_results` field, so the per-asset explanation is
Streamlit-only today.

See `frontendjs/CLAUDE.md` for the app's own architecture notes.

### Other top-level directories

`aurora_poster/` and `hdbrain_poster/` are self-contained HTML/CSS posters with
their own build scripts. `backend/t1/` is reference-only (its algorithms were
ported into `news_intelligence/`). `ui_design/` is a superseded, dead Next.js
attempt — don't edit it.
