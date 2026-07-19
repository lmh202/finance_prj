# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AURORA — a "portfolio intelligence copilot" built as a four-person group
capstone. Full product spec lives in `Architecture.md`; the day-to-day
contract between the four workstreams lives in `backend/src/interfaces.py`.
Read both before making non-trivial changes — this repo enforces a strict
ownership model, and violating it breaks other developers' branches.

## Commands

```bash
# Install all deps (backend + frontend)
pip install -r requirements.txt

# Start everything (one PowerShell command)
.\scripts\dev.ps1

# Or run the two processes by hand:
uvicorn main:app --app-dir backend --reload --port 8000   # terminal 1
streamlit run frontend/app.py                             # terminal 2

# Standalone RSS collector (grows data/news_raw.json)
python backend/src/news_intelligence/collector.py

# API docs (when backend is running)
# http://localhost:8000/docs
```

There is no test suite, linter, or CI config in this repo — don't assume
`pytest`/`ruff`/etc. exist unless you add them yourself.

## Architecture: FastAPI backend + Streamlit frontend (HTTP-only split)

The repo is split into two processes that talk **only over HTTP**. No
`streamlit` import anywhere under `backend/`; no `src` import anywhere under
`frontend/`.

```
backend/                     FastAPI service
  main.py                    entry point — uvicorn main:app --app-dir backend
  serialize.py               dataclass/DataFrame → JSON helpers (df_records, df_split, as_dict)
  routers/                   one thin router per owned engine (health, strategy, news, recommendation),
                              plus shared-kernel routers (market, portfolio) and analysis (see below)
    _common.py               shared request plumbing: load holdings → bail markers (409/502)
  src/
    interfaces.py            THE FROZEN CONTRACT (custodian: Developer 4)
    config.py                DATA_DIR resolution (AURORA_DATA_DIR override)
    data_loader.py           shared kernel: NASDAQ symbol universe + prices + history
    portfolio.py             shared kernel: holdings CSV persistence + valuation
    portfolio_health/        Developer 1 — compute_health, what_if_health  (+ owns Performance page)
    daily_strategy/          Developer 2 — classify_regime, score_assets, backtest (walk-forward ML)
    news_intelligence/       Developer 3 — fetch_headlines, essential_news, sentiment_features
    recommendation/          Developer 4 — reaction_risk, recommend_daily, recommend_event, apply_constraints
    analysis/                NOT one of the four contract engines — calendar alignment, constant-mix
                              portfolio construction, and risk stats. Ported from frontendjs' old
                              src/lib/metrics.ts; used only by routers/analysis.py's /analysis/explore,
                              which serves frontendjs (see "Separate projects" below), not frontend/.

frontend/                    Streamlit UI
  app.py                     Home — portfolio builder (Developer 4)
  api_client.py              typed HTTP wrappers over the backend API (the ONLY data-access path)
  views/                     one view per engine page (owned per developer)
    _common.py               shared Streamlit helpers
  pages/                     6-line routing shims — NEVER edit these (edit views/ instead)

scripts/dev.ps1              start backend (new window) + frontend (this window)
data/                        gitignored runtime state + committed fixtures
  processed/                 Dev 3's historical sentiment feature table (built once, read many times)
```

### The four engines and their contract functions

`backend/src/interfaces.py` defines every cross-engine dataclass
(`HealthReport`, `RegimeState`, `AssetSignal`, `NewsEvent`, `ReactionRisk`,
`ProposedTrade`, `Recommendation`) and the exact function signatures below.
It is **frozen** — changing it requires agreement across all four developers.

| Engine | Owner | Functions | Technique |
|---|---|---|---|
| `portfolio_health/` | Dev 1 | `compute_health()`, `what_if_health()` | Quant formulas + validation |
| `daily_strategy/` | Dev 2 | `classify_regime()`, `score_assets()`, `backtest()` | scikit-learn walk-forward ML |
| `news_intelligence/` | Dev 3 | `fetch_headlines()`, `essential_news()`, `sentiment_features()` | LLM API (live) + FinBERT/VADER (batch) |
| `recommendation/` | Dev 4 | `reaction_risk()`, `recommend_daily()`, `recommend_event()`, `apply_constraints()` | Decision formulas + integration |

Each `backend/src/<engine>/` folder has its own `README.md` with the mission,
contract, and definition-of-done — check it before working inside that folder.

### Ownership rules (one git branch per developer)

1. Treat each `backend/src/<engine>/` folder as owned — don't edit another
   engine's internals. Cross-engine calls go through the public contract
   functions, never through private helpers.
2. `backend/src/interfaces.py`, `backend/src/data_loader.py`, and
   `backend/src/portfolio.py` are the **shared kernel**: read-only from
   inside an engine folder. Changes to them are cross-cutting — flag them
   explicitly rather than making them silently.
3. `frontend/pages/*.py` are routing shims only — real page logic lives in
   each engine's view under `frontend/views/`.
4. New pip dependency → add it to the correct `requirements.txt`
   (`backend/requirements.txt` or `frontend/requirements.txt`) and say so.
5. `ANTHROPIC_API_KEY` (used by `news_intelligence` for live LLM
   classification) comes from an env var or the gitignored
   `.streamlit/secrets.toml` — never hardcode or commit it.

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

News flows on **two timescales**, and this is where the four engines'
contracts chain together:

- **Slow / training**: `sentiment_features()` produces a long-format,
  look-ahead-safe table (`date, symbol, sentiment, news_count, has_news`)
  scored by a local model (FinBERT/VADER). It's an *optional* input to
  `score_assets(history, holdings, sentiment=None)` — days with no news get
  `sentiment=0, has_news=0`. Dev 2's required ablation is price-only vs.
  price+news, validated with walk-forward splits only (never a random
  shuffle).
- **Fast / decision-time**: live events from `essential_news()` flow into
  `reaction_risk()`. Its `priced_in` factor must rise when Dev 2's
  sentiment-tilted scores already reflect a story, so the system never
  reacts twice to the same news. This reconciliation is the main
  intellectual problem in `backend/src/recommendation/engine.py`.

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
(`sample_portfolio.csv`, `tickers.csv` seed). `data/processed/` is where
Dev 3's historical sentiment feature table is cached (built once, read
many times — don't rebuild it from raw corpora on every run).

## Separate projects in this repo

### `frontendjs/` — Aurora (Next.js portfolio analytics)

A **tracked, separate** Next.js 16 App Router app with its own `CLAUDE.md`.
It is not part of the AURORA Python system's ownership model (no engine
folder, no Developer owns it) but it is no longer fully independent either:
it used to run its own PostgreSQL database and Next.js API routes, but that
was ripped out — it now has **no database and no server-side API routes of
its own**. Every data operation goes through `frontendjs/src/lib/api-client.ts`,
a typed fetch client that calls this repo's FastAPI backend directly at
`NEXT_PUBLIC_BACKEND_URL` (default `http://localhost:8000`).
`backend/main.py` sets CORS to allow `http://localhost:3000` specifically
for this.

The app has two kinds of pages:
- `/` — a clean home/search entry point; picking a symbol routes to
  `/portfolio?add=SYMBOL` (the old localStorage analyzer was merged into
  `/portfolio` and no longer exists).
- Engine pages ported from `frontend/`'s Streamlit views: `/portfolio`
  (builder + analytics for the backend-saved portfolio via `/portfolio*`
  plus `/analysis/explore` in shares mode), `/health`, `/strategy`,
  `/news`, `/react` and `/performance`, consuming the four engine routers
  (`/health/report`, `/strategy/*`, `/news/*`, `/recommendation/*`).
  These read the same `data/portfolio.csv` the Streamlit frontend edits —
  the `empty_portfolio` / `no_history` markers are surfaced through
  `ApiMarkerError` in the api-client.

See `frontendjs/CLAUDE.md` for the app's own architecture notes.
