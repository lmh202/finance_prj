# AURORA — AI-Powered Portfolio Intelligence Copilot

Group capstone project. Full design: [Architecture.md](Architecture.md).

## Run it

Two processes: the FastAPI backend and the Streamlit frontend. The frontend
talks to the backend over HTTP only (`frontend/api_client.py`); it never
imports backend code, and nothing under `backend/` imports streamlit.

```powershell
pip install -r requirements.txt
.\scripts\dev.ps1          # starts both (backend window + streamlit here)
```

Or by hand, from the repo root:

```bash
uvicorn main:app --app-dir backend --reload --port 8000   # terminal 1
streamlit run frontend/app.py                             # terminal 2
```

Interactive API docs: http://localhost:8000/docs. The frontend reads
`AURORA_API_URL` (default `http://localhost:8000`); the backend reads
`AURORA_DATA_DIR` (default `<repo>/data`).

Home page = portfolio builder (search ~13,000 US-listed securities, add
holdings, live valuation via yfinance, CSV import/export). The sidebar links
each engine's page.

## Team structure — one folder per developer

**Collaboration happens ONLY through
[`backend/src/interfaces.py`](backend/src/interfaces.py)** — the frozen
contract defining every cross-engine type and required function signature.
Read it first. It changes only by agreement of all four developers
(Developer 4 is its custodian).

Each developer now owns **three files**: the engine
(`backend/src/<engine>/engine.py` — pure logic, frozen signatures), the
router (`backend/routers/<engine>.py` — thin JSON translation, no logic),
and the view (`frontend/views/<engine>.py` — presentation, HTTP-only via
`api_client`).

| Engine folder | Owner | One-line goal | Technique | Pages |
|---|---|---|---|---|
| [`backend/src/portfolio_health/`](backend/src/portfolio_health/) | Developer 1 | Metrics + 0–100 Health Score, **validated empirically**, plus what-if analysis of proposed trades | Quant formulas + validation study | Portfolio Health, Performance |
| [`backend/src/daily_strategy/`](backend/src/daily_strategy/) | Developer 2 | Regime classification + daily asset scores from a **walk-forward-validated ML model**, with news sentiment as an optional feature (ablation) | scikit-learn (predictive ML) | Daily Strategy |
| [`backend/src/news_intelligence/`](backend/src/news_intelligence/) | Developer 3 | Live RSS → **LLM-classified** essential events (≤5/day), plus the **historical sentiment feature table** that feeds Dev 2's ML | LLM API (live) + FinBERT/VADER (batch) | Essential News |
| [`backend/src/recommendation/`](backend/src/recommendation/) | Developer 4 | Fuse all engines into daily + event recommendations **without double-counting news** (`priced_in` reconciliation), under §9 constraints; own the product | Decision formulas + integration | Should I React?, Home |

Each folder has a **README.md** with the mission, frozen contract, and a
definition-of-done checklist. Working baselines are in place — the app runs
end-to-end today; replace baseline logic without changing signatures.

### How news meets the ML (the agreed design)

News enters the system on **two timescales**:
- **Slow (training)**: Dev 3's historical `sentiment_features` table is an
  *optional* input channel to Dev 2's model (no news day = neutral +
  `has_news=0`). Dev 2's ablation (price-only vs price+news) is a headline
  result either way.
- **Fast (decision time)**: live essential events flow to Dev 4's
  reaction-risk formula, whose `priced_in` factor must rise when the model's
  sentiment-tilted scores already reflect the story — the system never reacts
  twice to the same news.

### Ground rules

1. **Commit only inside your own files** (one git branch each, e.g.
   `dev2-daily-strategy`): your engine, your router, your view. Exceptions:
   the requirements files (announce first); Developer 4 also maintains
   `frontend/app.py`, `backend/main.py` and the shared kernel.
2. **Never import another engine's internals** — only its contract functions
   and the types in `backend/src/interfaces.py`.
3. **The split is absolute**: no `streamlit` import anywhere under
   `backend/`; no `src`/backend import anywhere under `frontend/` — views
   talk to the API through `frontend/api_client.py` only.
4. The shared kernel is read-only for everyone:
   - `backend/src/data_loader.py` — ticker universe (NASDAQ symbol
     directory), `get_latest_prices()`, `get_history()`
   - `backend/src/portfolio.py` — holdings schema, persistence, `build_view()`
5. `frontend/pages/*.py` are 6-line routing shims — never edit them; edit
   your view in `frontend/views/` instead.
6. API keys (`ANTHROPIC_API_KEY`) live in env vars or `.streamlit/secrets.toml`
   — never in code, never committed.

## Layout

```
backend/                     FastAPI service (no streamlit anywhere)
  main.py                    API entry point — uvicorn main:app --app-dir backend
  serialize.py               dataclass/DataFrame -> JSON helpers
  routers/                   one thin router per engine + market/portfolio
  src/
    interfaces.py            THE CONTRACT — frozen (custodian: Developer 4)
    config.py                DATA_DIR resolution (AURORA_DATA_DIR override)
    data_loader.py           shared kernel: universe + prices + history
    portfolio.py             shared kernel: holdings + valuation
    portfolio_health/        Developer 1  (engine.py, README.md)
    daily_strategy/          Developer 2  (engine.py, README.md)
    news_intelligence/       Developer 3  (engine.py, collector.py, README.md)
    recommendation/          Developer 4  (engine.py, README.md)
frontend/                    Streamlit UI (backend access via HTTP only)
  app.py                     Home — portfolio builder (Developer 4)
  api_client.py              typed wrappers over the backend API
  views/                     one view per engine page (owned per developer)
  pages/                     routing shims only (DO NOT EDIT)
scripts/dev.ps1              start backend + frontend together
data/                        caches + user portfolio (gitignored) + sample
  processed/                 Dev 3's historical sentiment feature table
```

## Data sources

- **Ticker universe**: official NASDAQ Trader symbol directory (all NASDAQ /
  NYSE / NYSE American / Arca / Cboe / IEX listings), cached 7 days.
- **Prices**: Yahoo Finance via yfinance (latest + daily history; expand the
  asset universe here for ML training).
- **Live news**: RSS feeds (MarketWatch, CNBC, Yahoo Finance) + Anthropic LLM
  API for classification/sentiment/summaries.
- **Historical news**: FNSPID or a Kaggle financial-news corpus, scored
  locally (FinBERT/VADER) into `sentiment_features` — no look-ahead.
