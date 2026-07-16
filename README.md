# AURORA — AI-Powered Portfolio Intelligence Copilot

Group capstone project. Full design: [Architecture.md](Architecture.md).

## Run it

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Home page = portfolio builder (search ~13,000 US-listed securities, add
holdings, live valuation via yfinance, CSV import/export). The sidebar links
each engine's page.

## Team structure — one folder per developer

**Collaboration happens ONLY through [`src/interfaces.py`](src/interfaces.py)** —
the frozen contract defining every cross-engine type and required function
signature. Read it first. It changes only by agreement of all four developers
(Developer 4 is its custodian).

| Folder | Owner | One-line goal | Technique | Pages |
|---|---|---|---|---|
| [`src/portfolio_health/`](src/portfolio_health/) | Developer 1 | Metrics + 0–100 Health Score, **validated empirically**, plus what-if analysis of proposed trades | Quant formulas + validation study | Portfolio Health, Performance |
| [`src/daily_strategy/`](src/daily_strategy/) | Developer 2 | Regime classification + daily asset scores from a **walk-forward-validated ML model**, with news sentiment as an optional feature (ablation) | scikit-learn (predictive ML) | Daily Strategy |
| [`src/news_intelligence/`](src/news_intelligence/) | Developer 3 | Live RSS → **LLM-classified** essential events (≤5/day), plus the **historical sentiment feature table** that feeds Dev 2's ML | LLM API (live) + FinBERT/VADER (batch) | Essential News |
| [`src/recommendation/`](src/recommendation/) | Developer 4 | Fuse all engines into daily + event recommendations **without double-counting news** (`priced_in` reconciliation), under §9 constraints; own the product | Decision formulas + integration | Should I React?, Home |

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

1. **Commit only inside your own folder** (one git branch each, e.g.
   `dev2-daily-strategy`). Exceptions: `requirements.txt` (announce first);
   Developer 4 also maintains `app/` and the shared kernel.
2. **Never import another engine's internals** — only its contract functions
   and the types in `src/interfaces.py`.
3. The shared kernel is read-only for everyone:
   - `src/data_loader.py` — ticker universe (NASDAQ symbol directory),
     `get_latest_prices()`, `get_history()`
   - `src/portfolio.py` — holdings schema, persistence, `build_view()`
4. `app/pages/*.py` are 6-line routing shims — never edit them; edit the
   `page.py` in your own folder instead.
5. API keys (`ANTHROPIC_API_KEY`) live in env vars or `.streamlit/secrets.toml`
   — never in code, never committed.

## Layout

```
app/
  app.py                     Home — portfolio builder (Developer 4)
  pages/                     routing shims only (DO NOT EDIT)
src/
  interfaces.py              THE CONTRACT — frozen (custodian: Developer 4)
  data_loader.py             shared kernel: universe + prices + history
  portfolio.py               shared kernel: holdings + valuation
  portfolio_health/          Developer 1  (engine.py, page.py, README.md)
  daily_strategy/            Developer 2  (engine.py, page.py, README.md)
  news_intelligence/         Developer 3  (engine.py, page.py, README.md)
  recommendation/            Developer 4  (engine.py, page.py, README.md)
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
