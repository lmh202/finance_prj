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
signature. Read it first. It changes only by agreement of all four developers.

| Folder | Owner | Engine | Dashboard pages |
|---|---|---|---|
| [`src/portfolio_health/`](src/portfolio_health/) | Developer 1 | Portfolio Intelligence (§4) | Portfolio Health |
| [`src/daily_strategy/`](src/daily_strategy/) | Developer 2 | Regime-Aware Momentum (§5) | Daily Strategy, Performance |
| [`src/news_intelligence/`](src/news_intelligence/) | Developer 3 | Event Intelligence / RSS (§6) | Essential News |
| [`src/recommendation/`](src/recommendation/) | Developer 4 | Reaction Risk + Recommendation (§7–9) | Should I React?, Home/integration |

Each folder has a **README.md** with your mission, your frozen contract, and a
definition-of-done checklist. Working baselines are already in place — the app
runs end-to-end today; your job is to replace baseline logic with the real
thing without changing signatures.

### Ground rules

1. **Commit only inside your own folder** (one git branch each, e.g.
   `dev1-portfolio-health`). Exceptions: `requirements.txt` (announce first);
   Developer 4 also maintains `app/` and the shared kernel.
2. **Never import another engine's internals** — only its contract functions
   and the types in `src/interfaces.py`.
3. The shared kernel is read-only for everyone:
   - `src/data_loader.py` — ticker universe (NASDAQ symbol directory),
     `get_latest_prices()`, `get_history()`
   - `src/portfolio.py` — holdings schema, persistence, `build_view()`
4. `app/pages/*.py` are 6-line routing shims — never edit them; edit the
   `page.py` in your own folder instead.

## Layout

```
app/
  app.py                     Home — portfolio builder (Developer 4)
  pages/                     routing shims only (DO NOT EDIT)
src/
  interfaces.py              THE CONTRACT — frozen
  data_loader.py             shared kernel: universe + prices + history
  portfolio.py               shared kernel: holdings + valuation
  portfolio_health/          Developer 1  (engine.py, page.py, README.md)
  daily_strategy/            Developer 2  (engine.py, page.py, README.md)
  news_intelligence/         Developer 3  (engine.py, page.py, README.md)
  recommendation/            Developer 4  (engine.py, page.py, README.md)
data/                        caches + user portfolio (gitignored) + sample
```

## Data sources

- **Ticker universe**: official NASDAQ Trader symbol directory (all NASDAQ /
  NYSE / NYSE American / Arca / Cboe / IEX listings), cached 7 days in
  `data/tickers.csv`.
- **Prices**: Yahoo Finance via yfinance (latest + daily history).
- **News** (pending): RSS feeds, see `src/news_intelligence/engine.py`.
