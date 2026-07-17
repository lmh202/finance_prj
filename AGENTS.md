# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

AURORA — a Streamlit "portfolio intelligence copilot" built as a four-person
group capstone. Full product spec lives in `Architecture.md`; the day-to-day
contract between the four workstreams lives in `src/interfaces.py`. Read
both before making non-trivial changes — this repo enforces a strict
ownership model described below, and violating it breaks other developers'
branches.

## Commands

```bash
pip install -r requirements.txt
streamlit run app/app.py       # run the whole app
python src/news_intelligence/collector.py   # run the RSS collector standalone (grows data/news_raw.json)
```

There is no test suite, linter, or CI config in this repo — don't assume
`pytest`/`ruff`/etc. exist unless you add them yourself.

## Architecture: four engines behind one frozen contract

`src/interfaces.py` is the **only** channel through which the four
workstreams talk to each other. It defines every cross-engine dataclass
(`HealthReport`, `RegimeState`, `AssetSignal`, `NewsEvent`, `ReactionRisk`,
`ProposedTrade`, `Recommendation`) and the exact function signature each
engine must expose. It is frozen — changing it requires agreement across all
four developer roles, not a unilateral edit.

```
src/
  interfaces.py              THE CONTRACT (frozen)
  data_loader.py              shared kernel: ticker universe + prices + history (read-only for engines)
  portfolio.py                shared kernel: holdings CSV persistence + valuation (read-only for engines)
  portfolio_health/           Dev 1: compute_health, what_if_health  (+ owns the Performance page)
  daily_strategy/              Dev 2: classify_regime, score_assets, backtest  (walk-forward ML)
  news_intelligence/           Dev 3: fetch_headlines, essential_news, sentiment_features (live LLM + historical local-model)
  recommendation/              Dev 4: reaction_risk, recommend_daily, recommend_event, apply_constraints
app/
  app.py                       Home page — portfolio builder (owned by Dev 4)
  pages/*.py                   6-line routing shims that import each engine's page.py — NEVER edit these directly
```

Each `src/<engine>/` folder has its own `README.md` with that role's mission,
contract, and definition-of-done — check it before working inside that
folder.

### Ownership rules (this repo is organized as one branch per developer)

1. Treat each `src/<engine>/` folder as owned — don't edit another engine's
   `engine.py`/`page.py` internals. Cross-engine calls go through the public
   contract functions in `src/interfaces.py`, never through another engine's
   private helpers.
2. `src/interfaces.py`, `src/data_loader.py`, and `src/portfolio.py` are the
   shared kernel: read-only from inside an engine folder. Changes to them
   are cross-cutting and should be flagged explicitly rather than made
   silently.
3. `app/pages/*.py` are routing shims only — real page logic lives in each
   engine's own `page.py`.
4. New pip dependency → add it to `requirements.txt` and say so; don't add
   silently.
5. `ANTHROPIC_API_KEY` (used by `news_intelligence` for live LLM
   classification) comes from an env var or the gitignored
   `.streamlit/secrets.toml` — never hardcode or commit it.

### How news reaches the ML model (cross-engine data flow)

News flows on two timescales, and this is the one place the four engines'
contracts actually chain together:

- **Slow / training**: `news_intelligence.sentiment_features()` produces a
  long-format, look-ahead-safe table (`date, symbol, sentiment, news_count,
  has_news`) scored by a local model (FinBERT/VADER). It's an *optional*
  input to `daily_strategy.score_assets(history, holdings, sentiment=None)`
  — days with no news get `sentiment=0, has_news=0`. Dev 2's required
  ablation is price-only vs. price+news, validated with walk-forward splits
  only (never a random shuffle).
- **Fast / decision-time**: live events from `news_intelligence.essential_news()`
  flow into `recommendation.reaction_risk()`. Its `priced_in` factor must
  rise when Dev 2's sentiment-tilted scores already reflect a story, so the
  system never reacts twice to the same news. This reconciliation is the
  main intellectual problem in `src/recommendation/engine.py`.

### Shared kernel data shapes (used by every engine)

- `holdings` — `pd.DataFrame` from `src.portfolio.load_portfolio()`:
  columns `symbol, name, shares, buy_price`.
- `prices` — `Dict[str, float]` from `src.data_loader.get_latest_prices()`:
  latest close per symbol; a symbol may be **missing**, always handle that.
- `history` — `pd.DataFrame` from `src.data_loader.get_history()`: daily
  adjusted close, `DatetimeIndex`, one column per symbol; columns may be
  **missing** for some symbols.
- `weights` — `Dict[str, float]` derived from `portfolio.build_view()`:
  current weight in percent per symbol (cash excluded).

`src/data_loader.py` also owns the ticker universe: the full NASDAQ Trader
symbol directory (NASDAQ + NYSE/NYSE American/Arca/Cboe/IEX), cached to
`data/tickers.csv` for 7 days, with a small hardcoded `FALLBACK_UNIVERSE` if
the network and cache both fail. `to_yahoo_symbol()` translates directory
symbols (`BRK.B`) to yfinance symbols (`BRK-B`).

### Data directory

`data/` holds gitignored runtime state (`portfolio.csv`, `settings.json`,
`tickers.csv`, `news_raw.json` cache) alongside committed fixtures
(`sample_portfolio.csv`, `tickers.csv` seed). `data/processed/` is where
Dev 3's historical sentiment feature table is cached (built once, read
many times — don't rebuild it from raw corpora on every run).
