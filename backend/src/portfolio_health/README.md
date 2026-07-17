# Developer 1 — Portfolio Intelligence Engine

**Mission (Architecture.md §4 + extensions):** answer *"How healthy is this
portfolio — and does our Health Score actually mean anything?"* Metrics,
the 0–100 Health Score, an empirical validation of that score, what-if
analysis for proposed trades, and the Performance & Benchmark page.

## Your contract (frozen — see `src/interfaces.py`)

```python
compute_health(holdings, history, benchmark=BENCHMARK) -> HealthReport
what_if_health(holdings, history, trades, benchmark=BENCHMARK) -> HealthReport
```

Consumed by: your two pages, and Developer 4 (`recommend_daily` context +
the "Health 68 → 73 if you accept this" metric on the recommendation pages).

## What you get for free (shared kernel — read-only)

- `src.portfolio.load_portfolio()` → holdings DataFrame
- `src.data_loader.get_history(symbols, period)` → daily adjusted-close DataFrame
- `src.daily_strategy.engine.backtest(...)` → curves for your Performance page (Dev 2's contract)

## Files you own (edit ONLY inside this folder)

- `engine.py` — metrics, health score, `what_if_health`. Working baselines exist; **improve them.**
- `page.py` — TWO pages: `render()` (Portfolio Health) and `render_performance()`
  (Performance & Benchmark — you present Dev 2's backtest/ablation results).

## Definition of done

- [ ] Annualized return, volatility, Sharpe, Sortino, max drawdown, beta — verified against a hand calculation
- [ ] Diversification, single-asset and sector concentration (sector via yfinance info), VaR/CVaR if time allows
- [ ] **Health Score validation study**: compute the score on rolling historical windows across many random portfolios from the expanded universe; show low-health portfolios suffered worse forward drawdowns/Sharpe; calibrate the §4 weights from that evidence — this is your presentation result
- [ ] `what_if_health` handles cash and buying assets not currently held
- [ ] Performance page: cumulative-return chart, rolling metrics, metric table (return, Sharpe, Sortino, max DD, vol, turnover) for every backtest column Dev 2 produces
- [ ] Correlation heatmap; graceful handling of empty portfolio / missing columns / short history
- [ ] Relief valve: if you finish early, you own cross-team number verification (every metric checkable against a spreadsheet)

## Rules

1. Commit only inside `src/portfolio_health/`. Never edit `src/interfaces.py`,
   the shared kernel, other engines, or `app/` — propose changes in the group instead.
2. Collaborate only through `src/interfaces.py` types and other engines' public functions.
3. New pip dependency? Announce it, then add to `requirements.txt` (the one allowed outside-folder edit).
