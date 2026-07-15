# Developer 1 — Portfolio Intelligence Engine

**Mission (Architecture.md §4):** answer *"How healthy is the user's current
portfolio?"* — risk/return metrics plus a 0–100 Health Score that communicates
strengths and weaknesses in plain language.

## Your contract (frozen — see `src/interfaces.py`)

```python
compute_health(holdings, history, benchmark=BENCHMARK) -> HealthReport
```

Consumed by: your Health page, and Developer 4's `recommend_daily`.

## What you get for free (shared kernel — read-only)

- `src.portfolio.load_portfolio()` → holdings DataFrame (`symbol, name, shares, buy_price`)
- `src.data_loader.get_history(symbols, period)` → daily adjusted-close DataFrame
- `src.data_loader.get_latest_prices(symbols)` → dict of latest prices

## Files you own (edit ONLY inside this folder)

- `engine.py` — the metrics + scoring logic. A working baseline exists; **improve it.**
- `page.py` — the "Portfolio Health" dashboard page (`render()`).

## Definition of done (MVP)

- [ ] Annualized return, volatility, Sharpe, Sortino, max drawdown, beta — verified against a hand calculation
- [ ] Diversification + single-asset concentration; sector concentration if time allows
- [ ] Health Score using the §4 weightings (25/20/20/15/10/10) with sensible curves
- [ ] Strengths/weaknesses text that reads like the §4 example
- [ ] Correlation heatmap on the page (upgrade the current table)
- [ ] Handles gracefully: empty portfolio, missing price columns, <30 days history

## Rules

1. Commit only inside `src/portfolio_health/`. Never edit `src/interfaces.py`,
   the shared kernel, other engines, or `app/` — propose changes in the group instead.
2. Collaborate only through `src/interfaces.py` types and other engines' public
   functions.
3. New pip dependency? Announce it, then add to `requirements.txt` (the one
   allowed outside-folder edit).
