# Developer 2 — Daily Market Strategy Engine

**Mission (Architecture.md §5):** answer *"Given normal market conditions,
should the allocation change?"* — regime-aware momentum: classify the market
regime, score every held asset daily, and prove it with a backtest.

## Your contract (frozen — see `src/interfaces.py`)

```python
classify_regime(history, benchmark=BENCHMARK) -> RegimeState
score_assets(history, holdings) -> List[AssetSignal]
backtest(history, holdings, cash=0.0) -> pd.DataFrame   # buy_hold, equal_weight, + your strategy
```

Consumed by: your two pages, and Developer 4's `reaction_risk` + `recommend_daily`.

## What you get for free (shared kernel — read-only)

- `src.portfolio.load_portfolio()` → holdings DataFrame
- `src.data_loader.get_history(symbols, period)` → daily adjusted-close DataFrame

## Files you own (edit ONLY inside this folder)

- `engine.py` — regime rules, §5 scoring formula, backtest. Baseline exists; **improve it.**
- `page.py` — TWO pages: `render()` (Daily Strategy) and `render_performance()` (Performance & Benchmark).

## Definition of done (MVP)

- [ ] Four regimes per §5 rules (SMA50/SMA200/momentum/volatility), sanity-checked on 2020 (bearish→high-vol) and 2024 (bullish)
- [ ] Asset score = 30% momentum + 25% trend + 20% Sharpe − 15% vol − 10% drawdown, with increase/hold/reduce signals
- [ ] Backtest adds the regime-aware strategy itself (rebalance on regime change) + turnover and a transaction-cost assumption
- [ ] Performance page: cumulative-return chart + metric table (return, Sharpe, Sortino, max DD, vol, turnover)
- [ ] Handles gracefully: <200 days of history, missing benchmark column

## Rules

1. Commit only inside `src/daily_strategy/`. Never edit `src/interfaces.py`,
   the shared kernel, other engines, or `app/` — propose changes in the group instead.
2. Collaborate only through `src/interfaces.py` types and other engines' public functions.
3. New pip dependency? Announce it, then add to `requirements.txt` (the one allowed outside-folder edit).
