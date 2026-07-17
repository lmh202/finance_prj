# Developer 2 — Daily Market Strategy Engine (the predictive ML)

**Mission (Architecture.md §5 + ML upgrade):** answer *"Given market
conditions, should the allocation change?"* — regime classification and
daily asset scoring, upgraded from the rule-based baseline to a trained
ML model, with news sentiment as an **optional feature channel** and a
rigorous ablation proving whether it helps.

> **Split note:** your files are this engine, `backend/routers/strategy.py`,
> and `frontend/views/daily_strategy.py`. No streamlit in backend/, no `src`
> imports in frontend/.

## Your contract (frozen — see `src/interfaces.py`)

```python
classify_regime(history, benchmark=BENCHMARK) -> RegimeState
score_assets(history, holdings, sentiment=None) -> List[AssetSignal]
backtest(history, holdings, cash=0.0) -> pd.DataFrame
```

Consumed by: your Daily Strategy page, Developer 4 (`reaction_risk`,
`recommend_daily`), and Developer 1's Performance page (backtest curves).

## What you get for free

- Shared kernel: `portfolio.load_portfolio()`, `data_loader.get_history()`
- Developer 3's `news_intelligence.engine.sentiment_features(symbols, start, end)`
  → long-format frame (`date, symbol, sentiment, news_count, has_news`), no look-ahead

## Files you own (edit ONLY inside this folder)

- `engine.py` — rule-based baseline exists (regime rules + §5 percentile scores + naive backtest). **Your job is the ML upgrade.**
- `page.py` — the Daily Strategy page (Performance page moved to Developer 1; you supply its data via `backtest()`).

## The ML plan

1. **Features**: the indicators already computed (momentum, price-vs-SMA50/200,
   rolling vol, RSI, drawdown, beta) per asset per day over the expanded universe.
2. **Optional news channel**: left-join Dev 3's `sentiment_features`; days
   without news get `sentiment=0, news_count=0, has_news=0` — that's what makes
   the input optional.
3. **Model**: scikit-learn (logistic regression / random forest — small-data,
   explainable), predicting next-N-day direction or regime; probabilities feed
   `score_assets` scores and `classify_regime` confidence.
4. **Validation**: WALK-FORWARD splits only — never a random shuffle. This is
   the methodological point graders probe.
5. **The ablation** (your presentation result): price-only vs price+news,
   compared on walk-forward AUC and backtested strategy Sharpe. A null result
   is still a result — report it honestly.

## Definition of done

- [ ] Four regimes per §5, sanity-checked on 2020 (bear/high-vol) and 2024 (bull)
- [ ] Rule-based scores kept as the fallback when the model can't run
- [ ] ML model trained walk-forward on the expanded universe; probabilities → scores
- [ ] Sentiment channel merged per the schema; ablation run and reported
- [ ] `backtest()` returns buy_hold, equal_weight, rule-based, ML price-only, ML price+news columns + turnover/cost assumption
- [ ] Handles gracefully: <200 days history, missing benchmark, empty/None sentiment

## Rules

1. Commit only inside `src/daily_strategy/`. Never edit `src/interfaces.py`,
   the shared kernel, other engines, or `app/` — propose changes in the group instead.
2. Collaborate only through `src/interfaces.py` types and other engines' public functions.
3. New pip dependency (scikit-learn is already listed)? Announce it, then add to `requirements.txt`.
