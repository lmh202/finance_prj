# Developer 4 — Reaction Risk, Recommendation & Dashboard Integration

**Mission (Architecture.md §7–§9):** combine the other three engines into one
answer — *"what should the user do, and how risky is reacting?"* — plus own the
Streamlit app's overall integration and polish.

## Your contract (frozen — see `src/interfaces.py`)

```python
reaction_risk(event, weights, regime) -> ReactionRisk
recommend_daily(regime, signals, weights) -> Recommendation
recommend_event(event, risk) -> Recommendation
apply_constraints(trades, weights) -> List[ProposedTrade]
```

You CONSUME the other three engines (through their frozen contracts only):
`portfolio_health.compute_health`, `daily_strategy.classify_regime` /
`score_assets`, `news_intelligence.essential_news`.

## Files you own

- `engine.py` — §7 risk formula, §8 recommendation combiner, §9 constraints.
  Baseline exists; the TODOs (priced-in detection, corroboration) are yours.
- `page.py` — the "Should I React?" page (includes a clearly-labeled DEMO
  event until Developer 3 ships real news — remove it then).
- **Exception to the folder rule:** you also maintain `app/` (Home page,
  navigation, visual consistency) and the shared kernel (`src/data_loader.py`,
  `src/portfolio.py`) — but treat kernel changes as breaking changes:
  announce before touching.

## Definition of done (MVP)

- [ ] Reaction risk uses the §7 weights with real inputs: priced-in check
      (recent move of affected assets vs. event time), corroboration from the
      news engine, live volatility from the regime indicators
- [ ] Daily + event recommendations shown side by side (§14: "keep them separate")
- [ ] §9 constraints enforced on every displayed trade (incl. min holdings, sector caps)
- [ ] Home page shows: portfolio value, health score, regime, risk level, #essential news, main recommendation (§10 Page 1)
- [ ] "Should I React?" offers the three §10 choices (do nothing / moderate / aggressive) with risk beside each

## Rules

1. Outside your folder you may edit ONLY `app/` and the shared kernel (with
   announcement) — never another developer's engine folder.
2. `src/interfaces.py` changes require agreement from all four developers;
   you're its natural custodian — collect and apply agreed amendments.
3. New pip dependency? Announce it, then add to `requirements.txt`.
