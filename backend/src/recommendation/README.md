# Developer 4 — Decision Layer: Reaction Risk, Recommendation & Product

**Mission (Architecture.md §7–§9 + fusion):** combine the other three engines
into one answer — *"what should the user do, and how risky is reacting?"* —
**without double-counting news**, and own the Streamlit product end to end.

## Your contract (frozen — see `src/interfaces.py`)

```python
reaction_risk(event, weights, regime) -> ReactionRisk
recommend_daily(regime, signals, weights) -> Recommendation
recommend_event(event, risk) -> Recommendation
apply_constraints(trades, weights) -> List[ProposedTrade]
```

You CONSUME (through frozen contracts only): `portfolio_health.compute_health`
/ `what_if_health`, `daily_strategy.classify_regime` / `score_assets`,
`news_intelligence.essential_news`.

## Your key intellectual problem: double-counting news

Once Developer 2's ML scores include sentiment features, a headline influences
the recommendation through TWO paths — the model's tilted score AND the
event-driven recommendation. Reacting to both = reacting twice. Your §7
`priced_in` factor is where this gets solved: if the affected assets' current
signals already reflect today's sentiment (or prices already moved), raise the
risk of reacting further. This is the smartest slide of the demo — "AURORA
knows when the news is already priced in and tells you NOT to trade."

Everything else stays formula-based on purpose (explainability is the §7/§8
story) — resist the temptation to put an ML model or LLM in this layer.

## Files you own

- `engine.py` — §7 risk formula (implement `priced_in` + corroboration from
  real inputs), §8 combiner, §9 constraints (add sector caps, min holdings).
- `page.py` — "Should I React?" (remove the DEMO event once Dev 3 ships).
- **Exception to the folder rule:** you also maintain `app/` (Home page,
  navigation, visual consistency) and the shared kernel (`src/data_loader.py`,
  `src/portfolio.py`) — treat kernel changes as breaking: announce first.
  You are custodian of `src/interfaces.py` amendments (collect team sign-off).

## Definition of done

- [ ] `priced_in` implemented from real inputs (event sentiment/timestamp vs. current signals and recent price moves); corroboration wired from the news engine
- [ ] Daily + event recommendations shown side by side (§14: keep them separate), each with the what-if health delta (already wired for daily)
- [ ] §9 constraints enforced on every displayed trade (incl. min holdings, sector caps)
- [ ] Home page per §10 Page 1: value, health score, regime, risk level, #essential news, main recommendation
- [ ] "Should I React?" offers the three §10 choices (do nothing / moderate / aggressive) with risk beside each
- [ ] Demo-day narrative: you own the end-to-end story and present the ablation result (Dev 2's numbers) in product terms

## Rules

1. Outside your folder you may edit ONLY `app/` and the shared kernel (with
   announcement) — never another developer's engine folder.
2. `src/interfaces.py` changes require agreement from all four developers.
3. New pip dependency? Announce it, then add to `requirements.txt`.
