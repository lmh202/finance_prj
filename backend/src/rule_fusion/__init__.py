"""Financial Rule Fusion Engine — four signals, four ordered steps, one decision.

Combines the daily strategy signal, news sentiment, the portfolio health score
and the volatility percentile into a single explainable per-holding decision.
The four inputs are NEVER averaged into one score: each acts on exactly one
output dimension, in a fixed order, and every step is recorded in a confidence
ledger that ships with the answer.

Public surface:
  engine.decide(FusionInputs) -> FusionDecision      pure, no I/O
  engine.fuse(...)            -> List[FusionDecision]  batch over the other engines' output
  engine.FusionInputs / FusionDecision / Adjustment / RiskView / SizeView
  adapters.news_view / critical_scan / health_input / volatility_view

This is NOT one of the four contract engines: src/interfaces.py is frozen and
untouched, and the types above are declared locally (same pattern as
src/risk_engine/). Read-only w.r.t. the shared kernel.
"""
