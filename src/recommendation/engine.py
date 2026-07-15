"""Recommendation & Reaction-Risk Engine — Developer 4 (Architecture.md §7-§9).

WORKING BASELINE combining the other engines' outputs with the §7 risk
formula and §9 constraints. Refine factor estimation (priced-in detection,
source corroboration) as the news engine matures. Signatures are frozen
in src/interfaces.py.
"""

from typing import Dict, List

from src.interfaces import (
    CONSTRAINTS,
    AssetSignal,
    NewsEvent,
    ProposedTrade,
    ReactionRisk,
    Recommendation,
    RegimeState,
)

# §7 factor weights
RISK_WEIGHTS = {
    "news_uncertainty": 0.25,
    "technical_disagreement": 0.20,
    "market_volatility": 0.20,
    "priced_in": 0.15,
    "concentration": 0.10,
    "ambiguity": 0.10,
}


def reaction_risk(
    event: NewsEvent, weights: Dict[str, float], regime: RegimeState
) -> ReactionRisk:
    """How risky is it to trade on this event? 0-100 per the §7 formula."""
    exposure = sum(weights.get(s, 0.0) for s in event.affected_symbols) / 100.0

    factors = {
        # TODO: corroboration count from the news engine once available
        "news_uncertainty": 0.5,
        # News pushing against the prevailing regime is riskier to act on
        "technical_disagreement": 0.7 if (
            (event.sentiment < 0 and regime.regime == "bullish")
            or (event.sentiment > 0 and regime.regime == "bearish")
        ) else 0.3,
        "market_volatility": 0.8 if regime.regime == "high_volatility" else 0.4,
        # TODO: compare event timestamp vs. recent price move of affected assets
        "priced_in": 0.5,
        "concentration": min(1.0, exposure * 2),
        "ambiguity": 0.5 if abs(event.sentiment) < 0.3 else 0.2,
    }

    risk = 100 * sum(RISK_WEIGHTS[k] * v for k, v in factors.items())

    reasons = []
    if factors["technical_disagreement"] >= 0.7:
        reasons.append("The event contradicts the current market regime.")
    if factors["market_volatility"] >= 0.8:
        reasons.append("Markets are unusually volatile right now.")
    if exposure > 0.25:
        reasons.append(
            f"Affected holdings are {exposure:.0%} of the portfolio — reacting adds concentration risk."
        )
    if not reasons:
        reasons.append("No single factor dominates; uncertainty is broad-based.")

    suggestion = "do_nothing" if risk >= 60 else ("moderate" if risk >= 35 else "aggressive")
    return ReactionRisk(
        risk_pct=round(risk, 1), factors=factors, reasons=reasons, suggestion=suggestion
    )


def apply_constraints(
    trades: List[ProposedTrade], weights: Dict[str, float]
) -> List[ProposedTrade]:
    """Enforce §9 safety rules on proposed trades."""
    out = []
    for t in trades:
        change = t.weight_change_pct
        cap = CONSTRAINTS["max_weight_change_pct"]
        change = max(-cap, min(cap, change))
        # Don't push a single position above the max stock weight
        new_weight = weights.get(t.symbol, 0.0) + change
        if change > 0 and new_weight > CONSTRAINTS["max_stock_weight_pct"]:
            change = max(0.0, CONSTRAINTS["max_stock_weight_pct"] - weights.get(t.symbol, 0.0))
        if abs(change) < CONSTRAINTS["min_trade_pct"]:
            continue
        out.append(ProposedTrade(t.symbol, round(change, 2), t.reason))
    return out


def recommend_daily(
    regime: RegimeState, signals: List[AssetSignal], weights: Dict[str, float]
) -> Recommendation:
    """Normal-day suggestion: trim the weakest signal, add to the strongest."""
    trades: List[ProposedTrade] = []
    increases = [s for s in signals if s.action == "increase"]
    reduces = [s for s in signals if s.action == "reduce"]
    if increases:
        best = increases[0]
        trades.append(ProposedTrade(best.symbol, +2.0, best.rationale))
    if reduces:
        worst = reduces[-1]
        trades.append(ProposedTrade(worst.symbol, -2.0, worst.rationale))
    trades = apply_constraints(trades, weights)

    if trades:
        explanation = (
            f"Market regime is {regime.regime.replace('_', ' ')} "
            f"(confidence {regime.confidence:.0%}). Small tilt toward the "
            "strongest-scoring holding, away from the weakest."
        )
    else:
        explanation = (
            f"Market regime is {regime.regime.replace('_', ' ')}. No adjustment "
            "clears the constraints today — holding steady is the recommendation."
        )
    return Recommendation(
        kind="daily",
        trades=trades,
        confidence=regime.confidence,
        explanation=explanation,
    )


def recommend_event(event: NewsEvent, risk: ReactionRisk) -> Recommendation:
    """Event-driven suggestion, gated by reaction risk. The user decides."""
    trades: List[ProposedTrade] = []
    if risk.suggestion != "do_nothing":
        size = 1.0 if risk.suggestion == "moderate" else 2.0
        for sym, direction in event.impact.items():
            delta = -size if direction == "negative" else (size if direction == "positive" else 0.0)
            if delta:
                trades.append(
                    ProposedTrade(sym, delta, f"{event.category}: expected {direction} impact")
                )

    explanation = (
        f"Event: {event.title} ({event.category}). Risk of reacting: "
        f"{risk.risk_pct:.0f}% → suggested stance: {risk.suggestion.replace('_', ' ')}."
    )
    return Recommendation(
        kind="event",
        trades=trades,
        confidence=1 - risk.risk_pct / 100,
        explanation=explanation,
        reaction_risk=risk,
    )
