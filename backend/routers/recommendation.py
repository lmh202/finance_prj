"""Engine 4 — Reaction Risk & Recommendation (Developer 4).

Composes the other engines server-side so the frontend gets one small JSON
answer instead of shipping price history over the wire.
"""

from datetime import datetime
from typing import Dict

from fastapi import APIRouter
from pydantic import BaseModel

from routers._common import load_holdings, load_holdings_history
from serialize import as_dict
from src import data_loader
from src import portfolio as pf
from src.daily_strategy import engine as strategy
from src.daily_strategy import learned as learned_strategy
from src.interfaces import NewsEvent
from src.news_intelligence import engine as news
from src.portfolio_health import engine as health
from src.recommendation import engine
from src.recommendation import decision
from src.recommendation import gated_news
from src.recommendation import llm_client
from src.risk_engine import engine as risk_engine

router = APIRouter(prefix="/recommendation", tags=["recommendation"])

# Exercises the reaction-risk flow until Developer 3 ships real events.
# Clearly labeled as a demo — remove once essential_news returns data.
DEMO_EVENT = NewsEvent(
    title="[DEMO] Unexpected interest-rate increase announced",
    source="demo — not a real headline",
    url="",
    published=None,
    category="interest_rates",
    sentiment=-0.6,
    importance=85.0,
    affected_symbols=["AAPL", "MSFT"],
    impact={"AAPL": "negative", "MSFT": "negative", "GLD": "mixed"},
    summary="Placeholder event so the reaction-risk flow can be demoed "
    "before the news engine is implemented.",
)


class EventIn(BaseModel):
    event: Dict


def _weights_and_regime(holdings, history):
    prices = data_loader.get_latest_prices(list(holdings["symbol"]))
    view, _ = pf.build_view(holdings, prices, pf.load_cash())
    weights = dict(zip(view["symbol"], view["weight_pct"]))
    return weights, strategy.classify_regime(history)


def _event_from_dict(raw: Dict) -> NewsEvent:
    published = raw.get("published")
    if isinstance(published, str):
        published = datetime.fromisoformat(published)
    return NewsEvent(
        title=raw.get("title", ""),
        source=raw.get("source", ""),
        url=raw.get("url", ""),
        published=published,
        category=raw.get("category", ""),
        sentiment=float(raw.get("sentiment", 0.0)),
        importance=float(raw.get("importance", 0.0)),
        affected_symbols=list(raw.get("affected_symbols", [])),
        impact=dict(raw.get("impact", {})),
        summary=raw.get("summary", ""),
    )


@router.get("/daily")
def daily() -> dict:
    holdings, history = load_holdings_history()
    weights, regime = _weights_and_regime(holdings, history)
    fusion_results = []
    decision_meta = {
        "production_mode": "legacy_signal_fallback",
        "fallback_reason": None,
    }
    explanation_meta = {
        "source": "legacy_template",
        "reason": "decision_model_unavailable",
    }
    try:
        if not risk_engine.model_available():
            raise RuntimeError("risk_model_unavailable")
        symbols = sorted(set(holdings["symbol"]))
        ohlc = data_loader.get_ohlc_history(symbols)
        estimates = risk_engine.risk_estimates(ohlc)
        signals = strategy.score_assets(history, holdings)
        if learned_strategy.model_available(require_promoted=True):
            signals = learned_strategy.enhance_signals(
                history,
                signals,
                require_promoted=True,
            )
        health_report = health.compute_health(holdings, history)
        health_score = health_report.score if health_report.metrics else None
        max_events = min(50, max(15, len(symbols) * 3))
        events = news.essential_news(symbols, max_events=max_events)
        if gated_news.model_available(require_promoted=True):
            gated = gated_news.recommend_portfolio(
                history=history,
                signals=signals,
                news_events=events,
                risk_estimates=estimates,
                current_weights_pct=weights,
                health_score=health_score,
            )
            rec = gated.recommendation
            decision_meta = gated.metadata
            explanation_meta = {
                "source": "gated_news_plus_external_risk",
                "reason": None,
            }
        else:
            controlled = gated_news.recommend_strategy_risk_control(
                history=history,
                signals=signals,
                risk_estimates=estimates,
                current_weights_pct=weights,
                health_score=health_score,
            )
            rec = controlled.recommendation
            decision_meta = controlled.metadata
            explanation_meta = {
                "source": "strategy_plus_external_harx_news_risk",
                "reason": "direct_news_residual_not_promoted",
            }
    except (RuntimeError, ValueError, KeyError) as exc:
        # Preserve the already-validated numeric decision path as the first
        # fallback. The legacy signal recommendation remains the final
        # always-available path.
        try:
            if not decision.model_available() or not risk_engine.model_available():
                raise RuntimeError("decision_or_risk_model_unavailable")
            symbols = sorted(set(holdings["symbol"]))
            ohlc = data_loader.get_ohlc_history(symbols)
            estimates = risk_engine.risk_estimates(ohlc)
            numeric = decision.recommend_portfolio(history, estimates, weights)
            rec = numeric.recommendation
            decision_meta = numeric.metadata
            decision_meta["fallback_reason"] = type(exc).__name__
            explanation = llm_client.explain_decision(
                {
                    "production_mode": decision_meta["production_mode"],
                    "model_version": decision_meta["model_version"],
                    "trades": [as_dict(trade) for trade in rec.trades],
                    "confidence": rec.confidence,
                    "symbols": decision_meta["symbols"],
                    "confidence_note": (
                        "Confidence reflects numeric model availability and "
                        "constraint feasibility, not certainty about price direction."
                    ),
                }
            )
            rec.explanation = " ".join(
                [explanation["summary"], *explanation["reasons"]]
            )
            explanation_meta = explanation["_meta"]
        except (RuntimeError, ValueError, KeyError):
            signals = strategy.score_assets(history, holdings)
            rec = engine.recommend_daily(regime, signals, weights)
            decision_meta["fallback_reason"] = type(exc).__name__

    result = {
        "recommendation": as_dict(rec),
        "fusion_results": fusion_results,
        "decision_meta": decision_meta,
        "explanation_meta": explanation_meta,
        "health_before": None,
        "health_after": None,
    }
    if rec.trades:
        result["health_before"] = health.compute_health(holdings, history).score
        result["health_after"] = health.what_if_health(holdings, history, rec.trades).score
    return result


@router.get("/events")
def events(max_events: int = 5) -> dict:
    holdings = load_holdings()
    found = news.essential_news(list(holdings["symbol"]), max_events=max_events)
    demo = not found
    if demo:
        found = [DEMO_EVENT]
    return {"events": [as_dict(e) for e in found], "demo": demo}


@router.post("/react")
def react(body: EventIn) -> dict:
    event = _event_from_dict(body.event)
    holdings, history = load_holdings_history()
    weights, regime = _weights_and_regime(holdings, history)
    risk = engine.reaction_risk(event, weights, regime)
    rec = engine.recommend_event(event, risk)
    return {"risk": as_dict(risk), "recommendation": as_dict(rec)}
