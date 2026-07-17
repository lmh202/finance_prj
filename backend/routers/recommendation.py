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
from src.interfaces import NewsEvent
from src.news_intelligence import engine as news
from src.portfolio_health import engine as health
from src.recommendation import engine

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
    signals = strategy.score_assets(history, holdings)
    rec = engine.recommend_daily(regime, signals, weights)

    result = {"recommendation": as_dict(rec), "health_before": None, "health_after": None}
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
