"""Rule Fusion Engine — strategy + news + health + volatility -> one decision.

Composes the other engines server-side and runs the four-step rule pipeline
(src/rule_fusion/), so the frontend gets a decision plus the full audit trail
rather than four separate signals it would have to reconcile itself.

Markers: 409 empty_portfolio · 502 no_history. There is deliberately no
`no_model` marker — when the risk engine's artifact is missing the fusion
engine falls back to a price-derived volatility percentile and says so in
`volatility_source`, rather than failing the whole decision.
"""

import math
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from routers._common import NO_HISTORY, load_holdings
from serialize import as_dict
from src import data_loader
from src import portfolio as pf
from src.daily_strategy import engine as strategy
from src.interfaces import BENCHMARK
from src.news_intelligence import engine as news
from src.portfolio_health import engine as health
from src.risk_engine import engine as risk
from src.rule_fusion import engine as fusion

router = APIRouter(prefix="/fusion", tags=["fusion"])

# Bounds the RSS collector: it opens three feeds per ticker, so an unbounded
# scan set would turn one request into hundreds of network round-trips.
MAX_SCAN_SYMBOLS = 30


class SimulateIn(BaseModel):
    """The four inputs by hand — no market data, no network.

    Defaults mirror rule_fusion.engine.FusionInputs so a caller can send only
    the fields they care about.
    """

    symbol: str = "SIM"
    held: bool = False
    weight_pct: float = 0.0

    strategy_signal: str = Field("HOLD", description="BUY | SELL | HOLD")
    strategy_score: float = Field(0.0, description="-3.5 .. +3.5")
    strategy_reasons: List[str] = Field(default_factory=list)

    news_label: str = Field("none", description="positive | neutral | negative | none")
    news_sentiment: float = Field(0.0, description="-1 .. +1")
    news_importance: float = Field(0.0, description="0 .. 100")
    news_headline: str = ""
    critical_category: Optional[str] = Field(
        None, description="interest_rate | war_geopolitical | earnings_corporate | systemic_macro"
    )
    critical_keyword: Optional[str] = None

    health_score: Optional[float] = Field(None, description="0 .. 100, null if unknown")

    volatility_pct: Optional[float] = Field(None, description="0 .. 100, null if unknown")
    volatility_source: str = "unavailable"


def _risk_levels(symbols: List[str]) -> Dict[str, float]:
    """symbol -> RiskEstimate.risk_level at the 5-day horizon.

    Empty when the trained artifact is absent — the fusion adapter then falls
    back to a price-derived percentile, so a missing model degrades the answer
    instead of blocking it.
    """
    if not risk.model_available():
        return {}
    ohlc = data_loader.get_ohlc_history(symbols)
    if not ohlc:
        return {}
    return {
        est.symbol: est.risk_level
        for est in risk.risk_estimates(ohlc, horizons=(5,))
        if est.has_history
    }


@router.get("/decisions")
def decisions(universe: str = "", max_events: int = 8, held_only: bool = False) -> dict:
    """One fused decision per scanned symbol, most actionable first.

    Scans holdings PLUS a candidate universe (the `universe` query param, else
    daily_strategy.STRATEGY_WATCHLIST) so NEW_BUY can surface on something you
    do not own yet; `held_only=true` restricts the scan to your holdings.
    409 if the portfolio is empty; 502 if price history is unavailable.
    """
    holdings = load_holdings()
    held = set(holdings["symbol"])
    extra = [s.strip().upper() for s in universe.split(",") if s.strip()]
    scan = sorted(held if held_only else held | set(extra or strategy.STRATEGY_WATCHLIST))
    scan = scan[:MAX_SCAN_SYMBOLS]

    history = data_loader.get_history(scan + [BENCHMARK])
    if history.empty:
        raise HTTPException(status_code=502, detail=NO_HISTORY)

    prices = data_loader.get_latest_prices(scan)
    view, _ = pf.build_view(holdings, prices, pf.load_cash())
    weights = dict(zip(view["symbol"], view["weight_pct"])) if not view.empty else {}

    recs = strategy.recommend_signals(history, holdings, prices, universe=scan)
    events = news.essential_news(scan, max_events=max_events)
    report = health.compute_health(holdings, history)
    levels = _risk_levels(scan)

    fused = fusion.rank(
        fusion.fuse(
            recs,
            events=events,
            health=report,
            risk_levels=levels,
            history=history,
            weights=weights,
        )
    )
    sources = sorted({d.inputs.volatility_source for d in fused})
    return {
        "decisions": [as_dict(d) for d in fused],
        "context": {
            "scanned": scan,
            "health_score": report.score if report.metrics else None,
            "regime": as_dict(strategy.classify_regime(history)),
            "event_count": len(events),
            "volatility_sources": sources,
        },
    }


@router.post("/simulate")
def simulate(body: SimulateIn) -> dict:
    """Run the rule pipeline on hand-supplied inputs — no market data needed.

    The demo and debugging surface: it exercises the exact same decide() the
    live endpoint uses, so the rule table can be probed one case at a time.
    """
    return as_dict(fusion.decide(fusion.FusionInputs(**body.model_dump())))


@router.get("/rules")
def rules() -> dict:
    """The tunable rule table and the critical-event taxonomy, as served.

    Lets a UI (or a reviewer) show the thresholds a decision was made under
    instead of hardcoding a second copy of them.
    """
    return {
        "step1_strategy": {
            "neutral_base_confidence": fusion.NEUTRAL_BASE_CONFIDENCE,
            "directional_base_confidence": fusion.DIRECTIONAL_BASE_CONFIDENCE,
            "directional_conviction_bonus": fusion.DIRECTIONAL_CONVICTION_BONUS,
            "buy_threshold": fusion.BUY_THRESHOLD,
            "max_strategy_score": fusion.MAX_STRATEGY_SCORE,
        },
        "step2_news": {
            "agree_gain": fusion.NEWS_AGREE_GAIN,
            "disagree_loss": fusion.NEWS_DISAGREE_LOSS,
            "neutral_drag": fusion.NEWS_NEUTRAL_DRAG,
            "critical_min_importance": fusion.CRITICAL_MIN_IMPORTANCE,
            "critical_min_abs_sentiment": fusion.CRITICAL_MIN_ABS_SENTIMENT,
            "critical_override_cap": fusion.CRITICAL_OVERRIDE_CAP,
            "taxonomy": fusion.adapters.TAXONOMY,
            "category_priority": list(fusion.adapters.CRITICAL_PRIORITY),
        },
        "step3_health": {
            "strong_at": fusion.HEALTH_STRONG,
            "fragile_below": fusion.HEALTH_FRAGILE,
            "adjustments": fusion.HEALTH_ADJUSTMENTS,
        },
        "step4_volatility": {
            # The top rung's ceiling is +inf, which starlette's json.dumps
            # (allow_nan=False) refuses — report it as null.
            "size_ladder": [
                {"below_percentile": (c if math.isfinite(c) else None), "multiplier": m}
                for c, m in fusion.VOLATILITY_SIZE_LADDER
            ],
            "base_trade_weight_pct": fusion.BASE_TRADE_WEIGHT_PCT,
            "base_trim_fraction": fusion.BASE_TRIM_FRACTION,
        },
        "actions": {
            "vocabulary": {a: fusion.ACTION_LABELS[a] for a in fusion.ACTIONS},
            "confidence_floor": fusion.ACT_CONFIDENCE_FLOOR,
            "close_confidence": fusion.CLOSE_CONFIDENCE,
        },
    }
