"""Explainable rule-based fusion of AURORA's strategy, news, health, and risk.

The directional vote is:

    strategy + news + health

The HAR-X risk percentile never votes on direction.  It only pulls the
directional score toward neutral, reduces confidence, and limits position size.
The module is intentionally deterministic so every displayed result can be
reconstructed from its component values.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Iterable, Mapping, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd

from src.interfaces import AssetSignal, ProposedTrade, Recommendation
from src.recommendation.engine import apply_constraints

MODEL_VERSION = "aurora-rule-fusion-v1"

STRATEGY_LABEL_SCORES = {
    "strong buy": 1.0,
    "buy": 0.60,
    "weak buy": 0.25,
    "increase": 0.60,
    "add": 0.60,
    "neutral": 0.0,
    "hold": 0.0,
    "weak sell": -0.25,
    "sell": -0.60,
    "reduce": -0.60,
    "trim": -0.25,
    "strong sell": -1.0,
}

_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}

_CATEGORY_IMPORTANCE = {
    "interest_rates": 1.00,
    "geopolitical": 1.00,
    "technology_regulation": 1.00,
    "corporate_earnings": 0.80,
    "energy_commodities": 0.80,
    "financial_system": 0.80,
    "inflation": 0.80,
    "economic_growth": 0.50,
    "other": 0.20,
}

# Optional relevance refinements for the two commodity ETFs in AURORA's
# training universe. Explicit ticker-tagged corporate news still gets 1.0.
_CATEGORY_RELEVANCE = {
    "GLD": {
        "interest_rates": 0.90,
        "inflation": 0.85,
        "geopolitical": 0.90,
        "energy_commodities": 0.90,
        "economic_growth": 0.70,
    },
    "SLV": {
        "interest_rates": 0.80,
        "inflation": 0.80,
        "geopolitical": 0.70,
        "energy_commodities": 0.85,
        "economic_growth": 0.75,
    },
}


@dataclass(frozen=True)
class FusionConfig:
    strategy_weight: float = 0.50
    news_weight: float = 0.30
    health_weight: float = 0.20
    low_news_weight: float = 0.10
    minimum_news_articles: int = 3
    risk_strength: float = 0.45
    strong_conflict_threshold: float = 0.60
    extreme_risk_threshold: float = 90.0
    strategy_stale_business_days: int = 3
    risk_stale_business_days: int = 3
    health_stale_business_days: int = 5


DEFAULT_CONFIG = FusionConfig()


@dataclass
class NewsAggregate:
    score: float
    relevant_articles: int
    unique_articles: int
    confidence: str
    titles: list[str] = field(default_factory=list)
    denominator: float = 0.0


@dataclass
class AssetFusionResult:
    symbol: str
    aurora_score: float
    outlook: str
    risk_level: str
    action: str
    confidence: float
    confidence_label: str
    strategy_score: float
    news_score: float
    health_score: float
    health_normalized: float
    risk_percentile: float
    risk_factor: float
    raw_score: float
    adjusted_score: float
    component_weights: dict[str, float]
    news_articles: int
    news_confidence: str
    news_titles: list[str]
    position_change_pct: float
    conflict: bool
    extreme_volatility: bool
    stale_inputs: list[str]
    unavailable_inputs: list[str]
    as_of: dict[str, Optional[str]]
    why: list[str]


@dataclass
class FusionDecision:
    recommendation: Recommendation
    assets: list[AssetFusionResult]
    metadata: dict


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, value)))


def normalize_strategy(value: object) -> float:
    """Convert a continuous 0..100 score or strategy label to -1..1."""
    if isinstance(value, str):
        return STRATEGY_LABEL_SCORES.get(value.strip().lower(), 0.0)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    if 0.0 <= number <= 100.0:
        return _clip((number - 50.0) / 50.0, -1.0, 1.0)
    return _clip(number, -1.0, 1.0)


def normalize_health(health_score: Optional[float]) -> float:
    """Convert a 0..100 health score to -1..1; missing health is neutral."""
    if health_score is None:
        return 0.0
    try:
        value = float(health_score)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return _clip((value - 50.0) / 50.0, -1.0, 1.0)


def risk_factor(risk_percentile: float, strength: float = 0.45) -> float:
    """Risk attenuation in [0.55, 1]; it cannot reverse direction."""
    percentile = _clip(float(risk_percentile), 0.0, 100.0)
    return 1.0 - float(strength) * percentile / 100.0


def _event_value(event: object, name: str, default: object = None) -> object:
    if isinstance(event, Mapping):
        return event.get(name, default)
    return getattr(event, name, default)


def _sentiment_strength(event: object) -> float:
    positive = _event_value(event, "positive_probability")
    negative = _event_value(event, "negative_probability")
    if positive is not None and negative is not None:
        try:
            return _clip(float(positive) - float(negative), -1.0, 1.0)
        except (TypeError, ValueError):
            pass
    try:
        return _clip(float(_event_value(event, "sentiment", 0.0)), -1.0, 1.0)
    except (TypeError, ValueError):
        return 0.0


def _importance(event: object) -> float:
    raw = _event_value(event, "importance")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float("nan")
    if math.isfinite(value) and value > 0:
        return _clip(value / 100.0 if value > 1.0 else value, 0.0, 1.0)
    category = str(_event_value(event, "category", "other")).lower()
    return _CATEGORY_IMPORTANCE.get(category, 0.20)


def _relevance(event: object, symbol: str) -> float:
    explicit = _event_value(event, "relevance")
    if isinstance(explicit, Mapping):
        explicit = explicit.get(symbol)
    if explicit is not None:
        try:
            return _clip(float(explicit), 0.0, 1.0)
        except (TypeError, ValueError):
            pass

    affected = {
        str(item).upper()
        for item in (_event_value(event, "affected_symbols", []) or [])
    }
    impact = {
        str(item).upper()
        for item in (_event_value(event, "impact", {}) or {})
    }
    symbol = symbol.upper()
    if symbol not in affected and symbol not in impact:
        return 0.0

    category = str(_event_value(event, "category", "other")).lower()
    refined = _CATEGORY_RELEVANCE.get(symbol, {}).get(category)
    return float(refined) if refined is not None else 1.0


def _to_utc(value: object) -> Optional[datetime]:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def recency_weight(published: object, now: Optional[datetime] = None) -> float:
    """Daily-decision recency schedule; stories older than five days expire."""
    published_utc = _to_utc(published)
    if published_utc is None:
        return 0.0
    now_utc = _to_utc(now or datetime.now(timezone.utc))
    if now_utc is None:
        return 0.0
    age_hours = max(0.0, (now_utc - published_utc).total_seconds() / 3600.0)
    if age_hours < 24:
        return 1.00
    if age_hours < 48:
        return 0.75
    if age_hours < 72:
        return 0.50
    if age_hours <= 120:
        return 0.25
    return 0.0


def _normalized_title(event: object) -> str:
    title = str(_event_value(event, "title", "")).lower()
    return " ".join(re.findall(r"[a-z0-9]+", title))


def _canonical_url(event: object) -> str:
    raw = str(_event_value(event, "url", "") or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
        query = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if not key.lower().startswith("utm_")
            and key.lower() not in _TRACKING_QUERY_KEYS
        ]
        return urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                parts.path.rstrip("/"),
                urlencode(query),
                "",
            )
        )
    except ValueError:
        return raw.lower()


def deduplicate_news(events: Iterable[object]) -> list[object]:
    """Collapse the same URL or >=88%-similar title into one event."""
    groups: list[list[object]] = []
    for event in events:
        title = _normalized_title(event)
        url = _canonical_url(event)
        for group in groups:
            representative = group[0]
            rep_url = _canonical_url(representative)
            rep_title = _normalized_title(representative)
            same_url = bool(url and rep_url and url == rep_url)
            same_title = bool(
                title
                and rep_title
                and SequenceMatcher(None, title, rep_title).ratio() >= 0.88
            )
            if same_url or same_title:
                group.append(event)
                break
        else:
            groups.append([event])

    unique = []
    for group in groups:
        # Keep the most informative representative. Multiple publications of
        # one story must not create multiple independent votes.
        representative = max(
            group,
            key=lambda event: (
                _importance(event),
                _to_utc(_event_value(event, "published"))
                or datetime.min.replace(tzinfo=timezone.utc),
            ),
        )
        unique.append(representative)
    return unique


def aggregate_news(
    events: Iterable[object],
    symbol: str,
    now: Optional[datetime] = None,
    minimum_articles: int = 3,
) -> NewsAggregate:
    """Importance/relevance/recency-weighted news sentiment for one asset."""
    relevant = [
        event
        for event in events
        if _relevance(event, symbol) > 0
        and recency_weight(_event_value(event, "published"), now) > 0
    ]
    unique = deduplicate_news(relevant)
    numerator = 0.0
    denominator = 0.0
    titles = []
    for event in unique:
        base_weight = (
            _importance(event)
            * _relevance(event, symbol)
            * recency_weight(_event_value(event, "published"), now)
        )
        numerator += _sentiment_strength(event) * base_weight
        denominator += base_weight
        title = str(_event_value(event, "title", "")).strip()
        if title:
            titles.append(title)
    score = numerator / denominator if denominator > 0 else 0.0
    count = len(unique)
    return NewsAggregate(
        score=round(_clip(score, -1.0, 1.0), 6),
        relevant_articles=len(relevant),
        unique_articles=count,
        confidence="High" if count >= minimum_articles else "Low",
        titles=titles,
        denominator=denominator,
    )


def _outlook(score: float) -> str:
    if score >= 75:
        return "Strong Positive"
    if score >= 65:
        return "Moderately Positive"
    if score >= 55:
        return "Slightly Positive"
    if score >= 45:
        return "Neutral"
    if score >= 35:
        return "Slightly Negative"
    if score >= 25:
        return "Moderately Negative"
    return "Strong Negative"


def _risk_level(percentile: float) -> str:
    if percentile <= 35:
        return "Low"
    if percentile <= 70:
        return "Moderate"
    if percentile <= 90:
        return "High"
    return "Extreme"


def _direction(value: float, subject: str) -> str:
    magnitude = abs(value)
    if magnitude < 0.10:
        word = "neutral"
    elif value > 0:
        word = "strongly positive" if magnitude >= 0.60 else "positive"
    else:
        word = "strongly negative" if magnitude >= 0.60 else "negative"
    return f"{subject} is {word} ({value:+.2f})."


def _confidence_label(value: float) -> str:
    if value >= 0.70:
        return "High"
    if value >= 0.45:
        return "Medium"
    return "Low"


def _action(outlook: str, risk: str, blocked: bool) -> str:
    if blocked:
        return "Hold"
    if outlook in {"Strong Positive", "Moderately Positive", "Slightly Positive"}:
        if risk in {"High", "Extreme"}:
            return "Cautious Positive"
        if outlook == "Strong Positive":
            return "Increase"
        if outlook == "Moderately Positive":
            return "Moderate Positive Exposure"
        return "Small Positive Tilt"
    if outlook == "Neutral":
        return "Hold"
    if outlook == "Slightly Negative":
        return "Small Reduction"
    if outlook == "Moderately Negative":
        return "Reduce"
    return "Strong Reduction"


def _position_change(
    score: float,
    blocked: bool,
    extreme_volatility: bool,
) -> float:
    if blocked or 45 <= score < 55:
        return 0.0
    if score >= 75:
        change = 4.0
    elif score >= 65:
        change = 3.0
    elif score >= 55:
        change = 1.5
    elif score < 25:
        change = -4.0
    elif score < 35:
        change = -3.0
    else:
        change = -1.5
    if extreme_volatility and change > 0:
        change = min(change, 1.0)
    return change


def fuse_scores(
    symbol: str,
    strategy_score: Optional[float],
    news_score: float,
    health_score: Optional[float],
    risk_percentile: Optional[float],
    *,
    news_article_count: int = 3,
    news_titles: Optional[Sequence[str]] = None,
    stale_inputs: Optional[Sequence[str]] = None,
    unavailable_inputs: Optional[Sequence[str]] = None,
    as_of: Optional[Mapping[str, Optional[str]]] = None,
    config: FusionConfig = DEFAULT_CONFIG,
) -> AssetFusionResult:
    """Fuse already-normalized component scores into one auditable result."""
    unavailable = list(unavailable_inputs or [])
    stale = list(stale_inputs or [])

    strategy_available = strategy_score is not None
    risk_available = risk_percentile is not None
    health_available = health_score is not None
    if not strategy_available and "strategy" not in unavailable:
        unavailable.append("strategy")
    if not risk_available and "risk" not in unavailable:
        unavailable.append("risk")
    if not health_available and "health" not in unavailable:
        unavailable.append("health")

    strategy = _clip(float(strategy_score or 0.0), -1.0, 1.0)
    news_value = _clip(float(news_score), -1.0, 1.0)
    health_value = normalize_health(health_score)
    displayed_health = 50.0 if health_score is None else _clip(float(health_score), 0, 100)
    displayed_risk = 100.0 if risk_percentile is None else _clip(
        float(risk_percentile), 0, 100
    )

    if news_article_count >= config.minimum_news_articles:
        news_weight = config.news_weight
    elif news_article_count > 0:
        news_weight = config.low_news_weight
    else:
        news_weight = 0.0
    strategy_weight = config.strategy_weight + (
        config.news_weight - news_weight
    )
    health_weight = config.health_weight

    raw = (
        strategy_weight * strategy
        + news_weight * news_value
        + health_weight * health_value
    )
    factor = risk_factor(displayed_risk, config.risk_strength)
    adjusted = raw * factor
    formula_score = _clip(50.0 + 50.0 * adjusted, 0.0, 100.0)

    conflict = bool(
        news_article_count > 0
        and abs(strategy) >= config.strong_conflict_threshold
        and abs(news_value) >= config.strong_conflict_threshold
        and strategy * news_value < 0
    )
    extreme = displayed_risk > config.extreme_risk_threshold
    blocked = conflict or not strategy_available or not risk_available

    # Strong directional disagreement is not silently averaged into a trade.
    # Missing strategy/risk is also fail-closed.
    score = 50.0 if blocked else formula_score
    if extreme and score > 74.0:
        score = 74.0
    outlook = (
        "Conflicting Signals"
        if conflict
        else ("Input Unavailable" if blocked else _outlook(score))
    )

    confidence = (
        0.55
        + 0.25 * abs(raw)
        + (0.10 if news_article_count >= config.minimum_news_articles else 0.0)
    ) * factor
    if news_article_count < config.minimum_news_articles:
        confidence = min(confidence, 0.65)
    if stale:
        confidence = min(confidence, 0.40)
    if conflict or unavailable:
        confidence = min(confidence, 0.35)
    if extreme:
        confidence = min(confidence, 0.45)
    confidence = _clip(confidence, 0.0, 1.0)

    risk_name = _risk_level(displayed_risk)
    action = _action(outlook, risk_name, blocked)
    change = _position_change(score, blocked, extreme)
    why = [
        _direction(strategy, "Daily strategy"),
    ]
    if news_article_count:
        why.append(
            _direction(news_value, "Weighted news sentiment")
            + f" Based on {news_article_count} unique relevant article"
            + ("." if news_article_count == 1 else "s.")
        )
    else:
        why.append(
            "No current relevant news was available; its directional weight "
            "was reassigned to daily strategy."
        )
    why.append(
        f"Portfolio health is {displayed_health:.0f}/100 "
        f"({health_value:+.2f} after normalization)."
    )
    why.append(
        f"{risk_name} volatility ({displayed_risk:.0f}th percentile) reduced "
        f"directional conviction by {(1.0 - factor):.0%} without voting bearish."
    )
    if 0 < news_article_count < config.minimum_news_articles:
        why.append(
            "News confidence is low, so news weight was capped at 10% and "
            "the unused weight moved to strategy."
        )
    if conflict:
        why.append(
            "Strong strategy/news disagreement forced Hold instead of averaging "
            "the opposing signals into a trade."
        )
    if extreme:
        why.append(
            "Extreme volatility capped positive exposure and position size."
        )
    if stale:
        why.append("Stale inputs: " + ", ".join(stale) + ".")
    if unavailable:
        why.append("Unavailable inputs: " + ", ".join(unavailable) + ".")

    return AssetFusionResult(
        symbol=symbol,
        aurora_score=round(score, 1),
        outlook=outlook,
        risk_level=risk_name,
        action=action,
        confidence=round(confidence, 4),
        confidence_label=_confidence_label(confidence),
        strategy_score=round(strategy, 6),
        news_score=round(news_value, 6),
        health_score=round(displayed_health, 2),
        health_normalized=round(health_value, 6),
        risk_percentile=round(displayed_risk, 2),
        risk_factor=round(factor, 6),
        raw_score=round(raw, 6),
        adjusted_score=round(adjusted, 6),
        component_weights={
            "strategy": round(strategy_weight, 4),
            "news": round(news_weight, 4),
            "health": round(health_weight, 4),
        },
        news_articles=int(news_article_count),
        news_confidence=(
            "High"
            if news_article_count >= config.minimum_news_articles
            else "Low"
        ),
        news_titles=list(news_titles or []),
        position_change_pct=round(change, 2),
        conflict=conflict,
        extreme_volatility=extreme,
        stale_inputs=stale,
        unavailable_inputs=unavailable,
        as_of=dict(as_of or {}),
        why=why,
    )


def _business_age_days(value: object, now: datetime) -> Optional[int]:
    timestamp = _to_utc(value)
    if timestamp is None:
        return None
    start = np.datetime64(timestamp.date())
    end = np.datetime64(_to_utc(now).date())
    if start >= end:
        return 0
    return int(np.busday_count(start, end))


def _is_stale(
    value: object,
    now: datetime,
    maximum_business_days: int,
) -> bool:
    age = _business_age_days(value, now)
    return age is None or age > maximum_business_days


def _latest_relevant_news_time(
    events: Sequence[object],
    symbol: str,
) -> Optional[datetime]:
    timestamps = [
        timestamp
        for event in events
        if _relevance(event, symbol) > 0
        for timestamp in [_to_utc(_event_value(event, "published"))]
        if timestamp is not None
    ]
    return max(timestamps, default=None)


def directional_strategy_scores(
    signals: Sequence[AssetSignal],
) -> dict[str, float]:
    """Remove the upstream volatility rank from the directional strategy vote.

    The legacy AssetSignal score contains a 15% low-volatility preference.
    Volatility is not direction, so the fusion adapter rebuilds the score from
    momentum, trend, Sharpe, and drawdown ranks and lets HAR-X handle risk.
    If the indicator block is unavailable, it falls back to the supplied score.
    """
    rows = []
    for signal in signals:
        indicators = signal.indicators or {}
        required = ("momentum", "trend", "sharpe", "drawdown")
        try:
            values = {name: float(indicators[name]) for name in required}
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values.values()):
            rows.append({"symbol": signal.symbol, **values})

    directional: dict[str, float] = {}
    if rows:
        frame = pd.DataFrame(rows).set_index("symbol")
        ranks = frame.rank(pct=True)
        score_01 = (
            0.30 * ranks["momentum"]
            + 0.25 * ranks["trend"]
            + 0.20 * ranks["sharpe"]
            + 0.10 * ranks["drawdown"]
        ) / 0.85
        directional.update(
            {
                str(symbol): _clip(2.0 * float(value) - 1.0, -1.0, 1.0)
                for symbol, value in score_01.items()
            }
        )
    for signal in signals:
        directional.setdefault(signal.symbol, normalize_strategy(signal.score))
    return directional


def recommend_portfolio(
    symbols: Sequence[str],
    signals: Sequence[AssetSignal],
    news_events: Iterable[object],
    health_score: Optional[float],
    risk_estimates: Iterable[object],
    current_weights_pct: Mapping[str, float],
    *,
    strategy_as_of: object = None,
    health_as_of: object = None,
    now: Optional[datetime] = None,
    config: FusionConfig = DEFAULT_CONFIG,
) -> FusionDecision:
    """Build one result per asset and an API-compatible Recommendation."""
    now_utc = _to_utc(now or datetime.now(timezone.utc))
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    event_list = list(news_events)
    strategy_map = directional_strategy_scores(signals)
    risk_map = {
        str(getattr(estimate, "symbol", "")): estimate
        for estimate in risk_estimates
        if int(getattr(estimate, "horizon", 0)) == 5
        and bool(getattr(estimate, "has_history", False))
    }
    health_available = health_score is not None

    results: list[AssetFusionResult] = []
    for symbol in dict.fromkeys(str(item) for item in symbols):
        estimate = risk_map.get(symbol)
        risk_as_of = getattr(estimate, "as_of", None) if estimate else None
        stale = []
        if _is_stale(
            strategy_as_of,
            now_utc,
            config.strategy_stale_business_days,
        ):
            stale.append("strategy")
        if estimate is not None and _is_stale(
            risk_as_of,
            now_utc,
            config.risk_stale_business_days,
        ):
            stale.append("risk")
        if health_available and _is_stale(
            health_as_of,
            now_utc,
            config.health_stale_business_days,
        ):
            stale.append("health")

        unavailable = []
        if symbol not in strategy_map:
            unavailable.append("strategy")
        if estimate is None:
            unavailable.append("risk")
        if not health_available:
            unavailable.append("health")

        news = aggregate_news(
            event_list,
            symbol,
            now=now_utc,
            minimum_articles=config.minimum_news_articles,
        )
        latest_news = _latest_relevant_news_time(event_list, symbol)
        result = fuse_scores(
            symbol=symbol,
            strategy_score=strategy_map.get(symbol),
            news_score=news.score,
            health_score=health_score,
            risk_percentile=(
                float(getattr(estimate, "risk_level"))
                if estimate is not None
                else None
            ),
            news_article_count=news.unique_articles,
            news_titles=news.titles,
            stale_inputs=stale,
            unavailable_inputs=unavailable,
            as_of={
                "strategy": (
                    _to_utc(strategy_as_of).isoformat()
                    if _to_utc(strategy_as_of)
                    else None
                ),
                "news": latest_news.isoformat() if latest_news else None,
                "health": (
                    _to_utc(health_as_of).isoformat()
                    if _to_utc(health_as_of)
                    else None
                ),
                "risk": (
                    _to_utc(risk_as_of).isoformat()
                    if _to_utc(risk_as_of)
                    else None
                ),
            },
            config=config,
        )
        results.append(result)

    proposed = [
        ProposedTrade(
            result.symbol,
            result.position_change_pct,
            f"{result.outlook}; {result.risk_level.lower()} near-term risk. "
            + result.why[-1],
        )
        for result in results
        if result.position_change_pct
    ]
    constrained = apply_constraints(proposed, dict(current_weights_pct))
    constrained_map = {
        trade.symbol: trade.weight_change_pct for trade in constrained
    }
    for result in results:
        result.position_change_pct = float(
            constrained_map.get(result.symbol, 0.0)
        )

    positive = sum(result.aurora_score >= 55 for result in results)
    negative = sum(result.aurora_score < 45 for result in results)
    conflicts = sum(result.conflict for result in results)
    explanation = (
        f"Rule fusion evaluated {len(results)} assets: {positive} positive, "
        f"{negative} negative, and {conflicts} with strong signal conflict. "
        "Strategy, news, and portfolio health determine direction; HAR-X risk "
        "only reduces conviction and position size."
    )
    portfolio_confidence = (
        float(np.mean([result.confidence for result in results]))
        if results
        else 0.0
    )
    recommendation = Recommendation(
        kind="daily",
        trades=constrained,
        confidence=round(portfolio_confidence, 4),
        explanation=explanation,
    )
    metadata = {
        "production_mode": "rule_fusion",
        "model_version": MODEL_VERSION,
        "base_weights": {
            "strategy": config.strategy_weight,
            "news": config.news_weight,
            "health": config.health_weight,
        },
        "risk_formula": (
            "risk_factor = 1 - 0.45 * risk_percentile / 100"
        ),
        "health_scope": "portfolio",
        "strategy_adapter": "directional_only_excludes_volatility_rank",
        "news_policy": {
            "minimum_articles_for_full_weight": config.minimum_news_articles,
            "low_news_weight": config.low_news_weight,
            "zero_news_weight": 0.0,
            "duplicates_grouped": True,
            "maximum_age_days": 5,
        },
        "assets_evaluated": len(results),
    }
    return FusionDecision(recommendation, results, metadata)
