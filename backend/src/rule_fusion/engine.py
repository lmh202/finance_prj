"""Rule fusion — four inputs, four ordered steps, one explainable decision.

THE RULE THAT DEFINES THIS ENGINE: the four inputs are never averaged into a
single score. Each one acts on exactly one output dimension, in a fixed order,
and every step appends a row to a confidence ledger returned to the caller:

    STEP 1  daily strategy      -> direction + base confidence
    STEP 2  news sentiment      -> confidence; direction ONLY via a gated
                                   critical-event override
    STEP 3  health score        -> confidence      (never direction)
    STEP 4  volatility pctile   -> position size    (never direction,
                                                     never confidence)

decide() is pure — normalized inputs in, decision out, no I/O — so the whole
rule table can be exercised offline (scripts/fusion_selfcheck.py) and live
(POST /fusion/simulate). adapters.py normalizes the other engines' output into
FusionInputs; the router does the data loading.

Ledger invariant: every Adjustment carries confidence_before/delta/
confidence_after, and the deltas telescope — sum(a.delta) == final confidence.
A step that is not allowed to move confidence still records a row, with
delta 0.0 and a note saying so, which makes the invariant visible in the API
response rather than something you have to take on trust.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import pandas as pd

from src.interfaces import CONSTRAINTS, HealthReport, NewsEvent
from src.rule_fusion import adapters

# Dev 2's own public thresholds — imported rather than re-hardcoded so the two
# engines can never drift apart on what "a BUY" or "a maximal score" means.
from src.daily_strategy.engine import (
    BUY_THRESHOLD,
    W_EMA,
    W_MACD,
    W_RSI,
    StrategyRecommendation,
)

MAX_STRATEGY_SCORE = W_EMA + W_MACD + W_RSI          # 3.5 — a perfect confluence

DIRECTIONS = ("BUY", "SELL", "NEUTRAL")

ACTIONS = ("NEW_BUY", "ADD", "HOLD", "TRIM", "CLOSE")
ACTION_LABELS = {
    "NEW_BUY": "New Buy",              # not currently held — initiate a position
    "ADD": "Add to Position",          # already held — increase it
    "HOLD": "Hold",
    "TRIM": "Trim",                    # already held — partial sell-off
    "CLOSE": "Close Position",         # already held — fully liquidate
}

# ------------------------------------------------- STEP 1: strategy -> direction

DIRECTION_FROM_SIGNAL = {"BUY": "BUY", "SELL": "SELL", "HOLD": "NEUTRAL"}

NEUTRAL_BASE_CONFIDENCE = 0.20
DIRECTIONAL_BASE_CONFIDENCE = 0.45
DIRECTIONAL_CONVICTION_BONUS = 0.20      # so a directional call starts at 0.45–0.65

# ------------------------------------------------ STEP 2: news -> confidence

# Disagreement outweighs agreement on purpose: news that confirms the technicals
# tells you little you did not already know, news that contradicts them is
# genuinely new information.
NEWS_AGREE_GAIN = 0.15
NEWS_DISAGREE_LOSS = 0.20
NEWS_NEUTRAL_DRAG = 0.05

# |sentiment| at which the news channel counts as fully expressed.
NEWS_STRENGTH_FULL_SENTIMENT = 0.6

# The critical-event override is gated — a single loosely worded headline must
# not be able to invert a position.
CRITICAL_MIN_IMPORTANCE = 60.0
CRITICAL_MIN_ABS_SENTIMENT = 0.35
CRITICAL_OVERRIDE_BASE = 0.45
CRITICAL_OVERRIDE_GAIN = 0.25
CRITICAL_OVERRIDE_CAP = 0.75             # acting against the technicals is never certain
CRITICAL_AMBIGUOUS_CONFIDENCE = 0.15     # decisive event, unclear sign -> stand aside

# ---------------------------------------------- STEP 3: health -> confidence

HEALTH_STRONG = 70.0
HEALTH_FRAGILE = 40.0

# Direction-aware, and deliberately a table rather than a curve: a rule engine
# should be readable off the page. Weak health lowers conviction in a BUY but
# RAISES it in a SELL — de-risking is exactly what a fragile portfolio needs.
# This changes confidence only; direction is untouchable here.
HEALTH_ADJUSTMENTS = {
    "strong": {"BUY": +0.10, "SELL": -0.05, "NEUTRAL": 0.0},
    "steady": {"BUY": 0.0, "SELL": 0.0, "NEUTRAL": 0.0},
    "fragile": {"BUY": -0.15, "SELL": +0.10, "NEUTRAL": 0.0},
}

# ------------------------------------------------ STEP 4: volatility -> size

# Symmetric volatility targeting: the multiplier scales the magnitude of any
# exposure change so its risk contribution stays roughly constant.
VOLATILITY_SIZE_LADDER = (
    (25.0, 1.25),
    (50.0, 1.00),
    (75.0, 0.75),
    (90.0, 0.50),
    (float("inf"), 0.25),
)
UNKNOWN_VOLATILITY_MULTIPLIER = 1.00

BASE_TRADE_WEIGHT_PCT = 2.0              # weight points for a NEW_BUY / ADD at 1.00x
BASE_TRIM_FRACTION = 0.33                # share of the position for a TRIM at 1.00x
MAX_PARTIAL_TRIM_FRACTION = 0.75         # a partial trim never becomes a full exit

# --------------------------------------------------------- action selection

ACT_CONFIDENCE_FLOOR = 0.35              # below this, do nothing whatever the direction
CLOSE_CONFIDENCE = 0.70                  # a full exit needs real conviction

# ------------------------------------------------------------- risk banding

RISK_BANDS = ("low", "moderate", "elevated", "high", "extreme")
RISK_BAND_LADDER = ((25.0, "low"), (60.0, "moderate"), (85.0, "elevated"), (float("inf"), "high"))
UNKNOWN_RISK_BAND = "moderate"


# =========================================================== types

@dataclass
class Adjustment:
    """One row of the confidence ledger — what a step saw and what it did."""

    step: int                            # 1..4
    source: str                          # strategy | news | health | volatility
    observation: str                     # what the input said, as a sentence
    confidence_before: float
    delta: float                         # signed; 0.0 for a step that may not move it
    confidence_after: float
    note: str                            # which rule fired, and why


@dataclass
class FusionInputs:
    """Everything decide() needs. Hand-constructible — no engine call required."""

    symbol: str
    held: bool = False
    weight_pct: float = 0.0              # current portfolio weight, 0 if not held

    strategy_signal: str = "HOLD"        # BUY | SELL | HOLD
    strategy_score: float = 0.0          # -3.5 .. +3.5
    strategy_reasons: List[str] = field(default_factory=list)

    news_label: str = "none"             # positive | neutral | negative | none
    news_sentiment: float = 0.0          # -1..1, importance-weighted
    news_importance: float = 0.0         # 0..100 of the driving story
    news_headline: str = ""
    critical_category: Optional[str] = None   # interest_rate | war_geopolitical | ...
    critical_keyword: Optional[str] = None    # the phrase that matched

    health_score: Optional[float] = None      # None = unknown (NOT 0.0)

    volatility_pct: Optional[float] = None    # None = unknown
    volatility_source: str = "unavailable"    # risk_engine | realized_fallback | unavailable


@dataclass
class RiskView:
    """Risk is reported, not blended: `level` is the volatility percentile
    verbatim, and `band` escalates one notch per named driver."""

    level: Optional[float]
    band: str
    drivers: List[str] = field(default_factory=list)


@dataclass
class SizeView:
    multiplier: float                    # from volatility alone
    weight_points: float                 # NEW_BUY / ADD: weight-pct change to propose
    trim_fraction: float                 # TRIM / CLOSE: share of the position to sell


@dataclass
class FusionDecision:
    symbol: str
    direction: str                       # BUY | SELL | NEUTRAL
    confidence: float                    # 0..1
    risk: RiskView
    suggested_action: str                # NEW_BUY | ADD | HOLD | TRIM | CLOSE
    action_label: str                    # human form of suggested_action
    size: SizeView
    explanation: str
    adjustments: List[Adjustment]        # always 4 rows, one per step
    inputs: FusionInputs                 # echoed back so a UI can show what fed it
    overridden: bool = False             # True if a critical event replaced Step 1


# =========================================================== helpers

def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _record(
    ledger: List[Adjustment],
    step: int,
    source: str,
    observation: str,
    before: float,
    after: float,
    note: str,
) -> float:
    """Append a ledger row and return the (rounded) confidence to carry forward.

    Rounding here — not at the end — is what makes the deltas telescope exactly,
    so sum(delta) == final confidence holds for the caller too.
    """
    before_r, after_r = round(before, 4), round(_clip01(after), 4)
    ledger.append(
        Adjustment(
            step=step,
            source=source,
            observation=observation,
            confidence_before=before_r,
            delta=round(after_r - before_r, 4),
            confidence_after=after_r,
            note=note,
        )
    )
    return after_r


def _news_strength(sentiment: float, importance: float) -> float:
    """How loudly the news channel is speaking, 0..1. Tone drives it; importance
    scales it down but never to zero (a quiet source still said something)."""
    tone = _clip01(abs(sentiment) / NEWS_STRENGTH_FULL_SENTIMENT)
    weight = 0.5 + 0.5 * _clip01(importance / 100.0)
    return tone * weight


def _size_multiplier(volatility_pct: Optional[float]) -> float:
    if volatility_pct is None or not math.isfinite(volatility_pct):
        return UNKNOWN_VOLATILITY_MULTIPLIER
    for ceiling, multiplier in VOLATILITY_SIZE_LADDER:
        if volatility_pct < ceiling:
            return multiplier
    return VOLATILITY_SIZE_LADDER[-1][1]


def _base_risk_band(volatility_pct: Optional[float]) -> str:
    if volatility_pct is None or not math.isfinite(volatility_pct):
        return UNKNOWN_RISK_BAND
    for ceiling, band in RISK_BAND_LADDER:
        if volatility_pct < ceiling:
            return band
    return RISK_BAND_LADDER[-1][1]


def _escalate(band: str, notches: int) -> str:
    idx = min(len(RISK_BANDS) - 1, RISK_BANDS.index(band) + max(0, notches))
    return RISK_BANDS[idx]


def _critical_override_applies(inputs: FusionInputs) -> bool:
    """All three gates must hold. Relevance is the adapter's job — it only sets
    critical_category for an event that actually names this symbol."""
    return bool(inputs.critical_category) and inputs.news_importance >= CRITICAL_MIN_IMPORTANCE


def _select_action(direction: str, confidence: float, held: bool):
    """(direction, confidence, held) -> (action, why). Deterministic and total."""
    if confidence < ACT_CONFIDENCE_FLOOR:
        return "HOLD", "confidence is below the action floor — not convinced enough to trade"
    if direction == "NEUTRAL":
        return "HOLD", "no directional edge to act on"
    if direction == "BUY":
        return ("ADD", "already held and the case is bullish") if held else (
            "NEW_BUY",
            "not currently held and the case is bullish",
        )
    if not held:
        return "HOLD", "not held — there is nothing to sell, so stay out"
    if confidence >= CLOSE_CONFIDENCE:
        return "CLOSE", "held, bearish, and confident enough to exit in full"
    return "TRIM", "held and bearish, but not confident enough for a full exit"


def _size_for(action: str, multiplier: float) -> SizeView:
    if action in ("NEW_BUY", "ADD"):
        return SizeView(multiplier, round(BASE_TRADE_WEIGHT_PCT * multiplier, 2), 0.0)
    if action == "TRIM":
        fraction = min(MAX_PARTIAL_TRIM_FRACTION, BASE_TRIM_FRACTION * multiplier)
        return SizeView(multiplier, 0.0, round(fraction, 3))
    if action == "CLOSE":
        # A full exit is the decision itself, so the multiplier is reported but
        # not applied — volatility sizes exposure changes, it does not get to
        # veto a decision to be flat.
        return SizeView(multiplier, 0.0, 1.0)
    return SizeView(multiplier, 0.0, 0.0)


def _size_phrase(action: str, size: SizeView) -> str:
    if action in ("NEW_BUY", "ADD"):
        return f" — target a +{size.weight_points:.1f} weight-point change"
    if action == "TRIM":
        return f" — reduce the position by about {size.trim_fraction:.0%}"
    if action == "CLOSE":
        return " — exit the position in full"
    return ""


# =========================================================== the four steps

def decide(inputs: FusionInputs) -> FusionDecision:
    """Run the four steps in order. Pure: no I/O, no globals, deterministic."""
    ledger: List[Adjustment] = []
    signal = (inputs.strategy_signal or "HOLD").strip().upper()
    score = float(inputs.strategy_score)

    # ---------------------------------------------------------------- STEP 1
    # The daily strategy — and nothing else — decides the market direction.
    direction = DIRECTION_FROM_SIGNAL.get(signal, "NEUTRAL")
    strategy_direction = direction
    reasons = ", ".join(inputs.strategy_reasons[:2])
    detail = f": {reasons}" if reasons else ""

    if direction == "NEUTRAL":
        base = NEUTRAL_BASE_CONFIDENCE
        observation = f"Daily strategy is neutral on {inputs.symbol} (score {score:+.2f}{detail})."
        note = "No directional edge, so the fused direction starts neutral."
    else:
        span = max(1e-9, MAX_STRATEGY_SCORE - BUY_THRESHOLD)
        conviction = _clip01((abs(score) - BUY_THRESHOLD) / span)
        base = DIRECTIONAL_BASE_CONFIDENCE + DIRECTIONAL_CONVICTION_BONUS * conviction
        observation = f"Daily strategy says {direction} on {inputs.symbol} (score {score:+.2f}{detail})."
        note = (
            f"Direction comes from the strategy alone; its score sits {conviction:.0%} "
            "of the way through the signal's range, which sets the starting confidence."
        )
    confidence = _record(ledger, 1, "strategy", observation, 0.0, base, note)

    # ---------------------------------------------------------------- STEP 2
    # News adjusts confidence. Only a gated critical event may change direction.
    label = (inputs.news_label or "none").strip().lower()
    strength = _news_strength(inputs.news_sentiment, inputs.news_importance)
    before = confidence
    overridden = False
    disagreement = False

    if _critical_override_applies(inputs):
        overridden = True
        headline = inputs.news_headline or "a critical story"
        tag = f'{inputs.critical_category} ("{inputs.critical_keyword}")'
        if abs(inputs.news_sentiment) >= CRITICAL_MIN_ABS_SENTIMENT:
            news_direction = "BUY" if inputs.news_sentiment > 0 else "SELL"
            disagreement = direction != "NEUTRAL" and news_direction != direction
            verb = "overrides" if disagreement else "confirms"
            direction = news_direction
            after = min(
                CRITICAL_OVERRIDE_CAP, CRITICAL_OVERRIDE_BASE + CRITICAL_OVERRIDE_GAIN * strength
            )
            observation = (
                f"A critical {tag} story {verb} the strategy — "
                f'"{headline}" (importance {inputs.news_importance:.0f}, '
                f"sentiment {inputs.news_sentiment:+.2f}) reads as {news_direction}."
            )
            note = (
                "Critical news is allowed to set the direction. Confidence is rebased "
                f"rather than nudged, and capped at {CRITICAL_OVERRIDE_CAP:.0%} because "
                "an override means acting ahead of the technicals."
            )
        else:
            direction = "NEUTRAL"
            after = CRITICAL_AMBIGUOUS_CONFIDENCE
            observation = (
                f"A critical {tag} story is in play — "
                f'"{headline}" (importance {inputs.news_importance:.0f}) — but its '
                f"sentiment {inputs.news_sentiment:+.2f} is too ambiguous to read."
            )
            note = (
                "A decisive event with an unclear sign is a reason to stand aside, not "
                "to guess: direction is forced to neutral and confidence collapses."
            )
    elif label == "none":
        after = confidence
        observation = f"No recent story maps to {inputs.symbol}."
        note = "Step 2 is a no-op — the news channel had nothing to say."
    elif direction == "NEUTRAL":
        after = confidence
        observation = f"News on {inputs.symbol} reads {label}, but the strategy has no direction."
        note = "Nothing to agree or disagree with, so news leaves confidence alone."
    else:
        agrees = (label == "positive" and direction == "BUY") or (
            label == "negative" and direction == "SELL"
        )
        opposes = (label == "positive" and direction == "SELL") or (
            label == "negative" and direction == "BUY"
        )
        headline = inputs.news_headline or "recent coverage"
        if agrees:
            after = confidence + NEWS_AGREE_GAIN * strength
            observation = (
                f'News agrees with the {direction} case — "{headline}" reads {label} '
                f"(sentiment {inputs.news_sentiment:+.2f})."
            )
            note = "Agreement raises confidence; it does not touch direction."
        elif opposes:
            disagreement = True
            after = confidence - NEWS_DISAGREE_LOSS * strength
            observation = (
                f'News disagrees with the {direction} case — "{headline}" reads {label} '
                f"(sentiment {inputs.news_sentiment:+.2f})."
            )
            note = (
                "Disagreement cuts confidence harder than agreement raises it, but only "
                "a critical event may change direction."
            )
        else:
            after = confidence - NEWS_NEUTRAL_DRAG * strength
            observation = (
                f'News on {inputs.symbol} is neutral — "{headline}" '
                f"(sentiment {inputs.news_sentiment:+.2f})."
            )
            note = "Ambiguous coverage applies a small drag on confidence."
    confidence = _record(ledger, 2, "news", observation, before, after, note)

    # ---------------------------------------------------------------- STEP 3
    # Health changes confidence only — it can never change direction.
    before = confidence
    if inputs.health_score is None:
        after = confidence
        observation = "Portfolio health is unavailable (not enough history to score it)."
        note = "Step 3 is a no-op — an unknown health score is not treated as a bad one."
    else:
        health = float(inputs.health_score)
        tier = "strong" if health >= HEALTH_STRONG else (
            "fragile" if health < HEALTH_FRAGILE else "steady"
        )
        delta = HEALTH_ADJUSTMENTS[tier][direction]
        after = confidence + delta
        observation = f"Portfolio health is {health:.0f}/100 ({tier})."
        if delta > 0:
            note = (
                f"A {tier} portfolio strengthens the case for a {direction}; health moves "
                "confidence only, never direction."
            )
        elif delta < 0:
            note = (
                f"A {tier} portfolio weakens the case for a {direction}; health moves "
                "confidence only, never direction."
            )
        else:
            note = "Health is in the neutral band (or the call is neutral) — no change."
    confidence = _record(ledger, 3, "health", observation, before, after, note)

    # ---------------------------------------------------------------- STEP 4
    # Volatility sizes the position and touches nothing else. The row is
    # recorded as an explicit no-op so the invariant is visible, not implied.
    multiplier = _size_multiplier(inputs.volatility_pct)
    if inputs.volatility_pct is None or not math.isfinite(inputs.volatility_pct):
        observation = (
            f"Volatility percentile unavailable ({inputs.volatility_source}) — "
            f"sizing at {multiplier:.2f}x."
        )
    else:
        observation = (
            f"Volatility is in the {inputs.volatility_pct:.0f}th percentile of "
            f"{inputs.symbol}'s own history ({inputs.volatility_source}) — "
            f"sizing at {multiplier:.2f}x."
        )
    confidence = _record(
        ledger,
        4,
        "volatility",
        observation,
        confidence,
        confidence,
        "Volatility never changes direction or confidence — Step 4 only sizes the position.",
    )

    # ------------------------------------------------------ action, size, risk
    action, why = _select_action(direction, confidence, inputs.held)
    size = _size_for(action, multiplier)

    drivers: List[str] = []
    if inputs.critical_category:
        drivers.append(f"critical {inputs.critical_category} news")
    if disagreement:
        drivers.append("strategy and news disagree")
    if inputs.health_score is not None and inputs.health_score < HEALTH_FRAGILE:
        drivers.append("portfolio health is fragile")
    if inputs.weight_pct > CONSTRAINTS["max_stock_weight_pct"]:
        drivers.append(f"position is {inputs.weight_pct:.1f}% of the portfolio")
    risk = RiskView(
        level=inputs.volatility_pct,
        band=_escalate(_base_risk_band(inputs.volatility_pct), len(drivers)),
        drivers=drivers,
    )

    verdict = (
        f"{ACTION_LABELS[action]}{_size_phrase(action, size)} — {why} "
        f"(confidence {confidence:.0%}, risk {risk.band})."
    )
    explanation = " ".join(a.observation for a in ledger) + " -> " + verdict

    return FusionDecision(
        symbol=inputs.symbol,
        direction=direction,
        confidence=confidence,
        risk=risk,
        suggested_action=action,
        action_label=ACTION_LABELS[action],
        size=size,
        explanation=explanation,
        adjustments=ledger,
        inputs=inputs,
        overridden=overridden,
    )


# =========================================================== batch entry point

def fuse(
    strategy_recs: Sequence[StrategyRecommendation],
    events: Sequence[NewsEvent] = (),
    health: Optional[HealthReport] = None,
    risk_levels: Optional[Mapping[str, float]] = None,
    history: Optional[pd.DataFrame] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> List[FusionDecision]:
    """One decision per scored symbol.

    Everything is passed IN — this module performs no data loading (same
    contract as risk_engine.risk_estimates). `risk_levels` is symbol ->
    RiskEstimate.risk_level; where it is missing or NaN the adapter falls back
    to a realized-volatility percentile computed from `history`.
    """
    risk_levels = risk_levels or {}
    weights = weights or {}
    health_score = adapters.health_input(health)
    decisions: List[FusionDecision] = []

    for rec in strategy_recs:
        news = adapters.news_view(rec.symbol, events)
        critical = adapters.critical_scan(rec.symbol, events)
        vol_pct, vol_source = adapters.volatility_view(
            rec.symbol, risk_levels.get(rec.symbol), history
        )
        # When a critical story is in play the news_* fields describe THAT
        # story, not the portfolio-wide average — the override has to be judged
        # on the sentiment of the event that triggered it, otherwise unrelated
        # background chatter could dilute a decisive headline into ambiguity.
        driver = critical or news
        decisions.append(
            decide(
                FusionInputs(
                    symbol=rec.symbol,
                    held=bool(rec.held),
                    weight_pct=float(weights.get(rec.symbol, 0.0)),
                    strategy_signal=rec.raw_signal,
                    strategy_score=float(rec.score),
                    strategy_reasons=list(rec.reasons),
                    news_label=driver.label,
                    news_sentiment=driver.sentiment,
                    news_importance=driver.importance,
                    news_headline=driver.headline,
                    critical_category=(critical.category if critical else None),
                    critical_keyword=(critical.keyword if critical else None),
                    health_score=health_score,
                    volatility_pct=vol_pct,
                    volatility_source=vol_source,
                )
            )
        )
    return decisions


def rank(decisions: Sequence[FusionDecision]) -> List[FusionDecision]:
    """Most actionable first: acted-on calls before holds, then by confidence."""
    order: Dict[str, int] = {"CLOSE": 0, "TRIM": 1, "NEW_BUY": 2, "ADD": 3, "HOLD": 4}
    return sorted(decisions, key=lambda d: (order[d.suggested_action], -d.confidence, d.symbol))
