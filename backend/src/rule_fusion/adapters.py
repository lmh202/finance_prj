"""Normalizers — turn the other engines' output into the fusion engine's inputs.

Each of the four channels arrives in a shape the fusion rules cannot use
directly, and each needs a guard:

  news        NewsEvent carries only a float sentiment. analyzer.py computes a
              positive/neutral/negative label and then discards it, and
              NewsEvent is frozen (amending it needs all four developers), so
              the label is re-derived here at the SAME +/-0.15 cut the analyzer
              uses in its own fallback — the two engines must agree on what
              "neutral" means.
  critical    The four-bucket taxonomy in critical_events.json. Deliberately
              this engine's own file: news_intelligence/rules.json is Dev 3's,
              has different category names, and is tuned for a different job.
  health      compute_health() returns score=0.0 with an EMPTY metrics dict
              when there is too little history. Passed through, that reads as
              "catastrophic portfolio". It becomes None (unknown) instead.
  volatility  risk_engine.risk_level is preferred, but it is NaN without the
              trained artifact and data/ is gitignored. The fallback computes
              the same quantity (a percentile of current vol against the
              symbol's own history) straight from prices, so a teammate
              without data/processed/risk_model.json still gets a decision.

This module imports no engine — values are passed in by the caller — so it
stays cheap to import and trivially testable offline.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.interfaces import HealthReport, NewsEvent

# Same cut analyzer._sentiment_keywords uses, so "neutral" means one thing.
NEWS_LABEL_THRESHOLD = 0.15

# Which critical bucket wins when several fire on one symbol. Mirrors the
# impact_score ordering already in news_intelligence/rules.json (95/90/85/65),
# so the two engines rank systemic risk the same way.
CRITICAL_PRIORITY = ("systemic_macro", "war_geopolitical", "interest_rate", "earnings_corporate")

# Fallback volatility percentile: 20-session annualized realized vol, ranked
# against its own trailing distribution.
REALIZED_WINDOW = 20
TRADING_DAYS = 252
MIN_FALLBACK_ROWS = 120          # enough rows for the window + a usable distribution
MIN_FALLBACK_SAMPLES = 60        # ...and enough vol observations to rank against

_TAXONOMY_PATH = Path(__file__).with_name("critical_events.json")


@dataclass
class NewsView:
    """The news channel's aggregate opinion on one symbol."""

    label: str = "none"              # positive | neutral | negative | none
    sentiment: float = 0.0           # importance-weighted, -1..1
    importance: float = 0.0          # of the loudest story
    headline: str = ""
    event_count: int = 0


@dataclass
class CriticalHit:
    """A single story that clears the critical-taxonomy bar for one symbol."""

    category: str
    keyword: str
    headline: str
    sentiment: float
    importance: float
    label: str


def _load_taxonomy() -> Dict[str, List[str]]:
    return json.loads(_TAXONOMY_PATH.read_text(encoding="utf-8"))


def _compile(taxonomy: Dict[str, List[str]]) -> List[Tuple[str, str, "re.Pattern[str]"]]:
    """(category, keyword, word-boundary regex), highest-priority bucket first."""
    ordered = sorted(
        taxonomy.items(),
        key=lambda kv: CRITICAL_PRIORITY.index(kv[0]) if kv[0] in CRITICAL_PRIORITY else 99,
    )
    compiled = []
    for category, keywords in ordered:
        for keyword in keywords:
            pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
            compiled.append((category, keyword, pattern))
    return compiled


TAXONOMY = _load_taxonomy()
_PATTERNS = _compile(TAXONOMY)


def label_for(sentiment: float) -> str:
    if sentiment >= NEWS_LABEL_THRESHOLD:
        return "positive"
    if sentiment <= -NEWS_LABEL_THRESHOLD:
        return "negative"
    return "neutral"


def _mentions(symbol: str, event: NewsEvent) -> bool:
    """The news engine maps stories to holdings two ways — a direct ticker/name
    match lands in affected_symbols, an asset-class read-through lands in
    impact. Either counts as this story being about the symbol."""
    return symbol in (event.affected_symbols or []) or symbol in (event.impact or {})


def _text(event: NewsEvent) -> str:
    return f"{event.title or ''} {event.summary or ''}"


def news_view(symbol: str, events: Sequence[NewsEvent]) -> NewsView:
    """Aggregate every story that names `symbol` into one opinion.

    Sentiment is importance-weighted so a major story outweighs background
    chatter; the reported headline and importance come from the loudest story
    so the explanation quotes something a user can go and read.
    """
    relevant = [e for e in events if _mentions(symbol, e)]
    if not relevant:
        return NewsView()

    total_importance = sum(max(0.0, float(e.importance)) for e in relevant)
    if total_importance > 0:
        sentiment = sum(float(e.sentiment) * max(0.0, float(e.importance)) for e in relevant)
        sentiment /= total_importance
    else:
        sentiment = sum(float(e.sentiment) for e in relevant) / len(relevant)

    loudest = max(relevant, key=lambda e: float(e.importance))
    return NewsView(
        label=label_for(sentiment),
        sentiment=round(float(sentiment), 3),
        importance=round(float(loudest.importance), 1),
        headline=loudest.title or "",
        event_count=len(relevant),
    )


def critical_scan(symbol: str, events: Sequence[NewsEvent]) -> Optional[CriticalHit]:
    """The highest-priority critical story naming `symbol`, or None.

    Relevance is enforced here rather than in the rules, so the engine can
    treat `critical_category is not None` as "this symbol is genuinely
    affected" without re-checking. Ties break by category priority first, then
    importance — deterministic, no dependence on feed ordering.
    """
    best: Optional[CriticalHit] = None
    best_rank = (len(CRITICAL_PRIORITY), -1.0)

    for event in events:
        if not _mentions(symbol, event):
            continue
        text = _text(event)
        for category, keyword, pattern in _PATTERNS:
            if not pattern.search(text):
                continue
            priority = CRITICAL_PRIORITY.index(category)
            rank = (priority, -float(event.importance))
            if rank < best_rank:
                best_rank = rank
                best = CriticalHit(
                    category=category,
                    keyword=keyword,
                    headline=event.title or "",
                    sentiment=round(float(event.sentiment), 3),
                    importance=round(float(event.importance), 1),
                    label=label_for(float(event.sentiment)),
                )
            break        # first (highest-priority) keyword hit represents this story
    return best


def health_input(report: Optional[HealthReport]) -> Optional[float]:
    """The 0-100 score, or None when it could not really be computed.

    compute_health() signals "not enough history" by returning score=0.0 with
    an empty metrics dict. Treating that as a real 0 would make every decision
    behave as if the portfolio were catastrophic.
    """
    if report is None or not getattr(report, "metrics", None):
        return None
    score = float(report.score)
    return score if math.isfinite(score) else None


def _realized_vol_percentile(prices: pd.Series) -> Optional[float]:
    """Percentile of the latest 20-session annualized vol against its own past.

    Same quantity risk_engine reports as risk_level, computed from close prices
    only — so the Step 4 size ladder stays meaningful without the trained model.
    """
    series = pd.to_numeric(prices, errors="coerce").dropna()
    series = series[series > 0]
    if len(series) < MIN_FALLBACK_ROWS:
        return None

    returns = np.log(series / series.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    vol = returns.rolling(REALIZED_WINDOW).std().dropna() * math.sqrt(TRADING_DAYS)
    vol = vol[vol > 0]
    if len(vol) < MIN_FALLBACK_SAMPLES:
        return None

    latest = float(vol.iloc[-1])
    if not math.isfinite(latest):
        return None
    return round(float((vol <= latest).mean() * 100.0), 1)


def volatility_view(
    symbol: str,
    risk_level: Optional[float] = None,
    history: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[float], str]:
    """(percentile 0-100 or None, source). Prefers the risk engine, degrades
    to prices, and says which one it used so the trace never hides it."""
    if risk_level is not None:
        level = float(risk_level)
        if math.isfinite(level):
            return round(level, 1), "risk_engine"

    if history is not None and not history.empty and symbol in history.columns:
        fallback = _realized_vol_percentile(history[symbol])
        if fallback is not None:
            return fallback, "realized_fallback"

    return None, "unavailable"
