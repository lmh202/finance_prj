"""Event Intelligence Engine — Developer 3 (Architecture.md §6).

Live pipeline: collector.py fetches + normalizes RSS into data/news_raw.json
(one feed per portfolio ticker, generated automatically, plus a few general
market feeds) -> this module clusters near-duplicate stories, filters for
relevance, classifies, scores and maps them to holdings -> NewsEvent.

Classification (category, relevance, severity) is deterministic keyword
rules (see analyzer.py + rules.json); sentiment is scored by a local FinBERT
model (see finbert_sentiment.py), with a keyword fallback if that model
can't load. Zero API keys required either way, matching this engine's
README, which calls the keyword path out as the always-available fallback;
a batched LLM classification pass can sit in front of it later without
changing fetch_headlines'/essential_news' signatures or callers.

sentiment_features() (the HISTORICAL pipeline feeding Developer 2's ML
ablation) is a separate concern — untouched stub here, see fnspid_prep.py.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import pandas as pd

from src import portfolio as pf
from src.interfaces import NewsEvent
from src.news_intelligence import analyzer, collector

# General-feed URLs only (no per-ticker feeds) — the default for fetch_headlines()
# and what GET /news/feeds reports. Derived from collector.py so the two never drift.
DEFAULT_FEEDS: List[str] = [feed["url"] for feed in collector.GENERAL_FEEDS]

# How far back to look for stories, and the coarse noise floor below which an
# event isn't worth surfacing at all. Not part of the frozen signatures below —
# tune here rather than growing the public parameter list.
_LOOKBACK_HOURS = 48
_MIN_IMPORTANCE_FLOOR = 35.0


def _holding_names(symbols: List[str]) -> Dict[str, str]:
    """symbol -> company name for the given symbols, used as free-text
    aliases (news prose says "Apple", not "AAPL")."""
    if not symbols:
        return {}
    holdings = pf.load_portfolio()
    if holdings.empty:
        return {}
    subset = holdings[holdings["symbol"].isin(symbols)]
    return dict(zip(subset["symbol"], subset["name"]))


def _recent_records(lookback_hours: int, now: datetime) -> List[dict]:
    since = now - timedelta(hours=lookback_hours)
    records = []
    for record in collector.load_store().values():
        raw = record.get("published_utc") or record.get("fetched_utc")
        if raw and datetime.fromisoformat(raw) >= since:
            records.append(record)
    return records


def _rank(events: List[NewsEvent]) -> List[NewsEvent]:
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(events, key=lambda e: (e.importance, e.published or epoch), reverse=True)


def fetch_headlines(feeds: Optional[List[str]] = None, limit: int = 50) -> List[NewsEvent]:
    """Recent general-market stories, classified and scored — no portfolio
    filtering (that's essential_news' job). `feeds`, if given, overrides the
    default general feeds with an explicit list of feed URLs.
    """
    urls = feeds or DEFAULT_FEEDS
    collector.collect_urls(urls)
    now = datetime.now(timezone.utc)
    records = _recent_records(_LOOKBACK_HOURS, now)
    clusters = [c for c in analyzer.cluster(records) if analyzer.is_relevant(c, [])]
    events = [analyzer.analyze(c, [], {}, now, _LOOKBACK_HOURS) for c in clusters]
    return _rank(events)[:limit]


def essential_news(holding_symbols: List[str], max_events: int = 5) -> List[NewsEvent]:
    """The top `max_events` stories that matter for THIS portfolio.

    score = source credibility + corroboration + portfolio relevance +
    severity + recency + expected market impact (Architecture.md §6),
    keeping only the best few above a coarse importance floor.
    """
    symbols = sorted({s.strip().upper() for s in holding_symbols if s and s.strip()})
    collector.collect(symbols)
    now = datetime.now(timezone.utc)
    records = _recent_records(_LOOKBACK_HOURS, now)
    holding_names = _holding_names(symbols)
    clusters = [c for c in analyzer.cluster(records) if analyzer.is_relevant(c, symbols)]
    events = [analyzer.analyze(c, symbols, holding_names, now, _LOOKBACK_HOURS) for c in clusters]
    events = [e for e in events if e.importance >= _MIN_IMPORTANCE_FLOOR]
    return _rank(events)[:max_events]


def sentiment_features(
    symbols: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """HISTORICAL news features for Developer 2's ML (optional input channel).

    Long format, one row per symbol-day that has news:
        date (datetime), symbol (str), sentiment (-1..1 importance-weighted),
        news_count (int), has_news (always 1 here; Dev 2 fills 0 elsewhere)

    Rules (see src/interfaces.py):
    - NO LOOK-AHEAD: a row dated t may only use news published before day
      t's market open.
    - Score the historical corpus (e.g. FNSPID / Kaggle financial news)
      with a LOCAL model (FinBERT or VADER) — the LLM API is for the live
      feed only; batch-scoring millions of headlines through it wastes money.
    - Cache the finished feature table to data/processed/ so it is built once.

    NOT IMPLEMENTED YET — separate concern from the live pipeline above
    (needs the FNSPID corpus + a local FinBERT/VADER pass, see
    fnspid_prep.py). Returns an empty, correctly-shaped frame so Developer 2
    can already code the merge + ablation against it.
    """
    return pd.DataFrame(columns=["date", "symbol", "sentiment", "news_count", "has_news"])
