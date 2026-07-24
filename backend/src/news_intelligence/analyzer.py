"""Deterministic keyword-rule analysis: raw records -> src.interfaces.NewsEvent.

Ported from an earlier standalone prototype (an MVP built before the
four-engine split existed). No ML/LLM here by design — this is the always-
available fallback path described in this engine's README; a batched LLM
classification pass can sit in front of it later without touching this module.

Operates on collector.py's raw record schema (plain dicts, not a dataclass):
    id, title, summary, url, source, publisher, feed_url,
    published_utc (ISO 8601 UTC or None), fetched_utc (ISO 8601 UTC), symbols

`symbols` on a record is collector.py's own ticker tag: an entry fetched from
a per-ticker Yahoo feed is already known to be about that ticker, which is a
stronger relevance signal than any text match below.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.interfaces import NewsEvent

RULES_PATH = Path(__file__).resolve().parent / "rules.json"

# Architecture.md §6 weights for the six importance factors.
WEIGHTS = {
    "source_credibility": 0.20,
    "multi_source_confirmation": 0.15,
    "portfolio_relevance": 0.25,
    "event_severity": 0.15,
    "recency": 0.10,
    "expected_market_impact": 0.15,
}


def load_rules(path: Optional[Path] = None) -> Dict[str, Any]:
    return json.loads((path or RULES_PATH).read_text(encoding="utf-8"))


def _normalized_title(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def _record_time(record: dict) -> datetime:
    raw = record.get("published_utc") or record.get("fetched_utc")
    return datetime.fromisoformat(raw) if raw else datetime.now(timezone.utc)


def _contains(text: str, term: str, whole_word: bool = False) -> bool:
    if not term:
        return False
    if whole_word or (term.isalnum() and len(term) <= 5):
        return re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE) is not None
    return term.lower() in text.lower()


def _text(records: List[dict]) -> str:
    return " ".join(f"{r['title']}. {r.get('summary', '')}" for r in records)


def _tagged_symbols(records: List[dict]) -> set:
    return {s for r in records for s in r.get("symbols", [])}


# Memoizes the single most recent cluster() call, keyed on exactly which
# record ids went in. cluster() runs BEFORE any symbol filtering (fetch_
# headlines() and essential_news() both cluster the same all-symbols 48h
# window regardless of which tickers the caller passed), so calling
# essential_news() again right after a ticker add/remove — with no new
# articles collected in between — reclusters an identical record set. That
# used to mean redoing an O(records x groups) SequenceMatcher pass (the
# dominant cost once the store has a few hundred stories in the lookback
# window) purely to reproduce the same grouping. A cache keyed on the
# record-id set is exact, not an approximation: same ids in means same
# titles/urls in, so a hit is guaranteed to equal a fresh recompute.
_CLUSTER_CACHE: Optional[Tuple[Tuple[str, ...], List[List[dict]]]] = None


def cluster(records: List[dict]) -> List[List[dict]]:
    """Group near-duplicate stories: same canonical URL, or >=88% title match."""
    global _CLUSTER_CACHE
    key = tuple(sorted(r["id"] for r in records))
    if _CLUSTER_CACHE is not None and _CLUSTER_CACHE[0] == key:
        return _CLUSTER_CACHE[1]

    groups: List[List[dict]] = []
    # One SequenceMatcher per group, its (expensive-to-index) seq2 pinned to
    # the group's representative title and reused across every candidate
    # record — instead of rebuilding a fresh matcher (and recomputing the
    # representative's normalized title) on every single comparison.
    matchers: List[SequenceMatcher] = []
    for record in sorted(records, key=_record_time, reverse=True):
        normalized = _normalized_title(record["title"])
        for group, matcher in zip(groups, matchers):
            same_url = bool(record.get("url")) and record["url"] == group[0].get("url")
            if same_url:
                group.append(record)
                break
            matcher.set_seq1(normalized)
            if matcher.ratio() >= 0.88:
                group.append(record)
                break
        else:
            groups.append([record])
            new_matcher = SequenceMatcher(None, "", normalized)
            new_matcher.set_seq2(normalized)
            matchers.append(new_matcher)

    _CLUSTER_CACHE = (key, groups)
    return groups


def is_relevant(records: List[dict], holding_symbols: List[str], rules: Optional[dict] = None) -> bool:
    """Finance-relevance only (not holdings-specific): used by fetch_headlines
    with an empty holdings list, and as the first filter for essential_news."""
    if _tagged_symbols(records) & set(holding_symbols):
        return True
    rules = rules or load_rules()
    text = _text(records)
    terms = list(rules["finance_keywords"]) + list(holding_symbols)
    category_terms = [kw for cat in rules["categories"].values() for kw in cat["keywords"]]
    return any(_contains(text, term) for term in terms + category_terms)


def _category(text: str, rules: dict) -> Tuple[str, dict]:
    scores = {
        name: sum(_contains(text, keyword) for keyword in cat["keywords"])
        for name, cat in rules["categories"].items()
    }
    name = max(scores, key=scores.get)
    if scores[name] == 0:
        return "other", {"label": "Other", "impact_score": 30, "asset_impacts": {}}
    return name, rules["categories"][name]


def _sentiment(text: str, rules: dict) -> Tuple[float, str]:
    positives = sum(_contains(text, word) for word in rules["sentiment"]["positive"])
    negatives = sum(_contains(text, word) for word in rules["sentiment"]["negative"])
    matches = positives + negatives
    score = 0.0 if not matches else max(-1.0, min(1.0, (positives - negatives) / math.sqrt(matches)))
    label = "positive" if score >= 0.15 else "negative" if score <= -0.15 else "neutral"
    return round(score, 3), label


def _affected_symbols(
    text: str,
    records: List[dict],
    holding_symbols: List[str],
    holding_names: Dict[str, str],
    affected_assets: Dict[str, str],
    asset_map: Dict[str, List[str]],
    fallback_impact: str,
) -> Tuple[List[str], bool, Dict[str, str]]:
    """Two-tier holdings mapping: direct (ticker tagged/mentioned or company
    name mentioned) or indirect (symbol's mapped asset class intersects the
    event's affected asset classes, e.g. a rate-hike story implicating a
    long-duration bond ETF that it never names)."""
    tagged = _tagged_symbols(records)
    matched: List[str] = []
    direct = False
    impact: Dict[str, str] = {}
    for symbol in holding_symbols:
        name = holding_names.get(symbol, "")
        is_direct = (
            symbol in tagged
            or _contains(text, symbol, whole_word=True)
            or (bool(name) and _contains(text, name))
        )
        mapped = set(asset_map.get(symbol.upper(), []))
        overlap = mapped & set(affected_assets)
        if not (is_direct or overlap):
            continue
        matched.append(symbol)
        direct = direct or is_direct
        impact[symbol] = affected_assets[next(iter(overlap))] if overlap else fallback_impact
    return matched, direct, impact


def _summary(record: dict) -> str:
    source_text = record.get("summary") or record["title"]
    sentences = re.split(r"(?<=[.!?])\s+", source_text)
    summary = " ".join(sentences[:2]).strip()
    return summary if len(summary) <= 320 else summary[:317].rstrip() + "..."


def analyze(
    records: List[dict],
    holding_symbols: List[str],
    holding_names: Dict[str, str],
    now: datetime,
    lookback_hours: int,
    rules: Optional[dict] = None,
) -> NewsEvent:
    """Score one cluster of (near-)duplicate articles into a NewsEvent."""
    rules = rules or load_rules()
    text = _text(records)
    category, rule = _category(text, rules)
    sentiment_score, sentiment_label = _sentiment(text, rules)
    affected_assets = dict(rule.get("asset_impacts", {}))
    fallback_impact = "mixed" if sentiment_label == "neutral" else sentiment_label
    matched, direct, impact = _affected_symbols(
        text, records, holding_symbols, holding_names, affected_assets,
        rules.get("holding_asset_map", {}), fallback_impact,
    )

    credibility_map = rules.get("source_credibility", {})
    default_credibility = credibility_map.get("_default", 0.7)
    credibility = max(credibility_map.get(r["source"], default_credibility) for r in records) * 100
    source_count = len({r["source"] for r in records})
    confirmation = 100.0 if source_count >= 3 else 50.0 if source_count == 2 else 0.0
    portfolio_relevance = 100.0 if direct else 60.0 if matched else 20.0
    severity_hits = sum(_contains(text, word) for word in rule.get("severity_keywords", []))
    severity = min(100.0, 50.0 + severity_hits * 25.0)
    newest_record = max(records, key=_record_time)
    newest = _record_time(newest_record)
    age_hours = max(0.0, (now - newest).total_seconds() / 3600)
    recency = max(0.0, 100.0 * (1 - age_hours / lookback_hours)) if lookback_hours else 0.0
    market_impact = float(rule.get("impact_score", 30))

    reasons = {
        "source_credibility": credibility,
        "multi_source_confirmation": confirmation,
        "portfolio_relevance": portfolio_relevance,
        "event_severity": severity,
        "recency": recency,
        "expected_market_impact": market_impact,
    }
    importance = sum(reasons[name] * weight for name, weight in WEIGHTS.items())

    published = None
    if newest_record.get("published_utc"):
        published = datetime.fromisoformat(newest_record["published_utc"])

    return NewsEvent(
        title=newest_record["title"],
        source=newest_record["source"],
        url=newest_record["url"],
        published=published,
        category=category,
        sentiment=sentiment_score,
        importance=round(importance, 1),
        affected_symbols=sorted(matched),
        impact=impact,
        summary=_summary(newest_record),
    )
