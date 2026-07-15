"""Event Intelligence Engine — Developer 3 (Architecture.md §6). STUB.

Pipeline to build:

    RSS feeds -> fetch (feedparser) -> dedupe headlines -> relevance filter
              -> event classification (EVENT_CATEGORIES) -> sentiment
              -> importance score -> map to affected holdings

Both functions currently return [] so every consumer already renders an
empty news section without special-casing. Signatures are frozen in
src/interfaces.py.
"""

from typing import List, Optional

from src.interfaces import NewsEvent

# Candidate RSS feeds — extend/replace as you evaluate quality.
DEFAULT_FEEDS: List[str] = [
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",  # MarketWatch
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",       # CNBC Top News
    "https://finance.yahoo.com/news/rssindex",                     # Yahoo Finance
]


def fetch_headlines(feeds: Optional[List[str]] = None, limit: int = 50) -> List[NewsEvent]:
    """Fetch recent stories from RSS, dedupe, classify, and score them.

    TODO (Developer 3): implement with `feedparser` (announce + add to
    requirements.txt). Fill every NewsEvent field; leave importance crude
    at first and refine.
    """
    return []


def essential_news(holding_symbols: List[str], max_events: int = 5) -> List[NewsEvent]:
    """The top `max_events` stories that matter for THIS portfolio.

    TODO (Developer 3): score = source credibility + corroboration +
    portfolio relevance + severity + recency (Architecture.md §6), then
    keep the best few. Never return more than `max_events`.
    """
    return []
