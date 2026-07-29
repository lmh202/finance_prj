"""Generalized RSS collector — Developer 3 (Event Intelligence, stage 1).

Pulls any number of RSS feeds (Yahoo Finance per-ticker feeds are generated
from the portfolio automatically; add other sources as plain config entries)
and normalizes every entry into ONE stable record schema, accumulated in a
pretty-printed JSON store (data/news_raw.json) with idempotent dedupe — run
it as often as you like, e.g. on a schedule, and the archive grows without
duplicates.

Record schema (an array of objects, human-readable indentation):
    id             sha1 of the canonical URL (or title if no URL)
    title          headline text
    summary        plain-text summary/description ("" if the feed has none)
    url            canonical link
    source         feed display name (e.g. "Yahoo Finance: AAPL")
    publisher      original outlet if the feed names one, else ""
    feed_url       where it came from
    published_utc  ISO 8601 UTC (null if the feed omitted it)
    fetched_utc    ISO 8601 UTC when we saw it
    symbols        tickers this story is tagged with (union across feeds)

Run directly:
    python src/news_intelligence/collector.py            # portfolio symbols
    python src/news_intelligence/collector.py AAPL NVDA  # explicit symbols

The store is the RAW layer. The LLM classification stage reads from here;
training code can load it with:  pd.read_json("data/news_raw.json")
"""

import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import mktime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import feedparser
import requests

# backend/ dir — puts the `src` package on sys.path when run as a script
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR  # noqa: E402

STORE = DATA_DIR / "news_raw.json"

# ------------------------------------------------------------------ feed config
# To generalize to a new source, append an entry here — nothing else changes.
# `symbols`: tickers every entry of this feed should be tagged with ([] = general).

GENERAL_FEEDS: List[Dict] = [
    {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex", "symbols": []},
    {"name": "CNBC Top News", "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "symbols": []},
    {"name": "MarketWatch", "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories", "symbols": []},
]

YAHOO_TICKER_URL = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"
)

# Per-ticker feed templates: {sym} is substituted with the uppercased symbol.
# Every portfolio symbol gets one feed per template below — same "append an
# entry, nothing else changes" shape as GENERAL_FEEDS above. Yahoo was the
# only one for a while because it's the only general-market outlet with a
# public per-symbol RSS endpoint that's still alive; Seeking Alpha and
# Nasdaq's outbound feed both were verified live and per-symbol too.
TICKER_FEED_TEMPLATES: List[Dict] = [
    {"name": "Yahoo Finance", "url": YAHOO_TICKER_URL},
    {"name": "Seeking Alpha", "url": "https://seekingalpha.com/api/sa/combined/{sym}.xml"},
    {"name": "Nasdaq", "url": "https://www.nasdaq.com/feed/rssoutbound?symbol={sym}"},
]

# Skip re-fetching a feed URL if the store already holds an entry fetched
# more recently than this. collect() runs on every tracked-symbol change
# (add/remove/reset in the frontend ticker picker) and each fetch is a live,
# serial HTTP request per feed — without this, narrowing the symbol list
# (e.g. just removing one ticker) still re-fetched every remaining feed over
# the network for no reason, since dropping a symbol can't require fresher
# data for the symbols left over.
FEED_STALE_MINUTES = 5

# feedparser.parse(url) has no built-in timeout — a single slow/hung server
# blocks this whole (serial) collection run indefinitely, worse now that
# per-symbol collection means 3 feeds per ticker instead of 1. Fetch with a
# bounded request instead and hand feedparser the bytes. Some of the newer
# sources (Nasdaq, Seeking Alpha) also 403 a bare default User-Agent.
FEED_TIMEOUT_SECONDS = 10
FEED_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# collect_feeds() fetches feeds concurrently (I/O-bound — most of the wall
# clock is waiting on remote servers, not CPU) then merges results into the
# store single-threaded. 8 caps how many requests are in flight at once —
# high enough to matter (a 10-symbol portfolio now means 33 feeds, 3+ per
# host) without opening so many simultaneous connections to one host (e.g.
# Yahoo, hit by every symbol) that it reads as abuse.
FEED_FETCH_WORKERS = 8


def build_feeds(symbols: List[str]) -> List[Dict]:
    """General feeds + one feed per (portfolio symbol x per-ticker template),
    e.g. AAPL gets a Yahoo Finance, a Seeking Alpha, and a Nasdaq feed."""
    feeds = list(GENERAL_FEEDS)
    for sym in symbols:
        for template in TICKER_FEED_TEMPLATES:
            feeds.append({
                "name": f"{template['name']}: {sym}",
                "url": template["url"].format(sym=sym),
                "symbols": [sym],
            })
    return feeds


# ------------------------------------------------------------------ normalization

def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _record_id(url: str, title: str) -> str:
    key = (url or "").split("?")[0].rstrip("/").lower() or _clean(title).lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _entry_to_record(entry, feed: Dict, fetched_utc: str) -> Optional[Dict]:
    title = _clean(getattr(entry, "title", ""))
    if not title:
        return None
    url = getattr(entry, "link", "") or ""

    published = None
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            published = datetime.fromtimestamp(mktime(t), tz=timezone.utc).isoformat()
            break

    publisher = ""
    src = getattr(entry, "source", None)
    if src is not None:
        publisher = _clean(getattr(src, "title", "") or "")

    return {
        "id": _record_id(url, title),
        "title": title,
        "summary": _clean(getattr(entry, "summary", "")),
        "url": url,
        "source": feed["name"],
        "publisher": publisher,
        "feed_url": feed["url"],
        "published_utc": published,
        "fetched_utc": fetched_utc,
        "symbols": list(feed["symbols"]),
    }


# ------------------------------------------------------------------ store

def load_store(path: Path = STORE) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["id"]: r for r in json.load(f)}


def save_store(records: Dict[str, Dict], path: Path = STORE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records.values(), key=lambda r: r["published_utc"] or "")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)


# ------------------------------------------------------------------ collection

def _fetch_feed(feed: Dict) -> Tuple[Optional[List], bool]:
    """Fetch + parse one feed. Returns (entries, ok); ok=False on any
    network/parse failure or an empty result. Pure function of `feed` only —
    no access to `records`/`stats` — so it's safe to run from a worker
    thread; collect_feeds() does all the shared-state mutation itself,
    single-threaded, after every fetch has finished."""
    try:
        response = requests.get(
            feed["url"],
            timeout=FEED_TIMEOUT_SECONDS,
            headers={"User-Agent": FEED_USER_AGENT},
        )
        response.raise_for_status()
        entries = feedparser.parse(response.content).entries
    except Exception:
        return None, False
    if not entries:
        return None, False
    return entries, True


def collect(symbols: List[str], store_path: Path = STORE) -> Dict[str, int]:
    """Fetch all feeds (general + one per portfolio symbol), merge into the
    JSONL store, return run statistics."""
    return collect_feeds(build_feeds(symbols), store_path)


def collect_urls(urls: List[str], store_path: Path = STORE) -> Dict[str, int]:
    """Fetch arbitrary feed URLs (no ticker tagging) into the same store —
    used by fetch_headlines() when called with an explicit feed list."""
    feeds = [{"name": url, "url": url, "symbols": []} for url in urls]
    return collect_feeds(feeds, store_path)


def collect_feeds(feeds: List[Dict], store_path: Path = STORE) -> Dict[str, int]:
    """Fetch a specific list of {name, url, symbols} feed dicts, merge into
    the JSONL store, return run statistics. `collect()`/`collect_urls()` are
    thin wrappers over this that build the feed list differently.

    A feed fetched within the last FEED_STALE_MINUTES is skipped rather than
    re-parsed — the check is a lookup against records already loaded from
    the store, so it costs nothing like a network round trip does.

    Fetching itself runs on a small thread pool (network wait dominates, so
    this is a real wall-clock win); merging results into `records`/`stats`
    happens afterward on the main thread only, in original feed order, so
    the two shared dicts are never written from more than one thread and a
    run's outcome doesn't depend on which request happened to finish first.
    """
    now = datetime.now(timezone.utc)
    fetched_utc = now.isoformat()
    records = load_store(store_path)

    last_fetch_by_feed: Dict[str, datetime] = {}
    for r in records.values():
        ts, feed_url = r.get("fetched_utc"), r.get("feed_url")
        if not ts or not feed_url:
            continue
        t = datetime.fromisoformat(ts)
        if feed_url not in last_fetch_by_feed or t > last_fetch_by_feed[feed_url]:
            last_fetch_by_feed[feed_url] = t

    stats = {"feeds_ok": 0, "feeds_failed": 0, "feeds_skipped_fresh": 0,
             "entries_seen": 0, "new": 0, "updated_tags": 0, "total_in_store": 0}

    to_fetch = []
    for feed in feeds:
        last = last_fetch_by_feed.get(feed["url"])
        if last is not None and now - last < timedelta(minutes=FEED_STALE_MINUTES):
            stats["feeds_skipped_fresh"] += 1
            continue
        to_fetch.append(feed)

    results: Dict[int, Tuple[Optional[List], bool]] = {}
    if to_fetch:
        with ThreadPoolExecutor(max_workers=min(FEED_FETCH_WORKERS, len(to_fetch))) as pool:
            future_to_index = {pool.submit(_fetch_feed, feed): i for i, feed in enumerate(to_fetch)}
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()

    for i, feed in enumerate(to_fetch):
        entries, ok = results[i]
        if not ok:
            stats["feeds_failed"] += 1
            continue
        stats["feeds_ok"] += 1

        for entry in entries:
            rec = _entry_to_record(entry, feed, fetched_utc)
            if rec is None:
                continue
            stats["entries_seen"] += 1
            existing = records.get(rec["id"])
            if existing is None:
                records[rec["id"]] = rec
                stats["new"] += 1
            else:
                # same story from another feed — union the ticker tags
                merged = sorted(set(existing["symbols"]) | set(rec["symbols"]))
                if merged != existing["symbols"]:
                    existing["symbols"] = merged
                    stats["updated_tags"] += 1

    save_store(records, store_path)
    stats["total_in_store"] = len(records)
    return stats


def default_symbols() -> List[str]:
    """Symbols this collector runs for when none are given: the saved
    portfolio, falling back to the sample one while it's still empty.
    Public because `POST /news/collect` runs the same default as the
    command line does."""
    from src import portfolio as pf

    holdings = pf.load_portfolio()
    if holdings.empty:
        holdings = pf.load_portfolio(pf.SAMPLE_PORTFOLIO_CSV)
    return sorted(set(holdings["symbol"]))


if __name__ == "__main__":
    symbols = [s.upper() for s in sys.argv[1:]] or default_symbols()
    print(f"Collecting for symbols: {', '.join(symbols)}")
    result = collect(symbols)
    print(json.dumps(result, indent=2))
    print(f"Store: {STORE}")
