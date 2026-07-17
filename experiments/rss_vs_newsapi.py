"""Experiment: RSS feeds vs NewsAPI.org as AURORA's news source.

Compares the two on the dimensions that matter for our use cases:
  LIVE feed  (Essential News page): volume, freshness, portfolio relevance,
             metadata quality, outlet diversity
  TRAINING   (sentiment_features):  how far back each source reaches

Run:
    $env:NEWSAPI_KEY = "<your key>"        (PowerShell)
    python experiments/rss_vs_newsapi.py

Writes a markdown report to experiments/rss_vs_newsapi_results.md.
The API key is read from the environment on purpose — never commit it.
"""

import difflib
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import mktime
from typing import Dict, List, Optional

import feedparser
import requests

ROOT = Path(__file__).resolve().parent.parent
REPORT = Path(__file__).resolve().parent / "rss_vs_newsapi_results.md"

RSS_FEEDS = {
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Investing.com": "https://www.investing.com/rss/news.rss",
    "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
}

# Keywords per holding in the sample portfolio — used for relevance scoring
PORTFOLIO_KEYWORDS = {
    "AAPL": ["aapl", "apple"],
    "MSFT": ["msft", "microsoft"],
    "SPY": ["s&p 500", "s&p500", "sp500"],
    "XLV": ["healthcare", "health care", "pharma"],
    "GLD": ["gold"],
    "SLV": ["silver"],
}
MARKET_KEYWORDS = ["fed", "inflation", "interest rate", "stocks", "earnings",
                   "market", "tariff", "treasury", "nasdaq", "dow"]

NEWSAPI_QUERY = (
    '(Apple OR Microsoft OR "S&P 500" OR gold OR silver OR healthcare) '
    "OR (stocks AND market) OR Fed OR inflation OR earnings"
)


def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (t or "").lower())).strip()


def fetch_rss() -> List[Dict]:
    articles = []
    for name, url in RSS_FEEDS.items():
        try:
            parsed = feedparser.parse(url)
        except Exception as exc:
            print(f"  RSS {name}: FAILED ({exc})")
            continue
        for e in parsed.entries:
            published = None
            for attr in ("published_parsed", "updated_parsed"):
                t = getattr(e, attr, None)
                if t:
                    published = datetime.fromtimestamp(mktime(t), tz=timezone.utc)
                    break
            articles.append({
                "title": getattr(e, "title", ""),
                "description": re.sub(r"<[^>]+>", "", getattr(e, "summary", "") or ""),
                "source": name,
                "published": published,
                "url": getattr(e, "link", ""),
            })
        print(f"  RSS {name}: {len(parsed.entries)} entries")
    return articles


def fetch_newsapi(key: str) -> List[Dict]:
    articles = []

    def call(endpoint: str, label: str, **params) -> None:
        params.update({"apiKey": key, "pageSize": 100, "language": "en"})
        try:
            r = requests.get(f"https://newsapi.org/v2/{endpoint}", params=params, timeout=30)
            data = r.json()
        except Exception as exc:
            print(f"  NewsAPI {label}: FAILED ({exc})")
            return
        if data.get("status") != "ok":
            print(f"  NewsAPI {label}: API error — {data.get('code')}: {data.get('message')}")
            return
        for a in data.get("articles", []):
            published = None
            if a.get("publishedAt"):
                try:
                    published = datetime.fromisoformat(
                        a["publishedAt"].replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
            articles.append({
                "title": a.get("title") or "",
                "description": a.get("description") or "",
                "source": (a.get("source") or {}).get("name") or "unknown",
                "published": published,
                "url": a.get("url") or "",
            })
        print(f"  NewsAPI {label}: {len(data.get('articles', []))} returned "
              f"(totalResults={data.get('totalResults')})")

    # 3 calls, well inside the free tier's 100/day
    call("top-headlines", "top-headlines business", category="business", country="us")
    month_ago = (datetime.now(timezone.utc) - timedelta(days=29)).strftime("%Y-%m-%d")
    call("everything", "everything (finance query, 29d)", q=NEWSAPI_QUERY,
         sortBy="publishedAt", **{"from": month_ago})
    call("everything", "everything (oldest reachable)", q=NEWSAPI_QUERY,
         sortBy="publishedAt", **{"from": month_ago, "to": month_ago})
    return articles


def dedupe(articles: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for a in articles:
        k = _norm_title(a["title"])
        if k and k not in seen:
            seen.add(k)
            out.append(a)
    return out


def relevance(article: Dict) -> List[str]:
    text = f"{article['title']} {article['description']}".lower()
    hits = [sym for sym, kws in PORTFOLIO_KEYWORDS.items() if any(k in text for k in kws)]
    if not hits and any(k in text for k in MARKET_KEYWORDS):
        hits = ["MARKET"]
    return hits


def metrics(name: str, articles: List[Dict]) -> Dict:
    now = datetime.now(timezone.utc)
    dated = [a for a in articles if a["published"]]
    ages_h = [(now - a["published"]).total_seconds() / 3600 for a in dated]
    tagged = [(a, relevance(a)) for a in articles]
    holding_hits = [a for a, r in tagged if r and r != ["MARKET"]]
    market_hits = [a for a, r in tagged if r == ["MARKET"]]
    per_symbol = Counter(sym for _, r in tagged for sym in r if sym != "MARKET")
    return {
        "name": name,
        "total": len(articles),
        "outlets": len({a["source"] for a in articles}),
        "with_timestamp_pct": 100 * len(dated) / len(articles) if articles else 0,
        "with_description_pct": 100 * sum(1 for a in articles if len(a["description"]) > 20)
        / len(articles) if articles else 0,
        "newest_h": min(ages_h) if ages_h else None,
        "median_age_h": sorted(ages_h)[len(ages_h) // 2] if ages_h else None,
        "oldest_h": max(ages_h) if ages_h else None,
        "last24h": sum(1 for h in ages_h if h <= 24),
        "holding_relevant": len(holding_hits),
        "market_relevant": len(market_hits),
        "per_symbol": dict(per_symbol),
    }


def overlap(a: List[Dict], b: List[Dict], threshold: float = 0.75) -> int:
    bt = [_norm_title(x["title"]) for x in b]
    n = 0
    for x in a:
        xt = _norm_title(x["title"])
        if any(difflib.SequenceMatcher(None, xt, y).ratio() >= threshold for y in bt):
            n += 1
    return n


def fmt_h(h: Optional[float]) -> str:
    if h is None:
        return "—"
    return f"{h:.1f} h" if h < 48 else f"{h / 24:.1f} d"


def main() -> None:
    key = os.environ.get("NEWSAPI_KEY")
    if not key:
        sys.exit("Set NEWSAPI_KEY in the environment first.")

    print("Fetching RSS…")
    rss = dedupe(fetch_rss())
    print("Fetching NewsAPI…")
    napi = dedupe(fetch_newsapi(key))

    m_rss, m_napi = metrics("RSS (5 feeds)", rss), metrics("NewsAPI (3 calls)", napi)
    cross = overlap(rss, napi)

    rows = [
        ("Articles (deduped)", "total", "{}"),
        ("Distinct outlets", "outlets", "{}"),
        ("Have timestamp", "with_timestamp_pct", "{:.0f}%"),
        ("Have description", "with_description_pct", "{:.0f}%"),
        ("Newest article age", "newest_h", None),
        ("Median article age", "median_age_h", None),
        ("Oldest article age", "oldest_h", None),
        ("Published in last 24 h", "last24h", "{}"),
        ("Mention a holding", "holding_relevant", "{}"),
        ("General market news", "market_relevant", "{}"),
    ]
    lines = [
        "# Experiment: RSS vs NewsAPI as AURORA's news source",
        "",
        f"Run at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} — "
        "rerun with `python experiments/rss_vs_newsapi.py` (needs NEWSAPI_KEY env var).",
        "",
        "| Metric | RSS (5 feeds) | NewsAPI (3 calls) |",
        "|---|---|---|",
    ]
    for label, field, fmt in rows:
        va, vb = m_rss[field], m_napi[field]
        if fmt is None:
            va, vb = fmt_h(va), fmt_h(vb)
        else:
            va, vb = fmt.format(va), fmt.format(vb)
        lines.append(f"| {label} | {va} | {vb} |")
    lines += [
        "",
        f"**Overlap**: {cross} of {m_rss['total']} RSS stories also appear in the "
        "NewsAPI result set (title similarity ≥ 0.75).",
        "",
        f"**Per-holding mentions** — RSS: {m_rss['per_symbol']} · NewsAPI: {m_napi['per_symbol']}",
        "",
        "## Plan limits that the numbers can't show",
        "",
        "- **NewsAPI free tier**: 100 requests/day, ~24 h delay on articles, and a",
        "  hard ~1-month lookback — the `everything` endpoint rejects `from` dates",
        "  older than a month. Commercial tiers remove these but start at $449/mo.",
        "- **RSS**: free, no rate limit worth worrying about, near-real-time — but",
        "  each feed only exposes its current window (typically the last few hours",
        "  to days). There is **no archive at all**.",
        "",
        "## Interpretation for AURORA",
        "",
        "- **For ML training**: NEITHER source works — RSS has zero history and",
        "  NewsAPI free caps at ~1 month, far short of the years needed to train.",
        "  The historical `sentiment_features` table must come from an archival",
        "  corpus (FNSPID / Kaggle), as planned. This experiment settles the",
        "  question: the live-source choice is about the Essential News page, not",
        "  about training data.",
        "- **For the live feed**: compare the freshness rows (NewsAPI free delays",
        "  ~24 h; RSS is minutes-fresh), relevance counts, and outlet diversity",
        "  above, and weigh NewsAPI's structured JSON + keyword search against",
        "  RSS's freshness and zero cost.",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "\n".join(lines[4:len(rows) + 6]))
    print(f"\nOverlap: {cross} shared stories. Report written to {REPORT}")


if __name__ == "__main__":
    main()
