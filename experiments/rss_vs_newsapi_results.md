# Experiment: RSS vs NewsAPI as AURORA's news source

Run at 2026-07-17 02:44 UTC — rerun with `python experiments/rss_vs_newsapi.py` (needs NEWSAPI_KEY env var).

| Metric | RSS (5 feeds) | NewsAPI (3 calls) |
|---|---|---|
| Articles (deduped) | 98 | 234 |
| Distinct outlets | 5 | 106 |
| Have timestamp | 100% | 100% |
| Have description | 41% | 100% |
| Newest article age | 8.2 h | 24.0 h |
| Median article age | 18.1 h | 34.7 h |
| Oldest article age | 46.8 h | 28.2 d |
| Published in last 24 h | 56 | 0 |
| Mention a holding | 7 | 42 |
| General market news | 24 | 65 |

**Overlap**: 0 of 98 RSS stories also appear in the NewsAPI result set (title similarity ≥ 0.75).

**Per-holding mentions** — RSS: {'MSFT': 2, 'XLV': 1, 'SLV': 1, 'GLD': 2, 'AAPL': 1} · NewsAPI: {'SPY': 3, 'AAPL': 15, 'SLV': 7, 'XLV': 4, 'GLD': 8, 'MSFT': 5}

## Plan limits that the numbers can't show

- **NewsAPI free tier**: 100 requests/day, ~24 h delay on articles, and a
  hard ~1-month lookback — the `everything` endpoint rejects `from` dates
  older than a month. Commercial tiers remove these but start at $449/mo.
- **RSS**: free, no rate limit worth worrying about, near-real-time — but
  each feed only exposes its current window (typically the last few hours
  to days). There is **no archive at all**.

## Interpretation for AURORA

- **For ML training**: NEITHER source works — RSS has zero history and
  NewsAPI free caps at ~1 month, far short of the years needed to train.
  The historical `sentiment_features` table must come from an archival
  corpus (FNSPID / Kaggle), as planned. This experiment settles the
  question: the live-source choice is about the Essential News page, not
  about training data.
- **For the live feed**: compare the freshness rows (NewsAPI free delays
  ~24 h; RSS is minutes-fresh), relevance counts, and outlet diversity
  above, and weigh NewsAPI's structured JSON + keyword search against
  RSS's freshness and zero cost.