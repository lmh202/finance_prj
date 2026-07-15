# Developer 3 — Event Intelligence Engine (News & AI)

**Mission (Architecture.md §6):** answer *"What happened, which holdings may be
affected?"* — collect news via RSS, filter to at most ~5 essential events per
day, classify + sentiment-score them, and map each event to affected holdings.

## Your contract (frozen — see `src/interfaces.py`)

```python
fetch_headlines(feeds=None, limit=50) -> List[NewsEvent]
essential_news(holding_symbols, max_events=5) -> List[NewsEvent]
```

Consumed by: your News page, and Developer 4's `reaction_risk` / `recommend_event`
(the "Should I React?" page picks from YOUR events).

## What you get for free (shared kernel — read-only)

- `src.portfolio.load_portfolio()` → holdings (so you know which tickers matter)
- `src.interfaces.EVENT_CATEGORIES` → the fixed event taxonomy
- `src.data_loader.load_ticker_universe()` → symbol ↔ company-name mapping
  (useful for matching company names in headlines to tickers)

## Files you own (edit ONLY inside this folder)

- `engine.py` — currently returns `[]`. Implement the pipeline:
  1. Fetch feeds with `feedparser` (announce + add to requirements.txt)
  2. Dedupe near-identical headlines
  3. Classify into `EVENT_CATEGORIES` (keyword rules first; FinBERT later if time)
  4. Sentiment −1…+1 (VADER or keyword rules are fine for MVP)
  5. Importance 0–100: credibility + corroboration + portfolio relevance + severity + recency
  6. Map to `affected_symbols` via company-name/ticker matching
- `page.py` — the "Essential News" page; the event-card rendering is already
  sketched and lights up as soon as `essential_news` returns data.

## Definition of done (MVP)

- [ ] ≥2 feeds fetched and parsed reliably, with caching so reruns don't hammer feeds
- [ ] Duplicate stories collapsed (same event from 2 sources = 1 event, higher corroboration)
- [ ] Every NewsEvent field populated; never more than `max_events` returned
- [ ] At least one held ticker correctly mapped on a real news day
- [ ] Handles gracefully: feed down, malformed entries, zero relevant news

## Rules

1. Commit only inside `src/news_intelligence/`. Never edit `src/interfaces.py`,
   the shared kernel, other engines, or `app/` — propose changes in the group instead.
2. Collaborate only through `src/interfaces.py` types and other engines' public functions.
3. New pip dependency (feedparser, vaderSentiment, …)? Announce it, then add to
   `requirements.txt` (the one allowed outside-folder edit).
