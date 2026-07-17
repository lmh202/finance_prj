# Developer 3 — Event Intelligence Engine (News, LLM & NLP)

**Mission (Architecture.md §6 + ML feature channel):** two deliverables —
(A) the **live** pipeline: RSS → dedupe → LLM classification → at most ~5
essential events per day mapped to holdings; (B) the **historical** pipeline:
a news-sentiment feature table for Developer 2's ML ablation.

## Your contract (frozen — see `src/interfaces.py`)

```python
fetch_headlines(feeds=None, limit=50) -> List[NewsEvent]           # live
essential_news(holding_symbols, max_events=5) -> List[NewsEvent]   # live
sentiment_features(symbols, start=None, end=None) -> pd.DataFrame  # historical
```

Consumed by: your News page, Developer 4 ("Should I React?" picks from YOUR
events), and Developer 2 (`sentiment_features` as the optional ML input).

## A. Live pipeline (LLM API — you are the only developer who calls it)

1. Fetch feeds with `feedparser`; dedupe by URL/title similarity — no LLM here.
2. **One batched call per refresh cycle**: send all deduped headlines + the
   user's holdings in a single request; use the Anthropic SDK's structured
   outputs (`client.messages.parse` + a Pydantic schema) so every field of
   `NewsEvent` comes back validated. Default model `claude-opus-4-8`; swapping
   the string to `claude-haiku-4-5` is the budget option — team's call.
3. Cache by headline hash in `data/news_cache.json` + `@st.cache_data(ttl=900)`
   so Streamlit reruns never re-bill.
4. Key handling: `ANTHROPIC_API_KEY` env var or `.streamlit/secrets.toml`
   (gitignored) — never committed. If the key is missing or the call fails,
   fall back to keyword-rule classification so the demo never breaks.

## B. Historical pipeline (feeds the ML)

1. Source a timestamped, ticker-mapped corpus (FNSPID, Kaggle financial news, …).
2. Score it with a LOCAL model (FinBERT or VADER) — NOT the LLM API
   (millions of headlines through an API = real money for no extra grade).
3. Aggregate per symbol-day (importance-weighted mean sentiment, count) into
   the `sentiment_features` schema. **No look-ahead**: rows dated t use only
   news published before day t's market open.
4. Cache the finished table to `data/processed/` — built once, read many times.

## Files you own (edit ONLY inside this folder)

- `engine.py` — all three contract functions (currently stubs returning empty).
- `page.py` — the Essential News page (card rendering already sketched).

## Definition of done

- [ ] ≥2 live feeds parsed reliably, deduped, cached
- [ ] Every `NewsEvent` field populated via structured-output LLM call; ≤ `max_events` returned
- [ ] Keyword fallback path works with no API key
- [ ] Historical feature table built, look-ahead-safe, cached, schema-exact
- [ ] At least one held ticker correctly mapped on a real news day
- [ ] Handles gracefully: feed down, malformed entries, zero relevant news

## Rules

1. Commit only inside `src/news_intelligence/`. Never edit `src/interfaces.py`,
   the shared kernel, other engines, or `app/` — propose changes in the group instead.
2. Collaborate only through `src/interfaces.py` types and other engines' public functions.
3. New pip dependency (`feedparser`, `anthropic`, `vaderSentiment`/FinBERT stack)?
   Announce it, then add to `requirements.txt` (the one allowed outside-folder edit).
