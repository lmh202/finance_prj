"""Essential News page — Developer 3 owns everything rendered here.

The tracked ticker list defaults to the saved portfolio (via /news/essential,
which resolves it server-side from data/portfolio.csv when no explicit list
is sent) but is editable in the page body: remove a holding you don't care
about, or add any symbol you don't hold just to watch its news. The sidebar
is the page switcher for the whole app, so this — and the other controls —
live in the page body instead.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import streamlit as st

import api_client as api
from theme import apply_theme
from views._common import call, portfolio_key

# Fetch a generous candidate batch once per cache window; the sliders below
# then filter/slice it client-side with no extra backend calls.
_BATCH_SIZE = 20
_WINDOW_OPTIONS = [6, 12, 24, 48]


@st.cache_data(ttl=900, show_spinner="Collecting and scoring headlines…")
def _essential(pkey: str, batch_size: int, symbols_key: str) -> List[dict]:
    symbols = symbols_key.split(",") if symbols_key else []
    return api.essential_news(batch_size, symbols=symbols)


@st.cache_data(ttl=86400, show_spinner="Loading US stock universe…")
def _universe() -> Tuple[List[str], Dict[str, str]]:
    # Same source + label format as the Home page's "Add a holding" search
    # (frontend/app.py) so ticker autocomplete looks/behaves consistently.
    uni = api.get_universe()
    labels = uni["symbol"] + " — " + uni["name"] + "  (" + uni["exchange"] + ")"
    return uni["symbol"].tolist(), dict(zip(uni["symbol"], labels))


def _sentiment_label(score: float) -> str:
    # Mirrors the +-0.15 neutral band analyzer.py uses server-side.
    if score >= 0.15:
        return "🟢 Positive"
    if score <= -0.15:
        return "🔴 Negative"
    return "⚪ Neutral"


def _age_hours(published: Optional[str], now: datetime) -> Optional[float]:
    if not published:
        return None
    return (now - datetime.fromisoformat(published)).total_seconds() / 3600


def render() -> None:
    st.set_page_config(page_title="AURORA — Essential News", page_icon="📰", layout="wide")
    apply_theme()
    st.title("📰 Essential News")
    st.caption(
        "Engine 3 — Event Intelligence (Developer 3). Ranked by source credibility, "
        "corroboration, relevance to your holdings, severity, recency and expected "
        "market impact. Informational only — not investment advice."
    )

    pkey = call(portfolio_key)
    if pkey is None:
        return

    portfolio = call(api.get_portfolio)
    if portfolio is None:
        return
    portfolio_symbols = sorted(set(portfolio["symbol"])) if not portfolio.empty else []

    universe = call(_universe)
    if universe is None:
        return
    universe_symbols, universe_labels = universe

    if "news_symbols" not in st.session_state:
        st.session_state.news_symbols = list(portfolio_symbols)

    st.markdown("**Tracking news for**")

    # Two separate widgets on purpose: st.multiselect's format_func decorates
    # BOTH the dropdown options AND the selected chips, so one widget can't
    # show full company names in the picker while keeping chips as plain
    # tickers. Split instead: a plain-ticker "currently tracking" row you
    # remove from, and a separate name-searchable form you add through.
    current = st.session_state.news_symbols
    chips_col, reset_col = st.columns([5, 1])
    with chips_col:
        # Key is derived from the content itself so that whenever `current`
        # changes — via the reset button below or the add-form further down —
        # this renders as a fresh widget seeded with the new value, instead
        # of Streamlit replaying whatever was last clicked under an old key.
        chip_key = "news_chips::" + "|".join(sorted(current))
        kept = st.multiselect(
            "Currently tracking", options=current, default=current,
            key=chip_key, label_visibility="collapsed",
            placeholder="No tickers selected — showing general market news only.",
        )
    with reset_col:
        if st.button("↺ Reset to portfolio", width="stretch"):
            st.session_state.news_symbols = list(portfolio_symbols)
            st.rerun()
    if sorted(kept) != sorted(current):
        st.session_state.news_symbols = kept
        st.rerun()

    with st.form("news_add_ticker", clear_on_submit=True):
        add_col, submit_col = st.columns([5, 1])
        with add_col:
            # Not-yet-added portfolio tickers surface first (quick re-add);
            # the rest of the full NASDAQ/NYSE/etc. universe follows so any
            # other symbol can be found by ticker or company name too.
            not_yet_added = sorted(set(portfolio_symbols) - set(current))
            rest = sorted(set(universe_symbols) - set(current) - set(not_yet_added))
            pick = st.selectbox(
                "Add a ticker", options=not_yet_added + rest, index=None,
                format_func=lambda s: universe_labels.get(s, s),
                placeholder="Search a ticker or company name (e.g. Microsoft) to add…",
                label_visibility="collapsed",
            )
        with submit_col:
            add_clicked = st.form_submit_button("Add", width="stretch")
    if add_clicked and pick:
        symbol = pick.strip().upper()
        if symbol not in st.session_state.news_symbols:
            st.session_state.news_symbols = st.session_state.news_symbols + [symbol]
            st.rerun()

    active_symbols = sorted({s.strip().upper() for s in st.session_state.news_symbols if s and s.strip()})

    window, min_importance, show_count = st.columns(3)
    with window:
        lookback_hours = st.select_slider(
            "Lookback window", options=_WINDOW_OPTIONS, value=24,
            format_func=lambda h: f"last {h}h",
        )
    with min_importance:
        min_score = st.slider("Minimum importance", 0, 100, 40)
    with show_count:
        limit = st.slider("Number of stories", 1, _BATCH_SIZE, 5)

    events = call(_essential, pkey, _BATCH_SIZE, ",".join(active_symbols))
    if events is None:
        return

    now = datetime.now(timezone.utc)
    candidates = []
    for event in events:
        age = _age_hours(event["published"], now)
        if age is not None and age > lookback_hours:
            continue
        if event["importance"] < min_score:
            continue
        candidates.append(event)
    candidates = candidates[:limit]

    if not candidates:
        st.info("No stories met the current lookback window and importance threshold.")
        return

    st.subheader(f"Top {len(candidates)} essential event(s)")
    for event in candidates:
        with st.container(border=True):
            title_col, score_col = st.columns([4, 2])
            with title_col:
                st.markdown(f"### [{event['title']}]({event['url']})")
                published = event["published"]
                when = f" · {published[:16].replace('T', ' ')} UTC" if published else ""
                st.caption(f"{event['source']}{when}")
            with score_col:
                st.caption("Importance")
                st.markdown(f"**{event['importance']:.0f}/100**")

            if event["summary"]:
                st.write(event["summary"])

            # st.metric's value font is large and fixed-size — fine for a short
            # number like Importance above, but it truncates ("Geopolitical",
            # "Negative", a symbol list) once squeezed into a 1/3-width column.
            # Plain markdown wraps instead of clipping, so use that here.
            cat_col, sent_col, holdings_col = st.columns(3)
            symbols = event["affected_symbols"]
            with cat_col:
                st.caption("Category")
                st.markdown(f"**{event['category'].replace('_', ' ').title()}**")
            with sent_col:
                st.caption("Sentiment")
                st.markdown(f"**{_sentiment_label(event['sentiment'])}** ({event['sentiment']:+.2f})")
            with holdings_col:
                st.caption("Affected holdings")
                st.markdown(f"**{', '.join(symbols) if symbols else 'None identified'}**")

            impact = event.get("impact", {})
            if symbols:
                st.caption(
                    "Potential impact: "
                    + " · ".join(f"{sym} ({impact[sym]})" if impact.get(sym) else sym for sym in symbols)
                )

    st.caption("Headlines remain attributed to their original publishers — follow the source link for full context.")
