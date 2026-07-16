"""'Should I React?' page — Developer 4 owns everything rendered here."""

from datetime import datetime

import pandas as pd
import streamlit as st

from src import data_loader
from src import portfolio as pf
from src.daily_strategy import engine as strategy
from src.interfaces import BENCHMARK, NewsEvent
from src.news_intelligence import engine as news
from src.portfolio_health import engine as health
from src.recommendation import engine

# Exercises the reaction-risk UI until Developer 3 ships real events.
# Clearly labeled as a demo — remove once essential_news returns data.
DEMO_EVENT = NewsEvent(
    title="[DEMO] Unexpected interest-rate increase announced",
    source="demo — not a real headline",
    url="",
    published=None,
    category="interest_rates",
    sentiment=-0.6,
    importance=85.0,
    affected_symbols=["AAPL", "MSFT"],
    impact={"AAPL": "negative", "MSFT": "negative", "GLD": "mixed"},
    summary="Placeholder event so the reaction-risk flow can be demoed "
    "before the news engine is implemented.",
)

SUGGESTION_LABELS = {
    "do_nothing": "🟥 Wait — do nothing for now",
    "moderate": "🟨 Moderate adjustment may be considered",
    "aggressive": "🟩 Acting now carries relatively low risk",
}


@st.cache_data(ttl=3600, show_spinner="Loading price history…")
def _history(symbols: tuple) -> pd.DataFrame:
    return data_loader.get_history(list(symbols))


def render() -> None:
    st.set_page_config(page_title="AURORA — Should I React?", page_icon="🤔", layout="wide")
    st.title("🤔 Should I React?")
    st.caption("Engine 4 — Reaction Risk & Recommendation (Developer 4)")

    holdings = pf.load_portfolio()
    if holdings.empty:
        st.info("Your portfolio is empty — build it on the Home page first.")
        return

    history = _history(tuple(sorted(set(holdings["symbol"]))) + (BENCHMARK,))
    if history.empty:
        st.error("Could not load price history — check your connection and retry.")
        return

    prices = data_loader.get_latest_prices(list(holdings["symbol"]))
    view, totals = pf.build_view(holdings, prices, pf.load_cash())
    weights = dict(zip(view["symbol"], view["weight_pct"]))
    regime = strategy.classify_regime(history)
    signals = strategy.score_assets(history, holdings)

    # ---------------------------------------------------- daily recommendation
    st.subheader("Today's normal-day recommendation")
    daily = engine.recommend_daily(regime, signals, weights)
    st.write(daily.explanation)
    if daily.trades:
        st.dataframe(
            pd.DataFrame(
                [{"Ticker": t.symbol, "Weight change": f"{t.weight_change_pct:+.1f}%",
                  "Reason": t.reason} for t in daily.trades]
            ),
            hide_index=True,
            width="stretch",
        )
        base = health.compute_health(holdings, history)
        after = health.what_if_health(holdings, history, daily.trades)
        st.metric(
            "Portfolio health if applied",
            f"{after.score:.0f}/100",
            delta=f"{after.score - base.score:+.1f} vs current {base.score:.0f}",
        )

    # ---------------------------------------------------------- event reaction
    st.subheader("React to an event")
    events = news.essential_news(list(holdings["symbol"]), max_events=5)
    if not events:
        st.caption(
            "No real events yet (news engine pending — Developer 3). "
            "Showing a demo event so the flow is testable."
        )
        events = [DEMO_EVENT]

    picked = st.selectbox(
        "Event", options=range(len(events)), format_func=lambda i: events[i].title
    )
    event = events[picked]

    risk = engine.reaction_risk(event, weights, regime)
    rec = engine.recommend_event(event, risk)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Risk of reacting", f"{risk.risk_pct:.0f}%")
        st.markdown(SUGGESTION_LABELS.get(risk.suggestion, risk.suggestion))
    with c2:
        st.markdown("**Why:**")
        for r in risk.reasons:
            st.markdown(f"- {r}")

    st.markdown("**Factor breakdown** (0 = safe to act, 1 = risky):")
    st.dataframe(
        pd.Series(risk.factors, name="value").round(2),
        width="stretch",
    )

    st.markdown(f"**Recommendation:** {rec.explanation}")
    if rec.trades:
        st.dataframe(
            pd.DataFrame(
                [{"Ticker": t.symbol, "Weight change": f"{t.weight_change_pct:+.1f}%",
                  "Reason": t.reason} for t in rec.trades]
            ),
            hide_index=True,
            width="stretch",
        )
    st.caption("AURORA never executes trades — the final decision is always yours.")
