"""Essential News page — Developer 3 owns everything rendered here."""

import streamlit as st

from src import portfolio as pf
from src.news_intelligence import engine


def render() -> None:
    st.set_page_config(page_title="AURORA — Essential News", page_icon="📰", layout="wide")
    st.title("📰 Essential News")
    st.caption("Engine 3 — Event Intelligence (Developer 3)")

    holdings = pf.load_portfolio()
    events = engine.essential_news(list(holdings["symbol"]), max_events=5)

    if not events:
        st.info(
            "The news engine is not implemented yet — this page comes alive "
            "when Developer 3 ships `src/news_intelligence/engine.py`. "
            "Planned feeds:"
        )
        for feed in engine.DEFAULT_FEEDS:
            st.markdown(f"- `{feed}`")
        return

    for ev in events:
        with st.container(border=True):
            st.markdown(f"**{ev.title}**  \n{ev.source} · {ev.category} · importance {ev.importance:.0f}/100")
            if ev.summary:
                st.write(ev.summary)
            if ev.affected_symbols:
                st.caption("Affected holdings: " + ", ".join(ev.affected_symbols))
