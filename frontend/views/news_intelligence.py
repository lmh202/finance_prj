"""Essential News page — Developer 3 owns everything rendered here."""

import streamlit as st

import api_client as api
from views._common import call


def render() -> None:
    st.set_page_config(page_title="AURORA — Essential News", page_icon="📰", layout="wide")
    st.title("📰 Essential News")
    st.caption("Engine 3 — Event Intelligence (Developer 3)")

    events = call(api.essential_news, 5)
    if events is None:
        return

    if not events:
        st.info(
            "The news engine is not implemented yet — this page comes alive "
            "when Developer 3 ships `backend/src/news_intelligence/engine.py`. "
            "Planned feeds:"
        )
        for feed in call(api.news_feeds) or []:
            st.markdown(f"- `{feed}`")
        return

    for ev in events:
        with st.container(border=True):
            st.markdown(
                f"**{ev['title']}**  \n{ev['source']} · {ev['category']} · "
                f"importance {ev['importance']:.0f}/100"
            )
            if ev["summary"]:
                st.write(ev["summary"])
            if ev["affected_symbols"]:
                st.caption("Affected holdings: " + ", ".join(ev["affected_symbols"]))
