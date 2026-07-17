"""'Should I React?' page — Developer 4 owns everything rendered here."""

import pandas as pd
import streamlit as st

import api_client as api
from views._common import call, portfolio_key

SUGGESTION_LABELS = {
    "do_nothing": "🟥 Wait — do nothing for now",
    "moderate": "🟨 Moderate adjustment may be considered",
    "aggressive": "🟩 Acting now carries relatively low risk",
}


@st.cache_data(ttl=3600, show_spinner="Computing today's recommendation…")
def _daily(pkey: str) -> dict:
    return api.recommend_daily()


@st.cache_data(ttl=3600, show_spinner="Loading events…")
def _events(pkey: str) -> dict:
    return api.recommendation_events(5)


def _trades_table(trades: list) -> None:
    st.dataframe(
        pd.DataFrame(
            [{"Ticker": t["symbol"], "Weight change": f"{t['weight_change_pct']:+.1f}%",
              "Reason": t["reason"]} for t in trades]
        ),
        hide_index=True,
        width="stretch",
    )


def render() -> None:
    st.set_page_config(page_title="AURORA — Should I React?", page_icon="🤔", layout="wide")
    st.title("🤔 Should I React?")
    st.caption("Engine 4 — Reaction Risk & Recommendation (Developer 4)")

    pkey = call(portfolio_key)
    if pkey is None:
        return

    # ---------------------------------------------------- daily recommendation
    st.subheader("Today's normal-day recommendation")
    daily = call(_daily, pkey)
    if daily is None:
        return
    rec = daily["recommendation"]
    st.write(rec["explanation"])
    if rec["trades"]:
        _trades_table(rec["trades"])
        st.metric(
            "Portfolio health if applied",
            f"{daily['health_after']:.0f}/100",
            delta=f"{daily['health_after'] - daily['health_before']:+.1f} "
            f"vs current {daily['health_before']:.0f}",
        )

    # ---------------------------------------------------------- event reaction
    st.subheader("React to an event")
    payload = call(_events, pkey)
    if payload is None:
        return
    events = payload["events"]
    if payload["demo"]:
        st.caption(
            "No real events yet (news engine pending — Developer 3). "
            "Showing a demo event so the flow is testable."
        )

    picked = st.selectbox(
        "Event", options=range(len(events)), format_func=lambda i: events[i]["title"]
    )
    event = events[picked]

    reaction = call(api.react, event)
    if reaction is None:
        return
    risk = reaction["risk"]
    event_rec = reaction["recommendation"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Risk of reacting", f"{risk['risk_pct']:.0f}%")
        st.markdown(SUGGESTION_LABELS.get(risk["suggestion"], risk["suggestion"]))
    with c2:
        st.markdown("**Why:**")
        for r in risk["reasons"]:
            st.markdown(f"- {r}")

    st.markdown("**Factor breakdown** (0 = safe to act, 1 = risky):")
    st.dataframe(
        pd.Series(risk["factors"], name="value").round(2),
        width="stretch",
    )

    st.markdown(f"**Recommendation:** {event_rec['explanation']}")
    if event_rec["trades"]:
        _trades_table(event_rec["trades"])
    st.caption("AURORA never executes trades — the final decision is always yours.")
