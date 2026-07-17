"""Daily Strategy page — Developer 2 owns everything here.
(The Performance page is presented by Developer 1: views/portfolio_health.py)"""

from typing import List

import pandas as pd
import streamlit as st

import api_client as api
from views._common import call, portfolio_key

REGIME_LABELS = {
    "bullish": "🟢 Bullish",
    "bearish": "🔴 Bearish",
    "high_volatility": "🟠 High volatility",
    "sideways": "🟡 Sideways / uncertain",
}


@st.cache_data(ttl=3600, show_spinner="Classifying market regime…")
def _regime(pkey: str) -> dict:
    return api.regime()


@st.cache_data(ttl=3600, show_spinner="Scoring assets…")
def _signals(pkey: str) -> List[dict]:
    return api.signals()


def render() -> None:
    st.set_page_config(page_title="AURORA — Daily Strategy", page_icon="📈", layout="wide")
    st.title("📈 Daily Strategy")
    st.caption("Engine 2 — Regime-Aware Momentum (Developer 2)")

    pkey = call(portfolio_key)
    if pkey is None:
        return
    regime = call(_regime, pkey)
    if regime is None:
        return

    c1, c2 = st.columns([1, 2])
    c1.metric(
        "Market regime",
        REGIME_LABELS.get(regime["regime"], regime["regime"]),
        delta=f"confidence {regime['confidence']:.0%}",
        delta_color="off",
    )
    with c2:
        st.dataframe(
            pd.Series(regime["indicators"], name="value").round(4),
            width="stretch",
        )

    st.subheader("Daily asset ranking")
    signals = call(_signals, pkey)
    if signals is None:
        return
    if not signals:
        st.warning("No asset has enough history to score yet.")
        return
    table = pd.DataFrame(
        [
            {"Ticker": s["symbol"], "Score": s["score"], "Signal": s["action"],
             "Momentum": s["indicators"]["momentum"] * 100,
             "Sharpe": s["indicators"]["sharpe"],
             "Volatility": s["indicators"]["volatility"] * 100, "Why": s["rationale"]}
            for s in signals
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
            "Momentum": st.column_config.NumberColumn(format="%.1f%%"),
            "Volatility": st.column_config.NumberColumn(format="%.1f%%"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        },
    )
