"""Daily Strategy page — Developer 2 owns everything here.
(The Performance page is presented by Developer 1: views/portfolio_health.py)

This page shows ONLY the EMA/RSI/MACD confluence recommendations. The frozen
classify_regime() / score_assets() functions still live in the backend (Dev 4's
recommendation engine and the Performance page consume them) — they are just not
displayed here.
"""

from typing import List

import pandas as pd
import streamlit as st

import api_client as api
from theme import apply_theme
from views._common import call, portfolio_key

ACTION_LABELS = {
    "BUY": "🟢 BUY", "ADD": "🟢 ADD", "TRIM": "🟡 TRIM",
    "SELL": "🔴 SELL", "HOLD": "⚪ HOLD",
}


@st.cache_data(ttl=3600, show_spinner="Scanning for signals…")
def _recommendations(pkey: str) -> List[dict]:
    return api.strategy_recommendations()


def render() -> None:
    st.set_page_config(page_title="AURORA — Daily Strategy", page_icon="📈", layout="wide")
    apply_theme()
    st.title("📈 Daily Strategy")
    st.caption("Engine 2 — EMA/RSI/MACD confluence signals (Developer 2)")

    pkey = call(portfolio_key)
    if pkey is None:
        return

    st.subheader("Per-stock recommendations")
    st.caption(
        "Confluence score = EMA20/50 (±1.5) + fresh MACD crossover (±1.0) + RSI (±1.0). "
        "Raw BUY at score ≥ 2, SELL at ≤ −2. The portfolio layer then maps "
        "(what you hold + P&L) into BUY / ADD / TRIM / SELL / HOLD."
    )
    recs = call(_recommendations, pkey)
    if recs is None:
        return
    if not recs:
        st.warning("No symbol has enough history to score yet.")
        return

    table = pd.DataFrame(
        [
            {"Ticker": r["symbol"],
             "Action": ACTION_LABELS.get(r["recommendation"], r["recommendation"]),
             "Held": "✓" if r["held"] else "",
             "Score": r["score"],
             "P&L %": r["unrealized_pnl_pct"],
             "Price": r["current_price"],
             "Why": " · ".join(r["reasons"])}
            for r in recs
        ]
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.NumberColumn(format="%.1f"),
            "P&L %": st.column_config.NumberColumn(format="%.1f%%"),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Why": st.column_config.TextColumn(width="large"),
        },
    )
