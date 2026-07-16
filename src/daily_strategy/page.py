"""Daily Strategy page — Developer 2 owns everything here.
(The Performance page is presented by Developer 1: src/portfolio_health/page.py)"""

import pandas as pd
import streamlit as st

from src import data_loader
from src import portfolio as pf
from src.daily_strategy import engine
from src.interfaces import BENCHMARK

REGIME_LABELS = {
    "bullish": "🟢 Bullish",
    "bearish": "🔴 Bearish",
    "high_volatility": "🟠 High volatility",
    "sideways": "🟡 Sideways / uncertain",
}


@st.cache_data(ttl=3600, show_spinner="Loading price history…")
def _history(symbols: tuple) -> pd.DataFrame:
    return data_loader.get_history(list(symbols))


def _load():
    holdings = pf.load_portfolio()
    if holdings.empty:
        st.info("Your portfolio is empty — build it on the Home page first.")
        return None, None
    history = _history(tuple(sorted(set(holdings["symbol"]))) + (BENCHMARK,))
    if history.empty:
        st.error("Could not load price history — check your connection and retry.")
        return None, None
    return holdings, history


def render() -> None:
    st.set_page_config(page_title="AURORA — Daily Strategy", page_icon="📈", layout="wide")
    st.title("📈 Daily Strategy")
    st.caption("Engine 2 — Regime-Aware Momentum (Developer 2)")

    holdings, history = _load()
    if holdings is None:
        return

    regime = engine.classify_regime(history)
    c1, c2 = st.columns([1, 2])
    c1.metric(
        "Market regime",
        REGIME_LABELS.get(regime.regime, regime.regime),
        delta=f"confidence {regime.confidence:.0%}",
        delta_color="off",
    )
    with c2:
        st.dataframe(
            pd.Series(regime.indicators, name="value").round(4),
            width="stretch",
        )

    st.subheader("Daily asset ranking")
    signals = engine.score_assets(history, holdings)
    if not signals:
        st.warning("No asset has enough history to score yet.")
        return
    table = pd.DataFrame(
        [
            {"Ticker": s.symbol, "Score": s.score, "Signal": s.action,
             "Momentum": s.indicators["momentum"] * 100,
             "Sharpe": s.indicators["sharpe"],
             "Volatility": s.indicators["volatility"] * 100, "Why": s.rationale}
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


