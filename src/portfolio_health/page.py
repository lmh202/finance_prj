"""Portfolio Health page — Developer 1 owns everything rendered here."""

import pandas as pd
import streamlit as st

from src import data_loader
from src import portfolio as pf
from src.interfaces import BENCHMARK
from src.portfolio_health import engine


@st.cache_data(ttl=3600, show_spinner="Loading price history…")
def _history(symbols: tuple) -> pd.DataFrame:
    return data_loader.get_history(list(symbols))


def render() -> None:
    st.set_page_config(page_title="AURORA — Portfolio Health", page_icon="🩺", layout="wide")
    st.title("🩺 Portfolio Health")
    st.caption("Engine 1 — Portfolio Intelligence (Developer 1)")

    holdings = pf.load_portfolio()
    if holdings.empty:
        st.info("Your portfolio is empty — build it on the Home page first.")
        return

    history = _history(tuple(sorted(set(holdings["symbol"]))) + (BENCHMARK,))
    if history.empty:
        st.error("Could not load price history — check your connection and retry.")
        return

    report = engine.compute_health(holdings, history)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Health score", f"{report.score:.0f}/100")
    if report.metrics:
        c2.metric("Sharpe", f"{report.metrics['sharpe']:.2f}")
        c3.metric("Max drawdown", f"{report.metrics['max_drawdown']:.1%}")
        c4.metric("Volatility (ann.)", f"{report.metrics['annual_volatility']:.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Strengths")
        for s in report.strengths or ["—"]:
            st.markdown(f"- ✅ {s}")
        st.subheader("Weaknesses")
        for w in report.weaknesses or ["—"]:
            st.markdown(f"- ⚠️ {w}")
    with right:
        st.subheader("All metrics")
        st.dataframe(
            pd.Series(report.metrics, name="value").round(3),
            width="stretch",
        )

    if report.correlation is not None:
        st.subheader("Asset correlation")
        st.dataframe(report.correlation.round(2), width="stretch")
