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


def render_performance() -> None:
    """Performance & Benchmark page — owned by Developer 1.

    Presents Developer 2's backtest output (a contract function) alongside
    the portfolio metrics this engine computes.
    """
    from src.daily_strategy import engine as strategy

    st.set_page_config(page_title="AURORA — Performance", page_icon="🏁", layout="wide")
    st.title("🏁 Performance & Benchmark")
    st.caption("Presentation: Developer 1 · Backtest engine: Developer 2")

    holdings = pf.load_portfolio()
    if holdings.empty:
        st.info("Your portfolio is empty — build it on the Home page first.")
        return
    history = _history(tuple(sorted(set(holdings["symbol"]))) + (BENCHMARK,))
    if history.empty:
        st.error("Could not load price history — check your connection and retry.")
        return

    curves = strategy.backtest(history, holdings)
    if curves.empty:
        st.warning("Nothing to backtest yet.")
        return

    final = curves.iloc[-1]
    cols = st.columns(len(curves.columns))
    for col, name in zip(cols, curves.columns):
        col.metric(f"{name} (growth of $1)", f"${final[name]:.3f}")
    st.dataframe(curves.tail(10).round(4), width="stretch")
    st.caption(
        "TODO Developer 1: cumulative-return chart, rolling metrics, and the "
        "metric table (return, Sharpe, Sortino, max DD, vol, turnover). "
        "TODO Developer 2: add the ML strategy runs (price-only vs price+news) "
        "to backtest() so the ablation shows up here."
    )
