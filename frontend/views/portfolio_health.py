"""Portfolio Health page — Developer 1 owns everything rendered here."""

import pandas as pd
import streamlit as st

import api_client as api
from theme import apply_theme
from views._common import call, portfolio_key


@st.cache_data(ttl=3600, show_spinner="Computing health report…")
def _report(pkey: str) -> dict:
    return api.health_report()


@st.cache_data(ttl=3600, show_spinner="Running backtest…")
def _backtest(pkey: str) -> pd.DataFrame:
    return api.backtest()


def render() -> None:
    st.set_page_config(page_title="AURORA — Portfolio Health", page_icon="🩺", layout="wide")
    apply_theme()
    st.title("🩺 Portfolio Health")
    st.caption("Engine 1 — Portfolio Intelligence (Developer 1)")

    pkey = call(portfolio_key)
    if pkey is None:
        return
    report = call(_report, pkey)
    if report is None:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Health score", f"{report['score']:.0f}/100")
    if report["metrics"]:
        c2.metric("Sharpe", f"{report['metrics']['sharpe']:.2f}")
        c3.metric("Max drawdown", f"{report['metrics']['max_drawdown']:.1%}")
        c4.metric("Volatility (ann.)", f"{report['metrics']['annual_volatility']:.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Strengths")
        for s in report["strengths"] or ["—"]:
            st.markdown(f"- ✅ {s}")
        st.subheader("Weaknesses")
        for w in report["weaknesses"] or ["—"]:
            st.markdown(f"- ⚠️ {w}")
    with right:
        st.subheader("All metrics")
        st.dataframe(
            pd.Series(report["metrics"], name="value").round(3),
            width="stretch",
        )

    if report["correlation"] is not None:
        st.subheader("Asset correlation")
        st.dataframe(report["correlation"].round(2), width="stretch")


def render_performance() -> None:
    """Performance & Benchmark page — owned by Developer 1.

    Presents Developer 2's backtest output (served by /strategy/backtest)
    alongside the portfolio metrics this engine computes.
    """
    st.set_page_config(page_title="AURORA — Performance", page_icon="🏁", layout="wide")
    apply_theme()
    st.title("🏁 Performance & Benchmark")
    st.caption("Presentation: Developer 1 · Backtest engine: Developer 2")

    pkey = call(portfolio_key)
    if pkey is None:
        return
    curves = call(_backtest, pkey)
    if curves is None:
        return
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
