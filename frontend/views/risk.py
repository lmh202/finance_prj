"""Risk page — per-stock + portfolio downside risk from the risk engine.

Shows, in plain terms: how much each holding is expected to swing, its worst-case
loss, and a 0-100 danger level; plus one card for the whole portfolio. Data comes
from /risk/estimates and /risk/portfolio (HAR volatility forecast + Filtered
Historical Simulation), NOT the crude reaction-risk proxy on 'Should I React'.
"""

from typing import List

import pandas as pd
import streamlit as st

import api_client as api
from theme import apply_theme
from views._common import call, portfolio_key


@st.cache_data(ttl=1800, show_spinner="Estimating risk…")
def _estimates(pkey: str, horizon: int) -> List[dict]:
    return api.risk_estimates(horizon)


@st.cache_data(ttl=1800, show_spinner="Estimating portfolio risk…")
def _portfolio(pkey: str, horizon: int) -> dict:
    return api.risk_portfolio(horizon)


def render() -> None:
    st.set_page_config(page_title="AURORA — Risk", page_icon="🛡️", layout="wide")
    apply_theme()
    st.title("🛡️ Risk")
    st.caption(
        "How much each holding could swing, and the worst it would normally lose. "
        "This forecasts the *size* of moves (not the direction) — the number a "
        "position-sizing or reaction decision should lean on."
    )

    pkey = call(portfolio_key)
    if pkey is None:
        return

    horizon = st.radio("Look-ahead window", [5, 20], index=0, horizontal=True,
                       format_func=lambda h: f"next {h} trading days"
                       f" (~{'1 week' if h == 5 else '1 month'})")

    # ---- whole-portfolio card ----
    pr = call(_portfolio, pkey, horizon)
    if pr is None:
        return
    st.subheader("Your whole portfolio")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Typical swing", f"±{pr['sigma_h']*100:.1f}%",
              help="Expected size of the portfolio move over the window.")
    c2.metric("Worst-case loss (95%)", f"{pr['var_95']*100:.1f}%",
              help="19 weeks out of 20, losses stay smaller than this.")
    c3.metric("Severe loss (worst 1%)", f"{pr['var_99']*100:.1f}%",
              help="A rare bad-day figure — losses beyond the 95% line.")
    offset = (1 - pr["diversification_ratio"]) * 100
    c4.metric("Diversification benefit", f"{offset:.0f}%",
              help="How much your holdings offset each other vs. holding them alone.")
    st.caption(f"As of {pr['as_of']} · {pr['n_holdings']} holdings scored.")

    # ---- per-stock table ----
    st.subheader("Each holding")
    ests = call(_estimates, pkey, horizon)
    if ests is None:
        return
    ests = [e for e in ests if e.get("has_history")]
    if not ests:
        st.warning("No holding has enough price history to score yet.")
        return
    ests.sort(key=lambda e: e["risk_level"], reverse=True)

    table = pd.DataFrame([
        {"Ticker": e["symbol"],
         "Danger (0-100)": e["risk_level"],
         "Typical swing": e["sigma_h"] * 100,
         "Worst-case loss (95%)": e["var_95"] * 100,
         "Severe loss (1%)": e["var_99"] * 100}
        for e in ests
    ])
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Danger (0-100)": st.column_config.ProgressColumn(
                format="%.0f", min_value=0, max_value=100,
                help="How risky this stock is right now vs. its own last 2 years."),
            "Typical swing": st.column_config.NumberColumn(format="±%.1f%%"),
            "Worst-case loss (95%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Severe loss (1%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.caption(
        "Danger = today's expected swing ranked against the stock's own history "
        "(90+ = unusually turbulent). Losses are downside thresholds, validated on "
        "3 years of held-out data (95% line breached ~5% of the time, as intended)."
    )
