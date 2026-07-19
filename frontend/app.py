"""AURORA — Portfolio Builder (Streamlit entry point).

Run with:  streamlit run frontend/app.py
(the FastAPI backend must be up: uvicorn main:app --app-dir backend --port 8000,
or start both with scripts/dev.ps1)

Today's scope: let a user mirror their real portfolio here — pick any
US-listed security (full NASDAQ symbol directory behind the search box),
enter shares and cost basis, see live value / P&L / allocation.
All data comes from the backend over HTTP via api_client — no backend
imports anywhere under frontend/.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import api_client as api
from theme import CHART, apply_theme

st.set_page_config(page_title="AURORA — Portfolio Builder", page_icon="🌅", layout="wide")
apply_theme()

# Chart ink/palette per theme — dark mirrors the ui_design ("Prism") look.
PALETTE = {
    "light": {"bar": "#2a78d6", "cash": "#898781", "ink": "#0b0b0b",
              "muted": "#898781", "grid": "#e1e0d9"},
    "dark": {"bar": CHART["accent"], "cash": CHART["cash"], "ink": CHART["ink"],
             "muted": CHART["mut"], "grid": CHART["grid"]},
}


def theme_type() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------- data access

@st.cache_data(ttl=86400, show_spinner="Loading US stock universe…")
def load_universe() -> pd.DataFrame:
    return api.get_universe()


@st.cache_data(ttl=86400)
def universe_labels():
    uni = load_universe()
    labels = uni["symbol"] + " — " + uni["name"] + "  (" + uni["exchange"] + ")"
    return (
        uni["symbol"].tolist(),
        dict(zip(uni["symbol"], labels)),
        dict(zip(uni["symbol"], uni["name"])),
    )


@st.cache_data(ttl=300, show_spinner="Fetching latest prices…")
def get_prices(symbols: tuple) -> dict:
    return api.get_prices(list(symbols))


@st.cache_data(ttl=300, show_spinner="Fetching latest prices…")
def get_view(portfolio_key: str) -> tuple:
    """Valuation view + totals; keyed on the holdings/cash snapshot so edits
    invalidate immediately while prices stay cached for 5 minutes."""
    return api.get_view()


@st.cache_data(ttl=900, show_spinner="Loading price history…")
def get_symbol_history(symbol: str) -> pd.DataFrame:
    return api.get_history([symbol], period="1y")


def set_holdings(df: pd.DataFrame) -> None:
    clean, _ = api.put_portfolio(df)
    st.session_state.holdings = clean


try:
    if "holdings" not in st.session_state:
        st.session_state.holdings = api.get_portfolio()
    if "cash" not in st.session_state:
        st.session_state.cash = api.get_cash()
    symbols, labels, names = universe_labels()
except api.ApiUnavailable:
    st.error(
        "Backend not running — start it with `scripts/dev.ps1` or "
        "`uvicorn main:app --app-dir backend --port 8000`, then reload."
    )
    st.stop()

# ------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("⚙️ Portfolio setup")

    cash = st.number_input(
        "Cash balance ($)", min_value=0.0, step=100.0,
        value=float(st.session_state.cash),
    )
    if cash != st.session_state.cash:
        st.session_state.cash = cash
        api.put_cash(cash)

    st.divider()
    st.subheader("Import / export")
    uploaded = st.file_uploader(
        "Upload portfolio CSV", type="csv",
        help="Columns: symbol, shares and optionally buy_price, name",
    )
    if uploaded is not None:
        imported, problems = api.parse_csv(
            uploaded.getvalue().decode("utf-8", errors="replace")
        )
        for p in problems:
            st.warning(p)
        if not imported.empty:
            st.caption(f"{len(imported)} holdings found in file.")
            if st.button("Import (replaces current portfolio)", type="primary"):
                imported["name"] = imported.apply(
                    lambda r: r["name"] or names.get(r["symbol"], ""), axis=1
                )
                set_holdings(imported)
                st.rerun()

    st.download_button(
        "Download current portfolio (CSV)",
        st.session_state.holdings.to_csv(index=False),
        file_name="my_portfolio.csv",
        mime="text/csv",
        disabled=st.session_state.holdings.empty,
    )
    st.download_button(
        "Download CSV template",
        "symbol,shares,buy_price\nAAPL,10,250\nSPY,5,600\n",
        file_name="portfolio_template.csv",
        mime="text/csv",
    )

    st.divider()
    if st.button("Load sample portfolio"):
        sample, sample_cash = api.load_sample()
        st.session_state.holdings = sample
        st.session_state.cash = sample_cash
        st.rerun()
    if st.button("Refresh prices"):
        get_prices.clear()
        get_view.clear()
        st.rerun()

    st.divider()
    st.caption(
        "📰 **Coming soon:** essential-news feed (RSS) with rebalancing "
        "suggestions — see `backend/src/news_intelligence/`. Explore the "
        "engine pages in the sidebar above."
    )

# -------------------------------------------------------------------- header

st.title("🌅 AURORA")
st.caption(
    "AI-powered portfolio intelligence copilot — **step 1: mirror your real "
    "portfolio.** Every US-listed stock and ETF is searchable below "
    f"({len(symbols):,} securities, NASDAQ symbol directory)."
)

# -------------------------------------------------------------- add a holding

with st.container(border=True):
    st.subheader("➕ Add a holding")
    with st.form("add_holding", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1], vertical_alignment="bottom")
        pick = c1.selectbox(
            "Security", options=symbols,
            format_func=lambda s: labels.get(s, s),
            index=None, placeholder="Type a ticker or company name…",
        )
        shares = c2.number_input("Shares", min_value=0.0, value=1.0, step=1.0)
        buy_price = c3.number_input(
            "Avg. buy price ($)", min_value=0.0, value=0.0, step=1.0,
            help="Leave at 0 to use the latest market price as your cost basis.",
        )
        submitted = c4.form_submit_button("Add", type="primary", width="stretch")

    if submitted:
        if pick is None:
            st.error("Pick a security first.")
        elif shares <= 0:
            st.error("Shares must be greater than zero.")
        else:
            price = buy_price
            if price == 0:
                fetched = get_prices((pick,)).get(pick)
                if fetched is None:
                    st.error(
                        f"No live price available for {pick} — "
                        "please enter your buy price manually."
                    )
                    st.stop()
                price = fetched
            st.session_state.holdings = api.add_holding(
                pick, names.get(pick, ""), shares, price
            )
            st.toast(f"Added {shares:g} × {pick} @ ${price:,.2f}")
            st.rerun()

holdings = st.session_state.holdings

if holdings.empty:
    st.info(
        "Your portfolio is empty. Add your first holding above, upload a CSV, "
        "or load the sample portfolio from the sidebar."
    )
    st.stop()

# ------------------------------------------------------------------ valuation

view, totals = get_view(holdings.to_json() + f"|cash={st.session_state.cash}")

st.divider()
st.subheader("📊 Overview")

unpriced = view.loc[~view["has_price"], "symbol"].tolist()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total value (incl. cash)", f"${totals['market_value']:,.2f}", border=True)
m2.metric("Invested", f"${totals['invested']:,.2f}", border=True)
m3.metric("Cash", f"${totals['cash']:,.2f}", border=True)
pnl_pct = (totals["pnl"] / totals["cost"] * 100) if totals["cost"] > 0 else 0.0
m4.metric(
    "Unrealized P/L", f"${totals['pnl']:,.2f}", delta=f"{pnl_pct:+.2f}%", border=True
)

if unpriced:
    st.warning(
        "No live price for: " + ", ".join(unpriced) + " — showing cost basis instead."
    )

# --------------------------------------------------------- table + allocation

st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("Holdings")
    display = view[
        ["symbol", "name", "shares", "buy_price", "current_price",
         "market_value", "pnl", "pnl_pct", "weight_pct"]
    ].sort_values("market_value", ascending=False)

    def _pnl_color(v):
        if pd.isna(v) or v == 0:
            return ""
        return f"color: {CHART['gain']}" if v > 0 else f"color: {CHART['loss']}"

    styled = display.style.map(_pnl_color, subset=["pnl", "pnl_pct"])
    table_event = st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "symbol": st.column_config.TextColumn("Ticker"),
            "name": st.column_config.TextColumn("Name", width="medium"),
            "shares": st.column_config.NumberColumn("Shares", format="%.4g"),
            "buy_price": st.column_config.NumberColumn("Avg. cost", format="$%.2f"),
            "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "market_value": st.column_config.NumberColumn("Value", format="$%.2f"),
            "pnl": st.column_config.NumberColumn("P/L", format="$%.2f"),
            "pnl_pct": st.column_config.NumberColumn("P/L %", format="%.2f%%"),
            "weight_pct": st.column_config.ProgressColumn(
                "Weight", format="%.1f%%", min_value=0, max_value=100
            ),
        },
    )

with right:
    st.subheader("Allocation")
    colors = PALETTE[theme_type()]
    alloc = display[["symbol", "market_value", "weight_pct"]].copy()
    if totals["cash"] > 0:
        cash_weight = totals["cash"] / totals["market_value"] * 100
        alloc = pd.concat(
            [alloc, pd.DataFrame([{"symbol": "Cash", "market_value": totals["cash"],
                                   "weight_pct": cash_weight}])],
            ignore_index=True,
        )
    alloc = alloc.sort_values("weight_pct", ascending=True)  # largest on top

    fig = go.Figure(
        go.Bar(
            x=alloc["weight_pct"],
            y=alloc["symbol"],
            orientation="h",
            marker_color=[
                colors["cash"] if s == "Cash" else colors["bar"] for s in alloc["symbol"]
            ],
            text=[f"{w:.1f}%" for w in alloc["weight_pct"]],
            textposition="outside",
            textfont={"color": colors["muted"], "size": 12},
            customdata=alloc["market_value"],
            hovertemplate="<b>%{y}</b>: %{x:.1f}%  ($%{customdata:,.0f})<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(260, 34 * len(alloc) + 90),
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.35,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
              "color": colors["muted"]},
        xaxis={
            "title": None, "ticksuffix": "%", "gridcolor": colors["grid"],
            "zeroline": False, "range": [0, float(alloc["weight_pct"].max()) * 1.18],
        },
        yaxis={"title": None, "tickfont": {"color": colors["ink"], "size": 13}},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# --------------------------------------------------------------- stock chart

st.divider()
st.subheader("📈 Price history")

selected_rows = table_event.selection.rows if table_event is not None else []
if not selected_rows:
    st.caption("Select a row in the Holdings table above to see its price chart.")
else:
    sel_symbol = display.iloc[selected_rows[0]]["symbol"]
    hist = get_symbol_history(sel_symbol)

    if hist.empty or sel_symbol not in hist.columns:
        st.warning(f"No price history available for {sel_symbol}.")
    else:
        colors = PALETTE[theme_type()]
        series = hist[sel_symbol].dropna()
        last, first = series.iloc[-1], series.iloc[0]
        chg_pct = (last / first - 1) * 100 if first else 0.0

        st.markdown(
            f"**{sel_symbol}** — ${last:,.2f} "
            f"(:{'green' if chg_pct >= 0 else 'red'}[{chg_pct:+.1f}%] over 1y)"
        )
        fig2 = go.Figure(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line={"color": colors["bar"], "width": 2},
                fill="tozeroy",
                fillcolor=_hex_to_rgba(colors["bar"], 0.12),
                hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.2f}<extra></extra>",
            )
        )
        fig2.update_layout(
            height=320,
            margin={"l": 8, "r": 8, "t": 8, "b": 8},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
                  "color": colors["muted"]},
            xaxis={"gridcolor": colors["grid"], "showgrid": False},
            yaxis={"gridcolor": colors["grid"], "tickprefix": "$", "zeroline": False},
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ------------------------------------------------------------- edit holdings

with st.expander("✏️ Edit or remove holdings"):
    st.caption(
        "Edit shares / cost directly, or select rows and press Delete to remove "
        "them. Then save."
    )
    with st.form("edit_holdings"):
        edited = st.data_editor(
            st.session_state.holdings,
            hide_index=True,
            width="stretch",
            num_rows="dynamic",
            column_config={
                "symbol": st.column_config.TextColumn("Ticker", disabled=True),
                "name": st.column_config.TextColumn("Name", disabled=True),
                "shares": st.column_config.NumberColumn("Shares", min_value=0.0),
                "buy_price": st.column_config.NumberColumn("Avg. cost ($)", min_value=0.0),
            },
        )
        if st.form_submit_button("💾 Save changes"):
            clean, problems = api.put_portfolio(edited)
            for p in problems:
                st.warning(p)
            st.session_state.holdings = clean
            st.rerun()
