"""AURORA — Portfolio Builder (Streamlit entry point).

Run with:  streamlit run app/app.py

Today's scope: let a user mirror their real portfolio here — pick any
US-listed security (full NASDAQ symbol directory behind the search box),
enter shares and cost basis, see live value / P&L / allocation.
News-driven rebalancing (RSS) plugs in later via src/news_rss.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import data_loader
from src import portfolio as pf

st.set_page_config(page_title="AURORA — Portfolio Builder", page_icon="🌅", layout="wide")

# Chart ink/palette per theme (dataviz reference palette)
PALETTE = {
    "light": {"bar": "#2a78d6", "cash": "#898781", "ink": "#0b0b0b",
              "muted": "#898781", "grid": "#e1e0d9"},
    "dark": {"bar": "#3987e5", "cash": "#898781", "ink": "#ffffff",
             "muted": "#898781", "grid": "#2c2c2a"},
}


def theme_type() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


# ---------------------------------------------------------------- data access

@st.cache_data(ttl=86400, show_spinner="Loading US stock universe…")
def load_universe() -> pd.DataFrame:
    return data_loader.load_ticker_universe()


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
    return data_loader.get_latest_prices(list(symbols))


def set_holdings(df: pd.DataFrame) -> None:
    st.session_state.holdings = df
    pf.save_portfolio(df)


if "holdings" not in st.session_state:
    st.session_state.holdings = pf.load_portfolio()
if "cash" not in st.session_state:
    st.session_state.cash = pf.load_cash()

symbols, labels, names = universe_labels()

# ------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("⚙️ Portfolio setup")

    cash = st.number_input(
        "Cash balance ($)", min_value=0.0, step=100.0,
        value=float(st.session_state.cash),
    )
    if cash != st.session_state.cash:
        st.session_state.cash = cash
        pf.save_cash(cash)

    st.divider()
    st.subheader("Import / export")
    uploaded = st.file_uploader(
        "Upload portfolio CSV", type="csv",
        help="Columns: symbol, shares and optionally buy_price, name",
    )
    if uploaded is not None:
        imported, problems = pf.parse_uploaded_csv(uploaded)
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
        sample = pf.load_portfolio(pf.SAMPLE_PORTFOLIO_CSV)
        set_holdings(sample)
        st.session_state.cash = 23000.0
        pf.save_cash(23000.0)
        st.rerun()
    if st.button("Refresh prices"):
        get_prices.clear()
        st.rerun()

    st.divider()
    st.caption(
        "📰 **Coming soon:** essential-news feed (RSS) with rebalancing "
        "suggestions — see `src/news_intelligence/`. Explore the engine "
        "pages in the sidebar above."
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
            set_holdings(
                pf.add_holding(st.session_state.holdings, pick, names.get(pick, ""), shares, price)
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

prices = get_prices(tuple(holdings["symbol"]))
view, totals = pf.build_view(holdings, prices, st.session_state.cash)

unpriced = view.loc[~view["has_price"], "symbol"].tolist()
if unpriced:
    st.warning(
        "No live price for: " + ", ".join(unpriced) + " — showing cost basis instead."
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total value (incl. cash)", f"${totals['market_value']:,.2f}")
m2.metric("Invested", f"${totals['invested']:,.2f}")
m3.metric("Cash", f"${totals['cash']:,.2f}")
pnl_pct = (totals["pnl"] / totals["cost"] * 100) if totals["cost"] > 0 else 0.0
m4.metric("Unrealized P/L", f"${totals['pnl']:,.2f}", delta=f"{pnl_pct:+.2f}%")

# --------------------------------------------------------- table + allocation

left, right = st.columns([3, 2])

with left:
    st.subheader("Holdings")
    display = view[
        ["symbol", "name", "shares", "buy_price", "current_price",
         "market_value", "pnl", "pnl_pct", "weight_pct"]
    ].sort_values("market_value", ascending=False)
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
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
            clean, problems = pf.normalize_holdings(edited)
            for p in problems:
                st.warning(p)
            set_holdings(clean)
            st.rerun()
