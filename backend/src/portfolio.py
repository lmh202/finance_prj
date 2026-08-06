"""Portfolio shaping and valuation for AURORA.

A portfolio is a DataFrame with columns:
    symbol     str   ticker as listed in the NASDAQ symbol directory
    name       str   security name (display only)
    shares     float number of shares held
    buy_price  float average purchase price per share (cost basis)

The backend does NOT persist portfolios. Each client owns its own holdings
(the Next.js app keeps them in localStorage, the Streamlit app in a local
file) and sends them along with every request, so one backend process can
serve many users without a database, a disk or a session. The only file this
module still reads is the committed `sample_portfolio.csv` fixture, which is
handed to a client to store rather than saved here.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import DATA_DIR

SAMPLE_PORTFOLIO_CSV = DATA_DIR / "sample_portfolio.csv"

COLUMNS = ["symbol", "name", "shares", "buy_price"]


def empty_portfolio() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def load_portfolio_file(path: Path) -> pd.DataFrame:
    """Read a holdings CSV off disk — the sample fixture, or a test file."""
    if not path.exists():
        return empty_portfolio()
    return _normalize(pd.read_csv(path))[0]


def holdings_from_records(rows: Optional[List[Dict]]) -> Tuple[pd.DataFrame, List[str]]:
    """Canonicalise client-supplied holdings (the request-body entry point).

    Every engine endpoint receives untrusted rows from a browser, so this
    runs the same repair/merge pass that a CSV import does: bad rows are
    dropped and reported, duplicate symbols merge into one weighted-average
    lot.
    """
    if not rows:
        return empty_portfolio(), []
    return _normalize(pd.DataFrame(rows))


def _normalize(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Coerce a raw holdings frame to the canonical schema.

    Returns (clean_df, problems) — rows that cannot be repaired are dropped
    and reported rather than raising, so a half-good CSV still imports.
    """
    problems: List[str] = []
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "symbol" not in df.columns or "shares" not in df.columns:
        return empty_portfolio(), ["CSV must have at least 'symbol' and 'shares' columns."]

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["buy_price"] = pd.to_numeric(df.get("buy_price"), errors="coerce")
    if "name" not in df.columns:
        df["name"] = ""
    df["name"] = df["name"].fillna("").astype(str)

    bad = df["symbol"].eq("") | df["symbol"].eq("NAN") | df["shares"].isna() | (df["shares"] <= 0)
    for _, row in df[bad].iterrows():
        problems.append(f"Skipped row with symbol='{row['symbol']}', shares='{row['shares']}'.")
    df = df[~bad]

    # Same symbol twice -> merge into one lot with a weighted-average cost
    if df["symbol"].duplicated().any():
        df = (
            df.assign(_cost=df["shares"] * df["buy_price"].fillna(0.0))
            .groupby("symbol", as_index=False)
            .agg(name=("name", "first"), shares=("shares", "sum"), _cost=("_cost", "sum"))
        )
        df["buy_price"] = (df["_cost"] / df["shares"]).where(df["_cost"] > 0)
        df = df.drop(columns="_cost")

    return df[COLUMNS].reset_index(drop=True), problems


def parse_uploaded_csv(file) -> Tuple[pd.DataFrame, List[str]]:
    """Parse a user-uploaded portfolio CSV (symbol, shares[, buy_price, name])."""
    try:
        raw = pd.read_csv(file)
    except Exception as exc:
        return empty_portfolio(), [f"Could not read CSV: {exc}"]
    return _normalize(raw)


def build_view(
    holdings: pd.DataFrame, prices: Dict[str, float], cash: float = 0.0
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Join holdings with latest prices and compute values, P/L and weights.

    Where a live price is unavailable the cost basis stands in for market
    value (flagged via has_price=False) so totals stay meaningful.
    """
    view = holdings.copy()
    if view.empty:
        totals = {"market_value": cash, "invested": 0.0, "cost": 0.0, "pnl": 0.0, "cash": cash}
        return view, totals

    view["current_price"] = view["symbol"].map(prices)
    view["has_price"] = view["current_price"].notna()
    view["cost_value"] = view["shares"] * view["buy_price"]
    view["market_value"] = (view["shares"] * view["current_price"]).fillna(view["cost_value"])
    view["pnl"] = view["market_value"] - view["cost_value"]
    view["pnl_pct"] = (view["pnl"] / view["cost_value"].where(view["cost_value"] > 0)) * 100

    total_value = float(view["market_value"].sum()) + cash
    view["weight_pct"] = view["market_value"] / total_value * 100 if total_value > 0 else 0.0

    totals = {
        "market_value": total_value,
        "invested": float(view["market_value"].sum()),
        "cost": float(view["cost_value"].sum()),
        "pnl": float(view["pnl"].sum()),
        "cash": cash,
    }
    return view, totals
