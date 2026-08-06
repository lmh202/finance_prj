"""HTTP client for the AURORA backend — the frontend's ONLY data-access path.

No module under frontend/ may import from backend/ (or `src`); everything
goes through these wrappers over the FastAPI service. Streamlit-free on
purpose so it can be exercised from a plain Python shell.

The backend persists no portfolio: this app owns its own (see store.py) and
ships it in the body of every engine call. That is invisible to the views —
the function signatures below are unchanged — but it is why endpoints that
read like queries are POSTs.

Backend URL: AURORA_API_URL env var, default http://localhost:8000.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

import store

BASE_URL = os.environ.get("AURORA_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120  # engine endpoints may fetch 2y of history on a cold cache

HOLDING_COLUMNS = ["symbol", "name", "shares", "buy_price"]

# Marker details the backend uses for expected conditions (routers/_common.py)
EMPTY_PORTFOLIO = "empty_portfolio"
NO_HISTORY = "no_history"
NO_MODEL = "no_model"           # risk model artifact not built yet (routers/risk.py)

_session = requests.Session()


class ApiUnavailable(Exception):
    """The backend is not reachable — start it with scripts/dev.ps1."""


class ApiMarker(Exception):
    """An expected backend condition (EMPTY_PORTFOLIO / NO_HISTORY)."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(detail)


def _request(method: str, path: str, **kwargs):
    try:
        resp = _session.request(method, BASE_URL + path, timeout=TIMEOUT, **kwargs)
    except requests.ConnectionError as exc:
        raise ApiUnavailable(str(exc)) from exc
    if resp.status_code in (409, 502, 503):
        detail = resp.json().get("detail", "")
        if detail in (EMPTY_PORTFOLIO, NO_HISTORY, NO_MODEL):
            raise ApiMarker(detail)
    resp.raise_for_status()
    return resp.json()


def _get(path: str, **params):
    return _request("GET", path, params=params or None)


def _post(path: str, body: Optional[dict] = None):
    return _request("POST", path, json=body)


# ------------------------------------------------------------- converters

def _records_df(records: List[dict], columns: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty and columns:
        df = pd.DataFrame(columns=columns)
    return df


def _split_df(split: dict, datetime_index: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(
        split.get("data", []), index=split.get("index", []), columns=split.get("columns", [])
    )
    if datetime_index and not df.empty:
        df.index = pd.to_datetime(df.index)
    return df


def _df_records(df: pd.DataFrame) -> List[dict]:
    # to_json round-trip keeps NaN JSON-safe (-> null)
    return json.loads(df.to_json(orient="records"))


def _body(**extra) -> dict:
    """This client's portfolio, as the body every engine endpoint expects."""
    payload = {
        "holdings": _df_records(store.load_holdings()),
        "cash": store.load_cash(),
    }
    payload.update(extra)
    return payload


# ------------------------------------------------------------------ market

def get_universe() -> pd.DataFrame:
    return _records_df(_get("/market/universe"), ["symbol", "name", "exchange", "etf"])


def get_prices(symbols: List[str]) -> Dict[str, float]:
    if not symbols:
        return {}
    return _get("/market/prices", symbols=",".join(symbols))


def get_history(symbols: List[str], period: str = "1y") -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    return _split_df(
        _get("/market/history", symbols=",".join(symbols), period=period),
        datetime_index=True,
    )


# --------------------------------------------------------------- portfolio

def get_portfolio() -> pd.DataFrame:
    return store.load_holdings()


def put_portfolio(holdings: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Canonicalise server-side (repair rows, merge duplicates), then store."""
    data = _post("/portfolio/normalize", {"holdings": _df_records(holdings)})
    clean = _records_df(data["holdings"], HOLDING_COLUMNS)
    store.save_holdings(clean)
    return clean, data["problems"]


def add_holding(symbol: str, name: str, shares: float, buy_price: float) -> pd.DataFrame:
    # Appending then normalizing is how a repeat buy merges into the existing
    # lot at a weighted-average cost.
    rows = _df_records(store.load_holdings())
    rows.append({"symbol": symbol, "name": name, "shares": shares, "buy_price": buy_price})
    clean, _ = put_portfolio(pd.DataFrame(rows))
    return clean


def parse_csv(text: str) -> Tuple[pd.DataFrame, List[str]]:
    """Parse only — nothing is stored until put_portfolio() is called."""
    data = _post("/portfolio/parse-csv", {"csv": text})
    return _records_df(data["holdings"], HOLDING_COLUMNS), data["problems"]


def load_sample() -> Tuple[pd.DataFrame, float]:
    data = _get("/portfolio/sample")
    holdings = _records_df(data["holdings"], HOLDING_COLUMNS)
    store.save_holdings(holdings)
    store.save_cash(data["cash"])
    return holdings, data["cash"]


def get_cash() -> float:
    return store.load_cash()


def put_cash(cash: float) -> None:
    store.save_cash(cash)


def get_view() -> Tuple[pd.DataFrame, Dict[str, float]]:
    data = _post("/portfolio/view", _body())
    return _records_df(data["view"]), data["totals"]


# ----------------------------------------------------------------- engines

def health_report() -> dict:
    report = _post("/health/report", _body())
    if report.get("correlation") is not None:
        report["correlation"] = _split_df(report["correlation"])
    return report


def regime() -> dict:
    return _post("/strategy/regime", _body())


def signals() -> List[dict]:
    return _post("/strategy/signals", _body())


def backtest() -> pd.DataFrame:
    return _split_df(_post("/strategy/backtest", _body()), datetime_index=True)


def strategy_recommendations(universe: Optional[List[str]] = None) -> List[dict]:
    query = "?universe=" + ",".join(universe) if universe else ""
    return _post("/strategy/recommendations" + query, _body())


def essential_news(max_events: int = 5, symbols: Optional[List[str]] = None) -> List[dict]:
    """`symbols=None` keeps the old default of tracking the stored portfolio;
    the backend has no portfolio to fall back to, so it is resolved here."""
    if symbols is None:
        symbols = sorted(set(store.load_holdings()["symbol"]))
    return _get("/news/essential", max_events=max_events, symbols=",".join(symbols))


def news_feeds() -> List[str]:
    return _get("/news/feeds")["feeds"]


def recommend_daily() -> dict:
    return _post("/recommendation/daily", _body())


def recommendation_events(max_events: int = 5) -> dict:
    return _post(f"/recommendation/events?max_events={max_events}", _body())


def react(event: dict) -> dict:
    return _post("/recommendation/react", _body(event=event))


def risk_estimates(horizon: int = 5) -> List[dict]:
    return _post(f"/risk/estimates?horizon={horizon}", _body())


def risk_portfolio(horizon: int = 5) -> dict:
    return _post(f"/risk/portfolio?horizon={horizon}", _body())
