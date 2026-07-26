"""Shared request-time plumbing for the engine routers.

Every engine used to do the same dance inside its Streamlit page:
load holdings -> bail if empty -> fetch history incl. benchmark -> bail if
empty. That dance now lives here, server-side; the frontend translates the
two marker details back into its st.info / st.error messages.
"""

from typing import Iterable, Tuple

import pandas as pd
from fastapi import HTTPException

from src import data_loader
from src import portfolio as pf
from src.interfaces import BENCHMARK

EMPTY_PORTFOLIO = "empty_portfolio"
NO_HISTORY = "no_history"


def refresh_news_store(symbols: Iterable[str]) -> dict:
    """Pull current RSS into data/news_raw.json before risk is computed.

    The risk engine derives its causal news-attention input (`log_count`) from
    that store whenever a caller does not pass news features explicitly. If the
    store is refreshed *after* the risk call, the newest headlines only reach
    the model on the following request — and on a cold start the overlay never
    applies at all. Refreshing here keeps HAR-X + News reading the same
    headlines the rest of the response is built from.

    The collector already throttles per feed (FEED_STALE_MINUTES), so calling
    this on every request costs nothing when the store is warm. Network or
    parse failures are non-fatal: the risk engine degrades to its documented
    `stale_store` / `missing_store` path and reports it in `news_quality`.
    """
    from src.news_intelligence import collector

    try:
        return collector.collect(sorted({str(s) for s in symbols}))
    except Exception as exc:                                   # noqa: BLE001
        return {"error": type(exc).__name__}


def load_holdings() -> pd.DataFrame:
    holdings = pf.load_portfolio()
    if holdings.empty:
        raise HTTPException(status_code=409, detail=EMPTY_PORTFOLIO)
    return holdings


def load_holdings_history() -> Tuple[pd.DataFrame, pd.DataFrame]:
    holdings = load_holdings()
    symbols = sorted(set(holdings["symbol"])) + [BENCHMARK]
    history = data_loader.get_history(symbols)
    if history.empty:
        raise HTTPException(status_code=502, detail=NO_HISTORY)
    return holdings, history
