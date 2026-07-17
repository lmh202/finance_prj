"""Shared request-time plumbing for the engine routers.

Every engine used to do the same dance inside its Streamlit page:
load holdings -> bail if empty -> fetch history incl. benchmark -> bail if
empty. That dance now lives here, server-side; the frontend translates the
two marker details back into its st.info / st.error messages.
"""

from typing import Tuple

import pandas as pd
from fastapi import HTTPException

from src import data_loader
from src import portfolio as pf
from src.interfaces import BENCHMARK

EMPTY_PORTFOLIO = "empty_portfolio"
NO_HISTORY = "no_history"


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
