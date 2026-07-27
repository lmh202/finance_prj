"""Shared request-time plumbing for the engine routers.

Every engine used to do the same dance inside its Streamlit page:
load holdings -> bail if empty -> fetch history incl. benchmark -> bail if
empty. That dance now lives here, server-side; the frontend translates the
two marker details back into its st.info / st.error messages.
"""

import time
from typing import Iterable, Optional, Tuple

import pandas as pd
from fastapi import HTTPException

from src import data_loader
from src import portfolio as pf
from src.interfaces import BENCHMARK

EMPTY_PORTFOLIO = "empty_portfolio"
NO_HISTORY = "no_history"

# The market-stress signal needs a 60-session rolling volatility PLUS 252-504
# observations of that volatility to rank it — roughly 312 sessions minimum and
# 564 for a full reference window. `load_holdings_history()` deliberately stays
# at two years because four engines consume that frame and widening it would
# silently change all of them, so the benchmark is fetched separately.
BENCHMARK_HISTORY_PERIOD = "5y"
BENCHMARK_CACHE_MAX_AGE_SECONDS = 6 * 3600
_BENCHMARK_CACHE: Optional[Tuple[float, pd.Series]] = None


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


def load_benchmark_close() -> Optional[pd.Series]:
    """Long benchmark close series for the causal market-stress signal.

    Deliberately separate from `load_holdings_history()`: that frame is two
    years and is consumed by score_assets, compute_health, classify_regime
    (whose volatility median spans the whole frame) and the decision layer's
    runtime features. Widening it to satisfy one signal would silently move
    four engines, so this pays for one extra single-ticker download instead.

    Memoised in-process for BENCHMARK_CACHE_MAX_AGE_SECONDS — a daily bar
    changes once per session. Returns None on any failure: a missing stress
    signal must degrade the risk budget to its conservative default, never
    fail the recommendation.
    """
    global _BENCHMARK_CACHE

    now = time.monotonic()
    cached = _BENCHMARK_CACHE
    if cached is not None and now - cached[0] < BENCHMARK_CACHE_MAX_AGE_SECONDS:
        return cached[1]

    try:
        frame = data_loader.get_history(
            [BENCHMARK],
            period=BENCHMARK_HISTORY_PERIOD,
        )
        if frame is None or frame.empty or BENCHMARK not in frame.columns:
            return cached[1] if cached is not None else None
        close = frame[BENCHMARK].dropna()
        if close.empty:
            return cached[1] if cached is not None else None
    except Exception:                                          # noqa: BLE001
        return cached[1] if cached is not None else None

    _BENCHMARK_CACHE = (now, close)
    return close
