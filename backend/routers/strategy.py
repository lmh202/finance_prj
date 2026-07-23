"""Engine 2 — Daily Strategy: regime, asset ranking, backtest (Developer 2)."""

from fastapi import APIRouter, HTTPException

from routers._common import NO_HISTORY, load_holdings, load_holdings_history
from serialize import as_dict, df_split
from src import data_loader
from src.daily_strategy import engine
from src.interfaces import BENCHMARK

router = APIRouter(prefix="/strategy", tags=["strategy"])


@router.get("/regime")
def regime() -> dict:
    _, history = load_holdings_history()
    return as_dict(engine.classify_regime(history))


@router.get("/signals")
def signals() -> list:
    holdings, history = load_holdings_history()
    return [as_dict(s) for s in engine.score_assets(history, holdings)]


@router.get("/backtest")
def backtest() -> dict:
    holdings, history = load_holdings_history()
    return df_split(engine.backtest(history, holdings))


@router.get("/recommendations")
def recommendations(universe: str = "") -> list:
    """Confluence (EMA/MACD/RSI) recommendations cross-referenced with holdings.

    Scans holdings PLUS a candidate universe (the `universe` query param, else
    engine.STRATEGY_WATCHLIST) so new-position BUYs can surface. 409 if the
    portfolio is empty; 502 if price history is unavailable.
    """
    holdings = load_holdings()
    extra = [s.strip().upper() for s in universe.split(",") if s.strip()]
    scan = sorted(set(holdings["symbol"]) | set(extra or engine.STRATEGY_WATCHLIST))
    history = data_loader.get_history(scan + [BENCHMARK])
    if history.empty:
        raise HTTPException(status_code=502, detail=NO_HISTORY)
    prices = data_loader.get_latest_prices(scan)
    recs = engine.recommend_signals(history, holdings, prices, universe=scan)
    return [as_dict(r) for r in recs]
