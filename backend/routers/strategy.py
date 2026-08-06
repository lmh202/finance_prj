"""Engine 2 — Daily Strategy: regime, asset ranking, backtest (Developer 2)."""

from fastapi import APIRouter, HTTPException

from routers._common import NO_HISTORY, PortfolioIn, holdings_history, require_holdings
from serialize import as_dict, df_split
from src import data_loader
from src.daily_strategy import engine
from src.interfaces import BENCHMARK

router = APIRouter(prefix="/strategy", tags=["strategy"])


@router.post("/regime")
def regime(body: PortfolioIn) -> dict:
    _, history = holdings_history(body)
    return as_dict(engine.classify_regime(history))


@router.post("/signals")
def signals(body: PortfolioIn) -> list:
    holdings, history = holdings_history(body)
    return [as_dict(s) for s in engine.score_assets(history, holdings)]


@router.post("/backtest")
def backtest(body: PortfolioIn) -> dict:
    holdings, history = holdings_history(body)
    return df_split(engine.backtest(history, holdings))


@router.post("/recommendations")
def recommendations(body: PortfolioIn, universe: str = "") -> list:
    """Confluence (EMA/MACD/RSI) recommendations cross-referenced with holdings.

    Scans holdings PLUS a candidate universe (the `universe` query param, else
    engine.STRATEGY_WATCHLIST) so new-position BUYs can surface. 409 if the
    portfolio is empty; 502 if price history is unavailable.
    """
    holdings = require_holdings(body)
    extra = [s.strip().upper() for s in universe.split(",") if s.strip()]
    scan = sorted(set(holdings["symbol"]) | set(extra or engine.STRATEGY_WATCHLIST))
    history = data_loader.get_history(scan + [BENCHMARK])
    if history.empty:
        raise HTTPException(status_code=502, detail=NO_HISTORY)
    prices = data_loader.get_latest_prices(scan)
    recs = engine.recommend_signals(history, holdings, prices, universe=scan)
    return [as_dict(r) for r in recs]
