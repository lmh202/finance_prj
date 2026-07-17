"""Engine 2 — Daily Strategy: regime, asset ranking, backtest (Developer 2)."""

from fastapi import APIRouter

from routers._common import load_holdings_history
from serialize import as_dict, df_split
from src.daily_strategy import engine

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
