"""Engine 1 — Portfolio Health (Developer 1)."""

from fastapi import APIRouter

from routers._common import load_holdings_history
from serialize import as_dict
from src.portfolio_health import engine

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/report")
def report() -> dict:
    holdings, history = load_holdings_history()
    return as_dict(engine.compute_health(holdings, history))
