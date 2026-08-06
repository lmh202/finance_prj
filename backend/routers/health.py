"""Engine 1 — Portfolio Health (Developer 1)."""

from fastapi import APIRouter

from routers._common import PortfolioIn, holdings_history
from serialize import as_dict
from src.portfolio_health import engine

router = APIRouter(prefix="/health", tags=["health"])


@router.post("/report")
def report(body: PortfolioIn) -> dict:
    holdings, history = holdings_history(body)
    return as_dict(engine.compute_health(holdings, history))
