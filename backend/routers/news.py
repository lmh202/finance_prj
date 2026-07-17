"""Engine 3 — Event Intelligence (Developer 3). Engine is still a stub."""

from fastapi import APIRouter

from serialize import as_dict
from src import portfolio as pf
from src.news_intelligence import engine

router = APIRouter(prefix="/news", tags=["news"])


@router.get("/essential")
def essential(max_events: int = 5) -> list:
    holdings = pf.load_portfolio()
    events = engine.essential_news(list(holdings["symbol"]), max_events=max_events)
    return [as_dict(e) for e in events]


@router.get("/feeds")
def feeds() -> dict:
    return {"feeds": engine.DEFAULT_FEEDS}
