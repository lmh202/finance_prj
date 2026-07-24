"""Engine 3 — Event Intelligence (Developer 3)."""

from typing import Optional

from fastapi import APIRouter

from serialize import as_dict
from src import portfolio as pf
from src.news_intelligence import engine

router = APIRouter(prefix="/news", tags=["news"])


@router.get("/essential")
def essential(max_events: int = 5, symbols: Optional[str] = None) -> list:
    """`symbols`, if given, is a comma-separated ticker list that OVERRIDES the
    portfolio (the frontend's ticker editor sends this; it may be empty —
    that means "no tickers selected", not "fall back to the portfolio").
    Omitting it entirely keeps the old default: derive tickers from the
    saved portfolio.
    """
    if symbols is None:
        selected = list(pf.load_portfolio()["symbol"])
    else:
        selected = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    events = engine.essential_news(selected, max_events=max_events)
    return [as_dict(e) for e in events]


@router.get("/feeds")
def feeds() -> dict:
    return {"feeds": engine.DEFAULT_FEEDS}
