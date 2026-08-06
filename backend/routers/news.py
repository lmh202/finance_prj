"""Engine 3 — Event Intelligence (Developer 3)."""

from typing import Optional

from fastapi import APIRouter

from serialize import as_dict
from src.news_intelligence import engine

router = APIRouter(prefix="/news", tags=["news"])


@router.get("/essential")
def essential(max_events: int = 5, symbols: Optional[str] = None) -> list:
    """`symbols` is a comma-separated ticker list scoping the search.

    The backend holds no portfolio, so the caller always decides which
    tickers it cares about; omitting `symbols` (or passing it empty) means
    "no tickers selected" and yields general market news only.
    """
    selected = [s.strip().upper() for s in (symbols or "").split(",") if s.strip()]
    events = engine.essential_news(selected, max_events=max_events)
    return [as_dict(e) for e in events]


@router.get("/feeds")
def feeds() -> dict:
    return {"feeds": engine.DEFAULT_FEEDS}
