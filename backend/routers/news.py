"""Engine 3 — Event Intelligence (Developer 3)."""

from typing import Optional

from fastapi import APIRouter, HTTPException

from serialize import as_dict
from src import portfolio as pf
from src.news_intelligence import collector, engine

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


@router.post("/collect")
def collect(symbols: Optional[str] = None) -> dict:
    """Run the RSS collector and return its run statistics.

    This is the in-process equivalent of
    `python backend/src/news_intelligence/collector.py` — same entry point,
    same default symbol list, same `data/news_raw.json` store — exposed so
    the frontend's "Refresh news" button can trigger it. Calling the module
    directly rather than spawning an interpreter keeps the run inside this
    process's DATA_DIR and doesn't depend on which `python` is on PATH.

    `symbols`, if given, is a comma-separated ticker list that overrides the
    default (the script's positional arguments); omitting it collects for the
    saved portfolio, as running the script bare does.

    Defined with `def`, so FastAPI runs this blocking call in its threadpool
    rather than on the event loop — a full run is a few seconds of network.
    """
    if symbols is None:
        selected = collector.default_symbols()
    else:
        selected = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    try:
        stats = collector.collect(selected)
    except Exception as exc:                                   # noqa: BLE001
        # Per-feed network/parse failures are already swallowed inside
        # collect_feeds() and reported as `feeds_failed`; reaching here means
        # the run itself broke (e.g. the store could not be written).
        raise HTTPException(
            status_code=502, detail=f"News collection failed: {exc}"
        ) from exc
    return {"symbols": selected, "stats": stats}
