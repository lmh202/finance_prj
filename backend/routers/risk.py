"""Risk Engine — per-stock + portfolio downside risk (HAR forecast + FHS).

Serves the model trained offline (data/processed/risk_model.json). Markers:
  409 empty_portfolio · 502 no_history · 503 no_model (artifact not built yet).
"""

import pandas as pd
from fastapi import APIRouter, HTTPException

from routers._common import NO_HISTORY, PortfolioIn, refresh_news_store, require_holdings
from serialize import as_dict
from src import data_loader
from src import portfolio as pf
from src.risk_engine import engine

router = APIRouter(prefix="/risk", tags=["risk"])
NO_MODEL = "no_model"


def _ohlc_and_weights(body: PortfolioIn):
    holdings = require_holdings(body)                # 409 if empty
    if not engine.model_available():
        raise HTTPException(status_code=503, detail=NO_MODEL)
    symbols = sorted(set(holdings["symbol"]))
    # The formal five-session model is HAR-X + News: it reads live RSS
    # attention from data/news_raw.json. Refresh before estimating so these
    # endpoints report the same news state the recommendation path uses.
    refresh_news_store(symbols)
    ohlc = data_loader.get_ohlc_history(symbols)
    if not ohlc:
        raise HTTPException(status_code=502, detail=NO_HISTORY)
    prices = data_loader.get_latest_prices(symbols)
    view, _ = pf.build_view(holdings, prices)
    weights = {s: float(w) for s, w in zip(view["symbol"], view["weight_pct"])
               if pd.notna(w) and w > 0}
    return ohlc, weights


@router.post("/estimates")
def estimates(body: PortfolioIn, horizon: int = 0) -> list:
    """Per-stock RiskEstimate for the held symbols. horizon=5|20, or 0 for both."""
    ohlc, _ = _ohlc_and_weights(body)
    hz = (horizon,) if horizon in (5, 20) else engine.HORIZONS
    return [as_dict(r) for r in engine.risk_estimates(ohlc, horizons=hz)]


@router.post("/portfolio")
def portfolio(body: PortfolioIn, horizon: int = 5) -> dict:
    """Aggregate PortfolioRisk (EWMA-correlation VaR/ES) for the held portfolio."""
    ohlc, weights = _ohlc_and_weights(body)
    h = horizon if horizon in (5, 20) else 5
    pr = engine.portfolio_risk(ohlc, weights, h)
    if pr is None:
        raise HTTPException(status_code=502, detail=NO_HISTORY)
    return as_dict(pr)
