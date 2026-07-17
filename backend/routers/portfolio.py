"""Portfolio persistence, valuation and CSV import."""

import io
from typing import Dict, List

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from serialize import df_records
from src import data_loader
from src import portfolio as pf

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

SAMPLE_CASH = 23000.0


class HoldingIn(BaseModel):
    symbol: str
    name: str = ""
    shares: float
    buy_price: float


class HoldingsIn(BaseModel):
    holdings: List[Dict]


class CashIn(BaseModel):
    cash: float


class CsvIn(BaseModel):
    csv: str


def _frame(rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pf.empty_portfolio()


@router.get("")
def get_portfolio() -> dict:
    return {"holdings": df_records(pf.load_portfolio())}


@router.put("")
def put_portfolio(body: HoldingsIn) -> dict:
    clean, problems = pf.normalize_holdings(_frame(body.holdings))
    pf.save_portfolio(clean)
    return {"holdings": df_records(clean), "problems": problems}


@router.post("/holdings")
def add_holding(body: HoldingIn) -> dict:
    merged = pf.add_holding(
        pf.load_portfolio(), body.symbol, body.name, body.shares, body.buy_price
    )
    pf.save_portfolio(merged)
    return {"holdings": df_records(merged)}


@router.post("/parse-csv")
def parse_csv(body: CsvIn) -> dict:
    """Parse only — nothing is saved until the client PUTs the result back."""
    holdings, problems = pf.parse_uploaded_csv(io.StringIO(body.csv))
    return {"holdings": df_records(holdings), "problems": problems}


@router.post("/load-sample")
def load_sample() -> dict:
    sample = pf.load_portfolio(pf.SAMPLE_PORTFOLIO_CSV)
    pf.save_portfolio(sample)
    pf.save_cash(SAMPLE_CASH)
    return {"holdings": df_records(sample), "cash": SAMPLE_CASH}


@router.get("/cash")
def get_cash() -> dict:
    return {"cash": pf.load_cash()}


@router.put("/cash")
def put_cash(body: CashIn) -> dict:
    pf.save_cash(body.cash)
    return {"cash": body.cash}


@router.get("/view")
def get_view() -> dict:
    """Holdings joined with live prices: values, P/L, weights, totals."""
    holdings = pf.load_portfolio()
    cash = pf.load_cash()
    prices = data_loader.get_latest_prices(list(holdings["symbol"])) if not holdings.empty else {}
    view, totals = pf.build_view(holdings, prices, cash)
    return {"view": df_records(view), "totals": totals}
