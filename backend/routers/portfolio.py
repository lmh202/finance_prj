"""Stateless portfolio helpers: canonicalisation, CSV import, valuation.

Nothing here persists anything. The client owns its portfolio (localStorage
in the Next.js app, a local file in the Streamlit app) and posts it in for
the operations that need market data or the shared normalisation rules —
duplicate-symbol merging, row repair, weighted-average cost basis — which
must not be reimplemented per frontend.
"""

import io
from typing import Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

from serialize import df_records
from src import data_loader
from src import portfolio as pf

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

SAMPLE_CASH = 23000.0


class HoldingsIn(BaseModel):
    holdings: List[Dict] = []


class ValuationIn(BaseModel):
    holdings: List[Dict] = []
    cash: float = 0.0


class CsvIn(BaseModel):
    csv: str


@router.post("/normalize")
def normalize(body: HoldingsIn) -> dict:
    """Canonicalise edited rows: repair, drop bad ones, merge duplicate symbols.

    Replaces the old `PUT /portfolio` + `POST /portfolio/holdings` pair — the
    client sends whatever its editor holds and stores the cleaned result.
    """
    clean, problems = pf.holdings_from_records(body.holdings)
    return {"holdings": df_records(clean), "problems": problems}


@router.post("/parse-csv")
def parse_csv(body: CsvIn) -> dict:
    """Parse an uploaded portfolio CSV. The client stores the result itself."""
    holdings, problems = pf.parse_uploaded_csv(io.StringIO(body.csv))
    return {"holdings": df_records(holdings), "problems": problems}


@router.get("/sample")
def sample() -> dict:
    """The committed demo portfolio, for a client with nothing saved yet."""
    holdings = pf.load_portfolio_file(pf.SAMPLE_PORTFOLIO_CSV)
    return {"holdings": df_records(holdings), "cash": SAMPLE_CASH}


@router.post("/view")
def view(body: ValuationIn) -> dict:
    """Holdings joined with live prices: values, P/L, weights, totals."""
    holdings, _ = pf.holdings_from_records(body.holdings)
    prices = data_loader.get_latest_prices(list(holdings["symbol"])) if not holdings.empty else {}
    rows, totals = pf.build_view(holdings, prices, body.cash)
    return {"view": df_records(rows), "totals": totals}
