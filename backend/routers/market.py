"""Shared-kernel market data: ticker universe + latest prices + history."""

import re

from fastapi import APIRouter, Query

from serialize import df_records, df_split
from src import data_loader

router = APIRouter(prefix="/market", tags=["market"])

# Sector inference: industry keyword → GICS sector
_INDUSTRY_SECTORS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"semiconductor|software|technology|computer|internet|cloud|cybersec|telecom equipment", re.I), "Technology"),
    (re.compile(r"bank|finance|financial|investment|insurance|capital|broker|credit|mortgage", re.I), "Financials"),
    (re.compile(r"biotech|pharma|health|medical|diagnostic|life science|drug", re.I), "Health Care"),
    (re.compile(r"oil|gas|energy|pipeline|drilling|coal|refining", re.I), "Energy"),
    (re.compile(r"food|beverage|tobacco|household|personal products|staples", re.I), "Consumer Staples"),
    (re.compile(r"retail|restaurant|auto|leisure|apparel|consumer|hotel|casino|travel|commerce|footwear|home improvement", re.I), "Consumer Discretionary"),
    (re.compile(r"aerospace|industrial|transport|construct|machinery|logistic|railroad|defense|freight|engineering", re.I), "Industrials"),
    (re.compile(r"telecom|media|advertis|entertainment|broadcast|gaming|publish", re.I), "Communication Services"),
    (re.compile(r"utilit|electric|water", re.I), "Utilities"),
    (re.compile(r"real estate|reit", re.I), "Real Estate"),
    (re.compile(r"chemical|mining|metal|paper|material|steel|gold|silver|lumber|aluminum", re.I), "Materials"),
]


def _clean_name(name: str) -> str:
    """Strip verbose legal suffixes from company names."""
    name = re.sub(
        r"\s+(common stock|common shares|ordinary shares|american depositary shares|"
        r"ads|adr|preferred stock|units?|warrants?|shares|class [a-z] common stock)\.?$",
        "",
        name,
        flags=re.I,
    ).strip().rstrip("-").strip()
    return name[:90]


def _infer_sector(name: str, is_etf: bool) -> str:
    if is_etf:
        return "ETF"
    for pattern, sector in _INDUSTRY_SECTORS:
        if pattern.search(name):
            return sector
    return "US Equity"


def _rank_key(row, q: str):
    """Score a row for sorting: higher = better match."""
    sym = row["symbol"].upper()
    name = str(row.get("name", "")).upper()
    qq = q.upper()
    score = 0
    if sym == qq:
        score += 1000
    elif sym.startswith(qq):
        score += 500 - len(sym)
    elif name.startswith(qq):
        score += 250
    elif qq in name:
        score += 120
    elif qq in sym:
        score += 60
    return score


@router.get("/universe")
def universe(force_refresh: bool = False) -> list:
    return df_records(data_loader.load_ticker_universe(force_refresh))


@router.get("/prices")
def prices(symbols: str) -> dict:
    """Latest close per symbol. `symbols` is comma-separated; a symbol that
    could not be priced is absent from the response (contract in
    src/data_loader.get_latest_prices)."""
    wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    return data_loader.get_latest_prices(wanted)


@router.get("/search")
def search(q: str = Query(..., min_length=1, max_length=40), limit: int = Query(10, ge=1, le=20)):
    """Search the ticker universe by symbol or name substring.

    Returns ranked results with symbol, name, exchange, sector, and quoteType.
    Falls back to live Nasdaq / Yahoo search when local results are thin.
    """
    q = re.sub(r"[%_\\\"']", "", q).strip()
    if not q:
        return []

    df = data_loader.load_ticker_universe()

    # Filter: symbol or name contains query (case-insensitive)
    q_lower = q.lower()
    mask = (
        df["symbol"].str.lower().str.contains(q_lower, na=False, regex=False)
        | df["name"].str.lower().str.contains(q_lower, na=False, regex=False)
    )
    candidates = df[mask].copy()

    # Rank and take top N
    if not candidates.empty:
        candidates["_score"] = candidates.apply(lambda r: _rank_key(r, q), axis=1)
        candidates = candidates.sort_values("_score", ascending=False)
        candidates = candidates.head(limit)

    results: list[dict] = []
    for _, row in candidates.iterrows():
        name = _clean_name(str(row.get("name", row["symbol"])))
        is_etf = bool(row.get("is_etf", False))
        results.append({
            "symbol": str(row["symbol"]),
            "name": name if name else str(row["symbol"]),
            "exchange": str(row.get("exchange", "")),
            "sector": _infer_sector(name, is_etf),
            "quoteType": "ETF" if is_etf else "EQUITY",
        })

    return results

@router.get("/history")
def history(symbols: str, period: str = "2y") -> dict:
    """Daily adjusted-close history. `symbols` is comma-separated; a symbol's
    column is absent if it could not be fetched (contract in
    src/data_loader.get_history)."""
    wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    return df_split(data_loader.get_history(wanted, period=period))
