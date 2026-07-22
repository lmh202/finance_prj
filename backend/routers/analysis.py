"""Exploratory analysis endpoint — takes arbitrary holdings and returns full
risk/performance analytics without depending on a saved portfolio.

This is the backend counterpart of what the frontendjs POST /api/analyze used
to do locally with its own PostgreSQL and curl-based price fetching.
"""

from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src import data_loader
from src.analysis import engine as analyze

router = APIRouter(prefix="/analysis", tags=["analysis"])

MAX_HOLDINGS = 24
BENCHMARKS = [
    {"symbol": "SPY", "name": "S&P 500", "color": "#8DA2FB"},
    {"symbol": "QQQ", "name": "NASDAQ 100", "color": "#CF9FFF"},
    {"symbol": "IWM", "name": "Russell 2000", "color": "#FCD34D"},
    {"symbol": "EFA", "name": "MSCI EAFE", "color": "#5EEAD4"},
    {"symbol": "EEM", "name": "MSCI Emerging Mkts", "color": "#FB923C"},
]

# Sector inference — same regex map as routers/market.py
import re as _re

_INDUSTRY_SECTORS: list[tuple[_re.Pattern, str]] = [
    (_re.compile(r"semiconductor|software|technology|computer|internet|cloud|cybersec|telecom equipment", _re.I), "Technology"),
    (_re.compile(r"bank|finance|financial|investment|insurance|capital|broker|credit|mortgage", _re.I), "Financials"),
    (_re.compile(r"biotech|pharma|health|medical|diagnostic|life science|drug", _re.I), "Health Care"),
    (_re.compile(r"oil|gas|energy|pipeline|drilling|coal|refining", _re.I), "Energy"),
    (_re.compile(r"food|beverage|tobacco|household|personal products|staples", _re.I), "Consumer Staples"),
    (_re.compile(r"retail|restaurant|auto|leisure|apparel|consumer|hotel|casino|travel|commerce|footwear|home improvement", _re.I), "Consumer Discretionary"),
    (_re.compile(r"aerospace|industrial|transport|construct|machinery|logistic|railroad|defense|freight|engineering", _re.I), "Industrials"),
    (_re.compile(r"telecom|media|advertis|entertainment|broadcast|gaming|publish", _re.I), "Communication Services"),
    (_re.compile(r"utilit|electric|water", _re.I), "Utilities"),
    (_re.compile(r"real estate|reit", _re.I), "Real Estate"),
    (_re.compile(r"chemical|mining|metal|paper|material|steel|gold|silver|lumber|aluminum", _re.I), "Materials"),
]


class HoldingIn(BaseModel):
    symbol: str
    value: float


class ExploreRequest(BaseModel):
    holdings: List[HoldingIn]
    mode: str = "weight"  # "weight" | "shares"


def _normalize_symbol(raw: str) -> str:
    return raw.strip().upper().replace(".", "-")[:14]


def _infer_sector(name: str, is_etf: bool) -> str:
    if is_etf:
        return "ETF"
    for pattern, sector in _INDUSTRY_SECTORS:
        if pattern.search(name):
            return sector
    return "US Equity"


def _resolve_name(symbol: str) -> tuple[str, bool, str]:
    """Best-effort name + sector resolution from the ticker universe."""
    try:
        df = data_loader.load_ticker_universe()
        match = df[df["symbol"].str.upper() == symbol.upper()]
        if not match.empty:
            row = match.iloc[0]
            name = str(row.get("name", symbol))
            is_etf = bool(row.get("is_etf", False))
            sector = _infer_sector(name, is_etf)
            return name, is_etf, sector
    except Exception:
        pass
    return symbol, False, "Other"


@router.post("/explore")
def explore(body: ExploreRequest) -> dict:
    """Analyze an arbitrary portfolio of holdings.

    Takes a list of {symbol, value} pairs and an optional mode
    ("weight" or "shares"), fetches up to 5 years of daily price history,
    builds a daily-rebalanced constant-mix portfolio, and returns full
    risk/performance analytics benchmarked against SPY and QQQ.
    """
    mode = body.mode if body.mode in ("weight", "shares") else "weight"
    raw_holdings = body.holdings

    # --- validate & deduplicate ---
    merged: dict[str, float] = {}
    for h in raw_holdings:
        if not h.symbol or not isinstance(h.value, (int, float)):
            continue
        if not float(h.value) > 0:
            continue
        s = _normalize_symbol(h.symbol)
        if not s:
            continue
        merged[s] = merged.get(s, 0.0) + float(h.value)

    symbols = list(merged.keys())[:MAX_HOLDINGS]
    if not symbols:
        raise HTTPException(status_code=400, detail="No valid holdings provided")

    # --- fetch price history (5y) ---
    all_symbols = list(dict.fromkeys(symbols + [b["symbol"] for b in BENCHMARKS]))
    history = data_loader.get_history(all_symbols, period="5y")

    # Filter to symbols that actually have data (min 20 bars)
    usable = [s for s in symbols if s in history.columns and history[s].dropna().shape[0] >= 20]
    if not usable:
        raise HTTPException(
            status_code=422,
            detail="No usable price history found for these symbols",
        )

    # --- resolve latest prices for weight computation ---
    latest_prices: dict[str, Optional[float]] = {}
    for s in usable:
        col = history[s].dropna()
        latest_prices[s] = float(col.iloc[-1]) if len(col) > 0 else None

    # --- resolve weights ---
    raw_sum = 0.0
    raw_weights: dict[str, float] = {}
    for s in usable:
        inp = merged[s]
        raw = inp * (latest_prices[s] or 0.0) if mode == "shares" else inp
        raw_weights[s] = raw
        raw_sum += raw

    if not raw_sum > 0:
        raise HTTPException(status_code=422, detail="Could not resolve position values")

    weights = {s: raw_weights[s] / raw_sum for s in usable}

    # --- prepare aligned input ---
    align_input: dict[str, pd.DataFrame] = {}
    for s in usable:
        col = history[s].dropna()
        align_input[s] = col.to_frame("close")
    for b in BENCHMARKS:
        if b["symbol"] in history.columns:
            col = history[b["symbol"]].dropna()
            if len(col) >= 20:
                align_input[b["symbol"]] = col.to_frame("close")

    aligned = analyze.align_series(align_input)
    if not aligned or len(aligned["dates"]) < 30:
        raise HTTPException(
            status_code=422,
            detail="Insufficient overlapping history across holdings",
        )

    dates = aligned["dates"]
    values = aligned["values"]
    n_days = len(dates)

    # --- portfolio + benchmark curves ---
    portfolio = analyze.build_portfolio_index(usable, weights, values, n_days)
    portfolio_metrics_base = analyze.compute_stats(portfolio, dates)

    benchmarks = []
    for b in BENCHMARKS:
        if b["symbol"] not in values:
            continue
        idx = analyze.index_series(values[b["symbol"]])
        benchmarks.append({
            "symbol": b["symbol"],
            "name": b["name"],
            "color": b["color"],
            "values": analyze.round_series(idx),
            "stats": analyze.compute_stats(idx, dates),
        })

    # --- beta / alpha vs SPY ---
    spy_bench = next((b for b in benchmarks if b["symbol"] == "SPY"), None)
    ba = None
    if spy_bench and "SPY" in values:
        ba = analyze.beta_alpha(portfolio, analyze.index_series(values["SPY"]))

    # --- per-holding analytics ---
    holdings_out = []
    for s in usable:
        idx = analyze.index_series(values[s])
        stats = analyze.compute_stats(values[s], dates)
        name, is_etf, sector = _resolve_name(s)
        holdings_out.append({
            "symbol": s,
            "name": name,
            "sector": sector,
            "weight": analyze.round_n(weights[s], 4),
            "inputValue": analyze.round_n(merged[s], 4),
            "lastPrice": latest_prices[s],
            "simulated": False,
            "stats": stats,
            "contribution": analyze.round_n(weights[s] * stats["totalReturn"], 5),
            "values": analyze.round_series(idx),
        })
    holdings_out.sort(key=lambda h: h["weight"], reverse=True)

    # --- sector allocation ---
    sector_map: dict[str, float] = {}
    for h in holdings_out:
        sector_map[h["sector"]] = sector_map.get(h["sector"], 0.0) + h["weight"]
    sectors = [
        {"sector": k, "weight": analyze.round_n(v, 4)}
        for k, v in sorted(sector_map.items(), key=lambda x: x[1], reverse=True)
    ]

    # --- source flag ---
    source = "live"  # yfinance is always "live" in this backend

    # --- truncation ---
    truncated = aligned.get("truncated", False)
    truncated_by = aligned.get("truncated_by")
    truncated_note = None
    if truncated and truncated_by:
        tb_name, _, _ = _resolve_name(truncated_by)
        truncated_note = (
            f"History starts {dates[0]} — "
            f"{truncated_by} ({tb_name}) has the shortest track record in this portfolio."
        )

    return {
        "ok": True,
        "mode": mode,
        "source": source,
        "asOf": dates[-1],
        "range": {
            "start": dates[0],
            "end": dates[-1],
            "days": n_days,
            "truncated": truncated,
            "truncatedNote": truncated_note,
        },
        "dates": dates,
        "portfolio": analyze.round_series(portfolio),
        "portfolioMetrics": {
            **portfolio_metrics_base,
            "beta": ba["beta"] if ba else None,
            "alpha": ba["alpha"] if ba else None,
            "ytd": analyze.ytd_return(portfolio, dates),
        },
        "benchmarks": benchmarks,
        "holdings": holdings_out,
        "sectors": sectors,
        "monthly": analyze.monthly_returns(portfolio, dates),
    }
