"""Shared-kernel market data: ticker universe + latest prices + history."""

from fastapi import APIRouter

from serialize import df_records, df_split
from src import data_loader

router = APIRouter(prefix="/market", tags=["market"])


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


@router.get("/history")
def history(symbols: str, period: str = "2y") -> dict:
    """Daily adjusted-close history. `symbols` is comma-separated; a symbol's
    column is absent if it could not be fetched (contract in
    src/data_loader.get_history)."""
    wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    return df_split(data_loader.get_history(wanted, period=period))
