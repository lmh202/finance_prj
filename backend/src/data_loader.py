"""Market data access for AURORA.

Two responsibilities:
  1. Ticker universe — the official NASDAQ Trader symbol directory, which lists
     every security trading on NASDAQ (nasdaqlisted.txt) plus NYSE / NYSE American /
     NYSE Arca / Cboe / IEX listings (otherlisted.txt). Cached locally so the app
     works offline after the first download.
  2. Latest prices — batch fetch via yfinance.
"""

import io
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple

import pandas as pd
import requests

from src.config import DATA_DIR

TICKER_CACHE = DATA_DIR / "tickers.csv"
CACHE_MAX_AGE_DAYS = 7

# ---------------------------------------------------------------------------
# Per-symbol market-data cache
#
# Every client now sends its own portfolio, so N visitors used to mean N
# yfinance downloads of largely the same tickers — the fastest way to get
# rate-limited in production. Caching per SYMBOL rather than per request means
# overlapping portfolios share bars: the second visitor holding AAPL pays
# nothing for it.
#
# A daily bar changes once per session, so hours of staleness is correct for
# history; the latest close moves intraday, so it gets a much shorter life.
# FastAPI runs these sync endpoints in a threadpool, so concurrent requests
# are real and the store is mutated under a lock.
# ---------------------------------------------------------------------------
HISTORY_TTL_SECONDS = 6 * 3600
PRICE_TTL_SECONDS = 15 * 60

_CACHE: Dict[Tuple[str, ...], Tuple[float, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(keys: List[Tuple[str, ...]], ttl: float) -> Dict[Tuple[str, ...], Any]:
    """Return the still-fresh entries among `keys`, dropping expired ones."""
    now = time.monotonic()
    fresh: Dict[Tuple[str, ...], Any] = {}
    with _CACHE_LOCK:
        for key in keys:
            hit = _CACHE.get(key)
            if hit is None:
                continue
            if now - hit[0] < ttl:
                fresh[key] = hit[1]
            else:
                _CACHE.pop(key, None)
    return fresh


def _cache_put(entries: Dict[Tuple[str, ...], Any]) -> None:
    now = time.monotonic()
    with _CACHE_LOCK:
        for key, value in entries.items():
            _CACHE[key] = (now, value)


def clear_market_cache() -> None:
    """Drop every cached bar — for tests and for a manual refresh."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _cached_by_symbol(
    symbols: List[str],
    namespace: str,
    period: str,
    ttl: float,
    fetch: Callable[[List[str]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Serve `symbols` from cache, downloading only the ones that are missing.

    A symbol the fetch could not resolve is cached as None so a delisted or
    typo'd ticker doesn't re-hit the network on every single request; callers
    already tolerate absent symbols, so those are filtered back out here.
    """
    keys = {s: (namespace, period, s) for s in symbols}
    fresh = _cache_get(list(keys.values()), ttl)

    missing = [s for s in symbols if keys[s] not in fresh]
    if missing:
        fetched = fetch(missing)
        _cache_put({keys[s]: fetched.get(s) for s in missing})
        for s in missing:
            fresh[keys[s]] = fetched.get(s)

    return {s: fresh[keys[s]] for s in symbols if fresh.get(keys[s]) is not None}

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
REQUEST_TIMEOUT = 30

# Exchange codes used in otherlisted.txt
EXCHANGE_CODES = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}

# Minimal built-in universe so the app still functions if the NASDAQ directory
# is unreachable and no cache exists yet.
FALLBACK_UNIVERSE = pd.DataFrame(
    [
        ("AAPL", "Apple Inc.", "NASDAQ", False),
        ("MSFT", "Microsoft Corporation", "NASDAQ", False),
        ("NVDA", "NVIDIA Corporation", "NASDAQ", False),
        ("AMZN", "Amazon.com Inc.", "NASDAQ", False),
        ("GOOGL", "Alphabet Inc. Class A", "NASDAQ", False),
        ("META", "Meta Platforms Inc.", "NASDAQ", False),
        ("TSLA", "Tesla Inc.", "NASDAQ", False),
        ("JPM", "JPMorgan Chase & Co.", "NYSE", False),
        ("JNJ", "Johnson & Johnson", "NYSE", False),
        ("SPY", "SPDR S&P 500 ETF Trust", "NYSE Arca", True),
        ("QQQ", "Invesco QQQ Trust", "NASDAQ", True),
        ("XLV", "Health Care Select Sector SPDR Fund", "NYSE Arca", True),
        ("GLD", "SPDR Gold Shares", "NYSE Arca", True),
        ("SLV", "iShares Silver Trust", "NYSE Arca", True),
    ],
    columns=["symbol", "name", "exchange", "is_etf"],
)


def _fetch_symbol_file(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), sep="|")
    # Last line is a "File Creation Time" footer, not a security
    df = df[~df.iloc[:, 0].astype(str).str.startswith("File Creation Time")]
    return df


def download_ticker_universe() -> pd.DataFrame:
    """Download the full US-listed universe from the NASDAQ symbol directory."""
    nasdaq = _fetch_symbol_file(NASDAQ_LISTED_URL)
    nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
    nasdaq_df = pd.DataFrame(
        {
            "symbol": nasdaq["Symbol"].astype(str).str.strip(),
            "name": nasdaq["Security Name"].astype(str).str.strip(),
            "exchange": "NASDAQ",
            "is_etf": nasdaq["ETF"] == "Y",
        }
    )

    other = _fetch_symbol_file(OTHER_LISTED_URL)
    other = other[other["Test Issue"] == "N"]
    other_df = pd.DataFrame(
        {
            "symbol": other["ACT Symbol"].astype(str).str.strip(),
            "name": other["Security Name"].astype(str).str.strip(),
            "exchange": other["Exchange"].map(EXCHANGE_CODES).fillna("Other"),
            "is_etf": other["ETF"] == "Y",
        }
    )

    universe = pd.concat([nasdaq_df, other_df], ignore_index=True)
    universe = universe[universe["symbol"].str.len() > 0]
    universe = universe.drop_duplicates(subset="symbol").sort_values("symbol")
    return universe.reset_index(drop=True)


def load_ticker_universe(force_refresh: bool = False) -> pd.DataFrame:
    """Return the cached universe, refreshing from NASDAQ when stale or forced.

    Falls back to a stale cache, then to a small built-in list, so the app
    never hard-fails on network problems.
    """
    cache_fresh = (
        TICKER_CACHE.exists()
        and (time.time() - TICKER_CACHE.stat().st_mtime) < CACHE_MAX_AGE_DAYS * 86400
    )
    if cache_fresh and not force_refresh:
        return _read_cache()

    try:
        universe = download_ticker_universe()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        universe.to_csv(TICKER_CACHE, index=False)
        return universe
    except Exception:
        if TICKER_CACHE.exists():
            return _read_cache()
        return FALLBACK_UNIVERSE.copy()


def _read_cache() -> pd.DataFrame:
    # keep_default_na: real tickers like "NA" must not come back as NaN
    return pd.read_csv(
        TICKER_CACHE,
        dtype={"symbol": str, "name": str, "exchange": str},
        keep_default_na=False,
        na_values=[],
    )


def to_yahoo_symbol(symbol: str) -> str:
    """NASDAQ directory symbols use '.', '/' or '$' for share classes and
    preferreds; Yahoo Finance uses '-' (e.g. BRK.B -> BRK-B)."""
    return symbol.replace(".", "-").replace("/", "-").replace("$", "-P")


def get_latest_prices(symbols: Iterable[str]) -> Dict[str, float]:
    """Fetch the most recent closing price for each symbol via yfinance.

    Returns a dict keyed by the ORIGINAL symbol. Symbols that could not be
    priced are simply absent — callers must handle missing keys. Served from
    the per-symbol cache for PRICE_TTL_SECONDS.
    """
    wanted = [s for s in dict.fromkeys(symbols) if s]
    if not wanted:
        return {}
    return _cached_by_symbol(
        wanted, "price", "latest", PRICE_TTL_SECONDS, _download_latest_prices
    )


def _download_latest_prices(symbols: List[str]) -> Dict[str, float]:
    import yfinance as yf

    yahoo_map = {to_yahoo_symbol(s): s for s in symbols}
    prices: Dict[str, float] = {}
    try:
        data = yf.download(
            list(yahoo_map.keys()),
            period="5d",
            progress=False,
            auto_adjust=True,
            group_by="column",
        )
    except Exception:
        return prices
    if data is None or data.empty:
        return prices

    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    if isinstance(close, pd.Series):
        close = close.to_frame(name=list(yahoo_map.keys())[0])

    for yahoo_sym, original in yahoo_map.items():
        if yahoo_sym in close.columns:
            series = close[yahoo_sym].dropna()
            if not series.empty:
                prices[original] = float(series.iloc[-1])
    return prices


def get_history(symbols: Iterable[str], period: str = "2y") -> pd.DataFrame:
    """Daily adjusted-close history via yfinance.

    Returns a DataFrame indexed by date with one column per ORIGINAL symbol.
    Symbols that could not be fetched are simply absent — callers must
    tolerate missing columns. Empty DataFrame on total failure. Served from
    the per-symbol cache for HISTORY_TTL_SECONDS.
    """
    wanted = [s for s in dict.fromkeys(symbols) if s]
    if not wanted:
        return pd.DataFrame()

    series = _cached_by_symbol(
        wanted,
        "close",
        period,
        HISTORY_TTL_SECONDS,
        lambda missing: _download_history(missing, period),
    )
    if not series:
        return pd.DataFrame()
    # Symbols cached at different moments can carry different trailing bars;
    # concat unions the calendars and leaves NaN where one lags, which is the
    # same shape a single multi-symbol download produces.
    return pd.concat(series, axis=1).dropna(how="all")


def _download_history(symbols: List[str], period: str) -> Dict[str, pd.Series]:
    import yfinance as yf

    yahoo_map = {to_yahoo_symbol(s): s for s in symbols}
    try:
        data = yf.download(
            list(yahoo_map.keys()),
            period=period,
            progress=False,
            auto_adjust=True,
            group_by="column",
        )
    except Exception:
        return {}
    if data is None or data.empty:
        return {}

    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    if isinstance(close, pd.Series):
        close = close.to_frame(name=list(yahoo_map.keys())[0])

    out: Dict[str, pd.Series] = {}
    for ysym, original in yahoo_map.items():
        if ysym in close.columns:
            col = close[ysym].dropna()
            if not col.empty:
                out[original] = col.rename(original)
    return out


def get_ohlc_history(symbols: Iterable[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
    """Daily adjusted OHLC history via yfinance, per symbol — for the risk engine's
    HAR features (the Parkinson range estimator needs high/low, which the close-only
    get_history() does not provide).

    Returns {original_symbol: DataFrame[close, high, low]} indexed by date. Symbols
    that could not be fetched are simply absent — callers must tolerate that.
    Served from the per-symbol cache for HISTORY_TTL_SECONDS.
    """
    wanted = [s for s in dict.fromkeys(symbols) if s]
    if not wanted:
        return {}
    return _cached_by_symbol(
        wanted,
        "ohlc",
        period,
        HISTORY_TTL_SECONDS,
        lambda missing: _download_ohlc_history(missing, period),
    )


def _download_ohlc_history(symbols: List[str], period: str) -> Dict[str, pd.DataFrame]:
    import yfinance as yf

    yahoo_map = {to_yahoo_symbol(s): s for s in symbols}
    try:
        data = yf.download(
            list(yahoo_map.keys()),
            period=period,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception:
        return {}
    if data is None or data.empty:
        return {}

    multi = isinstance(data.columns, pd.MultiIndex)
    out: Dict[str, pd.DataFrame] = {}
    for ysym, original in yahoo_map.items():
        try:
            sub = data[ysym] if multi else data
            df = sub[["Close", "High", "Low"]]
        except (KeyError, IndexError):
            continue
        df = df.rename(columns={"Close": "close", "High": "high", "Low": "low"}).dropna(how="all")
        if not df.empty:
            out[original] = df
    return out
