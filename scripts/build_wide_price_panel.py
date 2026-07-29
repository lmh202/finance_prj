"""Build a sector-diversified wide price panel for Daily Strategy research.

WHY THIS EXISTS
---------------
The 21-symbol FNSPID panel has an average pairwise return correlation of 0.33,
which makes its effective breadth ~2.8 independent names.  A single day's
cross-sectional rank IC therefore carries a standard error of 1/sqrt(20)=0.224.
Averaged over 13 years that is still ~0.030 — the same order as the signal
being measured, so a genuine IC of 0.03 can only ever produce t~1.0.  Every
promotion gate that requires t>2 is unpassable by construction on that panel.

This script rebuilds the same feature table over a wider, sector-diversified
universe so the question becomes measurable.  It is price-only: FNSPID is NOT
reprocessed and FinBERT is NOT re-run.  The cached 21-symbol sentiment table is
left-joined on, and every new symbol gets sentiment=0 / news_count=0 /
has_news=0 — the "optional news channel" contract from CLAUDE.md.

SURVIVORSHIP BIAS — READ BEFORE QUOTING ANY RESULT
--------------------------------------------------
UNIVERSE is a fixed list chosen from companies that were ALREADY large caps in
2013, so membership is decided on information available at the start of the
sample rather than on 2013-2023 outcomes.  That removes the worst form of the
bias (picking today's winners) but NOT all of it: firms that were delisted,
acquired, or went bankrupt during the window are absent, so realised returns
are still optimistic.  Treat cross-sectional IC as the primary result and
absolute portfolio returns as indicative only.  State this in any report.

Outputs (data/processed/)
  wide_price_panel.parquet          the trainable table
  wide_price_panel_manifest.json    universe, coverage, sha256, caveats

Run:
  python scripts/build_wide_price_panel.py
  python scripts/build_wide_price_panel.py --diagnose        # + run the IC test
  python scripts/build_wide_price_panel.py --start 2010-01-01
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SENTIMENT = PROCESSED / "sentiment_features.parquet"   # cached, 21 symbols
OUTPUT = PROCESSED / "wide_price_panel.parquet"
MANIFEST = PROCESSED / "wide_price_panel_manifest.json"

BENCHMARK = "SPY"
TRADING_DAYS = 252
DEFAULT_START = "2013-01-01"
DEFAULT_END = "2023-12-31"

# --------------------------------------------------------------- the universe
# Sector-diversified US large caps that were already large in 2013, plus the
# original 21 FNSPID symbols (kept so the news join and the old results stay
# comparable).  Edit this dict to change the universe — nothing else depends
# on its contents.  Deliberately excludes tickers whose identity changed
# mid-sample (RTX/UTX, LIN, DOW, CARR, OTIS) to avoid splice artefacts.
UNIVERSE: dict[str, list[str]] = {
    "information_technology": [
        "AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "QCOM", "TXN", "ADBE",
        "ACN", "MU", "AMAT", "ADI", "NVDA", "AVGO", "AMD", "INTU", "ASML",
        "CRM", "HPQ", "GLW", "STX", "NTAP", "AKAM",
    ],
    "communication_services": [
        "GOOGL", "META", "VZ", "T", "DIS", "CMCSA", "NFLX", "EA", "OMC",
    ],
    "consumer_discretionary": [
        "AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "F", "GM", "TSLA",
        "YUM", "ROST", "BBY", "EBAY",
    ],
    "consumer_staples": [
        "PG", "KO", "PEP", "WMT", "COST", "MO", "CL", "KMB", "GIS", "SYY",
        "K", "HSY", "STZ", "ADM",
    ],
    "health_care": [
        "JNJ", "PFE", "MRK", "ABT", "AMGN", "GILD", "BMY", "LLY", "UNH",
        "MDT", "CVS", "CI", "BDX", "SYK", "ISRG", "BIIB", "REGN", "ZBH",
    ],
    "financials": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "BK",
        "SCHW", "TRV", "ALL", "MET", "PRU", "BLK", "SPGI", "CME", "ICE", "AFL",
    ],
    "industrials": [
        "GE", "BA", "CAT", "MMM", "HON", "UPS", "UNP", "LMT", "DE", "EMR",
        "ITW", "CSX", "NSC", "FDX", "GD", "NOC", "WM", "ETN", "PH", "ROK",
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "VLO", "MPC", "HAL",
        "KMI", "WMB", "BKR", "DVN",
    ],
    "materials": [
        "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "PPG", "IP", "VMC", "MLM",
    ],
    "utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "WEC", "ES",
        "PEG", "SRE",
    ],
    "real_estate": [
        "AMT", "PLD", "CCI", "SPG", "PSA", "EQIX", "O", "AVB", "EQR", "VTR",
    ],
    "commodity_etf": ["GLD", "SLV"],
}

# The original FNSPID panel — every one of these must survive into the output
# so the 21-symbol results stay reproducible as a subset.
LEGACY_21 = [
    "AAPL", "ADBE", "AMD", "AMZN", "ASML", "AVGO", "COST", "CSCO", "GLD",
    "GOOGL", "INTC", "INTU", "MSFT", "MU", "NFLX", "NVDA", "PEP", "QCOM",
    "SLV", "TSLA", "TXN",
]


def universe_symbols() -> list[str]:
    out: list[str] = []
    for names in UNIVERSE.values():
        out.extend(names)
    missing = sorted(set(LEGACY_21) - set(out))
    if missing:
        raise ValueError(f"UNIVERSE is missing legacy FNSPID symbols: {missing}")
    return sorted(set(out))


def sector_of(symbol: str) -> str:
    for sector, names in UNIVERSE.items():
        if symbol in names:
            return sector
    return "unknown"


# ------------------------------------------------------------------ download
def _extract(raw: pd.DataFrame, symbol: str) -> pd.Series | None:
    """Pull one symbol's adjusted close out of a yfinance batch frame."""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if symbol not in raw.columns.get_level_values(0):
                return None
            frame = raw[symbol]
        else:
            frame = raw
        column = "Close" if "Close" in frame.columns else "Adj Close"
        series = frame[column].dropna()
    except (KeyError, AttributeError):
        return None
    return series if len(series) else None


def download_closes(symbols: list[str], start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    """Batch download with a per-symbol retry — Yahoo silently drops tickers
    from otherwise successful batch responses."""
    import yfinance as yf

    print(f"[1/4] downloading {len(symbols)} symbols {start} -> {end} …")
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    closes: dict[str, pd.Series] = {}
    retry: list[str] = []
    for symbol in symbols:
        series = _extract(raw, symbol)
        if series is None:
            retry.append(symbol)
        else:
            closes[symbol] = series

    failed: list[str] = []
    if retry:
        print(f"      retrying {len(retry)} symbols individually: {', '.join(retry)}")
        for symbol in retry:
            try:
                one = yf.Ticker(symbol).history(
                    start=start, end=end, auto_adjust=True
                )
                series = one["Close"].dropna() if "Close" in one else None
            except Exception:                                  # noqa: BLE001
                series = None
            if series is None or series.empty:
                failed.append(symbol)
            else:
                series.index = series.index.tz_localize(None)
                closes[symbol] = series

    if BENCHMARK not in closes:
        raise RuntimeError(f"benchmark {BENCHMARK} could not be downloaded")

    wide = pd.DataFrame(closes).sort_index()
    wide.index = pd.to_datetime(wide.index).tz_localize(None)
    print(f"      got {wide.shape[1]} symbols x {wide.shape[0]} sessions"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return wide, failed


# ------------------------------------------------------------------ features
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


FEATURE_COLS = [
    "ret_1d", "mom_20d", "mom_60d", "mom_12_1", "price_vs_sma50",
    "sma50_vs_sma200", "vol_20d", "rsi_14", "drawdown", "risk_adj_mom",
    "beta_60d", "rel_str_20d",
]


def build_features(wide: pd.DataFrame, min_sessions: int = 260) -> pd.DataFrame:
    """Same definitions as build_training_dataset.build_price_features, plus
    mom_12_1 (12-1 momentum: 60-day return excluding the most recent 20 days —
    the classic skip-a-month construction)."""
    print("[2/4] engineering price features …")
    spy = wide[BENCHMARK]
    spy_ret = spy.pct_change()
    spy_mom20 = spy.pct_change(20)

    rows = []
    skipped = []
    for symbol in wide.columns:
        if symbol == BENCHMARK:
            continue
        close = wide[symbol].dropna()
        if len(close) < min_sessions:
            skipped.append(symbol)
            continue

        ret1 = close.pct_change()
        feat = pd.DataFrame(index=close.index)
        feat["ret_1d"] = ret1
        feat["mom_20d"] = close.pct_change(20)
        feat["mom_60d"] = close.pct_change(60)
        feat["mom_12_1"] = feat["mom_60d"] - feat["mom_20d"]
        sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
        feat["price_vs_sma50"] = close / sma50 - 1
        feat["sma50_vs_sma200"] = sma50 / sma200 - 1
        feat["vol_20d"] = ret1.rolling(20).std() * np.sqrt(TRADING_DAYS)
        feat["rsi_14"] = _rsi(close)
        feat["drawdown"] = close / close.cummax() - 1
        feat["risk_adj_mom"] = feat["mom_20d"] / feat["vol_20d"].replace(0, np.nan)

        aligned = spy_ret.reindex(close.index)
        feat["beta_60d"] = (
            ret1.rolling(60).cov(aligned) / aligned.rolling(60).var().replace(0, np.nan)
        )
        feat["rel_str_20d"] = feat["mom_20d"] - spy_mom20.reindex(close.index)

        # forward labels — the only look-ahead allowed, it is the target
        feat["fwd_ret_5d"] = close.shift(-5) / close - 1
        feat["fwd_ret_20d"] = close.shift(-20) / close - 1
        feat["label_up_5d"] = (feat["fwd_ret_5d"] > 0).astype("Int64")
        feat["label_up_20d"] = (feat["fwd_ret_20d"] > 0).astype("Int64")

        feat.insert(0, "sector", sector_of(symbol))
        feat.insert(0, "symbol", symbol)
        feat.index.name = "date"
        rows.append(feat.reset_index())

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=FEATURE_COLS + ["fwd_ret_20d"]).reset_index(drop=True)
    if skipped:
        print(f"      skipped {len(skipped)} symbols with <{min_sessions} sessions: "
              f"{', '.join(skipped)}")
    print(f"      {len(panel):,} symbol-days x {len(FEATURE_COLS)} features, "
          f"{panel['symbol'].nunique()} symbols")
    return panel


# --------------------------------------------------------------------- merge
def attach_news(panel: pd.DataFrame) -> pd.DataFrame:
    """Left-join the cached 21-symbol sentiment table.  Symbols outside FNSPID
    get the documented neutral state — this is the optional-channel contract,
    not an imputation choice."""
    print("[3/4] left-joining cached sentiment (no FNSPID reprocessing) …")
    if not SENTIMENT.exists():
        print(f"      {SENTIMENT.name} not found — writing price-only panel")
        panel["sentiment"] = 0.0
        panel["news_count"] = 0
        panel["has_news"] = 0
        return panel

    sent = pd.read_parquet(SENTIMENT)
    sent["date"] = pd.to_datetime(sent["date"])
    panel["date"] = pd.to_datetime(panel["date"])
    merged = panel.merge(sent, on=["date", "symbol"], how="left")
    merged["sentiment"] = merged["sentiment"].fillna(0.0)
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged["has_news"] = merged["has_news"].fillna(0).astype(int)
    covered = sorted(set(sent["symbol"]) & set(merged["symbol"]))
    print(f"      news coverage: {len(covered)}/{merged['symbol'].nunique()} symbols "
          f"({merged['has_news'].mean():.1%} of rows)")
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(panel: pd.DataFrame, failed: list[str], start: str, end: str) -> None:
    print("[4/4] writing outputs …")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel.to_parquet(OUTPUT, index=False)

    by_sector = (
        panel.groupby("sector")["symbol"].nunique().sort_values(ascending=False)
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance (auto_adjust=True)",
        "range": [start, end],
        "rows": int(len(panel)),
        "symbols": int(panel["symbol"].nunique()),
        "symbols_by_sector": {k: int(v) for k, v in by_sector.items()},
        "legacy_21_present": sorted(set(LEGACY_21) & set(panel["symbol"])),
        "download_failed": failed,
        "features": FEATURE_COLS,
        "news": {
            "reprocessed": False,
            "source": SENTIMENT.name,
            "symbols_with_news": int(panel.loc[panel["has_news"] > 0, "symbol"].nunique()),
            "policy": "left join; symbols outside FNSPID get sentiment=0/news_count=0/has_news=0",
        },
        "caveats": [
            "Survivorship bias: UNIVERSE is a fixed list of firms that were "
            "already large caps in 2013. Membership uses start-of-sample "
            "information, but firms delisted or acquired during 2013-2023 are "
            "absent, so realised returns are optimistic.",
            "Cross-sectional rank IC is the primary result; absolute portfolio "
            "returns are indicative only.",
            "Yahoo adjusted closes splice dividends and splits; they are not "
            "point-in-time and may be revised.",
        ],
        "sha256": _sha256(OUTPUT),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def show(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:          # output redirected outside the repo
            return str(path)

    print(f"      wrote {show(OUTPUT)} ({len(panel):,} rows, "
          f"{panel['symbol'].nunique()} symbols)")
    print(f"      wrote {show(MANIFEST)}")


# ----------------------------------------------------------------- diagnostic
def diagnose(panel: pd.DataFrame) -> None:
    """Answer the one question this panel was built for: does breadth turn the
    signal from unmeasurable (t~1) into measurable (t>2)?  Compares the wide
    panel against the legacy 21-symbol subset on identical code."""
    print("\n" + "=" * 78)
    print("BREADTH DIAGNOSTIC — same features, same years, two universes")
    print("=" * 78)

    def rank_ic(df: pd.DataFrame, feature: str, target: str, horizon: int):
        sub = df[["date", feature, target]].dropna()
        sub = sub[sub.groupby("date")[feature].transform("size") >= 8]
        if sub.empty:
            return None
        ic = sub.groupby("date").apply(
            lambda x: x[feature].rank().corr(x[target].rank())
        ).dropna()
        if len(ic) < 200:
            return None
        t = ic.mean() / ic.std() * np.sqrt(len(ic) / horizon)
        years = ic.groupby(ic.index.year).mean()
        return ic.mean(), t, int((years > 0).sum()), len(years)

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    subsets = {
        "legacy 21 symbols": panel[panel["symbol"].isin(LEGACY_21)],
        f"wide {panel['symbol'].nunique()} symbols": panel,
    }
    features = ["mom_12_1", "mom_60d", "sma50_vs_sma200", "beta_60d", "vol_20d"]

    for target, horizon in (("fwd_ret_5d", 5), ("fwd_ret_20d", 20)):
        print(f"\n--- {target} ---")
        print(f"{'feature':20s} " + " ".join(f"{k:>28s}" for k in subsets))
        print(f"{'':20s} " + " ".join(f"{'IC':>9s}{'t':>8s}{'pos yrs':>11s}"
                                      for _ in subsets))
        for feature in features:
            cells = []
            for frame in subsets.values():
                got = rank_ic(frame, feature, target, horizon)
                cells.append("        n/a" + " " * 17 if got is None else
                             f"{got[0]:>+9.4f}{got[1]:>+8.2f}{got[2]:>7d}/{got[3]:<3d}")
            print(f"{feature:20s} " + " ".join(cells))

    print("\nDecision rule:")
    print("  mom_12_1 t rises from ~1.6 to >3  -> breadth was the binding constraint;")
    print("                                       scaling further is justified.")
    print("  mom_12_1 t stays below ~2         -> the weak IC is real market")
    print("                                       efficiency; pivot to exposure/timing.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--min-sessions", type=int, default=260,
                    help="drop symbols with fewer sessions (warm-up needs 200d SMA)")
    ap.add_argument("--diagnose", action="store_true",
                    help="run the breadth IC diagnostic after building")
    args = ap.parse_args()

    symbols = universe_symbols()
    if BENCHMARK not in symbols:
        symbols.append(BENCHMARK)
    wide, failed = download_closes(symbols, args.start, args.end)
    panel = build_features(wide, args.min_sessions)
    panel = attach_news(panel)
    write_outputs(panel, failed, args.start, args.end)
    if args.diagnose:
        diagnose(panel)
    print("\ndone.")


if __name__ == "__main__":
    main()
