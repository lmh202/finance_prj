"""Build a current, broader out-of-sample test set for the AURORA risk engine.

The original research data ends in 2023. This script extends evaluation from
2024 to the latest completed market session and broadens the universe beyond
the 21 development assets:

  - original training-dataset symbols;
  - symbols in the user's current portfolio;
  - stable, liquid broad-market/sector/bond/commodity ETFs.

Two data products are intentionally separated:

1. ``risk_backtest_panel.parquet``
   Historical OHLC-based HAR forecasts and realized 5/20-day outcomes. This is
   immediately useful for an honest 2024-present risk backtest.

2. ``rss_event_observations.parquet``
   Prospective snapshots that join the latest pre-event risk forecast to RSS
   stories first observed during each collection run. RSS is not an archive,
   so earlier dates are never filled with false "zero news" values. Re-running
   this script grows the RSS archive and matures prior forward-return labels.

Raw RSS entries preserve both ``published_utc`` and first ``fetched_utc``.
Event timing uses first fetch, not publication, so a story is never inserted
retroactively into a historical decision.

Outputs under ``data/processed/current_risk_test/``:

  universe.csv
  ohlc.parquet
  risk_backtest_panel.parquet
  risk_backtest_summary.csv
  rss_headlines.json
  rss_event_observations.parquet
  manifest.json

Run:
  python scripts/build_current_risk_testset.py
  python scripts/build_current_risk_testset.py --no-rss
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src import portfolio as portfolio_store  # noqa: E402
from src.news_intelligence import collector  # noqa: E402

TRAINING_DATA = ROOT / "data" / "processed" / "training_dataset.parquet"
MODEL_PATH = ROOT / "data" / "processed" / "risk_model.json"
OUTPUT = ROOT / "data" / "processed" / "current_risk_test"
TEST_START = pd.Timestamp("2024-01-01")
WARMUP_START = pd.Timestamp("2022-01-01")
HORIZONS = (5, 20)
EPS = 1e-6

# Stable ETFs avoid using today's equity-index constituents to reconstruct
# earlier years, which would introduce survivorship bias.
BROAD_ETFS = {
    "SPY": "US large-cap equity",
    "QQQ": "US growth/technology equity",
    "IWM": "US small-cap equity",
    "DIA": "US blue-chip equity",
    "EEM": "emerging-market equity",
    "XLF": "financial sector",
    "XLK": "technology sector",
    "XLV": "health-care sector",
    "XLE": "energy sector",
    "XLY": "consumer discretionary sector",
    "XLP": "consumer staples sector",
    "XLI": "industrial sector",
    "XLU": "utilities sector",
    "VNQ": "US real estate",
    "TLT": "long-duration US Treasury",
    "HYG": "high-yield credit",
    "LQD": "investment-grade credit",
    "USO": "oil commodity",
    "GLD": "gold commodity",
    "SLV": "silver commodity",
}


def _groups(symbol: str, training: set[str], portfolio: set[str]) -> str:
    groups = []
    if symbol in training:
        groups.append("original_research")
    if symbol in portfolio:
        groups.append("current_portfolio")
    if symbol in BROAD_ETFS:
        groups.append("broad_etf")
    return "|".join(groups) or "additional"


def build_universe() -> pd.DataFrame:
    training_df = pd.read_parquet(TRAINING_DATA, columns=["symbol"])
    training = set(training_df["symbol"].dropna().astype(str).str.upper())
    holdings = portfolio_store.load_portfolio()
    portfolio = set(holdings["symbol"].dropna().astype(str).str.upper())
    symbols = sorted(training | portfolio | set(BROAD_ETFS))
    return pd.DataFrame(
        {
            "symbol": symbols,
            "groups": [_groups(s, training, portfolio) for s in symbols],
            "broad_etf_role": [BROAD_ETFS.get(s, "") for s in symbols],
        }
    )


def _normalise_download(symbol: str, data: pd.DataFrame) -> Optional[pd.DataFrame]:
    if data is None or data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        if symbol in data.columns.get_level_values(0):
            data = data[symbol]
        elif symbol in data.columns.get_level_values(-1):
            data = data.xs(symbol, axis=1, level=-1)
        else:
            return None
    wanted = ["Close", "High", "Low", "Open", "Volume"]
    if not set(wanted[:3]).issubset(data.columns):
        return None
    out = data[[c for c in wanted if c in data.columns]].copy()
    out.columns = [c.lower() for c in out.columns]
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out.index.name = "date"
    out["symbol"] = symbol
    return out.reset_index()


def download_ohlc(symbols: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    symbols = list(symbols)
    print(
        f"Downloading adjusted OHLC for {len(symbols)} symbols "
        f"from {WARMUP_START.date()} …"
    )
    raw = yf.download(
        symbols,
        start=WARMUP_START.date().isoformat(),
        auto_adjust=True,
        actions=False,
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=30,
    )
    frames: Dict[str, pd.DataFrame] = {}
    missing = []
    for symbol in symbols:
        frame = _normalise_download(symbol, raw)
        if frame is not None:
            frames[symbol] = frame
        else:
            missing.append(symbol)

    # Yahoo occasionally drops individual tickers from a successful batch.
    # Retry only those symbols with the independent Ticker.history route.
    still_missing = []
    for symbol in missing:
        frame = None
        for _ in range(2):
            try:
                one = yf.Ticker(symbol).history(
                    start=WARMUP_START.date().isoformat(),
                    auto_adjust=True,
                    actions=False,
                    timeout=30,
                )
                frame = _normalise_download(symbol, one)
            except Exception:
                frame = None
            if frame is not None:
                break
        if frame is None:
            still_missing.append(symbol)
        else:
            frames[symbol] = frame

    if not frames:
        raise RuntimeError("No OHLC data could be downloaded.")
    out = pd.concat(frames.values(), ignore_index=True)
    out = out.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    return out, still_missing


def _feature_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    ohlc = ohlc.sort_values("date").set_index("date")
    close, high, low = ohlc["close"], ohlc["high"], ohlc["low"]
    ret = close.pct_change()
    park = (np.log(high / low) ** 2) / (4 * np.log(2))
    out = pd.DataFrame(index=close.index)
    out["close"] = close
    out["ret_1d"] = ret
    out["l_rv5"] = np.log(ret.rolling(5).std() + EPS)
    out["l_rv22"] = np.log(ret.rolling(22).std() + EPS)
    out["l_rv66"] = np.log(ret.rolling(66).std() + EPS)
    out["l_park5"] = np.log(np.sqrt(park.rolling(5).mean()) + EPS)
    out["l_park22"] = np.log(np.sqrt(park.rolling(22).mean()) + EPS)
    out["l_absret"] = np.log(ret.abs() + EPS)
    return out


def build_risk_panel(
    ohlc: pd.DataFrame, universe: pd.DataFrame, model: dict
) -> pd.DataFrame:
    group_map = universe.set_index("symbol")["groups"].to_dict()
    frames = []
    for symbol, prices in ohlc.groupby("symbol", sort=True):
        frame = _feature_frame(prices)
        result = pd.DataFrame(index=frame.index)
        result["symbol"] = symbol
        result["groups"] = group_map[symbol]
        result["close"] = frame["close"]
        result["ret_1d"] = frame["ret_1d"]
        for horizon in HORIZONS:
            mh = model["horizons"][str(horizon)]
            fh = model["fhs"]["horizons"][str(horizon)]
            log_sigma = mh["intercept"] + sum(
                frame[name] * mh["coef"][name] for name in mh["features"]
            )
            sigma_daily = np.exp(log_sigma) * mh["smearing"]
            sigma_h = sigma_daily * np.sqrt(horizon)
            forward_return = frame["close"].shift(-horizon) / frame["close"] - 1
            realized_vol = (
                frame["ret_1d"].rolling(horizon).std().shift(-horizon)
            )

            result[f"sigma_daily_{horizon}d"] = sigma_daily
            result[f"sigma_{horizon}d"] = sigma_h
            result[f"risk_level_{horizon}d"] = (
                sigma_h.expanding(min_periods=1).rank(pct=True) * 100
            )
            result[f"var95_{horizon}d"] = sigma_h * fh["q05"]
            result[f"var99_{horizon}d"] = sigma_h * fh["q01"]
            result[f"es95_{horizon}d"] = sigma_h * fh["es05"]
            result[f"band_lo_{horizon}d"] = sigma_h * fh["q025"]
            result[f"band_hi_{horizon}d"] = sigma_h * fh["q975"]
            result[f"fwd_return_{horizon}d"] = forward_return
            result[f"realized_vol_{horizon}d"] = realized_vol
            result[f"mature_{horizon}d"] = forward_return.notna()
            result[f"breach95_{horizon}d"] = (
                forward_return < result[f"var95_{horizon}d"]
            ).where(forward_return.notna())
            result[f"breach99_{horizon}d"] = (
                forward_return < result[f"var99_{horizon}d"]
            ).where(forward_return.notna())
            result[f"inside_band_{horizon}d"] = (
                (forward_return >= result[f"band_lo_{horizon}d"])
                & (forward_return <= result[f"band_hi_{horizon}d"])
            ).where(forward_return.notna())
        result.index.name = "date"
        frames.append(result.reset_index())

    panel = pd.concat(frames, ignore_index=True)
    required = [f"sigma_{h}d" for h in HORIZONS]
    panel = panel.dropna(subset=required)
    panel = panel[panel["date"] >= TEST_START].copy()
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def _qlike(realized_vol: pd.Series, forecast_vol: pd.Series) -> float:
    mask = realized_vol.notna() & forecast_vol.notna() & (forecast_vol > 0)
    y2 = realized_vol[mask].to_numpy() ** 2
    f2 = np.maximum(forecast_vol[mask].to_numpy(), 1e-8) ** 2
    ratio = y2 / f2
    return float(np.mean(ratio - np.log(ratio) - 1))


def summarise_backtest(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {
        "all": pd.Series(True, index=panel.index),
        "original_research": panel["groups"].str.contains("original_research"),
        "external_generalization": ~panel["groups"].str.contains("original_research"),
        "current_portfolio": panel["groups"].str.contains("current_portfolio"),
    }
    for scope, scope_mask in scopes.items():
        for horizon in HORIZONS:
            mature = (
                scope_mask
                & panel[f"mature_{horizon}d"]
                & panel[f"sigma_daily_{horizon}d"].notna()
            )
            sample = panel.loc[mature]
            if sample.empty:
                continue
            rows.append(
                {
                    "scope": scope,
                    "horizon": horizon,
                    "n": len(sample),
                    "n_symbols": sample["symbol"].nunique(),
                    "start": sample["date"].min().date().isoformat(),
                    "end": sample["date"].max().date().isoformat(),
                    "qlike": _qlike(
                        sample[f"realized_vol_{horizon}d"],
                        sample[f"sigma_daily_{horizon}d"],
                    ),
                    "var95_breach_rate": sample[
                        f"breach95_{horizon}d"
                    ].astype(float).mean(),
                    "var99_breach_rate": sample[
                        f"breach99_{horizon}d"
                    ].astype(float).mean(),
                    "band_coverage": sample[
                        f"inside_band_{horizon}d"
                    ].astype(float).mean(),
                    "mean_es_ratio": (
                        sample.loc[
                            sample[f"breach95_{horizon}d"] == True,  # noqa: E712
                            f"fwd_return_{horizon}d",
                        ].mean()
                        / sample.loc[
                            sample[f"breach95_{horizon}d"] == True,  # noqa: E712
                            f"es95_{horizon}d",
                        ].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _effective_session(observed_utc: datetime) -> str:
    """Conservative US-market session associated with the first RSS fetch."""
    try:
        from zoneinfo import ZoneInfo

        eastern = observed_utc.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        eastern = observed_utc
    day = pd.Timestamp(eastern.date())
    if day.dayofweek < 5 and eastern.time() < time(9, 30):
        return day.date().isoformat()
    return (day + pd.offsets.BDay(1)).date().isoformat()


def _load_json_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def collect_rss_snapshot(
    universe: pd.DataFrame, panel: pd.DataFrame, rss_path: Path
) -> tuple[pd.DataFrame, dict]:
    before = {r["id"] for r in _load_json_records(rss_path)}
    is_baseline = not before
    stats = collector.collect(universe["symbol"].tolist(), store_path=rss_path)
    records = _load_json_records(rss_path)
    new_records = [r for r in records if r["id"] not in before]
    if new_records:
        observed_text = min(r["fetched_utc"] for r in new_records)
        observed_utc = datetime.fromisoformat(observed_text)
    else:
        observed_utc = datetime.now(timezone.utc)

    # Some feeds retain stale entries. They remain in the raw archive, but only
    # stories first seen within 72 hours of publication count as fresh events.
    fresh_records = []
    for record in new_records:
        published = record.get("published_utc")
        if not published:
            continue
        try:
            lag_hours = (
                datetime.fromisoformat(record["fetched_utc"])
                - datetime.fromisoformat(published)
            ).total_seconds() / 3600
        except (TypeError, ValueError):
            continue
        if -2 <= lag_hours <= 72:
            fresh_records.append(record)

    general_count = sum(1 for r in fresh_records if not r.get("symbols"))
    specific_counts = {symbol: 0 for symbol in universe["symbol"]}
    for record in fresh_records:
        for symbol in record.get("symbols", []):
            if symbol in specific_counts:
                specific_counts[symbol] += 1

    latest = (
        panel.sort_values("date").groupby("symbol", as_index=False).tail(1)
    )
    snapshot = latest[
        [
            "symbol",
            "groups",
            "date",
            "sigma_5d",
            "risk_level_5d",
            "var95_5d",
            "sigma_20d",
            "risk_level_20d",
            "var95_20d",
        ]
    ].rename(columns={"date": "forecast_as_of"})
    snapshot.insert(0, "collection_utc", observed_utc.isoformat())
    snapshot.insert(1, "effective_session", _effective_session(observed_utc))
    snapshot.insert(
        2,
        "rss_snapshot_kind",
        "baseline_feed_window" if is_baseline else "incremental",
    )
    # The initial feed window is useful as a baseline archive, but not as an
    # arrival-rate observation because every ticker feed starts nearly full.
    snapshot.insert(
        3,
        "eligible_for_rate_analysis",
        (not is_baseline) and stats["feeds_failed"] == 0,
    )
    snapshot["rss_symbol_news_count"] = snapshot["symbol"].map(specific_counts)
    snapshot["rss_market_news_count"] = general_count
    snapshot["rss_any_symbol_news"] = snapshot["rss_symbol_news_count"] > 0
    snapshot["new_rss_records_in_run"] = len(new_records)
    snapshot["fresh_rss_records_in_run"] = len(fresh_records)
    snapshot["stale_or_undated_records_in_run"] = (
        len(new_records) - len(fresh_records)
    )
    snapshot["fwd_return_5d"] = np.nan
    snapshot["fwd_return_20d"] = np.nan
    snapshot["label_status_5d"] = "pending"
    snapshot["label_status_20d"] = "pending"
    return snapshot, {**stats, "new_ids_this_run": len(new_records)}


def update_event_labels(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    lookup = panel.set_index(["symbol", "date"])
    events = events.copy()
    events["forecast_as_of"] = pd.to_datetime(events["forecast_as_of"])
    for idx, event in events.iterrows():
        key = (event["symbol"], event["forecast_as_of"])
        if key not in lookup.index:
            continue
        row = lookup.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        for horizon in HORIZONS:
            value = row[f"fwd_return_{horizon}d"]
            if pd.notna(value):
                events.at[idx, f"fwd_return_{horizon}d"] = float(value)
                events.at[idx, f"label_status_{horizon}d"] = "mature"
    return events


def append_events(path: Path, snapshot: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        old = pd.read_parquet(path)
        # Migrate the first file produced before baseline/incremental flags were
        # added. Its saturated feed-window counts must not enter rate analysis.
        if "rss_snapshot_kind" not in old:
            old["rss_snapshot_kind"] = "baseline_feed_window"
        if "eligible_for_rate_analysis" not in old:
            old["eligible_for_rate_analysis"] = False
        if "fresh_rss_records_in_run" not in old:
            old["fresh_rss_records_in_run"] = np.nan
        if "stale_or_undated_records_in_run" not in old:
            old["stale_or_undated_records_in_run"] = np.nan
        events = pd.concat([old, snapshot], ignore_index=True)
    else:
        events = snapshot
    events = events.drop_duplicates(["collection_utc", "symbol"], keep="last")
    events = update_event_labels(events, panel)
    events.to_parquet(path, index=False)
    return events


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-rss", action="store_true", help="skip RSS collection"
    )
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    universe = build_universe()
    universe.to_csv(OUTPUT / "universe.csv", index=False)
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

    ohlc, failed = download_ohlc(universe["symbol"])
    ohlc.to_parquet(OUTPUT / "ohlc.parquet", index=False)
    available = set(ohlc["symbol"])
    usable_universe = universe[universe["symbol"].isin(available)].copy()

    panel = build_risk_panel(ohlc, usable_universe, model)
    panel.to_parquet(OUTPUT / "risk_backtest_panel.parquet", index=False)
    summary = summarise_backtest(panel)
    summary.to_csv(OUTPUT / "risk_backtest_summary.csv", index=False)

    rss_stats = None
    event_rows = 0
    if not args.no_rss:
        rss_path = OUTPUT / "rss_headlines.json"
        snapshot, rss_stats = collect_rss_snapshot(
            usable_universe, panel, rss_path
        )
        events = append_events(
            OUTPUT / "rss_event_observations.parquet", snapshot, panel
        )
        event_rows = len(events)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_start": TEST_START.date().isoformat(),
        "latest_market_date": panel["date"].max().date().isoformat(),
        "model_created": model.get("created"),
        "model_name": model.get("model"),
        "universe_requested": len(universe),
        "universe_downloaded": len(available),
        "failed_symbols": failed,
        "ohlc_rows": len(ohlc),
        "risk_panel_rows": len(panel),
        "mature_5d_rows": int(panel["mature_5d"].sum()),
        "mature_20d_rows": int(panel["mature_20d"].sum()),
        "rss_event_rows": event_rows,
        "rss_stats": rss_stats,
        "sources": {
            "prices": "Yahoo Finance via yfinance (adjusted OHLC)",
            "rss_general": [
                feed["url"] for feed in collector.GENERAL_FEEDS
            ],
            "rss_ticker_template": collector.YAHOO_TICKER_URL,
        },
        "caveats": [
            "RSS has no historical archive; event observations begin with the first local fetch.",
            "Current ETF constituents were not used; stable ETF tickers avoid constituent survivorship bias.",
            "External-generalization symbols were not part of model development.",
            "Daily 5/20-day outcomes overlap; use block-aware inference for formal tests.",
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_safe), encoding="utf-8"
    )

    print("\nCurrent risk test set built")
    print(
        f"  universe: {len(available)}/{len(universe)} symbols "
        f"({', '.join(failed) if failed else 'no failures'})"
    )
    print(
        f"  market:   {panel['date'].min().date()}–"
        f"{panel['date'].max().date()} · {len(panel):,} forecast rows"
    )
    print(
        f"  mature:   {int(panel['mature_5d'].sum()):,} 5d · "
        f"{int(panel['mature_20d'].sum()):,} 20d"
    )
    print("\nBacktest summary")
    print(summary.to_string(index=False))
    if rss_stats is not None:
        print(f"\nRSS: {json.dumps(rss_stats)}")
        print(f"RSS event observations: {event_rows:,}")
    print(f"\nOutput -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
