"""Build leakage-safe panels for the final portfolio decision layer.

The predictive target is not price direction.  It is the next-20-session
return in excess of SPY:

    alpha_20d = stock_return_20d - spy_return_20d

The return model deliberately uses only close-derived price features that can
be reproduced by the online backend.  News reaches the decision through the
formal five-session HAR-X + News risk forecast, which is merged as an
out-of-fold risk input for historical evaluation.

Outputs
-------
data/processed/decision_dataset.parquet
data/processed/decision_external_dataset.parquet
data/processed/decision_dataset_manifest.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TRAINING_DATA = PROCESSED / "training_dataset.parquet"
RISK_OOF = (
    ROOT / "reports" / "risk_engine_optimization" / "oof_predictions.parquet"
)
HISTORICAL_SPY = (
    ROOT / "FNSPID" / "final_dataset" / "prices" / "SPY.csv"
)
CURRENT_OHLC = PROCESSED / "current_risk_test" / "ohlc.parquet"
CURRENT_RISK = (
    PROCESSED / "current_risk_test" / "risk_backtest_panel.parquet"
)

OUTPUT = PROCESSED / "decision_dataset.parquet"
EXTERNAL_OUTPUT = PROCESSED / "decision_external_dataset.parquet"
MANIFEST = PROCESSED / "decision_dataset_manifest.json"

BENCHMARK = "SPY"
TRADING_DAYS = 252
HORIZON = 20
PRICE_FEATURES = [
    "ret_1d",
    "mom_20d",
    "mom_60d",
    "price_vs_sma50",
    "sma50_vs_sma200",
    "vol_20d",
    "rsi_14",
    "drawdown",
    "risk_adj_mom",
    "beta_60d",
    "rel_str_20d",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + relative_strength)


def engineer_close_features(
    close_wide: pd.DataFrame,
    benchmark: str = BENCHMARK,
) -> pd.DataFrame:
    """Reproduce the deployable price schema from adjusted closes."""
    close_wide = close_wide.sort_index()
    if benchmark not in close_wide:
        raise ValueError(f"benchmark {benchmark} is missing")
    spy = close_wide[benchmark].dropna()
    spy_return = spy.pct_change()
    spy_momentum = spy.pct_change(20)

    rows: list[pd.DataFrame] = []
    for symbol in close_wide.columns:
        if symbol == benchmark:
            continue
        close = close_wide[symbol].dropna()
        if len(close) < 220:
            continue
        return_1d = close.pct_change()
        frame = pd.DataFrame(index=close.index)
        frame["ret_1d"] = return_1d
        frame["mom_20d"] = close.pct_change(20)
        frame["mom_60d"] = close.pct_change(60)
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        frame["price_vs_sma50"] = close / sma50 - 1
        frame["sma50_vs_sma200"] = sma50 / sma200 - 1
        frame["vol_20d"] = (
            return_1d.rolling(20).std() * np.sqrt(TRADING_DAYS)
        )
        frame["rsi_14"] = _rsi(close)
        frame["drawdown"] = close / close.cummax() - 1
        frame["risk_adj_mom"] = (
            frame["mom_20d"] / frame["vol_20d"].replace(0, np.nan)
        )
        aligned_spy_return = spy_return.reindex(close.index)
        covariance = return_1d.rolling(60).cov(aligned_spy_return)
        benchmark_variance = aligned_spy_return.rolling(60).var()
        frame["beta_60d"] = (
            covariance / benchmark_variance.replace(0, np.nan)
        )
        frame["rel_str_20d"] = (
            frame["mom_20d"] - spy_momentum.reindex(close.index)
        )
        frame["fwd_ret_5d"] = close.shift(-5) / close - 1
        frame["fwd_ret_20d"] = close.shift(-20) / close - 1
        frame["benchmark_fwd_ret_20d"] = (
            spy.shift(-20) / spy - 1
        ).reindex(close.index)
        frame["benchmark_ret_1d"] = spy_return.reindex(close.index)
        frame["excess_ret_20d"] = (
            frame["fwd_ret_20d"] - frame["benchmark_fwd_ret_20d"]
        )
        frame.insert(0, "symbol", symbol)
        frame.index.name = "date"
        rows.append(frame.reset_index())

    if not rows:
        raise ValueError("no symbols have enough history")
    result = pd.concat(rows, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    return result.sort_values(["date", "symbol"]).reset_index(drop=True)


def _historical_spy_forward_return() -> pd.DataFrame:
    spy = pd.read_csv(HISTORICAL_SPY, usecols=["date", "adj close"])
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
    spy = spy.sort_values("date")
    spy["benchmark_ret_1d"] = spy["adj close"].pct_change()
    spy["benchmark_fwd_ret_20d"] = (
        spy["adj close"].shift(-HORIZON) / spy["adj close"] - 1
    )
    return spy[
        ["date", "benchmark_ret_1d", "benchmark_fwd_ret_20d"]
    ]


def _formal_oof_risk() -> pd.DataFrame:
    risk = pd.read_parquet(RISK_OOF)
    risk["date"] = pd.to_datetime(risk["date"]).dt.normalize()
    risk = risk.loc[risk["horizon"].eq(5)].copy()
    risk = risk[
        [
            "date",
            "symbol",
            "test_year",
            "forecast__har_news_linear_deployable",
        ]
    ].rename(
        columns={
            "forecast__har_news_linear_deployable": "risk_sigma_daily_5d"
        }
    )
    risk = risk.sort_values(["symbol", "date"]).reset_index(drop=True)
    risk["risk_level_5d"] = (
        risk.groupby("symbol", group_keys=False)["risk_sigma_daily_5d"]
        .expanding(min_periods=1)
        .rank(pct=True)
        .reset_index(level=0, drop=True)
        .to_numpy()
        * 100
    )
    return risk


def build_historical_panel() -> pd.DataFrame:
    panel = pd.read_parquet(TRAINING_DATA)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.merge(
        _historical_spy_forward_return(),
        on="date",
        how="left",
        validate="many_to_one",
    )
    panel["excess_ret_20d"] = (
        panel["fwd_ret_20d"] - panel["benchmark_fwd_ret_20d"]
    )
    panel = panel.merge(
        _formal_oof_risk(),
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    panel["target_rank_20d"] = panel.groupby("date")[
        "excess_ret_20d"
    ].rank(pct=True)
    panel["development_split"] = np.select(
        [
            panel["date"].dt.year.le(2017),
            panel["date"].dt.year.between(2018, 2020),
            panel["date"].dt.year.between(2021, 2023),
        ],
        ["pretrain", "validation", "locked_test"],
        default="unused",
    )
    required = PRICE_FEATURES + [
        "fwd_ret_20d",
        "benchmark_fwd_ret_20d",
        "excess_ret_20d",
    ]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=required).copy()
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_external_panel() -> pd.DataFrame:
    raw = pd.read_parquet(CURRENT_OHLC)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    close_wide = raw.pivot(index="date", columns="symbol", values="close")
    panel = engineer_close_features(close_wide)

    risk = pd.read_parquet(CURRENT_RISK)
    risk["date"] = pd.to_datetime(risk["date"]).dt.normalize()
    risk = risk[
        [
            "date",
            "symbol",
            "groups",
            "sigma_daily_5d",
            "sigma_5d",
            "risk_level_5d",
            "var95_5d",
            "es95_5d",
            "mature_20d",
        ]
    ].rename(columns={"sigma_daily_5d": "risk_sigma_daily_5d"})
    panel = panel.merge(
        risk,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    panel["target_rank_20d"] = panel.groupby("date")[
        "excess_ret_20d"
    ].rank(pct=True)
    panel["development_split"] = "external_2024_2026"
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(
        subset=PRICE_FEATURES
        + [
            "excess_ret_20d",
            "risk_sigma_daily_5d",
            "risk_level_5d",
        ]
    ).copy()
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def main() -> None:
    historical = build_historical_panel()
    external = build_external_panel()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    historical.to_parquet(OUTPUT, index=False)
    external.to_parquet(EXTERNAL_OUTPUT, index=False)

    manifest = {
        "schema_version": 1,
        "target": "20-session stock return minus SPY return",
        "target_column": "excess_ret_20d",
        "target_rank_column": "target_rank_20d",
        "price_features": PRICE_FEATURES,
        "risk_feature": "formal HAR-X + News five-session OOF sigma",
        "embargo_sessions": HORIZON,
        "historical": {
            "rows": int(len(historical)),
            "symbols": int(historical["symbol"].nunique()),
            "start": str(historical["date"].min().date()),
            "end": str(historical["date"].max().date()),
            "sha256": _sha256(OUTPUT),
        },
        "external": {
            "rows": int(len(external)),
            "symbols": int(external["symbol"].nunique()),
            "start": str(external["date"].min().date()),
            "end": str(external["date"].max().date()),
            "sha256": _sha256(EXTERNAL_OUTPUT),
        },
        "source_hashes": {
            str(TRAINING_DATA.relative_to(ROOT)): _sha256(TRAINING_DATA),
            str(RISK_OOF.relative_to(ROOT)): _sha256(RISK_OOF),
            str(CURRENT_OHLC.relative_to(ROOT)): _sha256(CURRENT_OHLC),
            str(CURRENT_RISK.relative_to(ROOT)): _sha256(CURRENT_RISK),
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"historical decision panel: {len(historical):,} rows, "
        f"{historical['symbol'].nunique()} symbols -> "
        f"{OUTPUT.relative_to(ROOT)}"
    )
    print(
        f"external decision panel:   {len(external):,} rows, "
        f"{external['symbol'].nunique()} symbols -> "
        f"{EXTERNAL_OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
