"""Direction AUC ablation with price, news, and HAR risk features.

The existing ``training_dataset.parquet`` contains price/news features but not
the online risk engine's output.  This script reconstructs the exact HAR input
features from the local OHLC files, applies the serialized risk model, and
merges two non-redundant risk features into a derived dataset:

  risk_sigma_{h}d  forecast h-day volatility
  risk_level_{h}d expanding percentile of that forecast using history up to t

VaR, ES, and interval width are deliberately omitted because they are fixed
multiples of sigma and would duplicate the same information.

Evaluation is a single honest temporal holdout aligned with the risk engine:

  classifier train: through 2020-12-02 (30-day embargo before the seam)
  classifier test:  2021-01-01 through 2023-11-29

The serialized HAR coefficients were fit on 2013-2018 and selected/calibrated
on 2019-2020, so they are never fit on the direction test period.  Results use
the same deterministic HistGradientBoostingClassifier as train_ablation.py.

Outputs:
  data/processed/training_dataset_with_risk.parquet
  reports/direction_auc_risk/auc_results.csv
  reports/direction_auc_risk/auc_lift_bootstrap.csv

Run:
  python scripts/train_direction_risk_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "training_dataset.parquet"
MODEL = ROOT / "data" / "processed" / "risk_model.json"
PRICES = ROOT / "FNSPID" / "final_dataset" / "prices"
AUGMENTED = ROOT / "data" / "processed" / "training_dataset_with_risk.parquet"
OUT = ROOT / "reports" / "direction_auc_risk"

PRICE = [
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
NEWS = ["sentiment", "news_count", "has_news"]
HORIZONS = {5: "label_up_5d", 20: "label_up_20d"}
TEST_START = pd.Timestamp("2021-01-01")
TRAIN_END = TEST_START - pd.Timedelta(days=30)  # >20 trading-label days
EPS = 1e-6
BOOTSTRAP_REPS = 500
BOOTSTRAP_BLOCK_DAYS = 20


def new_model() -> HistGradientBoostingClassifier:
    """Keep the classifier identical to the existing direction ablation."""
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )


def _har_inputs(px: pd.DataFrame) -> pd.DataFrame:
    """Exact online HAR inputs from risk_engine.engine._feature_frame."""
    px = px.sort_values("date").set_index("date")
    close, high, low = px["adj close"], px["high"], px["low"]
    ret = close.pct_change()
    park = (np.log(high / low) ** 2) / (4 * np.log(2))
    out = pd.DataFrame(index=close.index)
    out["l_rv5"] = np.log(ret.rolling(5).std() + EPS)
    out["l_rv22"] = np.log(ret.rolling(22).std() + EPS)
    out["l_rv66"] = np.log(ret.rolling(66).std() + EPS)
    out["l_park5"] = np.log(np.sqrt(park.rolling(5).mean()) + EPS)
    out["l_park22"] = np.log(np.sqrt(park.rolling(22).mean()) + EPS)
    out["l_absret"] = np.log(ret.abs() + EPS)
    return out


def build_risk_features(symbols: Iterable[str], model: dict) -> pd.DataFrame:
    """Apply the frozen HAR model and calculate a no-look-ahead risk percentile."""
    rows = []
    for symbol in sorted(symbols):
        path = PRICES / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing OHLC file for {symbol}: {path}")
        px = pd.read_csv(path, usecols=["date", "adj close", "high", "low"])
        px["date"] = pd.to_datetime(px["date"])
        frame = _har_inputs(px)
        result = pd.DataFrame(index=frame.index)
        result["symbol"] = symbol
        for horizon in HORIZONS:
            mh = model["horizons"][str(horizon)]
            log_sigma = mh["intercept"] + sum(
                frame[name] * mh["coef"][name] for name in mh["features"]
            )
            sigma_h = np.exp(log_sigma) * mh["smearing"] * np.sqrt(horizon)
            result[f"risk_sigma_{horizon}d"] = sigma_h
            # Expanding.rank reports the current observation's percentile within
            # forecasts available through the current date only.
            result[f"risk_level_{horizon}d"] = (
                sigma_h.expanding(min_periods=1).rank(pct=True) * 100.0
            )
        result.index.name = "date"
        rows.append(result.reset_index())
    return pd.concat(rows, ignore_index=True)


def feature_sets(horizon: int) -> Dict[str, list[str]]:
    risk = [f"risk_sigma_{horizon}d", f"risk_level_{horizon}d"]
    return {
        "risk_only": risk,
        "price_only": PRICE,
        "price+news": PRICE + NEWS,
        "price+risk": PRICE + risk,
        "price+news+risk": PRICE + NEWS + risk,
    }


def moving_block_indices(
    dates: pd.Series,
    rng: np.random.Generator,
    block_days: int,
) -> np.ndarray:
    """Resample date blocks, preserving the cross-section within each date."""
    unique_dates = np.sort(dates.unique())
    positions = {
        date: np.flatnonzero(dates.to_numpy() == date) for date in unique_dates
    }
    sampled_dates = []
    max_start = max(1, len(unique_dates) - block_days + 1)
    while len(sampled_dates) < len(unique_dates):
        start = int(rng.integers(0, max_start))
        sampled_dates.extend(unique_dates[start : start + block_days])
    sampled_dates = sampled_dates[: len(unique_dates)]
    return np.concatenate([positions[date] for date in sampled_dates])


def bootstrap_lift(
    test_dates: pd.Series,
    y: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[float, float, float]:
    """20-trading-day moving-block bootstrap CI for paired AUC lift."""
    rng = np.random.default_rng(20260723)
    lifts = []
    for _ in range(BOOTSTRAP_REPS):
        idx = moving_block_indices(test_dates, rng, BOOTSTRAP_BLOCK_DAYS)
        if np.unique(y[idx]).size < 2:
            continue
        lifts.append(
            roc_auc_score(y[idx], candidate[idx])
            - roc_auc_score(y[idx], reference[idx])
        )
    lo, median, hi = np.quantile(lifts, [0.025, 0.5, 0.975])
    return float(lo), float(median), float(hi)


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df["date"] <= TRAIN_END
    test = df["date"] >= TEST_START
    result_rows = []
    lift_rows = []

    print("=" * 88)
    print("DIRECTION AUC · PRICE / NEWS / HAR RISK · TEMPORAL HOLDOUT")
    print(
        f"train through {TRAIN_END.date()} ({train.sum():,} rows) · "
        f"test {TEST_START.date()}–{df.loc[test, 'date'].max().date()} "
        f"({test.sum():,} rows)"
    )
    print("=" * 88)

    for horizon, label in HORIZONS.items():
        y_train = df.loc[train, label].astype(int).to_numpy()
        y_test = df.loc[test, label].astype(int).to_numpy()
        test_dates = df.loc[test, "date"].reset_index(drop=True)
        predictions = {}
        sets = feature_sets(horizon)

        for name, columns in sets.items():
            model = new_model()
            model.fit(df.loc[train, columns], y_train)
            predictions[name] = model.predict_proba(df.loc[test, columns])[:, 1]

        print(f"\nNext {horizon} trading days ({label})")
        print(f"{'feature set':>20} | {'pooled':>8} | {'2021':>8} | {'2022':>8} | {'2023':>8}")
        print("-" * 65)
        for name, prediction in predictions.items():
            pooled = roc_auc_score(y_test, prediction)
            per_year = {}
            for year in (2021, 2022, 2023):
                mask = test_dates.dt.year.to_numpy() == year
                per_year[year] = roc_auc_score(y_test[mask], prediction[mask])
            print(
                f"{name:>20} | {pooled:>8.4f} | {per_year[2021]:>8.4f} | "
                f"{per_year[2022]:>8.4f} | {per_year[2023]:>8.4f}"
            )
            result_rows.append(
                {
                    "horizon": horizon,
                    "feature_set": name,
                    "auc_pooled": pooled,
                    **{f"auc_{year}": value for year, value in per_year.items()},
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    "train_end": TRAIN_END.date().isoformat(),
                    "test_start": TEST_START.date().isoformat(),
                }
            )

        reference = predictions["price_only"]
        ref_auc = roc_auc_score(y_test, reference)
        print("\n  Paired lift over price_only (20-day moving-block bootstrap):")
        for name in ("price+news", "price+risk", "price+news+risk"):
            candidate = predictions[name]
            lift = roc_auc_score(y_test, candidate) - ref_auc
            lo, median, hi = bootstrap_lift(
                test_dates, y_test, reference, candidate
            )
            print(
                f"  {name:>17}: ΔAUC {lift:+.4f} · "
                f"95% CI [{lo:+.4f}, {hi:+.4f}]"
            )
            lift_rows.append(
                {
                    "horizon": horizon,
                    "candidate": name,
                    "reference": "price_only",
                    "auc_lift": lift,
                    "bootstrap_median": median,
                    "ci_2.5": lo,
                    "ci_97.5": hi,
                    "bootstrap_reps": BOOTSTRAP_REPS,
                    "block_days": BOOTSTRAP_BLOCK_DAYS,
                }
            )

    return pd.DataFrame(result_rows), pd.DataFrame(lift_rows)


def main() -> None:
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    model = json.loads(MODEL.read_text(encoding="utf-8"))

    risk = build_risk_features(df["symbol"].unique(), model)
    augmented = df.merge(risk, on=["date", "symbol"], how="left", validate="one_to_one")
    risk_columns = [
        f"risk_{kind}_{horizon}d"
        for horizon in HORIZONS
        for kind in ("sigma", "level")
    ]
    missing = augmented[risk_columns].isna().sum()
    if missing.any():
        raise ValueError(f"Risk-feature merge produced missing values:\n{missing}")

    AUGMENTED.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(AUGMENTED, index=False)
    results, lifts = evaluate(augmented)

    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "auc_results.csv", index=False)
    lifts.to_csv(OUT / "auc_lift_bootstrap.csv", index=False)
    print(f"\nAugmented data -> {AUGMENTED.relative_to(ROOT)}")
    print(f"AUC results    -> {(OUT / 'auc_results.csv').relative_to(ROOT)}")
    print(f"Lift intervals -> {(OUT / 'auc_lift_bootstrap.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
