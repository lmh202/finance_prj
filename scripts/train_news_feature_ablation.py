"""Walk-forward ablation for legacy, extended, and coverage-aware news features.

The experiment compares four feature sets:

1. price_only
2. price+legacy_news
3. price+extended_news
4. price+extended_news+causal_state

The stored ``coverage_active`` column is deliberately excluded. It was inferred
from each symbol's first and last FNSPID article and therefore uses a future
endpoint. Instead, three mutually exclusive, causal states are constructed
using only news observed through the current trading day:

* news_state_uncovered: no article has ever been observed for the symbol
* news_state_silent: no article today, but at least one in the last 60 sessions
* news_state_stale: coverage once started, but no article in the last 60 sessions

``has_news`` is already the fourth state (news observed today).

Validation uses expanding yearly walk-forward folds. Training rows whose
forward-return labels overlap the test year are embargoed.

Outputs (reports/news_feature_ablation/):
  fold_metrics.csv
  summary_metrics.csv
  auc_lift_bootstrap.csv
  oof_predictions.parquet
  feature_sets.json

Run:
  python scripts/train_news_feature_ablation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "training_dataset.parquet"
EXTENDED_DATA = ROOT / "data" / "processed" / "extended_news_features.parquet"
OUT = ROOT / "reports" / "news_feature_ablation"

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
LEGACY_NEWS = ["sentiment", "news_count", "has_news"]
CAUSAL_STATE = [
    "news_state_uncovered",
    "news_state_silent",
    "news_state_stale",
]
HORIZONS = {5: "label_up_5d", 20: "label_up_20d"}
DEFAULT_FIRST_TEST_YEAR = 2015
DEFAULT_BOOTSTRAP_REPS = 500
BOOTSTRAP_BLOCK_DAYS = 20
RANDOM_SEED = 20260723


def unique_in_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def new_model() -> HistGradientBoostingClassifier:
    """Keep the learner fixed so the comparison isolates feature content."""
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )


def prepare_dataset() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    if not EXTENDED_DATA.exists():
        raise FileNotFoundError(EXTENDED_DATA)

    frame = pd.read_parquet(DATA)
    extended_schema = pd.read_parquet(EXTENDED_DATA).columns.tolist()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    extended_news = unique_in_order(
        ["sentiment"]
        + [
            column
            for column in extended_schema
            if column not in {"date", "symbol", "coverage_active"}
        ]
    )
    required = set(PRICE + LEGACY_NEWS + extended_news + list(HORIZONS.values()))
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"training dataset is missing required columns: {missing}")

    # Build coverage state in symbol-date order, using no future observations.
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    coverage_started = (
        frame.groupby("symbol", sort=False)["has_news"].cummax().astype(int)
    )
    recent_news = (
        frame.groupby("symbol", sort=False)["has_news"]
        .rolling(60, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .astype(int)
    )
    no_news = frame["has_news"].eq(0)
    frame["news_state_uncovered"] = (
        no_news & coverage_started.eq(0)
    ).astype(int)
    frame["news_state_silent"] = (
        no_news & recent_news.eq(1)
    ).astype(int)
    frame["news_state_stale"] = (
        no_news & coverage_started.eq(1) & recent_news.eq(0)
    ).astype(int)

    state_sum = frame[CAUSAL_STATE].sum(axis=1) + frame["has_news"]
    if not state_sum.eq(1).all():
        raise AssertionError("causal news states are not mutually exhaustive")

    feature_sets = {
        "price_only": PRICE,
        "price+legacy_news": unique_in_order(PRICE + LEGACY_NEWS),
        "price+extended_news": unique_in_order(PRICE + extended_news),
        "price+extended_news+causal_state": unique_in_order(
            PRICE + extended_news + CAUSAL_STATE
        ),
    }
    all_features = unique_in_order(
        column for columns in feature_sets.values() for column in columns
    )
    if frame[all_features].isna().any().any():
        bad = frame[all_features].isna().sum()
        raise ValueError(
            f"feature matrix contains missing values: "
            f"{bad[bad.gt(0)].to_dict()}"
        )

    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    return frame, feature_sets


def fold_masks(
    frame: pd.DataFrame,
    test_year: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Create an expanding split with an h-session label embargo."""
    test = frame["date"].dt.year.eq(test_year).to_numpy()
    if not test.any():
        raise ValueError(f"no rows for test year {test_year}")

    unique_dates = np.sort(frame["date"].unique())
    first_test_date = frame.loc[test, "date"].min()
    first_test_position = int(
        np.searchsorted(unique_dates, np.datetime64(first_test_date))
    )
    cutoff_position = first_test_position - horizon - 1
    if cutoff_position < 0:
        raise ValueError(
            f"not enough history for {horizon}-session embargo before {test_year}"
        )
    train_cutoff = pd.Timestamp(unique_dates[cutoff_position])
    train = frame["date"].le(train_cutoff).to_numpy()
    return train, test, train_cutoff


def evaluate(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    first_test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    years = [
        year
        for year in range(first_test_year, int(frame["date"].dt.year.max()) + 1)
        if frame["date"].dt.year.eq(year).any()
    ]
    fold_rows: list[dict[str, object]] = []
    predictions: dict[int, dict[str, np.ndarray]] = {}

    print("=" * 104)
    print("NEWS FEATURE ABLATION | EXPANDING WALK-FORWARD | EMBARGOED LABELS")
    print(
        f"{len(frame):,} rows | {frame['symbol'].nunique()} symbols | "
        f"test years {years[0]}-{years[-1]} | "
        f"{len(feature_sets)} feature sets"
    )
    print("=" * 104)

    for horizon, label in HORIZONS.items():
        y_all = frame[label].astype(int).to_numpy()
        predictions[horizon] = {
            name: np.full(len(frame), np.nan, dtype=float)
            for name in feature_sets
        }
        print(f"\nHorizon: {horizon} trading days ({label})")

        for test_year in years:
            train, test, train_cutoff = fold_masks(frame, test_year, horizon)
            y_train, y_test = y_all[train], y_all[test]
            if len(y_train) < 500 or len(y_test) < 100:
                continue
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue

            fold_aucs = {}
            for name, columns in feature_sets.items():
                model = new_model()
                model.fit(frame.loc[train, columns], y_train)
                probability = model.predict_proba(frame.loc[test, columns])[:, 1]
                predictions[horizon][name][test] = probability

                auc = roc_auc_score(y_test, probability)
                fold_aucs[name] = auc
                fold_rows.append(
                    {
                        "horizon": horizon,
                        "test_year": test_year,
                        "feature_set": name,
                        "feature_count": len(columns),
                        "auc": auc,
                        "brier": brier_score_loss(y_test, probability),
                        "log_loss": log_loss(
                            y_test, probability, labels=[0, 1]
                        ),
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "train_cutoff": train_cutoff.date().isoformat(),
                        "test_start": frame.loc[test, "date"].min().date().isoformat(),
                        "test_end": frame.loc[test, "date"].max().date().isoformat(),
                    }
                )

            extended_lift = (
                fold_aucs["price+extended_news"]
                - fold_aucs["price_only"]
            )
            causal_lift = (
                fold_aucs["price+extended_news+causal_state"]
                - fold_aucs["price_only"]
            )
            print(
                f"  {test_year}: train<={train_cutoff.date()} "
                f"({train.sum():>6,}) test={test.sum():>5,} | "
                f"price {fold_aucs['price_only']:.4f} | "
                f"extended {fold_aucs['price+extended_news']:.4f} "
                f"({extended_lift:+.4f}) | "
                f"causal {fold_aucs['price+extended_news+causal_state']:.4f} "
                f"({causal_lift:+.4f})"
            )

    return pd.DataFrame(fold_rows), frame, predictions


def summarise(
    folds: pd.DataFrame,
    frame: pd.DataFrame,
    predictions: dict[int, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon, label in HORIZONS.items():
        y = frame[label].astype(int).to_numpy()
        reference_folds = (
            folds[
                (folds["horizon"] == horizon)
                & (folds["feature_set"] == "price_only")
            ]
            .set_index("test_year")["auc"]
        )
        for name, probability in predictions[horizon].items():
            selected = folds[
                (folds["horizon"] == horizon)
                & (folds["feature_set"] == name)
            ].copy()
            mask = ~np.isnan(probability)
            selected["auc_lift_vs_price"] = (
                selected["auc"]
                - selected["test_year"].map(reference_folds)
            )
            rows.append(
                {
                    "horizon": horizon,
                    "feature_set": name,
                    "feature_count": int(selected["feature_count"].iloc[0]),
                    "folds": len(selected),
                    "mean_fold_auc": selected["auc"].mean(),
                    "std_fold_auc": selected["auc"].std(ddof=1),
                    "pooled_oof_auc": roc_auc_score(y[mask], probability[mask]),
                    "pooled_oof_brier": brier_score_loss(
                        y[mask], probability[mask]
                    ),
                    "pooled_oof_log_loss": log_loss(
                        y[mask], probability[mask], labels=[0, 1]
                    ),
                    "mean_fold_auc_lift_vs_price": (
                        selected["auc_lift_vs_price"].mean()
                    ),
                    "positive_years_vs_price": int(
                        selected["auc_lift_vs_price"].gt(0).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def date_position_groups(dates: pd.Series) -> list[np.ndarray]:
    """Cache row positions for each date in an already time-sorted fold."""
    values = dates.to_numpy()
    _, inverse = np.unique(values, return_inverse=True)
    return [
        np.flatnonzero(inverse == index)
        for index in range(int(inverse.max()) + 1)
    ]


def moving_block_indices(
    positions: list[np.ndarray],
    rng: np.random.Generator,
    block_days: int,
) -> np.ndarray:
    sampled_positions: list[np.ndarray] = []
    sampled_day_count = 0
    max_start = max(1, len(positions) - block_days + 1)
    while sampled_day_count < len(positions):
        start = int(rng.integers(0, max_start))
        block = positions[start : start + block_days]
        sampled_positions.extend(block)
        sampled_day_count += len(block)
    sampled_positions = sampled_positions[: len(positions)]
    return np.concatenate(sampled_positions)


def bootstrap_lifts(
    oof: pd.DataFrame,
    reps: int,
) -> pd.DataFrame:
    """Bootstrap the mean annual-fold AUC lift without pooling model scales."""
    rows: list[dict[str, object]] = []
    probability_columns = [
        column
        for column in oof.columns
        if column.startswith("probability__")
    ]
    reference_column = "probability__price_only"
    candidates = [
        column.removeprefix("probability__")
        for column in probability_columns
        if column != reference_column
    ]

    for horizon in HORIZONS:
        horizon_oof = oof[oof["horizon"] == horizon].copy()
        horizon_oof["test_year"] = horizon_oof["date"].dt.year
        fold_data = []
        observed_fold_lifts = {name: [] for name in candidates}

        for _, fold in horizon_oof.groupby("test_year", sort=True):
            fold = fold.sort_values(["date", "symbol"]).reset_index(drop=True)
            y = fold["label"].astype(int).to_numpy()
            probabilities = {
                column.removeprefix("probability__"): fold[column].to_numpy()
                for column in probability_columns
            }
            reference_auc = roc_auc_score(y, probabilities["price_only"])
            for name in candidates:
                observed_fold_lifts[name].append(
                    roc_auc_score(y, probabilities[name]) - reference_auc
                )
            fold_data.append(
                {
                    "y": y,
                    "probabilities": probabilities,
                    "positions": date_position_groups(fold["date"]),
                }
            )

        candidates = [
            name for name in observed_fold_lifts
        ]
        samples = {name: [] for name in candidates}
        rng = np.random.default_rng(RANDOM_SEED + horizon)

        for _ in range(reps):
            replicate_lifts = {name: [] for name in candidates}
            valid_replicate = True
            for fold in fold_data:
                index = moving_block_indices(
                    fold["positions"], rng, BOOTSTRAP_BLOCK_DAYS
                )
                y = fold["y"]
                probabilities = fold["probabilities"]
                if np.unique(y[index]).size < 2:
                    valid_replicate = False
                    break
                reference_auc = roc_auc_score(
                    y[index], probabilities["price_only"][index]
                )
                for name in candidates:
                    replicate_lifts[name].append(
                        roc_auc_score(y[index], probabilities[name][index])
                        - reference_auc
                    )
            if not valid_replicate:
                continue
            for name in candidates:
                samples[name].append(float(np.mean(replicate_lifts[name])))

        for name in candidates:
            lifts = np.asarray(samples[name])
            low, median, high = np.quantile(lifts, [0.025, 0.5, 0.975])
            observed = float(np.mean(observed_fold_lifts[name]))
            rows.append(
                {
                    "horizon": horizon,
                    "candidate": name,
                    "reference": "price_only",
                    "observed_mean_fold_auc_lift": observed,
                    "bootstrap_median": median,
                    "ci_2.5": low,
                    "ci_97.5": high,
                    "bootstrap_reps": len(lifts),
                    "block_days": BOOTSTRAP_BLOCK_DAYS,
                }
            )
    return pd.DataFrame(rows)


def build_oof_output(
    frame: pd.DataFrame,
    predictions: dict[int, dict[str, np.ndarray]],
) -> pd.DataFrame:
    parts = []
    for horizon, label in HORIZONS.items():
        mask = ~np.isnan(predictions[horizon]["price_only"])
        part = frame.loc[mask, ["date", "symbol", label]].copy()
        part.insert(2, "horizon", horizon)
        part = part.rename(columns={label: "label"})
        for name, values in predictions[horizon].items():
            part[f"probability__{name}"] = values[mask]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def save_outputs(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    oof: pd.DataFrame,
    first_test_year: int,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    folds.to_csv(OUT / "fold_metrics.csv", index=False)
    summary.to_csv(OUT / "summary_metrics.csv", index=False)
    bootstrap.to_csv(OUT / "auc_lift_bootstrap.csv", index=False)
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False)

    config = {
        "dataset": str(DATA.relative_to(ROOT)),
        "rows": len(frame),
        "symbols": int(frame["symbol"].nunique()),
        "date_range": [
            frame["date"].min().date().isoformat(),
            frame["date"].max().date().isoformat(),
        ],
        "first_test_year": first_test_year,
        "horizons": HORIZONS,
        "embargo": (
            "For horizon h, training ends strictly more than h trading "
            "sessions before the first test date."
        ),
        "excluded_noncausal_feature": "coverage_active",
        "causal_state_definition": {
            "news_state_uncovered": "no news observed up to and including t",
            "news_state_silent": (
                "no news at t, but news observed within the trailing 60 sessions"
            ),
            "news_state_stale": (
                "news observed previously, but none in the trailing 60 sessions"
            ),
            "has_news": "news observed at t",
        },
        "feature_sets": feature_sets,
        "model": {
            "class": "HistGradientBoostingClassifier",
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": 0,
        },
    }
    with (OUT / "feature_sets.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)


def print_summary(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    print("\n" + "=" * 104)
    print("SUMMARY")
    print("=" * 104)
    for horizon in HORIZONS:
        print(f"\nHorizon: {horizon} trading days")
        subset = summary[summary["horizon"] == horizon]
        for row in subset.itertuples(index=False):
            print(
                f"  {row.feature_set:<38} "
                f"mean AUC {row.mean_fold_auc:.4f} | "
                f"pooled {row.pooled_oof_auc:.4f} | "
                f"mean lift {row.mean_fold_auc_lift_vs_price:+.4f} | "
                f"positive years {row.positive_years_vs_price}/{row.folds} | "
                f"Brier {row.pooled_oof_brier:.4f}"
            )
        print("  Moving-block bootstrap, mean annual-fold AUC lift over price_only:")
        for row in bootstrap[bootstrap["horizon"] == horizon].itertuples(
            index=False
        ):
            print(
                f"    {row.candidate:<36} "
                f"{row.observed_mean_fold_auc_lift:+.4f} "
                f"[{getattr(row, '_5'):+.4f}, {getattr(row, '_6'):+.4f}]"
            )
    print(f"\nSaved reports to: {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--first-test-year",
        type=int,
        default=DEFAULT_FIRST_TEST_YEAR,
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
    )
    args = parser.parse_args()

    frame, feature_sets = prepare_dataset()
    print(
        "Feature counts: "
        + ", ".join(
            f"{name}={len(columns)}"
            for name, columns in feature_sets.items()
        )
    )
    folds, frame, predictions = evaluate(
        frame, feature_sets, args.first_test_year
    )
    summary = summarise(folds, frame, predictions)
    oof = build_oof_output(frame, predictions)
    bootstrap = bootstrap_lifts(oof, args.bootstrap_reps)
    save_outputs(
        frame,
        feature_sets,
        folds,
        summary,
        bootstrap,
        oof,
        args.first_test_year,
    )
    print_summary(summary, bootstrap)


if __name__ == "__main__":
    main()
