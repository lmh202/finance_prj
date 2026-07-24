"""Test a regularised news residual layer on the price-only XGBoost model.

The direct Price + News XGBoost experiment can dilute a strong price signal
with sparse news inputs. This complementary experiment keeps the existing
price XGBoost specification fixed and asks whether news can explain its
remaining variance ratio:

    sigma = sigma_xgb_price * GammaRatio(news features)

News family and regularisation strength are selected using inner folds only.
Zero-news observations remain in every fit and naturally receive a unit
multiplier when all selected news inputs are zero.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from optimize_risk_engine import (  # noqa: E402
    _xgb_forecast,
    _xgb_target,
)
from risk_engine_optimization import (  # noqa: E402
    CAUSAL_NEWS_STATES,
    DEPLOYABLE_NEWS_BASE,
    OUTER_YEARS,
    PRICE_FEATURES,
    RANDOM_SEED,
    dm_test,
    fit_gamma_overlay,
    fit_har,
    fit_xgb_gamma_fixed,
    inner_folds,
    moving_block_bootstrap_gain,
    overlay_forecast,
    parameter_configs,
    qlike_loss,
    symbol_equal_qlike,
    time_fold,
)
from test_news_features_in_xgboost import (  # noqa: E402
    ATTENTION_FEATURES,
    FINBERT_FEATURES,
)

PANEL_PATH = ROOT / "data" / "processed" / "risk_optimization_panel.parquet"
SEARCH_PATH = (
    ROOT / "reports" / "risk_engine_optimization" / "search_results.csv"
)
BASE_OOF_PATH = (
    ROOT / "reports" / "risk_engine_optimization" / "oof_predictions.parquet"
)
OUTPUT = ROOT / "reports" / "risk_engine_xgb_news"
HORIZON = 5
TARGET = "realized_vol_5d"
BASE_FORECAST_COLUMN = "forecast__xgb_price"
DEVICE = "cuda"
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)

ALL_DEPLOYABLE = list(
    dict.fromkeys(
        DEPLOYABLE_NEWS_BASE
        + CAUSAL_NEWS_STATES
        + ["news_quality_available"]
    )
)
NEWS_FAMILIES = {
    "log_count": ["log_count"],
    "attention": ATTENTION_FEATURES,
    "attention_finbert": list(
        dict.fromkeys(ATTENTION_FEATURES + FINBERT_FEATURES)
    ),
    "all_deployable": ALL_DEPLOYABLE,
}


def specifications() -> dict[int, dict[str, object]]:
    search = pd.read_csv(SEARCH_PATH)
    rows = search[
        search["horizon"].eq(HORIZON)
        & search["stage"].eq("variant_selection")
    ]
    configs = {
        int(config["config_id"]): config
        for config in parameter_configs(96)
    }
    result = {}
    for row in rows.itertuples(index=False):
        config_id = int(row.config_id)
        result[int(row.outer_year)] = {
            "config": configs[config_id],
            "variant": str(row.variant),
            "n_estimators": int(row.iterations),
            "blend_weight": float(row.blend_weight),
            "scale": float(row.scale),
        }
    return result


def price_forecasts(
    train: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    specification: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    har = fit_har(train, HORIZON)
    har_train = har.predict(train)
    har_forecast = har.predict(forecast_frame)
    variant = str(specification["variant"])
    model = fit_xgb_gamma_fixed(
        train,
        PRICE_FEATURES,
        _xgb_target(train, HORIZON, variant, har_train),
        specification["config"],
        DEVICE,
        int(specification["n_estimators"]),
    )
    raw_train = _xgb_forecast(
        model,
        train,
        PRICE_FEATURES,
        variant,
        har_train,
    )
    raw_forecast = _xgb_forecast(
        model,
        forecast_frame,
        PRICE_FEATURES,
        variant,
        har_forecast,
    )
    weight = float(specification["blend_weight"])
    scale = float(specification["scale"])
    base_train = ((1 - weight) * har_train + weight * raw_train) * scale
    base_forecast = (
        (1 - weight) * har_forecast + weight * raw_forecast
    ) * scale
    return base_train, base_forecast


def inner_frames(
    panel: pd.DataFrame,
    fold,
    outer_cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = panel.loc[fold.train_mask].dropna(
        subset=PRICE_FEATURES + [TARGET]
    )
    validation_mask = fold.test_mask & panel["date"].le(
        outer_cutoff
    ).to_numpy()
    validation = panel.loc[validation_mask].dropna(
        subset=PRICE_FEATURES + [TARGET]
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def select_overlay(
    panel: pd.DataFrame,
    folds,
    outer_cutoff: pd.Timestamp,
    specification: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cached = []
    for fold in folds:
        train, validation = inner_frames(panel, fold, outer_cutoff)
        if validation.empty:
            continue
        base_train, base_validation = price_forecasts(
            train,
            validation,
            specification,
        )
        cached.append(
            (train, validation, base_train, base_validation)
        )

    rows = []
    for family, features in NEWS_FAMILIES.items():
        for alpha in ALPHAS:
            scores = []
            gains = []
            for train, validation, base_train, base_validation in cached:
                overlay = fit_gamma_overlay(
                    train,
                    base_train,
                    HORIZON,
                    features,
                    alpha,
                )
                candidate = overlay_forecast(
                    base_validation,
                    overlay,
                    validation,
                )
                base_score = symbol_equal_qlike(
                    validation,
                    TARGET,
                    base_validation,
                )
                candidate_score = symbol_equal_qlike(
                    validation,
                    TARGET,
                    candidate,
                )
                scores.append(candidate_score)
                gains.append(
                    (base_score - candidate_score) / base_score
                )
            rows.append(
                {
                    "news_family": family,
                    "news_feature_count": len(features),
                    "alpha": alpha,
                    "mean_inner_qlike": float(np.mean(scores)),
                    "median_gain": float(np.median(gains)),
                    "mean_gain": float(np.mean(gains)),
                    "positive_fold_share": float(
                        np.mean(np.asarray(gains) > 0)
                    ),
                    "selected": False,
                }
            )
    selected = choose_best(rows)
    for row in rows:
        if (
            row["news_family"] == selected["news_family"]
            and row["alpha"] == selected["alpha"]
        ):
            row["selected"] = True
    return selected, rows


def choose_best(rows: list[dict[str, object]]) -> dict[str, object]:
    stable = [
        row
        for row in rows
        if row["positive_fold_share"] >= 0.60
        and row["median_gain"] > 0
    ]
    pool = stable if stable else rows
    selected = min(
        pool,
        key=lambda row: (
            -row["median_gain"],
            -row["mean_gain"],
            row["mean_inner_qlike"],
            row["news_feature_count"],
        ),
    )
    return selected


def simple_candidate_summary(
    frame: pd.DataFrame,
    column: str,
) -> dict[str, object]:
    target = frame[TARGET].to_numpy(float)
    reference = frame["forecast__xgb_price_rebuilt"].to_numpy(float)
    candidate = frame[column].to_numpy(float)
    reference_loss = qlike_loss(target, reference)
    candidate_loss = qlike_loss(target, candidate)
    reference_score = symbol_equal_qlike(frame, TARGET, reference)
    candidate_score = symbol_equal_qlike(frame, TARGET, candidate)
    dm_t, dm_p = dm_test(
        frame["date"],
        candidate_loss,
        reference_loss,
        HORIZON,
    )
    yearly_gains = {}
    for year, group in frame.groupby("test_year"):
        ref = symbol_equal_qlike(
            group,
            TARGET,
            group["forecast__xgb_price_rebuilt"].to_numpy(float),
        )
        cand = symbol_equal_qlike(
            group,
            TARGET,
            group[column].to_numpy(float),
        )
        yearly_gains[str(int(year))] = (ref - cand) / ref

    high_cutoff = float(frame[TARGET].quantile(2 / 3))
    scope_gains = {}
    for scope, mask in {
        "news_active": frame["has_news"].eq(1),
        "zero_news": frame["has_news"].eq(0),
        "high_volatility": frame[TARGET].ge(high_cutoff),
    }.items():
        group = frame.loc[mask]
        ref = symbol_equal_qlike(
            group,
            TARGET,
            group["forecast__xgb_price_rebuilt"].to_numpy(float),
        )
        cand = symbol_equal_qlike(
            group,
            TARGET,
            group[column].to_numpy(float),
        )
        scope_gains[scope] = (ref - cand) / ref
    return {
        "forecast_column": column,
        "reference_qlike": reference_score,
        "candidate_qlike": candidate_score,
        "relative_gain": (
            reference_score - candidate_score
        )
        / reference_score,
        "dm_t": dm_t,
        "dm_p": dm_p,
        "positive_years": int(
            sum(value > 0 for value in yearly_gains.values())
        ),
        "yearly_gains": yearly_gains,
        **{f"{key}_gain": value for key, value in scope_gains.items()},
    }


def score(frame: pd.DataFrame, column: str) -> float:
    return symbol_equal_qlike(
        frame,
        TARGET,
        frame[column].to_numpy(float),
    )


def summarise(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    target = frame[TARGET].to_numpy(float)
    reference = frame["forecast__xgb_price_rebuilt"].to_numpy(float)
    candidate = frame["forecast__xgb_price_news_overlay"].to_numpy(float)
    reference_loss = qlike_loss(target, reference)
    candidate_loss = qlike_loss(target, candidate)
    reference_score = symbol_equal_qlike(frame, TARGET, reference)
    candidate_score = symbol_equal_qlike(frame, TARGET, candidate)
    dm_t, dm_p = dm_test(
        frame["date"],
        candidate_loss,
        reference_loss,
        HORIZON,
    )
    scored = pd.DataFrame(
        {
            "date": frame["date"].to_numpy(),
            "reference_loss": reference_loss,
            "candidate_loss": candidate_loss,
        }
    ).reset_index(drop=True)
    bootstrap = moving_block_bootstrap_gain(
        scored,
        reps=2_000,
        block_days=HORIZON,
        seed=RANDOM_SEED + 2,
    )
    yearly = {}
    for year, group in frame.groupby("test_year"):
        ref = score(group, "forecast__xgb_price_rebuilt")
        cand = score(group, "forecast__xgb_price_news_overlay")
        yearly[str(int(year))] = (ref - cand) / ref

    symbol_rows = []
    for symbol, group in frame.groupby("symbol"):
        ref = score(group, "forecast__xgb_price_rebuilt")
        cand = score(group, "forecast__xgb_price_news_overlay")
        symbol_rows.append(
            {
                "symbol": symbol,
                "reference_qlike": ref,
                "candidate_qlike": cand,
                "relative_gain": (ref - cand) / ref,
            }
        )
    symbols = pd.DataFrame(symbol_rows)
    high_cutoff = float(frame[TARGET].quantile(2 / 3))
    regime_rows = []
    for scope, mask in {
        "all": pd.Series(True, index=frame.index),
        "news_active": frame["has_news"].eq(1),
        "zero_news": frame["has_news"].eq(0),
        "high_volatility": frame[TARGET].ge(high_cutoff),
    }.items():
        group = frame.loc[mask].reset_index(drop=True)
        ref = score(group, "forecast__xgb_price_rebuilt")
        cand = score(group, "forecast__xgb_price_news_overlay")
        regime_rows.append(
            {
                "scope": scope,
                "rows": len(group),
                "reference_qlike": ref,
                "candidate_qlike": cand,
                "relative_gain": (ref - cand) / ref,
            }
        )
    regimes = pd.DataFrame(regime_rows)
    summary = {
        "reference_qlike": float(reference_score),
        "candidate_qlike": float(candidate_score),
        "relative_gain": float(
            (reference_score - candidate_score) / reference_score
        ),
        "dm_t": float(dm_t),
        "dm_p": float(dm_p),
        "bootstrap_95": list(bootstrap),
        "positive_years": int(
            sum(value > 0 for value in yearly.values())
        ),
        "yearly_gains": yearly,
        "positive_symbol_share": float(
            symbols["relative_gain"].gt(0).mean()
        ),
        "median_symbol_gain": float(
            symbols["relative_gain"].median()
        ),
        "news_active_gain": float(
            regimes.loc[
                regimes["scope"].eq("news_active"), "relative_gain"
            ].iloc[0]
        ),
        "zero_news_gain": float(
            regimes.loc[
                regimes["scope"].eq("zero_news"), "relative_gain"
            ].iloc[0]
        ),
        "high_volatility_gain": float(
            regimes.loc[
                regimes["scope"].eq("high_volatility"),
                "relative_gain",
            ].iloc[0]
        ),
        "rows": len(frame),
        "symbols": int(frame["symbol"].nunique()),
    }
    return summary, symbols, regimes


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    specs = specifications()
    predictions = []
    selection_rows = []
    for outer_year in OUTER_YEARS:
        outer_fold = time_fold(panel, outer_year, HORIZON)
        train = panel.loc[outer_fold.train_mask].dropna(
            subset=PRICE_FEATURES + [TARGET]
        ).reset_index(drop=True)
        test = panel.loc[outer_fold.test_mask].dropna(
            subset=PRICE_FEATURES + [TARGET]
        ).reset_index(drop=True)
        selected, rows = select_overlay(
            panel,
            inner_folds(panel, outer_year, HORIZON),
            outer_fold.train_cutoff,
            specs[outer_year],
        )
        for row in rows:
            selection_rows.append(
                {"outer_year": outer_year, **row}
            )
        base_train, base_test = price_forecasts(
            train,
            test,
            specs[outer_year],
        )
        family_forecasts = {}
        family_alphas = {}
        for family, features in NEWS_FAMILIES.items():
            family_rows = [
                row
                for row in rows
                if row["news_family"] == family
            ]
            family_selected = choose_best(family_rows)
            overlay = fit_gamma_overlay(
                train,
                base_train,
                HORIZON,
                features,
                float(family_selected["alpha"]),
            )
            family_forecasts[family] = overlay_forecast(
                base_test,
                overlay,
                test,
            )
            family_alphas[family] = float(family_selected["alpha"])
        candidate = family_forecasts[str(selected["news_family"])]
        prediction_payload = {
            f"forecast__overlay_{family}": forecast
            for family, forecast in family_forecasts.items()
        }
        predictions.append(
            pd.DataFrame(
                {
                    "date": test["date"].to_numpy(),
                    "symbol": test["symbol"].to_numpy(),
                    "test_year": outer_year,
                    TARGET: test[TARGET].to_numpy(float),
                    "has_news": test["has_news"].to_numpy(int),
                    "forecast__xgb_price_rebuilt": base_test,
                    "forecast__xgb_price_news_overlay": candidate,
                    "news_family": selected["news_family"],
                    "alpha": selected["alpha"],
                    **prediction_payload,
                }
            )
        )
        print(
            f"{outer_year}: family={selected['news_family']}, "
            f"alpha={selected['alpha']}, "
            f"median inner gain={selected['median_gain']:+.4%}",
            flush=True,
        )

    oof = pd.concat(predictions, ignore_index=True)
    original = pd.read_parquet(BASE_OOF_PATH)
    original["date"] = pd.to_datetime(original["date"]).dt.normalize()
    original = original[original["horizon"].eq(HORIZON)][
        ["date", "symbol", BASE_FORECAST_COLUMN]
    ]
    oof = oof.merge(
        original,
        on=["date", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    maximum_difference = float(
        np.max(
            np.abs(
                oof["forecast__xgb_price_rebuilt"]
                - oof[BASE_FORECAST_COLUMN]
            )
        )
    )
    summary, symbols, regimes = summarise(oof)
    summary["max_rebuilt_price_forecast_difference"] = maximum_difference
    selection = pd.DataFrame(selection_rows)
    summary["selected_news_families"] = (
        selection.loc[selection["selected"], "news_family"]
        .value_counts()
        .to_dict()
    )
    summary["selected_alphas"] = (
        selection.loc[selection["selected"], "alpha"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    family_comparison = []
    for family in NEWS_FAMILIES:
        result = simple_candidate_summary(
            oof,
            f"forecast__overlay_{family}",
        )
        family_comparison.append(
            {
                "news_family": family,
                **result,
            }
        )

    oof.to_parquet(OUTPUT / "overlay_oof_predictions.parquet", index=False)
    selection.to_csv(OUTPUT / "overlay_inner_selection.csv", index=False)
    symbols.to_csv(OUTPUT / "overlay_per_symbol_results.csv", index=False)
    regimes.to_csv(OUTPUT / "overlay_regime_results.csv", index=False)
    pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key != "yearly_gains"
            }
            for row in family_comparison
        ]
    ).to_csv(OUTPUT / "overlay_family_comparison.csv", index=False)
    (OUTPUT / "overlay_family_comparison.json").write_text(
        json.dumps(family_comparison, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "overlay_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("\nOVERLAY SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nFAMILY COMPARISON")
    print(json.dumps(family_comparison, indent=2))
    print(f"\nOutputs -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
