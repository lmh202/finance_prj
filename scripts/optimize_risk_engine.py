"""Search the generalisation ceiling of the AURORA volatility Risk Engine.

Primary target: future five-session realised volatility.
Secondary target: future twenty-session realised volatility.

The experiment is nested and embargoed.  Feature and hyperparameter decisions
are made only inside each outer annual fold (2018-2023).  The official
``risk_model.json`` is never replaced unless every promotion gate passes.

Run the full planned search:
    python scripts/optimize_risk_engine.py

Useful smoke run:
    python scripts/optimize_risk_engine.py --trials 4 --top-k 2 \
        --outer-years 2022,2023 --bootstrap-reps 50 --output-suffix smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from risk_engine_optimization import (  # noqa: E402
    CAUSAL_NEWS_STATES,
    CURRENT_MODEL_PATH,
    CURRENT_OHLC_PATH,
    EPS,
    HAR_FEATURES,
    HORIZONS,
    OUTER_YEARS,
    PRICE_FEATURES,
    RANDOM_SEED,
    GammaOverlay,
    HarModel,
    build_current_panel,
    build_historical_panel,
    current_checkpoint_forecast,
    dm_test,
    fit_gamma_overlay,
    fit_har,
    fit_xgb_gamma,
    fit_xgb_gamma_fixed,
    inner_folds,
    model_metrics,
    moving_block_bootstrap_gain,
    optimal_sigma_scale,
    overlay_forecast,
    parameter_configs,
    qlike_loss,
    sha256_file,
    symbol_equal_qlike,
    symbol_equal_weights,
    time_fold,
    xgb_sigma_prediction,
)

PROCESSED = ROOT / "data" / "processed"
DEFAULT_REPORT = ROOT / "reports" / "risk_engine_optimization"
DEFAULT_CANDIDATE = PROCESSED / "risk_model_candidate"
PANEL_CACHE = PROCESSED / "risk_optimization_panel.parquet"
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
BLEND_WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
DIRECT_VARIANCE_SCALE = 10_000.0

PROMOTION_5D = {
    "minimum_gain": 0.03,
    "minimum_positive_years": 4,
    "minimum_worst_year_gain": -0.02,
    "minimum_positive_symbol_share": 0.60,
    "var95_min": 0.04,
    "var95_max": 0.06,
    "band_min": 0.93,
    "band_max": 0.97,
    "es_ratio_min": 0.80,
    "es_ratio_max": 1.20,
}


@dataclass
class LinearPriceModel:
    model_type: str
    features: list[str]
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    calibration: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = (
            frame[self.features].to_numpy(dtype=float) - self.mean
        ) / self.scale
        linear = self.intercept + values @ self.coefficient
        if self.model_type == "ridge_har":
            return np.exp(linear) * self.calibration
        scaled_variance = np.exp(np.clip(linear, -30.0, 30.0))
        return (
            np.sqrt(scaled_variance / DIRECT_VARIANCE_SCALE)
            * self.calibration
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "features": self.features,
            "alpha": self.alpha,
            "mean": {
                name: float(value)
                for name, value in zip(self.features, self.mean)
            },
            "scale": {
                name: float(value)
                for name, value in zip(self.features, self.scale)
            },
            "coef": {
                name: float(value)
                for name, value in zip(self.features, self.coefficient)
            },
            "intercept": self.intercept,
            "calibration": self.calibration,
            "direct_variance_scale": DIRECT_VARIANCE_SCALE,
        }


def _weighted_qlike(
    frame: pd.DataFrame,
    horizon: int,
    forecast: np.ndarray,
) -> float:
    return symbol_equal_qlike(
        frame,
        f"realized_vol_{horizon}d",
        forecast,
    )


def fit_linear_price_model(
    frame: pd.DataFrame,
    horizon: int,
    model_type: str,
    alpha: float,
) -> LinearPriceModel:
    if model_type == "ridge_har":
        features = HAR_FEATURES
    elif model_type == "linear_gamma":
        features = PRICE_FEATURES
    else:
        raise ValueError(f"unsupported linear price model: {model_type}")

    values = frame[features].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > EPS, scale, 1.0)
    standardized = (values - mean) / scale
    weights = symbol_equal_weights(frame)

    if model_type == "ridge_har":
        target = np.log(
            np.maximum(
                frame[f"realized_vol_{horizon}d"].to_numpy(), EPS
            )
        )
        estimator = Ridge(alpha=alpha)
        estimator.fit(standardized, target, sample_weight=weights)
        residual = target - estimator.predict(standardized)
        calibration = float(
            np.average(np.exp(residual), weights=weights)
        )
    else:
        target = np.maximum(
            frame[f"target_variance_{horizon}d"].to_numpy()
            * DIRECT_VARIANCE_SCALE,
            EPS,
        )
        estimator = GammaRegressor(
            alpha=alpha,
            max_iter=2_000,
            tol=1e-8,
        )
        estimator.fit(standardized, target, sample_weight=weights)
        raw_volatility = np.sqrt(
            np.maximum(estimator.predict(standardized), EPS)
            / DIRECT_VARIANCE_SCALE
        )
        calibration = optimal_sigma_scale(
            frame[f"realized_vol_{horizon}d"].to_numpy(),
            raw_volatility,
            weights,
        )
    return LinearPriceModel(
        model_type=model_type,
        features=list(features),
        alpha=float(alpha),
        mean=np.asarray(mean, dtype=float),
        scale=np.asarray(scale, dtype=float),
        coefficient=np.asarray(estimator.coef_, dtype=float),
        intercept=float(estimator.intercept_),
        calibration=float(calibration),
    )


def tune_linear_price_models(
    panel: pd.DataFrame,
    folds,
    horizon: int,
    outer_year: int,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    selected: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    for model_type in ("ridge_har", "linear_gamma"):
        candidates = []
        for alpha in ALPHAS:
            scores = []
            for fold in folds:
                train, validation = _split_frames(panel, fold)
                model = fit_linear_price_model(
                    train, horizon, model_type, alpha
                )
                scores.append(
                    _weighted_qlike(
                        validation,
                        horizon,
                        model.predict(validation),
                    )
                )
            mean_score = float(np.mean(scores))
            worst_score = float(np.max(scores))
            candidates.append((mean_score, worst_score, alpha))
            rows.append(
                {
                    "outer_year": outer_year,
                    "horizon": horizon,
                    "stage": "linear_full_inner",
                    "config_id": f"{model_type}:{alpha}",
                    "variant": model_type,
                    "score": mean_score,
                    "worst_score": worst_score,
                    "alpha": alpha,
                }
            )
        _, _, best_alpha = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        selected[model_type] = float(best_alpha)
    return selected, rows


def _split_frames(
    panel: pd.DataFrame,
    fold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        panel.loc[fold.train_mask].reset_index(drop=True),
        panel.loc[fold.test_mask].reset_index(drop=True),
    )


def _overlay_fold_scores(
    panel: pd.DataFrame,
    folds,
    horizon: int,
    features: list[str],
    alpha: float,
) -> tuple[list[float], list[float]]:
    gains, candidate_scores = [], []
    for fold in folds:
        train, validation = _split_frames(panel, fold)
        har = fit_har(train, horizon)
        train_base = har.predict(train)
        validation_base = har.predict(validation)
        overlay = fit_gamma_overlay(
            train,
            train_base,
            horizon,
            features,
            alpha,
        )
        candidate = overlay_forecast(validation_base, overlay, validation)
        base_score = _weighted_qlike(validation, horizon, validation_base)
        candidate_score = _weighted_qlike(validation, horizon, candidate)
        gains.append((base_score - candidate_score) / base_score)
        candidate_scores.append(candidate_score)
    return gains, candidate_scores


def _screen_overlay_fold_scores(
    panel: pd.DataFrame,
    folds,
    horizon: int,
    features: list[str],
    alpha: float,
) -> tuple[list[float], list[float]]:
    """Fast ridge screen; candidates are still scored on final QLIKE."""
    gains, candidate_scores = [], []
    limit = 2 * math.log(2.0)
    for fold in folds:
        train, validation = _split_frames(panel, fold)
        har = fit_har(train, horizon)
        train_base = har.predict(train)
        validation_base = har.predict(validation)
        scale = train[features].std(ddof=0).replace(0, 1.0).fillna(1.0)
        target = np.log(
            np.clip(
                (
                    train[f"realized_vol_{horizon}d"].to_numpy()
                    / np.maximum(train_base, EPS)
                )
                ** 2,
                1e-4,
                1e4,
            )
        )
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(
            train[features].div(scale),
            target,
            sample_weight=symbol_equal_weights(train),
        )
        log_ratio = np.clip(
            model.predict(validation[features].div(scale)),
            -limit,
            limit,
        )
        available = validation["news_quality_available"].eq(1).to_numpy()
        candidate = validation_base * np.where(
            available, np.exp(0.5 * log_ratio), 1.0
        )
        base_score = _weighted_qlike(validation, horizon, validation_base)
        candidate_score = _weighted_qlike(validation, horizon, candidate)
        gains.append((base_score - candidate_score) / base_score)
        candidate_scores.append(candidate_score)
    return gains, candidate_scores


def _best_overlay_configuration(
    panel: pd.DataFrame,
    folds,
    horizon: int,
    features: list[str],
    alpha_grid=ALPHAS,
) -> dict[str, object]:
    candidates = []
    for alpha in alpha_grid:
        gains, scores = _screen_overlay_fold_scores(
            panel, folds, horizon, features, alpha
        )
        candidates.append(
            {
                "alpha": alpha,
                "gains": gains,
                "scores": scores,
                "positive_share": float(np.mean(np.asarray(gains) > 0)),
                "median_gain": float(np.median(gains)),
                "mean_gain": float(np.mean(gains)),
                "worst_gain": float(np.min(gains)),
            }
        )
    return max(
        candidates,
        key=lambda row: (
            row["positive_share"] >= 0.60
            and row["median_gain"] > 0
            and row["worst_gain"] > -0.01,
            row["median_gain"],
            row["mean_gain"],
        ),
    )


def select_news_features(
    panel: pd.DataFrame,
    folds,
    horizon: int,
    feature_groups: Mapping[str, list[str]],
    allowed_features: list[str],
    scope: str,
    outer_year: int,
) -> tuple[list[str], float, list[dict[str, object]]]:
    """Nested greedy family selection followed by drop-column stability."""
    allowed = set(allowed_features)
    groups = {
        name: [feature for feature in values if feature in allowed]
        for name, values in feature_groups.items()
    }
    groups = {name: values for name, values in groups.items() if values}
    selected: list[str] = []
    remaining = dict(groups)
    contribution_rows: list[dict[str, object]] = []

    while remaining:
        evaluated = []
        for group_name, group_features in remaining.items():
            candidate_features = list(
                dict.fromkeys(selected + group_features)
            )
            result = _best_overlay_configuration(
                panel,
                folds,
                horizon,
                candidate_features,
            )
            evaluated.append((group_name, group_features, result))
            contribution_rows.append(
                {
                    "outer_year": outer_year,
                    "horizon": horizon,
                    "scope": scope,
                    "stage": "family_forward",
                    "feature": group_name,
                    "alpha": result["alpha"],
                    "median_gain": result["median_gain"],
                    "mean_gain": result["mean_gain"],
                    "worst_gain": result["worst_gain"],
                    "positive_fold_share": result["positive_share"],
                    "accepted": False,
                }
            )
        best_name, best_features, best_result = max(
            evaluated,
            key=lambda item: (
                item[2]["positive_share"] >= 0.60
                and item[2]["median_gain"] > 0
                and item[2]["worst_gain"] > -0.01,
                item[2]["median_gain"],
                item[2]["mean_gain"],
            ),
        )
        passes = (
            best_result["positive_share"] >= 0.60
            and best_result["median_gain"] > 0
            and best_result["worst_gain"] > -0.01
        )
        row = contribution_rows[-len(remaining) :][
            list(remaining).index(best_name)
        ]
        row["accepted"] = passes
        if not passes:
            break
        selected = list(dict.fromkeys(selected + best_features))
        remaining.pop(best_name)

    if not selected:
        # A neutral minimal contract keeps the research runnable while the
        # acceptance result remains an honest rejection.
        selected = [
            feature
            for feature in ("has_news", "log_count") + tuple(CAUSAL_NEWS_STATES)
            if feature in allowed
        ]

    full_result = _best_overlay_configuration(
        panel, folds, horizon, selected
    )
    alpha = float(full_result["alpha"])
    _, full_scores = _screen_overlay_fold_scores(
        panel, folds, horizon, selected, alpha
    )
    stable = []
    for feature in selected:
        reduced = [value for value in selected if value != feature]
        if not reduced:
            stable.append(feature)
            continue
        _, reduced_scores = _screen_overlay_fold_scores(
            panel, folds, horizon, reduced, alpha
        )
        contributions = [
            (reduced_score - full_score) / reduced_score
            for reduced_score, full_score in zip(reduced_scores, full_scores)
        ]
        positive_share = float(np.mean(np.asarray(contributions) > 0))
        median = float(np.median(contributions))
        accepted = positive_share >= 0.60 and median > 0
        if accepted:
            stable.append(feature)
        contribution_rows.append(
            {
                "outer_year": outer_year,
                "horizon": horizon,
                "scope": scope,
                "stage": "drop_column",
                "feature": feature,
                "alpha": alpha,
                "median_gain": median,
                "mean_gain": float(np.mean(contributions)),
                "worst_gain": float(np.min(contributions)),
                "positive_fold_share": positive_share,
                "accepted": accepted,
            }
        )

    if not stable:
        stable = selected
    # Feature discovery uses fast ridge fits.  Once the stable set is fixed,
    # choose the production Gamma regularisation on the same inner folds.
    gamma_candidates = []
    for gamma_alpha in ALPHAS:
        gamma_gains, gamma_scores = _overlay_fold_scores(
            panel, folds, horizon, stable, gamma_alpha
        )
        gamma_candidates.append(
            {
                "alpha": gamma_alpha,
                "median_gain": float(np.median(gamma_gains)),
                "mean_gain": float(np.mean(gamma_gains)),
                "positive_share": float(
                    np.mean(np.asarray(gamma_gains) > 0)
                ),
                "mean_score": float(np.mean(gamma_scores)),
            }
        )
    stable_result = max(
        gamma_candidates,
        key=lambda row: (
            row["positive_share"] >= 0.60 and row["median_gain"] > 0,
            row["median_gain"],
            row["mean_gain"],
            -row["mean_score"],
        ),
    )
    return stable, float(stable_result["alpha"]), contribution_rows


def _xgb_target(
    frame: pd.DataFrame,
    horizon: int,
    variant: str,
    har_forecast: np.ndarray | None = None,
) -> np.ndarray:
    variance = frame[f"target_variance_{horizon}d"].to_numpy()
    if variant == "direct":
        # Daily variances are usually around 1e-4.  XGBoost's Gamma
        # objective can collapse to an all-zero predictor at that numerical
        # scale (and then every trial appears to stop at tree zero).  Express
        # direct variance in percent-squared units during fitting and invert
        # the deterministic scale at prediction time.
        return np.clip(variance * DIRECT_VARIANCE_SCALE, EPS, 1e4)
    if har_forecast is None:
        raise ValueError("HAR forecast is required for ratio target")
    return np.clip(
        variance / np.maximum(np.asarray(har_forecast) ** 2, EPS),
        1e-4,
        1e4,
    )


def _xgb_forecast(
    model,
    frame: pd.DataFrame,
    features: list[str],
    variant: str,
    har_forecast: np.ndarray,
) -> np.ndarray:
    if variant == "direct":
        scaled_variance = np.clip(
            model.predict(frame[features]), EPS, 1e4
        )
        return np.sqrt(scaled_variance / DIRECT_VARIANCE_SCALE)
    ratio = xgb_sigma_prediction(model, frame, features)
    return np.asarray(har_forecast) * np.clip(ratio, 0.5, 2.0)


def tune_price_xgb(
    panel: pd.DataFrame,
    folds,
    horizon: int,
    configs: list[dict[str, float | int]],
    top_k: int,
    device: str,
    outer_year: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """96-config coarse screen, top-k full inner review, then variant/blend."""
    search_rows: list[dict[str, object]] = []
    coarse_fold = folds[-1]
    coarse_train, coarse_validation = _split_frames(panel, coarse_fold)
    coarse_scores = []
    for config in configs:
        model = fit_xgb_gamma(
            coarse_train,
            coarse_validation,
            PRICE_FEATURES,
            _xgb_target(coarse_train, horizon, "direct"),
            _xgb_target(coarse_validation, horizon, "direct"),
            config,
            device,
        )
        forecast = _xgb_forecast(
            model,
            coarse_validation,
            PRICE_FEATURES,
            "direct",
            np.ones(len(coarse_validation)),
        )
        score = _weighted_qlike(coarse_validation, horizon, forecast)
        iterations = int(
            getattr(model, "best_iteration", config.get("n_estimators", 300))
            + 1
        )
        coarse_scores.append((score, config, iterations))
        search_rows.append(
            {
                "outer_year": outer_year,
                "horizon": horizon,
                "stage": "coarse",
                "config_id": config["config_id"],
                "variant": "direct",
                "score": score,
                "iterations": iterations,
                **{
                    key: value
                    for key, value in config.items()
                    if key != "config_id"
                },
            }
        )

    finalists = sorted(coarse_scores, key=lambda item: item[0])[:top_k]
    reviewed = []
    for _, config, _ in finalists:
        scores, iterations = [], []
        for fold in folds:
            train, validation = _split_frames(panel, fold)
            model = fit_xgb_gamma(
                train,
                validation,
                PRICE_FEATURES,
                _xgb_target(train, horizon, "direct"),
                _xgb_target(validation, horizon, "direct"),
                config,
                device,
            )
            forecast = _xgb_forecast(
                model,
                validation,
                PRICE_FEATURES,
                "direct",
                np.ones(len(validation)),
            )
            scores.append(_weighted_qlike(validation, horizon, forecast))
            iterations.append(int(getattr(model, "best_iteration", 299) + 1))
        reviewed.append(
            {
                "config": config,
                "mean_score": float(np.mean(scores)),
                "worst_score": float(np.max(scores)),
                "iterations": int(np.median(iterations)),
            }
        )
        search_rows.append(
            {
                "outer_year": outer_year,
                "horizon": horizon,
                "stage": "full_inner",
                "config_id": config["config_id"],
                "variant": "direct",
                "score": float(np.mean(scores)),
                "worst_score": float(np.max(scores)),
                "iterations": int(np.median(iterations)),
                **{
                    key: value
                    for key, value in config.items()
                    if key != "config_id"
                },
            }
        )
    best = min(reviewed, key=lambda row: (row["mean_score"], row["worst_score"]))
    config = best["config"]

    inner_predictions = []
    iteration_values = []
    for fold in folds:
        train, validation = _split_frames(panel, fold)
        har = fit_har(train, horizon)
        har_train = har.predict(train)
        har_validation = har.predict(validation)
        predictions = {"har": har_validation}
        for variant in ("direct", "ratio"):
            model = fit_xgb_gamma(
                train,
                validation,
                PRICE_FEATURES,
                _xgb_target(train, horizon, variant, har_train),
                _xgb_target(
                    validation, horizon, variant, har_validation
                ),
                config,
                device,
            )
            predictions[variant] = _xgb_forecast(
                model,
                validation,
                PRICE_FEATURES,
                variant,
                har_validation,
            )
            iteration_values.append(
                int(getattr(model, "best_iteration", 299) + 1)
            )
        inner_predictions.append((validation, predictions))

    variants = []
    for variant in ("direct", "ratio"):
        for weight in BLEND_WEIGHTS:
            frames, forecasts = [], []
            for validation, prediction in inner_predictions:
                blended = (
                    (1 - weight) * prediction["har"]
                    + weight * prediction[variant]
                )
                frames.append(validation)
                forecasts.append(blended)
            combined = pd.concat(frames, ignore_index=True)
            combined_forecast = np.concatenate(forecasts)
            scale = optimal_sigma_scale(
                combined[f"realized_vol_{horizon}d"].to_numpy(),
                combined_forecast,
                symbol_equal_weights(combined),
            )
            score = _weighted_qlike(
                combined, horizon, combined_forecast * scale
            )
            variants.append(
                {
                    "variant": variant,
                    "blend_weight": float(weight),
                    "scale": scale,
                    "score": score,
                }
            )
    chosen_variant = min(variants, key=lambda row: row["score"])
    result = {
        "config": config,
        "n_estimators": int(np.median(iteration_values)),
        **chosen_variant,
    }
    search_rows.append(
        {
            "outer_year": outer_year,
            "horizon": horizon,
            "stage": "variant_selection",
            "config_id": config["config_id"],
            "variant": result["variant"],
            "blend_weight": result["blend_weight"],
            "scale": result["scale"],
            "score": result["score"],
            "iterations": result["n_estimators"],
        }
    )
    return result, search_rows


def fit_price_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    specification: Mapping[str, object],
    device: str,
):
    har = fit_har(train, horizon)
    har_train, har_test = har.predict(train), har.predict(test)
    variant = str(specification["variant"])
    target = _xgb_target(train, horizon, variant, har_train)
    model = fit_xgb_gamma_fixed(
        train,
        PRICE_FEATURES,
        target,
        specification["config"],
        device,
        int(specification["n_estimators"]),
    )
    raw = _xgb_forecast(
        model, test, PRICE_FEATURES, variant, har_test
    )
    blended = (
        (1 - float(specification["blend_weight"])) * har_test
        + float(specification["blend_weight"]) * raw
    ) * float(specification["scale"])
    return har, model, har_train, har_test, blended


def fit_news_xgb(
    train: pd.DataFrame,
    base_train: np.ndarray,
    features: list[str],
    horizon: int,
    specification: Mapping[str, object],
    device: str,
):
    context = list(dict.fromkeys(features + ["l_rv22", "l_absret"]))
    target = _xgb_target(train, horizon, "ratio", base_train)
    model = fit_xgb_gamma_fixed(
        train,
        context,
        target,
        specification["config"],
        device,
        int(specification["n_estimators"]),
    )
    return model, context


def news_xgb_forecast(
    model,
    context: list[str],
    frame: pd.DataFrame,
    base_forecast: np.ndarray,
) -> np.ndarray:
    multiplier = np.clip(
        xgb_sigma_prediction(model, frame, context),
        0.5,
        2.0,
    )
    available = frame["news_quality_available"].eq(1).to_numpy()
    return np.asarray(base_forecast) * np.where(
        available, multiplier, 1.0
    )


def _fold_metric_rows(
    test: pd.DataFrame,
    horizon: int,
    test_year: int,
    forecasts: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for model_name, forecast in forecasts.items():
        metrics = model_metrics(test, horizon, forecast)
        rows.append(
            {
                "horizon": horizon,
                "test_year": test_year,
                "model": model_name,
                **metrics,
            }
        )
    return rows


def run_outer_experiment(
    panel: pd.DataFrame,
    groups: dict[str, list[str]],
    research_features: list[str],
    deployable_features: list[str],
    outer_years: list[int],
    configs: list[dict[str, float | int]],
    top_k: int,
    device: str,
):
    fold_rows, contribution_rows, search_rows, oof_parts = [], [], [], []
    specifications: dict[int, list[dict[str, object]]] = {5: [], 20: []}

    for horizon in HORIZONS:
        horizon_panel = panel.dropna(
            subset=[f"realized_vol_{horizon}d"]
        ).reset_index(drop=True)
        for outer_year in outer_years:
            outer = time_fold(horizon_panel, outer_year, horizon)
            folds = inner_folds(horizon_panel, outer_year, horizon)
            if len(folds) < 2:
                raise ValueError(
                    f"outer year {outer_year} has fewer than two inner folds"
                )
            print(
                f"\n[h={horizon}] outer={outer_year} | "
                f"inner={len(folds)} | train<={outer.train_cutoff.date()}"
            )

            research_selected, research_alpha, rows = select_news_features(
                horizon_panel,
                folds,
                horizon,
                groups,
                research_features,
                "research",
                outer_year,
            )
            contribution_rows.extend(rows)
            deploy_selected, deploy_alpha, rows = select_news_features(
                horizon_panel,
                folds,
                horizon,
                groups,
                deployable_features,
                "deployable",
                outer_year,
            )
            contribution_rows.extend(rows)
            print(
                f"  selected news: research={len(research_selected)} "
                f"deployable={len(deploy_selected)}"
            )

            linear_specs, rows = tune_linear_price_models(
                horizon_panel, folds, horizon, outer_year
            )
            search_rows.extend(rows)
            price_spec, rows = tune_price_xgb(
                horizon_panel,
                folds,
                horizon,
                configs,
                top_k,
                device,
                outer_year,
            )
            search_rows.extend(rows)
            specifications[horizon].append(price_spec)
            print(
                f"  linear: ridge_alpha={linear_specs['ridge_har']} "
                f"gamma_alpha={linear_specs['linear_gamma']}"
            )
            print(
                f"  price XGB: cfg={price_spec['config']['config_id']} "
                f"{price_spec['variant']} blend={price_spec['blend_weight']:.2f} "
                f"inner QLIKE={price_spec['score']:.4f}"
            )

            train, test = _split_frames(horizon_panel, outer)
            har, price_model, har_train, har_test, price_forecast = (
                fit_price_candidate(
                    train, test, horizon, price_spec, device
                )
            )
            forecasts: dict[str, np.ndarray] = {
                "naive_rv22": test["rv22"].to_numpy(),
                "ewma": test["ewma_sigma"].to_numpy(),
                "har": har_test,
                "ridge_har": fit_linear_price_model(
                    train,
                    horizon,
                    "ridge_har",
                    linear_specs["ridge_har"],
                ).predict(test),
                "linear_gamma": fit_linear_price_model(
                    train,
                    horizon,
                    "linear_gamma",
                    linear_specs["linear_gamma"],
                ).predict(test),
                "xgb_price": price_forecast,
                "current_frozen_har": current_checkpoint_forecast(
                    test, horizon
                ),
            }

            for scope, features, alpha in (
                ("research", research_selected, research_alpha),
                ("deployable", deploy_selected, deploy_alpha),
            ):
                linear = fit_gamma_overlay(
                    train, har_train, horizon, features, alpha
                )
                forecasts[f"har_news_linear_{scope}"] = overlay_forecast(
                    har_test, linear, test
                )
                news_model, context = fit_news_xgb(
                    train,
                    har_train,
                    features,
                    horizon,
                    price_spec,
                    device,
                )
                forecasts[f"har_news_xgb_{scope}"] = news_xgb_forecast(
                    news_model, context, test, har_test
                )

            fold_rows.extend(
                _fold_metric_rows(test, horizon, outer_year, forecasts)
            )
            oof = test[
                [
                    "date",
                    "symbol",
                    f"realized_vol_{horizon}d",
                    f"fwd_return_{horizon}d",
                    "rv22",
                ]
            ].copy()
            oof["horizon"] = horizon
            oof["test_year"] = outer_year
            for name, values in forecasts.items():
                oof[f"forecast__{name}"] = values
            oof_parts.append(oof)

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(contribution_rows),
        pd.DataFrame(search_rows),
        pd.concat(oof_parts, ignore_index=True),
        specifications,
    )


def build_leaderboard(folds: pd.DataFrame) -> pd.DataFrame:
    reference = (
        folds[folds["model"] == "har"]
        .set_index(["horizon", "test_year"])["qlike"]
    )
    rows = []
    for (horizon, model), group in folds.groupby(["horizon", "model"]):
        reference_scores = np.array(
            [
                reference.loc[(horizon, year)]
                for year in group["test_year"]
            ]
        )
        gains = (reference_scores - group["qlike"].to_numpy()) / reference_scores
        rows.append(
            {
                "horizon": horizon,
                "model": model,
                "mean_fold_qlike": group["qlike"].mean(),
                "mean_relative_gain_vs_har": gains.mean(),
                "median_relative_gain_vs_har": np.median(gains),
                "worst_year_gain_vs_har": gains.min(),
                "positive_years_vs_har": int((gains > 0).sum()),
                "folds": len(group),
                "mean_rmse_log": group["rmse_log"].mean(),
                "mean_r2_log": group["r2_log"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["horizon", "mean_fold_qlike"]
    )


def _candidate_name(leaderboard: pd.DataFrame, horizon: int) -> str:
    eligible = leaderboard[
        (leaderboard["horizon"] == horizon)
        & leaderboard["model"].isin(
            [
                "har",
                "ridge_har",
                "linear_gamma",
                "xgb_price",
                "har_news_linear_deployable",
                "har_news_xgb_deployable",
            ]
        )
    ]
    return str(eligible.sort_values("mean_fold_qlike").iloc[0]["model"])


def per_symbol_results(
    oof: pd.DataFrame,
    horizon: int,
    candidate: str,
    reference: str = "current_frozen_har",
) -> pd.DataFrame:
    subset = oof[oof["horizon"] == horizon].copy()
    realized = subset[f"realized_vol_{horizon}d"].to_numpy()
    subset["reference_loss"] = qlike_loss(
        realized, subset[f"forecast__{reference}"].to_numpy()
    )
    subset["candidate_loss"] = qlike_loss(
        realized, subset[f"forecast__{candidate}"].to_numpy()
    )
    result = (
        subset.groupby("symbol")
        .agg(
            n=("symbol", "size"),
            reference_qlike=("reference_loss", "mean"),
            candidate_qlike=("candidate_loss", "mean"),
        )
        .reset_index()
    )
    result["relative_gain"] = (
        result["reference_qlike"] - result["candidate_qlike"]
    ) / result["reference_qlike"]
    result.insert(0, "horizon", horizon)
    result.insert(1, "candidate", candidate)
    result.insert(2, "reference", reference)
    return result


def evaluate_gates(
    oof: pd.DataFrame,
    folds: pd.DataFrame,
    leaderboard: pd.DataFrame,
    bootstrap_reps: int,
    candidates: Mapping[int, str] | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    rss_shadow_mature: bool = False,
) -> tuple[dict[str, object], pd.DataFrame]:
    checkpoint = json.loads(CURRENT_MODEL_PATH.read_text(encoding="utf-8"))
    gates: dict[str, object] = {}
    symbol_parts = []
    reference = "current_frozen_har"
    candidates = candidates or {}
    for horizon in horizons:
        candidate = candidates.get(
            horizon, _candidate_name(leaderboard, horizon)
        )
        subset = oof[oof["horizon"] == horizon].reset_index(drop=True)
        realized = subset[f"realized_vol_{horizon}d"].to_numpy()
        reference_forecast = subset[f"forecast__{reference}"].to_numpy()
        candidate_forecast = subset[f"forecast__{candidate}"].to_numpy()
        reference_loss = qlike_loss(realized, reference_forecast)
        candidate_loss = qlike_loss(realized, candidate_forecast)
        weights = symbol_equal_weights(subset)
        scored = pd.DataFrame(
            {
                "date": subset["date"],
                "reference_loss": reference_loss * weights,
                "candidate_loss": candidate_loss * weights,
            }
        )
        ci_low, ci_median, ci_high = moving_block_bootstrap_gain(
            scored,
            bootstrap_reps,
            horizon,
            RANDOM_SEED + horizon,
        )
        dm_statistic, dm_p = dm_test(
            subset["date"],
            candidate_loss * weights,
            reference_loss * weights,
            horizon,
        )
        symbol_result = per_symbol_results(
            oof, horizon, candidate, reference
        )
        symbol_parts.append(symbol_result)
        positive_symbol_share = float(symbol_result["relative_gain"].gt(0).mean())
        median_symbol_gain = float(symbol_result["relative_gain"].median())

        candidate_folds = folds[
            (folds["horizon"] == horizon)
            & (folds["model"] == candidate)
        ].sort_values("test_year")
        reference_folds = folds[
            (folds["horizon"] == horizon)
            & (folds["model"] == reference)
        ].sort_values("test_year")
        yearly_gains = (
            (
                reference_folds["qlike"].to_numpy()
                - candidate_folds["qlike"].to_numpy()
            )
            / reference_folds["qlike"].to_numpy()
        )
        aggregate_gain = float(
            (np.average(reference_loss, weights=weights)
             - np.average(candidate_loss, weights=weights))
            / np.average(reference_loss, weights=weights)
        )

        high_threshold = np.quantile(realized, 2 / 3)
        high = realized >= high_threshold
        high_reference = reference_loss[high].mean()
        high_candidate = candidate_loss[high].mean()
        high_gain = float(
            (high_reference - high_candidate) / high_reference
        )

        fhs = checkpoint["fhs"]["horizons"][str(horizon)]
        sigma_h = candidate_forecast * np.sqrt(horizon)
        forward_return = subset[f"fwd_return_{horizon}d"].to_numpy()
        var95 = sigma_h * fhs["q05"]
        es95 = sigma_h * fhs["es05"]
        band_low = sigma_h * fhs["q025"]
        band_high = sigma_h * fhs["q975"]
        breaches = forward_return < var95
        var95_rate = float(breaches.mean())
        band_coverage = float(
            ((forward_return >= band_low) & (forward_return <= band_high)).mean()
        )
        if breaches.any():
            es_ratio = float(
                abs(forward_return[breaches].mean())
                / max(abs(es95[breaches].mean()), EPS)
            )
        else:
            es_ratio = float("nan")

        news_candidate = "_news_" in candidate
        if horizon == 5:
            checks = {
                "gain_at_least_3pct": aggregate_gain >= 0.03,
                "bootstrap_lower_positive": ci_low > 0,
                "dm_significant": dm_statistic < 0 and dm_p < 0.05,
                "positive_years": int((yearly_gains > 0).sum()) >= 4,
                "worst_year": float(yearly_gains.min()) >= -0.02,
                "positive_symbols": positive_symbol_share >= 0.60,
                "median_symbol_positive": median_symbol_gain > 0,
                "high_vol_not_worse": high_gain >= 0,
                "var95_calibrated": 0.04 <= var95_rate <= 0.06,
                "band_calibrated": 0.93 <= band_coverage <= 0.97,
                "es_calibrated": 0.80 <= es_ratio <= 1.20,
                "rss_shadow_60_mature": (
                    not news_candidate or rss_shadow_mature
                ),
            }
        else:
            checks = {
                "aggregate_not_worse_1pct": aggregate_gain >= -0.01,
                "significantly_better_if_promoted": (
                    aggregate_gain > 0 and dm_statistic < 0 and dm_p < 0.05
                ),
                "tail_not_worse": high_gain >= -0.01,
                "band_not_worse": 0.92 <= band_coverage <= 0.98,
                "rss_shadow_60_mature": (
                    not news_candidate or rss_shadow_mature
                ),
            }
        gates[str(horizon)] = {
            "candidate": candidate,
            "reference": reference,
            "aggregate_relative_gain": aggregate_gain,
            "yearly_gains": {
                str(year): float(gain)
                for year, gain in zip(candidate_folds["test_year"], yearly_gains)
            },
            "positive_symbol_share": positive_symbol_share,
            "median_symbol_gain": median_symbol_gain,
            "high_vol_relative_gain": high_gain,
            "bootstrap_ci": [ci_low, ci_median, ci_high],
            "dm_t": dm_statistic,
            "dm_p": dm_p,
            "var95_breach_rate": var95_rate,
            "band_coverage": band_coverage,
            "es_ratio": es_ratio,
            "checks": checks,
            "passed": bool(all(checks.values())),
        }
    return gates, pd.concat(symbol_parts, ignore_index=True)


def _stable_final_features(
    contributions: pd.DataFrame,
    horizon: int,
    scope: str,
) -> list[str]:
    selected = contributions[
        (contributions["horizon"] == horizon)
        & (contributions["scope"] == scope)
        & (contributions["stage"] == "drop_column")
        & (contributions["accepted"])
    ]
    counts = selected.groupby("feature")["outer_year"].nunique()
    return sorted(counts[counts.ge(4)].index.tolist())


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fhs_calibration(
    frame: pd.DataFrame,
    forecast: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    standardized = (
        frame[f"fwd_return_{horizon}d"].to_numpy()
        / np.maximum(np.asarray(forecast) * np.sqrt(horizon), EPS)
    )
    standardized = standardized[np.isfinite(standardized)]
    q05 = float(np.quantile(standardized, 0.05))
    q01 = float(np.quantile(standardized, 0.01))
    return {
        "q05": q05,
        "q01": q01,
        "q025": float(np.quantile(standardized, 0.025)),
        "q975": float(np.quantile(standardized, 0.975)),
        "es05": float(standardized[standardized <= q05].mean()),
        "es01": float(standardized[standardized <= q01].mean()),
    }


def train_final_bundle(
    panel: pd.DataFrame,
    groups: dict[str, list[str]],
    research_features: list[str],
    deployable_features: list[str],
    contributions: pd.DataFrame,
    leaderboard: pd.DataFrame,
    configs: list[dict[str, float | int]],
    top_k: int,
    device: str,
    candidate_dir: Path,
):
    candidate_dir.mkdir(parents=True, exist_ok=True)
    horizon_payload = {}
    final_models = {}
    final_forecasts = {}
    final_specs = {}

    for horizon in HORIZONS:
        horizon_panel = panel.dropna(
            subset=[f"realized_vol_{horizon}d"]
        ).reset_index(drop=True)
        folds = inner_folds(horizon_panel, 2024, horizon)
        linear_specs, _ = tune_linear_price_models(
            horizon_panel, folds, horizon, 2024
        )
        price_spec, _ = tune_price_xgb(
            horizon_panel,
            folds,
            horizon,
            configs,
            top_k,
            device,
            2024,
        )
        har = fit_har(horizon_panel, horizon)
        har_forecast = har.predict(horizon_panel)
        ridge_model = fit_linear_price_model(
            horizon_panel,
            horizon,
            "ridge_har",
            linear_specs["ridge_har"],
        )
        gamma_model = fit_linear_price_model(
            horizon_panel,
            horizon,
            "linear_gamma",
            linear_specs["linear_gamma"],
        )
        ridge_forecast = ridge_model.predict(horizon_panel)
        gamma_forecast = gamma_model.predict(horizon_panel)
        variant = str(price_spec["variant"])
        price_model = fit_xgb_gamma_fixed(
            horizon_panel,
            PRICE_FEATURES,
            _xgb_target(
                horizon_panel, horizon, variant, har_forecast
            ),
            price_spec["config"],
            device,
            int(price_spec["n_estimators"]),
        )
        price_model.get_booster().set_attr(
            risk_target_kind=variant,
            direct_variance_scale=str(DIRECT_VARIANCE_SCALE),
        )
        price_path = candidate_dir / f"price_{horizon}d_xgb.json"
        price_model.save_model(price_path)
        price_raw = _xgb_forecast(
            price_model,
            horizon_panel,
            PRICE_FEATURES,
            variant,
            har_forecast,
        )
        price_forecast = (
            (1 - float(price_spec["blend_weight"])) * har_forecast
            + float(price_spec["blend_weight"]) * price_raw
        ) * float(price_spec["scale"])

        research_selected = _stable_final_features(
            contributions, horizon, "research"
        )
        deploy_selected = _stable_final_features(
            contributions, horizon, "deployable"
        )
        if not research_selected:
            research_selected, _, _ = select_news_features(
                horizon_panel,
                folds,
                horizon,
                groups,
                research_features,
                "research_final",
                2024,
            )
        if not deploy_selected:
            deploy_selected, _, _ = select_news_features(
                horizon_panel,
                folds,
                horizon,
                groups,
                deployable_features,
                "deployable_final",
                2024,
            )
        deploy_result = _best_overlay_configuration(
            horizon_panel, folds, horizon, deploy_selected
        )
        linear_overlay = fit_gamma_overlay(
            horizon_panel,
            har_forecast,
            horizon,
            deploy_selected,
            float(deploy_result["alpha"]),
        )
        linear_forecast = overlay_forecast(
            har_forecast, linear_overlay, horizon_panel
        )
        news_xgb, news_context = fit_news_xgb(
            horizon_panel,
            har_forecast,
            deploy_selected,
            horizon,
            price_spec,
            device,
        )
        news_path = candidate_dir / f"news_{horizon}d_xgb.json"
        news_xgb.save_model(news_path)
        news_xgb_prediction = news_xgb_forecast(
            news_xgb,
            news_context,
            horizon_panel,
            har_forecast,
        )

        selected_model = _candidate_name(leaderboard, horizon)
        selected_price_model = str(
            leaderboard[
                (leaderboard["horizon"] == horizon)
                & leaderboard["model"].isin(
                    ["ridge_har", "linear_gamma", "xgb_price"]
                )
            ]
            .sort_values("mean_fold_qlike")
            .iloc[0]["model"]
        )
        forecast_lookup = {
            "har": har_forecast,
            "ridge_har": ridge_forecast,
            "linear_gamma": gamma_forecast,
            "xgb_price": price_forecast,
            "har_news_linear_deployable": linear_forecast,
            "har_news_xgb_deployable": news_xgb_prediction,
        }
        selected_forecast = forecast_lookup[selected_model]
        fhs = fhs_calibration(
            horizon_panel, selected_forecast, horizon
        )
        fhs_by_component = {
            "current_frozen_har": fhs_calibration(
                horizon_panel,
                current_checkpoint_forecast(horizon_panel, horizon),
                horizon,
            ),
            selected_price_model: fhs_calibration(
                horizon_panel,
                forecast_lookup[selected_price_model],
                horizon,
            ),
            "har_news_linear_deployable": fhs_calibration(
                horizon_panel,
                linear_forecast,
                horizon,
            ),
        }
        horizon_payload[str(horizon)] = {
            "selected_model": selected_model,
            "selected_price_model": selected_price_model,
            "har": har.to_dict(),
            "ridge_har": ridge_model.to_dict(),
            "linear_gamma": gamma_model.to_dict(),
            "price_xgb": {
                "model_type": "xgboost_gamma",
                "artifact": price_path.name,
                "features": PRICE_FEATURES,
                "variant": variant,
                "direct_variance_scale": DIRECT_VARIANCE_SCALE,
                "blend_weight": float(price_spec["blend_weight"]),
                "scale": float(price_spec["scale"]),
                "parameters": price_spec["config"],
                "n_estimators": int(price_spec["n_estimators"]),
            },
            "news_linear_deployable": linear_overlay.to_dict(),
            "news_xgb_deployable": {
                "model_type": "xgboost_gamma_variance_ratio",
                "artifact": news_path.name,
                "features": news_context,
                "news_features": deploy_selected,
                "max_sigma_multiplier": 2.0,
            },
            "research_features": research_selected,
            "deployable_features": deploy_selected,
            "fhs": fhs,
            "fhs_by_component": fhs_by_component,
        }
        final_models[horizon] = {
            "har": har,
            "ridge_har": ridge_model,
            "linear_gamma": gamma_model,
            "price": price_model,
            "news_xgb": news_xgb,
            "linear_overlay": linear_overlay,
        }
        final_forecasts[horizon] = forecast_lookup
        final_specs[horizon] = price_spec
    return horizon_payload, final_models, final_specs


def evaluate_external_price(
    final_models,
    final_specs,
    horizon_payload,
    device: str,
) -> pd.DataFrame:
    try:
        panel = build_current_panel()
    except Exception as exc:
        return pd.DataFrame(
            [{"scope": "unavailable", "error": str(exc)}]
        )
    rows = []
    research_symbols = set(
        pd.read_parquet(
            ROOT / "data" / "processed" / "training_dataset.parquet",
            columns=["symbol"],
        )["symbol"].unique()
    )
    for horizon in HORIZONS:
        sample = panel.dropna(
            subset=[f"realized_vol_{horizon}d"]
        ).reset_index(drop=True)
        har = final_models[horizon]["har"]
        har_forecast = har.predict(sample)
        reference_forecast = current_checkpoint_forecast(sample, horizon)
        spec = final_specs[horizon]
        selected_price = horizon_payload[str(horizon)][
            "selected_price_model"
        ]
        if selected_price == "xgb_price":
            raw = _xgb_forecast(
                final_models[horizon]["price"],
                sample,
                PRICE_FEATURES,
                str(spec["variant"]),
                har_forecast,
            )
            candidate = (
                (1 - float(spec["blend_weight"])) * har_forecast
                + float(spec["blend_weight"]) * raw
            ) * float(spec["scale"])
        else:
            candidate = final_models[horizon][selected_price].predict(
                sample
            )
        for scope, mask in {
            "all": np.ones(len(sample), dtype=bool),
            "original_research": sample["symbol"].isin(research_symbols).to_numpy(),
            "external_generalization": (~sample["symbol"].isin(research_symbols)).to_numpy(),
        }.items():
            if not mask.any():
                continue
            subset = sample.loc[mask].reset_index(drop=True)
            reference_score = _weighted_qlike(
                subset, horizon, reference_forecast[mask]
            )
            candidate_score = _weighted_qlike(
                subset, horizon, candidate[mask]
            )
            rows.append(
                {
                    "horizon": horizon,
                    "scope": scope,
                    "n": len(subset),
                    "symbols": int(subset["symbol"].nunique()),
                    "reference_model": "current_frozen_har",
                    "candidate_model": selected_price,
                    "reference_qlike": reference_score,
                    "candidate_qlike": candidate_score,
                    "relative_gain": (
                        reference_score - candidate_score
                    ) / reference_score,
                }
            )
    return pd.DataFrame(rows)


def build_regime_results(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        sample = oof[oof["horizon"] == horizon].copy()
        target = f"realized_vol_{horizon}d"
        sample["volatility_regime"] = pd.qcut(
            sample[target],
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        )
        forecast_columns = [
            column
            for column in sample.columns
            if column.startswith("forecast__")
        ]
        for regime, regime_frame in sample.groupby(
            "volatility_regime", observed=True
        ):
            reference_score = _weighted_qlike(
                regime_frame,
                horizon,
                regime_frame["forecast__current_frozen_har"].to_numpy(),
            )
            for column in forecast_columns:
                score = _weighted_qlike(
                    regime_frame,
                    horizon,
                    regime_frame[column].to_numpy(),
                )
                rows.append(
                    {
                        "horizon": horizon,
                        "volatility_regime": str(regime),
                        "model": column.removeprefix("forecast__"),
                        "n": len(regime_frame),
                        "symbols": int(regime_frame["symbol"].nunique()),
                        "qlike": score,
                        "relative_gain_vs_current_har": (
                            reference_score - score
                        )
                        / reference_score,
                    }
                )
    return pd.DataFrame(rows)


def build_tail_risk_report(
    gates: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    rows = []
    for component, gate in gates.items():
        rows.append(
            {
                "component": component,
                "candidate": gate["candidate"],
                "reference": gate["reference"],
                "var95_breach_rate": gate["var95_breach_rate"],
                "band_coverage": gate["band_coverage"],
                "es_ratio": gate["es_ratio"],
                "high_vol_relative_gain": gate[
                    "high_vol_relative_gain"
                ],
                "passed": gate["passed"],
                "failed_checks": ",".join(
                    name
                    for name, passed in gate["checks"].items()
                    if not passed
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    report_dir: Path,
    leaderboard: pd.DataFrame,
    gates: Mapping[str, object],
    external: pd.DataFrame,
    contributions: pd.DataFrame,
) -> None:
    lines = [
        "# Risk Engine volatility optimisation",
        "",
        "Primary target: 5-session realised volatility; secondary target: 20-session.",
        "All reported model choices are nested, annual walk-forward, and embargoed.",
        "",
        "## Leaderboard",
        "",
        "| h | model | mean QLIKE | gain vs HAR | positive years | worst year |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in leaderboard.itertuples(index=False):
        lines.append(
            f"| {row.horizon} | {row.model} | {row.mean_fold_qlike:.5f} | "
            f"{row.mean_relative_gain_vs_har:+.2%} | "
            f"{row.positive_years_vs_har}/{row.folds} | "
            f"{row.worst_year_gain_vs_har:+.2%} |"
        )
    legacy_path = (
        ROOT / "reports" / "risk_engine" / "risk_engine_v2_results.csv"
    )
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
        legacy = legacy[
            (legacy["split"] == "TEST")
            & legacy["model"].isin(["GARCH(1,1)", "EGARCH(1,1,1)"])
        ]
        lines += [
            "",
            "### Legacy GARCH controls (reference only)",
            "",
            "These controls use the earlier 2021–2023 fixed test split, so they "
            "are not mixed into the nested promotion ranking.",
            "",
            "| h | model | QLIKE | log-RMSE |",
            "|---:|---|---:|---:|",
        ]
        for row in legacy.itertuples(index=False):
            lines.append(
                f"| {row.horizon} | {row.model} | {row.qlike:.5f} | "
                f"{row.rmse_log:.5f} |"
            )
    lines += ["", "## Promotion gates", ""]
    for component, gate in gates.items():
        lines.append(
            f"- **{component} `{gate['candidate']}` vs "
            f"`{gate['reference']}`**: "
            f"gain {gate['aggregate_relative_gain']:+.2%}, "
            f"DM p={gate['dm_p']:.3g}, "
            f"bootstrap [{gate['bootstrap_ci'][0]:+.2%}, "
            f"{gate['bootstrap_ci'][2]:+.2%}] — "
            f"**{'PASS' if gate['passed'] else 'HOLD'}**"
        )
        failed = [
            name for name, passed in gate["checks"].items() if not passed
        ]
        if failed:
            lines.append(f"  - Failed: {', '.join(failed)}")
    lines += [
        "",
        "## External 2024+ price generalisation",
        "",
    ]
    if "relative_gain" in external.columns:
        for row in external.itertuples(index=False):
            lines.append(
                f"- {row.horizon}d `{row.scope}`: "
                f"{row.relative_gain:+.2%} QLIKE gain "
                f"({row.symbols} symbols, {row.n:,} rows)"
            )
    else:
        lines.append("- External evaluation unavailable.")
    stable = contributions[
        (contributions["stage"] == "drop_column")
        & (contributions["accepted"])
    ]
    lines += [
        "",
        "## Positive news features",
        "",
        "A feature is listed only when removing it worsened inner-fold QLIKE in",
        "at least 60% of folds with positive median contribution.",
        "",
    ]
    if stable.empty:
        lines.append("- No stable positive individual news feature.")
    else:
        counts = (
            stable.groupby(["horizon", "scope", "feature"])
            .agg(
                selected_outer_folds=("outer_year", "nunique"),
                median_gain=("median_gain", "median"),
            )
            .reset_index()
            .sort_values(
                ["horizon", "scope", "selected_outer_folds", "median_gain"],
                ascending=[True, True, False, False],
            )
        )
        counts = counts[counts["selected_outer_folds"] >= 4]
        if counts.empty:
            lines.append(
                "- No feature survived the final 4-of-6 outer-fold rule."
            )
        for row in counts.itertuples(index=False):
            lines.append(
                f"- {row.horizon}d `{row.scope}` `{row.feature}`: "
                f"{row.selected_outer_folds}/6 folds, "
                f"median drop-column gain {row.median_gain:+.3%}"
            )
    lines += [
        "",
        "## Guardrails",
        "",
        "- `coverage_active` is excluded because its upper endpoint is non-causal.",
        "- Missing/stale RSS never blocks the price forecast; the news multiplier is 1.",
        "- 2021-2023 is confirmation evidence, not a pristine untouched test.",
        "- The FNSPID-derived news model remains research/non-commercial.",
    ]
    (report_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def resolve_xgb_device(requested: str) -> str:
    if requested in {"cpu", "cuda"}:
        return requested
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=1,
            max_depth=1,
            tree_method="hist",
            device="cuda",
            objective="reg:gamma",
            verbosity=0,
        )
        model.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
        model.predict(np.array([[1.5]]))
        return "cuda"
    except Exception:
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument(
        "--outer-years",
        default=",".join(str(year) for year in OUTER_YEARS),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument(
        "--promote-if-passed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    outer_years = [int(value) for value in args.outer_years.split(",")]
    report_dir = DEFAULT_REPORT
    candidate_dir = DEFAULT_CANDIDATE
    if args.output_suffix:
        report_dir = Path(f"{DEFAULT_REPORT}_{args.output_suffix}")
        candidate_dir = Path(f"{DEFAULT_CANDIDATE}_{args.output_suffix}")
    report_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_xgb_device(args.device)
    print(f"XGBoost device: {device}")

    panel, groups, research_features, deployable_features = (
        build_historical_panel()
    )
    panel.to_parquet(PANEL_CACHE, index=False)
    print(
        f"panel: {len(panel):,} rows | {panel.symbol.nunique()} symbols | "
        f"price={len(PRICE_FEATURES)} research_news={len(research_features)} "
        f"deployable_news={len(deployable_features)}"
    )
    configs = parameter_configs(args.trials)

    folds, contributions, search, oof, _ = run_outer_experiment(
        panel,
        groups,
        research_features,
        deployable_features,
        outer_years,
        configs,
        min(args.top_k, len(configs)),
        device,
    )
    leaderboard = build_leaderboard(folds)
    deployable_news = leaderboard[
        (leaderboard["horizon"] == 5)
        & leaderboard["model"].isin(
            [
                "har_news_linear_deployable",
                "har_news_xgb_deployable",
            ]
        )
    ].sort_values("mean_fold_qlike")
    news_5d_candidate = str(deployable_news.iloc[0]["model"])
    price_5d_candidate = str(
        leaderboard[
            (leaderboard["horizon"] == 5)
            & leaderboard["model"].isin(
                ["ridge_har", "linear_gamma", "xgb_price"]
            )
        ]
        .sort_values("mean_fold_qlike")
        .iloc[0]["model"]
    )
    model_20d_candidate = _candidate_name(leaderboard, 20)

    price_5d, price_symbols = evaluate_gates(
        oof,
        folds,
        leaderboard,
        args.bootstrap_reps,
        candidates={5: price_5d_candidate},
        horizons=(5,),
    )
    news_5d, news_symbols = evaluate_gates(
        oof,
        folds,
        leaderboard,
        args.bootstrap_reps,
        candidates={5: news_5d_candidate},
        horizons=(5,),
        # No historical RSS archive exists.  This remains false until 60
        # mature live five-session forecasts have been scored.
        rss_shadow_mature=False,
    )
    model_20d, model_20d_symbols = evaluate_gates(
        oof,
        folds,
        leaderboard,
        args.bootstrap_reps,
        candidates={20: model_20d_candidate},
        horizons=(20,),
    )
    gates = {
        "price_5d": price_5d["5"],
        "news_5d": news_5d["5"],
        "model_20d": model_20d["20"],
    }
    per_symbol = pd.concat(
        [
            price_symbols.assign(component="price_5d"),
            news_symbols.assign(component="news_5d"),
            model_20d_symbols.assign(component="model_20d"),
        ],
        ignore_index=True,
    )
    horizon_payload, final_models, final_specs = train_final_bundle(
        panel,
        groups,
        research_features,
        deployable_features,
        contributions,
        leaderboard,
        configs,
        min(args.top_k, len(configs)),
        device,
        candidate_dir,
    )
    external = evaluate_external_price(
        final_models, final_specs, horizon_payload, device
    )
    if "relative_gain" in external.columns:
        external_5 = external[
            (external["horizon"] == 5)
            & (external["scope"] == "external_generalization")
        ]
        external_pass = (
            not external_5.empty
            and float(external_5.iloc[0]["relative_gain"]) >= 0
        )
    else:
        external_pass = False
    gates["price_5d"]["checks"]["external_price_not_worse"] = bool(
        external_pass
    )
    gates["price_5d"]["passed"] = bool(
        all(gates["price_5d"]["checks"].values())
    )
    if "relative_gain" in external.columns:
        external_20 = external[
            (external["horizon"] == 20)
            & (external["scope"] == "external_generalization")
        ]
        external_20_pass = (
            not external_20.empty
            and float(external_20.iloc[0]["relative_gain"]) >= 0
        )
    else:
        external_20_pass = False
    if gates["model_20d"]["candidate"] in {
        "ridge_har",
        "linear_gamma",
        "xgb_price",
    }:
        gates["model_20d"]["checks"][
            "external_price_not_worse"
        ] = bool(external_20_pass)
        gates["model_20d"]["passed"] = bool(
            all(gates["model_20d"]["checks"].values())
        )

    passed_components = [
        name for name, gate in gates.items() if gate["passed"]
    ]

    manifest = {
        "schema_version": 2,
        "model": "AURORA Risk Engine optimisation candidate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "partial_promotion_eligible_backend_parity_required"
            if passed_components
            else "candidate_hold"
        ),
        "primary_horizon": 5,
        "secondary_horizon": 20,
        "training_universe": sorted(panel["symbol"].unique().tolist()),
        "training_range": [
            panel["date"].min().date().isoformat(),
            panel["date"].max().date().isoformat(),
        ],
        "outer_years": outer_years,
        "xgboost_device": device,
        "trials": args.trials,
        "top_k": args.top_k,
        "data_hashes": {
            "training_dataset": sha256_file(
                ROOT / "data" / "processed" / "training_dataset.parquet"
            ),
            "extended_news_features": sha256_file(
                ROOT / "data" / "processed" / "extended_news_features.parquet"
            ),
            "risk_optimization_panel": sha256_file(PANEL_CACHE),
            "current_checkpoint": sha256_file(CURRENT_MODEL_PATH),
        },
        "fallback": {
            "news_optional": True,
            "missing_or_invalid_news_multiplier": 1.0,
            "price_forecast_always_available": True,
        },
        "horizons": horizon_payload,
        "gates": gates,
        "passed_components": passed_components,
        "external_price_generalization": external.to_dict("records"),
        "warning": (
            "News promotion requires 60 mature live RSS 5-day forecasts; "
            "2021-2023 has already been inspected by prior experiments."
        ),
    }
    _atomic_json(candidate_dir / "manifest.json", manifest)

    folds.to_csv(report_dir / "fold_metrics.csv", index=False)
    leaderboard.to_csv(report_dir / "leaderboard.csv", index=False)
    contributions.to_csv(
        report_dir / "feature_contributions.csv", index=False
    )
    search.to_csv(report_dir / "search_results.csv", index=False)
    oof.to_parquet(report_dir / "oof_predictions.parquet", index=False)
    per_symbol.to_csv(report_dir / "per_symbol_results.csv", index=False)
    build_regime_results(oof).to_csv(
        report_dir / "regime_results.csv", index=False
    )
    build_tail_risk_report(gates).to_csv(
        report_dir / "tail_risk_report.csv", index=False
    )
    external.to_csv(report_dir / "external_generalization.csv", index=False)
    _atomic_json(report_dir / "promotion_gates.json", gates)
    write_report(report_dir, leaderboard, gates, external, contributions)

    promotion_passed = bool(passed_components)
    promoted = False
    if args.promote_if_passed and promotion_passed:
        # The candidate bundle uses a schema-v2 multi-artifact layout.  Keep
        # the v1 official model intact until backend loading parity is proven
        # by the integration tests; record eligibility instead of making a
        # partial, non-atomic promotion.
        manifest["status"] = "promotion_eligible_backend_parity_required"
        _atomic_json(candidate_dir / "manifest.json", manifest)

    print("\nLEADERBOARD")
    print(
        leaderboard[
            [
                "horizon",
                "model",
                "mean_fold_qlike",
                "mean_relative_gain_vs_har",
                "positive_years_vs_har",
                "worst_year_gain_vs_har",
            ]
        ].to_string(index=False)
    )
    print("\nGATES")
    print(json.dumps(gates, indent=2))
    print(f"\nCandidate: {candidate_dir}")
    print(f"Report:    {report_dir}")
    print(f"Official checkpoint changed: {promoted}")


if __name__ == "__main__":
    main()
