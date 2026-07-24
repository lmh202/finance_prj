"""Test whether deployable news features improve the five-day XGBoost model.

This experiment is deliberately paired with the existing price-only XGBoost
OOF benchmark:

* the price-tree configuration for each outer year is inherited from the
  original inner-fold search;
* news feature family, direct/ratio target, HAR blend weight, calibration, and
  tree count are selected using inner folds only;
* the outer test year is evaluated once;
* zero-news observations remain in every fit with explicit causal state
  features.

The target is future five-session daily realised volatility. Outputs are
written to ``reports/risk_engine_xgb_news``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from optimize_risk_engine import (  # noqa: E402
    BLEND_WEIGHTS,
    _weighted_qlike,
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
    fit_har,
    fit_xgb_gamma,
    fit_xgb_gamma_fixed,
    inner_folds,
    moving_block_bootstrap_gain,
    optimal_sigma_scale,
    parameter_configs,
    qlike_loss,
    symbol_equal_qlike,
    symbol_equal_weights,
    time_fold,
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

ATTENTION_FEATURES = [
    "has_news",
    "log_count",
    "count_z20",
    "count_ratio20",
    "news_count_3d",
    "news_count_5d",
    "days_since_news",
    *CAUSAL_NEWS_STATES,
    "news_quality_available",
]

FINBERT_FEATURES = [
    "sent_mean",
    "sent_std",
    "sent_range",
    "sent_abs_mean",
    "sent_positive_share",
    "sent_negative_share",
    "sent_extreme_share",
    "sent_surprise20",
    "sent_abs_surprise20",
]

ALL_DEPLOYABLE_FEATURES = list(
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
    "all_deployable": ALL_DEPLOYABLE_FEATURES,
}


def split_inner(
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


def price_specifications() -> dict[int, dict[str, object]]:
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
            "price_variant": str(row.variant),
            "price_iterations": int(row.iterations),
            "price_blend_weight": float(row.blend_weight),
            "price_scale": float(row.scale),
        }
    missing = sorted(set(OUTER_YEARS) - set(result))
    if missing:
        raise ValueError(f"missing price XGBoost specifications: {missing}")
    return result


def family_inner_result(
    panel: pd.DataFrame,
    folds,
    outer_cutoff: pd.Timestamp,
    config: Mapping[str, float | int],
    news_features: list[str],
) -> dict[str, object]:
    features = PRICE_FEATURES + news_features
    fold_predictions = []
    iteration_lookup: dict[str, list[int]] = {
        "direct": [],
        "ratio": [],
    }
    for fold in folds:
        train, validation = split_inner(panel, fold, outer_cutoff)
        if validation.empty:
            continue
        har = fit_har(train, HORIZON)
        har_train = har.predict(train)
        har_validation = har.predict(validation)
        predictions = {"har": har_validation}
        for variant in ("direct", "ratio"):
            model = fit_xgb_gamma(
                train,
                validation,
                features,
                _xgb_target(train, HORIZON, variant, har_train),
                _xgb_target(
                    validation,
                    HORIZON,
                    variant,
                    har_validation,
                ),
                config,
                DEVICE,
            )
            predictions[variant] = _xgb_forecast(
                model,
                validation,
                features,
                variant,
                har_validation,
            )
            iteration_lookup[variant].append(
                int(getattr(model, "best_iteration", 299) + 1)
            )
        fold_predictions.append((validation, predictions))

    variants = []
    for variant in ("direct", "ratio"):
        for blend_weight in BLEND_WEIGHTS:
            frames = []
            forecasts = []
            for validation, predictions in fold_predictions:
                forecast = (
                    (1 - float(blend_weight)) * predictions["har"]
                    + float(blend_weight) * predictions[variant]
                )
                frames.append(validation)
                forecasts.append(forecast)
            combined = pd.concat(frames, ignore_index=True)
            combined_forecast = np.concatenate(forecasts)
            scale = optimal_sigma_scale(
                combined[TARGET].to_numpy(float),
                combined_forecast,
                symbol_equal_weights(combined),
            )
            score = _weighted_qlike(
                combined,
                HORIZON,
                combined_forecast * scale,
            )
            variants.append(
                {
                    "variant": variant,
                    "blend_weight": float(blend_weight),
                    "scale": float(scale),
                    "inner_qlike": float(score),
                    "n_estimators": int(
                        np.median(iteration_lookup[variant])
                    ),
                    "inner_rows": len(combined),
                }
            )
    return min(
        variants,
        key=lambda row: (
            row["inner_qlike"],
            row["blend_weight"],
        ),
    )


def fit_outer_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: Mapping[str, float | int],
    selected: Mapping[str, object],
    news_features: list[str],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    features = PRICE_FEATURES + news_features
    har = fit_har(train, HORIZON)
    har_train = har.predict(train)
    har_test = har.predict(test)
    variant = str(selected["variant"])
    model = fit_xgb_gamma_fixed(
        train,
        features,
        _xgb_target(train, HORIZON, variant, har_train),
        config,
        DEVICE,
        int(selected["n_estimators"]),
    )
    raw = _xgb_forecast(
        model,
        test,
        features,
        variant,
        har_test,
    )
    forecast = (
        (1 - float(selected["blend_weight"])) * har_test
        + float(selected["blend_weight"]) * raw
    ) * float(selected["scale"])

    importance = model.get_booster().get_score(
        importance_type="total_gain"
    )
    rows = [
        {
            "feature": feature,
            "total_gain": float(value),
            "is_news": feature in news_features,
        }
        for feature, value in importance.items()
    ]
    return forecast, rows


def stock_equal_score(
    frame: pd.DataFrame,
    realized: str,
    forecast: str,
) -> float:
    return symbol_equal_qlike(
        frame,
        realized,
        frame[forecast].to_numpy(float),
    )


def summarise(frame: pd.DataFrame) -> dict[str, object]:
    target = frame[TARGET].to_numpy(float)
    reference = frame[BASE_FORECAST_COLUMN].to_numpy(float)
    candidate = frame["forecast__xgb_price_news"].to_numpy(float)
    reference_loss = qlike_loss(target, reference)
    candidate_loss = qlike_loss(target, candidate)
    reference_qlike = symbol_equal_qlike(
        frame, TARGET, reference
    )
    candidate_qlike = symbol_equal_qlike(
        frame, TARGET, candidate
    )
    gain = (reference_qlike - candidate_qlike) / reference_qlike
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
        seed=RANDOM_SEED + 1,
    )

    yearly = {}
    for year, group in frame.groupby("test_year"):
        reference_score = stock_equal_score(
            group, TARGET, BASE_FORECAST_COLUMN
        )
        candidate_score = stock_equal_score(
            group, TARGET, "forecast__xgb_price_news"
        )
        yearly[str(int(year))] = (
            reference_score - candidate_score
        ) / reference_score

    symbol_rows = []
    for symbol, group in frame.groupby("symbol"):
        reference_score = stock_equal_score(
            group, TARGET, BASE_FORECAST_COLUMN
        )
        candidate_score = stock_equal_score(
            group, TARGET, "forecast__xgb_price_news"
        )
        symbol_rows.append(
            {
                "symbol": symbol,
                "reference_qlike": reference_score,
                "candidate_qlike": candidate_score,
                "relative_gain": (
                    reference_score - candidate_score
                )
                / reference_score,
            }
        )

    regime_rows = []
    high_cutoff = float(frame[TARGET].quantile(2 / 3))
    for scope, mask in {
        "all": pd.Series(True, index=frame.index),
        "news_active": frame["has_news"].eq(1),
        "zero_news": frame["has_news"].eq(0),
        "high_volatility": frame[TARGET].ge(high_cutoff),
    }.items():
        group = frame.loc[mask].reset_index(drop=True)
        reference_score = stock_equal_score(
            group, TARGET, BASE_FORECAST_COLUMN
        )
        candidate_score = stock_equal_score(
            group, TARGET, "forecast__xgb_price_news"
        )
        regime_rows.append(
            {
                "scope": scope,
                "rows": len(group),
                "symbols": int(group["symbol"].nunique()),
                "reference_qlike": reference_score,
                "candidate_qlike": candidate_score,
                "relative_gain": (
                    reference_score - candidate_score
                )
                / reference_score,
            }
        )

    symbols = pd.DataFrame(symbol_rows)
    regimes = pd.DataFrame(regime_rows)
    return {
        "reference_model": "XGBoost Gamma price-only",
        "candidate_model": "XGBoost Gamma price+news",
        "reference_qlike": float(reference_qlike),
        "candidate_qlike": float(candidate_qlike),
        "relative_gain": float(gain),
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
        "_symbol_frame": symbols,
        "_regime_frame": regimes,
    }


def write_report(
    summary: Mapping[str, object],
    selection: pd.DataFrame,
    news_gain_share: float,
) -> None:
    yearly = summary["yearly_gains"]
    lines = [
        "# Direct XGBoost news-feature experiment",
        "",
        "## Question",
        "",
        "Can deployable news features improve the existing price-only XGBoost "
        "forecast of future five-session realised volatility?",
        "",
        "## Result",
        "",
        f"- Price-only XGBoost QLIKE: **{summary['reference_qlike']:.6f}**",
        f"- Price + News XGBoost QLIKE: **{summary['candidate_qlike']:.6f}**",
        f"- Relative QLIKE gain: **{summary['relative_gain']:+.2%}**",
        f"- DM test: **p={summary['dm_p']:.6g}**",
        "- 95% moving-block bootstrap gain: "
        f"**[{summary['bootstrap_95'][0]:+.2%}, "
        f"{summary['bootstrap_95'][2]:+.2%}]**",
        f"- Positive outer years: **{summary['positive_years']}/6**",
        f"- Positive stocks: **{summary['positive_symbol_share']:.1%}**",
        f"- Median stock gain: **{summary['median_symbol_gain']:+.2%}**",
        f"- News-active rows: **{summary['news_active_gain']:+.2%}**",
        f"- Zero-news rows: **{summary['zero_news_gain']:+.2%}**",
        f"- High-volatility rows: **{summary['high_volatility_gain']:+.2%}**",
        f"- News share of final-tree total gain: **{news_gain_share:.1%}**",
        "",
        "## Outer-year gains",
        "",
        "| Test year | QLIKE gain vs price-only XGBoost |",
        "|---:|---:|",
    ]
    for year, gain in yearly.items():
        lines.append(f"| {year} | {gain:+.2%} |")
    lines += [
        "",
        "## Leakage controls",
        "",
        "- The price-tree configuration is inherited from the original inner "
        "search for each outer year.",
        "- News family, target form, HAR blend, calibration, and tree count "
        "are selected inside the outer training window.",
        "- The final inner validation year is truncated at the outer embargo "
        "cutoff.",
        "- Every zero-news observation remains in training and scoring.",
        "- The outer test year is not used for feature-family selection.",
        "",
        "## Selected news families",
        "",
    ]
    counts = Counter(
        selection.loc[selection["selected"], "news_family"].tolist()
    )
    for family, count in sorted(counts.items()):
        lines.append(f"- `{family}`: {count}/6 outer folds")
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if (
        float(summary["relative_gain"]) > 0
        and float(summary["dm_p"]) < 0.05
        and int(summary["positive_years"]) >= 4
    ):
        lines.append(
            "The direct Price + News XGBoost candidate shows a statistically "
            "credible incremental news contribution and should advance to "
            "external calibration and live-RSS validation."
        )
    else:
        lines.append(
            "The experiment does not establish a stable positive news "
            "increment for direct XGBoost. The result should not replace the "
            "formal HAR-X + News model."
        )
    (OUTPUT / "report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    required_news = sorted(
        set(feature for family in NEWS_FAMILIES.values() for feature in family)
    )
    missing = [
        feature
        for feature in PRICE_FEATURES + required_news
        if feature not in panel.columns
    ]
    if missing:
        raise ValueError(f"panel is missing features: {missing}")

    specifications = price_specifications()
    outer_predictions = []
    selection_rows = []
    importance_rows = []
    for outer_year in OUTER_YEARS:
        outer_fold = time_fold(panel, outer_year, HORIZON)
        outer_train = panel.loc[outer_fold.train_mask].dropna(
            subset=PRICE_FEATURES + [TARGET]
        ).reset_index(drop=True)
        outer_test = panel.loc[outer_fold.test_mask].dropna(
            subset=PRICE_FEATURES + [TARGET]
        ).reset_index(drop=True)
        folds = inner_folds(panel, outer_year, HORIZON)
        specification = specifications[outer_year]
        config = specification["config"]
        candidates = []
        for family_name, news_features in NEWS_FAMILIES.items():
            result = family_inner_result(
                panel,
                folds,
                outer_fold.train_cutoff,
                config,
                news_features,
            )
            result = {
                **result,
                "outer_year": outer_year,
                "news_family": family_name,
                "news_feature_count": len(news_features),
                "config_id": int(config["config_id"]),
                "selected": False,
            }
            candidates.append(result)
            selection_rows.append(result.copy())
        selected = min(
            candidates,
            key=lambda row: (
                row["inner_qlike"],
                row["news_feature_count"],
            ),
        )
        for row in selection_rows:
            if (
                row["outer_year"] == outer_year
                and row["news_family"] == selected["news_family"]
            ):
                row["selected"] = True
        news_features = NEWS_FAMILIES[str(selected["news_family"])]
        forecast, fold_importance = fit_outer_candidate(
            outer_train,
            outer_test,
            config,
            selected,
            news_features,
        )
        outer_predictions.append(
            pd.DataFrame(
                {
                    "date": outer_test["date"].to_numpy(),
                    "symbol": outer_test["symbol"].to_numpy(),
                    "test_year": outer_year,
                    TARGET: outer_test[TARGET].to_numpy(float),
                    "has_news": outer_test["has_news"].to_numpy(int),
                    "forecast__xgb_price_news": forecast,
                    "selected_news_family": selected["news_family"],
                }
            )
        )
        for row in fold_importance:
            importance_rows.append(
                {
                    "outer_year": outer_year,
                    "news_family": selected["news_family"],
                    **row,
                }
            )
        print(
            f"{outer_year}: family={selected['news_family']}, "
            f"variant={selected['variant']}, "
            f"blend={selected['blend_weight']:.2f}, "
            f"trees={selected['n_estimators']}, "
            f"inner QLIKE={selected['inner_qlike']:.6f}",
            flush=True,
        )

    candidate_oof = pd.concat(outer_predictions, ignore_index=True)
    baseline = pd.read_parquet(BASE_OOF_PATH)
    baseline["date"] = pd.to_datetime(baseline["date"]).dt.normalize()
    baseline = baseline[baseline["horizon"].eq(HORIZON)][
        ["date", "symbol", BASE_FORECAST_COLUMN]
    ]
    comparison = candidate_oof.merge(
        baseline,
        on=["date", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    summary = summarise(comparison)
    symbols = summary.pop("_symbol_frame")
    regimes = summary.pop("_regime_frame")

    selection = pd.DataFrame(selection_rows)
    importance = pd.DataFrame(importance_rows)
    total_gain = float(importance["total_gain"].sum())
    news_gain = float(
        importance.loc[importance["is_news"], "total_gain"].sum()
    )
    news_gain_share = news_gain / total_gain if total_gain > 0 else 0.0
    summary["selected_news_families"] = (
        selection.loc[selection["selected"], "news_family"]
        .value_counts()
        .to_dict()
    )
    summary["news_total_gain_share"] = news_gain_share

    comparison.to_parquet(OUTPUT / "oof_predictions.parquet", index=False)
    selection.to_csv(OUTPUT / "inner_selection.csv", index=False)
    importance.to_csv(OUTPUT / "feature_importance.csv", index=False)
    symbols.to_csv(OUTPUT / "per_symbol_results.csv", index=False)
    regimes.to_csv(OUTPUT / "regime_results.csv", index=False)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_report(summary, selection, news_gain_share)
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
