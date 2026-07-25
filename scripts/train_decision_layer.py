"""Train, validate, test, and serialize the AURORA final decision layer.

Model selection uses 2018-2020 expanding walk-forward validation.  The chosen
specification is then evaluated once on the locked 2021-2023 historical test
and on the independent 2024-2026 panel.  Portfolio evaluation is cost-aware
and uses the formal HAR-X + News volatility forecast for risk sizing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from decision_layer_core import (  # noqa: E402
    MODEL_SPECS,
    PRICE_FEATURES,
    RANDOM_SEED,
    TARGET,
    backtest_metrics,
    bounded_prediction,
    fit_estimator,
    model_spec,
    moving_block_utility_gain,
    prediction_metrics,
    run_backtest,
    walk_forward_predictions,
)

DATA = ROOT / "data" / "processed" / "decision_dataset.parquet"
EXTERNAL_DATA = (
    ROOT / "data" / "processed" / "decision_external_dataset.parquet"
)
REPORT = ROOT / "reports" / "decision_layer"
CHECKPOINT = ROOT / "data" / "processed" / "decision_model"
VALIDATION_YEARS = (2018, 2019, 2020)
TEST_YEARS = (2021, 2022, 2023)
COST_SCENARIOS = (10.0, 25.0, 50.0)
PRIMARY_COST_BPS = 25.0
RISK_AVERSION = 6.0
SIGNAL_SCALES = (
    0.0,
    0.005,
    0.010,
    0.020,
    0.025,
    0.030,
    0.035,
    0.040,
    0.045,
    0.050,
    0.100,
    0.250,
    0.500,
    0.750,
    1.000,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_prediction_intervals(
    predictions: pd.DataFrame,
    residuals: np.ndarray,
) -> pd.DataFrame:
    output = predictions.copy()
    q10, q90 = np.quantile(residuals, [0.10, 0.90])
    output["prediction_q10"] = np.clip(
        output["prediction"] + q10,
        -0.50,
        0.50,
    )
    output["prediction_q50"] = output["prediction"]
    output["prediction_q90"] = np.clip(
        output["prediction"] + q90,
        -0.50,
        0.50,
    )
    sorted_residual = np.sort(np.asarray(residuals, dtype=float))
    threshold = -output["prediction"].to_numpy(dtype=float)
    position = np.searchsorted(sorted_residual, threshold, side="right")
    output["probability_positive_excess"] = (
        len(sorted_residual) - position
    ) / max(len(sorted_residual), 1)
    return output


def scale_predictions(
    predictions: pd.DataFrame,
    signal_scale: float,
) -> pd.DataFrame:
    """Apply a validation-selected shrinkage to the tradable return signal."""
    output = predictions.copy()
    if "raw_prediction" not in output:
        output["raw_prediction"] = output["prediction"]
    output["prediction"] = (
        output["raw_prediction"].to_numpy(dtype=float) * signal_scale
    )
    output["signal_scale"] = signal_scale
    return output


def evaluate_backtest(
    panel: pd.DataFrame,
    predictions: pd.DataFrame | None,
    *,
    period: str,
    start: str,
    end: str,
    strategy: str,
    cost_bps: float,
    scope: str = "all",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    daily, trades = run_backtest(
        panel,
        predictions,
        start=start,
        end=end,
        strategy=strategy,
        transaction_cost_bps=cost_bps,
        risk_aversion=RISK_AVERSION,
    )
    metrics = backtest_metrics(daily, risk_aversion=RISK_AVERSION)
    metrics.update(
        {
            "period": period,
            "scope": scope,
            "strategy": strategy,
            "transaction_cost_bps": cost_bps,
        }
    )
    daily["period"] = period
    daily["scope"] = scope
    daily["transaction_cost_bps"] = cost_bps
    trades["period"] = period
    trades["scope"] = scope
    trades["transaction_cost_bps"] = cost_bps
    return metrics, daily, trades


def yearly_metrics(
    daily: pd.DataFrame,
    strategy_name: str,
    period: str,
) -> list[dict[str, Any]]:
    rows = []
    for year, group in daily.groupby(daily["date"].dt.year):
        metrics = backtest_metrics(group, risk_aversion=RISK_AVERSION)
        metrics.update(
            {
                "period": period,
                "year": int(year),
                "strategy": strategy_name,
            }
        )
        rows.append(metrics)
    return rows


def _report_markdown(
    selected_name: str,
    selected_strategy: str,
    signal_scale: float,
    production_mode: str,
    prediction_table: pd.DataFrame,
    portfolio_table: pd.DataFrame,
    gates: dict[str, Any],
) -> str:
    validation = portfolio_table.loc[
        portfolio_table["period"].eq("validation")
        & portfolio_table["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ]
    test = portfolio_table.loc[
        portfolio_table["period"].eq("locked_test")
        & portfolio_table["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ]
    external = portfolio_table.loc[
        portfolio_table["period"].eq("external")
        & portfolio_table["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
        & portfolio_table["scope"].eq("all")
    ]

    def markdown_table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        rule = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = [
            "| "
            + " | ".join(str(value) for value in row)
            + " |"
            for row in frame.itertuples(index=False, name=None)
        ]
        return "\n".join([header, rule, *rows])

    def table(frame: pd.DataFrame) -> str:
        columns = [
            "strategy",
            "certainty_equivalent",
            "sharpe",
            "max_drawdown",
            "annual_turnover",
            "total_transaction_cost",
        ]
        return markdown_table(frame[columns].round(4))

    predictive = prediction_table[
        [
            "period",
            "model",
            "rank_ic_mean",
            "top_bottom_20d_spread",
            "r2",
        ]
    ].round(4)
    passed = sum(bool(value) for value in gates["checks"].values())
    total = len(gates["checks"])
    return f"""# Final Decision Layer Validation

## Decision

- Selected return model: **{selected_name}**
- Validation-selected return-signal scale: **{signal_scale:.3f}**
- Evaluated decision strategy: **{selected_strategy}**
- Formal production mode: **{production_mode}**
- Promotion checks passed: **{passed}/{total}**

The return model predicts the next-20-session stock return relative to SPY.
The formal HAR-X + News five-session volatility forecast controls position
sizing and the portfolio covariance.  DeepSeek is not allowed to alter the
numeric decision; it is an explanation-only layer.

## Evaluation protocol

- Pretraining history: 2013-2017
- Walk-forward model selection and signal calibration: 2018-2020
- Locked historical test: 2021-2023
- External transfer test: 2024-2026, including previously unseen symbols
- Leakage control: 20 trading sessions between training labels and each fold
- Portfolio stress: 10, 25, and 50 bps one-way transaction costs

## Predictive results

{markdown_table(predictive)}

## 2018-2020 validation portfolio results (25 bps)

{table(validation)}

## 2021-2023 locked historical test (25 bps)

{table(test)}

## 2024-2026 external test (25 bps)

{table(external)}

## Promotion gates

```json
{json.dumps(gates, indent=2)}
```

## Interpretation

Average direction accuracy is not the promotion target.  A candidate is useful
only when its ranking survives transaction costs and improves realised
risk-adjusted utility over the risk-only allocation.  If the ML gates fail,
the checkpoint retains the selected model for research while the backend uses
the deterministic risk-only optimiser.  The selected shrinkage shows that a
small return tilt may be useful, but it remains disabled until both statistical
uncertainty and external transfer gates pass.
"""


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    historical = pd.read_parquet(DATA)
    external = pd.read_parquet(EXTERNAL_DATA)
    historical["date"] = pd.to_datetime(historical["date"]).dt.normalize()
    external["date"] = pd.to_datetime(external["date"]).dt.normalize()

    prediction_rows: list[dict[str, Any]] = []
    validation_predictions: dict[str, pd.DataFrame] = {}
    portfolio_rows: list[dict[str, Any]] = []
    daily_outputs: list[pd.DataFrame] = []
    trade_outputs: list[pd.DataFrame] = []

    print("=" * 88)
    print("DECISION LAYER MODEL SELECTION | 2018-2020 VALIDATION")
    print("=" * 88)
    for spec in MODEL_SPECS:
        name = str(spec["name"])
        print(f"fitting {name} ...", flush=True)
        predictions = walk_forward_predictions(
            historical,
            VALIDATION_YEARS,
            spec,
        )
        validation_predictions[name] = predictions
        metrics = prediction_metrics(predictions)
        prediction_rows.append(
            {"period": "validation", "model": name, **metrics}
        )
        bt_metrics, daily, trades = evaluate_backtest(
            historical,
            predictions,
            period="validation",
            start="2018-01-01",
            end="2020-12-31",
            strategy="ml",
            cost_bps=PRIMARY_COST_BPS,
        )
        bt_metrics["strategy"] = name
        portfolio_rows.append(bt_metrics)
        daily["strategy"] = name
        trades["strategy"] = name
        daily_outputs.append(daily)
        trade_outputs.append(trades)
        print(
            f"  rank IC={metrics['rank_ic_mean']:+.4f} | "
            f"CER={bt_metrics['certainty_equivalent']:+.4f} | "
            f"Sharpe={bt_metrics['sharpe']:+.3f}",
            flush=True,
        )

    for baseline in ("equal_weight", "risk_only", "momentum_rule"):
        bt_metrics, daily, trades = evaluate_backtest(
            historical,
            None,
            period="validation",
            start="2018-01-01",
            end="2020-12-31",
            strategy=baseline,
            cost_bps=PRIMARY_COST_BPS,
        )
        portfolio_rows.append(bt_metrics)
        daily_outputs.append(daily)
        trade_outputs.append(trades)

    selection = pd.DataFrame(portfolio_rows)
    selected_name = str(
        selection.loc[
            selection["period"].eq("validation")
            & ~selection["strategy"].isin(
                ["equal_weight", "risk_only", "momentum_rule"]
            )
        ]
        .sort_values(
            ["certainty_equivalent", "sharpe"],
            ascending=False,
        )
        .iloc[0]["strategy"]
    )
    selected_spec = model_spec(selected_name)
    selected_validation_raw = validation_predictions[selected_name]

    # Return forecasts are noisy and often too large for direct optimisation.
    # Select one fixed shrinkage factor on validation only, then lock it before
    # either holdout is evaluated.
    scale_rows: list[dict[str, Any]] = []
    for signal_scale in SIGNAL_SCALES:
        scaled = scale_predictions(selected_validation_raw, signal_scale)
        metrics, _, _ = evaluate_backtest(
            historical,
            scaled,
            period="signal_scale_validation",
            start="2018-01-01",
            end="2020-12-31",
            strategy="ml",
            cost_bps=PRIMARY_COST_BPS,
        )
        scale_rows.append(
            {
                "signal_scale": signal_scale,
                "certainty_equivalent": metrics["certainty_equivalent"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "annual_turnover": metrics["annual_turnover"],
            }
        )
    scale_table = pd.DataFrame(scale_rows).sort_values(
        ["certainty_equivalent", "sharpe"],
        ascending=False,
    )
    selected_scale = float(scale_table.iloc[0]["signal_scale"])
    selected_strategy = f"{selected_name}_scaled"
    selected_validation = scale_predictions(
        selected_validation_raw,
        selected_scale,
    )
    scaled_predictive = prediction_metrics(selected_validation)
    prediction_rows.append(
        {
            "period": "validation_deployed_signal",
            "model": selected_strategy,
            **scaled_predictive,
        }
    )
    scaled_metrics, scaled_daily, scaled_trades = evaluate_backtest(
        historical,
        selected_validation,
        period="validation",
        start="2018-01-01",
        end="2020-12-31",
        strategy="ml",
        cost_bps=PRIMARY_COST_BPS,
    )
    scaled_metrics["strategy"] = selected_strategy
    scaled_daily["strategy"] = selected_strategy
    scaled_trades["strategy"] = selected_strategy
    portfolio_rows.append(scaled_metrics)
    daily_outputs.append(scaled_daily)
    trade_outputs.append(scaled_trades)
    print(
        f"\nlocked model specification: {selected_name} | "
        f"signal scale={selected_scale:.3f}\n",
        flush=True,
    )

    print("=" * 88)
    print("LOCKED HISTORICAL TEST | 2021-2023")
    print("=" * 88)
    locked_predictions = scale_predictions(
        walk_forward_predictions(
            historical,
            TEST_YEARS,
            selected_spec,
        ),
        selected_scale,
    )
    locked_predictive = prediction_metrics(locked_predictions)
    prediction_rows.append(
        {
            "period": "locked_test",
            "model": selected_name,
            **locked_predictive,
        }
    )
    for strategy in ("equal_weight", "risk_only", "momentum_rule", "ml"):
        predictions = locked_predictions if strategy == "ml" else None
        metrics, daily, trades = evaluate_backtest(
            historical,
            predictions,
            period="locked_test",
            start="2021-01-01",
            end="2023-12-31",
            strategy=strategy,
            cost_bps=PRIMARY_COST_BPS,
        )
        if strategy == "ml":
            metrics["strategy"] = selected_strategy
            daily["strategy"] = selected_strategy
            trades["strategy"] = selected_strategy
        portfolio_rows.append(metrics)
        daily_outputs.append(daily)
        trade_outputs.append(trades)

    # The model format is locked.  Refit on all mature historical labels.
    final_train = historical.dropna(subset=PRICE_FEATURES + [TARGET])
    final_estimator = fit_estimator(final_train, selected_spec)
    joblib_path = CHECKPOINT / "return_model.joblib"
    joblib.dump(final_estimator, joblib_path)
    if selected_spec["kind"] == "xgboost":
        final_estimator.save_model(CHECKPOINT / "return_model.xgb.json")

    external_predictions = external[
        ["date", "symbol", TARGET]
    ].copy()
    external_predictions["raw_prediction"] = bounded_prediction(
        final_estimator,
        external,
    )
    external_predictions["prediction"] = (
        external_predictions["raw_prediction"] * selected_scale
    )
    external_predictions["signal_scale"] = selected_scale
    external_predictions["model"] = selected_strategy
    external_predictive = prediction_metrics(external_predictions)
    prediction_rows.append(
        {
            "period": "external",
            "model": selected_name,
            **external_predictive,
        }
    )

    print("=" * 88)
    print("EXTERNAL TEST | 2024-2026")
    print("=" * 88)
    for strategy in ("equal_weight", "risk_only", "momentum_rule", "ml"):
        predictions = external_predictions if strategy == "ml" else None
        metrics, daily, trades = evaluate_backtest(
            external,
            predictions,
            period="external",
            start="2024-01-01",
            end="2026-12-31",
            strategy=strategy,
            cost_bps=PRIMARY_COST_BPS,
        )
        if strategy == "ml":
            metrics["strategy"] = selected_strategy
            daily["strategy"] = selected_strategy
            trades["strategy"] = selected_strategy
        portfolio_rows.append(metrics)
        daily_outputs.append(daily)
        trade_outputs.append(trades)

    # External transfer is reported separately for seen and unseen symbols.
    scopes = {
        "original_research": external["groups"].str.contains(
            "original_research", na=False
        ),
        "unseen": ~external["groups"].str.contains(
            "original_research", na=False
        ),
    }
    for scope, mask in scopes.items():
        scoped_panel = external.loc[mask].copy()
        scoped_symbols = set(scoped_panel["symbol"])
        scoped_predictions = external_predictions.loc[
            external_predictions["symbol"].isin(scoped_symbols)
        ]
        for strategy in ("risk_only", "ml"):
            predictions = scoped_predictions if strategy == "ml" else None
            metrics, daily, trades = evaluate_backtest(
                scoped_panel,
                predictions,
                period="external",
                scope=scope,
                start="2024-01-01",
                end="2026-12-31",
                strategy=strategy,
                cost_bps=PRIMARY_COST_BPS,
            )
            if strategy == "ml":
                metrics["strategy"] = selected_strategy
                daily["strategy"] = selected_strategy
                trades["strategy"] = selected_strategy
            portfolio_rows.append(metrics)
            daily_outputs.append(daily)
            trade_outputs.append(trades)

    # Cost stress is run after selection and does not alter model choice.
    for period, panel, predictions, start, end in (
        (
            "validation",
            historical,
            selected_validation,
            "2018-01-01",
            "2020-12-31",
        ),
        (
            "locked_test",
            historical,
            locked_predictions,
            "2021-01-01",
            "2023-12-31",
        ),
        (
            "external",
            external,
            external_predictions,
            "2024-01-01",
            "2026-12-31",
        ),
    ):
        for cost in COST_SCENARIOS:
            if cost == PRIMARY_COST_BPS:
                continue
            for strategy in ("risk_only", "ml"):
                use_prediction = predictions if strategy == "ml" else None
                metrics, daily, trades = evaluate_backtest(
                    panel,
                    use_prediction,
                    period=period,
                    start=start,
                    end=end,
                    strategy=strategy,
                    cost_bps=cost,
                )
                if strategy == "ml":
                    metrics["strategy"] = selected_strategy
                    daily["strategy"] = selected_strategy
                    trades["strategy"] = selected_strategy
                portfolio_rows.append(metrics)
                daily_outputs.append(daily)
                trade_outputs.append(trades)

    # Prediction intervals use only walk-forward residuals.
    selected_oof = pd.concat(
        [selected_validation, locked_predictions],
        ignore_index=True,
    )
    residuals = (
        selected_oof[TARGET] - selected_oof["prediction"]
    ).to_numpy(dtype=float)
    selected_oof = add_prediction_intervals(selected_oof, residuals)
    external_predictions = add_prediction_intervals(
        external_predictions,
        residuals,
    )

    portfolio_table = pd.DataFrame(portfolio_rows)
    prediction_table = pd.DataFrame(prediction_rows)
    all_daily = pd.concat(daily_outputs, ignore_index=True)
    all_trades = pd.concat(
        [frame for frame in trade_outputs if not frame.empty],
        ignore_index=True,
    )

    primary_test = all_daily.loc[
        all_daily["period"].eq("locked_test")
        & all_daily["scope"].eq("all")
        & all_daily["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ]
    risk_test = primary_test.loc[
        primary_test["strategy"].eq("risk_only")
    ]
    ml_test = primary_test.loc[
        primary_test["strategy"].eq(selected_strategy)
    ]
    bootstrap = moving_block_utility_gain(risk_test, ml_test)

    yearly_rows: list[dict[str, Any]] = []
    for strategy in ("risk_only", selected_strategy):
        strategy_daily = primary_test.loc[
            primary_test["strategy"].eq(strategy)
        ]
        yearly_rows.extend(
            yearly_metrics(strategy_daily, strategy, "locked_test")
        )
    yearly_table = pd.DataFrame(yearly_rows)
    pivot_cer = yearly_table.pivot(
        index="year",
        columns="strategy",
        values="certainty_equivalent",
    )
    positive_test_years = int(
        (
            pivot_cer[selected_strategy] - pivot_cer["risk_only"]
        ).gt(0).sum()
    )

    def metric(
        period: str,
        strategy: str,
        column: str,
        scope: str = "all",
    ) -> float:
        selected = portfolio_table.loc[
            portfolio_table["period"].eq(period)
            & portfolio_table["scope"].eq(scope)
            & portfolio_table["strategy"].eq(strategy)
            & portfolio_table["transaction_cost_bps"].eq(PRIMARY_COST_BPS),
            column,
        ]
        return float(selected.iloc[0])

    test_ml_cer = metric(
        "locked_test",
        selected_strategy,
        "certainty_equivalent",
    )
    test_risk_cer = metric("locked_test", "risk_only", "certainty_equivalent")
    test_ml_sharpe = metric("locked_test", selected_strategy, "sharpe")
    test_risk_sharpe = metric("locked_test", "risk_only", "sharpe")
    test_ml_drawdown = metric(
        "locked_test",
        selected_strategy,
        "max_drawdown",
    )
    test_risk_drawdown = metric("locked_test", "risk_only", "max_drawdown")
    test_ml_es = metric("locked_test", selected_strategy, "daily_es95")
    test_risk_es = metric("locked_test", "risk_only", "daily_es95")
    external_ml_cer = metric(
        "external",
        selected_strategy,
        "certainty_equivalent",
    )
    external_risk_cer = metric("external", "risk_only", "certainty_equivalent")
    external_unseen_ml = metric(
        "external",
        selected_strategy,
        "certainty_equivalent",
        "unseen",
    )
    external_unseen_risk = metric(
        "external",
        "risk_only",
        "certainty_equivalent",
        "unseen",
    )
    max_weight = metric(
        "locked_test",
        selected_strategy,
        "maximum_weight",
    )
    max_change = metric(
        "locked_test",
        selected_strategy,
        "maximum_change",
    )
    min_active = metric(
        "locked_test",
        selected_strategy,
        "minimum_active_trade",
    )

    checks = {
        "locked_test_cer_above_risk_only": test_ml_cer > test_risk_cer,
        "locked_test_sharpe_above_risk_only": (
            test_ml_sharpe > test_risk_sharpe
        ),
        "positive_test_years_at_least_2_of_3": positive_test_years >= 2,
        "bootstrap_utility_gain_lower_bound_positive": bootstrap[0] > 0,
        "max_drawdown_not_worse_by_more_than_2pp": (
            test_ml_drawdown >= test_risk_drawdown - 0.02
        ),
        "es_not_worse_by_more_than_10pct": (
            test_ml_es >= test_risk_es - abs(test_risk_es) * 0.10
        ),
        "external_cer_not_below_risk_only": (
            external_ml_cer >= external_risk_cer
        ),
        "external_unseen_cer_not_below_risk_only": (
            external_unseen_ml >= external_unseen_risk
        ),
        "max_position_constraint": max_weight <= 0.2001,
        "max_change_constraint": max_change <= 0.0501,
        "minimum_trade_constraint": (
            min_active == 0 or min_active >= 0.0099
        ),
    }
    promoted = all(checks.values())
    production_mode = "ml_return_plus_risk" if promoted else "risk_only"
    gates = {
        "promoted": promoted,
        "production_mode": production_mode,
        "selected_model": selected_name,
        "selected_strategy": selected_strategy,
        "return_signal_scale": selected_scale,
        "checks": checks,
        "statistics": {
            "bootstrap_utility_gain_95": list(bootstrap),
            "positive_test_years": positive_test_years,
            "locked_test_cer_gain": test_ml_cer - test_risk_cer,
            "locked_test_sharpe_gain": test_ml_sharpe - test_risk_sharpe,
            "external_cer_gain": external_ml_cer - external_risk_cer,
            "external_unseen_cer_gain": (
                external_unseen_ml - external_unseen_risk
            ),
        },
    }

    metadata = {
        "schema_version": 1,
        "model_version": "decision-layer-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "production_mode": production_mode,
        "selected_model": selected_name,
        "selected_strategy": selected_strategy,
        "selected_spec": selected_spec,
        "return_signal_scale": selected_scale,
        "feature_order": PRICE_FEATURES,
        "target": TARGET,
        "target_definition": "20-session stock return minus SPY return",
        "risk_input": "formal HAR-X + News five-session sigma",
        "rebalance_sessions": 5,
        "risk_aversion": RISK_AVERSION,
        "primary_transaction_cost_bps": PRIMARY_COST_BPS,
        "residual_quantiles": {
            "q10": float(np.quantile(residuals, 0.10)),
            "q90": float(np.quantile(residuals, 0.90)),
        },
        "validation_years": list(VALIDATION_YEARS),
        "locked_test_years": list(TEST_YEARS),
        "external_range": [
            str(external["date"].min().date()),
            str(external["date"].max().date()),
        ],
        "training_range": [
            str(final_train["date"].min().date()),
            str(final_train["date"].max().date()),
        ],
        "artifacts": {
            "return_model": joblib_path.name,
            "training_data_sha256": sha256(DATA),
            "external_data_sha256": sha256(EXTERNAL_DATA),
            "model_sha256": sha256(joblib_path),
        },
        "promotion_gates": gates,
        "fallback": (
            "Use the deterministic risk-only optimiser whenever the model "
            "artifact, required features, or risk estimates are unavailable."
        ),
        "llm_policy": (
            "DeepSeek may explain a fixed numeric decision but may not alter "
            "actions, target weights, or risk limits."
        ),
        "random_seed": RANDOM_SEED,
    }
    (CHECKPOINT / "decision_model.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    prediction_table.to_csv(REPORT / "prediction_metrics.csv", index=False)
    portfolio_table.to_csv(REPORT / "portfolio_metrics.csv", index=False)
    yearly_table.to_csv(REPORT / "yearly_metrics.csv", index=False)
    scale_table.sort_values("signal_scale").to_csv(
        REPORT / "signal_scale_validation.csv",
        index=False,
    )
    selected_oof.to_parquet(REPORT / "oof_predictions.parquet", index=False)
    external_predictions.to_parquet(
        REPORT / "external_predictions.parquet",
        index=False,
    )
    all_daily.to_parquet(REPORT / "backtest_daily.parquet", index=False)
    all_trades.to_parquet(REPORT / "backtest_trades.parquet", index=False)
    (REPORT / "promotion_gates.json").write_text(
        json.dumps(gates, indent=2),
        encoding="utf-8",
    )
    (REPORT / "report.md").write_text(
        _report_markdown(
            selected_name,
            selected_strategy,
            selected_scale,
            production_mode,
            prediction_table,
            portfolio_table,
            gates,
        ),
        encoding="utf-8",
    )
    print(
        f"\nselected={selected_name} | production_mode={production_mode} | "
        f"gates={sum(checks.values())}/{len(checks)}"
    )
    print(f"report -> {(REPORT / 'report.md').relative_to(ROOT)}")
    print(
        "checkpoint -> "
        f"{(CHECKPOINT / 'decision_model.json').relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
