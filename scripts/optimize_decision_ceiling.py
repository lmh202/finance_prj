"""Search the decision-layer ceiling with the formal HAR-X + News risk fixed.

Only 2018-2020 expanding walk-forward validation is used to choose:

* the return-ranking model;
* the cross-sectional alpha strength;
* the rebalance interval; and
* the turnover penalty multiplier.

The locked 2021-2023 period and the 2024-2026 external panel are evaluated
after the specification is frozen.  This script never overwrites the formal
decision checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
    moving_block_utility_gain,
    prediction_metrics,
    run_backtest,
    walk_forward_predictions,
)

HISTORICAL_PATH = ROOT / "data" / "processed" / "decision_dataset.parquet"
EXTERNAL_PATH = (
    ROOT / "data" / "processed" / "decision_external_dataset.parquet"
)
REPORT_DIR = ROOT / "reports" / "decision_layer_ceiling"
CANDIDATE_DIR = (
    ROOT / "data" / "processed" / "decision_model_candidate_ceiling"
)

VALIDATION_YEARS = (2018, 2019, 2020)
TEST_YEARS = (2021, 2022, 2023)
PRIMARY_COST_BPS = 25.0
EVALUATION_RISK_AVERSION = 6.0
OPTIMIZER_RISK_AVERSION = 6.0
ALPHA_STRENGTHS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05)
REBALANCE_GRID = (5, 10, 20)
TURNOVER_PENALTY_GRID = (0.5, 1.0, 2.0, 4.0)
COST_GRID = (10.0, 25.0, 50.0)

RANK_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "xgb_rank_d2",
        "kind": "xgboost",
        "target_column": "target_rank_20d",
        "target_offset": 0.5,
        "prediction_bound": 0.5,
        "max_depth": 2,
        "learning_rate": 0.02,
        "n_estimators": 700,
        "min_child_weight": 32,
        "subsample": 0.80,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 20.0,
    },
    {
        "name": "xgb_rank_d3",
        "kind": "xgboost",
        "target_column": "target_rank_20d",
        "target_offset": 0.5,
        "prediction_bound": 0.5,
        "max_depth": 3,
        "learning_rate": 0.02,
        "n_estimators": 700,
        "min_child_weight": 32,
        "subsample": 0.80,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 30.0,
    },
)
CEILING_SPECS = tuple(dict(spec) for spec in MODEL_SPECS) + RANK_MODEL_SPECS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def cross_sectional_signal(
    predictions: pd.DataFrame,
    alpha_strength: float,
) -> pd.DataFrame:
    """Convert any model output into the same zero-centred daily rank signal."""
    output = predictions.copy()
    if "raw_prediction" not in output:
        output["raw_prediction"] = output["prediction"]
    rank = output.groupby("date")["raw_prediction"].rank(method="average")
    count = output.groupby("date")["raw_prediction"].transform("size")
    score = np.where(
        count.to_numpy() > 1,
        2.0
        * (rank.to_numpy(dtype=float) - 1.0)
        / (count.to_numpy(dtype=float) - 1.0)
        - 1.0,
        0.0,
    )
    output["cross_sectional_score"] = score
    output["prediction"] = score * alpha_strength
    output["alpha_strength"] = alpha_strength
    return output


def evaluate(
    panel: pd.DataFrame,
    predictions: pd.DataFrame | None,
    *,
    period: str,
    strategy: str,
    start: str,
    end: str,
    cost_bps: float = PRIMARY_COST_BPS,
    rebalance_sessions: int = 5,
    turnover_penalty_multiplier: float = 1.0,
    scope: str = "all",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    daily, trades = run_backtest(
        panel,
        predictions,
        start=start,
        end=end,
        strategy=strategy,
        transaction_cost_bps=cost_bps,
        rebalance_sessions=rebalance_sessions,
        risk_aversion=OPTIMIZER_RISK_AVERSION,
        turnover_penalty_multiplier=turnover_penalty_multiplier,
    )
    metrics = backtest_metrics(
        daily,
        risk_aversion=EVALUATION_RISK_AVERSION,
    )
    metrics.update(
        {
            "period": period,
            "scope": scope,
            "strategy": strategy,
            "transaction_cost_bps": cost_bps,
            "rebalance_sessions": rebalance_sessions,
            "turnover_penalty_multiplier": turnover_penalty_multiplier,
        }
    )
    daily = daily.copy()
    daily["period"] = period
    daily["scope"] = scope
    daily["transaction_cost_bps"] = cost_bps
    daily["rebalance_sessions"] = rebalance_sessions
    daily["turnover_penalty_multiplier"] = turnover_penalty_multiplier
    trades = trades.copy()
    if not trades.empty:
        trades["period"] = period
        trades["scope"] = scope
        trades["transaction_cost_bps"] = cost_bps
        trades["rebalance_sessions"] = rebalance_sessions
        trades["turnover_penalty_multiplier"] = turnover_penalty_multiplier
    return metrics, daily, trades


def spec_lookup(name: str) -> dict[str, Any]:
    for spec in CEILING_SPECS:
        if str(spec["name"]) == name:
            return dict(spec)
    raise KeyError(name)


def metric_row(
    table: pd.DataFrame,
    period: str,
    strategy: str,
    scope: str = "all",
    cost_bps: float = PRIMARY_COST_BPS,
) -> pd.Series:
    selected = table.loc[
        table["period"].eq(period)
        & table["strategy"].eq(strategy)
        & table["scope"].eq(scope)
        & table["transaction_cost_bps"].eq(cost_bps)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one metric row for {period}/{strategy}/{scope}, "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    historical = pd.read_parquet(HISTORICAL_PATH)
    external = pd.read_parquet(EXTERNAL_PATH)
    for frame in (historical, external):
        frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()

    validation_period = historical["date"].dt.year.between(2018, 2020)
    test_period = historical["date"].dt.year.between(2021, 2023)
    if historical.loc[
        validation_period | test_period,
        "risk_sigma_daily_5d",
    ].isna().any():
        raise ValueError("formal HAR-X + News OOF risk is incomplete")

    print("=" * 88, flush=True)
    print("DECISION CEILING | MODEL + ALPHA SCREEN ON 2018-2020 ONLY", flush=True)
    print("=" * 88, flush=True)
    validation_predictions: dict[str, pd.DataFrame] = {}
    predictive_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    for spec in CEILING_SPECS:
        name = str(spec["name"])
        raw = walk_forward_predictions(
            historical,
            VALIDATION_YEARS,
            spec,
        )
        validation_predictions[name] = raw
        predictive = prediction_metrics(raw)
        predictive_rows.append(
            {"period": "validation", "model": name, **predictive}
        )
        print(
            f"{name}: rank IC={predictive['rank_ic_mean']:+.4f}",
            flush=True,
        )
        for strength in ALPHA_STRENGTHS:
            signal = cross_sectional_signal(raw, strength)
            metrics, _, _ = evaluate(
                historical,
                signal,
                period="validation_screen",
                strategy="ml",
                start="2018-01-01",
                end="2020-12-31",
            )
            screen_rows.append(
                {
                    "model": name,
                    "alpha_strength": strength,
                    **metrics,
                }
            )

    screen = pd.DataFrame(screen_rows)
    screen_nonzero = screen.loc[screen["alpha_strength"].gt(0)]
    first_choice = screen_nonzero.sort_values(
        ["certainty_equivalent", "sharpe", "max_drawdown"],
        ascending=False,
    ).iloc[0]
    selected_model = str(first_choice["model"])
    selected_raw_validation = validation_predictions[selected_model]
    print(
        f"\nmodel screen winner: {selected_model}\n",
        flush=True,
    )

    print("=" * 88, flush=True)
    print("DECISION CEILING | OPTIMIZER TUNING ON 2018-2020 ONLY", flush=True)
    print("=" * 88, flush=True)
    tuning_rows: list[dict[str, Any]] = []
    for strength in ALPHA_STRENGTHS:
        signal = cross_sectional_signal(
            selected_raw_validation,
            strength,
        )
        for rebalance in REBALANCE_GRID:
            for penalty in TURNOVER_PENALTY_GRID:
                metrics, _, _ = evaluate(
                    historical,
                    signal,
                    period="validation_tuning",
                    strategy="ml",
                    start="2018-01-01",
                    end="2020-12-31",
                    rebalance_sessions=rebalance,
                    turnover_penalty_multiplier=penalty,
                )
                tuning_rows.append(
                    {
                        "model": selected_model,
                        "alpha_strength": strength,
                        **metrics,
                    }
                )
    tuning = pd.DataFrame(tuning_rows)
    selected = tuning.sort_values(
        ["certainty_equivalent", "sharpe", "max_drawdown"],
        ascending=False,
    ).iloc[0]
    selected_strength = float(selected["alpha_strength"])
    selected_rebalance = int(selected["rebalance_sessions"])
    selected_penalty = float(selected["turnover_penalty_multiplier"])
    selected_spec = spec_lookup(selected_model)
    selected_strategy = f"{selected_model}_rank_signal"
    print(
        "locked on validation: "
        f"model={selected_model}, alpha={selected_strength:.4f}, "
        f"rebalance={selected_rebalance}, penalty={selected_penalty:.1f}",
        flush=True,
    )

    validation_signal = cross_sectional_signal(
        selected_raw_validation,
        selected_strength,
    )
    portfolio_rows: list[dict[str, Any]] = []
    daily_outputs: list[pd.DataFrame] = []
    trade_outputs: list[pd.DataFrame] = []
    for strategy in ("equal_weight", "risk_only", "ml"):
        predictions = validation_signal if strategy == "ml" else None
        display = selected_strategy if strategy == "ml" else strategy
        metrics, daily, trades = evaluate(
            historical,
            predictions,
            period="validation",
            strategy=strategy,
            start="2018-01-01",
            end="2020-12-31",
            rebalance_sessions=selected_rebalance,
            turnover_penalty_multiplier=selected_penalty,
        )
        metrics["strategy"] = display
        daily["strategy"] = display
        if not trades.empty:
            trades["strategy"] = display
        portfolio_rows.append(metrics)
        daily_outputs.append(daily)
        trade_outputs.append(trades)

    print("=" * 88, flush=True)
    print("FROZEN SPECIFICATION | REUSED 2021-2023 HOLDOUT DIAGNOSTIC", flush=True)
    print("=" * 88, flush=True)
    locked_raw = walk_forward_predictions(
        historical,
        TEST_YEARS,
        selected_spec,
    )
    locked_signal = cross_sectional_signal(locked_raw, selected_strength)
    predictive_rows.append(
        {
            "period": "locked_test_reused",
            "model": selected_model,
            **prediction_metrics(locked_raw),
        }
    )
    for cost in COST_GRID:
        for strategy in ("risk_only", "ml"):
            predictions = locked_signal if strategy == "ml" else None
            display = selected_strategy if strategy == "ml" else strategy
            metrics, daily, trades = evaluate(
                historical,
                predictions,
                period="locked_test_reused",
                strategy=strategy,
                start="2021-01-01",
                end="2023-12-31",
                cost_bps=cost,
                rebalance_sessions=selected_rebalance,
                turnover_penalty_multiplier=selected_penalty,
            )
            metrics["strategy"] = display
            daily["strategy"] = display
            if not trades.empty:
                trades["strategy"] = display
            portfolio_rows.append(metrics)
            daily_outputs.append(daily)
            trade_outputs.append(trades)

    target_column = str(selected_spec.get("target_column", TARGET))
    final_train = historical.dropna(
        subset=PRICE_FEATURES + [TARGET, target_column]
    )
    final_estimator = fit_estimator(final_train, selected_spec)
    model_path = CANDIDATE_DIR / "return_model.joblib"
    joblib.dump(final_estimator, model_path)
    if selected_spec["kind"] == "xgboost":
        final_estimator.save_model(CANDIDATE_DIR / "return_model.xgb.json")

    external_raw = external[["date", "symbol", TARGET]].copy()
    external_raw["prediction"] = bounded_prediction(
        final_estimator,
        external,
        bound=float(selected_spec.get("prediction_bound", 0.25)),
    )
    external_signal = cross_sectional_signal(
        external_raw,
        selected_strength,
    )
    predictive_rows.append(
        {
            "period": "external_reused_price_risk_only",
            "model": selected_model,
            **prediction_metrics(external_raw),
        }
    )
    print("=" * 88, flush=True)
    print("FROZEN SPECIFICATION | REUSED 2024-2026 EXTERNAL DIAGNOSTIC", flush=True)
    print("=" * 88, flush=True)
    for cost in COST_GRID:
        for strategy in ("risk_only", "ml"):
            predictions = external_signal if strategy == "ml" else None
            display = selected_strategy if strategy == "ml" else strategy
            metrics, daily, trades = evaluate(
                external,
                predictions,
                period="external_reused",
                strategy=strategy,
                start="2024-01-01",
                end="2026-12-31",
                cost_bps=cost,
                rebalance_sessions=selected_rebalance,
                turnover_penalty_multiplier=selected_penalty,
            )
            metrics["strategy"] = display
            daily["strategy"] = display
            if not trades.empty:
                trades["strategy"] = display
            portfolio_rows.append(metrics)
            daily_outputs.append(daily)
            trade_outputs.append(trades)

    scopes = {
        "original_research": external["groups"].str.contains(
            "original_research",
            na=False,
        ),
        "unseen": ~external["groups"].str.contains(
            "original_research",
            na=False,
        ),
    }
    for scope, mask in scopes.items():
        scoped_panel = external.loc[mask].copy()
        symbols = set(scoped_panel["symbol"])
        scoped_signal = external_signal.loc[
            external_signal["symbol"].isin(symbols)
        ]
        for strategy in ("risk_only", "ml"):
            predictions = scoped_signal if strategy == "ml" else None
            display = selected_strategy if strategy == "ml" else strategy
            metrics, daily, trades = evaluate(
                scoped_panel,
                predictions,
                period="external_reused",
                scope=scope,
                strategy=strategy,
                start="2024-01-01",
                end="2026-12-31",
                rebalance_sessions=selected_rebalance,
                turnover_penalty_multiplier=selected_penalty,
            )
            metrics["strategy"] = display
            daily["strategy"] = display
            if not trades.empty:
                trades["strategy"] = display
            portfolio_rows.append(metrics)
            daily_outputs.append(daily)
            trade_outputs.append(trades)

    portfolio = pd.DataFrame(portfolio_rows)
    prediction = pd.DataFrame(predictive_rows)
    all_daily = pd.concat(daily_outputs, ignore_index=True)
    nonempty_trades = [frame for frame in trade_outputs if not frame.empty]
    all_trades = (
        pd.concat(nonempty_trades, ignore_index=True)
        if nonempty_trades
        else pd.DataFrame()
    )

    test_daily = all_daily.loc[
        all_daily["period"].eq("locked_test_reused")
        & all_daily["scope"].eq("all")
        & all_daily["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ]
    risk_daily = test_daily.loc[test_daily["strategy"].eq("risk_only")]
    candidate_daily = test_daily.loc[
        test_daily["strategy"].eq(selected_strategy)
    ]
    bootstrap = moving_block_utility_gain(risk_daily, candidate_daily)

    yearly_rows = []
    for strategy, frame in (
        ("risk_only", risk_daily),
        (selected_strategy, candidate_daily),
    ):
        for year, group in frame.groupby(frame["date"].dt.year):
            yearly_rows.append(
                {
                    "year": int(year),
                    "strategy": strategy,
                    **backtest_metrics(
                        group,
                        risk_aversion=EVALUATION_RISK_AVERSION,
                    ),
                }
            )
    yearly = pd.DataFrame(yearly_rows)
    pivot = yearly.pivot(
        index="year",
        columns="strategy",
        values="certainty_equivalent",
    )
    positive_years = int(
        (pivot[selected_strategy] - pivot["risk_only"]).gt(0).sum()
    )

    test_candidate = metric_row(
        portfolio,
        "locked_test_reused",
        selected_strategy,
    )
    test_risk = metric_row(portfolio, "locked_test_reused", "risk_only")
    external_candidate = metric_row(
        portfolio,
        "external_reused",
        selected_strategy,
    )
    external_risk = metric_row(portfolio, "external_reused", "risk_only")
    unseen_candidate = metric_row(
        portfolio,
        "external_reused",
        selected_strategy,
        scope="unseen",
    )
    unseen_risk = metric_row(
        portfolio,
        "external_reused",
        "risk_only",
        scope="unseen",
    )
    checks = {
        "validation_cer_above_risk_only": (
            float(selected["certainty_equivalent"])
            > float(
                metric_row(
                    portfolio,
                    "validation",
                    "risk_only",
                )["certainty_equivalent"]
            )
        ),
        "locked_test_cer_above_risk_only": (
            test_candidate["certainty_equivalent"]
            > test_risk["certainty_equivalent"]
        ),
        "locked_test_sharpe_above_risk_only": (
            test_candidate["sharpe"] > test_risk["sharpe"]
        ),
        "positive_test_years_at_least_2_of_3": positive_years >= 2,
        "bootstrap_lower_bound_positive": bootstrap[0] > 0,
        "locked_test_drawdown_not_worse_by_2pp": (
            test_candidate["max_drawdown"]
            >= test_risk["max_drawdown"] - 0.02
        ),
        "external_cer_not_below_risk_only": (
            external_candidate["certainty_equivalent"]
            >= external_risk["certainty_equivalent"]
        ),
        "external_unseen_cer_not_below_risk_only": (
            unseen_candidate["certainty_equivalent"]
            >= unseen_risk["certainty_equivalent"]
        ),
        "max_position_constraint": (
            test_candidate["maximum_weight"] <= 0.2001
        ),
        "max_change_constraint": (
            test_candidate["maximum_change"] <= 0.0501
        ),
        "minimum_trade_constraint": (
            test_candidate["minimum_active_trade"] == 0
            or test_candidate["minimum_active_trade"] >= 0.0099
        ),
    }

    diagnostics = {
        "candidate_passes_diagnostic_gates": all(checks.values()),
        "automatic_promotion_allowed": False,
        "reason_automatic_promotion_disabled": (
            "The 2021-2026 holdouts were observed by earlier experiments and "
            "are no longer pristine blind tests."
        ),
        "selected_model": selected_model,
        "selected_spec": selected_spec,
        "signal_transform": "daily_cross_sectional_rank_to_minus1_plus1",
        "alpha_strength": selected_strength,
        "rebalance_sessions": selected_rebalance,
        "turnover_penalty_multiplier": selected_penalty,
        "fixed_risk_input": (
            "formal HAR-X + News five-session OOF sigma for 2018-2023"
        ),
        "checks": {key: bool(value) for key, value in checks.items()},
        "statistics": {
            "bootstrap_utility_gain_95": list(bootstrap),
            "positive_test_years": positive_years,
            "locked_test_cer_gain": float(
                test_candidate["certainty_equivalent"]
                - test_risk["certainty_equivalent"]
            ),
            "locked_test_sharpe_gain": float(
                test_candidate["sharpe"] - test_risk["sharpe"]
            ),
            "external_cer_gain": float(
                external_candidate["certainty_equivalent"]
                - external_risk["certainty_equivalent"]
            ),
            "external_unseen_cer_gain": float(
                unseen_candidate["certainty_equivalent"]
                - unseen_risk["certainty_equivalent"]
            ),
        },
        "known_limitations": [
            (
                "The 2024-2026 external risk panel is price-only because no "
                "historical RSS archive exists; it does not test the full "
                "live news path."
            ),
            (
                "The optimizer is fully invested and does not yet enforce "
                "sector, commodity, or cash constraints."
            ),
        ],
    }

    metadata = {
        "schema_version": 1,
        "model_version": "decision-ceiling-candidate-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_production_checkpoint_replaced": False,
        "selected_model": selected_model,
        "selected_spec": selected_spec,
        "feature_order": PRICE_FEATURES,
        "target": TARGET,
        "target_definition": "20-session stock return minus SPY return",
        "signal_transform": diagnostics["signal_transform"],
        "alpha_strength": selected_strength,
        "rebalance_sessions": selected_rebalance,
        "optimizer_risk_aversion": OPTIMIZER_RISK_AVERSION,
        "turnover_penalty_multiplier": selected_penalty,
        "primary_transaction_cost_bps": PRIMARY_COST_BPS,
        "fixed_risk_input": diagnostics["fixed_risk_input"],
        "validation_years": list(VALIDATION_YEARS),
        "reused_test_years": list(TEST_YEARS),
        "artifacts": {
            "model": model_path.name,
            "model_sha256": sha256(model_path),
            "historical_panel_sha256": sha256(HISTORICAL_PATH),
            "external_panel_sha256": sha256(EXTERNAL_PATH),
        },
        "diagnostics": diagnostics,
        "random_seed": RANDOM_SEED,
    }
    (CANDIDATE_DIR / "decision_model.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    screen.to_csv(REPORT_DIR / "model_alpha_screen.csv", index=False)
    tuning.to_csv(REPORT_DIR / "optimizer_tuning.csv", index=False)
    prediction.to_csv(REPORT_DIR / "prediction_metrics.csv", index=False)
    portfolio.to_csv(REPORT_DIR / "portfolio_metrics.csv", index=False)
    yearly.to_csv(REPORT_DIR / "yearly_metrics.csv", index=False)
    validation_signal.to_parquet(
        REPORT_DIR / "validation_predictions.parquet",
        index=False,
    )
    locked_signal.to_parquet(
        REPORT_DIR / "locked_test_predictions.parquet",
        index=False,
    )
    external_signal.to_parquet(
        REPORT_DIR / "external_predictions.parquet",
        index=False,
    )
    all_daily.to_parquet(REPORT_DIR / "backtest_daily.parquet", index=False)
    if not all_trades.empty:
        all_trades.to_parquet(
            REPORT_DIR / "backtest_trades.parquet",
            index=False,
        )
    (REPORT_DIR / "diagnostic_gates.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )

    primary = portfolio.loc[
        portfolio["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
        & portfolio["scope"].eq("all")
    ][
        [
            "period",
            "strategy",
            "certainty_equivalent",
            "sharpe",
            "max_drawdown",
            "annual_turnover",
        ]
    ].round(4)
    predictive_view = prediction[
        ["period", "model", "rank_ic_mean", "top_bottom_20d_spread"]
    ].round(4)
    passed = sum(checks.values())
    report = f"""# Fixed-Risk Decision-Layer Ceiling

## Locked specification

- Risk input: formal HAR-X + News five-session OOF risk
- Return model: **{selected_model}**
- Signal: daily cross-sectional rank, amplitude **{selected_strength:.4f}**
- Rebalance interval: **{selected_rebalance} sessions**
- Turnover-penalty multiplier: **{selected_penalty:.1f}**
- Diagnostic gates passed: **{passed}/{len(checks)}**
- Automatic promotion: **disabled**

All model and optimiser choices were made on the 2018-2020 expanding
walk-forward validation period.  The 2021-2023 and 2024-2026 results were
computed only after the specification was locked.  They are labelled reused
diagnostics because prior experiments had already exposed those periods.

## Predictive ranking

{markdown_table(predictive_view)}

## Portfolio results at 25 bps

{markdown_table(primary)}

## Diagnostic gates

```json
{json.dumps(diagnostics, indent=2)}
```

## Interpretation

This experiment estimates the practical ceiling of the current decision
architecture with the risk engine frozen.  A positive result does not authorize
deployment without new prospective data, because both historical holdouts have
already been observed.  The 2024-2026 panel also lacks historical RSS and
therefore tests price-risk transfer rather than the complete live news path.
"""
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(
        f"\nselected={selected_model} alpha={selected_strength:.4f} "
        f"rebalance={selected_rebalance} penalty={selected_penalty:.1f}",
        flush=True,
    )
    print(
        f"diagnostic gates={passed}/{len(checks)} | "
        "automatic promotion disabled",
        flush=True,
    )
    print(f"report -> {REPORT_DIR.relative_to(ROOT) / 'report.md'}")
    print(
        "candidate -> "
        f"{CANDIDATE_DIR.relative_to(ROOT) / 'decision_model.json'}"
    )


if __name__ == "__main__":
    main()
