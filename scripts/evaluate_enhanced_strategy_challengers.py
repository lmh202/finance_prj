"""Evaluate validation-defined conservative residual-alpha challengers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from train_enhanced_daily_strategy import (
    DEPLOYABLE_FEATURES,
    EXTERNAL_PATH,
    HISTORICAL_PATH,
    BlendConfig,
    XGBSpec,
    apply_cost,
    engineer_panel,
    make_specs,
    metric_row,
    paired_tests,
    refit_predict_period,
    run_portfolio,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "daily_strategy_enhanced"
PRIMARY_COST_BPS = 25.0


def main() -> None:
    historical = engineer_panel(HISTORICAL_PATH, external=False)
    external = engineer_panel(EXTERNAL_PATH, external=True)
    specs: dict[str, XGBSpec] = {spec.name: spec for spec in make_specs()}
    spec = specs["xgb_13"]
    blends = {
        "conservative": BlendConfig(0.05, 0.10),
        "validation_ceiling": BlendConfig(0.50, 0.25),
    }

    development = historical.loc[
        historical["date"].dt.year.le(2020)
    ]
    diagnostic = historical.loc[
        historical["date"].dt.year.between(2021, 2023)
    ]
    _, diagnostic_prediction, _ = refit_predict_period(
        development,
        diagnostic,
        spec,
        DEPLOYABLE_FEATURES,
    )
    _, external_prediction, _ = refit_predict_period(
        historical,
        external,
        spec,
        DEPLOYABLE_FEATURES,
    )

    periods = {
        "diagnostic": {
            "panel": historical,
            "prediction": diagnostic_prediction,
            "start": "2021-01-01",
            "end": "2023-12-31",
            "groups": {"all": None},
        },
        "external": {
            "panel": external,
            "prediction": external_prediction,
            "start": "2024-01-01",
            "end": "2026-12-31",
            "groups": {
                "all": None,
                "seen": sorted(
                    external.loc[
                        external["is_seen_symbol"], "symbol"
                    ].unique()
                ),
                "unseen": sorted(
                    external.loc[
                        ~external["is_seen_symbol"], "symbol"
                    ].unique()
                ),
            },
        },
    }
    metric_rows = []
    tests = {}
    for period_name, period in periods.items():
        for group_name, symbols in period["groups"].items():
            baseline = apply_cost(
                run_portfolio(
                    period["panel"],
                    None,
                    None,
                    start=period["start"],
                    end=period["end"],
                    strategy_name="rule",
                    symbol_filter=symbols,
                ),
                PRIMARY_COST_BPS,
            )
            reference_metrics = metric_row(baseline)
            metric_rows.append(
                {
                    "period": period_name,
                    "group": group_name,
                    "strategy": "rule",
                    **reference_metrics,
                    "cer_gain_vs_rule": 0.0,
                    "sharpe_gain_vs_rule": 0.0,
                }
            )
            for blend_name, blend in blends.items():
                candidate = apply_cost(
                    run_portfolio(
                        period["panel"],
                        period["prediction"],
                        blend,
                        start=period["start"],
                        end=period["end"],
                        strategy_name=blend_name,
                        symbol_filter=symbols,
                    ),
                    PRIMARY_COST_BPS,
                )
                metrics = metric_row(candidate)
                metric_rows.append(
                    {
                        "period": period_name,
                        "group": group_name,
                        "strategy": blend_name,
                        **metrics,
                        "cer_gain_vs_rule": (
                            metrics["certainty_equivalent"]
                            - reference_metrics["certainty_equivalent"]
                        ),
                        "sharpe_gain_vs_rule": (
                            metrics["sharpe"]
                            - reference_metrics["sharpe"]
                        ),
                    }
                )
                tests[
                    f"{period_name}_{group_name}_{blend_name}_vs_rule"
                ] = paired_tests(baseline, candidate)

    pd.DataFrame(metric_rows).to_csv(
        OUT_DIR / "challenger_metrics_25bps.csv",
        index=False,
    )
    (OUT_DIR / "challenger_statistical_tests.json").write_text(
        json.dumps(tests, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
