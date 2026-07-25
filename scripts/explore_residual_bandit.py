"""Explore a residual contextual-bandit policy for AURORA decisions.

This is deliberately not presented as online RL. Historical returns reveal the
counterfactual reward of increase/hold/reduce for every asset, so a
full-information contextual bandit is more data-efficient and auditable.

Daily Strategy remains the prior action. A Ridge or MLP Q-model estimates
whether news, HAR-X risk, Health, and market state justify a residual change.
The final action is projected through the same cash/no-leverage safety layer as
the rule-fusion backtest.

Time protocol
-------------
train       2018-2019
validation  2020 (model/configuration selection)
diagnostic  2021-2023 (already observed in previous experiments; not blind)
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

import backtest_rule_fusion as rule_backtest  # noqa: E402
from decision_layer_core import backtest_metrics  # noqa: E402

OUT_DIR = ROOT / "reports" / "decision_layer_bandit"
TRAIN_START, TRAIN_END = "2018-01-01", "2019-12-31"
VALIDATION_START, VALIDATION_END = "2020-01-01", "2020-12-31"
DIAGNOSTIC_START, DIAGNOSTIC_END = "2021-01-01", "2023-12-31"

ACTIONS = np.array([-1.0, 0.0, 1.0])
REWARD_POSITION_SIZE = 0.03
REWARD_RISK_PRICE = 0.20
REWARD_COST_BPS = 25.0
REWARD_SCALE = 10_000.0
PRIMARY_COST_BPS = 25.0
REBALANCE_SESSIONS = 5
INITIAL_CASH_WEIGHT = 0.05
SEED = 20260724

CORE_FEATURES = [
    "strategy_score",
    "mom_20d",
    "mom_60d",
    "strategy_trend",
    "strategy_sharpe",
    "strategy_drawdown",
    "vol_20d",
    "rsi_scaled",
    "risk_level_scaled",
    "log_risk_sigma",
    "health_normalized",
    "benchmark_ret_1d",
]
NEWS_FEATURES = [
    "has_recent_news",
    "news_score_5d",
    "log_news_count_5d",
    "sent_mean",
    "sent_std",
    "sent_min",
    "sent_max",
    "sent_abs_mean",
    "sent_positive_share",
    "sent_negative_share",
    "sent_extreme_share",
    "sent_surprise20",
    "sent_abs_surprise20",
    "count_z20",
    "title_token_novelty",
    "duplicate_share",
    "firm_specific_share",
    "broad_story_share",
    "event_earnings_share",
    "event_analyst_share",
    "event_corporate_action_share",
    "event_legal_regulatory_share",
    "event_product_share",
    "event_macro_share",
    "event_management_share",
    "event_financing_share",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    alpha: float
    hidden: tuple[int, ...] = ()


@dataclass(frozen=True)
class PolicyConfig:
    residual_weight: float
    q_margin: float
    maximum_change: float


MODEL_SPECS = (
    ModelSpec("ridge_a1", "ridge", 1.0),
    ModelSpec("ridge_a10", "ridge", 10.0),
    ModelSpec("ridge_a100", "ridge", 100.0),
    ModelSpec("mlp_32x16_a001", "mlp", 0.001, (32, 16)),
    ModelSpec("mlp_64x32_a001", "mlp", 0.001, (64, 32)),
    ModelSpec("mlp_64x32_a01", "mlp", 0.01, (64, 32)),
)
POLICY_CONFIGS = tuple(
    PolicyConfig(residual_weight, margin, size)
    for residual_weight in (0.25, 0.50, 0.75)
    for margin in (0.0, 0.5)
    for size in (0.02, 0.04)
)


def prepare_panel() -> pd.DataFrame:
    panel = rule_backtest.load_and_engineer_panel()
    benchmark = (
        panel[["date", "benchmark_ret_1d"]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")["benchmark_ret_1d"]
    )
    benchmark_growth = pd.Series(1.0, index=benchmark.index)
    for offset in range(1, 6):
        benchmark_growth *= 1.0 + benchmark.shift(-offset)
    benchmark_forward = benchmark_growth - 1.0
    panel["benchmark_fwd_ret_5d"] = panel["date"].map(benchmark_forward)
    panel["excess_ret_5d"] = (
        panel["fwd_ret_5d"] - panel["benchmark_fwd_ret_5d"]
    )

    health = rule_backtest.equal_weight_health_by_date(panel)
    panel["health_normalized"] = (
        panel["date"].map(health).fillna(50.0) - 50.0
    ) / 50.0
    panel["rsi_scaled"] = (panel["rsi_14"] - 50.0) / 50.0
    panel["risk_level_scaled"] = panel["risk_level_5d"] / 100.0
    panel["log_risk_sigma"] = np.log(
        panel["risk_sigma_daily_5d"].clip(lower=1e-6)
    )
    panel["has_recent_news"] = panel["news_unique_5d"].gt(0).astype(float)
    panel["log_news_count_5d"] = np.log1p(panel["news_unique_5d"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    required = list(
        dict.fromkeys(
            CORE_FEATURES
            + NEWS_FEATURES
            + [
                "excess_ret_5d",
                "risk_sigma_daily_5d",
                "ret_1d",
            ]
        )
    )
    panel[required] = panel[required].fillna(0.0)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def feature_columns(news_enabled: bool) -> list[str]:
    return CORE_FEATURES + (NEWS_FEATURES if news_enabled else [])


def fit_bounds(frame: pd.DataFrame, columns: Sequence[str]) -> dict:
    bounds = {}
    for column in columns:
        values = frame[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            bounds[column] = (-1.0, 1.0)
        else:
            lower, upper = values.quantile([0.01, 0.99])
            if not np.isfinite(lower) or not np.isfinite(upper):
                lower, upper = -1.0, 1.0
            if upper <= lower:
                upper = lower + 1e-6
            bounds[column] = (float(lower), float(upper))
    return bounds


def state_matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    bounds: dict,
) -> np.ndarray:
    values = frame[list(columns)].copy()
    for column in columns:
        lower, upper = bounds[column]
        values[column] = (
            values[column]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(lower, upper)
        )
    return values.to_numpy(dtype=float)


def state_action_matrix(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=float).reshape(-1, 1)
    return np.column_stack([states, actions, states * actions])


def counterfactual_training_data(
    frame: pd.DataFrame,
    columns: Sequence[str],
    bounds: dict,
) -> tuple[np.ndarray, np.ndarray]:
    states = state_matrix(frame, columns, bounds)
    repeated_states = np.repeat(states, len(ACTIONS), axis=0)
    actions = np.tile(ACTIONS, len(frame))
    delta = REWARD_POSITION_SIZE * actions
    excess = np.repeat(frame["excess_ret_5d"].to_numpy(dtype=float), len(ACTIONS))
    sigma_h = np.repeat(
        frame["risk_sigma_daily_5d"].to_numpy(dtype=float) * math.sqrt(5),
        len(ACTIONS),
    )
    reward = (
        delta * excess
        - REWARD_RISK_PRICE * np.abs(delta) * sigma_h
        - REWARD_COST_BPS / 10_000.0 * np.abs(delta)
    )
    return (
        state_action_matrix(repeated_states, actions),
        reward * REWARD_SCALE,
    )


def make_model(spec: ModelSpec) -> Pipeline:
    if spec.kind == "ridge":
        estimator = Ridge(alpha=spec.alpha)
    elif spec.kind == "mlp":
        estimator = MLPRegressor(
            hidden_layer_sizes=spec.hidden,
            activation="relu",
            solver="adam",
            alpha=spec.alpha,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=SEED,
        )
    else:
        raise ValueError(spec.kind)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def fit_q_model(
    frame: pd.DataFrame,
    spec: ModelSpec,
    *,
    news_enabled: bool,
) -> tuple[Pipeline, list[str], dict]:
    columns = feature_columns(news_enabled)
    bounds = fit_bounds(frame, columns)
    x_train, y_train = counterfactual_training_data(frame, columns, bounds)
    model = make_model(spec)
    model.fit(x_train, y_train)
    return model, columns, bounds


def predict_q(
    model: Pipeline,
    frame: pd.DataFrame,
    columns: Sequence[str],
    bounds: dict,
) -> np.ndarray:
    states = state_matrix(frame, columns, bounds)
    predictions = []
    for action in ACTIONS:
        actions = np.full(len(states), action)
        predictions.append(
            model.predict(state_action_matrix(states, actions))
        )
    return np.column_stack(predictions)


def requested_policy_changes(
    model: Pipeline,
    rows: pd.DataFrame,
    columns: Sequence[str],
    bounds: dict,
    config: PolicyConfig,
    *,
    health_score: float,
    zero_news: bool = False,
) -> tuple[np.ndarray, dict]:
    states = rows.copy()
    states["health_normalized"] = (health_score - 50.0) / 50.0
    if zero_news:
        for column in NEWS_FEATURES:
            if column in states:
                states[column] = 0.0
    q_values = predict_q(model, states, columns, bounds)
    best_index = np.argmax(q_values, axis=1)
    learned_action = ACTIONS[best_index]
    hold_q = q_values[:, 1]
    best_q = q_values[np.arange(len(q_values)), best_index]
    insufficient = (best_q - hold_q) < config.q_margin
    learned_action = np.where(insufficient, 0.0, learned_action)

    strategy_prior = states["strategy_score"].to_numpy(dtype=float).clip(-1, 1)
    combined = (
        (1.0 - config.residual_weight) * strategy_prior
        + config.residual_weight * learned_action
    ).clip(-1, 1)
    conservatism = (
        1.0
        - 0.45
        * states["risk_level_5d"].to_numpy(dtype=float).clip(0, 100)
        / 100.0
    )
    requested = config.maximum_change * combined * conservatism
    requested[np.abs(requested) < rule_backtest.MIN_TRADE] = 0.0
    diagnostics = {
        "learned_increase_share": float(np.mean(learned_action > 0)),
        "learned_reduce_share": float(np.mean(learned_action < 0)),
        "learned_hold_share": float(np.mean(learned_action == 0)),
        "mean_abs_requested_change": float(np.mean(np.abs(requested))),
        "mean_q_margin": float(np.mean(best_q - hold_q)),
    }
    return requested, diagnostics


def eligible_symbols(
    panel: pd.DataFrame,
    start: str,
    end: str,
) -> list[str]:
    return rule_backtest.eligible_symbols(panel, start, end)


def run_policy(
    panel: pd.DataFrame,
    model: Pipeline,
    columns: Sequence[str],
    bounds: dict,
    config: PolicyConfig,
    *,
    start: str,
    end: str,
    period: str,
    strategy_name: str,
    zero_news: bool = False,
) -> pd.DataFrame:
    symbols = eligible_symbols(panel, start, end)
    filtered = panel.loc[panel["symbol"].isin(symbols)].copy()
    returns = filtered.pivot(index="date", columns="symbol", values="ret_1d")
    returns = returns.reindex(columns=symbols).sort_index()
    benchmark = (
        filtered.groupby("date")["benchmark_ret_1d"].first().sort_index()
    )
    rows_by_date = {
        date: group.set_index("symbol").reindex(symbols).reset_index()
        for date, group in filtered.groupby("date")
    }

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    dates = returns.index[returns.index.to_series().between(start_ts, end_ts)]
    asset_weights = np.full(
        len(symbols), (1.0 - INITIAL_CASH_WEIGHT) / len(symbols)
    )
    cash_weight = INITIAL_CASH_WEIGHT
    rebalance_set = set(dates[::REBALANCE_SESSIONS])
    output = []

    for date in dates:
        daily_asset_return = (
            returns.loc[date].reindex(symbols).fillna(0.0).to_numpy(dtype=float)
        )
        gross_return = float(asset_weights @ daily_asset_return)
        growth = max(1.0 + gross_return, 1e-12)
        asset_weights = asset_weights * (1.0 + daily_asset_return) / growth
        cash_weight /= growth
        total = float(asset_weights.sum() + cash_weight)
        asset_weights /= max(total, 1e-12)
        cash_weight /= max(total, 1e-12)

        turnover = 0.0
        maximum_change = 0.0
        diagnostics = {
            "learned_increase_share": 0.0,
            "learned_reduce_share": 0.0,
            "learned_hold_share": 1.0,
            "mean_abs_requested_change": 0.0,
            "mean_q_margin": 0.0,
        }
        health_score = rule_backtest.portfolio_health_score(
            returns.loc[:date, symbols],
            asset_weights,
        )
        if date in rebalance_set:
            requested, diagnostics = requested_policy_changes(
                model,
                rows_by_date[date],
                columns,
                bounds,
                config,
                health_score=health_score,
                zero_news=zero_news,
            )
            (
                asset_weights,
                cash_weight,
                turnover,
                maximum_change,
            ) = rule_backtest.apply_position_changes(
                asset_weights,
                cash_weight,
                requested,
            )
        output.append(
            {
                "date": date,
                "period": period,
                "strategy": strategy_name,
                "gross_return": gross_return,
                "benchmark_return": float(benchmark.loc[date]),
                "turnover": turnover,
                "cash_weight": cash_weight,
                "maximum_weight": float(asset_weights.max()),
                "maximum_change": maximum_change,
                "health_score": health_score,
                "n_positions": int(np.sum(asset_weights > 1e-4)),
                **diagnostics,
            }
        )
    return pd.DataFrame(output)


def apply_cost(frame: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    result = frame.copy()
    result["transaction_cost_bps"] = cost_bps
    result["transaction_cost"] = (
        result["turnover"] * cost_bps / 10_000.0
    )
    result["net_return"] = (
        result["gross_return"] - result["transaction_cost"]
    )
    result["optimizer_success"] = True
    result["max_weight"] = result["maximum_weight"]
    result["max_change"] = result["maximum_change"]
    result["minimum_active_trade"] = 0.0
    return result


def metric_row(frame: pd.DataFrame) -> dict:
    return {
        **backtest_metrics(frame),
        "average_cash_weight": float(frame["cash_weight"].mean()),
        "average_health_score": float(frame["health_score"].mean()),
        "learned_increase_share": float(
            frame.loc[frame["turnover"].gt(0), "learned_increase_share"].mean()
        ),
        "learned_reduce_share": float(
            frame.loc[frame["turnover"].gt(0), "learned_reduce_share"].mean()
        ),
        "learned_hold_share": float(
            frame.loc[frame["turnover"].gt(0), "learned_hold_share"].mean()
        ),
    }


def search_models(
    panel: pd.DataFrame,
    *,
    news_enabled: bool,
) -> tuple[pd.DataFrame, dict]:
    train = panel.loc[panel["date"].between(TRAIN_START, TRAIN_END)]
    rows = []
    fitted = {}
    for spec in MODEL_SPECS:
        print(
            f"[fit] {'full' if news_enabled else 'no_news':7s} {spec.name}"
        )
        model, columns, bounds = fit_q_model(
            train,
            spec,
            news_enabled=news_enabled,
        )
        fitted[spec.name] = (model, columns, bounds, spec)
        for config in POLICY_CONFIGS:
            daily = run_policy(
                panel,
                model,
                columns,
                bounds,
                config,
                start=VALIDATION_START,
                end=VALIDATION_END,
                period="validation_2020",
                strategy_name=spec.name,
            )
            metrics = metric_row(apply_cost(daily, PRIMARY_COST_BPS))
            rows.append(
                {
                    "news_enabled": news_enabled,
                    "model": spec.name,
                    "kind": spec.kind,
                    "alpha": spec.alpha,
                    "hidden": "x".join(map(str, spec.hidden)),
                    "residual_weight": config.residual_weight,
                    "q_margin": config.q_margin,
                    "policy_maximum_change": config.maximum_change,
                    **metrics,
                }
            )
    table = pd.DataFrame(rows).sort_values(
        ["certainty_equivalent", "sharpe"],
        ascending=False,
    ).reset_index(drop=True)
    winner = table.iloc[0]
    selected = {
        "row": winner.to_dict(),
        "spec": fitted[str(winner["model"])][3],
        "config": PolicyConfig(
            residual_weight=float(winner["residual_weight"]),
            q_margin=float(winner["q_margin"]),
            maximum_change=float(winner["policy_maximum_change"]),
        ),
    }
    return table, selected


def refit_selected(
    panel: pd.DataFrame,
    selected: dict,
    *,
    news_enabled: bool,
) -> tuple[Pipeline, list[str], dict]:
    development = panel.loc[
        panel["date"].between(TRAIN_START, VALIDATION_END)
    ]
    return fit_q_model(
        development,
        selected["spec"],
        news_enabled=news_enabled,
    )


def baseline_daily(
    panel: pd.DataFrame,
    variant: str,
    *,
    start: str,
    end: str,
    period: str,
) -> pd.DataFrame:
    frame = rule_backtest.run_variant(
        panel,
        period=period,
        start=start,
        end=end,
        variant=variant,
    )
    return apply_cost(frame, PRIMARY_COST_BPS)


def paired_tests(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict:
    left = reference.sort_values("date").set_index("date")["net_return"]
    right = candidate.sort_values("date").set_index("date")["net_return"]
    common = left.index.intersection(right.index)
    difference = right.reindex(common).to_numpy() - left.reindex(common).to_numpy()
    return {
        **rule_backtest.newey_west_mean_test(difference),
        **rule_backtest.block_bootstrap_cer_gain(
            left.reindex(common).to_numpy(),
            right.reindex(common).to_numpy(),
            seed=SEED,
        ),
    }


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    return rule_backtest.markdown_table(frame, digits=digits)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    full_search, full_selected = search_models(panel, news_enabled=True)
    no_news_search, no_news_selected = search_models(
        panel,
        news_enabled=False,
    )
    full_search.to_csv(OUT_DIR / "validation_search_full.csv", index=False)
    no_news_search.to_csv(
        OUT_DIR / "validation_search_no_news.csv",
        index=False,
    )

    full_model, full_columns, full_bounds = refit_selected(
        panel,
        full_selected,
        news_enabled=True,
    )
    no_news_model, no_news_columns, no_news_bounds = refit_selected(
        panel,
        no_news_selected,
        news_enabled=False,
    )
    full_config = full_selected["config"]
    no_news_config = no_news_selected["config"]

    daily_paths = []
    for cost in (0.0, 25.0, 50.0):
        full_base = run_policy(
            panel,
            full_model,
            full_columns,
            full_bounds,
            full_config,
            start=DIAGNOSTIC_START,
            end=DIAGNOSTIC_END,
            period="diagnostic_2021_2023",
            strategy_name="bandit_full",
        )
        zeroed_base = run_policy(
            panel,
            full_model,
            full_columns,
            full_bounds,
            full_config,
            start=DIAGNOSTIC_START,
            end=DIAGNOSTIC_END,
            period="diagnostic_2021_2023",
            strategy_name="bandit_full_news_zeroed",
            zero_news=True,
        )
        no_news_base = run_policy(
            panel,
            no_news_model,
            no_news_columns,
            no_news_bounds,
            no_news_config,
            start=DIAGNOSTIC_START,
            end=DIAGNOSTIC_END,
            period="diagnostic_2021_2023",
            strategy_name="bandit_no_news",
        )
        daily_paths.extend(
            [
                apply_cost(full_base, cost),
                apply_cost(zeroed_base, cost),
                apply_cost(no_news_base, cost),
            ]
        )

    # Baselines need only one portfolio path; costs are applied afterwards.
    for variant in ("equal_weight", "strategy_only", "strategy_risk"):
        base = rule_backtest.run_variant(
            panel,
            period="diagnostic_2021_2023",
            start=DIAGNOSTIC_START,
            end=DIAGNOSTIC_END,
            variant=variant,
        )
        for cost in (0.0, 25.0, 50.0):
            daily_paths.append(apply_cost(base, cost))

    daily = pd.concat(daily_paths, ignore_index=True)
    metric_rows = []
    for (period, strategy, cost), frame in daily.groupby(
        ["period", "strategy", "transaction_cost_bps"]
    ):
        metric_rows.append(
            {
                "period": period,
                "strategy": strategy,
                "transaction_cost_bps": cost,
                **metric_row(frame.sort_values("date")),
            }
        )
    metrics = pd.DataFrame(metric_rows)

    primary = daily.loc[
        daily["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ]
    full_daily = primary.loc[primary["strategy"].eq("bandit_full")]
    zeroed_daily = primary.loc[
        primary["strategy"].eq("bandit_full_news_zeroed")
    ]
    strategy_risk_daily = primary.loc[
        primary["strategy"].eq("strategy_risk")
    ]
    tests = {
        "full_vs_strategy_risk": paired_tests(
            strategy_risk_daily,
            full_daily,
        ),
        "full_vs_same_model_news_zeroed": paired_tests(
            zeroed_daily,
            full_daily,
        ),
    }

    metric_primary = metrics.loc[
        metrics["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ].set_index("strategy")
    full_metric = metric_primary.loc["bandit_full"]
    strategy_metric = metric_primary.loc["strategy_risk"]
    zeroed_metric = metric_primary.loc["bandit_full_news_zeroed"]
    no_news_metric = metric_primary.loc["bandit_no_news"]
    checks = {
        "diagnostic_cer_above_strategy_risk": bool(
            full_metric["certainty_equivalent"]
            > strategy_metric["certainty_equivalent"]
        ),
        "diagnostic_sharpe_above_strategy_risk": bool(
            full_metric["sharpe"] > strategy_metric["sharpe"]
        ),
        "diagnostic_cer_above_same_model_news_zeroed": bool(
            full_metric["certainty_equivalent"]
            > zeroed_metric["certainty_equivalent"]
        ),
        "diagnostic_cer_above_retrained_no_news": bool(
            full_metric["certainty_equivalent"]
            > no_news_metric["certainty_equivalent"]
        ),
        "full_vs_strategy_bootstrap_ci_positive": bool(
            tests["full_vs_strategy_risk"]["cer_gain_ci_low"] > 0
        ),
        "news_bootstrap_ci_positive": bool(
            tests["full_vs_same_model_news_zeroed"]["cer_gain_ci_low"] > 0
        ),
    }
    worth_continuing = bool(
        checks["diagnostic_cer_above_strategy_risk"]
        and checks["diagnostic_sharpe_above_strategy_risk"]
    )
    news_helped = bool(
        checks["diagnostic_cer_above_same_model_news_zeroed"]
        and checks["diagnostic_cer_above_retrained_no_news"]
    )
    conclusion = {
        "worth_continuing": worth_continuing,
        "news_helped_diagnostic": news_helped,
        "promoted": False,
        "promotion_blocker": (
            "2021-2023 is already observed and no news-enabled external "
            "holdout is available"
        ),
        "checks": checks,
        "selected_full": {
            "model": full_selected["spec"].name,
            "model_kind": full_selected["spec"].kind,
            "policy": full_selected["row"],
        },
        "selected_no_news": {
            "model": no_news_selected["spec"].name,
            "model_kind": no_news_selected["spec"].kind,
            "policy": no_news_selected["row"],
        },
    }

    joblib.dump(
        {
            "model": full_model,
            "feature_columns": full_columns,
            "feature_bounds": full_bounds,
            "policy_config": {
                "residual_weight": full_config.residual_weight,
                "q_margin": full_config.q_margin,
                "maximum_change": full_config.maximum_change,
            },
            "model_spec": {
                "name": full_selected["spec"].name,
                "kind": full_selected["spec"].kind,
                "alpha": full_selected["spec"].alpha,
                "hidden": list(full_selected["spec"].hidden),
            },
            "training_range": [TRAIN_START, VALIDATION_END],
            "experimental_only": True,
        },
        OUT_DIR / "candidate_bandit_full.joblib",
    )
    daily.to_parquet(OUT_DIR / "daily_returns.parquet", index=False)
    metrics.to_csv(OUT_DIR / "portfolio_metrics.csv", index=False)
    (OUT_DIR / "paired_tests.json").write_text(
        json.dumps(tests, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "conclusion.json").write_text(
        json.dumps(conclusion, indent=2, default=str),
        encoding="utf-8",
    )

    display = metrics.loc[
        metrics["transaction_cost_bps"].eq(PRIMARY_COST_BPS),
        [
            "strategy",
            "cagr",
            "sharpe",
            "certainty_equivalent",
            "max_drawdown",
            "annual_turnover",
            "average_cash_weight",
        ],
    ].sort_values("certainty_equivalent", ascending=False)
    report = f"""# Residual Contextual-Bandit Exploration

## Bottom line

Worth continuing: **{str(worth_continuing).lower()}**.
News helped on the reused diagnostic period: **{str(news_helped).lower()}**.
Promoted: **false**.

The model is a full-information contextual bandit, not PPO/DQN. Daily
Strategy is the prior action; the learned Ridge/MLP Q-policy supplies a
residual action, HAR-X controls action conservatism, and a deterministic
safety layer enforces cash/no-leverage execution.

## Time protocol

- Train: **2018-2019**
- Configuration selection: **2020**
- Diagnostic: **2021-2023**
- Diagnostic status: previously observed, therefore not a fresh blind test
- Rebalance: every **{REBALANCE_SESSIONS} sessions**
- Primary one-way transaction cost: **{PRIMARY_COST_BPS:.0f} bps**

## Selected full policy

```json
{json.dumps(conclusion["selected_full"], indent=2, default=str)}
```

## Diagnostic portfolio performance

{markdown_table(display)}

## Paired statistical tests

```json
{json.dumps(tests, indent=2)}
```

## Checks

```json
{json.dumps(checks, indent=2)}
```

## Interpretation

- `bandit_full_news_zeroed` uses the exact same fitted model but zeros all
  news inputs at inference, isolating the operational effect of news.
- `bandit_no_news` is independently trained without news and estimates the
  no-news model ceiling.
- The candidate cannot be promoted even if diagnostic metrics improve,
  because 2021-2023 has already influenced project decisions and the
  2024-2026 panel has no archived news input.
- Counterfactual rewards use next-five-session excess return minus formal
  risk and transaction-cost penalties. New actions first earn t+1 returns.
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(conclusion, indent=2, default=str))
    print(f"Report: {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
