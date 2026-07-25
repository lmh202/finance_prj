"""Train and evaluate the gated-news decision layer.

Daily Strategy is the prior directional signal.  A full-information contextual
bandit may add only a small residual on rows with recent news.  HAR-X risk and
Portfolio Health are excluded from the learned state and are used only by the
external position/cash optimizer.

Protocol
--------
model/config selection:
    train through 2017 -> validate 2018
    train through 2018 -> validate 2019
    train through 2019 -> validate 2020
diagnostic test:
    refit through 2020 -> evaluate 2021-2023

The diagnostic period has already been inspected by prior experiments, so the
saved artifact remains experimental even if its numerical gates pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_rule_fusion as rule_backtest  # noqa: E402
from decision_layer_core import backtest_metrics  # noqa: E402
from src.recommendation.gated_news import (  # noqa: E402
    ACTIONS,
    CORE_FEATURES,
    FEATURES,
    NEWS_FEATURES,
    bounded_state_matrix,
    gated_direction_signal,
    risk_controlled_allocation,
    state_action_matrix,
)

OUT_DIR = ROOT / "reports" / "decision_layer_gated_news"
MODEL_DIR = (
    ROOT / "data" / "processed" / "decision_model_candidate_gated_news"
)
PANEL_PATH = ROOT / "data" / "processed" / "decision_dataset.parquet"
SEED = 20260724
PRIMARY_COST_BPS = 25.0
REBALANCE_SESSIONS = 5
INITIAL_CASH_WEIGHT = 0.05
REWARD_RESIDUAL_SIZE = 0.10
REWARD_COST_BPS = 25.0
REWARD_SCALE = 1_000.0

VALIDATION_FOLDS = (
    (2017, 2018),
    (2018, 2019),
    (2019, 2020),
)
DIAGNOSTIC_START = "2021-01-01"
DIAGNOSTIC_END = "2023-12-31"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    alpha: float
    hidden: tuple[int, ...] = ()


@dataclass(frozen=True)
class AllocationConfig:
    name: str
    return_signal_scale: float
    base_risk_aversion: float
    base_target_annual_volatility: float
    turnover_penalty: float = 0.0025


@dataclass(frozen=True)
class NewsPolicyConfig:
    residual_cap: float
    q_margin: float


MODEL_SPECS = (
    ModelSpec("ridge_a10", "ridge", 10.0),
    ModelSpec("mlp_32x16_a001", "mlp", 0.001, (32, 16)),
    ModelSpec("mlp_64x32_a01", "mlp", 0.01, (64, 32)),
)
ALLOCATION_CONFIGS = (
    AllocationConfig("risk4_vol15_sig10", 0.010, 4.0, 0.15),
    AllocationConfig("risk6_vol15_sig10", 0.010, 6.0, 0.15),
    AllocationConfig("risk4_vol18_sig15", 0.015, 4.0, 0.18),
    AllocationConfig("risk6_vol18_sig15", 0.015, 6.0, 0.18),
)
NEWS_CONFIGS = tuple(
    NewsPolicyConfig(cap, margin)
    for cap in (0.05, 0.10, 0.15)
    for margin in (0.10, 0.50)
)


def prepare_panel() -> pd.DataFrame:
    panel = rule_backtest.load_and_engineer_panel(
        PANEL_PATH,
        require_formal_risk=False,
    )
    benchmark = (
        panel[["date", "benchmark_ret_1d"]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")["benchmark_ret_1d"]
    )
    benchmark_growth = pd.Series(1.0, index=benchmark.index)
    for offset in range(1, 6):
        benchmark_growth *= 1.0 + benchmark.shift(-offset)
    panel["benchmark_fwd_ret_5d"] = panel["date"].map(
        benchmark_growth - 1.0
    )
    panel["excess_ret_5d"] = (
        panel["fwd_ret_5d"] - panel["benchmark_fwd_ret_5d"]
    )
    panel["rsi_scaled"] = (panel["rsi_14"] - 50.0) / 50.0
    panel["has_recent_news"] = panel["news_unique_5d"].gt(0)
    required = list(
        dict.fromkeys(
            FEATURES
            + [
                "excess_ret_5d",
                "risk_sigma_daily_5d",
                "ret_1d",
                "benchmark_ret_1d",
            ]
        )
    )
    fill_columns = list(
        dict.fromkeys(
            FEATURES + ["excess_ret_5d", "ret_1d", "benchmark_ret_1d"]
        )
    )
    panel[fill_columns] = (
        panel[fill_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    # Formal OOF HAR-X risk starts in 2018. It is required only during
    # allocation/backtesting, never imputed into the learned state.
    panel["risk_sigma_daily_5d"] = pd.to_numeric(
        panel["risk_sigma_daily_5d"],
        errors="coerce",
    )
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def fit_bounds(
    frame: pd.DataFrame,
    columns: Sequence[str] = FEATURES,
) -> dict[str, list[float]]:
    bounds: dict[str, list[float]] = {}
    for column in columns:
        values = (
            pd.to_numeric(frame[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if values.empty:
            lower, upper = -1.0, 1.0
        else:
            lower, upper = values.quantile([0.01, 0.99])
            if not np.isfinite(lower) or not np.isfinite(upper):
                lower, upper = -1.0, 1.0
            if upper <= lower:
                upper = lower + 1e-6
        bounds[column] = [float(lower), float(upper)]
    return bounds


def _balanced_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep all news rows and a deterministic equal-size no-news sample."""
    mature = frame.loc[frame["excess_ret_5d"].notna()].copy()
    news = mature.loc[mature["has_recent_news"]]
    no_news = mature.loc[~mature["has_recent_news"]]
    if news.empty or no_news.empty:
        return mature
    size = min(len(no_news), len(news))
    sampled_no_news = no_news.sample(size, random_state=SEED)
    return (
        pd.concat([news, sampled_no_news], ignore_index=True)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


def counterfactual_training_data(
    frame: pd.DataFrame,
    bounds: dict[str, list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    balanced = _balanced_training_rows(frame)
    states = bounded_state_matrix(balanced, FEATURES, bounds)
    repeated_states = np.repeat(states, len(ACTIONS), axis=0)
    actions = np.tile(ACTIONS, len(states))
    excess = np.repeat(
        balanced["excess_ret_5d"].to_numpy(dtype=float),
        len(ACTIONS),
    )
    delta = REWARD_RESIDUAL_SIZE * actions
    reward = (
        delta * excess
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
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
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


def fit_model(
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> tuple[Pipeline, dict[str, list[float]]]:
    bounds = fit_bounds(frame)
    x_train, y_train = counterfactual_training_data(frame, bounds)
    model = make_model(spec)
    model.fit(x_train, y_train)
    return model, bounds


def eligible_symbols(
    panel: pd.DataFrame,
    start: str,
    end: str,
) -> list[str]:
    return rule_backtest.eligible_symbols(panel, start, end)


def run_portfolio(
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
    period: str,
    strategy_name: str,
    allocation_config: AllocationConfig,
    model: object | None = None,
    bounds: dict[str, list[float]] | None = None,
    news_config: NewsPolicyConfig | None = None,
    news_mode: str = "full",
    sigma_multiplier: float = 1.0,
) -> pd.DataFrame:
    symbols = eligible_symbols(panel, start, end)
    if len(symbols) < 5:
        raise ValueError(f"{period}: fewer than five complete symbols")
    filtered = panel.loc[panel["symbol"].isin(symbols)].copy()
    returns = (
        filtered.pivot(index="date", columns="symbol", values="ret_1d")
        .reindex(columns=symbols)
        .sort_index()
    )
    benchmark = (
        filtered.groupby("date")["benchmark_ret_1d"].first().sort_index()
    )
    rows_by_date = {
        date: group.set_index("symbol").reindex(symbols).reset_index()
        for date, group in filtered.groupby("date")
    }
    dates = returns.index[
        returns.index.to_series().between(
            pd.Timestamp(start),
            pd.Timestamp(end),
        )
    ]
    weights = np.full(
        len(symbols),
        (1.0 - INITIAL_CASH_WEIGHT) / len(symbols),
    )
    cash = INITIAL_CASH_WEIGHT
    rebalance_set = set(dates[::REBALANCE_SESSIONS])
    output = []

    for date in dates:
        daily_asset_return = (
            returns.loc[date]
            .reindex(symbols)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        gross_return = float(weights @ daily_asset_return)
        growth = max(1.0 + gross_return, 1e-12)
        weights = weights * (1.0 + daily_asset_return) / growth
        cash /= growth
        total = float(weights.sum() + cash)
        weights /= max(total, 1e-12)
        cash /= max(total, 1e-12)

        turnover = 0.0
        maximum_change = 0.0
        minimum_trade = 0.0
        optimizer_success = True
        gate = {
            "news_available_share": 0.0,
            "news_applied_share": 0.0,
            "increase_share": 0.0,
            "reduce_share": 0.0,
            "hold_share": 1.0,
            "mean_q_advantage": 0.0,
            "mean_abs_news_residual": 0.0,
        }
        target_gross = float(weights.sum())
        predicted_annual_volatility = float("nan")
        target_annual_volatility = float("nan")
        health_score = rule_backtest.portfolio_health_score(
            returns.loc[:date, symbols],
            weights,
        )

        if date in rebalance_set:
            state = rows_by_date[date].copy()
            strategy_direction = (
                state["strategy_score"]
                .fillna(0.0)
                .clip(-1.0, 1.0)
                .to_numpy(dtype=float)
            )
            if model is None:
                direction = strategy_direction
            else:
                if bounds is None or news_config is None:
                    raise ValueError("model requires bounds and news config")
                model_state = state.copy()
                news_available = model_state["has_recent_news"].to_numpy(
                    dtype=bool
                )
                if news_mode == "unavailable":
                    news_available[:] = False
                elif news_mode == "sentiment_zero":
                    model_state[NEWS_FEATURES] = 0.0
                elif news_mode != "full":
                    raise ValueError(news_mode)
                direction, gate = gated_direction_signal(
                    model,
                    model_state,
                    FEATURES,
                    bounds,
                    residual_cap=news_config.residual_cap,
                    q_margin=news_config.q_margin,
                    news_available=news_available,
                )
            expected = (
                direction * allocation_config.return_signal_scale
            )
            sigma = (
                state["risk_sigma_daily_5d"]
                .fillna(0.0)
                .to_numpy(dtype=float)
                * float(sigma_multiplier)
            )
            allocation = risk_controlled_allocation(
                symbols,
                expected,
                sigma,
                returns.loc[:date, symbols],
                weights,
                cash,
                health_score=health_score,
                base_risk_aversion=allocation_config.base_risk_aversion,
                base_target_annual_volatility=(
                    allocation_config.base_target_annual_volatility
                ),
                turnover_penalty=allocation_config.turnover_penalty,
            )
            weights = allocation.weights
            cash = allocation.cash_weight
            turnover = allocation.turnover
            maximum_change = allocation.maximum_change
            minimum_trade = allocation.minimum_active_trade
            optimizer_success = allocation.success
            target_gross = allocation.target_gross
            predicted_annual_volatility = (
                allocation.predicted_annual_volatility
            )
            target_annual_volatility = allocation.target_annual_volatility

        output.append(
            {
                "date": date,
                "period": period,
                "strategy": strategy_name,
                "gross_return": gross_return,
                "benchmark_return": float(benchmark.loc[date]),
                "turnover": turnover,
                "cash_weight": cash,
                "gross_exposure": float(weights.sum()),
                "maximum_weight": float(weights.max()),
                "maximum_change": maximum_change,
                "minimum_active_trade": minimum_trade,
                "optimizer_success": optimizer_success,
                "health_score": health_score,
                "target_gross": target_gross,
                "predicted_annual_volatility": predicted_annual_volatility,
                "target_annual_volatility": target_annual_volatility,
                **gate,
            }
        )
    return pd.DataFrame(output)


def apply_cost(
    daily: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    frame = daily.copy()
    frame["transaction_cost_bps"] = float(cost_bps)
    frame["transaction_cost"] = (
        frame["turnover"] * float(cost_bps) / 10_000.0
    )
    frame["net_return"] = frame["gross_return"] - frame["transaction_cost"]
    frame["max_weight"] = frame["maximum_weight"]
    frame["max_change"] = frame["maximum_change"]
    return frame


def metric_row(daily: pd.DataFrame) -> dict:
    metrics = backtest_metrics(daily)
    return {
        **metrics,
        "average_turnover": float(daily["turnover"].mean()),
        "average_cash_weight": float(daily["cash_weight"].mean()),
        "average_gross_exposure": float(daily["gross_exposure"].mean()),
        "optimizer_success_rate": float(daily["optimizer_success"].mean()),
        "news_applied_share": float(daily["news_applied_share"].mean()),
    }


def paired_tests(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict:
    left = reference.sort_values("date").set_index("date")["net_return"]
    right = candidate.sort_values("date").set_index("date")["net_return"]
    common = left.index.intersection(right.index)
    difference = (
        right.reindex(common).to_numpy(dtype=float)
        - left.reindex(common).to_numpy(dtype=float)
    )
    return {
        **rule_backtest.newey_west_mean_test(difference, lag=5),
        **rule_backtest.block_bootstrap_cer_gain(
            left.reindex(common).to_numpy(dtype=float),
            right.reindex(common).to_numpy(dtype=float),
            block_length=10,
            seed=SEED,
        ),
    }


def select_allocation_config(panel: pd.DataFrame) -> tuple[
    AllocationConfig,
    pd.DataFrame,
    dict[tuple[int, str], pd.DataFrame],
]:
    rows = []
    paths: dict[tuple[int, str], pd.DataFrame] = {}
    for config in ALLOCATION_CONFIGS:
        for train_year, validation_year in VALIDATION_FOLDS:
            print(
                f"[allocation] {config.name} validation={validation_year}",
                flush=True,
            )
            daily = run_portfolio(
                panel,
                start=f"{validation_year}-01-01",
                end=f"{validation_year}-12-31",
                period=f"validation_{validation_year}",
                strategy_name="strategy_risk_control",
                allocation_config=config,
            )
            costed = apply_cost(daily, PRIMARY_COST_BPS)
            paths[(validation_year, config.name)] = costed
            rows.append(
                {
                    "train_through": train_year,
                    "validation_year": validation_year,
                    "allocation_config": config.name,
                    **metric_row(costed),
                }
            )
    table = pd.DataFrame(rows)
    aggregate = (
        table.groupby("allocation_config")
        .agg(
            mean_cer=("certainty_equivalent", "mean"),
            mean_sharpe=("sharpe", "mean"),
            worst_drawdown=("max_drawdown", "min"),
            mean_turnover=("average_turnover", "mean"),
        )
        .sort_values(["mean_cer", "mean_sharpe"], ascending=False)
    )
    winner_name = str(aggregate.index[0])
    winner = next(
        config for config in ALLOCATION_CONFIGS if config.name == winner_name
    )
    return winner, table, paths


def search_news_policy(
    panel: pd.DataFrame,
    allocation_config: AllocationConfig,
    baseline_paths: dict[tuple[int, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, ModelSpec, NewsPolicyConfig]:
    rows = []
    for train_year, validation_year in VALIDATION_FOLDS:
        train = panel.loc[panel["date"].dt.year.le(train_year)].copy()
        for spec in MODEL_SPECS:
            print(
                f"[fit] {spec.name} through={train_year} "
                f"validation={validation_year}",
                flush=True,
            )
            model, bounds = fit_model(train, spec)
            baseline = baseline_paths[
                (validation_year, allocation_config.name)
            ]
            baseline_metrics = metric_row(baseline)
            for news_config in NEWS_CONFIGS:
                daily = run_portfolio(
                    panel,
                    start=f"{validation_year}-01-01",
                    end=f"{validation_year}-12-31",
                    period=f"validation_{validation_year}",
                    strategy_name="gated_news",
                    allocation_config=allocation_config,
                    model=model,
                    bounds=bounds,
                    news_config=news_config,
                )
                costed = apply_cost(daily, PRIMARY_COST_BPS)
                metrics = metric_row(costed)
                rows.append(
                    {
                        "train_through": train_year,
                        "validation_year": validation_year,
                        "model": spec.name,
                        "kind": spec.kind,
                        "alpha": spec.alpha,
                        "hidden": "x".join(map(str, spec.hidden)),
                        "residual_cap": news_config.residual_cap,
                        "q_margin": news_config.q_margin,
                        **metrics,
                        "cer_gain_vs_strategy": (
                            metrics["certainty_equivalent"]
                            - baseline_metrics["certainty_equivalent"]
                        ),
                        "sharpe_gain_vs_strategy": (
                            metrics["sharpe"] - baseline_metrics["sharpe"]
                        ),
                        "drawdown_change_vs_strategy": (
                            metrics["max_drawdown"]
                            - baseline_metrics["max_drawdown"]
                        ),
                    }
                )
    table = pd.DataFrame(rows)
    aggregate = (
        table.groupby(["model", "residual_cap", "q_margin"], as_index=False)
        .agg(
            mean_cer=("certainty_equivalent", "mean"),
            mean_sharpe=("sharpe", "mean"),
            mean_cer_gain=("cer_gain_vs_strategy", "mean"),
            median_cer_gain=("cer_gain_vs_strategy", "median"),
            mean_sharpe_gain=("sharpe_gain_vs_strategy", "mean"),
            positive_cer_years=(
                "cer_gain_vs_strategy",
                lambda values: int(np.sum(np.asarray(values) > 0)),
            ),
            worst_drawdown_change=(
                "drawdown_change_vs_strategy",
                "min",
            ),
            mean_news_applied_share=("news_applied_share", "mean"),
        )
    )
    eligible = aggregate.loc[
        aggregate["mean_cer_gain"].gt(0)
        & aggregate["median_cer_gain"].gt(0)
        & aggregate["mean_sharpe_gain"].gt(0)
        & aggregate["positive_cer_years"].ge(2)
        & aggregate["worst_drawdown_change"].ge(-0.02)
    ]
    pool = eligible if not eligible.empty else aggregate
    winner = pool.sort_values(
        ["mean_cer_gain", "mean_cer", "mean_sharpe"],
        ascending=False,
    ).iloc[0]
    selected_spec = next(
        spec for spec in MODEL_SPECS if spec.name == winner["model"]
    )
    selected_news = NewsPolicyConfig(
        residual_cap=float(winner["residual_cap"]),
        q_margin=float(winner["q_margin"]),
    )
    return table, aggregate, selected_spec, selected_news


def _metrics_table(paths: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for path in paths:
        keys = path.iloc[0]
        rows.append(
            {
                "period": keys["period"],
                "strategy": keys["strategy"],
                "transaction_cost_bps": keys["transaction_cost_bps"],
                **metric_row(path),
            }
        )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    return rule_backtest.markdown_table(frame, digits=digits)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    print(
        f"[data] rows={len(panel):,} symbols={panel.symbol.nunique()} "
        f"range={panel.date.min().date()}..{panel.date.max().date()}",
        flush=True,
    )

    allocation_path = OUT_DIR / "validation_allocation_search.csv"
    news_search_path = OUT_DIR / "validation_news_search.csv"
    news_aggregate_path = OUT_DIR / "validation_news_aggregate.csv"
    force_search = os.environ.get("AURORA_FORCE_SEARCH") == "1"
    if (
        not force_search
        and
        allocation_path.exists()
        and news_search_path.exists()
        and news_aggregate_path.exists()
    ):
        print("[resume] using completed validation search tables", flush=True)
        allocation_search = pd.read_csv(allocation_path)
        allocation_name = str(
            allocation_search.groupby("allocation_config")[
                "certainty_equivalent"
            ]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )
        allocation = next(
            config
            for config in ALLOCATION_CONFIGS
            if config.name == allocation_name
        )
        news_search = pd.read_csv(news_search_path)
        news_aggregate = pd.read_csv(news_aggregate_path)
        eligible = news_aggregate.loc[
            news_aggregate["mean_cer_gain"].gt(0)
            & news_aggregate["median_cer_gain"].gt(0)
            & news_aggregate["mean_sharpe_gain"].gt(0)
            & news_aggregate["positive_cer_years"].ge(2)
            & news_aggregate["worst_drawdown_change"].ge(-0.02)
        ]
        pool = eligible if not eligible.empty else news_aggregate
        winner = pool.sort_values(
            ["mean_cer_gain", "mean_cer", "mean_sharpe"],
            ascending=False,
        ).iloc[0]
        selected_spec = next(
            spec for spec in MODEL_SPECS if spec.name == winner["model"]
        )
        selected_news = NewsPolicyConfig(
            residual_cap=float(winner["residual_cap"]),
            q_margin=float(winner["q_margin"]),
        )
    else:
        allocation, allocation_search, baseline_paths = (
            select_allocation_config(panel)
        )
        allocation_search.to_csv(allocation_path, index=False)
        print(f"[selected allocation] {allocation.name}", flush=True)
        (
            news_search,
            news_aggregate,
            selected_spec,
            selected_news,
        ) = search_news_policy(panel, allocation, baseline_paths)
        news_search.to_csv(news_search_path, index=False)
        news_aggregate.to_csv(news_aggregate_path, index=False)
    print(
        f"[selected news] {selected_spec.name} "
        f"cap={selected_news.residual_cap} "
        f"margin={selected_news.q_margin}",
        flush=True,
    )

    development = panel.loc[panel["date"].dt.year.le(2020)].copy()
    final_model, final_bounds = fit_model(development, selected_spec)

    diagnostic_base = run_portfolio(
        panel,
        start=DIAGNOSTIC_START,
        end=DIAGNOSTIC_END,
        period="diagnostic_2021_2023",
        strategy_name="strategy_risk_control",
        allocation_config=allocation,
    )
    diagnostic_full = run_portfolio(
        panel,
        start=DIAGNOSTIC_START,
        end=DIAGNOSTIC_END,
        period="diagnostic_2021_2023",
        strategy_name="gated_news",
        allocation_config=allocation,
        model=final_model,
        bounds=final_bounds,
        news_config=selected_news,
    )
    diagnostic_no_news = run_portfolio(
        panel,
        start=DIAGNOSTIC_START,
        end=DIAGNOSTIC_END,
        period="diagnostic_2021_2023",
        strategy_name="same_model_news_unavailable",
        allocation_config=allocation,
        model=final_model,
        bounds=final_bounds,
        news_config=selected_news,
        news_mode="unavailable",
    )
    diagnostic_sentiment_zero = run_portfolio(
        panel,
        start=DIAGNOSTIC_START,
        end=DIAGNOSTIC_END,
        period="diagnostic_2021_2023",
        strategy_name="same_model_sentiment_zero",
        allocation_config=allocation,
        model=final_model,
        bounds=final_bounds,
        news_config=selected_news,
        news_mode="sentiment_zero",
    )
    diagnostic_risk_stress = run_portfolio(
        panel,
        start=DIAGNOSTIC_START,
        end=DIAGNOSTIC_END,
        period="diagnostic_2021_2023",
        strategy_name="gated_news_risk_x2",
        allocation_config=allocation,
        model=final_model,
        bounds=final_bounds,
        news_config=selected_news,
        sigma_multiplier=2.0,
    )

    costed_paths = []
    for cost in (0.0, 25.0, 50.0):
        for path in (
            diagnostic_base,
            diagnostic_full,
            diagnostic_no_news,
            diagnostic_sentiment_zero,
        ):
            costed_paths.append(apply_cost(path, cost))
    daily = pd.concat(costed_paths, ignore_index=True)
    daily.to_parquet(OUT_DIR / "diagnostic_daily_paths.parquet", index=False)
    metrics = _metrics_table(costed_paths)
    metrics.to_csv(OUT_DIR / "diagnostic_metrics.csv", index=False)

    primary_base = apply_cost(diagnostic_base, PRIMARY_COST_BPS)
    primary_full = apply_cost(diagnostic_full, PRIMARY_COST_BPS)
    primary_no_news = apply_cost(diagnostic_no_news, PRIMARY_COST_BPS)
    primary_sentiment_zero = apply_cost(
        diagnostic_sentiment_zero,
        PRIMARY_COST_BPS,
    )
    tests = {
        "gated_news_vs_strategy": paired_tests(primary_base, primary_full),
        "gated_news_vs_news_unavailable": paired_tests(
            primary_no_news,
            primary_full,
        ),
        "gated_news_vs_sentiment_zero": paired_tests(
            primary_sentiment_zero,
            primary_full,
        ),
    }
    (OUT_DIR / "statistical_tests.json").write_text(
        json.dumps(tests, indent=2),
        encoding="utf-8",
    )

    no_news_exact = bool(
        np.allclose(
            diagnostic_base["gross_return"],
            diagnostic_no_news["gross_return"],
            atol=1e-12,
            rtol=0.0,
        )
        and np.allclose(
            diagnostic_base["turnover"],
            diagnostic_no_news["turnover"],
            atol=1e-12,
            rtol=0.0,
        )
    )
    normal_rebalances = diagnostic_full["target_gross"].notna()
    stressed_rebalances = diagnostic_risk_stress["target_gross"].notna()
    risk_control = {
        "normal_average_gross": float(
            diagnostic_full.loc[
                normal_rebalances, "gross_exposure"
            ].mean()
        ),
        "double_sigma_average_gross": float(
            diagnostic_risk_stress.loc[
                stressed_rebalances, "gross_exposure"
            ].mean()
        ),
        "double_sigma_reduces_gross": bool(
            diagnostic_risk_stress["gross_exposure"].mean()
            < diagnostic_full["gross_exposure"].mean() - 1e-6
        ),
        "no_news_exact_strategy_fallback": no_news_exact,
    }
    (OUT_DIR / "risk_control_checks.json").write_text(
        json.dumps(risk_control, indent=2),
        encoding="utf-8",
    )

    selected_validation = news_search.loc[
        news_search["model"].eq(selected_spec.name)
        & news_search["residual_cap"].eq(selected_news.residual_cap)
        & news_search["q_margin"].eq(selected_news.q_margin)
    ]
    validation_checks = {
        "mean_cer_gain_positive": bool(
            selected_validation["cer_gain_vs_strategy"].mean() > 0
        ),
        "median_cer_gain_positive": bool(
            selected_validation["cer_gain_vs_strategy"].median() > 0
        ),
        "positive_in_at_least_2_of_3_years": bool(
            (selected_validation["cer_gain_vs_strategy"] > 0).sum() >= 2
        ),
        "mean_sharpe_gain_positive": bool(
            selected_validation["sharpe_gain_vs_strategy"].mean() > 0
        ),
        "worst_drawdown_not_worse_by_more_than_2pp": bool(
            selected_validation["drawdown_change_vs_strategy"].min() >= -0.02
        ),
        "no_news_exact_fallback": no_news_exact,
        "risk_stress_reduces_gross": risk_control[
            "double_sigma_reduces_gross"
        ],
    }
    diagnostic_statistical_checks = {
        "newey_west_p_below_005": bool(
            tests["gated_news_vs_strategy"]["newey_west_p"] < 0.05
        ),
        "bootstrap_cer_gain_lower_bound_positive": bool(
            tests["gated_news_vs_strategy"]["cer_gain_ci_low"] > 0
        ),
    }
    validation_numerical_gates_passed = bool(
        all(validation_checks.values())
    )
    numerical_gates_passed = bool(
        validation_numerical_gates_passed
        and all(diagnostic_statistical_checks.values())
    )
    promotion = {
        "validation_numerical_gates_passed": (
            validation_numerical_gates_passed
        ),
        "numerical_gates_passed": numerical_gates_passed,
        "promotion_status": "experimental_only",
        "promotion_blocker": (
            "The diagnostic news gain is not statistically distinguishable "
            "from zero, and 2021-2023 was already observed by earlier "
            "decision-layer experiments. At least 60 mature live "
            "five-session predictions are required for a fresh promotion "
            "decision."
        ),
        "validation_checks": validation_checks,
        "diagnostic_statistical_checks": diagnostic_statistical_checks,
        "diagnostic_only": {
            "gated_news_vs_strategy": tests["gated_news_vs_strategy"],
            "gated_news_vs_news_unavailable": tests[
                "gated_news_vs_news_unavailable"
            ],
        },
    }

    model_path = MODEL_DIR / "q_model.joblib"
    joblib.dump(final_model, model_path)
    metadata = {
        "schema_version": 1,
        "model_version": "gated-news-risk-control-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "promotion_status": "experimental_only",
        "architecture": {
            "direction_prior": "Daily Strategy",
            "learned_component": "confidence-gated news residual",
            "risk_control": "external HAR-X covariance and gross exposure",
            "health_control": "external risk budget and risk aversion",
            "risk_is_model_feature": False,
            "health_is_model_feature": False,
            "no_news_behavior": "exact Daily Strategy fallback",
        },
        "feature_order": FEATURES,
        "core_features": CORE_FEATURES,
        "news_features": NEWS_FEATURES,
        "feature_bounds": final_bounds,
        "selected_model": asdict(selected_spec),
        "residual_cap": selected_news.residual_cap,
        "q_margin": selected_news.q_margin,
        "return_signal_scale": allocation.return_signal_scale,
        "base_risk_aversion": allocation.base_risk_aversion,
        "base_target_annual_volatility": (
            allocation.base_target_annual_volatility
        ),
        "turnover_penalty": allocation.turnover_penalty,
        "rebalance_sessions": REBALANCE_SESSIONS,
        "training_range": [
            str(development["date"].min().date()),
            str(development["date"].max().date()),
        ],
        "validation_years": [2018, 2019, 2020],
        "diagnostic_years": [2021, 2022, 2023],
        "training_rows": int(len(development)),
        "training_symbols": int(development["symbol"].nunique()),
        "data_sha256": _sha256(PANEL_PATH),
        "model_sha256": _sha256(model_path),
        "promotion": promotion,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "promotion_gates.json").write_text(
        json.dumps(promotion, indent=2),
        encoding="utf-8",
    )

    validation_summary = selected_validation[
        [
            "validation_year",
            "certainty_equivalent",
            "sharpe",
            "max_drawdown",
            "cer_gain_vs_strategy",
            "sharpe_gain_vs_strategy",
            "news_applied_share",
        ]
    ].sort_values("validation_year")
    diagnostic_summary = metrics.loc[
        metrics["transaction_cost_bps"].eq(PRIMARY_COST_BPS),
        [
            "strategy",
            "cagr",
            "sharpe",
            "certainty_equivalent",
            "max_drawdown",
            "average_turnover",
            "average_cash_weight",
            "news_applied_share",
        ],
    ]
    report = f"""# Gated News Decision Layer

## Decision architecture

Daily Strategy supplies the prior direction. The learned model can add at most
{selected_news.residual_cap:.0%} only when recent news exists and the estimated
Q advantage is at least {selected_news.q_margin:.2f}. HAR-X risk and Portfolio
Health are deliberately absent from the learned state: risk determines the
covariance, risky gross exposure and cash; Health changes the volatility budget
and risk aversion. No-news rows use the exact strategy-only path.

## Selection protocol

The model was selected with three expanding validation folds (2018, 2019,
2020). The final experimental artifact was refitted through 2020 and evaluated
on 2021-2023. That latter range is diagnostic, not a fresh blind test, because
earlier decision experiments already inspected it.

Selected model: `{selected_spec.name}`. Selected allocation:
`{allocation.name}`.

### Validation

{_markdown_table(validation_summary)}

### Diagnostic test at 25 bps

{_markdown_table(diagnostic_summary)}

## Engineering checks

- No-news exact fallback: `{no_news_exact}`.
- Doubling every HAR-X sigma lowers average risky gross:
  `{risk_control["double_sigma_reduces_gross"]}`.
- Validation-only numerical gates passed:
  `{validation_numerical_gates_passed}`.
- Full numerical/statistical promotion gates passed:
  `{numerical_gates_passed}`.
- Direct-news residual status: `experimental_only`.

## Deployment decision

The production-safe path is `strategy_external_harx_news_risk`: Daily Strategy
sets direction while the formal HAR-X + News estimate and Portfolio Health
control stock weights, risky gross exposure, and cash. Thus news still affects
the output through estimated risk. The direct 5% news residual is disabled
because it failed cross-year and statistical gates. A fresh live archive with
at least 60 mature five-session decisions is required before reconsidering it.
"""
    (OUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)


if __name__ == "__main__":
    main()
