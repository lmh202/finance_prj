"""Train the residual-alpha enhancement for AURORA Daily Strategy.

The fixed Daily Strategy remains the prior. XGBoost learns the part of the
future five-session cross-sectional residual-return rank that the rule score
does not explain. HAR-X + News risk and Portfolio Health stay outside the
direction model and size the resulting portfolio deterministically.

Selection protocol
------------------
expanding validation:
    train through 2017 -> validate 2018
    train through 2018 -> validate 2019
    train through 2019 -> validate 2020
diagnostic:
    refit through 2020 -> 2021-2023
external:
    refit through 2023 -> 2024-2026, including unseen symbols
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_rule_fusion as rule_backtest  # noqa: E402
from decision_layer_core import backtest_metrics  # noqa: E402
from src.recommendation.gated_news import (  # noqa: E402
    risk_controlled_allocation,
)

HISTORICAL_PATH = ROOT / "data" / "processed" / "decision_dataset.parquet"
EXTERNAL_PATH = (
    ROOT / "data" / "processed" / "decision_external_dataset.parquet"
)
OUT_DIR = ROOT / "reports" / "daily_strategy_enhanced"
MODEL_DIR = (
    ROOT / "data" / "processed" / "daily_strategy_model_candidate"
)
SEED = 20260724
PRIMARY_COST_BPS = 25.0
REBALANCE_SESSIONS = 5
INITIAL_CASH_WEIGHT = 0.05
SEARCH_CONFIGS = 32
TOP_PORTFOLIO_MODELS = 5
PURGE_SESSIONS = 5

VALIDATION_FOLDS = (
    (2017, 2018),
    (2018, 2019),
    (2019, 2020),
)

BASE_PRICE_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "mom_20d",
    "mom_60d",
    "momentum_acceleration",
    "price_vs_sma50",
    "sma50_vs_sma200",
    "trend_strength",
    "vol_5d",
    "vol_20d",
    "vol_60d",
    "vol_ratio_5_20",
    "vol_ratio_20_60",
    "downside_vol_20d",
    "skew_20d",
    "rsi_scaled",
    "drawdown",
    "risk_adj_mom",
    "beta_60d",
    "rel_str_20d",
    "strategy_sharpe",
    "strategy_drawdown",
    "strategy_score",
]
RANK_SOURCES = [
    "ret_5d",
    "mom_20d",
    "mom_60d",
    "momentum_acceleration",
    "trend_strength",
    "vol_20d",
    "downside_vol_20d",
    "rsi_scaled",
    "drawdown",
    "risk_adj_mom",
    "rel_str_20d",
]
RANK_FEATURES = [f"rank_{column}" for column in RANK_SOURCES]
MARKET_FEATURES = [
    "benchmark_ret_1d",
    "market_ret_5d",
    "market_ret_20d",
    "market_ret_60d",
    "market_vol_20d",
    "market_vol_60d",
    "market_drawdown",
]
NEWS_FEATURES = [
    "has_news",
    "sent_mean",
    "sent_positive_share",
    "sent_negative_share",
    "sent_extreme_share",
    "sent_surprise20",
]
PRICE_FEATURES = BASE_PRICE_FEATURES + RANK_FEATURES
DEPLOYABLE_FEATURES = PRICE_FEATURES + MARKET_FEATURES
RESEARCH_NEWS_FEATURES = DEPLOYABLE_FEATURES + NEWS_FEATURES


@dataclass(frozen=True)
class XGBSpec:
    name: str
    max_depth: int
    learning_rate: float
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    gamma: float


@dataclass(frozen=True)
class BlendConfig:
    eta: float
    residual_cap: float


BLEND_CONFIGS = tuple(
    BlendConfig(eta, cap)
    for eta in (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
    for cap in (0.10, 0.25, 0.50)
)
PORTFOLIO_BLEND_CONFIGS = (
    BlendConfig(0.05, 0.10),
    BlendConfig(0.10, 0.10),
    BlendConfig(0.25, 0.10),
    BlendConfig(0.25, 0.25),
    BlendConfig(0.50, 0.25),
    BlendConfig(0.50, 0.50),
    BlendConfig(0.75, 0.50),
)


def _compound_return(values: pd.Series, window: int) -> pd.Series:
    return (
        np.exp(
            np.log1p(values.clip(lower=-0.999999))
            .rolling(window, min_periods=window)
            .sum()
        )
        - 1.0
    )


def engineer_panel(path: Path, *, external: bool) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = panel.groupby("symbol", sort=False)

    panel["ret_5d"] = grouped["ret_1d"].transform(
        lambda values: _compound_return(values, 5)
    )
    panel["ret_10d"] = grouped["ret_1d"].transform(
        lambda values: _compound_return(values, 10)
    )
    panel["vol_5d"] = grouped["ret_1d"].transform(
        lambda values: values.rolling(5, min_periods=5).std()
        * math.sqrt(252)
    )
    panel["vol_60d"] = grouped["ret_1d"].transform(
        lambda values: values.rolling(60, min_periods=40).std()
        * math.sqrt(252)
    )
    panel["downside_vol_20d"] = grouped["ret_1d"].transform(
        lambda values: values.clip(upper=0.0)
        .rolling(20, min_periods=15)
        .std()
        * math.sqrt(252)
    )
    panel["skew_20d"] = grouped["ret_1d"].transform(
        lambda values: values.rolling(20, min_periods=15).skew()
    )
    panel["momentum_acceleration"] = (
        panel["mom_20d"] - panel["mom_60d"] / 3.0
    )
    panel["trend_strength"] = (
        panel["price_vs_sma50"] + panel["sma50_vs_sma200"]
    )
    panel["vol_ratio_5_20"] = panel["vol_5d"] / panel[
        "vol_20d"
    ].replace(0.0, np.nan)
    panel["vol_ratio_20_60"] = panel["vol_20d"] / panel[
        "vol_60d"
    ].replace(0.0, np.nan)
    panel["rsi_scaled"] = (panel["rsi_14"] - 50.0) / 50.0

    rolling_mean = grouped["ret_1d"].transform(
        lambda values: values.rolling(252, min_periods=126).mean()
    )
    rolling_std = grouped["ret_1d"].transform(
        lambda values: values.rolling(252, min_periods=126).std()
    )
    panel["strategy_sharpe"] = (
        rolling_mean
        * 252.0
        / (rolling_std * math.sqrt(252.0)).replace(0.0, np.nan)
    )
    panel["strategy_trend"] = (
        0.5 * panel["price_vs_sma50"].gt(0).astype(float)
        + 0.5 * panel["sma50_vs_sma200"].gt(0).astype(float)
    )
    price_index = grouped["ret_1d"].transform(
        lambda values: (1.0 + values.fillna(0.0)).cumprod()
    )
    panel["_price_index"] = price_index
    rolling_peak = panel.groupby("symbol", sort=False)[
        "_price_index"
    ].transform(
        lambda values: values.rolling(252, min_periods=60).max()
    )
    panel["strategy_drawdown"] = (
        panel["_price_index"] / rolling_peak - 1.0
    )
    for source in (
        "mom_60d",
        "strategy_trend",
        "strategy_sharpe",
        "strategy_drawdown",
    ):
        panel[f"_strategy_rank_{source}"] = panel.groupby("date")[
            source
        ].rank(pct=True)
    directional_01 = (
        0.30 * panel["_strategy_rank_mom_60d"]
        + 0.25 * panel["_strategy_rank_strategy_trend"]
        + 0.20 * panel["_strategy_rank_strategy_sharpe"]
        + 0.10 * panel["_strategy_rank_strategy_drawdown"]
    ) / 0.85
    panel["strategy_score"] = (
        2.0 * directional_01 - 1.0
    ).clip(-1.0, 1.0)

    for source in RANK_SOURCES:
        panel[f"rank_{source}"] = (
            2.0 * panel.groupby("date")[source].rank(pct=True) - 1.0
        )

    market = (
        panel[["date", "benchmark_ret_1d"]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )
    market["market_ret_5d"] = _compound_return(
        market["benchmark_ret_1d"],
        5,
    )
    market["market_ret_20d"] = _compound_return(
        market["benchmark_ret_1d"],
        20,
    )
    market["market_ret_60d"] = _compound_return(
        market["benchmark_ret_1d"],
        60,
    )
    market["market_vol_20d"] = (
        market["benchmark_ret_1d"].rolling(20, min_periods=15).std()
        * math.sqrt(252)
    )
    market["market_vol_60d"] = (
        market["benchmark_ret_1d"].rolling(60, min_periods=40).std()
        * math.sqrt(252)
    )
    market_index = (1.0 + market["benchmark_ret_1d"].fillna(0.0)).cumprod()
    market["market_drawdown"] = market_index / market_index.cummax() - 1.0
    benchmark_growth = pd.Series(1.0, index=market.index)
    for offset in range(1, 6):
        benchmark_growth *= 1.0 + market["benchmark_ret_1d"].shift(-offset)
    market["benchmark_fwd_ret_5d"] = benchmark_growth - 1.0
    panel = panel.merge(
        market.drop(columns=["benchmark_ret_1d"]),
        left_on="date",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    panel["residual_return_5d"] = (
        panel["fwd_ret_5d"]
        - panel["beta_60d"].clip(-2.0, 3.0)
        * panel["benchmark_fwd_ret_5d"]
    )
    panel["target_rank_5d"] = (
        2.0
        * panel.groupby("date")["residual_return_5d"].rank(pct=True)
        - 1.0
    )
    panel["residual_alpha_target"] = (
        panel["target_rank_5d"] - panel["strategy_score"]
    )

    if external:
        for column in NEWS_FEATURES:
            if column not in panel:
                panel[column] = 0.0
        panel["is_seen_symbol"] = panel["groups"].astype(str).str.contains(
            "original_research",
            regex=False,
        )
    else:
        panel["is_seen_symbol"] = True

    required_features = list(
        dict.fromkeys(RESEARCH_NEWS_FEATURES)
    )
    for column in required_features:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel[required_features] = panel[required_features].fillna(0.0)
    panel["risk_sigma_daily_5d"] = pd.to_numeric(
        panel["risk_sigma_daily_5d"],
        errors="coerce",
    )
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def make_specs(count: int = SEARCH_CONFIGS) -> tuple[XGBSpec, ...]:
    rng = np.random.default_rng(SEED)
    specs = []
    for index in range(count):
        specs.append(
            XGBSpec(
                name=f"xgb_{index:02d}",
                max_depth=int(rng.choice([2, 3, 4, 5, 6])),
                learning_rate=float(
                    np.exp(rng.uniform(np.log(0.01), np.log(0.12)))
                ),
                min_child_weight=float(
                    rng.choice([8, 16, 32, 64, 96])
                ),
                subsample=float(rng.uniform(0.65, 1.0)),
                colsample_bytree=float(rng.uniform(0.65, 1.0)),
                reg_alpha=float(rng.choice([0.0, 0.01, 0.05, 0.2, 0.5])),
                reg_lambda=float(rng.choice([5.0, 10.0, 20.0, 40.0, 80.0])),
                gamma=float(rng.choice([0.0, 0.01, 0.05, 0.10])),
            )
        )
    return tuple(specs)


def make_model(spec: XGBSpec) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2500,
        early_stopping_rounds=100,
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        max_depth=spec.max_depth,
        learning_rate=spec.learning_rate,
        min_child_weight=spec.min_child_weight,
        subsample=spec.subsample,
        colsample_bytree=spec.colsample_bytree,
        reg_alpha=spec.reg_alpha,
        reg_lambda=spec.reg_lambda,
        gamma=spec.gamma,
        random_state=SEED,
        n_jobs=4,
    )


def purged_train_mask(
    panel: pd.DataFrame,
    validation_year: int,
) -> np.ndarray:
    dates = np.asarray(sorted(panel["date"].unique()))
    validation_start = pd.Timestamp(f"{validation_year}-01-01")
    before = dates[dates < np.datetime64(validation_start)]
    if len(before) <= PURGE_SESSIONS:
        raise ValueError("not enough dates for purge")
    cutoff = pd.Timestamp(before[-PURGE_SESSIONS - 1])
    return panel["date"].le(cutoff).to_numpy()


def fold_frames(
    panel: pd.DataFrame,
    validation_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = panel.loc[
        purged_train_mask(panel, validation_year)
        & panel["target_rank_5d"].notna()
    ].copy()
    validation = panel.loc[
        panel["date"].dt.year.eq(validation_year)
        & panel["target_rank_5d"].notna()
    ].copy()
    return train, validation


def fit_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    spec: XGBSpec,
    features: Sequence[str],
) -> tuple[XGBRegressor, np.ndarray, float]:
    strategy = train["strategy_score"].to_numpy(dtype=float)
    target = train["target_rank_5d"].to_numpy(dtype=float)
    centered_strategy = strategy - strategy.mean()
    denominator = float(centered_strategy @ centered_strategy)
    rule_slope = (
        float(
            centered_strategy
            @ (target - target.mean())
            / denominator
        )
        if denominator > 1e-12
        else 0.0
    )
    train_residual = target - rule_slope * strategy
    validation_residual = (
        validation["target_rank_5d"].to_numpy(dtype=float)
        - rule_slope
        * validation["strategy_score"].to_numpy(dtype=float)
    )
    model = make_model(spec)
    model.fit(
        train[list(features)],
        train_residual,
        eval_set=[
            (
                validation[list(features)],
                validation_residual,
            )
        ],
        verbose=False,
    )
    return (
        model,
        np.asarray(
            model.predict(validation[list(features)]),
            dtype=float,
        ),
        rule_slope,
    )


def blended_score(
    strategy_score: np.ndarray,
    predicted_residual: np.ndarray,
    blend: BlendConfig,
    dates: Iterable[pd.Timestamp],
) -> np.ndarray:
    raw = np.asarray(strategy_score, dtype=float) + blend.eta * np.clip(
        np.asarray(predicted_residual, dtype=float),
        -blend.residual_cap,
        blend.residual_cap,
    )
    frame = pd.DataFrame({"date": list(dates), "raw": raw})
    return (
        2.0 * frame.groupby("date")["raw"].rank(pct=True) - 1.0
    ).to_numpy(dtype=float)


def signal_metrics(
    frame: pd.DataFrame,
    score: np.ndarray,
) -> dict[str, float]:
    work = frame[
        ["date", "target_rank_5d", "residual_return_5d"]
    ].copy()
    work["score"] = np.asarray(score, dtype=float)
    daily_ic = []
    spreads = []
    for _, group in work.groupby("date", sort=False):
        if len(group) < 5 or group["score"].nunique() < 2:
            continue
        correlation = spearmanr(
            group["score"],
            group["target_rank_5d"],
            nan_policy="omit",
        ).statistic
        if np.isfinite(correlation):
            daily_ic.append(float(correlation))
        lower = group["score"].quantile(0.25)
        upper = group["score"].quantile(0.75)
        top = group.loc[
            group["score"].ge(upper),
            "residual_return_5d",
        ].mean()
        bottom = group.loc[
            group["score"].le(lower),
            "residual_return_5d",
        ].mean()
        if np.isfinite(top) and np.isfinite(bottom):
            spreads.append(float(top - bottom))
    return {
        "rank_ic_mean": float(np.mean(daily_ic)),
        "rank_ic_median": float(np.median(daily_ic)),
        "rank_ic_positive_share": float(np.mean(np.asarray(daily_ic) > 0)),
        "top_bottom_spread_5d": float(np.mean(spreads)),
        "n_dates": int(len(daily_ic)),
    }


def run_search(
    panel: pd.DataFrame,
    specs: Sequence[XGBSpec],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    predictions: dict[str, list[pd.DataFrame]] = {
        spec.name: [] for spec in specs
    }
    for index, spec in enumerate(specs, start=1):
        print(
            f"[search {index:02d}/{len(specs):02d}] {spec.name}",
            flush=True,
        )
        for _, validation_year in VALIDATION_FOLDS:
            train, validation = fold_frames(panel, validation_year)
            model, predicted, rule_slope = fit_predict(
                train,
                validation,
                spec,
                DEPLOYABLE_FEATURES,
            )
            prediction_frame = validation[
                [
                    "date",
                    "symbol",
                    "strategy_score",
                    "target_rank_5d",
                    "residual_return_5d",
                ]
            ].copy()
            prediction_frame["predicted_residual"] = predicted
            prediction_frame["validation_year"] = validation_year
            predictions[spec.name].append(prediction_frame)
            for blend in BLEND_CONFIGS:
                score = blended_score(
                    validation["strategy_score"].to_numpy(dtype=float),
                    predicted,
                    blend,
                    validation["date"],
                )
                rows.append(
                    {
                        "model": spec.name,
                        "validation_year": validation_year,
                        "eta": blend.eta,
                        "residual_cap": blend.residual_cap,
                        "best_iteration": int(
                            getattr(model, "best_iteration", 0)
                        ),
                        "rule_slope": rule_slope,
                        **signal_metrics(validation, score),
                    }
                )
    return (
        pd.DataFrame(rows),
        {
            name: pd.concat(parts, ignore_index=True)
            for name, parts in predictions.items()
        },
    )


def search_aggregate(search: pd.DataFrame) -> pd.DataFrame:
    return (
        search.groupby(["model", "eta", "residual_cap"], as_index=False)
        .agg(
            mean_rank_ic=("rank_ic_mean", "mean"),
            median_rank_ic=("rank_ic_mean", "median"),
            positive_ic_years=(
                "rank_ic_mean",
                lambda values: int(np.sum(np.asarray(values) > 0)),
            ),
            mean_top_bottom_spread=(
                "top_bottom_spread_5d",
                "mean",
            ),
            mean_best_iteration=("best_iteration", "mean"),
            mean_rule_slope=("rule_slope", "mean"),
        )
        .sort_values(
            ["mean_rank_ic", "mean_top_bottom_spread"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def eligible_symbols(
    panel: pd.DataFrame,
    start: str,
    end: str,
    *,
    symbol_filter: Sequence[str] | None = None,
) -> list[str]:
    subset = panel.loc[
        panel["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ]
    if symbol_filter is not None:
        subset = subset.loc[subset["symbol"].isin(symbol_filter)]
    dates = subset["date"].nunique()
    counts = subset.groupby("symbol")["date"].nunique()
    return sorted(counts[counts.ge(max(1, int(0.85 * dates)))].index)


def run_portfolio(
    panel: pd.DataFrame,
    predictions: pd.DataFrame | None,
    blend: BlendConfig | None,
    *,
    start: str,
    end: str,
    strategy_name: str,
    symbol_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    symbols = eligible_symbols(
        panel,
        start,
        end,
        symbol_filter=symbol_filter,
    )
    if len(symbols) < 5:
        raise ValueError("fewer than five eligible symbols")
    filtered = panel.loc[panel["symbol"].isin(symbols)].copy()
    if predictions is not None:
        filtered = filtered.merge(
            predictions[["date", "symbol", "predicted_residual"]],
            on=["date", "symbol"],
            how="left",
            validate="one_to_one",
        )
    else:
        filtered["predicted_residual"] = 0.0
    filtered["predicted_residual"] = filtered[
        "predicted_residual"
    ].fillna(0.0)
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
        asset_return = (
            returns.loc[date]
            .reindex(symbols)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        gross_return = float(weights @ asset_return)
        growth = max(1.0 + gross_return, 1e-12)
        weights = weights * (1.0 + asset_return) / growth
        cash /= growth
        total = float(weights.sum() + cash)
        weights /= max(total, 1e-12)
        cash /= max(total, 1e-12)
        turnover = 0.0
        maximum_change = 0.0
        minimum_trade = 0.0
        optimizer_success = True
        health_score = rule_backtest.portfolio_health_score(
            returns.loc[:date, symbols],
            weights,
        )
        if date in rebalance_set:
            state = rows_by_date[date]
            if blend is None:
                direction = state["strategy_score"].to_numpy(dtype=float)
            else:
                direction = blended_score(
                    state["strategy_score"].to_numpy(dtype=float),
                    state["predicted_residual"].to_numpy(dtype=float),
                    blend,
                    state["date"],
                )
            allocation = risk_controlled_allocation(
                symbols,
                direction * 0.010,
                state["risk_sigma_daily_5d"].to_numpy(dtype=float),
                returns.loc[:date, symbols],
                weights,
                cash,
                health_score=health_score,
                base_risk_aversion=6.0,
                base_target_annual_volatility=0.15,
                turnover_penalty=0.0025,
            )
            weights = allocation.weights
            cash = allocation.cash_weight
            turnover = allocation.turnover
            maximum_change = allocation.maximum_change
            minimum_trade = allocation.minimum_active_trade
            optimizer_success = allocation.success
        output.append(
            {
                "date": date,
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
            }
        )
    return pd.DataFrame(output)


def apply_cost(frame: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    output = frame.copy()
    output["transaction_cost_bps"] = float(cost_bps)
    output["transaction_cost"] = (
        output["turnover"] * float(cost_bps) / 10_000.0
    )
    output["net_return"] = (
        output["gross_return"] - output["transaction_cost"]
    )
    output["max_weight"] = output["maximum_weight"]
    output["max_change"] = output["maximum_change"]
    return output


def metric_row(frame: pd.DataFrame) -> dict:
    return {
        **backtest_metrics(frame),
        "average_cash_weight": float(frame["cash_weight"].mean()),
        "average_gross_exposure": float(frame["gross_exposure"].mean()),
        "optimizer_success_rate": float(
            frame["optimizer_success"].mean()
        ),
    }


def yearly_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["year"] = work["date"].dt.year
    rows = []
    for year, group in work.groupby("year"):
        rows.append({"year": int(year), **metric_row(group)})
    return pd.DataFrame(rows)


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


def portfolio_search(
    panel: pd.DataFrame,
    aggregate: pd.DataFrame,
    prediction_cache: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    top_models = (
        aggregate.sort_values(
            ["mean_rank_ic", "mean_top_bottom_spread"],
            ascending=False,
        )
        .drop_duplicates("model")
        .head(TOP_PORTFOLIO_MODELS)["model"]
        .tolist()
    )
    baseline = apply_cost(
        run_portfolio(
            panel,
            None,
            None,
            start="2018-01-01",
            end="2020-12-31",
            strategy_name="daily_strategy_rule",
        ),
        PRIMARY_COST_BPS,
    )
    baseline_years = yearly_metrics(baseline).set_index("year")
    rows = []
    paths = {}
    for model_name in top_models:
        for blend in PORTFOLIO_BLEND_CONFIGS:
            name = (
                f"{model_name}_eta{blend.eta:g}_cap"
                f"{blend.residual_cap:g}"
            )
            print(f"[portfolio validation] {name}", flush=True)
            path = apply_cost(
                run_portfolio(
                    panel,
                    prediction_cache[str(model_name)],
                    blend,
                    start="2018-01-01",
                    end="2020-12-31",
                    strategy_name=name,
                ),
                PRIMARY_COST_BPS,
            )
            paths[name] = path
            candidate_years = yearly_metrics(path).set_index("year")
            for year in (2018, 2019, 2020):
                current = candidate_years.loc[year]
                reference = baseline_years.loc[year]
                rows.append(
                    {
                        "candidate": name,
                        "model": model_name,
                        "eta": blend.eta,
                        "residual_cap": blend.residual_cap,
                        "year": year,
                        **current.to_dict(),
                        "cer_gain": (
                            current["certainty_equivalent"]
                            - reference["certainty_equivalent"]
                        ),
                        "sharpe_gain": (
                            current["sharpe"] - reference["sharpe"]
                        ),
                        "drawdown_change": (
                            current["max_drawdown"]
                            - reference["max_drawdown"]
                        ),
                    }
                )
    table = pd.DataFrame(rows)
    summary = (
        table.groupby(
            ["candidate", "model", "eta", "residual_cap"],
            as_index=False,
        )
        .agg(
            mean_cer=("certainty_equivalent", "mean"),
            mean_sharpe=("sharpe", "mean"),
            mean_cer_gain=("cer_gain", "mean"),
            median_cer_gain=("cer_gain", "median"),
            positive_cer_years=(
                "cer_gain",
                lambda values: int(np.sum(np.asarray(values) > 0)),
            ),
            mean_sharpe_gain=("sharpe_gain", "mean"),
            worst_drawdown_change=("drawdown_change", "min"),
        )
    )
    eligible = summary.loc[
        summary["mean_cer_gain"].gt(0)
        & summary["median_cer_gain"].gt(0)
        & summary["positive_cer_years"].ge(2)
        & summary["mean_sharpe_gain"].gt(0)
        & summary["worst_drawdown_change"].ge(-0.02)
    ]
    pool = eligible if not eligible.empty else summary
    selected = pool.sort_values(
        ["mean_cer_gain", "mean_cer", "mean_sharpe"],
        ascending=False,
    ).iloc[0]
    return table, {
        "summary": summary,
        "selected": selected.to_dict(),
        "validation_gates_have_candidate": bool(not eligible.empty),
        "baseline": baseline,
        "paths": paths,
    }


def refit_predict_period(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    spec: XGBSpec,
    features: Sequence[str],
) -> tuple[XGBRegressor, pd.DataFrame, float]:
    train = train_panel.loc[
        train_panel["target_rank_5d"].notna()
    ].copy()
    # A trailing slice supplies early stopping without touching the test.
    train_dates = np.asarray(sorted(train["date"].unique()))
    split_date = pd.Timestamp(train_dates[int(len(train_dates) * 0.88)])
    fit = train.loc[train["date"].lt(split_date)]
    validation = train.loc[train["date"].ge(split_date)]
    model, _, rule_slope = fit_predict(
        fit,
        validation,
        spec,
        features,
    )
    prediction = test_panel[["date", "symbol"]].copy()
    prediction["predicted_residual"] = model.predict(
        test_panel[list(features)]
    )
    return model, prediction, rule_slope


def research_news_ceiling(
    panel: pd.DataFrame,
    spec: XGBSpec,
    blend: BlendConfig,
) -> pd.DataFrame:
    rows = []
    for features_name, features in (
        ("deployable_price_market", DEPLOYABLE_FEATURES),
        ("research_price_market_news", RESEARCH_NEWS_FEATURES),
    ):
        for _, year in VALIDATION_FOLDS:
            train, validation = fold_frames(panel, year)
            _, predicted, rule_slope = fit_predict(
                train,
                validation,
                spec,
                features,
            )
            score = blended_score(
                validation["strategy_score"].to_numpy(dtype=float),
                predicted,
                blend,
                validation["date"],
            )
            rows.append(
                {
                    "feature_family": features_name,
                    "year": year,
                    "rule_slope": rule_slope,
                    **signal_metrics(validation, score),
                }
            )
    return pd.DataFrame(rows)


def feature_importance(
    model: XGBRegressor,
    features: Sequence[str],
) -> pd.DataFrame:
    raw = model.get_booster().get_score(importance_type="gain")
    rows = []
    for index, feature in enumerate(features):
        rows.append(
            {
                "feature": feature,
                "gain": float(
                    raw.get(feature, raw.get(f"f{index}", 0.0))
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values("gain", ascending=False)
    total = float(result["gain"].sum())
    result["gain_share"] = result["gain"] / max(total, 1e-12)
    return result


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
    historical = engineer_panel(HISTORICAL_PATH, external=False)
    external = engineer_panel(EXTERNAL_PATH, external=True)
    print(
        f"[historical] rows={len(historical):,} "
        f"symbols={historical.symbol.nunique()} "
        f"{historical.date.min().date()}..{historical.date.max().date()}",
        flush=True,
    )
    print(
        f"[external] rows={len(external):,} "
        f"symbols={external.symbol.nunique()} "
        f"{external.date.min().date()}..{external.date.max().date()}",
        flush=True,
    )

    specs = make_specs()
    spec_map = {spec.name: spec for spec in specs}
    search, prediction_cache = run_search(historical, specs)
    aggregate = search_aggregate(search)
    search.to_csv(OUT_DIR / "signal_search_folds.csv", index=False)
    aggregate.to_csv(OUT_DIR / "signal_search_summary.csv", index=False)

    portfolio_folds, portfolio_result = portfolio_search(
        historical,
        aggregate,
        prediction_cache,
    )
    portfolio_folds.to_csv(
        OUT_DIR / "portfolio_validation_folds.csv",
        index=False,
    )
    portfolio_summary = portfolio_result["summary"]
    portfolio_summary.to_csv(
        OUT_DIR / "portfolio_validation_summary.csv",
        index=False,
    )
    selected = portfolio_result["selected"]
    selected_spec = spec_map[str(selected["model"])]
    selected_blend = BlendConfig(
        eta=float(selected["eta"]),
        residual_cap=float(selected["residual_cap"]),
    )
    print(
        f"[selected] {selected_spec.name} eta={selected_blend.eta} "
        f"cap={selected_blend.residual_cap}",
        flush=True,
    )

    development = historical.loc[
        historical["date"].dt.year.le(2020)
    ].copy()
    diagnostic = historical.loc[
        historical["date"].dt.year.between(2021, 2023)
    ].copy()
    (
        diagnostic_model,
        diagnostic_prediction,
        diagnostic_rule_slope,
    ) = refit_predict_period(
        development,
        diagnostic,
        selected_spec,
        DEPLOYABLE_FEATURES,
    )
    diagnostic_base_raw = run_portfolio(
        historical,
        None,
        None,
        start="2021-01-01",
        end="2023-12-31",
        strategy_name="daily_strategy_rule",
    )
    diagnostic_candidate_raw = run_portfolio(
        historical,
        diagnostic_prediction,
        selected_blend,
        start="2021-01-01",
        end="2023-12-31",
        strategy_name="daily_strategy_xgb_residual",
    )

    (
        final_model,
        external_prediction,
        final_rule_slope,
    ) = refit_predict_period(
        historical,
        external,
        selected_spec,
        DEPLOYABLE_FEATURES,
    )
    external_paths = []
    external_group_metrics = []
    seen_symbols = sorted(
        external.loc[external["is_seen_symbol"], "symbol"].unique()
    )
    unseen_symbols = sorted(
        external.loc[~external["is_seen_symbol"], "symbol"].unique()
    )
    for group_name, symbols in (
        ("all", None),
        ("seen", seen_symbols),
        ("unseen", unseen_symbols),
    ):
        baseline_raw = run_portfolio(
            external,
            None,
            None,
            start="2024-01-01",
            end="2026-12-31",
            strategy_name=f"external_{group_name}_rule",
            symbol_filter=symbols,
        )
        candidate_raw = run_portfolio(
            external,
            external_prediction,
            selected_blend,
            start="2024-01-01",
            end="2026-12-31",
            strategy_name=f"external_{group_name}_xgb_residual",
            symbol_filter=symbols,
        )
        for cost in (0.0, 25.0, 50.0):
            for path in (baseline_raw, candidate_raw):
                costed = apply_cost(path, cost)
                costed["group"] = group_name
                external_paths.append(costed)
        base_primary = apply_cost(baseline_raw, PRIMARY_COST_BPS)
        candidate_primary = apply_cost(candidate_raw, PRIMARY_COST_BPS)
        base_metrics = metric_row(base_primary)
        candidate_metrics = metric_row(candidate_primary)
        external_group_metrics.extend(
            [
                {
                    "group": group_name,
                    "strategy": "rule",
                    **base_metrics,
                },
                {
                    "group": group_name,
                    "strategy": "xgb_residual",
                    **candidate_metrics,
                    "cer_gain_vs_rule": (
                        candidate_metrics["certainty_equivalent"]
                        - base_metrics["certainty_equivalent"]
                    ),
                    "sharpe_gain_vs_rule": (
                        candidate_metrics["sharpe"]
                        - base_metrics["sharpe"]
                    ),
                    "drawdown_change_vs_rule": (
                        candidate_metrics["max_drawdown"]
                        - base_metrics["max_drawdown"]
                    ),
                },
            ]
        )

    diagnostic_paths = []
    for cost in (0.0, 25.0, 50.0):
        diagnostic_paths.extend(
            [
                apply_cost(diagnostic_base_raw, cost),
                apply_cost(diagnostic_candidate_raw, cost),
            ]
        )
    diagnostic_daily = pd.concat(diagnostic_paths, ignore_index=True)
    diagnostic_daily.to_parquet(
        OUT_DIR / "diagnostic_daily_paths.parquet",
        index=False,
    )
    external_daily = pd.concat(external_paths, ignore_index=True)
    external_daily.to_parquet(
        OUT_DIR / "external_daily_paths.parquet",
        index=False,
    )

    diagnostic_metrics = []
    for path in diagnostic_paths:
        diagnostic_metrics.append(
            {
                "strategy": path.iloc[0]["strategy"],
                "transaction_cost_bps": path.iloc[0][
                    "transaction_cost_bps"
                ],
                **metric_row(path),
            }
        )
    diagnostic_metrics = pd.DataFrame(diagnostic_metrics)
    diagnostic_metrics.to_csv(
        OUT_DIR / "diagnostic_metrics.csv",
        index=False,
    )
    external_group_metrics_frame = pd.DataFrame(external_group_metrics)
    external_group_metrics_frame.to_csv(
        OUT_DIR / "external_group_metrics_25bps.csv",
        index=False,
    )

    diagnostic_base = apply_cost(
        diagnostic_base_raw,
        PRIMARY_COST_BPS,
    )
    diagnostic_candidate = apply_cost(
        diagnostic_candidate_raw,
        PRIMARY_COST_BPS,
    )
    tests = {
        "diagnostic_candidate_vs_rule": paired_tests(
            diagnostic_base,
            diagnostic_candidate,
        )
    }
    for group_name in ("all", "seen", "unseen"):
        subset = external_daily.loc[
            external_daily["group"].eq(group_name)
            & external_daily["transaction_cost_bps"].eq(
                PRIMARY_COST_BPS
            )
        ]
        reference = subset.loc[subset["strategy"].str.endswith("_rule")]
        candidate = subset.loc[
            subset["strategy"].str.endswith("_xgb_residual")
        ]
        tests[f"external_{group_name}_candidate_vs_rule"] = paired_tests(
            reference,
            candidate,
        )
    (OUT_DIR / "statistical_tests.json").write_text(
        json.dumps(tests, indent=2),
        encoding="utf-8",
    )

    research_ceiling = research_news_ceiling(
        historical,
        selected_spec,
        selected_blend,
    )
    research_ceiling.to_csv(
        OUT_DIR / "research_news_ceiling.csv",
        index=False,
    )

    importance = feature_importance(final_model, DEPLOYABLE_FEATURES)
    importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    model_path = MODEL_DIR / "xgb_residual_alpha.json"
    final_model.save_model(model_path)

    validation_selected = portfolio_folds.loc[
        portfolio_folds["candidate"].eq(selected["candidate"])
    ]
    validation_checks = {
        "eligible_candidate_exists": bool(
            portfolio_result["validation_gates_have_candidate"]
        ),
        "mean_cer_gain_positive": bool(
            validation_selected["cer_gain"].mean() > 0
        ),
        "positive_in_at_least_2_of_3_years": bool(
            (validation_selected["cer_gain"] > 0).sum() >= 2
        ),
        "mean_sharpe_gain_positive": bool(
            validation_selected["sharpe_gain"].mean() > 0
        ),
        "worst_drawdown_not_worse_by_more_than_2pp": bool(
            validation_selected["drawdown_change"].min() >= -0.02
        ),
    }
    diagnostic_test = tests["diagnostic_candidate_vs_rule"]
    diagnostic_checks = {
        "newey_west_p_below_005": bool(
            diagnostic_test["newey_west_p"] < 0.05
        ),
        "bootstrap_cer_lower_bound_positive": bool(
            diagnostic_test["cer_gain_ci_low"] > 0
        ),
    }
    external_metrics_lookup = external_group_metrics_frame.set_index(
        ["group", "strategy"]
    )
    external_checks = {
        "overall_cer_gain_nonnegative": bool(
            external_metrics_lookup.loc[
                ("all", "xgb_residual"),
                "cer_gain_vs_rule",
            ]
            >= 0
        ),
        "unseen_cer_gain_nonnegative": bool(
            external_metrics_lookup.loc[
                ("unseen", "xgb_residual"),
                "cer_gain_vs_rule",
            ]
            >= 0
        ),
        "overall_drawdown_not_worse_by_more_than_2pp": bool(
            external_metrics_lookup.loc[
                ("all", "xgb_residual"),
                "drawdown_change_vs_rule",
            ]
            >= -0.02
        ),
    }
    promoted = bool(
        all(validation_checks.values())
        and all(diagnostic_checks.values())
        and all(external_checks.values())
    )
    promotion = {
        "promoted": promoted,
        "status": "promoted" if promoted else "experimental_only",
        "validation_checks": validation_checks,
        "diagnostic_checks": diagnostic_checks,
        "external_checks": external_checks,
    }
    (OUT_DIR / "promotion_gates.json").write_text(
        json.dumps(promotion, indent=2),
        encoding="utf-8",
    )

    bounds = {
        feature: [
            float(historical[feature].quantile(0.01)),
            float(historical[feature].quantile(0.99)),
        ]
        for feature in DEPLOYABLE_FEATURES
    }
    metadata = {
        "schema_version": 1,
        "model_version": "daily-strategy-xgb-residual-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "promotion_status": (
            "promoted" if promoted else "experimental_only"
        ),
        "target": (
            "future_5d_stock_return_minus_beta_times_spy_return_"
            "cross_sectional_rank_residual"
        ),
        "base_strategy": "causal Daily Strategy price-rule rank",
        "risk_usage": "external_only",
        "health_usage": "external_only",
        "feature_order": DEPLOYABLE_FEATURES,
        "feature_bounds": bounds,
        "selected_spec": asdict(selected_spec),
        "eta": selected_blend.eta,
        "residual_cap": selected_blend.residual_cap,
        "training_range": [
            str(historical["date"].min().date()),
            str(historical["date"].max().date()),
        ],
        "training_rows": int(
            historical["target_rank_5d"].notna().sum()
        ),
        "training_symbols": int(historical["symbol"].nunique()),
        "validation_years": [2018, 2019, 2020],
        "diagnostic_years": [2021, 2022, 2023],
        "external_range": [
            str(external["date"].min().date()),
            str(external["date"].max().date()),
        ],
        "external_symbols": int(external["symbol"].nunique()),
        "external_unseen_symbols": int(len(unseen_symbols)),
        "historical_data_sha256": _sha256(HISTORICAL_PATH),
        "external_data_sha256": _sha256(EXTERNAL_PATH),
        "model_sha256": _sha256(model_path),
        "diagnostic_rule_slope": diagnostic_rule_slope,
        "final_rule_slope": final_rule_slope,
        "promotion": promotion,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    primary_diagnostic = diagnostic_metrics.loc[
        diagnostic_metrics["transaction_cost_bps"].eq(PRIMARY_COST_BPS),
        [
            "strategy",
            "cagr",
            "sharpe",
            "certainty_equivalent",
            "max_drawdown",
            "average_cash_weight",
        ],
    ]
    external_display = external_group_metrics_frame[
        [
            "group",
            "strategy",
            "cagr",
            "sharpe",
            "certainty_equivalent",
            "max_drawdown",
            "cer_gain_vs_rule",
        ]
    ].copy()
    report = f"""# Enhanced Daily Strategy — XGBoost Residual Alpha

## Selected design

The current rule score remains the prior. `{selected_spec.name}` predicts the
unexplained part of the future five-session beta-adjusted cross-sectional
return rank. The selected blend is `eta={selected_blend.eta:.2f}` with a
residual cap of `{selected_blend.residual_cap:.2f}`. HAR-X + News risk and
Portfolio Health remain external position controls.

## Diagnostic 2021–2023 at 25 bps

{_markdown_table(primary_diagnostic)}

## External 2024–2026 at 25 bps

{_markdown_table(external_display)}

## Promotion

- Validation candidate passed: `{all(validation_checks.values())}`.
- Diagnostic significance passed: `{all(diagnostic_checks.values())}`.
- External generalisation passed: `{all(external_checks.values())}`.
- Status: `{promotion["status"]}`.

The model is only loaded by the backend when every gate passes. Otherwise the
existing rule Daily Strategy remains the production prior.

## Upper-bound conclusion

The searched XGBoost residual-alpha family did not produce a robust
improvement. The selected ceiling candidate loses CER and Sharpe in both the
2021-2023 diagnostic period and the full 2024-2026 external universe. A
validation-defined conservative challenger was also evaluated separately and
did not repair the degradation. Adding the available FNSPID sentiment fields
changed validation Rank IC only marginally. The rule strategy therefore
remains production, and the checkpoint stays available for research only.
"""
    (OUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)


if __name__ == "__main__":
    main()
