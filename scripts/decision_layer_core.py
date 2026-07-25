"""Reusable model, optimisation, and backtest primitives for the decision layer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PRICE_FEATURES = [
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
TARGET = "excess_ret_20d"
HORIZON = 20
RANDOM_SEED = 20260724
EPS = 1e-10

MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "elastic_net_a0005",
        "kind": "elastic_net",
        "alpha": 0.0005,
        "l1_ratio": 0.10,
    },
    {
        "name": "elastic_net_a002",
        "kind": "elastic_net",
        "alpha": 0.002,
        "l1_ratio": 0.10,
    },
    {
        "name": "xgb_shallow",
        "kind": "xgboost",
        "max_depth": 2,
        "learning_rate": 0.025,
        "n_estimators": 500,
        "min_child_weight": 32,
        "subsample": 0.80,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 20.0,
    },
    {
        "name": "xgb_medium",
        "kind": "xgboost",
        "max_depth": 3,
        "learning_rate": 0.025,
        "n_estimators": 500,
        "min_child_weight": 32,
        "subsample": 0.80,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.10,
        "reg_lambda": 30.0,
    },
    {
        "name": "quantile_hgb",
        "kind": "quantile_hgb",
        "max_depth": 3,
        "learning_rate": 0.04,
        "max_iter": 300,
        "min_samples_leaf": 40,
        "l2_regularization": 2.0,
    },
)


@dataclass(frozen=True)
class Fold:
    year: int
    train_cutoff: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_mask: np.ndarray
    test_mask: np.ndarray


@dataclass
class OptimizerResult:
    weights: np.ndarray
    success: bool
    message: str
    turnover: float
    maximum_position: float
    maximum_change: float
    minimum_active_trade: float


def model_spec(name: str) -> dict[str, Any]:
    for spec in MODEL_SPECS:
        if spec["name"] == name:
            return dict(spec)
    raise KeyError(name)


def time_fold(
    frame: pd.DataFrame,
    year: int,
    embargo_sessions: int = HORIZON,
) -> Fold:
    dates = np.sort(frame["date"].unique())
    test_mask = frame["date"].dt.year.eq(year).to_numpy()
    if not test_mask.any():
        raise ValueError(f"no observations for {year}")
    test_start = pd.Timestamp(frame.loc[test_mask, "date"].min())
    position = int(np.searchsorted(dates, np.datetime64(test_start)))
    cutoff_position = position - embargo_sessions - 1
    if cutoff_position < 0:
        raise ValueError(f"not enough history before {year}")
    train_cutoff = pd.Timestamp(dates[cutoff_position])
    train_mask = frame["date"].le(train_cutoff).to_numpy()
    return Fold(
        year=year,
        train_cutoff=train_cutoff,
        test_start=test_start,
        test_end=pd.Timestamp(frame.loc[test_mask, "date"].max()),
        train_mask=train_mask,
        test_mask=test_mask,
    )


def symbol_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("symbol")["symbol"].transform("size").to_numpy()
    weights = 1.0 / np.maximum(count, 1)
    return weights / weights.mean()


def _model_target(
    train: pd.DataFrame,
    spec: Mapping[str, Any],
) -> np.ndarray:
    target_column = str(spec.get("target_column", TARGET))
    target = train[target_column].to_numpy(dtype=float)
    target = target - float(spec.get("target_offset", 0.0))
    if target_column != TARGET:
        return target
    lower, upper = np.quantile(target, [0.005, 0.995])
    return np.clip(target, lower, upper)


def fit_estimator(
    train: pd.DataFrame,
    spec: Mapping[str, Any],
):
    name = str(spec["name"])
    kind = str(spec["kind"])
    values = train[PRICE_FEATURES]
    target = _model_target(train, spec)
    sample_weight = symbol_equal_weights(train)

    if kind == "elastic_net":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=float(spec["alpha"]),
                        l1_ratio=float(spec["l1_ratio"]),
                        max_iter=20_000,
                        tol=1e-7,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        estimator.fit(
            values,
            target,
            model__sample_weight=sample_weight,
        )
    elif kind == "xgboost":
        estimator = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            max_depth=int(spec["max_depth"]),
            learning_rate=float(spec["learning_rate"]),
            n_estimators=int(spec["n_estimators"]),
            min_child_weight=float(spec["min_child_weight"]),
            subsample=float(spec["subsample"]),
            colsample_bytree=float(spec["colsample_bytree"]),
            reg_alpha=float(spec["reg_alpha"]),
            reg_lambda=float(spec["reg_lambda"]),
            random_state=RANDOM_SEED,
            n_jobs=4,
        )
        estimator.fit(values, target, sample_weight=sample_weight)
    elif kind == "quantile_hgb":
        estimator = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.5,
            max_depth=int(spec["max_depth"]),
            learning_rate=float(spec["learning_rate"]),
            max_iter=int(spec["max_iter"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            l2_regularization=float(spec["l2_regularization"]),
            early_stopping=False,
            random_state=RANDOM_SEED,
        )
        estimator.fit(values, target, sample_weight=sample_weight)
    else:
        raise ValueError(f"unsupported model kind for {name}: {kind}")
    return estimator


def bounded_prediction(
    estimator,
    frame: pd.DataFrame,
    bound: float = 0.25,
) -> np.ndarray:
    prediction = np.asarray(
        estimator.predict(frame[PRICE_FEATURES]),
        dtype=float,
    )
    return np.clip(prediction, -bound, bound)


def walk_forward_predictions(
    frame: pd.DataFrame,
    years: Sequence[int],
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    target_column = str(spec.get("target_column", TARGET))
    for year in years:
        fold = time_fold(frame, year)
        train = frame.loc[fold.train_mask].dropna(
            subset=PRICE_FEATURES + [TARGET, target_column]
        )
        test = frame.loc[fold.test_mask].dropna(
            subset=PRICE_FEATURES + [TARGET, target_column]
        ).copy()
        estimator = fit_estimator(train, spec)
        test["prediction"] = bounded_prediction(
            estimator,
            test,
            bound=float(spec.get("prediction_bound", 0.25)),
        )
        test["model"] = spec["name"]
        test["test_year"] = year
        test["train_cutoff"] = fold.train_cutoff
        rows.append(
            test[
                [
                    "date",
                    "symbol",
                    TARGET,
                    "prediction",
                    "model",
                    "test_year",
                    "train_cutoff",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    actual = predictions[TARGET].to_numpy(dtype=float)
    forecast = predictions["prediction"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((actual - forecast) ** 2)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1 - np.sum((actual - forecast) ** 2) / denominator)

    daily_ic: list[float] = []
    spreads: list[float] = []
    for _, group in predictions.groupby("date"):
        if len(group) < 5 or group["prediction"].nunique() < 2:
            continue
        correlation = spearmanr(
            group["prediction"],
            group[TARGET],
            nan_policy="omit",
        ).statistic
        if np.isfinite(correlation):
            daily_ic.append(float(correlation))
        ranked = group.assign(
            bucket=pd.qcut(
                group["prediction"],
                q=5,
                labels=False,
                duplicates="drop",
            )
        )
        if ranked["bucket"].nunique() >= 2:
            means = ranked.groupby("bucket")[TARGET].mean()
            spreads.append(float(means.iloc[-1] - means.iloc[0]))

    return {
        "rmse": rmse,
        "r2": r2,
        "rank_ic_mean": float(np.mean(daily_ic)),
        "rank_ic_median": float(np.median(daily_ic)),
        "rank_ic_positive_share": float(np.mean(np.asarray(daily_ic) > 0)),
        "top_bottom_20d_spread": float(np.mean(spreads)),
        "n": int(len(predictions)),
        "n_dates": int(predictions["date"].nunique()),
    }


def _correlation_matrix(
    return_history: pd.DataFrame,
    symbols: Sequence[str],
) -> np.ndarray:
    history = return_history.reindex(columns=symbols).tail(126)
    correlation = history.corr(min_periods=40).to_numpy(dtype=float)
    correlation = np.where(np.isfinite(correlation), correlation, 0.0)
    np.fill_diagonal(correlation, 1.0)
    # Numerical shrinkage makes the matrix stable in short samples.
    correlation = 0.90 * correlation + 0.10 * np.eye(len(symbols))
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1e-5)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    return repaired / np.outer(scale, scale)


def optimize_weights(
    symbols: Sequence[str],
    expected_return_20d: np.ndarray,
    sigma_daily: np.ndarray,
    return_history: pd.DataFrame,
    previous_weights: np.ndarray,
    *,
    risk_aversion: float = 6.0,
    turnover_penalty: float = 0.0025,
    max_position: float = 0.20,
    max_change: float = 0.05,
    min_trade: float = 0.01,
) -> OptimizerResult:
    """Solve a long-only, fully-invested, cost-aware allocation problem."""
    symbols = list(symbols)
    expected = np.nan_to_num(
        np.asarray(expected_return_20d, dtype=float),
        nan=0.0,
    )
    sigma = np.asarray(sigma_daily, dtype=float)
    fallback_sigma = np.nanmedian(sigma[np.isfinite(sigma) & (sigma > 0)])
    if not np.isfinite(fallback_sigma):
        fallback_sigma = 0.02
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback_sigma)
    previous = np.asarray(previous_weights, dtype=float)
    previous = np.maximum(previous, 0.0)
    previous = previous / max(previous.sum(), EPS)

    correlation = _correlation_matrix(return_history, symbols)
    covariance = (
        np.diag(sigma)
        @ correlation
        @ np.diag(sigma)
        * HORIZON
    )
    horizon_sigma = sigma * math.sqrt(HORIZON)
    expected = np.clip(expected, -horizon_sigma, horizon_sigma)

    lower = np.maximum(0.0, previous - max_change)
    upper = np.minimum(max_position, previous + max_change)
    # Existing overweight positions may be held or reduced, never increased.
    upper = np.where(previous > max_position, previous, upper)
    if lower.sum() > 1 + 1e-8 or upper.sum() < 1 - 1e-8:
        lower = np.zeros_like(previous)
        upper = np.maximum(upper, 1.0 / len(previous))
        upper = np.minimum(np.maximum(upper, max_position), 1.0)

    def objective(weights: np.ndarray) -> float:
        delta = weights - previous
        smooth_turnover = np.sum(np.sqrt(delta * delta + 1e-10))
        return float(
            -(expected @ weights)
            + 0.5 * risk_aversion * (weights @ covariance @ weights)
            + turnover_penalty * smooth_turnover
        )

    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]
    bounds = list(zip(lower, upper))
    result = minimize(
        objective,
        x0=np.clip(previous, lower, upper),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-10},
    )
    weights = result.x if result.success else previous.copy()

    fixed = np.zeros(len(previous), dtype=bool)
    for _ in range(4):
        small = (np.abs(weights - previous) < min_trade - 1e-7) & (
            np.abs(weights - previous) > 1e-5
        )
        new_fixed = small & ~fixed
        if not result.success or not new_fixed.any():
            break
        fixed |= new_fixed
        fixed_lower = lower.copy()
        fixed_upper = upper.copy()
        fixed_lower[fixed] = previous[fixed]
        fixed_upper[fixed] = previous[fixed]
        if fixed_lower.sum() > 1 + 1e-8 or fixed_upper.sum() < 1 - 1e-8:
            break
        second = minimize(
            objective,
            x0=weights,
            method="SLSQP",
            bounds=list(zip(fixed_lower, fixed_upper)),
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if not second.success:
            break
        weights = second.x
        result = second

    weights = np.maximum(weights, 0.0)
    weights = weights / max(weights.sum(), EPS)
    change = weights - previous
    active = np.abs(change) > 1e-5
    return OptimizerResult(
        weights=weights,
        success=bool(result.success),
        message=str(result.message),
        turnover=float(0.5 * np.abs(change).sum()),
        maximum_position=float(weights.max()),
        maximum_change=float(np.abs(change).max()),
        minimum_active_trade=(
            float(np.abs(change[active]).min()) if active.any() else 0.0
        ),
    )


def _strategy_expected_return(
    strategy: str,
    date: pd.Timestamp,
    prediction: pd.DataFrame,
    momentum: pd.DataFrame,
    symbols: Sequence[str],
) -> np.ndarray:
    if strategy == "risk_only":
        return np.zeros(len(symbols), dtype=float)
    if strategy == "momentum_rule":
        values = momentum.reindex(index=[date], columns=symbols).iloc[0]
        values = values.to_numpy(dtype=float)
        center = np.nanmedian(values)
        scale = np.nanstd(values)
        if not np.isfinite(scale) or scale < EPS:
            return np.zeros(len(symbols), dtype=float)
        return np.clip((values - center) / scale, -2, 2) * 0.015
    if strategy == "ml":
        return (
            prediction.reindex(index=[date], columns=symbols)
            .iloc[0]
            .to_numpy(dtype=float)
        )
    raise ValueError(strategy)


def run_backtest(
    full_panel: pd.DataFrame,
    predictions: pd.DataFrame | None,
    *,
    start: str,
    end: str,
    strategy: str,
    transaction_cost_bps: float = 25.0,
    rebalance_sessions: int = 5,
    risk_aversion: float = 6.0,
    turnover_penalty_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest decisions at the close; new weights apply from the next day."""
    panel = full_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    evaluation = panel["date"].between(start_ts, end_ts)
    counts = panel.loc[evaluation].groupby("symbol")["date"].nunique()
    required_dates = max(1, int(panel.loc[evaluation, "date"].nunique() * 0.85))
    symbols = sorted(counts[counts.ge(required_dates)].index)
    if len(symbols) < 5:
        raise ValueError("fewer than five sufficiently complete symbols")

    returns = panel.pivot(index="date", columns="symbol", values="ret_1d")
    returns = returns.reindex(columns=symbols).sort_index()
    momentum = panel.pivot(index="date", columns="symbol", values="mom_20d")
    risk = panel.pivot(
        index="date",
        columns="symbol",
        values="risk_sigma_daily_5d",
    )
    if predictions is not None:
        prediction = predictions.pivot(
            index="date",
            columns="symbol",
            values="prediction",
        )
    else:
        prediction = pd.DataFrame(index=returns.index, columns=symbols)

    dates = returns.index[
        returns.index.to_series().between(start_ts, end_ts)
    ]
    weights = np.full(len(symbols), 1.0 / len(symbols))
    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    cost_rate = transaction_cost_bps / 10_000.0
    rebalance_set = set(dates[::rebalance_sessions])

    for date in dates:
        daily_asset_return = (
            returns.loc[date].reindex(symbols).fillna(0.0).to_numpy(dtype=float)
        )
        gross_return = float(weights @ daily_asset_return)
        turnover = 0.0
        success = True
        message = "no rebalance"
        maximum_change = 0.0
        minimum_trade = 0.0

        if strategy != "equal_weight" and date in rebalance_set:
            expected = _strategy_expected_return(
                strategy,
                date,
                prediction,
                momentum,
                symbols,
            )
            sigma = (
                risk.reindex(index=[date], columns=symbols)
                .iloc[0]
                .to_numpy(dtype=float)
            )
            result = optimize_weights(
                symbols,
                expected,
                sigma,
                returns.loc[:date],
                weights,
                risk_aversion=risk_aversion,
                turnover_penalty=(
                    cost_rate * turnover_penalty_multiplier
                ),
            )
            target = result.weights
            turnover = result.turnover
            success = result.success
            message = result.message
            maximum_change = result.maximum_change
            minimum_trade = result.minimum_active_trade
            for symbol, before, after in zip(symbols, weights, target):
                if abs(after - before) >= 0.005:
                    weight_rows.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "weight_before": before,
                            "weight_after": after,
                            "weight_change": after - before,
                            "strategy": strategy,
                        }
                    )
            weights = target

        cost = turnover * cost_rate
        net_return = gross_return - cost
        benchmark_return = float(
            panel.loc[
                panel["date"].eq(date), "benchmark_ret_1d"
            ].dropna().iloc[0]
        )
        rows.append(
            {
                "date": date,
                "strategy": strategy,
                "gross_return": gross_return,
                "transaction_cost": cost,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "turnover": turnover,
                "optimizer_success": success,
                "optimizer_message": message,
                "max_weight": float(weights.max()),
                "max_change": maximum_change,
                "minimum_active_trade": minimum_trade,
                "n_positions": int(np.sum(weights > 1e-4)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def backtest_metrics(
    returns: pd.DataFrame,
    risk_aversion: float = 6.0,
) -> dict[str, float]:
    daily = returns["net_return"].to_numpy(dtype=float)
    benchmark = returns["benchmark_return"].to_numpy(dtype=float)
    wealth = np.cumprod(1 + daily)
    benchmark_wealth = np.cumprod(1 + benchmark)
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / np.maximum(running_max, EPS) - 1
    downside = daily[daily < 0]
    annual_return = float(np.mean(daily) * 252)
    annual_variance = float(np.var(daily, ddof=1) * 252)
    annual_volatility = math.sqrt(max(annual_variance, 0.0))
    sharpe = annual_return / annual_volatility if annual_volatility else 0.0
    downside_volatility = (
        float(np.std(downside, ddof=1) * math.sqrt(252))
        if len(downside) > 1
        else 0.0
    )
    sortino = annual_return / downside_volatility if downside_volatility else 0.0
    tail_count = max(1, int(math.ceil(0.05 * len(daily))))
    expected_shortfall = float(np.sort(daily)[:tail_count].mean())
    certainty_equivalent = annual_return - 0.5 * risk_aversion * annual_variance
    years = max(len(daily) / 252, EPS)
    cagr = float(wealth[-1] ** (1 / years) - 1)
    benchmark_cagr = float(benchmark_wealth[-1] ** (1 / years) - 1)
    return {
        "total_return": float(wealth[-1] - 1),
        "cagr": cagr,
        "benchmark_cagr": benchmark_cagr,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "certainty_equivalent": certainty_equivalent,
        "max_drawdown": float(drawdown.min()),
        "daily_es95": expected_shortfall,
        "average_one_way_turnover": float(returns["turnover"].mean()),
        "annual_turnover": float(returns["turnover"].sum() / years),
        "total_transaction_cost": float(returns["transaction_cost"].sum()),
        "optimizer_success_rate": float(returns["optimizer_success"].mean()),
        "maximum_weight": float(returns["max_weight"].max()),
        "maximum_change": float(returns["max_change"].max()),
        "minimum_active_trade": float(
            returns.loc[
                returns["minimum_active_trade"].gt(0),
                "minimum_active_trade",
            ].min()
            if returns["minimum_active_trade"].gt(0).any()
            else 0.0
        ),
        "n_days": int(len(returns)),
    }


def moving_block_utility_gain(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    reps: int = 1000,
    block_days: int = 20,
    seed: int = RANDOM_SEED,
    risk_aversion: float = 6.0,
) -> tuple[float, float, float]:
    merged = reference[["date", "net_return"]].merge(
        candidate[["date", "net_return"]],
        on="date",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    ref = merged["net_return_reference"].to_numpy(dtype=float)
    cand = merged["net_return_candidate"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    gains: list[float] = []
    maximum_start = max(1, len(merged) - block_days + 1)
    for _ in range(reps):
        index: list[int] = []
        while len(index) < len(merged):
            start = int(rng.integers(0, maximum_start))
            index.extend(range(start, min(start + block_days, len(merged))))
        sample = np.asarray(index[: len(merged)])
        ref_sample = ref[sample]
        cand_sample = cand[sample]
        ref_utility = (
            np.mean(ref_sample) * 252
            - 0.5 * risk_aversion * np.var(ref_sample, ddof=1) * 252
        )
        cand_utility = (
            np.mean(cand_sample) * 252
            - 0.5 * risk_aversion * np.var(cand_sample, ddof=1) * 252
        )
        gains.append(float(cand_utility - ref_utility))
    return tuple(
        float(value)
        for value in np.quantile(gains, [0.025, 0.5, 0.975])
    )
