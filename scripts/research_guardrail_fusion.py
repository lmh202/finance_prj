"""Research a hierarchical Strategy-first, Risk-as-guardrail Fusion policy.

Unlike the production mean-variance objective, this candidate does not charge
every position a permanent covariance penalty. Daily Strategy first supplies a
fully invested target. The optimiser then finds the closest feasible portfolio
subject to a causal, stress-dependent volatility ceiling and the production
position/change constraints. Risk therefore intervenes only when the Strategy
target would exceed the current risk budget.

This is a research script. It does not change the production recommendation
path or promote a candidate automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SCRIPTS = ROOT / "scripts"
for path in (BACKEND, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research_adaptive_fusion import (  # noqa: E402
    Policy,
    build_stress_series,
    cagr,
    maximum_drawdown,
    metrics,
    paired_block_bootstrap,
    run_daily_strategy,
    run_policy,
    select_period,
)
from src.recommendation.gated_news import _repair_correlation  # noqa: E402

TRADING_DAYS = 252
HORIZON = 5
TRANSACTION_COST = 0.0025
HEALTH_SCORE = 60.0
HEALTH_BUDGET_FACTOR = 0.65 + 0.35 * HEALTH_SCORE / 100.0


def strategy_target(
    direction: np.ndarray,
    *,
    max_position: float,
) -> np.ndarray:
    ranks = pd.Series(np.asarray(direction, dtype=float)).rank(
        pct=True,
    ).to_numpy(dtype=float)
    weights = ranks / max(ranks.sum(), 1e-10)
    weights = np.clip(weights, 0.0, max_position)
    return weights / max(weights.sum(), 1e-10)


def portfolio_variance(
    weights: np.ndarray,
    covariance_annual: np.ndarray,
) -> float:
    return float(max(weights @ covariance_annual @ weights, 0.0))


def solve_guardrail(
    target: np.ndarray,
    previous: np.ndarray,
    covariance_annual: np.ndarray,
    volatility_budget: float,
    *,
    max_position: float,
    max_change: float = 0.05,
    min_trade: float = 0.01,
) -> tuple[np.ndarray, bool, float]:
    """Project Strategy target onto risk and production trading constraints."""
    target = np.asarray(target, dtype=float)
    previous = np.asarray(previous, dtype=float)
    lower = np.maximum(0.0, previous - max_change)
    upper = np.minimum(max_position, previous + max_change)

    def variance(weights: np.ndarray) -> float:
        return portfolio_variance(weights, covariance_annual)

    # Find the minimum attainable risk under weekly change bounds. If a sudden
    # shock makes the requested budget temporarily infeasible, use the tightest
    # feasible ceiling instead of failing or violating the trading contract.
    min_variance_result = minimize(
        variance,
        x0=np.clip(previous, lower, upper),
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=[
            {"type": "ineq", "fun": lambda candidate: 1.0 - candidate.sum()}
        ],
        options={"maxiter": 250, "ftol": 1e-12},
    )
    minimum_variance = (
        variance(min_variance_result.x)
        if min_variance_result.success
        else variance(lower)
    )
    effective_budget_squared = max(
        float(volatility_budget) ** 2,
        minimum_variance * (1.0 + 1e-8),
    )

    def objective(weights: np.ndarray) -> float:
        delta_target = weights - target
        delta_trade = weights - previous
        smooth_turnover = np.sqrt(delta_trade * delta_trade + 1e-10).sum()
        return float(
            np.square(delta_target).sum()
            + TRANSACTION_COST * smooth_turnover
        )

    constraints = [
        {"type": "ineq", "fun": lambda candidate: 1.0 - candidate.sum()},
        {
            "type": "ineq",
            "fun": lambda candidate: (
                effective_budget_squared - variance(candidate)
            ),
        },
    ]
    start = (
        previous
        if variance(previous) <= effective_budget_squared
        else min_variance_result.x
    )
    result = minimize(
        objective,
        x0=np.clip(start, lower, upper),
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        options={"maxiter": 350, "ftol": 1e-11},
    )
    weights = result.x if result.success else previous.copy()

    # Enforce the shared minimum trade without silently violating the risk
    # ceiling: fix sub-threshold positions and resolve the remaining names.
    fixed = np.zeros(len(previous), dtype=bool)
    for _ in range(len(previous) + 1):
        small = (
            (np.abs(weights - previous) < min_trade - 1e-7)
            & (np.abs(weights - previous) > 1e-6)
            & ~fixed
        )
        if not result.success or not small.any():
            break
        fixed |= small
        fixed_lower = lower.copy()
        fixed_upper = upper.copy()
        fixed_lower[fixed] = previous[fixed]
        fixed_upper[fixed] = previous[fixed]
        second = minimize(
            objective,
            x0=weights,
            method="SLSQP",
            bounds=list(zip(fixed_lower, fixed_upper)),
            constraints=constraints,
            options={"maxiter": 350, "ftol": 1e-11},
        )
        if not second.success:
            break
        result = second
        weights = second.x

    achieved_volatility = math.sqrt(variance(weights))
    return np.maximum(weights, 0.0), bool(result.success), achieved_volatility


def run_guardrail(
    cache_data: Mapping,
    stress: pd.DataFrame,
    *,
    calm_base_target: float,
    stress_base_target: float = 0.15,
    strategy_strength: float = 1.0,
    max_position: float,
    return_path: bool = False,
) -> tuple[pd.Series, pd.DataFrame] | tuple[
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    cache = cache_data["cache"]
    returns = pd.DataFrame(cache_data["rets"])
    dates = pd.DatetimeIndex(cache_data["dates"])
    symbols = list(cache_data["SYMS"])
    warmup = int(cache_data["WARMUP"])
    step = int(cache_data["STEP"])
    weights = np.full(len(symbols), 0.95 / len(symbols))
    path = pd.DataFrame(0.0, index=dates, columns=symbols)
    path.iloc[:warmup] = weights
    rows = []

    for index in sorted(cache):
        direction, sigma, return_window, _ = cache[index]
        formal_stress = float(stress.loc[index, "modulation"])
        bullish = float(bool(stress.loc[index, "bullish_regime"]))
        # A long bear cannot become risk-on merely because volatility has
        # normalised. Relaxation requires both bullish trend and low stress.
        defence = 1.0 - bullish * (1.0 - formal_stress)
        base_target = float(
            calm_base_target
            + defence * (stress_base_target - calm_base_target)
        )
        volatility_budget = base_target * HEALTH_BUDGET_FACTOR

        sigma = np.asarray(sigma, dtype=float)
        correlation = _repair_correlation(
            pd.DataFrame(return_window, columns=symbols),
            symbols,
        )
        covariance_annual = (
            np.diag(sigma) @ correlation @ np.diag(sigma) * TRADING_DAYS
        )
        ranked_strategy = strategy_target(
            direction,
            max_position=max_position,
        )
        neutral = np.full(len(symbols), 1.0 / len(symbols))
        desired = (
            (1.0 - float(strategy_strength)) * neutral
            + float(strategy_strength) * ranked_strategy
        )
        weights, success, achieved_volatility = solve_guardrail(
            desired,
            weights,
            covariance_annual,
            volatility_budget,
            max_position=max_position,
        )
        path.iloc[index : min(index + step, len(dates))] = weights
        rows.append(
            {
                "date": pd.Timestamp(dates[index - 1]),
                "defence": defence,
                "base_target": base_target,
                "volatility_budget": volatility_budget,
                "achieved_volatility": achieved_volatility,
                "gross": float(weights.sum()),
                "success": success,
                "risk_constraint_binding": bool(
                    achieved_volatility >= volatility_budget * 0.995
                ),
                "strategy_strength": float(strategy_strength),
            }
        )

    turnover = path.diff().abs().sum(axis=1).fillna(0.0)
    gross_return = (path.shift(1) * returns.reindex(path.index)).sum(axis=1)
    net = (gross_return - TRANSACTION_COST * turnover).iloc[warmup:]
    if return_path:
        return net, pd.DataFrame(rows), path.iloc[warmup:]
    return net, pd.DataFrame(rows)


def evaluate(
    cache_path: Path,
    periods: list[tuple[int, int, str]],
) -> dict:
    with cache_path.open("rb") as handle:
        cache_data = pickle.load(handle)
    max_position = 0.25 if len(cache_data["SYMS"]) <= 10 else 0.20
    stress = build_stress_series(cache_data)
    daily = run_daily_strategy(cache_data, max_position)
    current_policy = Policy(
        "current",
        0.15,
        0.15,
        6.0,
        6.0,
        adaptive_target=False,
        adaptive_risk_aversion=False,
    )
    current, _ = run_policy(
        cache_data,
        stress,
        current_policy,
        max_position,
    )
    guardrail_25, decisions_25 = run_guardrail(
        cache_data,
        stress,
        calm_base_target=0.25,
        max_position=max_position,
    )
    guardrail_30, decisions_30 = run_guardrail(
        cache_data,
        stress,
        calm_base_target=0.30,
        max_position=max_position,
    )
    neutral_guardrail, neutral_decisions = run_guardrail(
        cache_data,
        stress,
        calm_base_target=0.25,
        strategy_strength=0.0,
        max_position=max_position,
    )
    tilted_guardrail, tilted_decisions = run_guardrail(
        cache_data,
        stress,
        calm_base_target=0.25,
        strategy_strength=0.25,
        max_position=max_position,
    )
    series = {
        "daily_strategy": daily,
        "current_fusion": current,
        "guardrail_25": guardrail_25,
        "guardrail_30": guardrail_30,
        "neutral_guardrail": neutral_guardrail,
        "strategy_tilt_25_guardrail": tilted_guardrail,
    }
    output_periods = {}
    for start, end, label in periods:
        selected = {
            name: select_period(values, start, end)
            for name, values in series.items()
        }
        output_periods[label] = {
            "period": [start, end],
            "metrics": {
                name: metrics(values) for name, values in selected.items()
            },
            "vs_current": {
                name: paired_block_bootstrap(
                    selected["current_fusion"],
                    values,
                )
                for name, values in selected.items()
                if name != "current_fusion"
            },
        }
    return {
        "cache": str(cache_path),
        "symbols": list(cache_data["SYMS"]),
        "decision_summary": {
            "guardrail_25": {
                "mean_base_target": float(decisions_25["base_target"].mean()),
                "mean_gross": float(decisions_25["gross"].mean()),
                "risk_binding_share": float(
                    decisions_25["risk_constraint_binding"].mean()
                ),
                "success_rate": float(decisions_25["success"].mean()),
            },
            "guardrail_30": {
                "mean_base_target": float(decisions_30["base_target"].mean()),
                "mean_gross": float(decisions_30["gross"].mean()),
                "risk_binding_share": float(
                    decisions_30["risk_constraint_binding"].mean()
                ),
                "success_rate": float(decisions_30["success"].mean()),
            },
            "neutral_guardrail": {
                "mean_base_target": float(
                    neutral_decisions["base_target"].mean()
                ),
                "mean_gross": float(neutral_decisions["gross"].mean()),
                "risk_binding_share": float(
                    neutral_decisions["risk_constraint_binding"].mean()
                ),
                "success_rate": float(neutral_decisions["success"].mean()),
            },
            "strategy_tilt_25_guardrail": {
                "mean_base_target": float(
                    tilted_decisions["base_target"].mean()
                ),
                "mean_gross": float(tilted_decisions["gross"].mean()),
                "risk_binding_share": float(
                    tilted_decisions["risk_constraint_binding"].mean()
                ),
                "success_rate": float(tilted_decisions["success"].mean()),
            },
        },
        "periods": output_periods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="append", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "guardrail_fusion_research.json",
    )
    args = parser.parse_args()
    results = []
    for cache_path in args.cache:
        with cache_path.open("rb") as handle:
            cache_data = pickle.load(handle)
        first_year = pd.DatetimeIndex(cache_data["dates"])[
            int(cache_data["WARMUP"])
        ].year
        periods = (
            [
                (2000, 2013, "external_2000_2013"),
                (2014, 2023, "external_2014_2023"),
                (2000, 2023, "external_full"),
            ]
            if first_year <= 2000
            else [
                (2014, 2018, "design_2014_2018"),
                (2019, 2023, "validation_2019_2023"),
                (2014, 2023, "full_2014_2023"),
            ]
        )
        results.append(evaluate(cache_path, periods))
    args.output.write_text(
        json.dumps(
            {
                "method": "strategy_first_risk_guardrail",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
