"""Research an adaptive risk overlay for AURORA's production Fusion path.

This script deliberately leaves the production recommendation code unchanged.
It consumes the causal rebalance caches built during Fusion validation and
compares a small, pre-declared set of policies:

* current: fixed 15% base volatility target and risk aversion 6;
* relaxed: fixed 30% target and risk aversion 2;
* adaptive_target: only the volatility target responds to stress;
* adaptive_balanced: 25%/3 in calm markets to 15%/6 under stress;
* adaptive_growth: 30%/2 in calm markets to 15%/6 under stress.

Stress uses only information available at the rebalance date: the expanding
percentiles of median HAR-X sigma, SPY 20-day volatility, average correlation,
and the breadth of assets whose HAR-X sigma is above its own 80th percentile.
No drawdown level or future crisis label enters the policy.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.recommendation.gated_news import (  # noqa: E402
    risk_controlled_allocation,
    strategy_alpha,
)

TRADING_DAYS = 252
TRANSACTION_COST = 0.0025
MIN_PERCENTILE_HISTORY = 52


@dataclass(frozen=True)
class Policy:
    name: str
    calm_target: float
    stress_target: float
    calm_risk_aversion: float
    stress_risk_aversion: float
    adaptive_target: bool = True
    adaptive_risk_aversion: bool = True
    requires_bullish_regime: bool = False
    maximum_strategy_sleeve: float = 0.0


POLICIES = (
    Policy(
        "current",
        calm_target=0.15,
        stress_target=0.15,
        calm_risk_aversion=6.0,
        stress_risk_aversion=6.0,
        adaptive_target=False,
        adaptive_risk_aversion=False,
    ),
    Policy(
        "relaxed_fixed",
        calm_target=0.30,
        stress_target=0.30,
        calm_risk_aversion=2.0,
        stress_risk_aversion=2.0,
        adaptive_target=False,
        adaptive_risk_aversion=False,
    ),
    Policy(
        "adaptive_target",
        calm_target=0.30,
        stress_target=0.15,
        calm_risk_aversion=6.0,
        stress_risk_aversion=6.0,
        adaptive_target=True,
        adaptive_risk_aversion=False,
    ),
    Policy(
        "adaptive_balanced",
        calm_target=0.25,
        stress_target=0.15,
        calm_risk_aversion=3.0,
        stress_risk_aversion=6.0,
    ),
    Policy(
        "adaptive_growth",
        calm_target=0.30,
        stress_target=0.15,
        calm_risk_aversion=2.0,
        stress_risk_aversion=6.0,
    ),
    Policy(
        "bullish_balanced",
        calm_target=0.25,
        stress_target=0.15,
        calm_risk_aversion=3.0,
        stress_risk_aversion=6.0,
        requires_bullish_regime=True,
    ),
    Policy(
        "bullish_growth",
        calm_target=0.30,
        stress_target=0.15,
        calm_risk_aversion=2.0,
        stress_risk_aversion=6.0,
        requires_bullish_regime=True,
    ),
    Policy(
        "strategy_sleeve_25",
        calm_target=0.15,
        stress_target=0.15,
        calm_risk_aversion=6.0,
        stress_risk_aversion=6.0,
        adaptive_target=False,
        adaptive_risk_aversion=False,
        maximum_strategy_sleeve=0.25,
    ),
    Policy(
        "strategy_sleeve_40",
        calm_target=0.15,
        stress_target=0.15,
        calm_risk_aversion=6.0,
        stress_risk_aversion=6.0,
        adaptive_target=False,
        adaptive_risk_aversion=False,
        maximum_strategy_sleeve=0.40,
    ),
)


def prior_percentile(history: list[float], value: float) -> float:
    """Causal expanding percentile; current observation is never in history."""
    clean = np.asarray([item for item in history if np.isfinite(item)], dtype=float)
    if len(clean) < MIN_PERCENTILE_HISTORY or not np.isfinite(value):
        return 0.5
    return float((np.count_nonzero(clean <= value) + 0.5) / (len(clean) + 1.0))


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def build_stress_series(cache_data: Mapping) -> pd.DataFrame:
    """Build a causal, continuous stress score for every rebalance point."""
    cache = cache_data["cache"]
    dates = pd.DatetimeIndex(cache_data["dates"])
    symbols = list(cache_data["SYMS"])
    spy = pd.Series(cache_data["spy"]).sort_index()
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_vol = spy.pct_change().rolling(20, min_periods=20).std() * math.sqrt(
        TRADING_DAYS
    )

    median_sigma_history: list[float] = []
    spy_vol_history: list[float] = []
    correlation_history: list[float] = []
    symbol_sigma_history: dict[str, list[float]] = {
        symbol: [] for symbol in symbols
    }
    filtered_stress = 0.5
    rows = []

    for index in sorted(cache):
        _, sigma, return_window, _ = cache[index]
        decision_date = pd.Timestamp(dates[index - 1]).tz_localize(None)
        sigma = np.asarray(sigma, dtype=float)
        median_sigma = float(np.nanmedian(sigma))

        correlation = pd.DataFrame(return_window).corr().to_numpy(dtype=float)
        upper = correlation[np.triu_indices_from(correlation, k=1)]
        average_correlation = float(np.nanmean(upper))

        available_spy_vol = spy_vol.loc[:decision_date].dropna()
        current_spy_vol = (
            float(available_spy_vol.iloc[-1])
            if len(available_spy_vol)
            else float("nan")
        )
        available_spy = spy.loc[:decision_date].dropna()
        spy_price = float(available_spy.iloc[-1])
        spy_sma50 = float(available_spy.iloc[-50:].mean())
        spy_sma200 = (
            float(available_spy.iloc[-200:].mean())
            if len(available_spy) >= 200
            else spy_sma50
        )
        spy_momentum20 = (
            spy_price / float(available_spy.iloc[-21]) - 1.0
            if len(available_spy) > 21
            else 0.0
        )
        bullish_regime = bool(
            spy_price > spy_sma50
            and spy_sma50 > spy_sma200
            and spy_momentum20 > 0.0
        )

        sigma_percentile = prior_percentile(
            median_sigma_history,
            median_sigma,
        )
        spy_percentile = prior_percentile(spy_vol_history, current_spy_vol)
        correlation_percentile = prior_percentile(
            correlation_history,
            average_correlation,
        )
        individual_percentiles = [
            prior_percentile(symbol_sigma_history[symbol], sigma[position])
            for position, symbol in enumerate(symbols)
        ]
        risk_breadth = float(
            np.mean(np.asarray(individual_percentiles, dtype=float) >= 0.80)
        )

        # Formal risk forecast is the largest input. Market volatility and
        # correlation capture common stress, while breadth prevents one noisy
        # asset from switching the whole portfolio into defence.
        raw_stress = float(
            0.45 * sigma_percentile
            + 0.25 * spy_percentile
            + 0.15 * correlation_percentile
            + 0.15 * risk_breadth
        )

        # Fast attack, slower release. The asymmetric filter avoids one quiet
        # observation immediately disabling protection after a volatility shock.
        if raw_stress >= filtered_stress:
            filtered_stress = 0.25 * filtered_stress + 0.75 * raw_stress
        else:
            filtered_stress = 0.70 * filtered_stress + 0.30 * raw_stress

        # Percentile semantics pre-declare 50 as calm and 80 as full stress.
        modulation = smoothstep((filtered_stress - 0.50) / 0.30)
        rows.append(
            {
                "index": int(index),
                "date": decision_date,
                "sigma_percentile": sigma_percentile,
                "spy_vol_percentile": spy_percentile,
                "correlation_percentile": correlation_percentile,
                "risk_breadth": risk_breadth,
                "raw_stress": raw_stress,
                "stress": filtered_stress,
                "modulation": modulation,
                "bullish_regime": bullish_regime,
            }
        )

        median_sigma_history.append(median_sigma)
        spy_vol_history.append(current_spy_vol)
        correlation_history.append(average_correlation)
        for position, symbol in enumerate(symbols):
            symbol_sigma_history[symbol].append(float(sigma[position]))

    return pd.DataFrame(rows).set_index("index", drop=False)


def interpolate(calm: float, stress: float, modulation: float) -> float:
    return float(calm + modulation * (stress - calm))


def project_trade(
    desired: np.ndarray,
    previous: np.ndarray,
    *,
    max_position: float,
    max_change: float = 0.05,
    min_trade: float = 0.01,
) -> np.ndarray:
    """Project a sleeve blend onto the production position/trade constraints."""
    desired = np.asarray(desired, dtype=float)
    previous = np.asarray(previous, dtype=float)
    lower = np.maximum(0.0, previous - max_change)
    upper = np.minimum(max_position, previous + max_change)
    target_gross = float(
        np.clip(desired.sum(), lower.sum(), min(1.0, upper.sum()))
    )
    start = np.clip(desired, lower, upper)
    if abs(start.sum() - target_gross) > 1e-8:
        start = previous.copy()
    result = minimize(
        lambda candidate: float(np.square(candidate - desired).sum()),
        x0=start,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=[
            {
                "type": "eq",
                "fun": lambda candidate: candidate.sum() - target_gross,
            }
        ],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    projected = result.x if result.success else previous.copy()
    fixed = np.zeros(len(previous), dtype=bool)
    for _ in range(len(previous) + 1):
        sub_threshold = (
            (np.abs(projected - previous) < min_trade - 1e-7)
            & (np.abs(projected - previous) > 1e-7)
            & ~fixed
        )
        if not result.success or not sub_threshold.any():
            break
        candidate_fixed = fixed | sub_threshold
        fixed_lower = lower.copy()
        fixed_upper = upper.copy()
        fixed_lower[candidate_fixed] = previous[candidate_fixed]
        fixed_upper[candidate_fixed] = previous[candidate_fixed]
        if (
            fixed_lower.sum() > target_gross + 1e-8
            or fixed_upper.sum() < target_gross - 1e-8
        ):
            break
        second = minimize(
            lambda candidate: float(np.square(candidate - desired).sum()),
            x0=projected,
            method="SLSQP",
            bounds=list(zip(fixed_lower, fixed_upper)),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda candidate: (
                        candidate.sum() - target_gross
                    ),
                }
            ],
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if not second.success:
            break
        fixed = candidate_fixed
        result = second
        projected = second.x
    if projected.sum() > 1.0 + 1e-8:
        projected *= 1.0 / projected.sum()
    return np.maximum(projected, 0.0)


def run_daily_strategy(cache_data: Mapping, max_position: float) -> pd.Series:
    cache = cache_data["cache"]
    returns = pd.DataFrame(cache_data["rets"])
    dates = pd.DatetimeIndex(cache_data["dates"])
    symbols = list(cache_data["SYMS"])
    warmup = int(cache_data["WARMUP"])
    step = int(cache_data["STEP"])
    weights = np.full(len(symbols), 1.0 / len(symbols))
    path = pd.DataFrame(0.0, index=dates, columns=symbols)
    path.iloc[:warmup] = weights

    for index in sorted(cache):
        direction = np.asarray(cache[index][0], dtype=float)
        ranks = pd.Series(direction).rank(pct=True).to_numpy(dtype=float)
        weights = np.clip(ranks / max(ranks.sum(), 1e-10), 0.0, max_position)
        weights /= max(weights.sum(), 1e-10)
        path.iloc[index : min(index + step, len(dates))] = weights

    turnover = path.diff().abs().sum(axis=1).fillna(0.0)
    gross_return = (path.shift(1) * returns.reindex(path.index)).sum(axis=1)
    return (gross_return - TRANSACTION_COST * turnover).iloc[warmup:]


def run_policy(
    cache_data: Mapping,
    stress: pd.DataFrame,
    policy: Policy,
    max_position: float,
) -> tuple[pd.Series, pd.DataFrame]:
    cache = cache_data["cache"]
    returns = pd.DataFrame(cache_data["rets"])
    dates = pd.DatetimeIndex(cache_data["dates"])
    symbols = list(cache_data["SYMS"])
    warmup = int(cache_data["WARMUP"])
    step = int(cache_data["STEP"])
    weights = np.full(len(symbols), 0.95 / len(symbols))
    cash = 1.0 - float(weights.sum())
    path = pd.DataFrame(0.0, index=dates, columns=symbols)
    path.iloc[:warmup] = weights
    decisions = []

    for index in sorted(cache):
        direction, sigma, return_window, _ = cache[index]
        raw_modulation = float(stress.loc[index, "modulation"])
        bullish_regime = bool(stress.loc[index, "bullish_regime"])
        modulation = raw_modulation
        if policy.requires_bullish_regime:
            # Relax only when both conditions agree: formal risk is calm and
            # the broad market has a confirmed positive trend. A secular bear
            # cannot become "safe" merely because volatility normalised.
            bullish = float(bullish_regime)
            modulation = 1.0 - bullish * (1.0 - modulation)
        target = (
            interpolate(
                policy.calm_target,
                policy.stress_target,
                modulation,
            )
            if policy.adaptive_target
            else policy.calm_target
        )
        risk_aversion = (
            interpolate(
                policy.calm_risk_aversion,
                policy.stress_risk_aversion,
                modulation,
            )
            if policy.adaptive_risk_aversion
            else policy.calm_risk_aversion
        )
        allocation = risk_controlled_allocation(
            symbols,
            strategy_alpha(
                np.asarray(direction, dtype=float),
                np.asarray(sigma, dtype=float),
                information_coefficient=0.02,
            ),
            np.asarray(sigma, dtype=float),
            pd.DataFrame(return_window, columns=symbols),
            weights,
            cash,
            health_score=60.0,
            base_risk_aversion=risk_aversion,
            base_target_annual_volatility=target,
            turnover_penalty=TRANSACTION_COST,
            max_position=max_position,
        )
        if allocation.success:
            weights = allocation.weights
            cash = allocation.cash_weight
        strategy_sleeve = (
            float(policy.maximum_strategy_sleeve)
            * float(bullish_regime)
            * (1.0 - raw_modulation)
        )
        if allocation.success and strategy_sleeve > 0.0:
            ranks = pd.Series(direction).rank(pct=True).to_numpy(dtype=float)
            strategy_weights = ranks / max(ranks.sum(), 1e-10)
            strategy_weights = np.clip(strategy_weights, 0.0, max_position)
            strategy_weights /= max(strategy_weights.sum(), 1e-10)
            desired = (
                (1.0 - strategy_sleeve) * weights
                + strategy_sleeve * strategy_weights
            )
            weights = project_trade(
                desired,
                previous=np.asarray(
                    [
                        float(path.iloc[index - 1][symbol])
                        for symbol in symbols
                    ],
                    dtype=float,
                ),
                max_position=max_position,
            )
            cash = max(0.0, 1.0 - float(weights.sum()))
        path.iloc[index : min(index + step, len(dates))] = weights
        decisions.append(
            {
                "date": pd.Timestamp(dates[index - 1]),
                "policy": policy.name,
                "modulation": modulation,
                "base_target": target,
                "risk_aversion": risk_aversion,
                "gross": float(weights.sum()),
                "cash": cash,
                "success": allocation.success,
                "strategy_sleeve": strategy_sleeve,
            }
        )

    turnover = path.diff().abs().sum(axis=1).fillna(0.0)
    gross_return = (path.shift(1) * returns.reindex(path.index)).sum(axis=1)
    net = (gross_return - TRANSACTION_COST * turnover).iloc[warmup:]
    return net, pd.DataFrame(decisions)


def sharpe(returns: pd.Series) -> float:
    return float(returns.mean() / returns.std() * math.sqrt(TRADING_DAYS))


def cagr(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1.0)


def maximum_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def metrics(returns: pd.Series) -> dict[str, float]:
    annual_volatility = float(returns.std() * math.sqrt(TRADING_DAYS))
    drawdown = maximum_drawdown(returns)
    annual_return = cagr(returns)
    return {
        "cagr": annual_return,
        "sharpe": sharpe(returns),
        "annual_volatility": annual_volatility,
        "max_drawdown": drawdown,
        "calmar": annual_return / abs(drawdown) if drawdown else float("nan"),
    }


def paired_block_bootstrap(
    baseline: pd.Series,
    candidate: pd.Series,
    *,
    samples: int = 2000,
    block: int = 20,
    seed: int = 20260726,
) -> dict[str, float | list[float]]:
    aligned = pd.concat([baseline, candidate], axis=1).dropna()
    left = aligned.iloc[:, 0].to_numpy(dtype=float)
    right = aligned.iloc[:, 1].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(len(aligned) / block))
    gains = []
    for _ in range(samples):
        starts = rng.integers(0, len(aligned) - block + 1, blocks)
        indices = np.concatenate(
            [np.arange(start, start + block) for start in starts]
        )[: len(aligned)]
        gains.append(
            sharpe(pd.Series(right[indices]))
            - sharpe(pd.Series(left[indices]))
        )
    gain_array = np.asarray(gains, dtype=float)
    return {
        "point_delta_sharpe": sharpe(candidate) - sharpe(baseline),
        "bootstrap_delta_sharpe_95": np.quantile(
            gain_array,
            [0.025, 0.50, 0.975],
        ).tolist(),
        "probability_positive": float(np.mean(gain_array > 0.0)),
    }


def select_period(
    series: pd.Series,
    start_year: int,
    end_year: int,
) -> pd.Series:
    return series[
        (series.index.year >= start_year) & (series.index.year <= end_year)
    ]


def evaluate_cache(
    cache_path: Path,
    periods: list[tuple[int, int, str]],
) -> dict:
    with cache_path.open("rb") as handle:
        cache_data = pickle.load(handle)
    max_position = 0.25 if len(cache_data["SYMS"]) <= 10 else 0.20
    stress = build_stress_series(cache_data)
    daily = run_daily_strategy(cache_data, max_position)
    series: dict[str, pd.Series] = {"daily_strategy": daily}
    decisions: dict[str, pd.DataFrame] = {}

    for policy in POLICIES:
        policy_series, policy_decisions = run_policy(
            cache_data,
            stress,
            policy,
            max_position,
        )
        series[policy.name] = policy_series
        decisions[policy.name] = policy_decisions

    period_results = {}
    for start, end, label in periods:
        period_series = {
            name: select_period(values, start, end)
            for name, values in series.items()
        }
        current = period_series["current"]
        period_results[label] = {
            "period": [start, end],
            "metrics": {
                name: metrics(values)
                for name, values in period_series.items()
                if len(values)
            },
            "vs_current": {
                name: paired_block_bootstrap(current, values)
                for name, values in period_series.items()
                if name not in {"current"} and len(values)
            },
        }

    stress_dates = stress.loc[stress["modulation"] >= 0.80, "date"]
    return {
        "cache": str(cache_path),
        "symbols": list(cache_data["SYMS"]),
        "stress_summary": {
            "mean_modulation": float(stress["modulation"].mean()),
            "full_stress_share": float((stress["modulation"] >= 0.80).mean()),
            "calm_share": float((stress["modulation"] <= 0.20).mean()),
            "first_full_stress_dates": [
                value.date().isoformat() for value in stress_dates.iloc[:20]
            ],
        },
        "decision_summary": {
            name: {
                "mean_target": float(frame["base_target"].mean()),
                "mean_risk_aversion": float(frame["risk_aversion"].mean()),
                "mean_gross": float(frame["gross"].mean()),
                "optimizer_success_rate": float(frame["success"].mean()),
                "mean_strategy_sleeve": float(frame["strategy_sleeve"].mean()),
            }
            for name, frame in decisions.items()
        },
        "periods": period_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        type=Path,
        help="Causal Fusion cache; may be supplied more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "adaptive_fusion_research.json",
    )
    args = parser.parse_args()

    outputs = []
    for cache_path in args.cache:
        with cache_path.open("rb") as handle:
            cache_data = pickle.load(handle)
        first_year = pd.DatetimeIndex(cache_data["dates"])[
            int(cache_data["WARMUP"])
        ].year
        if first_year <= 2000:
            periods = [
                (2000, 2013, "external_2000_2013"),
                (2014, 2023, "external_2014_2023"),
                (2000, 2023, "external_full"),
            ]
        else:
            periods = [
                (2014, 2018, "design_2014_2018"),
                (2019, 2023, "validation_2019_2023"),
                (2014, 2023, "full_2014_2023"),
            ]
        outputs.append(evaluate_cache(cache_path, periods))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "method": "causal_adaptive_dual_channel_fusion",
                "policies": [policy.__dict__ for policy in POLICIES],
                "results": outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
