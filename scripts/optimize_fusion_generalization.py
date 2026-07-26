"""Diagnose and test a gross-only systemic-risk Fusion overlay.

The candidate preserves the relative portfolio supplied by Daily Strategy (or
a fixed neutral/Strategy blend). Risk cannot reorder stocks. It can only scale
the whole risky portfolio uniformly when causal systemic stress raises the
forecast portfolio volatility above a dynamic budget.

This script is research-only and does not alter the production endpoint.
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

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SCRIPTS = ROOT / "scripts"
for path in (BACKEND, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from research_adaptive_fusion import (  # noqa: E402
    build_stress_series,
    metrics,
    paired_block_bootstrap,
    project_trade,
    run_daily_strategy,
    smoothstep,
)
from src.recommendation.gated_news import _repair_correlation  # noqa: E402

TRADING_DAYS = 252
TRANSACTION_COST = 0.0025
HEALTH_SCORE = 60.0
HEALTH_BUDGET_FACTOR = 0.65 + 0.35 * HEALTH_SCORE / 100.0


@dataclass(frozen=True)
class OverlayPolicy:
    name: str
    strategy_strength: float
    calm_base_target: float = 0.30
    stress_base_target: float = 0.15
    non_bullish_defence_floor: float = 0.35
    stress_release_speed: float = 0.30


POLICIES = (
    OverlayPolicy("systemic_gross_tilt_25", strategy_strength=0.25),
    OverlayPolicy("systemic_gross_tilt_50", strategy_strength=0.50),
    OverlayPolicy("systemic_gross_strategy_100", strategy_strength=1.00),
    OverlayPolicy(
        "systemic_gross_strategy_100_light_trend",
        strategy_strength=1.00,
        non_bullish_defence_floor=0.15,
    ),
    OverlayPolicy(
        "systemic_gross_strategy_100_no_trend",
        strategy_strength=1.00,
        non_bullish_defence_floor=0.00,
    ),
    OverlayPolicy(
        "systemic_gross_strategy_100_fast_release",
        strategy_strength=1.00,
        non_bullish_defence_floor=0.00,
        stress_release_speed=0.70,
    ),
    OverlayPolicy(
        "systemic_gross_strategy_100_immediate_release",
        strategy_strength=1.00,
        non_bullish_defence_floor=0.00,
        stress_release_speed=1.00,
    ),
)


def add_systemic_stress(stress: pd.DataFrame) -> pd.DataFrame:
    """Separate systemic crisis risk from merely high individual volatility."""
    output = stress.copy()
    raw_values = (
        0.45 * output["spy_vol_percentile"]
        + 0.30 * output["correlation_percentile"]
        + 0.25 * output["risk_breadth"]
    )
    output["systemic_stress_raw"] = raw_values
    for release_speed in sorted(
        {policy.stress_release_speed for policy in POLICIES}
    ):
        filtered = 0.5
        systemic_values = []
        for raw in raw_values:
            raw = float(raw)
            if raw >= filtered:
                filtered = 0.25 * filtered + 0.75 * raw
            else:
                filtered = (
                    (1.0 - release_speed) * filtered
                    + release_speed * raw
                )
            systemic_values.append(
                smoothstep((filtered - 0.50) / 0.30)
            )
        suffix = int(round(release_speed * 100))
        output[f"systemic_stress_release_{suffix}"] = systemic_values
    return output


def ranked_strategy_target(
    direction: np.ndarray,
    *,
    strategy_strength: float,
    max_position: float,
) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    ranks = pd.Series(direction).rank(pct=True).to_numpy(dtype=float)
    strategy = ranks / max(ranks.sum(), 1e-10)
    neutral = np.full(len(direction), 1.0 / len(direction))
    target = (
        (1.0 - float(strategy_strength)) * neutral
        + float(strategy_strength) * strategy
    )
    target = np.clip(target, 0.0, max_position)
    return target / max(target.sum(), 1e-10)


def run_overlay(
    cache_data: Mapping,
    stress: pd.DataFrame,
    policy: OverlayPolicy,
    *,
    max_position: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
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
        release_suffix = int(round(policy.stress_release_speed * 100))
        systemic = float(
            stress.loc[index, f"systemic_stress_release_{release_suffix}"]
        )
        trend_floor = (
            0.0
            if bool(stress.loc[index, "bullish_regime"])
            else float(policy.non_bullish_defence_floor)
        )
        defence = systemic + (1.0 - systemic) * trend_floor
        base_target = float(
            policy.calm_base_target
            + defence
            * (policy.stress_base_target - policy.calm_base_target)
        )
        volatility_budget = base_target * HEALTH_BUDGET_FACTOR
        relative_target = ranked_strategy_target(
            direction,
            strategy_strength=policy.strategy_strength,
            max_position=max_position,
        )

        sigma = np.asarray(sigma, dtype=float)
        correlation = _repair_correlation(
            pd.DataFrame(return_window, columns=symbols),
            symbols,
        )
        covariance_annual = (
            np.diag(sigma) @ correlation @ np.diag(sigma) * TRADING_DAYS
        )
        predicted_full_volatility = float(
            math.sqrt(
                max(
                    relative_target
                    @ covariance_annual
                    @ relative_target,
                    0.0,
                )
            )
        )
        raw_gross = (
            1.0
            if predicted_full_volatility <= 1e-10
            else volatility_budget / predicted_full_volatility
        )
        target_gross = float(np.clip(raw_gross, 0.35, 1.0))
        desired = relative_target * target_gross
        previous = weights.copy()
        weights = project_trade(
            desired,
            previous,
            max_position=max_position,
        )
        achieved_gross = float(weights.sum())
        achieved_volatility = float(
            math.sqrt(
                max(weights @ covariance_annual @ weights, 0.0)
            )
        )
        path.iloc[index : min(index + step, len(dates))] = weights
        rows.append(
            {
                "date": pd.Timestamp(dates[index - 1]),
                "policy": policy.name,
                "systemic_stress": systemic,
                "defence": defence,
                "base_target": base_target,
                "volatility_budget": volatility_budget,
                "predicted_full_volatility": predicted_full_volatility,
                "target_gross": target_gross,
                "gross": achieved_gross,
                "achieved_volatility": achieved_volatility,
                "turnover": float(np.abs(weights - previous).sum()),
            }
        )

    turnover = path.diff().abs().sum(axis=1).fillna(0.0)
    gross_return = (path.shift(1) * returns.reindex(path.index)).sum(axis=1)
    net = (gross_return - TRANSACTION_COST * turnover).iloc[warmup:]
    return net, pd.DataFrame(rows), path.iloc[warmup:]


def subset(series: pd.Series, start: str, end: str) -> pd.Series:
    return series.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def evaluate_period(
    series: Mapping[str, pd.Series],
    *,
    start: str,
    end: str,
) -> dict:
    selected = {
        name: subset(values, start, end).dropna()
        for name, values in series.items()
    }
    daily = selected["daily_strategy"]
    return {
        "period": [start, end],
        "metrics": {
            name: metrics(values) for name, values in selected.items()
        },
        "vs_daily_strategy": {
            name: paired_block_bootstrap(
                daily,
                values,
                samples=3000,
                block=20,
                seed=20260726,
            )
            for name, values in selected.items()
            if name != "daily_strategy"
        },
    }


def periods_for(label: str) -> list[tuple[str, str, str]]:
    if label == "historical_21":
        return [
            ("2014-01-01", "2018-12-31", "design_2014_2018"),
            ("2019-01-01", "2023-12-31", "validation_2019_2023"),
            ("2014-01-01", "2023-12-31", "full_2014_2023"),
        ]
    if label == "forward_21":
        return [
            ("2024-01-01", "2024-12-31", "diagnostic_2024"),
            ("2025-01-01", "2026-07-24", "post_diagnostic_2025_current"),
            ("2024-01-01", "2026-07-24", "forward_full"),
        ]
    if label == "external_etf":
        return [
            ("2000-01-01", "2013-12-31", "external_2000_2013"),
            ("2014-01-01", "2023-12-31", "external_2014_2023"),
            ("2000-01-01", "2023-12-31", "external_full"),
        ]
    raise ValueError(f"unknown cache label: {label}")


def evaluate_cache(
    cache_path: Path,
    label: str,
) -> dict:
    with cache_path.open("rb") as handle:
        cache_data = pickle.load(handle)
    max_position = 0.25 if len(cache_data["SYMS"]) <= 10 else 0.20
    stress = add_systemic_stress(build_stress_series(cache_data))
    series: dict[str, pd.Series] = {
        "daily_strategy": run_daily_strategy(cache_data, max_position)
    }
    decisions = {}
    for policy in POLICIES:
        returns, decision_frame, _ = run_overlay(
            cache_data,
            stress,
            policy,
            max_position=max_position,
        )
        series[policy.name] = returns
        decisions[policy.name] = {
            "mean_gross": float(decision_frame["gross"].mean()),
            "mean_base_target": float(
                decision_frame["base_target"].mean()
            ),
            "mean_turnover_per_rebalance": float(
                decision_frame["turnover"].mean()
            ),
            "full_stress_share": float(
                (decision_frame["systemic_stress"] >= 0.80).mean()
            ),
        }
    output_periods = {
        name: evaluate_period(series, start=start, end=end)
        for start, end, name in periods_for(label)
    }
    return {
        "label": label,
        "cache": str(cache_path),
        "symbols": list(cache_data["SYMS"]),
        "decision_summary": decisions,
        "periods": output_periods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "fusion_generalization_research.json",
    )
    args = parser.parse_args()
    results = [
        evaluate_cache(Path(path), label) for label, path in args.cache
    ]
    payload = {
        "method": "systemic_risk_gross_only_overlay",
        "policies": [policy.__dict__ for policy in POLICIES],
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
