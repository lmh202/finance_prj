"""Research a crisis-gated, rebound-aware Fusion gross overlay.

The candidate keeps 100% of Daily Strategy's relative stock weights. Risk can
only scale the portfolio uniformly, and forecast volatility is allowed to cut
gross exposure only after causal "bad risk" confirms a systemic event.

The state machine separates three questions:

1. Is observed volatility harmful or merely rewarded upside volatility?
2. If a crisis is confirmed, how much gross exposure fits the risk budget?
3. Has a broad recovery started even though the slow volatility forecast is
   still elevated?

This script is research-only. It does not modify the production endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass
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

from optimize_fusion_generalization import (  # noqa: E402
    OverlayPolicy,
    add_systemic_stress,
    ranked_strategy_target,
    run_overlay,
)
from research_adaptive_fusion import (  # noqa: E402
    build_stress_series,
    metrics,
    paired_block_bootstrap,
    prior_percentile,
    project_trade,
    run_daily_strategy,
    smoothstep,
)
from src.recommendation.gated_news import _repair_correlation  # noqa: E402

TRADING_DAYS = 252
TRANSACTION_COST = 0.0025
HEALTH_SCORE = 60.0
HEALTH_BUDGET_FACTOR = 0.65 + 0.35 * HEALTH_SCORE / 100.0

CURRENT_POLICY = OverlayPolicy(
    "current_systemic_gross_fast_release",
    strategy_strength=1.0,
    non_bullish_defence_floor=0.0,
    stress_release_speed=0.70,
)


@dataclass(frozen=True)
class RegimePolicy:
    name: str
    activation_start: float = 0.50
    activation_full: float = 0.80
    attack_speed: float = 0.75
    release_speed: float = 0.70
    bull_attenuation: float = 0.25
    recovery_strength: float = 0.75
    recovery_threshold: float = 0.50
    stress_base_target: float = 0.15
    minimum_gross: float = 0.35


# The variants isolate the economic contribution of each state transition.
# The parameter grid is intentionally small to limit period-specific tuning.
POLICIES = (
    RegimePolicy(
        "bad_risk_gate_only",
        bull_attenuation=1.0,
        recovery_strength=0.0,
    ),
    RegimePolicy(
        "bad_risk_with_bull_guard",
        recovery_strength=0.0,
    ),
    RegimePolicy(
        "bad_risk_bull_recovery_50",
        recovery_strength=0.50,
    ),
    RegimePolicy(
        "bad_risk_bull_recovery_75",
        recovery_strength=0.75,
    ),
    RegimePolicy(
        "bad_risk_bull_recovery_100",
        recovery_strength=1.00,
    ),
    RegimePolicy(
        "bad_risk_bull_recovery_75_later_gate",
        activation_start=0.55,
        activation_full=0.85,
        recovery_strength=0.75,
    ),
)


@dataclass(frozen=True)
class HybridPolicy:
    """Minimal overlays on top of the existing systemic-risk policy."""

    name: str
    bull_gross_floor: float = 0.0
    bull_systemic_ceiling: float = 0.60
    bull_requires_rewarded_high_vol: bool = False
    recovery_gross_floor: float = 0.95
    recovery_trigger_systemic: float = 0.80
    recovery_impulse_threshold: float = 0.70
    recovery_hold_rebalances: int = 2
    shock_recovery_mode: bool = False
    shock_speed_threshold: float = 0.08
    shock_window_rebalances: int = 16
    minimum_rebound_return: float = 0.0
    minimum_recovery_from_trough: float = 0.0
    maximum_sma200_decline: float = -1.0
    pure_uniform_gross: bool = False


HYBRID_POLICIES = (
    HybridPolicy(
        "current_plus_recovery_85",
        recovery_gross_floor=0.85,
    ),
    HybridPolicy(
        "current_plus_recovery_95",
        recovery_gross_floor=0.95,
    ),
    HybridPolicy(
        "current_plus_recovery_100",
        recovery_gross_floor=1.00,
    ),
    HybridPolicy(
        "current_plus_bull95_recovery95",
        bull_gross_floor=0.95,
        recovery_gross_floor=0.95,
    ),
    HybridPolicy(
        "current_plus_bull100_recovery95",
        bull_gross_floor=1.00,
        recovery_gross_floor=0.95,
    ),
    HybridPolicy(
        "current_plus_bull95_recovery100",
        bull_gross_floor=0.95,
        recovery_gross_floor=1.00,
    ),
    HybridPolicy(
        "current_plus_bull95_only",
        bull_gross_floor=0.95,
        recovery_trigger_systemic=2.0,
    ),
    HybridPolicy(
        "current_plus_bull100_only",
        bull_gross_floor=1.00,
        recovery_trigger_systemic=2.0,
    ),
    HybridPolicy(
        "current_plus_rewarded_high_vol_bull95_only",
        bull_gross_floor=0.95,
        bull_requires_rewarded_high_vol=True,
        recovery_trigger_systemic=2.0,
    ),
    HybridPolicy(
        "shock_v_recovery_75",
        recovery_gross_floor=0.75,
        shock_recovery_mode=True,
        minimum_rebound_return=0.03,
        minimum_recovery_from_trough=0.05,
        maximum_sma200_decline=-0.005,
    ),
    HybridPolicy(
        "shock_v_recovery_85",
        recovery_gross_floor=0.85,
        shock_recovery_mode=True,
        minimum_rebound_return=0.03,
        minimum_recovery_from_trough=0.05,
        maximum_sma200_decline=-0.005,
    ),
    HybridPolicy(
        "shock_v_recovery_90",
        recovery_gross_floor=0.90,
        shock_recovery_mode=True,
        minimum_rebound_return=0.03,
        minimum_recovery_from_trough=0.05,
        maximum_sma200_decline=-0.005,
    ),
    HybridPolicy(
        "shock_v_recovery_90_uniform",
        recovery_gross_floor=0.90,
        shock_recovery_mode=True,
        minimum_rebound_return=0.03,
        minimum_recovery_from_trough=0.05,
        maximum_sma200_decline=-0.005,
        pure_uniform_gross=True,
    ),
    HybridPolicy(
        "shock_v_recovery_90_plus_rewarded_bull95",
        bull_gross_floor=0.95,
        bull_requires_rewarded_high_vol=True,
        recovery_gross_floor=0.90,
        shock_recovery_mode=True,
        minimum_rebound_return=0.03,
        minimum_recovery_from_trough=0.05,
        maximum_sma200_decline=-0.005,
    ),
    HybridPolicy(
        "shock_v_recovery_95",
        recovery_gross_floor=0.95,
        shock_recovery_mode=True,
        minimum_rebound_return=0.03,
        minimum_recovery_from_trough=0.05,
        maximum_sma200_decline=-0.005,
    ),
    HybridPolicy(
        "shock_v_recovery_85_strong",
        recovery_gross_floor=0.85,
        shock_recovery_mode=True,
        minimum_rebound_return=0.05,
        minimum_recovery_from_trough=0.05,
        maximum_sma200_decline=-0.005,
    ),
    HybridPolicy(
        "shock_v_recovery_90_hold_1",
        recovery_gross_floor=0.90,
        recovery_hold_rebalances=1,
        shock_recovery_mode=True,
        minimum_rebound_return=0.03,
        minimum_recovery_from_trough=0.05,
        maximum_sma200_decline=-0.005,
    ),
    HybridPolicy(
        "shock_v_recovery_90_hold_3",
        recovery_gross_floor=0.90,
        recovery_hold_rebalances=3,
        shock_recovery_mode=True,
        minimum_rebound_return=0.03,
        minimum_recovery_from_trough=0.05,
        maximum_sma200_decline=-0.005,
    ),
    HybridPolicy(
        "shock_v_recovery_90_shock_07",
        recovery_gross_floor=0.90,
        shock_recovery_mode=True,
        shock_speed_threshold=0.07,
        minimum_rebound_return=0.03,
        minimum_recovery_from_trough=0.05,
        maximum_sma200_decline=-0.005,
    ),
    HybridPolicy(
        "shock_v_recovery_90_shock_09",
        recovery_gross_floor=0.90,
        shock_recovery_mode=True,
        shock_speed_threshold=0.09,
        minimum_rebound_return=0.03,
        minimum_recovery_from_trough=0.05,
        maximum_sma200_decline=-0.005,
    ),
)

SELECTED_POLICY_NAME = "shock_v_recovery_90_uniform"


def _average_correlation(frame: pd.DataFrame) -> float:
    """Return the finite mean pairwise correlation, or zero if unavailable."""
    if frame.shape[0] < 3 or frame.shape[1] < 2:
        return 0.0
    correlation = frame.corr().to_numpy(dtype=float)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    finite = upper[np.isfinite(upper)]
    return float(finite.mean()) if len(finite) else 0.0


def _period_return(values: pd.Series, sessions: int) -> float:
    if len(values) <= sessions:
        return 0.0
    return float(values.iloc[-1] / values.iloc[-sessions - 1] - 1.0)


def _semivolatility(returns: pd.Series) -> float:
    values = np.minimum(returns.to_numpy(dtype=float), 0.0)
    if not len(values):
        return 0.0
    return float(math.sqrt(np.mean(np.square(values)) * TRADING_DAYS))


def build_bad_risk_features(cache_data: Mapping) -> pd.DataFrame:
    """Build causal bad-risk, bull-market, and recovery features.

    Every percentile excludes the current observation. Cross-asset downside
    correlation is measured only on negative equal-weight portfolio days.
    """
    cache = cache_data["cache"]
    dates = pd.DatetimeIndex(cache_data["dates"])
    symbols = list(cache_data["SYMS"])
    spy = pd.Series(cache_data["spy"]).sort_index()
    spy.index = pd.to_datetime(spy.index).tz_localize(None)

    downside_vol_history: list[float] = []
    downside_correlation_history: list[float] = []
    drawdown_speed_history: list[float] = []
    rebound_history: list[float] = []
    symbol_sigma_history: dict[str, list[float]] = {
        symbol: [] for symbol in symbols
    }
    rows: list[dict] = []

    for index in sorted(cache):
        _, sigma, return_window, cached_drawdown = cache[index]
        decision_date = pd.Timestamp(dates[index - 1]).tz_localize(None)
        sigma = np.asarray(sigma, dtype=float)
        asset_returns = pd.DataFrame(return_window, columns=symbols).tail(60)

        available_spy = spy.loc[:decision_date].dropna()
        spy_returns = available_spy.pct_change().dropna()
        current_price = float(available_spy.iloc[-1])
        sma50 = float(available_spy.iloc[-50:].mean())
        sma200 = (
            float(available_spy.iloc[-200:].mean())
            if len(available_spy) >= 200
            else sma50
        )
        prior_sma200 = (
            float(available_spy.iloc[-220:-20].mean())
            if len(available_spy) >= 220
            else sma200
        )
        sma200_slope20 = (
            sma200 / prior_sma200 - 1.0
            if abs(prior_sma200) > 1e-10
            else 0.0
        )
        momentum5 = _period_return(available_spy, 5)
        momentum10 = _period_return(available_spy, 10)
        momentum20 = _period_return(available_spy, 20)
        momentum60 = _period_return(available_spy, 60)

        downside_volatility = _semivolatility(spy_returns.tail(20))
        market_proxy = asset_returns.mean(axis=1)
        downside_days = asset_returns.loc[market_proxy < 0.0]
        downside_correlation = _average_correlation(downside_days)
        drawdown_speed = max(0.0, -momentum5, -momentum10)

        cumulative5 = (1.0 + asset_returns.tail(5)).prod() - 1.0
        cumulative20 = (1.0 + asset_returns.tail(20)).prod() - 1.0
        positive_breadth5 = float((cumulative5 > 0.0).mean())
        positive_breadth20 = float((cumulative20 > 0.0).mean())
        loss_breadth20 = 1.0 - positive_breadth20

        downside_vol_percentile = prior_percentile(
            downside_vol_history,
            downside_volatility,
        )
        downside_correlation_percentile = prior_percentile(
            downside_correlation_history,
            downside_correlation,
        )
        drawdown_speed_percentile = prior_percentile(
            drawdown_speed_history,
            drawdown_speed,
        )
        rebound_percentile = prior_percentile(rebound_history, momentum5)
        individual_sigma_percentiles = [
            prior_percentile(symbol_sigma_history[symbol], sigma[position])
            for position, symbol in enumerate(symbols)
        ]
        high_risk_breadth = float(
            np.mean(
                np.asarray(individual_sigma_percentiles, dtype=float) >= 0.80
            )
        )

        # Volatility alone is insufficient. A high score requires some
        # combination of downside volatility, downside co-movement, broad
        # losses, and a rapid benchmark decline.
        bad_risk_raw = float(
            0.35 * downside_vol_percentile
            + 0.25 * downside_correlation_percentile
            + 0.25 * loss_breadth20
            + 0.15 * drawdown_speed_percentile
        )
        bullish_regime = bool(
            current_price > sma50
            and sma50 > sma200
            and momentum20 > 0.0
            and momentum60 > 0.0
            and positive_breadth20 >= 0.55
        )

        # A hard emergency requires corroboration across independent channels.
        emergency_votes = sum(
            (
                downside_vol_percentile >= 0.90,
                downside_correlation_percentile >= 0.85,
                loss_breadth20 >= 0.80,
                drawdown_speed_percentile >= 0.90
                and drawdown_speed >= 0.03,
                high_risk_breadth >= 0.80,
            )
        )
        emergency = bool(
            emergency_votes >= 3
            or (
                float(cached_drawdown) <= -0.15
                and loss_breadth20 >= 0.75
                and downside_vol_percentile >= 0.80
            )
        )

        rebound_impulse = float(
            smoothstep((rebound_percentile - 0.65) / 0.25)
            * smoothstep((positive_breadth5 - 0.50) / 0.30)
        )
        high_vol_bull = bool(
            downside_vol_percentile >= 0.75 and bullish_regime
        )

        rows.append(
            {
                "index": int(index),
                "date": decision_date,
                "downside_volatility": downside_volatility,
                "downside_vol_percentile": downside_vol_percentile,
                "downside_correlation": downside_correlation,
                "downside_correlation_percentile": (
                    downside_correlation_percentile
                ),
                "drawdown_speed": drawdown_speed,
                "drawdown_speed_percentile": drawdown_speed_percentile,
                "loss_breadth20": loss_breadth20,
                "high_risk_breadth": high_risk_breadth,
                "positive_breadth5": positive_breadth5,
                "positive_breadth20": positive_breadth20,
                "momentum5": momentum5,
                "momentum10": momentum10,
                "momentum20": momentum20,
                "momentum60": momentum60,
                "price_vs_sma200": current_price / sma200 - 1.0,
                "sma200_slope20": sma200_slope20,
                "rebound_percentile": rebound_percentile,
                "rebound_impulse": rebound_impulse,
                "bad_risk_raw": bad_risk_raw,
                "bullish_regime": bullish_regime,
                "high_vol_bull": high_vol_bull,
                "emergency": emergency,
                "benchmark_drawdown": float(cached_drawdown),
            }
        )

        downside_vol_history.append(downside_volatility)
        downside_correlation_history.append(downside_correlation)
        drawdown_speed_history.append(drawdown_speed)
        rebound_history.append(momentum5)
        for position, symbol in enumerate(symbols):
            symbol_sigma_history[symbol].append(float(sigma[position]))

    return pd.DataFrame(rows).set_index("index", drop=False)


def run_regime_overlay(
    cache_data: Mapping,
    features: pd.DataFrame,
    policy: RegimePolicy,
    *,
    max_position: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Run one asymmetric regime-gated gross policy."""
    cache = cache_data["cache"]
    returns = pd.DataFrame(cache_data["rets"])
    dates = pd.DatetimeIndex(cache_data["dates"])
    symbols = list(cache_data["SYMS"])
    warmup = int(cache_data["WARMUP"])
    step = int(cache_data["STEP"])
    weights = np.full(len(symbols), 0.95 / len(symbols))
    path = pd.DataFrame(0.0, index=dates, columns=symbols)
    path.iloc[:warmup] = weights
    rows: list[dict] = []
    filtered_bad_risk = 0.50
    previous_raw = 0.50
    previous_gate = 0.0

    for index in sorted(cache):
        direction, sigma, return_window, _ = cache[index]
        feature = features.loc[index]
        raw_bad_risk = float(feature["bad_risk_raw"])
        if raw_bad_risk >= filtered_bad_risk:
            filtered_bad_risk = (
                (1.0 - policy.attack_speed) * filtered_bad_risk
                + policy.attack_speed * raw_bad_risk
            )
        else:
            filtered_bad_risk = (
                (1.0 - policy.release_speed) * filtered_bad_risk
                + policy.release_speed * raw_bad_risk
            )

        base_gate = smoothstep(
            (filtered_bad_risk - policy.activation_start)
            / (policy.activation_full - policy.activation_start)
        )
        emergency = bool(feature["emergency"])
        if emergency:
            base_gate = max(base_gate, 0.95)

        bull_guard_applied = bool(
            feature["bullish_regime"] and not emergency
        )
        if bull_guard_applied:
            base_gate *= policy.bull_attenuation

        stress_falling = bool(
            raw_bad_risk < previous_raw
            and filtered_bad_risk < previous_raw + 0.02
        )
        recovery_signal = 0.0
        if (
            previous_gate >= policy.recovery_threshold
            and stress_falling
            and not emergency
        ):
            recovery_signal = float(feature["rebound_impulse"])
        effective_gate = float(
            np.clip(
                base_gate
                * (1.0 - policy.recovery_strength * recovery_signal),
                0.0,
                1.0,
            )
        )

        relative_target = ranked_strategy_target(
            direction,
            strategy_strength=1.0,
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
        volatility_budget = (
            policy.stress_base_target * HEALTH_BUDGET_FACTOR
        )
        defensive_gross = float(
            np.clip(
                (
                    1.0
                    if predicted_full_volatility <= 1e-10
                    else volatility_budget / predicted_full_volatility
                ),
                policy.minimum_gross,
                1.0,
            )
        )

        # The critical change: volatility targeting is conditional on the
        # crisis gate. With no confirmed bad risk, target gross is exactly 100%.
        target_gross = float(
            1.0 - effective_gate * (1.0 - defensive_gross)
        )
        desired = relative_target * target_gross
        previous = weights.copy()
        weights = project_trade(
            desired,
            previous,
            max_position=max_position,
        )
        achieved_gross = float(weights.sum())
        achieved_volatility = float(
            math.sqrt(max(weights @ covariance_annual @ weights, 0.0))
        )
        path.iloc[index : min(index + step, len(dates))] = weights
        rows.append(
            {
                "date": pd.Timestamp(dates[index - 1]),
                "policy": policy.name,
                "bad_risk_raw": raw_bad_risk,
                "bad_risk_filtered": filtered_bad_risk,
                "base_crisis_gate": base_gate,
                "effective_crisis_gate": effective_gate,
                "bull_guard_applied": bull_guard_applied,
                "emergency": emergency,
                "recovery_signal": recovery_signal,
                "stress_falling": stress_falling,
                "high_vol_bull": bool(feature["high_vol_bull"]),
                "volatility_budget": volatility_budget,
                "predicted_full_volatility": predicted_full_volatility,
                "defensive_gross": defensive_gross,
                "target_gross": target_gross,
                "gross": achieved_gross,
                "achieved_volatility": achieved_volatility,
                "turnover": float(np.abs(weights - previous).sum()),
            }
        )
        previous_raw = raw_bad_risk
        previous_gate = effective_gate

    turnover = path.diff().abs().sum(axis=1).fillna(0.0)
    gross_return = (path.shift(1) * returns.reindex(path.index)).sum(axis=1)
    net = (gross_return - TRANSACTION_COST * turnover).iloc[warmup:]
    return net, pd.DataFrame(rows), path.iloc[warmup:]


def run_hybrid_overlay(
    cache_data: Mapping,
    old_stress: pd.DataFrame,
    features: pd.DataFrame,
    policy: HybridPolicy,
    *,
    max_position: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Add only bull/recovery overrides to the current Fusion policy."""
    cache = cache_data["cache"]
    returns = pd.DataFrame(cache_data["rets"])
    dates = pd.DatetimeIndex(cache_data["dates"])
    symbols = list(cache_data["SYMS"])
    warmup = int(cache_data["WARMUP"])
    step = int(cache_data["STEP"])
    weights = np.full(len(symbols), 0.95 / len(symbols))
    path = pd.DataFrame(0.0, index=dates, columns=symbols)
    path.iloc[:warmup] = weights
    rows: list[dict] = []
    previous_systemic = 0.0
    recovery_remaining = 0
    shock_timer = 0
    shock_trough = 0.0

    for index in sorted(cache):
        direction, sigma, return_window, _ = cache[index]
        feature = features.loc[index]
        systemic = float(old_stress.loc[index, "systemic_stress_release_70"])
        defence = systemic
        base_target = float(0.30 + defence * (0.15 - 0.30))
        volatility_budget = base_target * HEALTH_BUDGET_FACTOR
        relative_target = ranked_strategy_target(
            direction,
            strategy_strength=1.0,
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
        baseline_gross = float(
            np.clip(
                (
                    1.0
                    if predicted_full_volatility <= 1e-10
                    else volatility_budget / predicted_full_volatility
                ),
                0.35,
                1.0,
            )
        )

        emergency = bool(feature["emergency"])
        current_drawdown = float(feature["benchmark_drawdown"])
        shock_detected = bool(
            float(feature["drawdown_speed"]) >= policy.shock_speed_threshold
            and (systemic >= policy.recovery_trigger_systemic or emergency)
        )
        if shock_detected:
            if shock_timer <= 0:
                shock_trough = current_drawdown
            else:
                shock_trough = min(shock_trough, current_drawdown)
            shock_timer = policy.shock_window_rebalances
        elif shock_timer > 0:
            shock_timer -= 1
            shock_trough = min(shock_trough, current_drawdown)
        recovery_from_trough = current_drawdown - shock_trough

        if policy.shock_recovery_mode:
            recovery_trigger = bool(
                shock_timer > 0
                and systemic >= policy.recovery_trigger_systemic
                and float(feature["rebound_impulse"])
                >= policy.recovery_impulse_threshold
                and float(feature["momentum5"])
                >= policy.minimum_rebound_return
                and recovery_from_trough
                >= policy.minimum_recovery_from_trough
                and float(feature["sma200_slope20"])
                >= policy.maximum_sma200_decline
                and not emergency
            )
        else:
            recovery_trigger = bool(
                previous_systemic >= policy.recovery_trigger_systemic
                and systemic < previous_systemic
                and float(feature["rebound_impulse"])
                >= policy.recovery_impulse_threshold
                and not emergency
            )
        if recovery_trigger:
            recovery_remaining = policy.recovery_hold_rebalances
            if policy.shock_recovery_mode:
                shock_timer = 0
        elif (
            emergency
            or systemic > previous_systemic + 0.05
            or float(feature["momentum5"]) < -0.02
        ):
            recovery_remaining = 0

        recovery_active = recovery_remaining > 0
        rewarded_high_vol_bull = bool(
            bool(old_stress.loc[index, "bullish_regime"])
            and float(old_stress.loc[index, "spy_vol_percentile"]) >= 0.75
            and float(old_stress.loc[index, "correlation_percentile"]) < 0.75
            and float(old_stress.loc[index, "risk_breadth"]) < 0.50
        )
        bull_eligible = (
            rewarded_high_vol_bull
            if policy.bull_requires_rewarded_high_vol
            else bool(old_stress.loc[index, "bullish_regime"])
        )
        bull_override = bool(
            policy.bull_gross_floor > 0.0
            and bull_eligible
            and systemic <= policy.bull_systemic_ceiling
            and not emergency
        )
        target_gross = baseline_gross
        if bull_override:
            target_gross = max(target_gross, policy.bull_gross_floor)
        if recovery_active:
            target_gross = max(
                target_gross,
                policy.recovery_gross_floor,
            )
            recovery_remaining -= 1

        desired = relative_target * target_gross
        previous = weights.copy()
        weights = project_trade(
            desired,
            previous,
            max_position=max_position,
        )
        achieved_gross = float(weights.sum())
        achieved_volatility = float(
            math.sqrt(max(weights @ covariance_annual @ weights, 0.0))
        )
        path.iloc[index : min(index + step, len(dates))] = weights
        rows.append(
            {
                "date": pd.Timestamp(dates[index - 1]),
                "policy": policy.name,
                "systemic_stress": systemic,
                "base_target": base_target,
                "volatility_budget": volatility_budget,
                "predicted_full_volatility": predicted_full_volatility,
                "baseline_gross": baseline_gross,
                "target_gross": target_gross,
                "gross": achieved_gross,
                "achieved_volatility": achieved_volatility,
                "bull_override": bull_override,
                "recovery_trigger": recovery_trigger,
                "recovery_active": recovery_active,
                "shock_detected": shock_detected,
                "shock_timer": shock_timer,
                "recovery_from_trough": recovery_from_trough,
                "emergency": emergency,
                "high_vol_bull": bool(feature["high_vol_bull"]),
                "rewarded_high_vol_bull": rewarded_high_vol_bull,
                "turnover": float(np.abs(weights - previous).sum()),
            }
        )
        previous_systemic = systemic

    turnover = path.diff().abs().sum(axis=1).fillna(0.0)
    gross_return = (path.shift(1) * returns.reindex(path.index)).sum(axis=1)
    net = (gross_return - TRANSACTION_COST * turnover).iloc[warmup:]
    return net, pd.DataFrame(rows), path.iloc[warmup:]


def run_uniform_hybrid_overlay(
    cache_data: Mapping,
    old_stress: pd.DataFrame,
    features: pd.DataFrame,
    policy: HybridPolicy,
    baseline_decisions: pd.DataFrame,
    baseline_path: pd.DataFrame,
    *,
    max_position: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Apply recovery as a scalar to the current Fusion relative portfolio.

    This is the strict gross-only implementation: at every date the candidate
    has exactly the same relative stock weights as current Fusion. It prevents
    the position projection's minimum-trade path dependence from masquerading
    as stock-selection alpha months after a recovery event.
    """
    cache = cache_data["cache"]
    returns = pd.DataFrame(cache_data["rets"])
    dates = pd.DatetimeIndex(cache_data["dates"])
    symbols = list(cache_data["SYMS"])
    step = int(cache_data["STEP"])
    initial_weights = np.full(len(symbols), 0.95 / len(symbols))
    path = baseline_path.copy()
    rows: list[dict] = []
    previous_systemic = 0.0
    recovery_remaining = 0
    shock_timer = 0
    shock_trough = 0.0
    baseline_rows = baseline_decisions.reset_index(drop=True)
    previous_candidate_weights = initial_weights.copy()

    for position, index in enumerate(sorted(cache)):
        feature = features.loc[index]
        baseline_row = baseline_rows.iloc[position]
        systemic = float(old_stress.loc[index, "systemic_stress_release_70"])
        emergency = bool(feature["emergency"])
        current_drawdown = float(feature["benchmark_drawdown"])
        shock_detected = bool(
            float(feature["drawdown_speed"]) >= policy.shock_speed_threshold
            and (systemic >= policy.recovery_trigger_systemic or emergency)
        )
        if shock_detected:
            if shock_timer <= 0:
                shock_trough = current_drawdown
            else:
                shock_trough = min(shock_trough, current_drawdown)
            shock_timer = policy.shock_window_rebalances
        elif shock_timer > 0:
            shock_timer -= 1
            shock_trough = min(shock_trough, current_drawdown)
        recovery_from_trough = current_drawdown - shock_trough

        recovery_trigger = bool(
            shock_timer > 0
            and systemic >= policy.recovery_trigger_systemic
            and float(feature["rebound_impulse"])
            >= policy.recovery_impulse_threshold
            and float(feature["momentum5"])
            >= policy.minimum_rebound_return
            and recovery_from_trough >= policy.minimum_recovery_from_trough
            and float(feature["sma200_slope20"])
            >= policy.maximum_sma200_decline
            and not emergency
        )
        if recovery_trigger:
            recovery_remaining = policy.recovery_hold_rebalances
            shock_timer = 0
        elif (
            emergency
            or systemic > previous_systemic + 0.05
            or float(feature["momentum5"]) < -0.02
        ):
            recovery_remaining = 0

        recovery_active = recovery_remaining > 0
        rewarded_high_vol_bull = bool(
            bool(old_stress.loc[index, "bullish_regime"])
            and float(old_stress.loc[index, "spy_vol_percentile"]) >= 0.75
            and float(old_stress.loc[index, "correlation_percentile"]) < 0.75
            and float(old_stress.loc[index, "risk_breadth"]) < 0.50
        )
        interval_dates = dates[index : min(index + step, len(dates))]
        baseline_weights = baseline_path.reindex(interval_dates).iloc[0]
        baseline_gross = float(baseline_weights.sum())
        relative_weights = (
            baseline_weights.to_numpy(dtype=float) / baseline_gross
            if baseline_gross > 1e-10
            else np.full(len(symbols), 1.0 / len(symbols))
        )
        target_gross = baseline_gross
        if recovery_active:
            target_gross = max(
                baseline_gross,
                policy.recovery_gross_floor,
            )
            recovery_remaining -= 1
        positive = relative_weights > 1e-10
        feasible_low = 0.0
        feasible_high = 1.0
        if positive.any():
            feasible_low = max(
                0.0,
                float(
                    np.max(
                        (
                            previous_candidate_weights[positive] - 0.05
                        )
                        / relative_weights[positive]
                    )
                ),
            )
            feasible_high = min(
                1.0,
                float(
                    np.min(
                        (
                            previous_candidate_weights[positive] + 0.05
                        )
                        / relative_weights[positive]
                    )
                ),
                float(
                    np.min(
                        max_position / relative_weights[positive]
                    )
                ),
            )
        if feasible_low > feasible_high + 1e-10:
            raise RuntimeError(
                "no scalar gross satisfies the position-change constraints"
            )
        achieved_gross = float(
            np.clip(target_gross, feasible_low, feasible_high)
        )
        scale = (
            achieved_gross / baseline_gross
            if baseline_gross > 1e-10
            else 1.0
        )
        path.loc[interval_dates] = (
            baseline_path.reindex(interval_dates).to_numpy(dtype=float) * scale
        )
        previous_candidate_weights = relative_weights * achieved_gross
        rows.append(
            {
                "date": pd.Timestamp(dates[index - 1]),
                "policy": policy.name,
                "systemic_stress": systemic,
                "base_target": float(baseline_row["base_target"]),
                "volatility_budget": float(
                    baseline_row["volatility_budget"]
                ),
                "predicted_full_volatility": float(
                    baseline_row["predicted_full_volatility"]
                ),
                "baseline_gross": baseline_gross,
                "target_gross": target_gross,
                "gross": achieved_gross,
                "feasible_gross_low": feasible_low,
                "feasible_gross_high": feasible_high,
                "bull_override": False,
                "recovery_trigger": recovery_trigger,
                "recovery_active": recovery_active,
                "shock_detected": shock_detected,
                "shock_timer": shock_timer,
                "recovery_from_trough": recovery_from_trough,
                "emergency": emergency,
                "high_vol_bull": bool(feature["high_vol_bull"]),
                "rewarded_high_vol_bull": rewarded_high_vol_bull,
                "turnover": 0.0,
            }
        )
        previous_systemic = systemic

    turnover = path.diff().abs().sum(axis=1).fillna(0.0)
    if len(path):
        turnover.iloc[0] = float(
            np.abs(path.iloc[0].to_numpy(dtype=float) - initial_weights).sum()
        )
    lagged = path.shift(1)
    if len(lagged):
        lagged.iloc[0] = initial_weights
    gross_return = (lagged * returns.reindex(path.index)).sum(axis=1)
    net = gross_return - TRANSACTION_COST * turnover
    decisions = pd.DataFrame(rows)
    decisions["turnover"] = [
        float(turnover.reindex(
            dates[index : min(index + step, len(dates))]
        ).sum())
        for index in sorted(cache)
    ]
    return net, decisions, path


def _total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def metric_block(returns: pd.Series) -> dict:
    output = metrics(returns)
    output["total_return"] = _total_return(returns)
    output["observations"] = int(len(returns))
    return output


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
            ("2025-01-01", "2025-12-31", "diagnostic_2025"),
            ("2026-01-01", "2026-07-24", "diagnostic_2026_to_july_24"),
            ("2024-01-01", "2026-07-24", "forward_full"),
        ]
    if label == "external_etf":
        return [
            ("2000-01-01", "2013-12-31", "external_2000_2013"),
            ("2014-01-01", "2023-12-31", "external_2014_2023"),
            ("2000-01-01", "2023-12-31", "external_full"),
        ]
    raise ValueError(f"unknown cache label: {label}")


CRISIS_WINDOWS = {
    "2015_2016_adjustment": ("2015-07-01", "2016-02-29"),
    "2018_q4": ("2018-10-01", "2018-12-31"),
    "covid": ("2020-02-19", "2020-04-30"),
    "2022_bear": ("2022-01-03", "2022-12-30"),
    "2025_v_rebound": ("2025-04-08", "2025-06-30"),
}


def _evaluate_period(
    series: Mapping[str, pd.Series],
    *,
    start: str,
    end: str,
    bootstrap_samples: int,
) -> dict:
    selected = {
        name: values.loc[pd.Timestamp(start) : pd.Timestamp(end)].dropna()
        for name, values in series.items()
    }
    daily = selected["daily_strategy"]
    current = selected["current_fusion"]
    return {
        "period": [start, end],
        "metrics": {
            name: metric_block(values)
            for name, values in selected.items()
            if len(values)
        },
        "vs_daily_strategy": {
            name: paired_block_bootstrap(
                daily,
                values,
                samples=bootstrap_samples,
                block=20,
                seed=20260726,
            )
            for name, values in selected.items()
            if name != "daily_strategy" and len(values)
        },
        "vs_current_fusion": {
            name: paired_block_bootstrap(
                current,
                values,
                samples=bootstrap_samples,
                block=20,
                seed=20260726,
            )
            for name, values in selected.items()
            if name not in {"current_fusion"} and len(values)
        },
    }


def evaluate_cache(
    cache_path: Path,
    label: str,
    *,
    bootstrap_samples: int,
    policy_names: set[str] | None = None,
) -> dict:
    with cache_path.open("rb") as handle:
        cache_data = pickle.load(handle)
    max_position = 0.25 if len(cache_data["SYMS"]) <= 10 else 0.20
    features = build_bad_risk_features(cache_data)
    old_stress = add_systemic_stress(build_stress_series(cache_data))

    daily = run_daily_strategy(cache_data, max_position)
    current, current_decisions, current_path = run_overlay(
        cache_data,
        old_stress,
        CURRENT_POLICY,
        max_position=max_position,
    )
    series: dict[str, pd.Series] = {
        "daily_strategy": daily,
        "current_fusion": current,
    }
    decisions: dict[str, pd.DataFrame] = {
        "current_fusion": current_decisions
    }
    paths: dict[str, pd.DataFrame] = {"current_fusion": current_path}

    for policy in POLICIES:
        if policy_names is not None and policy.name not in policy_names:
            continue
        policy_returns, policy_decisions, policy_path = run_regime_overlay(
            cache_data,
            features,
            policy,
            max_position=max_position,
        )
        series[policy.name] = policy_returns
        decisions[policy.name] = policy_decisions
        paths[policy.name] = policy_path

    for policy in HYBRID_POLICIES:
        if policy_names is not None and policy.name not in policy_names:
            continue
        if policy.pure_uniform_gross:
            policy_returns, policy_decisions, policy_path = (
                run_uniform_hybrid_overlay(
                    cache_data,
                    old_stress,
                    features,
                    policy,
                    current_decisions,
                    current_path,
                    max_position=max_position,
                )
            )
        else:
            policy_returns, policy_decisions, policy_path = run_hybrid_overlay(
                cache_data,
                old_stress,
                features,
                policy,
                max_position=max_position,
            )
        series[policy.name] = policy_returns
        decisions[policy.name] = policy_decisions
        paths[policy.name] = policy_path

    periods = {
        name: _evaluate_period(
            series,
            start=start,
            end=end,
            bootstrap_samples=bootstrap_samples,
        )
        for start, end, name in periods_for(label)
    }
    crises = {}
    available_start = min(values.index.min() for values in series.values())
    available_end = max(values.index.max() for values in series.values())
    for name, (start, end) in CRISIS_WINDOWS.items():
        if (
            pd.Timestamp(end) < available_start
            or pd.Timestamp(start) > available_end
        ):
            continue
        crises[name] = _evaluate_period(
            series,
            start=start,
            end=end,
            bootstrap_samples=bootstrap_samples,
        )

    summaries = {}
    for name, frame in decisions.items():
        policy_path = paths[name]
        changes = policy_path.diff().abs().fillna(0.0)
        summary = {
            "mean_gross": float(frame["gross"].mean()),
            "mean_turnover_per_rebalance": float(frame["turnover"].mean()),
            "maximum_position": float(policy_path.max().max()),
            "maximum_position_change": float(changes.max().max()),
            "sub_minimum_trade_count": int(
                ((changes > 1e-8) & (changes < 0.01 - 1e-7)).sum().sum()
            ),
        }
        if "effective_crisis_gate" in frame:
            summary.update(
                {
                    "mean_effective_crisis_gate": float(
                        frame["effective_crisis_gate"].mean()
                    ),
                    "crisis_gate_above_50_share": float(
                        (frame["effective_crisis_gate"] >= 0.50).mean()
                    ),
                    "bull_guard_share": float(
                        frame["bull_guard_applied"].mean()
                    ),
                    "recovery_share": float(
                        (frame["recovery_signal"] > 0.0).mean()
                    ),
                    "emergency_share": float(frame["emergency"].mean()),
                    "high_vol_bull_mean_gross": float(
                        frame.loc[frame["high_vol_bull"], "gross"].mean()
                    )
                    if frame["high_vol_bull"].any()
                    else float("nan"),
                }
            )
        elif "recovery_trigger" in frame:
            baseline_relative = current_path.div(
                current_path.sum(axis=1),
                axis=0,
            )
            candidate_relative = policy_path.div(
                policy_path.sum(axis=1),
                axis=0,
            )
            summary.update(
                {
                    "mean_baseline_gross": float(
                        frame["baseline_gross"].mean()
                    ),
                    "bull_override_share": float(
                        frame["bull_override"].mean()
                    ),
                    "recovery_trigger_count": int(
                        frame["recovery_trigger"].sum()
                    ),
                    "recovery_active_share": float(
                        frame["recovery_active"].mean()
                    ),
                    "emergency_share": float(frame["emergency"].mean()),
                    "rewarded_high_vol_bull_share": float(
                        frame["rewarded_high_vol_bull"].mean()
                    ),
                    "rewarded_high_vol_bull_mean_gross": (
                        float(
                            frame.loc[
                                frame["rewarded_high_vol_bull"],
                                "gross",
                            ].mean()
                        )
                        if frame["rewarded_high_vol_bull"].any()
                        else float("nan")
                    ),
                    "recovery_events": [
                        {
                            "date": pd.Timestamp(row["date"])
                            .date()
                            .isoformat(),
                            "baseline_gross": float(row["baseline_gross"]),
                            "target_gross": float(row["target_gross"]),
                            "achieved_gross": float(row["gross"]),
                            "recovery_from_trough": float(
                                row["recovery_from_trough"]
                            ),
                        }
                        for _, row in frame.loc[
                            frame["recovery_trigger"]
                        ].iterrows()
                    ],
                    "maximum_relative_weight_deviation": float(
                        (candidate_relative - baseline_relative)
                        .abs()
                        .max()
                        .max()
                    ),
                }
            )
        summaries[name] = summary

    return {
        "label": label,
        "cache": str(cache_path),
        "symbols": list(cache_data["SYMS"]),
        "feature_summary": {
            "mean_bad_risk_raw": float(features["bad_risk_raw"].mean()),
            "bullish_regime_share": float(features["bullish_regime"].mean()),
            "high_vol_bull_share": float(features["high_vol_bull"].mean()),
            "emergency_share": float(features["emergency"].mean()),
        },
        "decision_summary": summaries,
        "periods": periods,
        "crises": crises,
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
        default=ROOT / "reports" / "regime_gated_fusion_research.json",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=3000,
        help="Moving-block bootstrap samples per comparison.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        help="Evaluate only the named candidate policy; may be repeated.",
    )
    args = parser.parse_args()
    available_policies = {
        policy.name for policy in (*POLICIES, *HYBRID_POLICIES)
    }
    requested_policies = set(args.policy) if args.policy else None
    unknown = (
        requested_policies - available_policies
        if requested_policies is not None
        else set()
    )
    if unknown:
        parser.error("unknown policy: " + ", ".join(sorted(unknown)))

    results = [
        evaluate_cache(
            Path(path),
            label,
            bootstrap_samples=args.bootstrap_samples,
            policy_names=requested_policies,
        )
        for label, path in args.cache
    ]
    payload = {
        "method": "regime_gated_asymmetric_gross_overlay",
        "current_policy": asdict(CURRENT_POLICY),
        "selected_policy": SELECTED_POLICY_NAME,
        "candidate_policies": [
            asdict(policy)
            for policy in POLICIES
            if requested_policies is None or policy.name in requested_policies
        ],
        "hybrid_policies": [
            asdict(policy)
            for policy in HYBRID_POLICIES
            if requested_policies is None or policy.name in requested_policies
        ],
        "requested_policies": (
            sorted(requested_policies)
            if requested_policies is not None
            else "all"
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
