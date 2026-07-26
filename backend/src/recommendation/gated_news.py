"""Gated news residual policy with external HAR-X position control.

The learned model is deliberately narrow: it may only make a small residual
change to Daily Strategy when recent news exists and its estimated advantage
clears a validation-selected confidence gate.  HAR-X risk and Portfolio Health
are not model features and never vote on direction.  They control the risky
gross exposure and cross-sectional allocation in a deterministic optimizer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import DATA_DIR
from src.interfaces import AssetSignal, ProposedTrade, Recommendation
from src.recommendation import fusion

MODEL_DIR = DATA_DIR / "processed" / "decision_model_candidate_gated_news"
METADATA_PATH = MODEL_DIR / "metadata.json"
MODEL_PATH = MODEL_DIR / "q_model.joblib"
MODEL_VERSION = "gated-news-risk-control-v1"
BASELINE_MODEL_VERSION = "strategy-external-harx-news-risk-v1"
HORIZON = 5
EPS = 1e-10
ACTIONS = np.array([-1.0, 0.0, 1.0])

# Grinold-Kahn alpha scaling: alpha = IC * sigma_horizon * z.  The optimiser
# needs the Daily Strategy rank expressed as an expected excess return, and
# that conversion is where an unvalidated signal does the most damage — the
# assumed information coefficient sets the tilt size directly.
#
# Measured cross-sectional rank IC of the Daily Strategy direction against the
# five-session forward return is ~0.01-0.02 and not distinguishable from zero
# (t~1) on both the 21-symbol FNSPID panel and the 165-symbol wide panel;
# quintile forward returns are flat.  The previous hardcoded 0.010 expected
# return per unit of direction implied an IC near 0.14 — about ten times the
# measured value — so the optimiser tilted on noise and paid turnover for it.
# Raise this only with a walk-forward measurement that supports it.
STRATEGY_INFORMATION_COEFFICIENT = 0.02

CORE_FEATURES = [
    "strategy_score",
    "mom_20d",
    "mom_60d",
    "strategy_trend",
    "strategy_sharpe",
    "strategy_drawdown",
    "rsi_scaled",
    "benchmark_ret_1d",
]
NEWS_FEATURES = [
    "news_score_5d",
    "sent_positive_share",
    "sent_negative_share",
    "sent_extreme_share",
]
FEATURES = CORE_FEATURES + NEWS_FEATURES

_ARTIFACT_CACHE: tuple[int, dict, object] | None = None


@dataclass
class AllocationResult:
    weights: np.ndarray
    cash_weight: float
    success: bool
    message: str
    turnover: float
    maximum_position: float
    maximum_change: float
    minimum_active_trade: float
    target_gross: float
    predicted_annual_volatility: float
    target_annual_volatility: float
    effective_risk_aversion: float


@dataclass
class GatedNewsDecision:
    recommendation: Recommendation
    metadata: dict


def state_action_matrix(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Add action and state/action interactions to a feature matrix."""
    state_values = np.asarray(states, dtype=float)
    action_values = np.asarray(actions, dtype=float).reshape(-1, 1)
    return np.column_stack(
        [state_values, action_values, state_values * action_values]
    )


def bounded_state_matrix(
    frame: pd.DataFrame,
    feature_order: Sequence[str],
    bounds: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Apply training-time causal winsorisation in the recorded feature order."""
    values = frame.reindex(columns=feature_order).copy()
    for column in feature_order:
        lower, upper = bounds[column]
        values[column] = (
            pd.to_numeric(values[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(float(lower), float(upper))
        )
    return values.to_numpy(dtype=float)


def predict_q_values(
    model: object,
    frame: pd.DataFrame,
    feature_order: Sequence[str],
    bounds: Mapping[str, Sequence[float]],
) -> np.ndarray:
    states = bounded_state_matrix(frame, feature_order, bounds)
    predictions = []
    for action in ACTIONS:
        action_column = np.full(len(states), action)
        predictions.append(
            np.asarray(
                model.predict(state_action_matrix(states, action_column)),
                dtype=float,
            )
        )
    return np.column_stack(predictions)


def gated_direction_signal(
    model: object,
    frame: pd.DataFrame,
    feature_order: Sequence[str],
    bounds: Mapping[str, Sequence[float]],
    *,
    residual_cap: float,
    q_margin: float,
    news_available: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Return Daily Strategy plus an optional, confidence-gated news residual."""
    strategy = (
        pd.to_numeric(frame["strategy_score"], errors="coerce")
        .fillna(0.0)
        .clip(-1.0, 1.0)
        .to_numpy(dtype=float)
    )
    q_values = predict_q_values(model, frame, feature_order, bounds)
    best_indices = np.argmax(q_values, axis=1)
    learned_actions = ACTIONS[best_indices]
    hold_q = q_values[:, 1]
    best_q = q_values[np.arange(len(q_values)), best_indices]
    advantages = best_q - hold_q
    news_mask = np.asarray(news_available, dtype=bool)
    applied = news_mask & (advantages >= float(q_margin))
    learned_actions = np.where(applied, learned_actions, 0.0)
    residual = float(residual_cap) * learned_actions
    combined = np.clip(strategy + residual, -1.0, 1.0)
    return combined, {
        "news_available_share": float(news_mask.mean()) if len(news_mask) else 0.0,
        "news_applied_share": float(applied.mean()) if len(applied) else 0.0,
        "increase_share": float(np.mean(learned_actions > 0)),
        "reduce_share": float(np.mean(learned_actions < 0)),
        "hold_share": float(np.mean(learned_actions == 0)),
        "mean_q_advantage": float(np.mean(advantages)),
        "mean_abs_news_residual": float(np.mean(np.abs(residual))),
    }


def _repair_correlation(
    return_history: pd.DataFrame,
    symbols: Sequence[str],
) -> np.ndarray:
    values = return_history.reindex(columns=list(symbols)).tail(126)
    correlation = values.corr(min_periods=40).to_numpy(dtype=float)
    correlation = np.where(np.isfinite(correlation), correlation, 0.0)
    np.fill_diagonal(correlation, 1.0)
    correlation = 0.90 * correlation + 0.10 * np.eye(len(symbols))
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1e-5)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.maximum(np.diag(repaired), EPS))
    return repaired / np.outer(scale, scale)


def _normalise_portfolio(
    previous_weights: np.ndarray,
    previous_cash: float,
) -> tuple[np.ndarray, float]:
    weights = np.maximum(np.asarray(previous_weights, dtype=float), 0.0)
    cash = max(float(previous_cash), 0.0)
    total = float(weights.sum() + cash)
    if total <= EPS:
        weights = np.full(len(weights), 1.0 / max(len(weights), 1))
        cash = 0.0
    else:
        weights /= total
        cash /= total
    return weights, cash


def strategy_alpha(
    direction: np.ndarray,
    sigma_daily: np.ndarray,
    *,
    information_coefficient: float = STRATEGY_INFORMATION_COEFFICIENT,
) -> np.ndarray:
    """Convert a cross-sectional direction rank into an expected horizon return.

    Grinold-Kahn: ``alpha = IC * sigma_h * z``.  ``direction`` is standardised
    cross-sectionally so the scale of the incoming rank cannot smuggle in an
    implied information coefficient of its own.  A degenerate cross-section
    (every asset ranked alike) yields zero alpha, which leaves the optimiser
    with the covariance term only — the correct behaviour when the signal
    carries no cross-sectional information.
    """
    direction = np.asarray(direction, dtype=float)
    direction = np.nan_to_num(direction, nan=0.0)
    spread = float(np.std(direction))
    if spread <= EPS:
        return np.zeros_like(direction)
    standardised = (direction - float(np.mean(direction))) / spread
    horizon_sigma = np.asarray(sigma_daily, dtype=float) * math.sqrt(HORIZON)
    return float(information_coefficient) * horizon_sigma * standardised


def risk_controlled_allocation(
    symbols: Sequence[str],
    expected_return_5d: np.ndarray,
    sigma_daily: np.ndarray,
    return_history: pd.DataFrame,
    previous_weights: np.ndarray,
    previous_cash: float,
    *,
    health_score: float = 50.0,
    base_risk_aversion: float = 6.0,
    base_target_annual_volatility: float = 0.18,
    turnover_penalty: float = 0.0025,
    max_position: float = 0.20,
    max_change: float = 0.05,
    min_trade: float = 0.01,
    minimum_gross: float = 0.35,
) -> AllocationResult:
    """Use HAR-X sigma externally to choose risky weights and cash exposure."""
    symbols = list(symbols)
    if len(symbols) < 5:
        raise ValueError("at least five assets are required")
    previous, cash_before = _normalise_portfolio(
        previous_weights,
        previous_cash,
    )
    expected = np.nan_to_num(
        np.asarray(expected_return_5d, dtype=float),
        nan=0.0,
    )
    sigma = np.asarray(sigma_daily, dtype=float)
    valid_sigma = sigma[np.isfinite(sigma) & (sigma > 0)]
    fallback_sigma = float(np.median(valid_sigma)) if len(valid_sigma) else 0.02
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback_sigma)

    correlation = _repair_correlation(return_history, symbols)
    covariance_daily = np.diag(sigma) @ correlation @ np.diag(sigma)
    covariance_horizon = covariance_daily * HORIZON
    horizon_sigma = sigma * math.sqrt(HORIZON)
    expected = np.clip(expected, -horizon_sigma, horizon_sigma)

    health = float(np.clip(health_score, 0.0, 100.0))
    health_budget_factor = 0.65 + 0.35 * health / 100.0
    target_annual_volatility = (
        float(base_target_annual_volatility) * health_budget_factor
    )
    effective_risk_aversion = float(base_risk_aversion) * (
        1.0 + 0.75 * (1.0 - health / 100.0)
    )

    def base_objective(weights: np.ndarray) -> float:
        return float(
            -(expected @ weights)
            + 0.5
            * effective_risk_aversion
            * (weights @ covariance_horizon @ weights)
        )

    relative_previous = previous / max(previous.sum(), EPS)
    relative_start = np.clip(relative_previous, 0.0, max_position)
    relative_start /= max(relative_start.sum(), EPS)
    relative_result = minimize(
        base_objective,
        x0=relative_start,
        method="SLSQP",
        bounds=[(0.0, max_position)] * len(symbols),
        constraints=[
            {"type": "eq", "fun": lambda candidate: candidate.sum() - 1.0}
        ],
        options={"maxiter": 250, "ftol": 1e-10},
    )
    relative = (
        relative_result.x
        if relative_result.success
        else np.full(len(symbols), 1.0 / len(symbols))
    )
    relative = np.maximum(relative, 0.0)
    relative /= max(relative.sum(), EPS)
    predicted_annual_volatility = float(
        math.sqrt(
            max(relative @ covariance_daily @ relative, 0.0) * 252.0
        )
    )
    raw_target_gross = (
        1.0
        if predicted_annual_volatility <= EPS
        else target_annual_volatility / predicted_annual_volatility
    )
    raw_target_gross = float(
        np.clip(raw_target_gross, minimum_gross, 1.0)
    )

    lower = np.maximum(0.0, previous - max_change)
    upper = np.minimum(max_position, previous + max_change)
    upper = np.where(previous > max_position, previous, upper)
    feasible_low = float(lower.sum())
    feasible_high = float(min(1.0, upper.sum()))
    target_gross = float(
        np.clip(raw_target_gross, feasible_low, feasible_high)
    )

    def objective(weights: np.ndarray) -> float:
        delta = weights - previous
        cash_after = 1.0 - float(weights.sum())
        smooth_turnover = (
            np.sum(np.sqrt(delta * delta + 1e-10))
            + math.sqrt((cash_after - cash_before) ** 2 + 1e-10)
        )
        return float(
            base_objective(weights)
            + float(turnover_penalty) * smooth_turnover
        )

    start = relative * target_gross
    start = np.clip(start, lower, upper)
    if abs(start.sum() - target_gross) > 1e-7:
        start = previous.copy()
    constraints = [
        {
            "type": "eq",
            "fun": lambda candidate: candidate.sum() - target_gross,
        }
    ]
    result = minimize(
        objective,
        x0=start,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-10},
    )
    weights = result.x if result.success else previous.copy()

    fixed = np.zeros(len(previous), dtype=bool)
    for _ in range(len(previous) + 1):
        small = (
            (np.abs(weights - previous) < min_trade - 1e-7)
            & (np.abs(weights - previous) > 1e-5)
        )
        new_fixed = small & ~fixed
        if not result.success or not new_fixed.any():
            break
        fixed |= new_fixed
        fixed_lower = lower.copy()
        fixed_upper = upper.copy()
        fixed_lower[fixed] = previous[fixed]
        fixed_upper[fixed] = previous[fixed]
        if (
            fixed_lower.sum() > target_gross + 1e-8
            or fixed_upper.sum() < target_gross - 1e-8
        ):
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
        result = second
        weights = second.x

    if result.success:
        weights = np.maximum(weights, 0.0)
        # Cash is an explicit buffer, so any remaining sub-threshold numerical
        # changes can be cancelled exactly instead of being silently traded.
        sub_threshold = (
            (np.abs(weights - previous) < min_trade - 1e-7)
            & (np.abs(weights - previous) > 1e-10)
        )
        weights[sub_threshold] = previous[sub_threshold]
        cash_after = max(0.0, 1.0 - float(weights.sum()))
        target_gross = float(weights.sum())
    else:
        weights = previous.copy()
        cash_after = cash_before
        target_gross = float(previous.sum())
    delta = weights - previous
    cash_delta = cash_after - cash_before
    active = np.abs(delta) >= min_trade - 1e-7
    turnover = 0.5 * (float(np.abs(delta).sum()) + abs(cash_delta))
    return AllocationResult(
        weights=weights,
        cash_weight=cash_after,
        success=bool(result.success),
        message=str(result.message),
        turnover=turnover,
        maximum_position=float(weights.max()),
        maximum_change=float(np.abs(delta).max()),
        minimum_active_trade=(
            float(np.abs(delta[active]).min()) if active.any() else 0.0
        ),
        target_gross=target_gross,
        predicted_annual_volatility=predicted_annual_volatility,
        target_annual_volatility=target_annual_volatility,
        effective_risk_aversion=effective_risk_aversion,
    )


def _load_artifact() -> tuple[Optional[dict], Optional[object]]:
    global _ARTIFACT_CACHE
    if not METADATA_PATH.exists() or not MODEL_PATH.exists():
        return None, None
    modified = max(
        METADATA_PATH.stat().st_mtime_ns,
        MODEL_PATH.stat().st_mtime_ns,
    )
    if _ARTIFACT_CACHE is None or _ARTIFACT_CACHE[0] != modified:
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            model = joblib.load(MODEL_PATH)
            _ARTIFACT_CACHE = (modified, metadata, model)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None, None
    return _ARTIFACT_CACHE[1], _ARTIFACT_CACHE[2]


def model_available(*, require_promoted: bool = True) -> bool:
    metadata, model = _load_artifact()
    if metadata is None or model is None:
        return False
    return (
        not require_promoted
        or metadata.get("promotion_status") == "promoted"
    )


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    relative = gain / loss.replace(0.0, np.nan)
    value = (100.0 - 100.0 / (1.0 + relative)).iloc[-1]
    return float(value) if np.isfinite(value) else 50.0


def _runtime_features(
    symbols: Sequence[str],
    history: pd.DataFrame,
    signals: Iterable[AssetSignal],
    news_events: Iterable[object],
) -> tuple[pd.DataFrame, np.ndarray]:
    signal_map = {str(signal.symbol): signal for signal in signals}
    event_list = list(news_events)
    benchmark_return = (
        float(history["SPY"].pct_change().iloc[-1])
        if "SPY" in history and len(history["SPY"].dropna()) > 1
        else 0.0
    )
    raw_rows = []
    for symbol in symbols:
        close = history[symbol].dropna() if symbol in history else pd.Series()
        signal = signal_map.get(symbol)
        indicators = signal.indicators if signal is not None else {}
        aggregate = fusion.aggregate_news(event_list, symbol)
        relevant_events = [
            event
            for event in event_list
            if symbol
            in {
                str(item).upper()
                for item in (
                    getattr(event, "affected_symbols", [])
                    if not isinstance(event, Mapping)
                    else event.get("affected_symbols", [])
                )
            }
        ]
        sentiments = np.asarray(
            [
                float(
                    getattr(event, "sentiment", 0.0)
                    if not isinstance(event, Mapping)
                    else event.get("sentiment", 0.0)
                )
                for event in relevant_events
            ],
            dtype=float,
        )
        mom20 = (
            float(close.iloc[-1] / close.iloc[-21] - 1.0)
            if len(close) > 20
            else 0.0
        )
        mom60 = (
            float(close.iloc[-1] / close.iloc[-61] - 1.0)
            if len(close) > 60
            else float(indicators.get("momentum", 0.0))
        )
        raw_rows.append(
            {
                "symbol": symbol,
                "mom_20d": mom20,
                "mom_60d": mom60,
                "strategy_trend": float(indicators.get("trend", 0.5)),
                "strategy_sharpe": float(indicators.get("sharpe", 0.0)),
                "strategy_drawdown": float(indicators.get("drawdown", 0.0)),
                "rsi_scaled": (_rsi(close) - 50.0) / 50.0
                if len(close) > 15
                else 0.0,
                "benchmark_ret_1d": benchmark_return,
                "news_score_5d": float(aggregate.score),
                "sent_positive_share": float(np.mean(sentiments > 0.15))
                if len(sentiments)
                else 0.0,
                "sent_negative_share": float(np.mean(sentiments < -0.15))
                if len(sentiments)
                else 0.0,
                "sent_extreme_share": float(np.mean(np.abs(sentiments) >= 0.6))
                if len(sentiments)
                else 0.0,
                "news_available": float(aggregate.unique_articles > 0),
            }
        )
    frame = pd.DataFrame(raw_rows).set_index("symbol")
    ranks = frame[
        [
            "mom_60d",
            "strategy_trend",
            "strategy_sharpe",
            "strategy_drawdown",
        ]
    ].rank(pct=True)
    direction_01 = (
        0.30 * ranks["mom_60d"]
        + 0.25 * ranks["strategy_trend"]
        + 0.20 * ranks["strategy_sharpe"]
        + 0.10 * ranks["strategy_drawdown"]
    ) / 0.85
    frame["strategy_score"] = (2.0 * direction_01 - 1.0).clip(-1, 1)
    enhanced_direction = pd.Series(
        {
            symbol: (
                float(
                    signal_map[symbol].indicators.get(
                        "enhanced_direction",
                        np.nan,
                    )
                )
                if symbol in signal_map
                else np.nan
            )
            for symbol in symbols
        }
    ).reindex(frame.index)
    frame["strategy_score"] = frame["strategy_score"].where(
        ~np.isfinite(enhanced_direction),
        enhanced_direction,
    )
    frame = frame.reindex(symbols)
    return frame, frame["news_available"].gt(0).to_numpy(dtype=bool)


def recommend_portfolio(
    *,
    history: pd.DataFrame,
    signals: Iterable[AssetSignal],
    news_events: Iterable[object],
    risk_estimates: Iterable[object],
    current_weights_pct: Mapping[str, float],
    health_score: Optional[float],
) -> GatedNewsDecision:
    """Run a promoted artifact; experimental candidates are never auto-served."""
    metadata, model = _load_artifact()
    if metadata is None or model is None:
        raise RuntimeError("gated news model is unavailable")
    if metadata.get("promotion_status") != "promoted":
        raise RuntimeError("gated news model has not passed promotion gates")

    risk_map = {
        str(estimate.symbol): estimate
        for estimate in risk_estimates
        if int(getattr(estimate, "horizon", 0)) == HORIZON
        and bool(getattr(estimate, "has_history", False))
        and np.isfinite(float(getattr(estimate, "sigma_daily", np.nan)))
    }
    held = sorted(
        symbol
        for symbol, weight in current_weights_pct.items()
        if symbol != "SPY" and float(weight) > 0 and symbol in risk_map
    )
    if len(held) < 5:
        raise RuntimeError("fewer than five held assets have formal risk")
    frame, news_available = _runtime_features(
        held,
        history,
        signals,
        news_events,
    )
    direction, gate = gated_direction_signal(
        model,
        frame,
        metadata["feature_order"],
        metadata["feature_bounds"],
        residual_cap=float(metadata["residual_cap"]),
        q_margin=float(metadata["q_margin"]),
        news_available=news_available,
    )
    signal_scale = float(metadata["return_signal_scale"])
    expected = direction * signal_scale
    sigma = np.asarray(
        [float(risk_map[symbol].sigma_daily) for symbol in held],
        dtype=float,
    )
    previous = np.asarray(
        [float(current_weights_pct.get(symbol, 0.0)) / 100.0 for symbol in held],
        dtype=float,
    )
    previous_cash = max(0.0, 1.0 - float(previous.sum()))
    returns = history.reindex(columns=held).pct_change()
    allocation = risk_controlled_allocation(
        held,
        expected,
        sigma,
        returns,
        previous,
        previous_cash,
        health_score=50.0 if health_score is None else float(health_score),
        base_risk_aversion=float(metadata["base_risk_aversion"]),
        base_target_annual_volatility=float(
            metadata["base_target_annual_volatility"]
        ),
        turnover_penalty=float(metadata["turnover_penalty"]),
    )
    changes = (allocation.weights - previous) * 100.0
    trades = []
    for index, symbol in enumerate(held):
        if abs(changes[index]) < 1.0:
            continue
        risk_level = float(getattr(risk_map[symbol], "risk_level", np.nan))
        news_note = (
            "News residual passed the confidence gate"
            if news_available[index]
            else "No recent news; Daily Strategy fallback was used"
        )
        trades.append(
            ProposedTrade(
                symbol=symbol,
                weight_change_pct=round(float(changes[index]), 2),
                reason=(
                    f"{news_note}; HAR-X risk percentile {risk_level:.0f}/100 "
                    "then determined the risk-controlled target weight."
                ),
            )
        )
    recommendation = Recommendation(
        kind="daily",
        trades=trades,
        confidence=0.70 if allocation.success else 0.0,
        explanation=(
            "Daily Strategy supplies the prior direction, gated news may make "
            "a small residual correction, and HAR-X risk plus Portfolio Health "
            "control position size and cash exposure."
        ),
    )
    return GatedNewsDecision(
        recommendation=recommendation,
        metadata={
            "production_mode": "gated_news_external_risk",
            "model_version": metadata.get("model_version", MODEL_VERSION),
            "promotion_status": metadata["promotion_status"],
            "optimizer_success": allocation.success,
            "risk_is_external_only": True,
            "health_is_external_only": True,
            "gate": gate,
            "cash_before_pct": previous_cash * 100.0,
            "cash_after_pct": allocation.cash_weight * 100.0,
            "target_gross_pct": allocation.target_gross * 100.0,
            "predicted_annual_volatility": (
                allocation.predicted_annual_volatility
            ),
            "target_annual_volatility": allocation.target_annual_volatility,
            "symbols": {
                symbol: {
                    "direction_signal": float(direction[index]),
                    "expected_return_5d_proxy": float(expected[index]),
                    "risk_sigma_daily_5d": float(sigma[index]),
                    "risk_level_5d": float(
                        getattr(risk_map[symbol], "risk_level", np.nan)
                    ),
                    "news_available": bool(news_available[index]),
                    "weight_before_pct": float(previous[index] * 100.0),
                    "weight_after_pct": float(
                        allocation.weights[index] * 100.0
                    ),
                }
                for index, symbol in enumerate(held)
            },
        },
    )


def recommend_strategy_risk_control(
    *,
    history: pd.DataFrame,
    signals: Iterable[AssetSignal],
    risk_estimates: Iterable[object],
    current_weights_pct: Mapping[str, float],
    health_score: Optional[float],
) -> GatedNewsDecision:
    """Production-safe prior: Strategy direction, HAR-X + News sizing only."""
    risk_map = {
        str(estimate.symbol): estimate
        for estimate in risk_estimates
        if int(getattr(estimate, "horizon", 0)) == HORIZON
        and bool(getattr(estimate, "has_history", False))
        and np.isfinite(float(getattr(estimate, "sigma_daily", np.nan)))
    }
    held = sorted(
        symbol
        for symbol, weight in current_weights_pct.items()
        if symbol != "SPY" and float(weight) > 0 and symbol in risk_map
    )
    if len(held) < 5:
        raise RuntimeError("fewer than five held assets have formal risk")
    frame, _ = _runtime_features(held, history, signals, [])
    direction = frame["strategy_score"].to_numpy(dtype=float)
    sigma = np.asarray(
        [float(risk_map[symbol].sigma_daily) for symbol in held],
        dtype=float,
    )
    # Size the tilt by the strategy's MEASURED information coefficient rather
    # than a fixed expected return — see STRATEGY_INFORMATION_COEFFICIENT.
    expected = strategy_alpha(direction, sigma)
    previous = np.asarray(
        [float(current_weights_pct.get(symbol, 0.0)) / 100.0 for symbol in held],
        dtype=float,
    )
    previous_cash = max(0.0, 1.0 - float(previous.sum()))
    allocation = risk_controlled_allocation(
        held,
        expected,
        sigma,
        history.reindex(columns=held).pct_change(),
        previous,
        previous_cash,
        health_score=50.0 if health_score is None else float(health_score),
        base_risk_aversion=6.0,
        base_target_annual_volatility=0.15,
        turnover_penalty=0.0025,
    )
    changes = (allocation.weights - previous) * 100.0
    trades = []
    for index, symbol in enumerate(held):
        if abs(changes[index]) < 1.0:
            continue
        estimate = risk_map[symbol]
        risk_level = float(getattr(estimate, "risk_level", np.nan))
        news_note = (
            "HAR-X risk includes the current news channel"
            if bool(getattr(estimate, "news_applied", False))
            else "HAR-X used its supported no-news fallback"
        )
        trades.append(
            ProposedTrade(
                symbol=symbol,
                weight_change_pct=round(float(changes[index]), 2),
                reason=(
                    f"Daily Strategy supplies direction; {news_note}. "
                    f"Risk percentile {risk_level:.0f}/100 sets the external "
                    "risk-controlled target weight."
                ),
            )
        )
    recommendation = Recommendation(
        kind="daily",
        trades=trades,
        confidence=0.70 if allocation.success else 0.0,
        explanation=(
            "Daily Strategy determines relative direction. Formal HAR-X + "
            "News risk and Portfolio Health then determine stock weights, "
            "total risky exposure, and cash. The experimental direct-news "
            "residual is disabled until it passes promotion gates."
        ),
    )
    news_applied = np.asarray(
        [
            bool(getattr(risk_map[symbol], "news_applied", False))
            for symbol in held
        ],
        dtype=bool,
    )
    return GatedNewsDecision(
        recommendation=recommendation,
        metadata={
            "production_mode": "strategy_external_harx_news_risk",
            "model_version": BASELINE_MODEL_VERSION,
            "direct_news_residual_applied": False,
            "direct_news_residual_status": "experimental_only",
            "news_via_risk_share": float(news_applied.mean()),
            "risk_is_external_only": True,
            "health_is_external_only": True,
            "alpha_scaling": "grinold_kahn_ic_times_horizon_sigma",
            "strategy_information_coefficient": (
                STRATEGY_INFORMATION_COEFFICIENT
            ),
            "optimizer_success": allocation.success,
            "cash_before_pct": previous_cash * 100.0,
            "cash_after_pct": allocation.cash_weight * 100.0,
            "target_gross_pct": allocation.target_gross * 100.0,
            "predicted_annual_volatility": (
                allocation.predicted_annual_volatility
            ),
            "target_annual_volatility": allocation.target_annual_volatility,
            "symbols": {
                symbol: {
                    "strategy_direction": float(direction[index]),
                    "expected_return_5d_proxy": float(expected[index]),
                    "risk_sigma_daily_5d": float(sigma[index]),
                    "risk_level_5d": float(
                        getattr(risk_map[symbol], "risk_level", np.nan)
                    ),
                    "risk_news_applied": bool(news_applied[index]),
                    # Surfaced so a broken RSS pipeline is visible downstream.
                    # The risk engine treats missing/stale stores as an
                    # observed no-news state, which understates risk exactly
                    # when the news feed has failed.
                    "risk_news_quality": str(
                        getattr(risk_map[symbol], "news_quality", "unknown")
                    ),
                    "weight_before_pct": float(previous[index] * 100.0),
                    "weight_after_pct": float(
                        allocation.weights[index] * 100.0
                    ),
                }
                for index, symbol in enumerate(held)
            },
        },
    )
