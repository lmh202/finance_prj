"""Numeric final-decision layer: return ranking + HAR-X + News risk + constraints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import DATA_DIR
from src.interfaces import CONSTRAINTS, ProposedTrade, Recommendation

CHECKPOINT_DIR = DATA_DIR / "processed" / "decision_model"
METADATA_PATH = CHECKPOINT_DIR / "decision_model.json"
MODEL_PATH = CHECKPOINT_DIR / "return_model.joblib"
BENCHMARK = "SPY"
HORIZON = 20
EPS = 1e-10
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

_ARTIFACT_CACHE: tuple[int, dict, object] | None = None


@dataclass
class NumericDecision:
    recommendation: Recommendation
    metadata: dict


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


def model_available() -> bool:
    metadata, model = _load_artifact()
    return metadata is not None and model is not None


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + relative_strength)


def latest_feature_frame(
    history: pd.DataFrame,
    benchmark: str = BENCHMARK,
) -> pd.DataFrame:
    """Build exactly the close-derived feature order used offline."""
    history = history.sort_index()
    if benchmark not in history:
        return pd.DataFrame(columns=PRICE_FEATURES)
    spy = history[benchmark].dropna()
    spy_return = spy.pct_change()
    spy_momentum = spy.pct_change(20)
    rows = []
    for symbol in history.columns:
        if symbol == benchmark:
            continue
        close = history[symbol].dropna()
        if len(close) < 220:
            continue
        return_1d = close.pct_change()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        vol20 = return_1d.rolling(20).std() * math.sqrt(252)
        momentum20 = close.pct_change(20)
        aligned_spy_return = spy_return.reindex(close.index)
        covariance = return_1d.rolling(60).cov(aligned_spy_return)
        benchmark_variance = aligned_spy_return.rolling(60).var()
        row = {
            "symbol": symbol,
            "ret_1d": return_1d.iloc[-1],
            "mom_20d": momentum20.iloc[-1],
            "mom_60d": close.pct_change(60).iloc[-1],
            "price_vs_sma50": close.iloc[-1] / sma50.iloc[-1] - 1,
            "sma50_vs_sma200": sma50.iloc[-1] / sma200.iloc[-1] - 1,
            "vol_20d": vol20.iloc[-1],
            "rsi_14": _rsi(close).iloc[-1],
            "drawdown": close.iloc[-1] / close.cummax().iloc[-1] - 1,
            "risk_adj_mom": momentum20.iloc[-1] / vol20.iloc[-1],
            "beta_60d": covariance.iloc[-1] / benchmark_variance.iloc[-1],
            "rel_str_20d": (
                momentum20.iloc[-1]
                - spy_momentum.reindex(close.index).iloc[-1]
            ),
        }
        if all(np.isfinite(row[name]) for name in PRICE_FEATURES):
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=PRICE_FEATURES)
    return pd.DataFrame(rows).set_index("symbol")[PRICE_FEATURES]


def _risk_map(risk_estimates: Iterable[object]) -> dict[str, object]:
    result = {}
    for estimate in risk_estimates:
        if int(getattr(estimate, "horizon", 0)) != 5:
            continue
        if not bool(getattr(estimate, "has_history", False)):
            continue
        sigma = float(getattr(estimate, "sigma_daily", np.nan))
        if np.isfinite(sigma) and sigma > 0:
            result[str(getattr(estimate, "symbol"))] = estimate
    return result


def _repair_correlation(values: pd.DataFrame) -> np.ndarray:
    correlation = values.tail(126).corr(min_periods=40).to_numpy(dtype=float)
    correlation = np.where(np.isfinite(correlation), correlation, 0.0)
    np.fill_diagonal(correlation, 1.0)
    correlation = 0.90 * correlation + 0.10 * np.eye(len(correlation))
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1e-5)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(repaired))
    return repaired / np.outer(scale, scale)


def _target_weights(
    symbols: Sequence[str],
    expected_return: np.ndarray,
    sigma_daily: np.ndarray,
    history: pd.DataFrame,
    previous_weight: np.ndarray,
    risk_aversion: float,
    cost_rate: float,
    min_trade: float,
) -> tuple[np.ndarray, bool]:
    previous = np.maximum(np.asarray(previous_weight, dtype=float), 0.0)
    previous = previous / max(previous.sum(), EPS)
    returns = history[list(symbols)].pct_change()
    correlation = _repair_correlation(returns)
    covariance = (
        np.diag(sigma_daily)
        @ correlation
        @ np.diag(sigma_daily)
        * HORIZON
    )
    horizon_sigma = sigma_daily * math.sqrt(HORIZON)
    expected = np.clip(
        np.nan_to_num(expected_return, nan=0.0),
        -horizon_sigma,
        horizon_sigma,
    )
    max_position = CONSTRAINTS["max_stock_weight_pct"] / 100
    max_change = CONSTRAINTS["max_weight_change_pct"] / 100
    lower = np.maximum(0.0, previous - max_change)
    upper = np.minimum(max_position, previous + max_change)
    upper = np.where(previous > max_position, previous, upper)
    if lower.sum() > 1 + 1e-8 or upper.sum() < 1 - 1e-8:
        return previous, False

    def objective(weights: np.ndarray) -> float:
        delta = weights - previous
        return float(
            -(expected @ weights)
            + 0.5 * risk_aversion * (weights @ covariance @ weights)
            + cost_rate * np.sum(np.sqrt(delta * delta + 1e-10))
        )

    result = minimize(
        objective,
        x0=previous,
        method="SLSQP",
        bounds=list(zip(lower, upper)),
        constraints=[
            {"type": "eq", "fun": lambda weights: weights.sum() - 1.0}
        ],
        options={"maxiter": 300, "ftol": 1e-10},
    )
    if not result.success:
        return previous, False
    weights = result.x

    # A displayed recommendation must be directly executable: freeze changes
    # below the minimum trade threshold and re-solve the remaining allocation.
    fixed = np.zeros(len(previous), dtype=bool)
    for _ in range(4):
        small = (np.abs(weights - previous) < min_trade - 1e-7) & (
            np.abs(weights - previous) > 1e-5
        )
        new_fixed = small & ~fixed
        if not new_fixed.any():
            break
        fixed |= new_fixed
        fixed_lower = lower.copy()
        fixed_upper = upper.copy()
        fixed_lower[fixed] = previous[fixed]
        fixed_upper[fixed] = previous[fixed]
        if (
            fixed_lower.sum() > 1 + 1e-8
            or fixed_upper.sum() < 1 - 1e-8
        ):
            return previous, False
        result = minimize(
            objective,
            x0=weights,
            method="SLSQP",
            bounds=list(zip(fixed_lower, fixed_upper)),
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda candidate: candidate.sum() - 1.0,
                }
            ],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if not result.success:
            return previous, False
        weights = result.x

    weights = np.maximum(weights, 0.0)
    return weights / max(weights.sum(), EPS), True


def recommend_portfolio(
    history: pd.DataFrame,
    risk_estimates: Iterable[object],
    current_weights_pct: Mapping[str, float],
) -> NumericDecision:
    metadata, model = _load_artifact()
    if metadata is None or model is None:
        raise RuntimeError("decision model is unavailable")

    features = latest_feature_frame(history)
    risks = _risk_map(risk_estimates)
    held_symbols = {
        str(symbol)
        for symbol, weight in current_weights_pct.items()
        if symbol != BENCHMARK and float(weight) > 0
    }
    available = set(features.index) & set(risks)
    missing = held_symbols - available
    if missing:
        raise RuntimeError(
            "missing features or formal risk for held symbols: "
            + ", ".join(sorted(missing))
        )
    symbols = sorted(held_symbols)
    if len(symbols) < 5:
        raise RuntimeError("fewer than five symbols have features and risk")

    feature_values = features.loc[symbols, metadata["feature_order"]]
    raw_model_prediction = np.asarray(
        model.predict(feature_values),
        dtype=float,
    )
    raw_model_prediction = np.clip(raw_model_prediction, -0.25, 0.25)
    signal_scale = float(metadata.get("return_signal_scale", 1.0))
    model_prediction = raw_model_prediction * signal_scale
    production_mode = str(metadata.get("production_mode", "risk_only"))
    expected = (
        model_prediction
        if production_mode == "ml_return_plus_risk"
        else np.zeros_like(model_prediction)
    )
    sigma = np.array(
        [float(getattr(risks[symbol], "sigma_daily")) for symbol in symbols]
    )
    previous = np.array(
        [float(current_weights_pct.get(symbol, 0.0)) / 100 for symbol in symbols]
    )
    if previous.sum() <= 0:
        raise RuntimeError("no current portfolio weights are available")
    target, success = _target_weights(
        symbols,
        expected,
        sigma,
        history,
        previous,
        float(metadata.get("risk_aversion", 6.0)),
        float(metadata.get("primary_transaction_cost_bps", 25.0)) / 10_000,
        float(CONSTRAINTS["min_trade_pct"]) / 100,
    )
    previous = previous / previous.sum()
    change_pct = (target - previous) * 100
    min_trade = float(CONSTRAINTS["min_trade_pct"])
    trades = []
    for index, symbol in enumerate(symbols):
        change = float(change_pct[index])
        if abs(change) < min_trade:
            continue
        risk_level = float(getattr(risks[symbol], "risk_level", np.nan))
        if change > 0:
            reason = (
                f"Target allocation increases after risk and cost controls; "
                f"near-term risk percentile is {risk_level:.0f}/100."
            )
        else:
            reason = (
                f"Target allocation decreases after risk and cost controls; "
                f"near-term risk percentile is {risk_level:.0f}/100."
            )
        trades.append(ProposedTrade(symbol, round(change, 2), reason))

    if trades:
        explanation = (
            f"The {production_mode.replace('_', ' ')} decision layer found "
            "portfolio changes that clear the risk, turnover, and position limits."
        )
    else:
        explanation = (
            "No position change clears the risk, turnover, and position limits today."
        )
    confidence = 0.70 if success else 0.0
    recommendation = Recommendation(
        kind="daily",
        trades=trades,
        confidence=confidence,
        explanation=explanation,
    )
    detail = {
        "model_version": metadata.get("model_version"),
        "selected_model": metadata.get("selected_model"),
        "return_signal_scale": signal_scale,
        "production_mode": production_mode,
        "optimizer_success": success,
        "risk_model_versions": sorted(
            {
                str(getattr(risks[symbol], "model_version", ""))
                for symbol in symbols
            }
        ),
        "news_applied_share": float(
            np.mean(
                [
                    bool(getattr(risks[symbol], "news_applied", False))
                    for symbol in symbols
                ]
            )
        ),
        "symbols": {
            symbol: {
                "predicted_excess_return_20d": float(
                    model_prediction[index]
                ),
                "raw_predicted_excess_return_20d": float(
                    raw_model_prediction[index]
                ),
                "risk_sigma_daily_5d": float(sigma[index]),
                "risk_level_5d": float(
                    getattr(risks[symbol], "risk_level", np.nan)
                ),
                "weight_before_pct": float(previous[index] * 100),
                "weight_after_pct": float(target[index] * 100),
            }
            for index, symbol in enumerate(symbols)
        },
    }
    return NumericDecision(recommendation, detail)
