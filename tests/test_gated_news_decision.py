from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.recommendation.gated_news import (  # noqa: E402
    CORE_FEATURES,
    FEATURES,
    NEWS_FEATURES,
    gated_direction_signal,
    recommend_strategy_risk_control,
    risk_controlled_allocation,
)
from src.interfaces import AssetSignal  # noqa: E402


class PreferIncreaseModel:
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        # The action column follows the state columns.
        return matrix[:, len(FEATURES)]


def _state(rows: int = 3) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=range(rows), columns=FEATURES)
    frame["strategy_score"] = np.linspace(-0.4, 0.4, rows)
    frame["news_score_5d"] = 0.5
    return frame


def _bounds() -> dict[str, list[float]]:
    return {column: [-2.0, 2.0] for column in FEATURES}


def test_learned_state_excludes_risk_and_health() -> None:
    assert not any("risk" in column for column in FEATURES)
    assert not any("health" in column for column in FEATURES)
    assert set(FEATURES) == set(CORE_FEATURES + NEWS_FEATURES)


def test_no_news_is_exact_strategy_fallback() -> None:
    frame = _state()
    combined, diagnostics = gated_direction_signal(
        PreferIncreaseModel(),
        frame,
        FEATURES,
        _bounds(),
        residual_cap=0.15,
        q_margin=0.0,
        news_available=np.zeros(len(frame), dtype=bool),
    )
    np.testing.assert_array_equal(
        combined,
        frame["strategy_score"].to_numpy(dtype=float),
    )
    assert diagnostics["news_applied_share"] == 0.0


def test_news_residual_is_bounded_and_gated() -> None:
    frame = _state()
    combined, diagnostics = gated_direction_signal(
        PreferIncreaseModel(),
        frame,
        FEATURES,
        _bounds(),
        residual_cap=0.15,
        q_margin=0.5,
        news_available=np.ones(len(frame), dtype=bool),
    )
    residual = combined - frame["strategy_score"].to_numpy(dtype=float)
    assert np.all(residual <= 0.15 + 1e-12)
    assert np.all(residual >= -0.15 - 1e-12)
    assert diagnostics["news_applied_share"] == 1.0


def _allocation(
    sigma: np.ndarray,
    *,
    health_score: float = 50.0,
):
    rng = np.random.default_rng(20260724)
    symbols = [f"S{index}" for index in range(len(sigma))]
    returns = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(200, len(symbols))),
        columns=symbols,
    )
    previous = np.full(len(symbols), 0.95 / len(symbols))
    return risk_controlled_allocation(
        symbols,
        np.zeros(len(symbols)),
        sigma,
        returns,
        previous,
        0.05,
        health_score=health_score,
        base_risk_aversion=6.0,
        base_target_annual_volatility=0.15,
        turnover_penalty=0.0025,
    )


def test_higher_har_x_risk_reduces_external_gross_exposure() -> None:
    low = _allocation(np.full(10, 0.01))
    high = _allocation(np.full(10, 0.05))
    assert low.success and high.success
    assert high.target_gross < low.target_gross
    assert high.cash_weight > low.cash_weight


def test_lower_health_reduces_external_risk_budget() -> None:
    sigma = np.full(10, 0.035)
    healthy = _allocation(sigma, health_score=90.0)
    weak = _allocation(sigma, health_score=20.0)
    assert healthy.success and weak.success
    assert weak.target_annual_volatility < healthy.target_annual_volatility
    assert weak.target_gross <= healthy.target_gross
    assert weak.effective_risk_aversion > healthy.effective_risk_aversion


def test_allocation_respects_trade_and_position_constraints() -> None:
    result = _allocation(
        np.array([0.06, 0.04, 0.03, 0.025, 0.02, 0.018, 0.016, 0.014, 0.012, 0.01])
    )
    assert result.success
    assert result.maximum_position <= 0.20 + 1e-7
    assert result.maximum_change <= 0.05 + 1e-7
    assert (
        result.minimum_active_trade == 0.0
        or result.minimum_active_trade >= 0.01 - 1e-7
    )
    assert np.isclose(result.weights.sum() + result.cash_weight, 1.0)


def test_production_baseline_uses_news_affected_risk_externally() -> None:
    rng = np.random.default_rng(7)
    symbols = [f"S{index}" for index in range(10)]
    dates = pd.bdate_range("2024-01-01", periods=260)
    prices = pd.DataFrame(index=dates)
    for symbol in symbols + ["SPY"]:
        prices[symbol] = 100.0 * np.cumprod(
            1.0 + rng.normal(0.0002, 0.012, len(dates))
        )
    signals = [
        AssetSignal(
            symbol=symbol,
            score=50.0,
            action="hold",
            indicators={
                "momentum": 0.05,
                "trend": 1.0,
                "sharpe": 0.8,
                "volatility": 0.2,
                "drawdown": -0.1,
            },
            rationale="test",
        )
        for symbol in symbols
    ]
    risks = [
        SimpleNamespace(
            symbol=symbol,
            horizon=5,
            has_history=True,
            sigma_daily=0.03,
            risk_level=70.0,
            news_applied=index % 2 == 0,
        )
        for index, symbol in enumerate(symbols)
    ]
    result = recommend_strategy_risk_control(
        history=prices,
        signals=signals,
        risk_estimates=risks,
        current_weights_pct={symbol: 9.5 for symbol in symbols},
        health_score=50.0,
    )
    assert result.metadata["production_mode"] == (
        "strategy_external_harx_news_risk"
    )
    assert result.metadata["direct_news_residual_applied"] is False
    assert result.metadata["risk_is_external_only"] is True
    assert result.metadata["news_via_risk_share"] == 0.5
