from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.recommendation.gated_news import (  # noqa: E402
    CALM_RISK_AVERSION,
    CORE_FEATURES,
    FEATURES,
    NEWS_FEATURES,
    STRESSED_RISK_AVERSION,
    gated_direction_signal,
    recommend_strategy_risk_control,
    risk_controlled_allocation,
)
from src.recommendation import market_stress  # noqa: E402
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


# ------------------------------------------------- adaptive risk aversion


def _adaptive_inputs(*, starting_weight_pct: float = 6.0):
    """Ten holdings with a deliberately low starting gross, so the
    max_change=0.05 band does not clamp the second solve and mask the
    risk-aversion effect."""
    rng = np.random.default_rng(41)
    symbols = [f"S{index}" for index in range(10)]
    dates = pd.bdate_range("2024-01-01", periods=320)
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
            sigma_daily=0.02 + 0.002 * index,
            risk_level=50.0,
            news_applied=True,
            news_quality="fresh",
        )
        for index, symbol in enumerate(symbols)
    ]
    weights = {symbol: starting_weight_pct for symbol in symbols}
    return prices, signals, risks, weights


def _benchmark(segments: list[tuple[int, float]], seed: int = 5) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = np.concatenate(
        [rng.normal(0.0, sigma, sessions) for sessions, sigma in segments]
    )
    dates = pd.bdate_range("2014-01-01", periods=len(returns))
    return pd.Series(100.0 * np.cumprod(1.0 + returns), index=dates)


def _decide(benchmark_close, *, health_score: float = 60.0):
    prices, signals, risks, weights = _adaptive_inputs()
    return recommend_strategy_risk_control(
        history=prices,
        signals=signals,
        risk_estimates=risks,
        current_weights_pct=weights,
        health_score=health_score,
        benchmark_close=benchmark_close,
    )


def test_missing_benchmark_falls_back_to_the_stressed_risk_aversion() -> None:
    """Fail closed: no benchmark must never widen the risk budget. This also
    keeps the pre-adaptive behaviour byte-identical for callers that omit it."""
    result = _decide(None)
    assert result.metadata["base_risk_aversion"] == STRESSED_RISK_AVERSION
    assert result.metadata["market_stress"]["state"] == market_stress.UNKNOWN
    assert result.metadata["market_stress"]["unavailable_reason"] is not None
    assert result.metadata["risk_aversion_policy"] == "adaptive_market_stress_v1"


def test_calm_market_lowers_the_base_risk_aversion() -> None:
    calm = _decide(_benchmark([(700, 0.012), (90, 0.004)]))
    assert calm.metadata["market_stress"]["state"] == market_stress.CALM
    assert calm.metadata["base_risk_aversion"] == CALM_RISK_AVERSION


def test_stressed_market_raises_the_base_risk_aversion() -> None:
    stressed = _decide(_benchmark([(700, 0.012), (90, 0.004), (90, 0.040)]))
    assert stressed.metadata["market_stress"]["state"] == market_stress.STRESSED
    assert stressed.metadata["base_risk_aversion"] == STRESSED_RISK_AVERSION


def test_calm_market_builds_a_less_minimum_variance_portfolio() -> None:
    """What the state actually changes is COMPOSITION, not gross exposure.

    A lower risk aversion moves the relative portfolio away from minimum
    variance, so its predicted volatility rises — and the volatility target
    then compensates by holding MORE cash, not less. Asserting on gross would
    get the sign backwards; the distinguishing quantity is the predicted
    volatility of the relative portfolio.
    """
    calm = _decide(_benchmark([(700, 0.012), (90, 0.004)]))
    stressed = _decide(_benchmark([(700, 0.012), (90, 0.004), (90, 0.040)]))
    assert calm.metadata["optimizer_success"]
    assert stressed.metadata["optimizer_success"]
    assert (
        calm.metadata["predicted_annual_volatility"]
        > stressed.metadata["predicted_annual_volatility"]
    )
    # The risk budget itself is unchanged by the market state — only health
    # scales the target — so both must aim at the same volatility.
    assert calm.metadata["target_annual_volatility"] == pytest.approx(
        stressed.metadata["target_annual_volatility"]
    )


def test_effective_risk_aversion_applies_the_health_multiplier() -> None:
    health = 40.0
    result = _decide(_benchmark([(700, 0.012), (90, 0.004)]), health_score=health)
    base = result.metadata["base_risk_aversion"]
    expected = base * (1.0 + 0.75 * (1.0 - health / 100.0))
    assert result.metadata["effective_risk_aversion"] == pytest.approx(expected)


def test_market_stress_metadata_is_json_serialisable() -> None:
    import json

    json.dumps(_decide(_benchmark([(700, 0.012), (90, 0.004)])).metadata)


# ------------------------------------------- unmanaged holdings are not cash


def test_unmanaged_holdings_are_locked_not_spent_as_cash() -> None:
    """Regression: the benchmark (and anything without a risk estimate) used to
    fall out of the managed sleeve and be counted as cash, so the optimiser
    funded increases from a sale it never proposed."""
    prices, signals, risks, weights = _adaptive_inputs()
    weights = dict(weights)
    weights["SPY"] = 20.0  # held, but excluded from the decision universe
    result = recommend_strategy_risk_control(
        history=prices,
        signals=signals,
        risk_estimates=risks,
        current_weights_pct=weights,
        health_score=60.0,
        benchmark_close=_benchmark([(700, 0.012), (90, 0.004)]),
    )
    meta = result.metadata

    assert meta["locked_positions"] == {"SPY": pytest.approx(20.0)}
    assert meta["locked_weight_pct"] == pytest.approx(20.0)
    # The managed sleeve may never exceed what is actually available.
    assert meta["target_gross_pct"] <= 100.0 - meta["locked_weight_pct"] + 1e-6
    # Cash reported must be TRUE cash, not cash + the locked position.
    managed_before = sum(
        block["weight_before_pct"] for block in meta["symbols"].values()
    )
    assert meta["cash_before_pct"] == pytest.approx(
        100.0 - managed_before - meta["locked_weight_pct"], abs=1e-6
    )
    # Everything adds up.
    managed_after = sum(
        block["weight_after_pct"] for block in meta["symbols"].values()
    )
    assert managed_after + meta["locked_weight_pct"] + meta[
        "cash_after_pct"
    ] == pytest.approx(100.0, abs=1e-4)


def test_no_unmanaged_holdings_leaves_the_gross_budget_untouched() -> None:
    """With nothing locked the cap is 1.0, i.e. the previous behaviour."""
    result = _decide(_benchmark([(700, 0.012), (90, 0.004)]))
    assert result.metadata["locked_weight_pct"] == pytest.approx(0.0)
    assert result.metadata["locked_positions"] == {}
