from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.recommendation import market_stress as ms  # noqa: E402


def _series(daily_vols: list[tuple[int, float]], seed: int = 11) -> pd.Series:
    """Build a close series from (sessions, daily_sigma) segments."""
    rng = np.random.default_rng(seed)
    returns = np.concatenate(
        [rng.normal(0.0, sigma, sessions) for sessions, sigma in daily_vols]
    )
    dates = pd.bdate_range("2015-01-01", periods=len(returns))
    return pd.Series(100.0 * np.cumprod(1.0 + returns), index=dates)


def test_calm_and_stressed_flip_across_the_threshold() -> None:
    # A long ordinary stretch, then a deliberately quiet tail (the calm
    # reading), then a volatility explosion (the stressed reading). The tail
    # segments must be longer than VOLATILITY_WINDOW so the trailing realised
    # volatility fully reflects them.
    full = _series([(700, 0.012), (90, 0.004), (90, 0.040)])
    calm = ms.assess(full.iloc[:790])
    stressed = ms.assess(full)

    assert calm.state == ms.CALM
    assert calm.stressed is False
    assert calm.volatility_percentile < ms.STRESS_PERCENTILE

    assert stressed.state == ms.STRESSED
    assert stressed.stressed is True
    assert stressed.volatility_percentile >= ms.STRESS_PERCENTILE
    assert stressed.realised_volatility_annualised > (
        calm.realised_volatility_annualised
    )


def test_short_history_is_unknown_not_calm() -> None:
    """The fail-closed guarantee: too little data must never read as calm,
    because the caller maps calm to the aggressive risk aversion."""
    short = ms.assess(_series([(ms.MIN_OBSERVATIONS - 1, 0.01)]))
    assert short.state == ms.UNKNOWN
    assert short.stressed is False
    assert short.unavailable_reason == "insufficient_history"
    assert short.volatility_percentile is None

    enough = ms.assess(_series([(ms.MIN_OBSERVATIONS + 5, 0.01)]))
    assert enough.state in {ms.CALM, ms.STRESSED}


def test_missing_and_malformed_inputs_are_unknown() -> None:
    for candidate in (
        None,
        pd.Series(dtype=float),
        pd.Series([np.nan] * 600),
        12345,
        "not a series",
    ):
        state = ms.assess(candidate)
        assert state.state == ms.UNKNOWN, candidate
        assert state.stressed is False
        assert state.unavailable_reason is not None


def test_percentile_uses_only_data_up_to_the_evaluation_point() -> None:
    """Causality: appending future bars must not change the reading taken
    at the earlier point."""
    full = _series([(700, 0.005), (100, 0.030)])
    early = ms.assess(full.iloc[:600])
    early_again = ms.assess(full.iloc[:600])  # same slice of a longer series
    assert early.volatility_percentile == early_again.volatility_percentile
    assert early.as_of == early_again.as_of == str(full.index[599].date())


def test_matches_the_backtested_expression() -> None:
    """Pin the definition: rolling rank INCLUDING the current observation."""
    close = _series([(800, 0.01)])
    volatility = close.pct_change().rolling(
        ms.VOLATILITY_WINDOW
    ).std() * np.sqrt(ms.TRADING_DAYS)
    expected = float(
        volatility.rolling(
            ms.PERCENTILE_WINDOW,
            min_periods=ms.PERCENTILE_MIN_PERIODS,
        )
        .rank(pct=True)
        .iloc[-1]
    )
    assert ms.assess(close).volatility_percentile == expected


def test_as_metadata_is_json_native() -> None:
    import json

    for candidate in (_series([(800, 0.01)]), None):
        payload = ms.as_metadata(ms.assess(candidate))
        json.dumps(payload)  # must not raise
        assert payload["stress_threshold"] == ms.STRESS_PERCENTILE
        assert payload["benchmark"] == "SPY"
        for key in ("volatility_percentile", "realised_volatility_annualised"):
            assert payload[key] is None or isinstance(payload[key], float)
        assert isinstance(payload["observations"], int)
        assert isinstance(payload["stressed"], bool)
