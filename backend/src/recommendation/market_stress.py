"""Causal market-stress state for the adaptive risk budget.

The decision layer runs a mean-variance allocation whose risk aversion decides
how far the portfolio sits from minimum variance. Backtesting settled on making
that one number state-dependent rather than fixed: be closer to the directional
strategy when the market is calm, closer to minimum variance when it is not.

The state is a single causal quantity — SPY's 60-session realised volatility,
ranked as a percentile against its own trailing history. Nothing here votes on
direction; it only sizes the risk budget.

This module is pure: a price series in, a dataclass out. No I/O, no network, no
model artifacts. `assess()` never raises — the caller treats an unknown state as
stressed, so a bad input degrades the risk budget conservatively instead of
failing the recommendation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# Pinned to the backtested definition. `rolling(...).rank(pct=True)` ranks the
# latest observation INSIDE its own window — do not substitute a strictly-prior
# reference window. Measured on SPY 2019-2026 the two conventions differ by
# 0.0010 on average and disagree on the stress flag on 1 of 1336 sessions, so
# the choice is immaterial to results but free to get exactly right.
VOLATILITY_WINDOW = 60
PERCENTILE_WINDOW = 504
PERCENTILE_MIN_PERIODS = 252
STRESS_PERCENTILE = 0.75
TRADING_DAYS = 252

# Closes required before a percentile can be emitted at all, and before the
# reference window is full.
MIN_OBSERVATIONS = VOLATILITY_WINDOW + PERCENTILE_MIN_PERIODS      # 312
FULL_WINDOW_OBSERVATIONS = VOLATILITY_WINDOW + PERCENTILE_WINDOW   # 564

CALM = "calm"
STRESSED = "stressed"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class MarketStressState:
    """One causal reading of how stressed the market is."""

    state: str                                    # CALM | STRESSED | UNKNOWN
    stressed: bool
    volatility_percentile: Optional[float]        # 0..1
    realised_volatility_annualised: Optional[float]
    observations: int
    as_of: Optional[str]                          # ISO date of the last close
    unavailable_reason: Optional[str]


def _unknown(observations: int, reason: str) -> MarketStressState:
    return MarketStressState(
        state=UNKNOWN,
        stressed=False,
        volatility_percentile=None,
        realised_volatility_annualised=None,
        observations=observations,
        as_of=None,
        unavailable_reason=reason,
    )


def assess(benchmark_close: Optional[pd.Series]) -> MarketStressState:
    """Classify the market from a benchmark close series.

    `benchmark_close` must be long enough to cover a `VOLATILITY_WINDOW`
    rolling volatility plus `PERCENTILE_MIN_PERIODS` observations of it. The
    two-year frame the other engines share is NOT long enough — see
    `routers/_common.load_benchmark_close`.

    Causality: the ranked observation is itself built only from past returns,
    and callers evaluate the series through the previous session, so no future
    information enters the state.
    """
    if benchmark_close is None:
        return _unknown(0, "no_benchmark_history")
    try:
        close = pd.Series(benchmark_close).dropna()
    except (TypeError, ValueError):
        return _unknown(0, "no_benchmark_history")
    if close.empty:
        return _unknown(0, "no_benchmark_history")
    if len(close) < MIN_OBSERVATIONS:
        return _unknown(len(close), "insufficient_history")

    try:
        returns = close.pct_change()
        volatility = (
            returns.rolling(VOLATILITY_WINDOW).std() * math.sqrt(TRADING_DAYS)
        )
        percentile_series = volatility.rolling(
            PERCENTILE_WINDOW,
            min_periods=PERCENTILE_MIN_PERIODS,
        ).rank(pct=True)

        percentile = float(percentile_series.iloc[-1])
        realised = float(volatility.iloc[-1])
        if not (math.isfinite(percentile) and math.isfinite(realised)):
            return _unknown(len(close), "insufficient_history")

        as_of = str(pd.Timestamp(close.index[-1]).date())
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        # Never propagate: the daily endpoint catches ValueError/KeyError and
        # would drop an otherwise valid recommendation to its fallback path.
        return _unknown(len(close), "no_benchmark_history")

    stressed = percentile >= STRESS_PERCENTILE
    return MarketStressState(
        state=STRESSED if stressed else CALM,
        stressed=stressed,
        volatility_percentile=percentile,
        realised_volatility_annualised=realised,
        observations=int(len(close)),
        as_of=as_of,
        unavailable_reason=None,
    )


def as_metadata(state: MarketStressState, benchmark: str = "SPY") -> dict:
    """JSON-native block for `decision_meta`.

    `decision_meta` bypasses `serialize.as_dict` and goes straight to FastAPI's
    encoder, so every value here must be a plain Python scalar — no numpy types,
    no `pd.Timestamp`.
    """
    return {
        "state": str(state.state),
        "stressed": bool(state.stressed),
        "benchmark": str(benchmark),
        "volatility_percentile": (
            None
            if state.volatility_percentile is None
            else float(state.volatility_percentile)
        ),
        "realised_volatility_annualised": (
            None
            if state.realised_volatility_annualised is None
            else float(state.realised_volatility_annualised)
        ),
        "stress_threshold": float(STRESS_PERCENTILE),
        "volatility_window_sessions": int(VOLATILITY_WINDOW),
        "percentile_window_sessions": int(PERCENTILE_WINDOW),
        "percentile_min_periods": int(PERCENTILE_MIN_PERIODS),
        "observations": int(state.observations),
        "as_of": None if state.as_of is None else str(state.as_of),
        "unavailable_reason": (
            None
            if state.unavailable_reason is None
            else str(state.unavailable_reason)
        ),
    }
