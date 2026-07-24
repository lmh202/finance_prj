from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.risk_engine import engine  # noqa: E402


def synthetic_backend_ohlc(rows: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-01", periods=rows)
    returns = 0.0005 + 0.012 * np.sin(np.arange(rows) / 9)
    close = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
        },
        index=dates,
    )


def test_official_primary_output_is_changed_by_news_attention():
    ohlc = synthetic_backend_ohlc()
    baseline = engine.risk_estimate("TEST", ohlc, 5)
    supplied = engine.risk_estimate(
        "TEST", ohlc, 5, news_features={"log_count": 3.0}
    )
    explicit_no_news = engine.risk_estimate(
        "TEST",
        ohlc,
        5,
        news_features={"log_count": 0.0, "__quality__": "no_news"},
    )
    assert supplied.sigma_daily > baseline.sigma_daily
    assert supplied.sigma_h > baseline.sigma_h
    assert supplied.news_applied is True
    assert supplied.news_quality == "ok"
    assert explicit_no_news.sigma_daily == baseline.sigma_daily
    assert explicit_no_news.news_applied is True


def test_default_batch_output_is_the_news_integrated_five_day_risk():
    estimates = engine.risk_estimates(
        {"TEST": synthetic_backend_ohlc()}
    )
    assert [estimate.horizon for estimate in estimates] == [5]
    assert estimates[0].news_applied is True
    assert estimates[0].model_version == "risk-har-news-5d-v1"


def test_configured_overlay_applies_only_to_complete_fresh_features():
    original = engine._MODEL
    try:
        engine._MODEL = copy.deepcopy(original)
        engine._MODEL["news_overlays"] = {
            "5": {
                "model_type": "linear_gamma_variance_ratio",
                "features": ["log_count"],
                "scale": {"log_count": 1.0},
                "coef": {"log_count": math.log(1.44)},
                "intercept": 0.0,
                "max_sigma_multiplier": 2.0,
            }
        }
        ohlc = synthetic_backend_ohlc()
        baseline = engine.risk_estimate("TEST", ohlc, 5)
        applied = engine.risk_estimate(
            "TEST",
            ohlc,
            5,
            news_features={"log_count": 1.0, "__quality__": "fresh"},
        )
        stale = engine.risk_estimate(
            "TEST",
            ohlc,
            5,
            news_features={"log_count": 1.0, "__quality__": "stale"},
        )
        incomplete = engine.risk_estimate(
            "TEST", ohlc, 5, news_features={"__quality__": "fresh"}
        )
        assert np.isclose(applied.sigma_daily, baseline.sigma_daily * 1.2)
        assert applied.news_applied is True
        assert applied.news_quality == "fresh"
        assert stale.sigma_daily == baseline.sigma_daily
        assert stale.news_quality == "stale"
        assert incomplete.sigma_daily == baseline.sigma_daily
        assert incomplete.news_quality == "invalid"
    finally:
        engine._MODEL = original


def test_portfolio_risk_uses_news_adjusted_marginal_volatility(monkeypatch):
    first = synthetic_backend_ohlc()
    second = synthetic_backend_ohlc().copy()
    second["close"] *= 1 + 0.002 * np.cos(np.arange(len(second)) / 5)
    second["open"] = second["close"] * 0.999
    second["high"] = second["close"] * 1.01
    second["low"] = second["close"] * 0.99
    frames = {"A": first, "B": second}
    weights = {"A": 0.5, "B": 0.5}

    original = engine._MODEL
    try:
        price_only = copy.deepcopy(original)
        price_only.pop("news_overlays", None)
        engine._MODEL = price_only
        baseline = engine.portfolio_risk(frames, weights, 5)

        engine._MODEL = original
        monkeypatch.setattr(
            engine,
            "_news_features_from_store",
            lambda symbol, as_of: {
                "log_count": 3.0,
                "__quality__": "fresh",
            },
        )
        integrated = engine.portfolio_risk(frames, weights, 5)
        assert baseline is not None and integrated is not None
        assert integrated.sigma_h > baseline.sigma_h
    finally:
        engine._MODEL = original
