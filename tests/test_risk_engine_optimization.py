from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from risk_engine_optimization import (  # noqa: E402
    CAUSAL_NEWS_STATES,
    DEPLOYABLE_NEWS_BASE,
    add_causal_news_states,
    engineer_symbol_frame,
    fit_xgb_gamma_fixed,
    parameter_configs,
    time_fold,
)
from optimize_risk_engine import (  # noqa: E402
    DIRECT_VARIANCE_SCALE,
    _xgb_target,
)


def synthetic_ohlc(rows: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2013-01-01", periods=rows)
    returns = 0.001 + 0.01 * np.sin(np.arange(rows) / 7)
    close = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj close": close,
            "volume": 1_000_000 + np.arange(rows),
        }
    )


def test_forward_realized_volatility_uses_only_t_plus_1_through_t_plus_h():
    raw = synthetic_ohlc()
    engineered = engineer_symbol_frame(raw, "TEST")
    horizon = 5
    position = 100
    returns = raw["adj close"].pct_change()
    expected = returns.iloc[position + 1 : position + horizon + 1].std()
    actual = engineered.loc[position, f"realized_vol_{horizon}d"]
    assert np.isclose(actual, expected)


def test_time_fold_embargo_has_exact_trading_session_count():
    frame = engineer_symbol_frame(synthetic_ohlc(800), "TEST")
    frame = frame.dropna(subset=["realized_vol_20d"]).reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    test_year = int(frame["date"].dt.year.max())
    fold = time_fold(frame, test_year, 20)
    dates = np.sort(frame["date"].unique())
    between = dates[
        (dates > np.datetime64(fold.train_cutoff))
        & (dates < np.datetime64(fold.test_start))
    ]
    assert len(between) == 20
    assert not (fold.train_mask & fold.test_mask).any()


def test_causal_news_states_are_exhaustive_and_do_not_use_future_news():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=70),
            "symbol": "TEST",
            "has_news": [0, 0, 1] + [0] * 67,
        }
    )
    result = add_causal_news_states(frame)
    state_total = result[CAUSAL_NEWS_STATES].sum(axis=1) + result["has_news"]
    assert state_total.eq(1).all()
    assert result.loc[0, "news_state_uncovered"] == 1
    assert result.loc[3, "news_state_silent"] == 1
    assert result.loc[65, "news_state_stale"] == 1


def test_deployable_rss_contract_is_supported_by_fnspid_features():
    spec = json.loads(
        (
            ROOT
            / "data"
            / "processed"
            / "extended_news_feature_spec.json"
        ).read_text(encoding="utf-8")
    )
    fnspid_features = {
        feature
        for family in spec["families"].values()
        for feature in family
    }
    assert set(DEPLOYABLE_NEWS_BASE).issubset(fnspid_features)


def test_xgboost_parameter_design_is_deterministic_and_bounded():
    first = parameter_configs(10)
    second = parameter_configs(10)
    assert first == second
    assert all(2 <= row["max_depth"] <= 6 for row in first)
    assert all(0.01 <= row["learning_rate"] <= 0.15 for row in first)
    assert all(0.6 <= row["subsample"] <= 1.0 for row in first)


def test_direct_gamma_target_is_numerically_scaled_and_reversible():
    frame = pd.DataFrame({"target_variance_5d": [1e-6, 1e-4, 1e-2]})
    scaled = _xgb_target(frame, 5, "direct")
    assert np.allclose(
        scaled / DIRECT_VARIANCE_SCALE,
        frame["target_variance_5d"].to_numpy(),
    )
    assert np.median(scaled) == 1.0


def test_xgboost_fixed_seed_and_serialization_roundtrip(tmp_path):
    xgboost = pytest.importorskip("xgboost")
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 20 + ["B"] * 20,
            "f1": np.linspace(-1, 1, 40),
            "f2": np.sin(np.arange(40)),
        }
    )
    target = np.exp(0.2 * frame["f1"].to_numpy()) + 0.1
    parameters = parameter_configs(1)[0]
    first = fit_xgb_gamma_fixed(
        frame, ["f1", "f2"], target, parameters, "cpu", 8
    )
    second = fit_xgb_gamma_fixed(
        frame, ["f1", "f2"], target, parameters, "cpu", 8
    )
    expected = first.predict(frame[["f1", "f2"]])
    assert np.array_equal(
        expected, second.predict(frame[["f1", "f2"]])
    )

    path = tmp_path / "booster.json"
    first.save_model(path)
    restored = xgboost.XGBRegressor()
    restored.load_model(path)
    assert np.array_equal(
        expected, restored.predict(frame[["f1", "f2"]])
    )
