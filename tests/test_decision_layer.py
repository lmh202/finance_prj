from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

from decision_layer_core import (  # noqa: E402
    PRICE_FEATURES,
    optimize_weights,
    time_fold,
)
from build_decision_dataset import engineer_close_features  # noqa: E402
from src.recommendation import decision, llm_client  # noqa: E402


def synthetic_history(rows: int = 280, symbols: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=rows)
    values = {}
    market_return = rng.normal(0.0003, 0.009, rows)
    values["SPY"] = 100 * np.cumprod(1 + market_return)
    for index in range(symbols):
        returns = (
            0.0002
            + 0.7 * market_return
            + rng.normal(0, 0.012 + index * 0.0003, rows)
        )
        values[f"S{index}"] = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(values, index=dates)


def test_online_feature_schema_is_finite_and_ordered():
    features = decision.latest_feature_frame(synthetic_history())
    assert list(features.columns) == PRICE_FEATURES
    assert len(features) == 8
    assert np.isfinite(features.to_numpy()).all()


def test_online_and_offline_feature_engineering_match():
    history = synthetic_history()
    online = decision.latest_feature_frame(history).sort_index()
    offline_panel = engineer_close_features(history)
    offline = (
        offline_panel.sort_values("date")
        .groupby("symbol")
        .tail(1)
        .set_index("symbol")[PRICE_FEATURES]
        .sort_index()
    )
    pd.testing.assert_frame_equal(
        online,
        offline,
        check_dtype=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_walk_forward_fold_has_twenty_session_embargo():
    panel_path = ROOT / "data" / "processed" / "decision_dataset.parquet"
    if not panel_path.exists():
        return
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    fold = time_fold(panel, 2021)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    embargo = dates[(dates > fold.train_cutoff) & (dates < fold.test_start)]
    assert len(embargo) == 20
    assert panel.loc[fold.train_mask, "date"].max() == fold.train_cutoff


def test_optimizer_enforces_position_and_change_limits():
    history = synthetic_history(symbols=8).pct_change().dropna()
    symbols = [f"S{index}" for index in range(8)]
    previous = np.full(8, 1 / 8)
    result = optimize_weights(
        symbols,
        expected_return_20d=np.linspace(-0.04, 0.06, 8),
        sigma_daily=np.linspace(0.012, 0.025, 8),
        return_history=history[symbols],
        previous_weights=previous,
    )
    assert result.success
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.maximum_position <= 0.2001
    assert result.maximum_change <= 0.0501
    assert (
        result.minimum_active_trade == 0
        or result.minimum_active_trade >= 0.0099
    )


def test_missing_deepseek_key_has_deterministic_fallback(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    payload = {
        "production_mode": "risk_only",
        "trades": [
            {
                "symbol": "AAPL",
                "weight_change_pct": -2.0,
                "reason": "risk budget",
            }
        ],
    }
    before = json.loads(json.dumps(payload))
    explanation = llm_client.explain_decision(payload)
    assert explanation["_meta"]["source"] == "deterministic_template"
    assert "reduce AAPL by 2.0" in explanation["summary"]
    assert payload == before


def test_deepseek_json_is_validated_and_cached(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "id": "test-request",
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "Hold the current allocation.",
                                    "reasons": ["Risk limits are satisfied."],
                                    "cautions": ["No trade is risk-free."],
                                    "confidence_note": "Numeric confidence only.",
                                }
                            )
                        }
                    }
                ],
            }

    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        calls["count"] += 1
        assert kwargs["json"]["response_format"] == {"type": "json_object"}
        assert "Authorization" in kwargs["headers"]
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only-key")
    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    llm_client._CACHE.clear()
    payload = {"production_mode": "risk_only", "trades": []}
    first = llm_client.explain_decision(payload)
    second = llm_client.explain_decision(payload)
    assert first["_meta"]["source"] == "deepseek"
    assert second["_meta"]["cache_hit"] is True
    assert calls["count"] == 1


def test_checkpoint_roundtrip_when_present():
    metadata_path = (
        ROOT / "data" / "processed" / "decision_model" / "decision_model.json"
    )
    model_path = (
        ROOT / "data" / "processed" / "decision_model" / "return_model.joblib"
    )
    if not metadata_path.exists() or not model_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model = joblib.load(model_path)
    sample = pd.read_parquet(
        ROOT / "data" / "processed" / "decision_dataset.parquet"
    ).tail(16)
    first = model.predict(sample[metadata["feature_order"]])
    reloaded = joblib.load(model_path)
    second = reloaded.predict(sample[metadata["feature_order"]])
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
