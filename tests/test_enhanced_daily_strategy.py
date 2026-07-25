from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.daily_strategy import learned  # noqa: E402
from src.interfaces import AssetSignal  # noqa: E402


def _history() -> pd.DataFrame:
    rng = np.random.default_rng(20260724)
    dates = pd.bdate_range("2023-01-02", periods=320)
    frame = pd.DataFrame(index=dates)
    for symbol in [f"S{index}" for index in range(10)] + ["SPY"]:
        frame[symbol] = 100.0 * np.cumprod(
            1.0 + rng.normal(0.0003, 0.012, len(dates))
        )
    return frame


def _signals() -> list[AssetSignal]:
    return [
        AssetSignal(
            symbol=f"S{index}",
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
        for index in range(10)
    ]


def test_direction_features_exclude_risk_health_and_news() -> None:
    assert "risk_sigma_daily_5d" not in learned.FEATURES
    assert "risk_level_5d" not in learned.FEATURES
    assert not any("health" in feature for feature in learned.FEATURES)
    assert not any("news" in feature for feature in learned.FEATURES)


def test_runtime_feature_order_matches_artifact() -> None:
    metadata = json.loads(
        learned.METADATA_PATH.read_text(encoding="utf-8")
    )
    frame = learned.latest_feature_frame(
        _history(),
        [signal.symbol for signal in _signals()],
    )
    assert not frame.empty
    assert list(frame.columns) == metadata["feature_order"]
    assert np.isfinite(frame.to_numpy(dtype=float)).all()


def test_unpromoted_model_is_exact_fallback() -> None:
    signals = _signals()
    result = learned.enhance_signals(
        _history(),
        signals,
        require_promoted=True,
    )
    assert result == signals
    assert learned.model_available(require_promoted=False)
    assert not learned.model_available(require_promoted=True)


def test_experimental_inference_is_bounded_and_reproducible() -> None:
    first = learned.enhance_signals(
        _history(),
        _signals(),
        require_promoted=False,
    )
    second = learned.enhance_signals(
        _history(),
        _signals(),
        require_promoted=False,
    )
    assert first == second
    assert len(first) == 10
    assert all(0.0 <= signal.score <= 100.0 for signal in first)
    assert all(
        -1.0 <= signal.indicators["enhanced_direction"] <= 1.0
        for signal in first
    )


def test_checkpoint_hash_matches_metadata() -> None:
    metadata = json.loads(
        learned.METADATA_PATH.read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(learned.MODEL_PATH.read_bytes()).hexdigest()
    assert digest == metadata["model_sha256"]
