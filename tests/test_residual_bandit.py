from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

import explore_residual_bandit as bandit  # noqa: E402


def test_bandit_time_splits_are_strictly_ordered():
    assert pd.Timestamp(bandit.TRAIN_END) < pd.Timestamp(
        bandit.VALIDATION_START
    )
    assert pd.Timestamp(bandit.VALIDATION_END) < pd.Timestamp(
        bandit.DIAGNOSTIC_START
    )


def test_counterfactual_hold_action_has_zero_reward():
    panel = bandit.prepare_panel().head(12)
    columns = bandit.feature_columns(news_enabled=False)
    bounds = bandit.fit_bounds(panel, columns)
    _, reward = bandit.counterfactual_training_data(panel, columns, bounds)
    reward = reward.reshape(len(panel), len(bandit.ACTIONS))
    np.testing.assert_allclose(reward[:, 1], 0.0, atol=0, rtol=0)


def test_higher_risk_reduces_identical_requested_action():
    columns = bandit.feature_columns(news_enabled=False)
    rows = pd.DataFrame(0.0, index=range(2), columns=columns)
    rows["strategy_score"] = 1.0
    rows["risk_level_5d"] = [0.0, 100.0]
    bounds = {column: (-10.0, 10.0) for column in columns}

    class PreferIncrease:
        def predict(self, values):
            # state_action_matrix = state, action, state*action
            return values[:, len(columns)]

    requested, _ = bandit.requested_policy_changes(
        PreferIncrease(),
        rows,
        columns,
        bounds,
        bandit.PolicyConfig(0.75, 0.0, 0.04),
        health_score=50.0,
    )
    assert requested[0] > requested[1] > 0


def test_experimental_checkpoint_roundtrip_when_present():
    path = (
        ROOT
        / "reports"
        / "decision_layer_bandit"
        / "candidate_bandit_full.joblib"
    )
    if not path.exists():
        return
    artifact = joblib.load(path)
    assert artifact["experimental_only"] is True
    assert artifact["training_range"] == [
        bandit.TRAIN_START,
        bandit.VALIDATION_END,
    ]
    assert artifact["feature_columns"]
    assert hasattr(artifact["model"], "predict")
