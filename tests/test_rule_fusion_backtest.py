from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))

import backtest_rule_fusion as backtest  # noqa: E402


def test_position_changes_are_funded_without_leverage():
    weights = np.array([0.18, 0.12, 0.10, 0.10, 0.10])
    cash = 0.40
    requested = np.array([0.04, 0.03, -0.03, 0.015, -0.015])
    after, after_cash, turnover, maximum_change = (
        backtest.apply_position_changes(weights, cash, requested)
    )
    assert np.isclose(after.sum() + after_cash, 1.0)
    assert (after >= 0).all()
    assert after_cash >= 0
    assert after.max() <= backtest.MAX_POSITION + 1e-12
    assert maximum_change <= 0.04 + 1e-12
    assert turnover >= 0


def test_buys_are_scaled_when_cash_and_sales_are_insufficient():
    weights = np.full(5, 0.20)
    requested = np.full(5, 0.04)
    after, cash, _, _ = backtest.apply_position_changes(
        weights,
        0.0,
        requested,
    )
    np.testing.assert_allclose(after, weights)
    assert cash == 0


def test_fixed_development_periods_do_not_overlap():
    validation_end = pd.Timestamp(backtest.PERIODS["validation"][1])
    test_start = pd.Timestamp(backtest.PERIODS["locked_test"][0])
    assert validation_end < test_start


def test_historical_feature_engineering_is_future_invariant(tmp_path):
    raw = pd.read_parquet(backtest.PANEL_PATH)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    cutoff = pd.Timestamp("2021-06-30")
    original = backtest.load_and_engineer_panel()

    altered = raw.copy()
    future = altered["date"].gt(cutoff)
    altered.loc[future, "ret_1d"] = altered.loc[future, "ret_1d"] * -7
    altered.loc[future, "sent_mean"] = 1.0
    altered.loc[future, "unique_story_count"] = 99
    temporary = tmp_path / "fusion_future_invariance.parquet"
    altered.to_parquet(temporary, index=False)
    try:
        changed = backtest.load_and_engineer_panel(temporary)
    finally:
        temporary.unlink(missing_ok=True)

    columns = [
        "date",
        "symbol",
        "strategy_score",
        "news_score_5d",
        "news_unique_5d",
    ]
    expected = original.loc[original["date"].le(cutoff), columns].reset_index(
        drop=True
    )
    actual = changed.loc[changed["date"].le(cutoff), columns].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(expected, actual)
