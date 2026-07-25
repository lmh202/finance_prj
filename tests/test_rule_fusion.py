from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.interfaces import AssetSignal, NewsEvent  # noqa: E402
from src.recommendation import fusion  # noqa: E402


def event(
    title: str,
    sentiment: float,
    importance: float,
    published: datetime,
    *,
    url: str = "",
    symbol: str = "GLD",
) -> NewsEvent:
    return NewsEvent(
        title=title,
        source="test",
        url=url,
        published=published,
        category="interest_rates",
        sentiment=sentiment,
        importance=importance,
        affected_symbols=[symbol],
        impact={symbol: "positive" if sentiment > 0 else "negative"},
    )


def test_gold_rulebook_example_reproduces_68_15():
    result = fusion.fuse_scores(
        "GLD",
        strategy_score=0.60,
        news_score=0.50,
        health_score=70,
        risk_percentile=70,
        news_article_count=3,
    )
    assert result.raw_score == pytest.approx(0.53)
    assert result.risk_factor == pytest.approx(0.685)
    assert result.adjusted_score == pytest.approx(0.36305)
    assert result.aurora_score == pytest.approx(68.2)
    assert result.outlook == "Moderately Positive"
    assert result.risk_level == "Moderate"


@pytest.mark.parametrize("direction", [-0.8, 0.8])
def test_risk_attenuates_but_never_reverses_direction(direction):
    low = fusion.fuse_scores(
        "AAPL",
        direction,
        0.0,
        50,
        0,
        news_article_count=0,
    )
    high = fusion.fuse_scores(
        "AAPL",
        direction,
        0.0,
        50,
        90,
        news_article_count=0,
    )
    assert low.adjusted_score * direction > 0
    assert high.adjusted_score * direction > 0
    assert abs(high.adjusted_score) < abs(low.adjusted_score)


def test_zero_and_sparse_news_reassign_weight_to_strategy():
    no_news = fusion.fuse_scores(
        "AAPL", 0.5, 0.0, 50, 20, news_article_count=0
    )
    sparse_news = fusion.fuse_scores(
        "AAPL", 0.5, -0.5, 50, 20, news_article_count=2
    )
    assert no_news.component_weights == {
        "strategy": 0.8,
        "news": 0.0,
        "health": 0.2,
    }
    assert sparse_news.component_weights == {
        "strategy": 0.7,
        "news": 0.1,
        "health": 0.2,
    }
    assert no_news.news_confidence == "Low"
    assert no_news.outlook != "Input Unavailable"


def test_news_is_weighted_and_duplicate_stories_count_once():
    now = datetime(2026, 7, 24, 12, tzinfo=timezone.utc)
    stories = [
        event(
            "Fed cuts rates after policy meeting",
            0.8,
            90,
            now - timedelta(hours=2),
            url="https://example.com/story?utm_source=rss",
        ),
        event(
            "Fed cuts rates after policy meeting",
            0.8,
            80,
            now - timedelta(hours=3),
            url="https://example.com/story",
        ),
        event(
            "Minor negative analyst comment",
            -0.5,
            20,
            now - timedelta(hours=30),
            url="https://example.com/other",
        ),
    ]
    aggregate = fusion.aggregate_news(stories, "GLD", now=now)
    assert aggregate.relevant_articles == 3
    assert aggregate.unique_articles == 2
    assert aggregate.score > 0.6
    assert aggregate.confidence == "Low"


def test_strong_strategy_news_conflict_forces_hold():
    result = fusion.fuse_scores(
        "TSLA",
        strategy_score=0.8,
        news_score=-0.9,
        health_score=70,
        risk_percentile=30,
        news_article_count=3,
    )
    assert result.conflict
    assert result.aurora_score == 50
    assert result.outlook == "Conflicting Signals"
    assert result.action == "Hold"
    assert result.position_change_pct == 0
    assert result.confidence_label == "Low"


def test_extreme_volatility_caps_positive_recommendation_and_size():
    result = fusion.fuse_scores(
        "NVDA",
        strategy_score=1.0,
        news_score=1.0,
        health_score=100,
        risk_percentile=95,
        news_article_count=4,
    )
    assert result.extreme_volatility
    assert result.aurora_score <= 74
    assert result.outlook != "Strong Positive"
    assert result.action == "Cautious Positive"
    assert result.position_change_pct <= 1
    assert result.risk_level == "Extreme"


def test_strategy_adapter_removes_legacy_volatility_directional_vote():
    signals = [
        AssetSignal(
            "LOWVOL",
            80,
            "increase",
            {
                "momentum": 0.1,
                "trend": 1,
                "sharpe": 1,
                "volatility": 0.1,
                "drawdown": -0.05,
            },
            "",
        ),
        AssetSignal(
            "HIGHVOL",
            20,
            "reduce",
            {
                "momentum": 0.1,
                "trend": 1,
                "sharpe": 1,
                "volatility": 0.8,
                "drawdown": -0.05,
            },
            "",
        ),
    ]
    scores = fusion.directional_strategy_scores(signals)
    assert scores["LOWVOL"] == pytest.approx(scores["HIGHVOL"])


def test_stale_inputs_are_explicit_and_lower_confidence():
    now = datetime(2026, 7, 24, 12, tzinfo=timezone.utc)
    signal = AssetSignal(
        "AAPL",
        70,
        "increase",
        {
            "momentum": 0.1,
            "trend": 1,
            "sharpe": 1,
            "volatility": 0.2,
            "drawdown": -0.05,
        },
        "",
    )
    estimate = SimpleNamespace(
        symbol="AAPL",
        horizon=5,
        has_history=True,
        risk_level=40.0,
        as_of="2026-07-10",
    )
    decision = fusion.recommend_portfolio(
        symbols=["AAPL"],
        signals=[signal],
        news_events=[],
        health_score=70,
        risk_estimates=[estimate],
        current_weights_pct={"AAPL": 10},
        strategy_as_of="2026-07-10",
        health_as_of="2026-07-10",
        now=now,
    )
    result = decision.assets[0]
    assert set(result.stale_inputs) == {"strategy", "risk", "health"}
    assert result.confidence <= 0.4


def test_missing_formal_risk_fails_closed_for_that_asset():
    signal = AssetSignal("AAPL", 90, "increase", {}, "bullish")
    decision = fusion.recommend_portfolio(
        symbols=["AAPL"],
        signals=[signal],
        news_events=[],
        health_score=70,
        risk_estimates=[],
        current_weights_pct={"AAPL": 10},
        strategy_as_of="2026-07-24",
        health_as_of="2026-07-24",
        now=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )
    result = decision.assets[0]
    assert "risk" in result.unavailable_inputs
    assert result.outlook == "Input Unavailable"
    assert result.action == "Hold"
    assert decision.recommendation.trades == []
