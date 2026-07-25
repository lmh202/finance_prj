"""Causal historical evaluation of the production rule-fusion decision layer.

The fixed rulebook is not fitted here.  It is evaluated on:

* validation: 2018-2020
* locked historical test: 2021-2023

Historical inputs reproduce the online components as closely as the stored
panel permits:

* strategy: cross-sectional momentum/trend/Sharpe/drawdown score, excluding
  volatility as a directional vote;
* news: next-session FNSPID/FinBERT sentiment, deduplicated story count,
  event-family importance proxy, and a five-session recency schedule;
* health: trailing two-year portfolio health calculated only from information
  available by the decision date;
* risk: formal out-of-fold HAR-X + News five-session risk percentile.

Signals observed at the close of t alter weights only after t's return, so the
new allocation first earns the return on t+1.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from decision_layer_core import backtest_metrics  # noqa: E402
from src.recommendation import fusion  # noqa: E402

PANEL_PATH = ROOT / "data" / "processed" / "decision_dataset.parquet"
OUT_DIR = ROOT / "reports" / "decision_layer_rule_fusion"

PERIODS = {
    "validation": ("2018-01-01", "2020-12-31"),
    "locked_test": ("2021-01-01", "2023-12-31"),
}
VARIANTS = (
    "equal_weight",
    "strategy_only",
    "strategy_risk",
    "fusion_no_news",
    "fusion_full",
    "fusion_full_no_risk",
)
COSTS_BPS = (0.0, 10.0, 25.0, 50.0)
PRIMARY_COST_BPS = 25.0
REBALANCE_SESSIONS = 5
INITIAL_CASH_WEIGHT = 0.05
MAX_POSITION = 0.20
MIN_TRADE = 0.01
TRADING_DAYS = 252
EPS = 1e-12
SEED = 20260724

STRATEGY_ONLY_CONFIG = fusion.FusionConfig(
    strategy_weight=1.0,
    news_weight=0.0,
    health_weight=0.0,
    low_news_weight=0.0,
)


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def load_and_engineer_panel(
    path: Path = PANEL_PATH,
    *,
    require_formal_risk: bool = True,
) -> pd.DataFrame:
    """Build causal strategy and recent-news inputs from the stored panel."""
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    grouped = panel.groupby("symbol", sort=False)
    rolling_mean = grouped["ret_1d"].transform(
        lambda values: values.rolling(252, min_periods=126).mean()
    )
    rolling_std = grouped["ret_1d"].transform(
        lambda values: values.rolling(252, min_periods=126).std()
    )
    panel["strategy_sharpe"] = (
        rolling_mean * TRADING_DAYS
        / (rolling_std * math.sqrt(TRADING_DAYS)).replace(0, np.nan)
    )
    panel["strategy_trend"] = (
        0.5 * panel["price_vs_sma50"].gt(0).astype(float)
        + 0.5 * panel["sma50_vs_sma200"].gt(0).astype(float)
    )

    price_index = grouped["ret_1d"].transform(
        lambda values: (1.0 + values.fillna(0.0)).cumprod()
    )
    panel["_price_index"] = price_index
    rolling_peak = panel.groupby("symbol", sort=False)["_price_index"].transform(
        lambda values: values.rolling(252, min_periods=60).max()
    )
    panel["strategy_drawdown"] = panel["_price_index"] / rolling_peak - 1.0

    rank_columns = {
        "mom_60d": "_rank_momentum",
        "strategy_trend": "_rank_trend",
        "strategy_sharpe": "_rank_sharpe",
        "strategy_drawdown": "_rank_drawdown",
    }
    for source, target in rank_columns.items():
        panel[target] = panel.groupby("date")[source].rank(pct=True)
    directional_01 = (
        0.30 * panel["_rank_momentum"]
        + 0.25 * panel["_rank_trend"]
        + 0.20 * panel["_rank_sharpe"]
        + 0.10 * panel["_rank_drawdown"]
    ) / 0.85
    panel["strategy_score"] = (2.0 * directional_01 - 1.0).clip(-1, 1)

    # The historical artifact has event-family shares rather than the live
    # per-article importance score. This is a causal daily importance proxy
    # using the same 1.0 / 0.8 / 0.5 / 0.2 hierarchy.
    event_weights = {
        "event_macro_share": 1.00,
        "event_legal_regulatory_share": 1.00,
        "event_earnings_share": 0.80,
        "event_corporate_action_share": 0.80,
        "event_analyst_share": 0.50,
        "event_product_share": 0.50,
        "event_management_share": 0.50,
        "event_financing_share": 0.50,
    }
    importance = sum(
        panel[column].fillna(0.0) * weight
        for column, weight in event_weights.items()
    )
    detected_share = sum(
        panel[column].fillna(0.0) for column in event_weights
    ).clip(0, 1)
    panel["_news_importance"] = (
        importance + (1.0 - detected_share) * 0.20
    ).clip(0.20, 1.00)
    panel["_news_unique"] = panel["unique_story_count"].fillna(0.0).clip(lower=0)
    panel["_news_daily_weight"] = (
        panel["_news_unique"] * panel["_news_importance"]
    )
    panel["_news_daily_numerator"] = (
        panel["_news_daily_weight"] * panel["sent_mean"].fillna(0.0)
    )

    recency = (1.00, 0.75, 0.50, 0.25, 0.25)
    numerator = pd.Series(0.0, index=panel.index)
    denominator = pd.Series(0.0, index=panel.index)
    story_count = pd.Series(0.0, index=panel.index)
    for lag, weight in enumerate(recency):
        numerator += (
            panel.groupby("symbol", sort=False)["_news_daily_numerator"]
            .shift(lag)
            .fillna(0.0)
            * weight
        )
        denominator += (
            panel.groupby("symbol", sort=False)["_news_daily_weight"]
            .shift(lag)
            .fillna(0.0)
            * weight
        )
        story_count += (
            panel.groupby("symbol", sort=False)["_news_unique"]
            .shift(lag)
            .fillna(0.0)
        )
    panel["news_score_5d"] = (
        numerator / denominator.replace(0, np.nan)
    ).fillna(0.0).clip(-1, 1)
    panel["news_unique_5d"] = story_count.round().clip(lower=0)

    panel["risk_level_5d"] = panel["risk_level_5d"].clip(0, 100)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    required = [
        "strategy_score",
        "news_score_5d",
        "news_unique_5d",
        "ret_1d",
        "benchmark_ret_1d",
        "fwd_ret_5d",
        "fwd_ret_20d",
    ]
    if require_formal_risk:
        required += ["risk_level_5d", "risk_sigma_daily_5d"]
    return panel.dropna(subset=required).sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)


def portfolio_health_score(
    returns_history: pd.DataFrame,
    asset_weights: np.ndarray,
) -> float:
    """Production Health formula using only the trailing two-year window."""
    total_asset_weight = float(asset_weights.sum())
    if total_asset_weight <= EPS or returns_history.empty:
        return 50.0
    weights = asset_weights / total_asset_weight
    history = returns_history.tail(504).fillna(0.0)
    if len(history) < 30:
        return 50.0
    portfolio_return = history.to_numpy(dtype=float) @ weights
    annual_return = float(np.mean(portfolio_return) * TRADING_DAYS)
    annual_volatility = float(
        np.std(portfolio_return, ddof=1) * math.sqrt(TRADING_DAYS)
    )
    sharpe = (
        annual_return / annual_volatility
        if annual_volatility > EPS
        else 0.0
    )
    curve = np.cumprod(1.0 + portfolio_return)
    max_drawdown = float(
        np.min(curve / np.maximum.accumulate(curve) - 1.0)
    )
    diversification = 1.0 - float(np.sum(weights**2))
    largest = float(np.max(weights))
    score = (
        25 * _clip01(sharpe / 2)
        + 20 * _clip01(diversification / 0.9)
        + 20 * _clip01(1 + max_drawdown / 0.4)
        + 15 * _clip01(1 - annual_volatility / 0.4)
        + 10 * _clip01(1 - largest / 0.5)
        + 10 * _clip01(0.5 + (annual_return - 0.08) / 0.3)
    )
    return float(np.clip(score, 0, 100))


def _variant_result(
    variant: str,
    symbol: str,
    strategy_score: float,
    news_score: float,
    news_count: int,
    health_score: float,
    risk_percentile: float,
) -> fusion.AssetFusionResult:
    if variant == "strategy_only":
        return fusion.fuse_scores(
            symbol,
            strategy_score,
            0.0,
            50.0,
            0.0,
            news_article_count=0,
            config=STRATEGY_ONLY_CONFIG,
        )
    if variant == "strategy_risk":
        return fusion.fuse_scores(
            symbol,
            strategy_score,
            0.0,
            50.0,
            risk_percentile,
            news_article_count=0,
            config=STRATEGY_ONLY_CONFIG,
        )
    if variant == "fusion_no_news":
        return fusion.fuse_scores(
            symbol,
            strategy_score,
            0.0,
            health_score,
            risk_percentile,
            news_article_count=0,
        )
    if variant == "fusion_full_no_risk":
        return fusion.fuse_scores(
            symbol,
            strategy_score,
            news_score,
            health_score,
            0.0,
            news_article_count=news_count,
        )
    if variant == "fusion_full":
        return fusion.fuse_scores(
            symbol,
            strategy_score,
            news_score,
            health_score,
            risk_percentile,
            news_article_count=news_count,
        )
    raise ValueError(f"unsupported variant: {variant}")


def apply_position_changes(
    asset_weights: np.ndarray,
    cash_weight: float,
    requested_change: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Apply independent recommendations without leverage, using cash as buffer."""
    before = asset_weights.copy()
    before_cash = float(cash_weight)
    weights = before.copy()
    cash = before_cash

    sells = np.minimum(np.minimum(requested_change, 0.0), 0.0)
    sells = np.maximum(sells, -weights)
    weights += sells
    cash -= float(sells.sum())

    buys = np.maximum(requested_change, 0.0)
    capacity = np.maximum(MAX_POSITION - weights, 0.0)
    buys = np.minimum(buys, capacity)
    requested_total = float(buys.sum())
    if requested_total > cash + EPS:
        buys *= cash / requested_total
    buys[buys < MIN_TRADE] = 0.0
    requested_total = float(buys.sum())
    if requested_total > cash + EPS:
        buys *= cash / requested_total
    weights += buys
    cash -= float(buys.sum())

    weights = np.maximum(weights, 0.0)
    cash = max(0.0, cash)
    total = float(weights.sum() + cash)
    weights /= max(total, EPS)
    cash /= max(total, EPS)

    delta = weights - before
    cash_delta = cash - before_cash
    turnover = 0.5 * (float(np.abs(delta).sum()) + abs(cash_delta))
    return weights, cash, turnover, float(np.max(np.abs(delta)))


def eligible_symbols(
    panel: pd.DataFrame,
    start: str,
    end: str,
) -> list[str]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    subset = panel.loc[panel["date"].between(start_ts, end_ts)]
    dates = subset["date"].nunique()
    counts = subset.groupby("symbol")["date"].nunique()
    return sorted(counts[counts.ge(max(1, int(dates * 0.85)))].index)


def run_variant(
    panel: pd.DataFrame,
    *,
    period: str,
    start: str,
    end: str,
    variant: str,
    initial_cash_weight: float = INITIAL_CASH_WEIGHT,
) -> pd.DataFrame:
    """Run one causal portfolio path before applying transaction costs."""
    symbols = eligible_symbols(panel, start, end)
    if len(symbols) < 5:
        raise ValueError(f"{period}: fewer than five complete symbols")
    filtered = panel.loc[panel["symbol"].isin(symbols)].copy()
    returns = filtered.pivot(index="date", columns="symbol", values="ret_1d")
    returns = returns.reindex(columns=symbols).sort_index()
    benchmark = (
        filtered.groupby("date")["benchmark_ret_1d"].first().sort_index()
    )
    strategy = filtered.pivot(
        index="date", columns="symbol", values="strategy_score"
    ).reindex(columns=symbols)
    news = filtered.pivot(
        index="date", columns="symbol", values="news_score_5d"
    ).reindex(columns=symbols)
    news_count = filtered.pivot(
        index="date", columns="symbol", values="news_unique_5d"
    ).reindex(columns=symbols)
    risk_level = filtered.pivot(
        index="date", columns="symbol", values="risk_level_5d"
    ).reindex(columns=symbols)

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    dates = returns.index[returns.index.to_series().between(start_ts, end_ts)]
    asset_weights = np.full(
        len(symbols), (1.0 - initial_cash_weight) / len(symbols)
    )
    cash_weight = initial_cash_weight
    rebalance_set = set(dates[::REBALANCE_SESSIONS])
    rows = []

    for date in dates:
        daily_returns = (
            returns.loc[date].reindex(symbols).fillna(0.0).to_numpy(dtype=float)
        )
        gross_return = float(asset_weights @ daily_returns)

        # Let weights drift through today's close before making a decision.
        portfolio_growth = max(1.0 + gross_return, EPS)
        asset_weights = asset_weights * (1.0 + daily_returns) / portfolio_growth
        cash_weight = cash_weight / portfolio_growth
        total = float(asset_weights.sum() + cash_weight)
        asset_weights /= max(total, EPS)
        cash_weight /= max(total, EPS)

        turnover = 0.0
        maximum_change = 0.0
        health_score = portfolio_health_score(
            returns.loc[:date, symbols],
            asset_weights,
        )
        if variant != "equal_weight" and date in rebalance_set:
            requested = np.zeros(len(symbols), dtype=float)
            for index, symbol in enumerate(symbols):
                result = _variant_result(
                    variant,
                    symbol,
                    float(strategy.at[date, symbol]),
                    float(news.at[date, symbol]),
                    int(news_count.at[date, symbol]),
                    health_score,
                    float(risk_level.at[date, symbol]),
                )
                requested[index] = result.position_change_pct / 100.0
            (
                asset_weights,
                cash_weight,
                turnover,
                maximum_change,
            ) = apply_position_changes(
                asset_weights,
                cash_weight,
                requested,
            )

        rows.append(
            {
                "date": date,
                "period": period,
                "strategy": variant,
                "gross_return": gross_return,
                "benchmark_return": float(benchmark.loc[date]),
                "turnover": turnover,
                "cash_weight": cash_weight,
                "maximum_weight": float(asset_weights.max()),
                "maximum_change": maximum_change,
                "health_score": health_score,
                "n_positions": int(np.sum(asset_weights > 1e-4)),
                "n_symbols": len(symbols),
            }
        )
    return pd.DataFrame(rows)


def add_cost_scenarios(base_daily: pd.DataFrame) -> pd.DataFrame:
    scenarios = []
    for cost_bps in COSTS_BPS:
        frame = base_daily.copy()
        frame["transaction_cost_bps"] = cost_bps
        frame["transaction_cost"] = (
            frame["turnover"] * cost_bps / 10_000.0
        )
        frame["net_return"] = (
            frame["gross_return"] - frame["transaction_cost"]
        )
        frame["optimizer_success"] = True
        frame["max_weight"] = frame["maximum_weight"]
        frame["max_change"] = frame["maximum_change"]
        frame["minimum_active_trade"] = 0.0
        scenarios.append(frame)
    return pd.concat(scenarios, ignore_index=True)


def evaluate_portfolios(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["period", "strategy", "transaction_cost_bps"]
    for keys, frame in daily.groupby(group_columns, sort=False):
        metrics = backtest_metrics(frame.sort_values("date"))
        rows.append(
            {
                "period": keys[0],
                "strategy": keys[1],
                "transaction_cost_bps": keys[2],
                **metrics,
                "average_cash_weight": float(frame["cash_weight"].mean()),
                "average_health_score": float(frame["health_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_years(daily: pd.DataFrame) -> pd.DataFrame:
    primary = daily.loc[
        daily["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ].copy()
    primary["year"] = primary["date"].dt.year
    rows = []
    for (period, strategy, year), frame in primary.groupby(
        ["period", "strategy", "year"], sort=False
    ):
        metrics = backtest_metrics(frame.sort_values("date"))
        rows.append(
            {
                "period": period,
                "strategy": strategy,
                "year": int(year),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def evaluate_cash_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    """Check that the locked-test conclusion is not a 5% cash artefact."""
    rows = []
    start, end = PERIODS["locked_test"]
    for initial_cash in (0.0, 0.05, 0.10):
        for variant in (
            "equal_weight",
            "strategy_only",
            "fusion_no_news",
            "fusion_full",
        ):
            daily = run_variant(
                panel,
                period="locked_test",
                start=start,
                end=end,
                variant=variant,
                initial_cash_weight=initial_cash,
            )
            daily["transaction_cost_bps"] = PRIMARY_COST_BPS
            daily["transaction_cost"] = (
                daily["turnover"] * PRIMARY_COST_BPS / 10_000.0
            )
            daily["net_return"] = (
                daily["gross_return"] - daily["transaction_cost"]
            )
            daily["optimizer_success"] = True
            daily["max_weight"] = daily["maximum_weight"]
            daily["max_change"] = daily["maximum_change"]
            daily["minimum_active_trade"] = 0.0
            rows.append(
                {
                    "initial_cash_weight": initial_cash,
                    "strategy": variant,
                    **backtest_metrics(daily),
                }
            )
    return pd.DataFrame(rows)


def equal_weight_health_by_date(panel: pd.DataFrame) -> pd.Series:
    returns = panel.pivot(index="date", columns="symbol", values="ret_1d")
    values = {}
    for date in returns.index:
        available = returns.loc[:date].tail(504).dropna(axis=1, how="all")
        weights = np.full(len(available.columns), 1.0 / len(available.columns))
        values[date] = portfolio_health_score(available, weights)
    return pd.Series(values, name="health_score")


def score_variant_frame(
    panel: pd.DataFrame,
    variant: str,
    health_by_date: pd.Series,
) -> pd.DataFrame:
    frame = panel.copy()
    frame["health_score"] = frame["date"].map(health_by_date).fillna(50.0)
    strategy = frame["strategy_score"].clip(-1, 1)
    news = frame["news_score_5d"].clip(-1, 1)
    health = ((frame["health_score"] - 50.0) / 50.0).clip(-1, 1)
    risk = frame["risk_level_5d"].clip(0, 100)

    if variant == "strategy_only":
        raw = strategy
        factor = pd.Series(1.0, index=frame.index)
    elif variant == "strategy_risk":
        raw = strategy
        factor = 1.0 - 0.45 * risk / 100.0
    else:
        if variant == "fusion_no_news":
            news_weight = pd.Series(0.0, index=frame.index)
            news = pd.Series(0.0, index=frame.index)
        else:
            news_weight = pd.Series(
                np.select(
                    [
                        frame["news_unique_5d"].ge(3),
                        frame["news_unique_5d"].gt(0),
                    ],
                    [0.30, 0.10],
                    default=0.0,
                ),
                index=frame.index,
            )
        strategy_weight = 0.50 + (0.30 - news_weight)
        raw = strategy_weight * strategy + news_weight * news + 0.20 * health
        factor = (
            pd.Series(1.0, index=frame.index)
            if variant == "fusion_full_no_risk"
            else 1.0 - 0.45 * risk / 100.0
        )

    adjusted = raw * factor
    score = (50.0 + 50.0 * adjusted).clip(0, 100)
    if variant in {"fusion_full", "fusion_full_no_risk"}:
        conflict = (
            strategy.abs().ge(0.60)
            & news.abs().ge(0.60)
            & strategy.mul(news).lt(0)
            & frame["news_unique_5d"].gt(0)
        )
        score = score.mask(conflict, 50.0)
    score = score.mask(risk.gt(90) & score.gt(74), 74.0)
    frame["decision_score"] = score
    frame["decision_signal"] = (score - 50.0) / 50.0
    frame["strategy"] = variant
    return frame


def _daily_rank_ic(frame: pd.DataFrame, target: str) -> float:
    values = []
    for _, day in frame.groupby("date"):
        if day["decision_score"].nunique() < 2 or day[target].nunique() < 2:
            continue
        value = day["decision_score"].corr(day[target], method="spearman")
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _top_bottom_spread(frame: pd.DataFrame, target: str) -> float:
    spreads = []
    for _, day in frame.groupby("date"):
        if len(day) < 5 or day["decision_score"].nunique() < 2:
            continue
        ranks = day["decision_score"].rank(pct=True)
        top = day.loc[ranks.gt(0.8), target].mean()
        bottom = day.loc[ranks.le(0.2), target].mean()
        if np.isfinite(top) and np.isfinite(bottom):
            spreads.append(float(top - bottom))
    return float(np.mean(spreads)) if spreads else float("nan")


def signal_metrics(
    scored: pd.DataFrame,
    *,
    period: str,
    scope: str,
) -> dict:
    frame = scored
    if scope == "news_covered":
        frame = frame.loc[frame["news_unique_5d"].gt(0)]
    active = frame.loc[
        frame["decision_score"].ge(55)
        | frame["decision_score"].lt(45)
    ]
    actual_positive = active["fwd_ret_20d"].gt(0).astype(int)
    if len(active) and actual_positive.nunique() > 1:
        auc = float(
            roc_auc_score(actual_positive, active["decision_score"])
        )
    else:
        auc = float("nan")
    predicted_positive = active["decision_score"].ge(55)
    accuracy = (
        float((predicted_positive == actual_positive.astype(bool)).mean())
        if len(active)
        else float("nan")
    )
    return {
        "period": period,
        "strategy": str(scored["strategy"].iloc[0]),
        "scope": scope,
        "rows": int(len(frame)),
        "dates": int(frame["date"].nunique()),
        "active_share": float(len(active) / max(len(frame), 1)),
        "direction_accuracy_20d": accuracy,
        "auc_20d": auc,
        "rank_ic_5d": _daily_rank_ic(frame, "fwd_ret_5d"),
        "rank_ic_20d": _daily_rank_ic(frame, "fwd_ret_20d"),
        "top_bottom_spread_5d": _top_bottom_spread(frame, "fwd_ret_5d"),
        "top_bottom_spread_20d": _top_bottom_spread(frame, "fwd_ret_20d"),
    }


def evaluate_signals(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    health = equal_weight_health_by_date(panel)
    metric_rows = []
    score_frames = []
    for period, (start, end) in PERIODS.items():
        subset = panel.loc[panel["date"].between(start, end)]
        for variant in VARIANTS:
            if variant == "equal_weight":
                continue
            scored = score_variant_frame(subset, variant, health)
            score_frames.append(
                scored[
                    [
                        "date",
                        "symbol",
                        "strategy",
                        "decision_score",
                        "strategy_score",
                        "news_score_5d",
                        "news_unique_5d",
                        "risk_level_5d",
                        "health_score",
                        "fwd_ret_5d",
                        "fwd_ret_20d",
                    ]
                ].assign(period=period)
            )
            for scope in ("all", "news_covered"):
                metric_rows.append(
                    signal_metrics(scored, period=period, scope=scope)
                )
    return pd.DataFrame(metric_rows), pd.concat(score_frames, ignore_index=True)


def newey_west_mean_test(values: np.ndarray, lag: int = 5) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    centered = values - values.mean()
    long_run_variance = float(centered @ centered / n)
    for offset in range(1, min(lag, n - 1) + 1):
        covariance = float(centered[offset:] @ centered[:-offset] / n)
        long_run_variance += (
            2.0 * (1.0 - offset / (lag + 1.0)) * covariance
        )
    standard_error = math.sqrt(max(long_run_variance, 0.0) / max(n, 1))
    statistic = float(values.mean() / standard_error) if standard_error else 0.0
    return {
        "mean_daily_return_gain": float(values.mean()),
        "newey_west_lag": lag,
        "newey_west_t": statistic,
        "newey_west_p": float(2.0 * norm.sf(abs(statistic))),
    }


def _certainty_equivalent(values: np.ndarray) -> float:
    return float(
        np.mean(values) * TRADING_DAYS
        - 0.5 * 6.0 * np.var(values, ddof=1) * TRADING_DAYS
    )


def block_bootstrap_cer_gain(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    block_length: int = 20,
    samples: int = 2000,
    seed: int = SEED,
) -> dict:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if len(reference) != len(candidate):
        raise ValueError("paired bootstrap inputs must have equal length")
    rng = np.random.default_rng(seed)
    n = len(reference)
    gains = np.empty(samples)
    maximum_start = max(n - block_length + 1, 1)
    for sample in range(samples):
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, maximum_start))
            indices.extend(range(start, min(start + block_length, n)))
        chosen = np.asarray(indices[:n], dtype=int)
        gains[sample] = (
            _certainty_equivalent(candidate[chosen])
            - _certainty_equivalent(reference[chosen])
        )
    return {
        "block_length": block_length,
        "bootstrap_samples": samples,
        "cer_gain_ci_low": float(np.quantile(gains, 0.025)),
        "cer_gain_ci_high": float(np.quantile(gains, 0.975)),
        "probability_cer_gain_positive": float(np.mean(gains > 0)),
    }


def news_contribution_tests(daily: pd.DataFrame) -> dict:
    primary = daily.loc[
        daily["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ]
    output = {}
    for period in PERIODS:
        reference = (
            primary.loc[
                primary["period"].eq(period)
                & primary["strategy"].eq("fusion_no_news")
            ]
            .sort_values("date")
            .set_index("date")["net_return"]
        )
        candidate = (
            primary.loc[
                primary["period"].eq(period)
                & primary["strategy"].eq("fusion_full")
            ]
            .sort_values("date")
            .set_index("date")["net_return"]
        )
        common = reference.index.intersection(candidate.index)
        difference = (
            candidate.reindex(common).to_numpy()
            - reference.reindex(common).to_numpy()
        )
        output[period] = {
            **newey_west_mean_test(difference),
            **block_bootstrap_cer_gain(
                reference.reindex(common).to_numpy(),
                candidate.reindex(common).to_numpy(),
            ),
        }
    return output


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: (
                f"{value:.{digits}f}" if pd.notna(value) else ""
            )
        )
    columns = list(display.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| "
        + " | ".join(str(value) for value in row)
        + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def write_report(
    panel: pd.DataFrame,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    signal: pd.DataFrame,
    contribution: dict,
    cash_sensitivity: pd.DataFrame,
) -> dict:
    primary = metrics.loc[
        metrics["transaction_cost_bps"].eq(PRIMARY_COST_BPS)
    ]
    columns = [
        "period",
        "strategy",
        "cagr",
        "sharpe",
        "certainty_equivalent",
        "max_drawdown",
        "annual_turnover",
        "average_cash_weight",
    ]
    portfolio_display = primary[columns].sort_values(
        ["period", "strategy"]
    )
    signal_display = signal.loc[
        signal["scope"].eq("all"),
        [
            "period",
            "strategy",
            "direction_accuracy_20d",
            "auc_20d",
            "rank_ic_5d",
            "rank_ic_20d",
            "top_bottom_spread_20d",
        ],
    ].sort_values(["period", "strategy"])
    cash_display = cash_sensitivity[
        [
            "initial_cash_weight",
            "strategy",
            "cagr",
            "sharpe",
            "certainty_equivalent",
            "max_drawdown",
        ]
    ]

    def row(period: str, strategy: str) -> pd.Series:
        return primary.loc[
            primary["period"].eq(period)
            & primary["strategy"].eq(strategy)
        ].iloc[0]

    validation_full = row("validation", "fusion_full")
    validation_no_news = row("validation", "fusion_no_news")
    test_full = row("locked_test", "fusion_full")
    test_no_news = row("locked_test", "fusion_no_news")
    test_strategy = row("locked_test", "strategy_only")
    test_equal = row("locked_test", "equal_weight")
    test_stats = contribution["locked_test"]

    checks = {
        "validation_cer_above_no_news": bool(
            validation_full["certainty_equivalent"]
            > validation_no_news["certainty_equivalent"]
        ),
        "locked_test_cer_above_no_news": bool(
            test_full["certainty_equivalent"]
            > test_no_news["certainty_equivalent"]
        ),
        "locked_test_sharpe_above_no_news": bool(
            test_full["sharpe"] > test_no_news["sharpe"]
        ),
        "locked_test_cer_above_strategy_only": bool(
            test_full["certainty_equivalent"]
            > test_strategy["certainty_equivalent"]
        ),
        "locked_test_cer_above_equal_weight": bool(
            test_full["certainty_equivalent"]
            > test_equal["certainty_equivalent"]
        ),
        "locked_test_news_nw_p_below_005": bool(
            test_stats["newey_west_p"] < 0.05
            and test_stats["mean_daily_return_gain"] > 0
        ),
        "locked_test_news_bootstrap_ci_positive": bool(
            test_stats["cer_gain_ci_low"] > 0
        ),
        "locked_test_drawdown_not_worse_than_no_news_2pp": bool(
            test_full["max_drawdown"]
            >= test_no_news["max_drawdown"] - 0.02
        ),
    }
    news_validated = bool(
        checks["validation_cer_above_no_news"]
        and checks["locked_test_cer_above_no_news"]
        and checks["locked_test_sharpe_above_no_news"]
        and checks["locked_test_news_nw_p_below_005"]
        and checks["locked_test_news_bootstrap_ci_positive"]
    )
    deployment_validated = bool(
        news_validated
        and checks["locked_test_cer_above_strategy_only"]
        and checks["locked_test_drawdown_not_worse_than_no_news_2pp"]
    )
    conclusion = {
        "news_contribution_validated": news_validated,
        "fusion_deployment_validated": deployment_validated,
        "checks": checks,
        "primary_transaction_cost_bps": PRIMARY_COST_BPS,
    }

    coverage = (
        panel.loc[panel["date"].dt.year.between(2018, 2023)]
        .assign(year=lambda frame: frame["date"].dt.year)
        .groupby("year")
        .agg(
            rows=("symbol", "size"),
            symbols=("symbol", "nunique"),
            news_day_share=("has_news", "mean"),
            unique_stories=("unique_story_count", "sum"),
        )
        .reset_index()
    )
    status = (
        "PASSED"
        if deployment_validated
        else "DID NOT PASS"
    )
    report = f"""# Rule-Fusion Decision Layer Backtest

## Bottom line

The fixed rule-fusion candidate **{status}** the historical deployment gates.
News contribution validated: **{str(news_validated).lower()}**.

This result evaluates the actual rule structure, not the previous ML optimiser.
The 50/30/20 weights were not retuned on the locked test.

## Protocol

- Validation: **2018-2020**
- Locked historical test: **2021-2023**
- Rebalance interval: **{REBALANCE_SESSIONS} sessions**
- Primary one-way transaction cost: **{PRIMARY_COST_BPS:.0f} bps**
- Initial cash buffer: **{INITIAL_CASH_WEIGHT:.0%}**
- Directional inputs at close t first affect returns on t+1
- Risk input: out-of-fold HAR-X + News five-session percentile
- News input: next-session FNSPID/FinBERT features with causal five-session decay
- Health input: trailing two-year portfolio health, recalculated through t

## Portfolio performance

{markdown_table(portfolio_display)}

## Signal diagnostics

{markdown_table(signal_display)}

## News contribution: full fusion minus no-news fusion

```json
{json.dumps(contribution, indent=2)}
```

## Locked-test initial-cash sensitivity

{markdown_table(cash_display)}

## Deployment checks

```json
{json.dumps(conclusion, indent=2)}
```

## Data coverage

{markdown_table(coverage)}

## Interpretation limits

- The historical dataset contains deduplicated story counts, FinBERT
  sentiment, and event-family shares, but not the live per-article importance
  field. Event-family weights are therefore used as a documented importance
  proxy.
- Historical article timestamps are represented by their causal next-trading
  session. The five-step decay uses trading sessions rather than exact hours.
- The full news-enabled fusion cannot be tested on the 2024-2026 external
  panel because that panel has no archived RSS/FinBERT news channel.
- Portfolio results assume a 5% initial cash buffer because independent
  increase/reduce recommendations require a funding convention. Buys cannot
  exceed available cash plus sale proceeds, and leverage is prohibited.
- This is an evaluation of a deterministic allocation rule, not evidence that
  any individual recommendation will be profitable.
"""
    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    (OUT_DIR / "conclusion.json").write_text(
        json.dumps(conclusion, indent=2),
        encoding="utf-8",
    )
    return conclusion


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_and_engineer_panel()

    base_paths = []
    for period, (start, end) in PERIODS.items():
        for variant in VARIANTS:
            print(f"[backtest] {period:11s} {variant}")
            base_paths.append(
                run_variant(
                    panel,
                    period=period,
                    start=start,
                    end=end,
                    variant=variant,
                )
            )
    base_daily = pd.concat(base_paths, ignore_index=True)
    daily = add_cost_scenarios(base_daily)
    metrics = evaluate_portfolios(daily)
    yearly = evaluate_years(daily)
    signal, scores = evaluate_signals(panel)
    contribution = news_contribution_tests(daily)
    cash_sensitivity = evaluate_cash_sensitivity(panel)

    daily.to_parquet(OUT_DIR / "daily_returns.parquet", index=False)
    scores.to_parquet(OUT_DIR / "signal_scores.parquet", index=False)
    metrics.to_csv(OUT_DIR / "portfolio_metrics.csv", index=False)
    yearly.to_csv(OUT_DIR / "yearly_metrics.csv", index=False)
    signal.to_csv(OUT_DIR / "signal_metrics.csv", index=False)
    cash_sensitivity.to_csv(
        OUT_DIR / "cash_sensitivity.csv",
        index=False,
    )
    (OUT_DIR / "news_contribution.json").write_text(
        json.dumps(contribution, indent=2),
        encoding="utf-8",
    )
    config = {
        "panel": str(PANEL_PATH.relative_to(ROOT)),
        "periods": PERIODS,
        "variants": VARIANTS,
        "costs_bps": COSTS_BPS,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "rebalance_sessions": REBALANCE_SESSIONS,
        "initial_cash_weight": INITIAL_CASH_WEIGHT,
        "fusion_config": asdict(fusion.DEFAULT_CONFIG),
        "seed": SEED,
    }
    (OUT_DIR / "backtest_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    conclusion = write_report(
        panel,
        metrics,
        yearly,
        signal,
        contribution,
        cash_sensitivity,
    )
    print(json.dumps(conclusion, indent=2))
    print(f"Report: {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
