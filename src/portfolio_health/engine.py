"""Portfolio Intelligence Engine — Developer 1 (Architecture.md §4).

This is a WORKING BASELINE so the app runs end-to-end from day one.
It is intentionally simple — your job is to refine it (see README.md).
The public signature must stay exactly as declared in src/interfaces.py.
"""

import numpy as np
import pandas as pd

from src.interfaces import BENCHMARK, HealthReport

TRADING_DAYS = 252


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def compute_health(
    holdings: pd.DataFrame, history: pd.DataFrame, benchmark: str = BENCHMARK
) -> HealthReport:
    """Evaluate the portfolio: risk/return metrics + 0-100 health score."""
    symbols = [s for s in holdings["symbol"] if s in history.columns]
    if not symbols or len(history) < 30:
        return HealthReport(
            score=0.0, metrics={}, strengths=[],
            weaknesses=["Not enough price history to assess the portfolio."],
        )

    last = history[symbols].ffill().iloc[-1]
    shares = holdings.set_index("symbol").loc[symbols, "shares"]
    values = last * shares
    weights = values / values.sum()

    rets = history[symbols].pct_change().dropna(how="all")
    port = (rets * weights).sum(axis=1)

    ann_ret = float(port.mean() * TRADING_DAYS)
    ann_vol = float(port.std() * np.sqrt(TRADING_DAYS))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    downside = float(port[port < 0].std() * np.sqrt(TRADING_DAYS))
    sortino = ann_ret / downside if downside > 0 else 0.0
    curve = (1 + port).cumprod()
    max_dd = float((curve / curve.cummax() - 1).min())

    beta = float("nan")
    if benchmark in history.columns:
        bret = history[benchmark].pct_change().reindex(port.index)
        var = float(bret.var())
        if var > 0:
            beta = float(port.cov(bret) / var)

    herfindahl = float((weights ** 2).sum())
    diversification = 1.0 - herfindahl
    largest = float(weights.max())

    # Crude blend per Architecture.md §4 weightings — improve the curves!
    score = (
        25 * _clip01(sharpe / 2)
        + 20 * _clip01(diversification / 0.9)
        + 20 * _clip01(1 + max_dd / 0.4)
        + 15 * _clip01(1 - ann_vol / 0.4)
        + 10 * _clip01(1 - largest / 0.5)
        + 10 * _clip01(0.5 + (ann_ret - 0.08) / 0.3)
    )

    strengths, weaknesses = [], []
    if sharpe >= 1.0:
        strengths.append(f"Strong risk-adjusted return (Sharpe {sharpe:.2f}).")
    elif sharpe < 0.5:
        weaknesses.append(f"Weak risk-adjusted return (Sharpe {sharpe:.2f}).")
    if diversification >= 0.7:
        strengths.append("Well diversified across holdings.")
    if largest > 0.25:
        weaknesses.append(f"Largest position is {largest:.0%} of the portfolio.")
    if max_dd < -0.25:
        weaknesses.append(f"Deep historical drawdown ({max_dd:.0%}).")
    elif max_dd > -0.12:
        strengths.append(f"Moderate drawdown ({max_dd:.0%}).")
    if ann_vol > 0.25:
        weaknesses.append(f"High volatility ({ann_vol:.0%} annualized).")

    metrics = {
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "beta": beta,
        "diversification": diversification,
        "largest_position": largest,
    }
    return HealthReport(
        score=round(float(score), 1),
        metrics=metrics,
        strengths=strengths,
        weaknesses=weaknesses,
        correlation=rets.corr(),
    )
