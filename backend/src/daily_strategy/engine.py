"""Daily Market Strategy Engine — Developer 2 (Architecture.md §5).

WORKING BASELINE: regime rules and the §5 asset-score formula on
cross-sectional percentiles. Refine thresholds, add RSI/correlation,
and build a real backtest. Signatures are frozen in src/interfaces.py.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.interfaces import BENCHMARK, AssetSignal, RegimeState

TRADING_DAYS = 252


def classify_regime(history: pd.DataFrame, benchmark: str = BENCHMARK) -> RegimeState:
    """Classify the market into bullish / bearish / high_volatility / sideways."""
    if benchmark not in history.columns:
        return RegimeState("sideways", 0.0, {}, None)

    b = history[benchmark].dropna()
    if len(b) < 60:
        return RegimeState("sideways", 0.0, {}, None)

    price = float(b.iloc[-1])
    sma50 = float(b.rolling(50).mean().iloc[-1])
    sma200 = float(b.rolling(200).mean().iloc[-1]) if len(b) >= 200 else float("nan")
    mom20 = price / float(b.iloc[-21]) - 1 if len(b) > 21 else 0.0

    daily_vol = b.pct_change().rolling(20).std()
    vol20 = float(daily_vol.iloc[-1] * np.sqrt(TRADING_DAYS))
    vol_median = float(daily_vol.median() * np.sqrt(TRADING_DAYS))

    indicators = {
        "price": price,
        "sma50": sma50,
        "sma200": sma200,
        "momentum_20d": mom20,
        "volatility_20d": vol20,
        "volatility_median": vol_median,
    }

    above50 = price > sma50
    golden = sma50 > sma200 if not np.isnan(sma200) else above50
    high_vol = vol_median > 0 and vol20 > 1.5 * vol_median

    if high_vol:
        regime, hits = "high_volatility", 2 + int(abs(mom20) > 0.03)
    elif above50 and golden and mom20 > 0:
        regime, hits = "bullish", 3
    elif not above50 and not golden and mom20 < 0:
        regime, hits = "bearish", 3
    else:
        regime, hits = "sideways", 1

    return RegimeState(
        regime=regime,
        confidence=round(hits / 3, 2),
        indicators=indicators,
        as_of=b.index[-1].to_pydatetime(),
    )


def score_assets(
    history: pd.DataFrame,
    holdings: pd.DataFrame,
    sentiment: Optional[pd.DataFrame] = None,
) -> List[AssetSignal]:
    """Rank each held asset 0-100 per the §5 formula:
    30% momentum + 25% trend + 20% Sharpe − 15% volatility − 10% drawdown.

    `sentiment` is the OPTIONAL news feature channel (long-format frame from
    news_intelligence.sentiment_features). This baseline ignores it — the ML
    upgrade uses it as extra features, and the model must be evaluated with
    and without it (ablation). Missing days mean neutral (has_news=0).
    """
    symbols = [s for s in holdings["symbol"] if s in history.columns]
    rows = []
    for sym in symbols:
        p = history[sym].dropna()
        if len(p) < 60:
            continue
        price = float(p.iloc[-1])
        sma50 = float(p.rolling(50).mean().iloc[-1])
        sma200 = float(p.rolling(200).mean().iloc[-1]) if len(p) >= 200 else sma50
        rets = p.pct_change().dropna()
        year = rets.tail(TRADING_DAYS)
        vol = float(year.std() * np.sqrt(TRADING_DAYS))
        mean = float(year.mean() * TRADING_DAYS)
        peak = float(p.tail(TRADING_DAYS).max())
        rows.append(
            {
                "symbol": sym,
                "momentum": price / float(p.iloc[-61]) - 1 if len(p) > 61 else 0.0,
                "trend": (price > sma50) * 0.5 + (sma50 > sma200) * 0.5,
                "sharpe": mean / vol if vol > 0 else 0.0,
                "volatility": vol,
                "drawdown": price / peak - 1 if peak > 0 else 0.0,
            }
        )
    if not rows:
        return []

    df = pd.DataFrame(rows).set_index("symbol")
    # Cross-sectional percentile rank (0..1) per indicator
    r = df.rank(pct=True)
    score = 100 * (
        0.30 * r["momentum"]
        + 0.25 * r["trend"]
        + 0.20 * r["sharpe"]
        + 0.15 * (1 - r["volatility"])
        + 0.10 * r["drawdown"]  # less negative drawdown ranks higher
    )

    signals = []
    for sym in df.index:
        s = float(score[sym])
        action = "increase" if s >= 65 else ("reduce" if s <= 40 else "hold")
        ind = df.loc[sym].to_dict()
        signals.append(
            AssetSignal(
                symbol=sym,
                score=round(s, 1),
                action=action,
                indicators={k: round(float(v), 4) for k, v in ind.items()},
                rationale=(
                    f"{sym}: momentum {ind['momentum']:+.1%}, "
                    f"Sharpe {ind['sharpe']:.2f}, vol {ind['volatility']:.0%} "
                    f"→ {action}."
                ),
            )
        )
    return sorted(signals, key=lambda x: x.score, reverse=True)


def backtest(history: pd.DataFrame, holdings: pd.DataFrame, cash: float = 0.0) -> pd.DataFrame:
    """Growth of $1: buy-and-hold current weights vs equal weight.

    TODO (Developer 2): add the regime-aware daily strategy as a third column
    and turnover / transaction-cost accounting (Architecture.md page 6 specs).
    """
    symbols = [s for s in holdings["symbol"] if s in history.columns]
    if not symbols:
        return pd.DataFrame()

    rets = history[symbols].pct_change().dropna(how="all").fillna(0.0)
    last = history[symbols].ffill().iloc[-1]
    shares = holdings.set_index("symbol").loc[symbols, "shares"]
    w = (last * shares) / float((last * shares).sum())

    out = pd.DataFrame(index=rets.index)
    out["buy_hold"] = (1 + (rets * w).sum(axis=1)).cumprod()
    out["equal_weight"] = (1 + rets.mean(axis=1)).cumprod()
    return out
