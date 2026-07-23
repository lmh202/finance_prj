"""Daily Market Strategy Engine — Developer 2 (Architecture.md §5).

WORKING BASELINE: regime rules and the §5 asset-score formula on
cross-sectional percentiles. Refine thresholds, add RSI/correlation,
and build a real backtest. Signatures are frozen in src/interfaces.py.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

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


# ==========================================================================
# Indicator-confluence recommendation layer  (ADDITIVE — not the frozen §5
# score_assets / AssetSignal contract). Turns EMA20/50 + MACD(12,26,9) + RSI14
# into a weighted score and a raw BUY/SELL/HOLD, then cross-references the
# user's actual holdings into a final per-stock action (BUY/ADD/TRIM/SELL/HOLD).
# Consumes the shared kernel only (history, holdings, latest prices); it does
# NOT touch Developer 4's recommendation engine.
# ==========================================================================

# --- indicator params ---
EMA_FAST, EMA_SLOW = 20, 50
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
RSI_OVERSOLD, RSI_OVERBOUGHT = 30.0, 70.0

# --- scoring weights & thresholds ---
W_EMA, W_MACD, W_RSI = 1.5, 1.0, 1.0
BUY_THRESHOLD, SELL_THRESHOLD = 2.0, -2.0
TRIM_THRESHOLD = -1.0  # weakening zone: SELL_THRESHOLD < score <= TRIM_THRESHOLD

# Default candidate universe for NEW-position discovery: the top-20 US large-caps
# + gold/silver (matches the FNSPID top20_gold_silver universe the rest of the
# project is built on). Holdings are always scanned too; pass ?universe=SYM,SYM
# to the endpoint to override for a single request.
STRATEGY_WATCHLIST = [
    # mega-caps
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "COST", "NFLX",
    # other large-cap tech + staples
    "AMD", "ADBE", "QCOM", "TXN", "INTC", "MU", "INTU", "ASML", "CSCO", "PEP",
    # gold / silver
    "GLD", "SLV",
]

_ACTION_ORDER = {"SELL": 0, "TRIM": 1, "BUY": 2, "ADD": 3, "HOLD": 4}


@dataclass
class StrategyRecommendation:
    """One per-stock action from the confluence layer. Local to this engine —
    deliberately NOT interfaces.Recommendation (that is Developer 4's type)."""

    symbol: str
    recommendation: str                 # BUY | ADD | TRIM | SELL | HOLD
    held: bool
    raw_signal: str                     # BUY | SELL | HOLD (pre-portfolio)
    score: float
    quantity: float
    avg_buy_price: Optional[float]
    current_price: Optional[float]
    unrealized_pnl_pct: Optional[float]
    reasons: List[str]


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(s: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI (EWM with alpha=1/period)."""
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def macd(s: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW,
         signal: int = MACD_SIGNAL):
    """Return (macd line, signal line, histogram)."""
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def confluence_score(prices: pd.Series):
    """Weighted EMA/MACD/RSI vote -> (score, raw_signal, reasons, rsi_now).

    EMA20>EMA50 -> +W_EMA (else -W_EMA); a FRESH MACD crossover -> ±W_MACD;
    RSI<30 -> +W_RSI, RSI>70 -> -W_RSI. score>=BUY_THRESHOLD -> BUY,
    score<=SELL_THRESHOLD -> SELL, else HOLD.
    """
    p = prices.dropna()
    if len(p) < EMA_SLOW + MACD_SLOW:
        return 0.0, "HOLD", ["insufficient price history to score"], float("nan")

    e_fast, e_slow = ema(p, EMA_FAST), ema(p, EMA_SLOW)
    r = rsi(p)
    m_line, m_sig, _ = macd(p)

    score, reasons = 0.0, []
    if e_fast.iloc[-1] > e_slow.iloc[-1]:
        score += W_EMA
        reasons.append(f"EMA{EMA_FAST}>EMA{EMA_SLOW} uptrend (+{W_EMA})")
    else:
        score -= W_EMA
        reasons.append(f"EMA{EMA_FAST}<EMA{EMA_SLOW} downtrend (-{W_EMA})")

    crossed_up = m_line.iloc[-2] <= m_sig.iloc[-2] and m_line.iloc[-1] > m_sig.iloc[-1]
    crossed_dn = m_line.iloc[-2] >= m_sig.iloc[-2] and m_line.iloc[-1] < m_sig.iloc[-1]
    if crossed_up:
        score += W_MACD
        reasons.append(f"MACD crossed above signal (+{W_MACD})")
    elif crossed_dn:
        score -= W_MACD
        reasons.append(f"MACD crossed below signal (-{W_MACD})")
    else:
        reasons.append("no fresh MACD crossover")

    rv = float(r.iloc[-1])
    if rv < RSI_OVERSOLD:
        score += W_RSI
        reasons.append(f"RSI {rv:.0f} oversold (+{W_RSI})")
    elif rv > RSI_OVERBOUGHT:
        score -= W_RSI
        reasons.append(f"RSI {rv:.0f} overbought (-{W_RSI})")
    else:
        reasons.append(f"RSI {rv:.0f} neutral")

    raw = "BUY" if score >= BUY_THRESHOLD else ("SELL" if score <= SELL_THRESHOLD else "HOLD")
    return round(score, 2), raw, reasons, rv


def _final_action(held: bool, raw: str, score: float,
                  overbought: bool, profitable: bool):
    """Map (held/not-held + raw signal + score + P&L) -> final action + reason.
    SELL takes priority over TRIM."""
    if held:
        if raw == "SELL":
            return "SELL", "held and SELL signal -> full exit"
        if overbought and profitable:
            return "TRIM", "held, overbought and in profit -> take partial profit"
        if SELL_THRESHOLD < score <= TRIM_THRESHOLD:
            return "TRIM", "held and momentum weakening -> trim"
        if raw == "BUY":
            return "ADD", "held and still bullish -> add"
        return "HOLD", "held, no strong signal -> hold"
    if raw == "BUY":
        return "BUY", "not held and BUY signal -> new position"
    return "HOLD", "not held, no buy signal -> stay out"


def recommend_signals(
    history: pd.DataFrame,
    holdings: pd.DataFrame,
    prices: Optional[Dict[str, float]] = None,
    universe: Optional[List[str]] = None,
) -> List[StrategyRecommendation]:
    """Score every symbol in (universe ∪ holdings) and cross-reference what the
    user holds into a final action. `prices` supplies current price for the P&L
    check (a symbol may be missing — handled). Sorted most-actionable first."""
    prices = prices or {}
    has_holdings = not holdings.empty
    held = holdings.set_index("symbol") if has_holdings else holdings
    held_syms = set(held.index) if has_holdings else set()
    scan = list(dict.fromkeys(list(universe or []) + list(held_syms)))

    recs: List[StrategyRecommendation] = []
    for sym in scan:
        if sym not in history.columns:
            continue
        score, raw, reasons, rv = confluence_score(history[sym])
        is_held = sym in held_syms

        qty, avg = 0.0, None
        if is_held:
            qty = float(held.loc[sym, "shares"])
            a = held.loc[sym, "buy_price"]
            avg = float(a) if pd.notna(a) else None

        price = prices.get(sym)
        pnl = ((price / avg - 1) * 100) if (is_held and price and avg and avg > 0) else None
        profitable = pnl is not None and pnl > 0
        overbought = np.isfinite(rv) and rv > RSI_OVERBOUGHT

        action, why = _final_action(is_held, raw, score, overbought, profitable)
        recs.append(StrategyRecommendation(
            symbol=sym,
            recommendation=action,
            held=is_held,
            raw_signal=raw,
            score=score,
            quantity=qty,
            avg_buy_price=avg,
            current_price=price,
            unrealized_pnl_pct=(round(pnl, 2) if pnl is not None else None),
            reasons=reasons + [why],
        ))
    return sorted(recs, key=lambda r: (_ACTION_ORDER[r.recommendation], -r.score))
