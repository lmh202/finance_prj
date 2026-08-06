"""AURORA engine interfaces — the ONLY way the four workstreams talk to each other.

RULES
=====
1. This file is FROZEN. It changes only by team agreement (all four developers),
   never unilaterally — everyone's code depends on it.
2. Each developer implements the functions listed under their engine below, in
   their own folder, with EXACTLY these signatures and return types.
3. Consumers import a producer's public functions (src.<engine>.engine) and the
   types in this file — never a producer's internals.

SHARED INPUTS (produced by the shared kernel, read-only for everyone)
=====================================================================
holdings : pd.DataFrame                          src.portfolio.holdings_from_records()
    columns: symbol (str), name (str), shares (float), buy_price (float)
    supplied by the CLIENT in the request body — the backend persists no
    portfolio; the dataclasses and signatures below are unchanged
prices   : Dict[str, float]                      src.data_loader.get_latest_prices()
    latest close per symbol; a symbol may be MISSING — always handle that
history  : pd.DataFrame                          src.data_loader.get_history()
    daily adjusted close; index=DatetimeIndex, one column per symbol; a symbol's
    column may be missing — always handle that
weights  : Dict[str, float]                      derived from portfolio.build_view()
    current weight in percent per symbol (cash excluded)

REQUIRED FUNCTIONS PER ENGINE
=============================
Developer 1 — src/portfolio_health/engine.py
    compute_health(holdings, history, benchmark=BENCHMARK) -> HealthReport
    what_if_health(holdings, history, trades, benchmark=BENCHMARK) -> HealthReport
        trades: List[ProposedTrade] — the portfolio's health as if the
        proposed weight changes were applied (consumed by Developer 4's
        recommendation pages: "Health 68 -> 73 if you accept this").
    Also owns the Performance & Benchmark page (render_performance in
    page.py), presenting Developer 2's backtest + ablation results.

Developer 2 — src/daily_strategy/engine.py
    classify_regime(history, benchmark=BENCHMARK) -> RegimeState
    score_assets(history, holdings, sentiment=None) -> List[AssetSignal]
        sentiment: OPTIONAL long-format DataFrame produced by Developer 3's
        sentiment_features(). The ML model treats news as an optional
        feature channel (no news on a day == neutral + has_news=0) and must
        be evaluated both with and without it (the ablation).
    backtest(history, holdings, cash=0.0) -> pd.DataFrame
        columns: 'buy_hold', 'equal_weight', + strategy runs (rule-based,
        ML price-only, ML price+news). Rendered by Developer 1's page.

Developer 3 — src/news_intelligence/engine.py
    fetch_headlines(feeds=None, limit=50) -> List[NewsEvent]          # live RSS + LLM
    essential_news(holding_symbols, max_events=5) -> List[NewsEvent]  # live, <=5/day
    sentiment_features(symbols, start=None, end=None) -> pd.DataFrame # historical
        Long format, one row per symbol-day that has news, columns:
        date, symbol, sentiment (-1..1), news_count (int), has_news (0/1).
        NO LOOK-AHEAD: rows dated t may only use news published before
        day t's market open. Historical corpus scored with a LOCAL model
        (FinBERT/VADER); the LLM API is reserved for the live feed.

Developer 4 — src/recommendation/engine.py
    reaction_risk(event, weights, regime) -> ReactionRisk
        Owns the double-counting reconciliation: news already reflected in
        Developer 2's sentiment-tilted scores raises the 'priced_in' factor.
    recommend_daily(regime, signals, weights) -> Recommendation
    recommend_event(event, risk) -> Recommendation
    apply_constraints(trades, weights) -> List[ProposedTrade]
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# ------------------------------------------------------------------ constants

BENCHMARK = "SPY"

REGIMES = ("bullish", "bearish", "high_volatility", "sideways")

ACTIONS = ("increase", "hold", "reduce")

# Architecture.md §3 — event taxonomy for news classification
EVENT_CATEGORIES = (
    "interest_rates",
    "inflation",
    "economic_growth",
    "geopolitical",
    "energy_commodities",
    "technology_regulation",
    "corporate_earnings",
    "financial_system",
)

# Architecture.md §9 — safety rules applied before any recommendation is shown
CONSTRAINTS = {
    "max_stock_weight_pct": 20.0,
    "max_sector_weight_pct": 35.0,
    "max_commodity_weight_pct": 25.0,
    "max_weight_change_pct": 5.0,
    "min_holdings": 5,
    "min_trade_pct": 1.0,
}

# ---------------------------------------------------- Developer 1: health

@dataclass
class HealthReport:
    """Produced by portfolio_health.compute_health; consumed by the Health page
    and by recommendation.recommend_daily."""

    score: float                        # 0–100 (Architecture.md §4 blend)
    metrics: Dict[str, float]           # annual_return, annual_volatility, sharpe,
                                        # sortino, max_drawdown, beta,
                                        # diversification, largest_position
    strengths: List[str]
    weaknesses: List[str]
    correlation: Optional[pd.DataFrame] = None   # symbols × symbols


# -------------------------------------------------- Developer 2: strategy

@dataclass
class RegimeState:
    """Produced by daily_strategy.classify_regime; consumed by the Strategy
    page, reaction_risk and recommend_daily."""

    regime: str                         # one of REGIMES
    confidence: float                   # 0–1, how clearly the rules fired
    indicators: Dict[str, float]        # price_vs_sma50, sma50_vs_sma200,
                                        # momentum_20d, volatility_20d, ...
    as_of: Optional[datetime] = None


@dataclass
class AssetSignal:
    """One row of the daily asset ranking (Architecture.md §5 score formula)."""

    symbol: str
    score: float                        # 0–100
    action: str                         # one of ACTIONS
    indicators: Dict[str, float]        # momentum, trend, sharpe, volatility, drawdown
    rationale: str                      # one human-readable sentence


# ------------------------------------------------------ Developer 3: news

@dataclass
class NewsEvent:
    """One filtered, classified news story (Architecture.md §6)."""

    title: str
    source: str
    url: str
    published: Optional[datetime]
    category: str                       # one of EVENT_CATEGORIES
    sentiment: float                    # -1 (very negative) … +1 (very positive)
    importance: float                   # 0–100 essential-news score
    affected_symbols: List[str] = field(default_factory=list)
    impact: Dict[str, str] = field(default_factory=dict)  # symbol -> positive|negative|mixed
    summary: str = ""


# -------------------------------------- Developer 4: risk & recommendation

@dataclass
class ReactionRisk:
    """How risky it is to trade on an event (Architecture.md §7 formula)."""

    risk_pct: float                     # 0–100
    factors: Dict[str, float]           # news_uncertainty, technical_disagreement,
                                        # market_volatility, priced_in,
                                        # concentration, ambiguity — each 0–1
    reasons: List[str]
    suggestion: str                     # "do_nothing" | "moderate" | "aggressive"


@dataclass
class ProposedTrade:
    symbol: str
    weight_change_pct: float            # +2.0 = increase weight by 2 points
    reason: str


@dataclass
class Recommendation:
    """Final output shown to the user (Architecture.md §8). Never auto-executed."""

    kind: str                           # "daily" | "event"
    trades: List[ProposedTrade]
    confidence: float                   # 0–1
    explanation: str
    reaction_risk: Optional[ReactionRisk] = None
