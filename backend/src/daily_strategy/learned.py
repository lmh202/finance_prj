"""Optional XGBoost residual-alpha enhancement for Daily Strategy.

The artifact is inert unless its metadata records ``promotion_status`` as
``promoted``. Risk and Portfolio Health are intentionally absent: the model
only adjusts the cross-sectional direction rank, while the recommendation
layer controls positions with formal HAR-X + News risk.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.config import DATA_DIR
from src.interfaces import AssetSignal

MODEL_DIR = DATA_DIR / "processed" / "daily_strategy_model_candidate"
METADATA_PATH = MODEL_DIR / "metadata.json"
MODEL_PATH = MODEL_DIR / "xgb_residual_alpha.json"

BASE_PRICE_FEATURES = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "mom_20d",
    "mom_60d",
    "momentum_acceleration",
    "price_vs_sma50",
    "sma50_vs_sma200",
    "trend_strength",
    "vol_5d",
    "vol_20d",
    "vol_60d",
    "vol_ratio_5_20",
    "vol_ratio_20_60",
    "downside_vol_20d",
    "skew_20d",
    "rsi_scaled",
    "drawdown",
    "risk_adj_mom",
    "beta_60d",
    "rel_str_20d",
    "strategy_sharpe",
    "strategy_drawdown",
    "strategy_score",
]
RANK_SOURCES = [
    "ret_5d",
    "mom_20d",
    "mom_60d",
    "momentum_acceleration",
    "trend_strength",
    "vol_20d",
    "downside_vol_20d",
    "rsi_scaled",
    "drawdown",
    "risk_adj_mom",
    "rel_str_20d",
]
RANK_FEATURES = [f"rank_{column}" for column in RANK_SOURCES]
MARKET_FEATURES = [
    "benchmark_ret_1d",
    "market_ret_5d",
    "market_ret_20d",
    "market_ret_60d",
    "market_vol_20d",
    "market_vol_60d",
    "market_drawdown",
]
FEATURES = BASE_PRICE_FEATURES + RANK_FEATURES + MARKET_FEATURES

_CACHE: tuple[int, dict, XGBRegressor] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_artifact() -> tuple[Optional[dict], Optional[XGBRegressor]]:
    global _CACHE
    if not METADATA_PATH.exists() or not MODEL_PATH.exists():
        return None, None
    modified = max(
        METADATA_PATH.stat().st_mtime_ns,
        MODEL_PATH.stat().st_mtime_ns,
    )
    if _CACHE is None or _CACHE[0] != modified:
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            if metadata.get("model_sha256") != _sha256(MODEL_PATH):
                return None, None
            model = XGBRegressor()
            model.load_model(MODEL_PATH)
            _CACHE = (modified, metadata, model)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None, None
    return _CACHE[1], _CACHE[2]


def model_available(*, require_promoted: bool = True) -> bool:
    metadata, model = _load_artifact()
    if metadata is None or model is None:
        return False
    return (
        not require_promoted
        or metadata.get("promotion_status") == "promoted"
    )


def _compound_return(values: pd.Series, window: int) -> pd.Series:
    return (
        np.exp(
            np.log1p(values.clip(lower=-0.999999))
            .rolling(window, min_periods=window)
            .sum()
        )
        - 1.0
    )


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    relative = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + relative)


def latest_feature_frame(
    history: pd.DataFrame,
    symbols: Iterable[str],
    *,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Build the latest feature row in the recorded training order."""
    history = history.sort_index()
    if benchmark not in history:
        return pd.DataFrame(columns=FEATURES)
    benchmark_close = history[benchmark].dropna()
    if len(benchmark_close) < 126:
        return pd.DataFrame(columns=FEATURES)
    benchmark_return = benchmark_close.pct_change()
    market_index = (1.0 + benchmark_return.fillna(0.0)).cumprod()
    market = {
        "benchmark_ret_1d": float(benchmark_return.iloc[-1]),
        "market_ret_5d": float(
            benchmark_close.iloc[-1] / benchmark_close.iloc[-6] - 1.0
        ),
        "market_ret_20d": float(
            benchmark_close.iloc[-1] / benchmark_close.iloc[-21] - 1.0
        ),
        "market_ret_60d": float(
            benchmark_close.iloc[-1] / benchmark_close.iloc[-61] - 1.0
        ),
        "market_vol_20d": float(
            benchmark_return.tail(20).std() * math.sqrt(252)
        ),
        "market_vol_60d": float(
            benchmark_return.tail(60).std() * math.sqrt(252)
        ),
        "market_drawdown": float(
            market_index.iloc[-1] / market_index.cummax().iloc[-1] - 1.0
        ),
    }
    rows = []
    for symbol in symbols:
        if symbol == benchmark or symbol not in history:
            continue
        close = history[symbol].dropna()
        if len(close) < 220:
            continue
        returns = close.pct_change()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        vol5 = returns.rolling(5).std() * math.sqrt(252)
        vol20 = returns.rolling(20).std() * math.sqrt(252)
        vol60 = returns.rolling(60).std() * math.sqrt(252)
        momentum20 = close.pct_change(20)
        momentum60 = close.pct_change(60)
        aligned_benchmark = benchmark_return.reindex(close.index)
        beta = (
            returns.rolling(60).cov(aligned_benchmark)
            / aligned_benchmark.rolling(60).var()
        )
        year = returns.tail(252)
        annual_volatility = float(year.std() * math.sqrt(252))
        strategy_sharpe = (
            float(year.mean() * 252 / annual_volatility)
            if annual_volatility > 0
            else 0.0
        )
        strategy_trend = (
            0.5 * float(close.iloc[-1] > sma50.iloc[-1])
            + 0.5 * float(sma50.iloc[-1] > sma200.iloc[-1])
        )
        strategy_drawdown = float(
            close.iloc[-1] / close.tail(252).max() - 1.0
        )
        row = {
            "symbol": symbol,
            "ret_1d": float(returns.iloc[-1]),
            "ret_5d": float(close.iloc[-1] / close.iloc[-6] - 1.0),
            "ret_10d": float(close.iloc[-1] / close.iloc[-11] - 1.0),
            "mom_20d": float(momentum20.iloc[-1]),
            "mom_60d": float(momentum60.iloc[-1]),
            "momentum_acceleration": float(
                momentum20.iloc[-1] - momentum60.iloc[-1] / 3.0
            ),
            "price_vs_sma50": float(
                close.iloc[-1] / sma50.iloc[-1] - 1.0
            ),
            "sma50_vs_sma200": float(
                sma50.iloc[-1] / sma200.iloc[-1] - 1.0
            ),
            "trend_strength": float(
                close.iloc[-1] / sma50.iloc[-1] - 1.0
                + sma50.iloc[-1] / sma200.iloc[-1]
                - 1.0
            ),
            "vol_5d": float(vol5.iloc[-1]),
            "vol_20d": float(vol20.iloc[-1]),
            "vol_60d": float(vol60.iloc[-1]),
            "vol_ratio_5_20": float(vol5.iloc[-1] / vol20.iloc[-1]),
            "vol_ratio_20_60": float(vol20.iloc[-1] / vol60.iloc[-1]),
            "downside_vol_20d": float(
                returns.clip(upper=0.0).tail(20).std()
                * math.sqrt(252)
            ),
            "skew_20d": float(returns.tail(20).skew()),
            "rsi_scaled": float((_rsi(close).iloc[-1] - 50.0) / 50.0),
            "drawdown": float(
                close.iloc[-1] / close.cummax().iloc[-1] - 1.0
            ),
            "risk_adj_mom": float(
                momentum20.iloc[-1] / vol20.iloc[-1]
            ),
            "beta_60d": float(beta.iloc[-1]),
            "rel_str_20d": float(
                momentum20.iloc[-1]
                - benchmark_close.pct_change(20).iloc[-1]
            ),
            "strategy_sharpe": strategy_sharpe,
            "strategy_drawdown": strategy_drawdown,
            "_strategy_trend": strategy_trend,
            **market,
        }
        rows.append(row)
    if len(rows) < 5:
        return pd.DataFrame(columns=FEATURES)
    frame = pd.DataFrame(rows).set_index("symbol")
    strategy_ranks = frame[
        ["mom_60d", "_strategy_trend", "strategy_sharpe", "strategy_drawdown"]
    ].rank(pct=True)
    direction_01 = (
        0.30 * strategy_ranks["mom_60d"]
        + 0.25 * strategy_ranks["_strategy_trend"]
        + 0.20 * strategy_ranks["strategy_sharpe"]
        + 0.10 * strategy_ranks["strategy_drawdown"]
    ) / 0.85
    frame["strategy_score"] = (
        2.0 * direction_01 - 1.0
    ).clip(-1.0, 1.0)
    for source in RANK_SOURCES:
        frame[f"rank_{source}"] = (
            2.0 * frame[source].rank(pct=True) - 1.0
        )
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame.reindex(columns=FEATURES).dropna()


def enhance_signals(
    history: pd.DataFrame,
    signals: Iterable[AssetSignal],
    *,
    require_promoted: bool = True,
) -> list[AssetSignal]:
    """Apply the candidate to existing signals, preserving exact fallback."""
    metadata, model = _load_artifact()
    original = list(signals)
    if metadata is None or model is None:
        return original
    if (
        require_promoted
        and metadata.get("promotion_status") != "promoted"
    ):
        return original
    signal_map = {signal.symbol: signal for signal in original}
    frame = latest_feature_frame(history, signal_map)
    if frame.empty:
        return original
    for feature in metadata["feature_order"]:
        lower, upper = metadata["feature_bounds"][feature]
        frame[feature] = frame[feature].clip(float(lower), float(upper))
    residual = np.asarray(
        model.predict(frame[metadata["feature_order"]]),
        dtype=float,
    )
    raw = (
        frame["strategy_score"].to_numpy(dtype=float)
        + float(metadata["eta"])
        * np.clip(
            residual,
            -float(metadata["residual_cap"]),
            float(metadata["residual_cap"]),
        )
    )
    enhanced = (
        2.0 * pd.Series(raw, index=frame.index).rank(pct=True) - 1.0
    )
    output = []
    for signal in original:
        if signal.symbol not in frame.index:
            output.append(signal)
            continue
        direction = float(enhanced.loc[signal.symbol])
        score = float(np.clip(50.0 + 50.0 * direction, 0.0, 100.0))
        action = "increase" if score >= 65 else (
            "reduce" if score <= 40 else "hold"
        )
        indicators = dict(signal.indicators)
        indicators.update(
            {
                "base_strategy_direction": float(
                    frame.at[signal.symbol, "strategy_score"]
                ),
                "xgb_residual_alpha": float(
                    residual[frame.index.get_loc(signal.symbol)]
                ),
                "enhanced_direction": direction,
            }
        )
        output.append(
            AssetSignal(
                symbol=signal.symbol,
                score=round(score, 1),
                action=action,
                indicators=indicators,
                rationale=(
                    f"{signal.rationale} XGBoost residual-alpha rank "
                    f"adjustment produced {direction:+.2f}."
                ),
            )
        )
    return sorted(output, key=lambda item: item.score, reverse=True)
