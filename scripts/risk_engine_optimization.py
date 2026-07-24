"""Core utilities for leakage-safe volatility model optimisation.

This module contains only reusable data, feature, split, model, and metric
primitives.  The experiment orchestration and artifact writing live in
``optimize_risk_engine.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TweedieRegressor

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "FNSPID" / "final_dataset" / "prices"
NEWS_PATH = ROOT / "data" / "processed" / "extended_news_features.parquet"
NEWS_SPEC_PATH = ROOT / "data" / "processed" / "extended_news_feature_spec.json"
CURRENT_MODEL_PATH = ROOT / "data" / "processed" / "risk_model.json"
CURRENT_OHLC_PATH = (
    ROOT / "data" / "processed" / "current_risk_test" / "ohlc.parquet"
)

HORIZONS = (5, 20)
EPS = 1e-8
RANDOM_SEED = 20260723
OUTER_YEARS = tuple(range(2018, 2024))

HAR_FEATURES = [
    "l_rv5",
    "l_rv22",
    "l_rv66",
    "l_park5",
    "l_park22",
    "l_absret",
]

EXTRA_PRICE_FEATURES = [
    "l_gk5",
    "l_gk22",
    "l_rs5",
    "l_rs22",
    "l_gapvol5",
    "l_gapvol22",
    "l_downsemi5",
    "l_downsemi22",
    "l_upsemi5",
    "l_upsemi22",
    "signed_ret1",
    "negative_return",
    "max_absret5",
    "max_absret22",
    "ret_skew22",
    "ret_kurt22",
    "drawdown",
    "log_volume",
    "volume_z22",
    "l_amihud22",
    "mkt_ret1",
    "mkt_l_rv5",
    "mkt_l_rv22",
    "mkt_l_rv66",
    "mkt_drawdown",
    "cross_median_l_rv22",
    "cross_ret_dispersion",
    "cross_median_absret",
]
PRICE_FEATURES = HAR_FEATURES + EXTRA_PRICE_FEATURES

CAUSAL_NEWS_STATES = [
    "news_state_uncovered",
    "news_state_silent",
    "news_state_stale",
]

DEPLOYABLE_NEWS_BASE = [
    "has_news",
    "log_count",
    "count_z20",
    "count_ratio20",
    "news_count_3d",
    "news_count_5d",
    "days_since_news",
    "sent_mean",
    "sent_std",
    "sent_range",
    "sent_abs_mean",
    "sent_positive_share",
    "sent_negative_share",
    "sent_extreme_share",
    "sent_surprise20",
    "sent_abs_surprise20",
    "unique_story_count",
    "duplicate_share",
    "unique_publisher_count",
    "publisher_entropy",
    "publisher_missing_share",
    "title_token_novelty",
    "event_earnings_share",
    "event_analyst_share",
    "event_corporate_action_share",
    "event_legal_regulatory_share",
    "event_product_share",
    "event_macro_share",
    "event_management_share",
    "event_financing_share",
]


@dataclass(frozen=True)
class TimeFold:
    test_year: int
    horizon: int
    train_mask: np.ndarray
    test_mask: np.ndarray
    train_cutoff: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class HarModel:
    features: list[str]
    coefficient: np.ndarray
    intercept: float
    smearing: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        log_sigma = (
            self.intercept
            + frame[self.features].to_numpy(dtype=float) @ self.coefficient
        )
        return np.exp(log_sigma) * self.smearing

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "linear_har",
            "features": self.features,
            "coef": {
                feature: float(value)
                for feature, value in zip(self.features, self.coefficient)
            },
            "intercept": float(self.intercept),
            "smearing": float(self.smearing),
        }


@dataclass
class GammaOverlay:
    features: list[str]
    alpha: float
    scale: dict[str, float]
    coefficient: np.ndarray
    intercept: float
    maximum_sigma_multiplier: float = 2.0

    def predict_multiplier(self, frame: pd.DataFrame) -> np.ndarray:
        scale = pd.Series(self.scale)
        values = frame[self.features].fillna(0.0).div(scale).to_numpy()
        log_variance_ratio = self.intercept + values @ self.coefficient
        limit = 2.0 * math.log(self.maximum_sigma_multiplier)
        return np.exp(0.5 * np.clip(log_variance_ratio, -limit, limit))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "linear_gamma_variance_ratio",
            "features": self.features,
            "alpha": float(self.alpha),
            "scale": self.scale,
            "coef": {
                feature: float(value)
                for feature, value in zip(self.features, self.coefficient)
            },
            "intercept": float(self.intercept),
            "max_sigma_multiplier": float(self.maximum_sigma_multiplier),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_log(value: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.log(np.maximum(value, EPS))


def _rolling_semivol(return_series: pd.Series, window: int, positive: bool) -> pd.Series:
    selected = return_series.where(
        return_series.gt(0) if positive else return_series.lt(0),
        0.0,
    )
    return np.sqrt(selected.pow(2).rolling(window).mean())


def _ewma_sigma(return_series: pd.Series, decay: float = 0.94) -> pd.Series:
    variance = return_series.pow(2).ewm(alpha=1 - decay, adjust=False).mean()
    return np.sqrt(variance)


def engineer_symbol_frame(
    prices: pd.DataFrame,
    symbol: str,
    horizons: Sequence[int] = HORIZONS,
) -> pd.DataFrame:
    """Create causal OHLC features and forward realised-volatility targets."""
    frame = prices.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.sort_values("date").drop_duplicates("date").set_index("date")

    close_column = "adj close" if "adj close" in frame.columns else "close"
    adjusted_close = frame[close_column].astype(float)
    raw_close = frame["close"].astype(float)
    adjustment = adjusted_close / raw_close.replace(0, np.nan)
    open_price = frame["open"].astype(float) * adjustment
    high = frame["high"].astype(float) * adjustment
    low = frame["low"].astype(float) * adjustment
    volume = frame["volume"].astype(float).clip(lower=0)

    ret = adjusted_close.pct_change()
    log_hl = np.log(high / low.replace(0, np.nan))
    log_co = np.log(adjusted_close / open_price.replace(0, np.nan))
    log_ho = np.log(high / open_price.replace(0, np.nan))
    log_hc = np.log(high / adjusted_close.replace(0, np.nan))
    log_lo = np.log(low / open_price.replace(0, np.nan))
    log_lc = np.log(low / adjusted_close.replace(0, np.nan))
    gap = np.log(open_price / adjusted_close.shift(1).replace(0, np.nan))

    parkinson_variance = log_hl.pow(2) / (4 * np.log(2))
    gk_variance = (
        0.5 * log_hl.pow(2)
        - (2 * np.log(2) - 1) * log_co.pow(2)
    ).clip(lower=0)
    rs_variance = (
        log_ho * log_hc + log_lo * log_lc
    ).clip(lower=0)

    result = pd.DataFrame(index=frame.index)
    result["ret1"] = ret
    result["signed_ret1"] = ret
    result["negative_return"] = ret.lt(0).astype(float)
    result["absret"] = ret.abs()
    result["rv5"] = ret.rolling(5).std()
    result["rv22"] = ret.rolling(22).std()
    result["rv66"] = ret.rolling(66).std()
    result["park5"] = np.sqrt(parkinson_variance.rolling(5).mean())
    result["park22"] = np.sqrt(parkinson_variance.rolling(22).mean())
    result["gk5"] = np.sqrt(gk_variance.rolling(5).mean())
    result["gk22"] = np.sqrt(gk_variance.rolling(22).mean())
    result["rs5"] = np.sqrt(rs_variance.rolling(5).mean())
    result["rs22"] = np.sqrt(rs_variance.rolling(22).mean())
    result["gapvol5"] = gap.rolling(5).std()
    result["gapvol22"] = gap.rolling(22).std()
    result["downsemi5"] = _rolling_semivol(ret, 5, positive=False)
    result["downsemi22"] = _rolling_semivol(ret, 22, positive=False)
    result["upsemi5"] = _rolling_semivol(ret, 5, positive=True)
    result["upsemi22"] = _rolling_semivol(ret, 22, positive=True)
    result["max_absret5"] = ret.abs().rolling(5).max()
    result["max_absret22"] = ret.abs().rolling(22).max()
    result["ret_skew22"] = ret.rolling(22).skew()
    result["ret_kurt22"] = ret.rolling(22).kurt()
    result["drawdown"] = adjusted_close / adjusted_close.cummax() - 1.0
    result["log_volume"] = np.log1p(volume)
    volume_mean = result["log_volume"].rolling(22).mean()
    volume_std = result["log_volume"].rolling(22).std()
    result["volume_z22"] = (
        (result["log_volume"] - volume_mean) / (volume_std + EPS)
    )
    dollar_volume = adjusted_close * volume
    amihud = ret.abs() / dollar_volume.replace(0, np.nan)
    result["amihud22"] = amihud.rolling(22).mean()
    result["ewma_sigma"] = _ewma_sigma(ret)

    log_columns = {
        "rv5": "l_rv5",
        "rv22": "l_rv22",
        "rv66": "l_rv66",
        "park5": "l_park5",
        "park22": "l_park22",
        "absret": "l_absret",
        "gk5": "l_gk5",
        "gk22": "l_gk22",
        "rs5": "l_rs5",
        "rs22": "l_rs22",
        "gapvol5": "l_gapvol5",
        "gapvol22": "l_gapvol22",
        "downsemi5": "l_downsemi5",
        "downsemi22": "l_downsemi22",
        "upsemi5": "l_upsemi5",
        "upsemi22": "l_upsemi22",
        "amihud22": "l_amihud22",
    }
    for raw_name, log_name in log_columns.items():
        result[log_name] = _safe_log(result[raw_name])

    for horizon in horizons:
        realised = ret.rolling(horizon).std().shift(-horizon)
        result[f"realized_vol_{horizon}d"] = realised
        result[f"target_variance_{horizon}d"] = realised.pow(2)
        result[f"fwd_return_{horizon}d"] = (
            adjusted_close.shift(-horizon) / adjusted_close - 1.0
        )

    result["symbol"] = symbol
    result.index.name = "date"
    return result.reset_index()


def _market_features(spy: pd.DataFrame) -> pd.DataFrame:
    keep = ["date", "ret1", "l_rv5", "l_rv22", "l_rv66", "drawdown"]
    market = spy[keep].copy()
    return market.rename(
        columns={
            "ret1": "mkt_ret1",
            "l_rv5": "mkt_l_rv5",
            "l_rv22": "mkt_l_rv22",
            "l_rv66": "mkt_l_rv66",
            "drawdown": "mkt_drawdown",
        }
    )


def _add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    grouped = panel.groupby("date", sort=False)
    panel["cross_median_l_rv22"] = grouped["l_rv22"].transform("median")
    panel["cross_ret_dispersion"] = grouped["ret1"].transform("std")
    panel["cross_median_absret"] = grouped["absret"].transform("median")
    return panel


def _news_feature_groups(spec: Mapping[str, object]) -> dict[str, list[str]]:
    families = spec["families"]
    return {
        "attention": list(families["attention"]),
        "sentiment": list(families["sentiment_distribution"]),
        "source_diffusion_novelty": list(
            families["source_diffusion_novelty"]
        ),
        "events": list(families["event_categories"]),
    }


def add_causal_news_states(panel: pd.DataFrame) -> pd.DataFrame:
    """Replace two-sided coverage metadata with states known at time t."""
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    has_news = panel["has_news"].fillna(0).astype(int)
    coverage_started = has_news.groupby(panel["symbol"], sort=False).cummax()
    recent_news = (
        has_news.groupby(panel["symbol"], sort=False)
        .rolling(60, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
        .reindex(panel.index)
        .fillna(0)
        .astype(int)
    )
    no_news = has_news.eq(0)
    panel["news_state_uncovered"] = (
        no_news & coverage_started.eq(0)
    ).astype(int)
    panel["news_state_silent"] = (
        no_news & recent_news.eq(1)
    ).astype(int)
    panel["news_state_stale"] = (
        no_news & coverage_started.eq(1) & recent_news.eq(0)
    ).astype(int)
    panel["news_quality_available"] = coverage_started.astype(int)
    state_total = panel[CAUSAL_NEWS_STATES].sum(axis=1) + has_news
    if not state_total.eq(1).all():
        raise AssertionError("causal news states are not mutually exhaustive")
    return panel


def build_historical_panel() -> tuple[
    pd.DataFrame,
    dict[str, list[str]],
    list[str],
    list[str],
]:
    """Build the 21-symbol OHLC/news panel used by the optimiser."""
    symbols = sorted(
        path.stem for path in PRICES.glob("*.csv") if path.stem != "SPY"
    )
    if len(symbols) != 21:
        raise ValueError(f"expected 21 research symbols, found {len(symbols)}")

    spy_raw = pd.read_csv(PRICES / "SPY.csv")
    spy = engineer_symbol_frame(spy_raw, "SPY")
    market = _market_features(spy)
    parts = []
    for symbol in symbols:
        part = engineer_symbol_frame(pd.read_csv(PRICES / f"{symbol}.csv"), symbol)
        parts.append(part.merge(market, on="date", how="left"))
    panel = _add_cross_sectional_features(pd.concat(parts, ignore_index=True))

    news = pd.read_parquet(NEWS_PATH)
    news["date"] = pd.to_datetime(news["date"]).dt.normalize()
    news = news.drop(columns=["coverage_active"], errors="ignore")
    spec = json.loads(NEWS_SPEC_PATH.read_text(encoding="utf-8"))
    groups = _news_feature_groups(spec)
    historical_news = list(
        dict.fromkeys(
            column
            for columns in groups.values()
            for column in columns
        )
    )
    extra_distribution = ["sent_min", "sent_max"]
    historical_news = list(dict.fromkeys(historical_news + extra_distribution))
    news_columns = [
        column for column in historical_news if column in news.columns
    ]
    contract = ["date", "symbol"] + news_columns
    panel = panel.merge(
        news[contract],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    panel[news_columns] = panel[news_columns].fillna(0.0)
    if "has_news" not in panel.columns:
        raise ValueError("extended news data is missing has_news")
    panel = add_causal_news_states(panel)

    research_news = list(dict.fromkeys(news_columns + CAUSAL_NEWS_STATES))
    deployable_news = [
        feature
        for feature in DEPLOYABLE_NEWS_BASE + CAUSAL_NEWS_STATES
        if feature in panel.columns
    ]
    groups = {
        name: [feature for feature in features if feature in panel.columns]
        for name, features in groups.items()
    }
    groups["causal_state"] = CAUSAL_NEWS_STATES.copy()

    required = PRICE_FEATURES + research_news
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=PRICE_FEATURES).copy()
    panel[research_news] = panel[research_news].fillna(0.0)
    if panel[required].isna().any().any():
        missing = panel[required].isna().sum()
        raise ValueError(
            f"panel contains missing features: {missing[missing.gt(0)].to_dict()}"
        )
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel, groups, research_news, deployable_news


def build_current_panel() -> pd.DataFrame:
    """Build the same price feature schema for the 2024+ 40-symbol panel."""
    if not CURRENT_OHLC_PATH.exists():
        raise FileNotFoundError(CURRENT_OHLC_PATH)
    raw = pd.read_parquet(CURRENT_OHLC_PATH)
    raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
    symbols = sorted(raw["symbol"].unique())
    if "SPY" in symbols:
        spy = engineer_symbol_frame(raw[raw["symbol"] == "SPY"], "SPY")
    else:
        spy = engineer_symbol_frame(pd.read_csv(PRICES / "SPY.csv"), "SPY")
    market = _market_features(spy)
    parts = []
    for symbol in symbols:
        part = engineer_symbol_frame(raw[raw["symbol"] == symbol], symbol)
        parts.append(part.merge(market, on="date", how="left"))
    panel = _add_cross_sectional_features(pd.concat(parts, ignore_index=True))
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=PRICE_FEATURES).copy()
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def time_fold(frame: pd.DataFrame, test_year: int, horizon: int) -> TimeFold:
    test_mask = frame["date"].dt.year.eq(test_year).to_numpy()
    if not test_mask.any():
        raise ValueError(f"no rows for test year {test_year}")
    dates = np.sort(frame["date"].unique())
    test_start = frame.loc[test_mask, "date"].min()
    first_position = int(np.searchsorted(dates, np.datetime64(test_start)))
    cutoff_position = first_position - horizon - 1
    if cutoff_position < 0:
        raise ValueError("insufficient history for embargo")
    cutoff = pd.Timestamp(dates[cutoff_position])
    train_mask = frame["date"].le(cutoff).to_numpy()
    return TimeFold(
        test_year=test_year,
        horizon=horizon,
        train_mask=train_mask,
        test_mask=test_mask,
        train_cutoff=cutoff,
        test_start=test_start,
        test_end=frame.loc[test_mask, "date"].max(),
    )


def inner_folds(
    frame: pd.DataFrame,
    outer_year: int,
    horizon: int,
    first_inner_year: int = 2016,
) -> list[TimeFold]:
    return [
        time_fold(frame, year, horizon)
        for year in range(first_inner_year, outer_year)
        if frame["date"].dt.year.eq(year).any()
    ]


def symbol_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("symbol")["symbol"].transform("size").to_numpy()
    weights = 1.0 / counts
    return weights / weights.mean()


def qlike_loss(realized_vol: np.ndarray, forecast_vol: np.ndarray) -> np.ndarray:
    realized_variance = np.maximum(np.asarray(realized_vol, float) ** 2, EPS)
    forecast_variance = np.maximum(np.asarray(forecast_vol, float) ** 2, EPS)
    ratio = realized_variance / forecast_variance
    return ratio - np.log(ratio) - 1.0


def symbol_equal_qlike(
    frame: pd.DataFrame,
    realized_column: str,
    forecast: np.ndarray,
) -> float:
    loss = qlike_loss(frame[realized_column].to_numpy(), forecast)
    scored = pd.DataFrame({"symbol": frame["symbol"].to_numpy(), "loss": loss})
    return float(scored.groupby("symbol")["loss"].mean().mean())


def fit_har(frame: pd.DataFrame, horizon: int) -> HarModel:
    target = f"realized_vol_{horizon}d"
    train = frame.dropna(subset=HAR_FEATURES + [target])
    log_target = np.log(np.maximum(train[target].to_numpy(), EPS))
    weights = symbol_equal_weights(train)
    model = LinearRegression().fit(
        train[HAR_FEATURES],
        log_target,
        sample_weight=weights,
    )
    residual = log_target - model.predict(train[HAR_FEATURES])
    smearing = float(
        np.average(np.exp(residual), weights=weights)
    )
    return HarModel(
        features=HAR_FEATURES.copy(),
        coefficient=np.asarray(model.coef_, dtype=float),
        intercept=float(model.intercept_),
        smearing=smearing,
    )


def fit_gamma_overlay(
    frame: pd.DataFrame,
    base_forecast: np.ndarray,
    horizon: int,
    features: list[str],
    alpha: float,
) -> GammaOverlay:
    target_column = f"realized_vol_{horizon}d"
    valid = frame[target_column].notna().to_numpy()
    train = frame.loc[valid].copy()
    base = np.asarray(base_forecast, float)[valid]
    scale_series = (
        train[features].std(ddof=0).replace(0, 1.0).fillna(1.0)
    )
    values = train[features].fillna(0.0).div(scale_series)
    variance_ratio = np.clip(
        (train[target_column].to_numpy() / np.maximum(base, EPS)) ** 2,
        1e-4,
        1e4,
    )
    model = TweedieRegressor(
        power=2,
        link="log",
        alpha=alpha,
        fit_intercept=False,
        max_iter=1000,
        tol=1e-7,
    )
    model.fit(
        values,
        variance_ratio,
        sample_weight=symbol_equal_weights(train),
    )
    return GammaOverlay(
        features=features.copy(),
        alpha=alpha,
        scale={key: float(value) for key, value in scale_series.items()},
        coefficient=np.asarray(model.coef_, dtype=float),
        intercept=float(model.intercept_),
    )


def overlay_forecast(
    base_forecast: np.ndarray,
    overlay: GammaOverlay,
    frame: pd.DataFrame,
) -> np.ndarray:
    multiplier = overlay.predict_multiplier(frame)
    available = frame["news_quality_available"].eq(1).to_numpy()
    return np.asarray(base_forecast, float) * np.where(
        available, multiplier, 1.0
    )


def dm_test(
    dates: pd.Series,
    candidate_loss: np.ndarray,
    reference_loss: np.ndarray,
    horizon: int,
) -> tuple[float, float]:
    daily = (
        pd.DataFrame(
            {
                "date": dates.to_numpy(),
                "difference": candidate_loss - reference_loss,
            }
        )
        .groupby("date")["difference"]
        .mean()
        .dropna()
        .to_numpy()
    )
    count = len(daily)
    mean = float(daily.mean())
    variance = float(np.var(daily, ddof=1))
    for lag in range(1, min(horizon, count - 2) + 1):
        covariance = float(
            np.cov(daily[lag:], daily[:-lag], ddof=1)[0, 1]
        )
        variance += 2 * (1 - lag / (horizon + 1)) * covariance
    standard_error = math.sqrt(max(variance, 1e-18) / count)
    statistic = mean / standard_error
    p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))
    return float(statistic), float(p_value)


def moving_block_bootstrap_gain(
    scored: pd.DataFrame,
    reps: int,
    block_days: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap relative QLIKE gain using whole cross-sections by date."""
    daily_groups = [
        group.index.to_numpy()
        for _, group in scored.groupby("date", sort=True)
    ]
    rng = np.random.default_rng(seed)
    gains = []
    maximum_start = max(1, len(daily_groups) - block_days + 1)
    for _ in range(reps):
        sampled: list[np.ndarray] = []
        while len(sampled) < len(daily_groups):
            start = int(rng.integers(0, maximum_start))
            sampled.extend(daily_groups[start : start + block_days])
        index = np.concatenate(sampled[: len(daily_groups)])
        reference = scored.loc[index, "reference_loss"].mean()
        candidate = scored.loc[index, "candidate_loss"].mean()
        gains.append((reference - candidate) / reference)
    return tuple(float(value) for value in np.quantile(gains, [0.025, 0.5, 0.975]))


def model_metrics(
    frame: pd.DataFrame,
    horizon: int,
    forecast: np.ndarray,
) -> dict[str, float]:
    target = frame[f"realized_vol_{horizon}d"].to_numpy()
    valid = np.isfinite(target) & np.isfinite(forecast) & (forecast > 0)
    target = target[valid]
    prediction = np.asarray(forecast)[valid]
    log_target = np.log(np.maximum(target, EPS))
    log_prediction = np.log(np.maximum(prediction, EPS))
    slope, intercept = np.polyfit(log_prediction, log_target, 1)
    denominator = np.sum((log_target - log_target.mean()) ** 2)
    return {
        "qlike": symbol_equal_qlike(
            frame.loc[valid],
            f"realized_vol_{horizon}d",
            prediction,
        ),
        "rmse_log": float(
            np.sqrt(np.mean((log_target - log_prediction) ** 2))
        ),
        "r2_log": float(
            1 - np.sum((log_target - log_prediction) ** 2) / denominator
        ),
        "mz_intercept": float(intercept),
        "mz_slope": float(slope),
        "mean_realized_to_forecast": float(np.mean(target / prediction)),
        "n": int(valid.sum()),
    }


def current_checkpoint_forecast(
    frame: pd.DataFrame,
    horizon: int,
    checkpoint: Mapping[str, object] | None = None,
) -> np.ndarray:
    if checkpoint is None:
        checkpoint = json.loads(
            CURRENT_MODEL_PATH.read_text(encoding="utf-8")
        )
    model = checkpoint["horizons"][str(horizon)]
    log_sigma = model["intercept"] + sum(
        frame[feature].to_numpy() * coefficient
        for feature, coefficient in model["coef"].items()
    )
    return np.exp(log_sigma) * model["smearing"]


def parameter_configs(count: int, seed: int = RANDOM_SEED) -> list[dict[str, float | int]]:
    """Deterministic random design for XGBoost's regularised shallow trees."""
    rng = np.random.default_rng(seed)
    configs = []
    for index in range(count):
        configs.append(
            {
                "config_id": index,
                "max_depth": int(rng.integers(2, 7)),
                "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.15)))),
                "min_child_weight": float(np.exp(rng.uniform(np.log(1), np.log(64)))),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.5, 1.0)),
                "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(100)))),
                "reg_alpha": float(
                    0.0 if rng.random() < 0.2 else np.exp(rng.uniform(np.log(1e-4), np.log(10)))
                ),
                "gamma": float(rng.uniform(0.0, 5.0)),
            }
        )
    return configs


def fit_xgb_gamma(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target_train: np.ndarray,
    target_validation: np.ndarray,
    parameters: Mapping[str, float | int],
    device: str,
):
    from xgboost import XGBRegressor

    kwargs = {
        key: value for key, value in parameters.items() if key != "config_id"
    }
    model = XGBRegressor(
        objective="reg:gamma",
        eval_metric="gamma-deviance",
        n_estimators=3000,
        early_stopping_rounds=100,
        tree_method="hist",
        device=device,
        random_state=RANDOM_SEED,
        n_jobs=4,
        **kwargs,
    )
    model.fit(
        train[features],
        np.maximum(target_train, EPS),
        sample_weight=symbol_equal_weights(train),
        eval_set=[(validation[features], np.maximum(target_validation, EPS))],
        verbose=False,
    )
    return model


def fit_xgb_gamma_fixed(
    frame: pd.DataFrame,
    features: list[str],
    target: np.ndarray,
    parameters: Mapping[str, float | int],
    device: str,
    n_estimators: int,
):
    """Refit a tuned Gamma booster on all available pre-test observations."""
    from xgboost import XGBRegressor

    kwargs = {
        key: value for key, value in parameters.items() if key != "config_id"
    }
    model = XGBRegressor(
        objective="reg:gamma",
        eval_metric="gamma-deviance",
        n_estimators=max(1, int(n_estimators)),
        tree_method="hist",
        device=device,
        random_state=RANDOM_SEED,
        n_jobs=4,
        **kwargs,
    )
    model.fit(
        frame[features],
        np.maximum(target, EPS),
        sample_weight=symbol_equal_weights(frame),
        verbose=False,
    )
    return model


def xgb_sigma_prediction(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    variance = np.clip(model.predict(frame[features]), EPS, 1.0)
    return np.sqrt(variance)


def optimal_sigma_scale(
    realized_vol: np.ndarray,
    forecast_vol: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    ratio = (
        np.maximum(np.asarray(realized_vol), EPS)
        / np.maximum(np.asarray(forecast_vol), EPS)
    ) ** 2
    variance_scale = (
        float(np.average(ratio, weights=weights))
        if weights is not None
        else float(ratio.mean())
    )
    return float(np.sqrt(np.clip(variance_scale, 0.25, 4.0)))
