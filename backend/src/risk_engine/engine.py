"""Risk Engine — per-stock and portfolio downside risk from the HAR + FHS model.

Loads the model trained OFFLINE (scripts/train_risk_engine_v2.py + risk_measures.py,
serialized to data/processed/risk_model.json) and does pure ONLINE inference:

    OHLC history ─► HAR σ̂ (a dot product on price features)
                 ─► Filtered Historical Simulation ─► VaR / ES / band / risk-level

No training, no refitting — just the serialized coefficients + the empirical
standardized-return quantiles. HAR forecasts the volatility; the FHS quantiles
(fat-tailed, left-skewed, calibrated on ≤2020) turn σ̂ into downside risk that a
formula-based decision layer can consume directly.

Inputs are per-symbol OHLC frames (data_loader.get_ohlc_history) — the Parkinson
feature needs high/low. This engine is READ-ONLY w.r.t. the shared kernel and does
not import any other engine. Consumed by routers/risk.py; risk_level is the natural
input to the recommendation engine's `market_volatility` factor.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.config import DATA_DIR

MODEL_PATH = DATA_DIR / "processed" / "risk_model.json"
EPS = 1e-6
HORIZONS = (5,)
AUXILIARY_HORIZONS = (20,)
_MIN_HISTORY = 80          # need ≥66 for rv66 + a little slack


@dataclass
class RiskEstimate:
    symbol: str
    horizon: int
    sigma_daily: float           # HAR per-day vol forecast
    sigma_h: float               # forecast vol over the horizon
    var_95: float                # downside VaR (h-day return threshold, negative)
    var_99: float
    es_95: float                 # expected shortfall / CVaR
    band_lo: float
    band_hi: float
    risk_level: float            # 0-100, σ̂ percentile vs the stock's own history
    as_of: Optional[str]
    has_history: bool
    model_version: str = ""
    news_applied: bool = False
    news_quality: str = "not_configured"


@dataclass
class PortfolioRisk:
    horizon: int
    n_holdings: int
    sigma_h: float
    var_95: float
    var_99: float
    es_95: float
    diversification_ratio: float   # portfolio VaR / Σ w·VaR_i  (<1 = diversified)
    as_of: Optional[str]


def _load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        m = json.loads(MODEL_PATH.read_text())
        return m if "fhs" in m else None
    except Exception:
        return None


_MODEL = _load_model()
_NEWS_BOOSTERS: Dict[str, object] = {}
_NEWS_STORE_CACHE: tuple[int, list[dict]] | None = None
_NEWS_ALIASES = {
    "AAPL": ("apple",),
    "ADBE": ("adobe",),
    "AMD": ("amd", "advanced micro devices"),
    "AMZN": ("amazon",),
    "ASML": ("asml",),
    "AVGO": ("broadcom",),
    "COST": ("costco",),
    "CSCO": ("cisco",),
    "GLD": ("gold",),
    "GOOGL": ("google", "alphabet"),
    "INTC": ("intel",),
    "INTU": ("intuit",),
    "MSFT": ("microsoft",),
    "MU": ("micron",),
    "NFLX": ("netflix",),
    "NVDA": ("nvidia",),
    "PEP": ("pepsico",),
    "QCOM": ("qualcomm",),
    "SLV": ("silver",),
    "SPY": ("s&p 500", "stock market", "wall street"),
    "TSLA": ("tesla", "elon musk"),
    "TXN": ("texas instruments",),
    "XLV": ("healthcare", "pharma", "fda"),
}


def model_available() -> bool:
    return _MODEL is not None


# --------------------------------------------------------------- features
def _feature_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    """The exact HAR feature block used in training (log RV 5/22/66 + Parkinson
    5/22 + |r|), plus the raw daily return for the portfolio EWMA covariance."""
    c, hi, lo = ohlc["close"], ohlc["high"], ohlc["low"]
    ret = c.pct_change()
    park = (np.log(hi / lo) ** 2) / (4 * np.log(2))
    df = pd.DataFrame(index=c.index)
    df["l_rv5"] = np.log(ret.rolling(5).std() + EPS)
    df["l_rv22"] = np.log(ret.rolling(22).std() + EPS)
    df["l_rv66"] = np.log(ret.rolling(66).std() + EPS)
    df["l_park5"] = np.log(np.sqrt(park.rolling(5).mean()) + EPS)
    df["l_park22"] = np.log(np.sqrt(park.rolling(22).mean()) + EPS)
    df["l_absret"] = np.log(ret.abs() + EPS)
    df["ret1"] = ret
    return df


def _sigma_series(fframe: pd.DataFrame, mh: dict) -> pd.Series:
    logv = mh["intercept"] + sum(fframe[f] * mh["coef"][f] for f in mh["features"])
    return np.exp(logv) * mh["smearing"]


def _model_version() -> str:
    if _MODEL is None:
        return ""
    return str(
        _MODEL.get("version")
        or _MODEL.get("created")
        or _MODEL.get("model", "")
    )


def _news_overlay_spec(h: int) -> Optional[dict]:
    if _MODEL is None:
        return None
    root_spec = _MODEL.get("news_overlays", {}).get(str(h))
    horizon_spec = _MODEL.get("horizons", {}).get(str(h), {}).get(
        "news_overlay"
    )
    return root_spec or horizon_spec


def _news_features_from_store(
    symbol: str,
    as_of: pd.Timestamp,
) -> Mapping[str, float]:
    """Build the deployed FNSPID-compatible attention input from live RSS.

    Training moves each article strictly to its next trading session. For
    session t, stories since the previous business session and before t
    become t's news count. An observed zero is an explicit news state.
    """
    global _NEWS_STORE_CACHE
    path = DATA_DIR / "news_raw.json"
    if not path.exists():
        return {
            "log_count": 0.0,
            "news_count": 0.0,
            "__quality__": "missing_store",
        }
    try:
        modified = path.stat().st_mtime_ns
        if _NEWS_STORE_CACHE is None or _NEWS_STORE_CACHE[0] != modified:
            records = json.loads(path.read_text(encoding="utf-8"))
            _NEWS_STORE_CACHE = (modified, records)
        records = _NEWS_STORE_CACHE[1]

        session = pd.Timestamp(as_of).normalize()
        if session.tzinfo is None:
            session = session.tz_localize("UTC")
        else:
            session = session.tz_convert("UTC")
        previous = session - pd.tseries.offsets.BDay(1)
        seen: set[str] = set()
        latest_fetch = None
        for record in records:
            fetched = pd.to_datetime(
                record.get("fetched_utc"), utc=True, errors="coerce"
            )
            if pd.notna(fetched) and (
                latest_fetch is None or fetched > latest_fetch
            ):
                latest_fetch = fetched
            tagged = {
                str(value).upper()
                for value in record.get("symbols", [])
            }
            text = " ".join(
                [
                    str(record.get("title") or ""),
                    str(record.get("summary") or ""),
                ]
            ).lower()
            aliases = _NEWS_ALIASES.get(symbol.upper(), ())
            mentioned = any(alias in text for alias in aliases)
            if symbol.upper() not in tagged and not mentioned:
                continue
            published = pd.to_datetime(
                record.get("published_utc"), utc=True, errors="coerce"
            )
            if pd.isna(published) or not (
                previous <= published < session
            ):
                continue
            seen.add(str(record.get("id") or record.get("url") or published))

        count = len(seen)
        now = pd.Timestamp(datetime.now(timezone.utc))
        if latest_fetch is None or now - latest_fetch > pd.Timedelta(hours=48):
            quality = "stale_store"
        else:
            quality = "fresh" if count else "no_news"
        return {
            "log_count": float(np.log1p(count)),
            "news_count": float(count),
            "__quality__": quality,
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "log_count": 0.0,
            "news_count": 0.0,
            "__quality__": "invalid_store",
        }


def _linear_news_multiplier(
    spec: Mapping[str, object],
    features: Mapping[str, float],
) -> float:
    required = list(spec.get("features", []))
    coefficient = dict(spec.get("coef", {}))
    scale = dict(spec.get("scale", {}))
    values = []
    for name in required:
        if name not in features:
            raise KeyError(name)
        value = float(features[name])
        if not np.isfinite(value):
            raise ValueError(name)
        values.append(value / max(float(scale.get(name, 1.0)), EPS))
    log_variance_ratio = float(spec.get("intercept", 0.0)) + sum(
        coefficient[name] * value
        for name, value in zip(required, values)
    )
    maximum = float(spec.get("max_sigma_multiplier", 2.0))
    limit = 2.0 * np.log(max(maximum, 1.0))
    return float(
        np.exp(0.5 * np.clip(log_variance_ratio, -limit, limit))
    )


def _xgb_news_multiplier(
    spec: Mapping[str, object],
    features: Mapping[str, float],
    h: int,
) -> float:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is required by the news overlay") from exc
    artifact = str(spec["artifact"])
    path = MODEL_PATH.parent / artifact
    key = f"{h}:{path.resolve()}"
    if key not in _NEWS_BOOSTERS:
        booster = xgb.Booster()
        booster.load_model(path)
        _NEWS_BOOSTERS[key] = booster
    required = list(spec.get("features", []))
    values = []
    for name in required:
        if name not in features:
            raise KeyError(name)
        value = float(features[name])
        if not np.isfinite(value):
            raise ValueError(name)
        values.append(value)
    prediction = float(
        _NEWS_BOOSTERS[key].predict(
            xgb.DMatrix(
                np.asarray([values], dtype=float),
                feature_names=required,
            )
        )[0]
    )
    maximum = float(spec.get("max_sigma_multiplier", 2.0))
    return float(
        np.clip(np.sqrt(max(prediction, EPS)), 1 / maximum, maximum)
    )


def _apply_news_overlay(
    sigma_daily: float,
    h: int,
    news_features: Optional[Mapping[str, float]],
) -> tuple[float, bool, str]:
    spec = _news_overlay_spec(h)
    if spec is None:
        return sigma_daily, False, "not_configured"
    if news_features is None:
        return sigma_daily, False, "missing"
    supplied_quality = str(news_features.get("__quality__", "ok")).lower()
    # An OBSERVED zero is a real news state and must pass through the joint
    # model. A DEGRADED store is not an observed zero — it is an unknown, and
    # reporting news_applied=True for it claims a channel that carried no data.
    #
    # Numerically this costs nothing: the calibrated multiplier at log_count=0
    # is exactly 1.0, so a degraded store already produced the price-only HAR
    # sigma. Excluding these qualities changes only the honesty of the report —
    # news_applied becomes False and the quality string reaches the caller, so
    # a broken RSS feed is visible instead of looking like a calm market.
    # (Deliberately NOT imputing a non-zero count: inventing news attention to
    # fail conservative would fabricate an input the model never observed.)
    if supplied_quality not in {
        "ok",
        "fresh",
        "complete",
        "no_news",
    }:
        return sigma_daily, False, supplied_quality
    try:
        model_type = str(spec.get("model_type", ""))
        if model_type == "linear_gamma_variance_ratio":
            multiplier = _linear_news_multiplier(spec, news_features)
        elif model_type == "xgboost_gamma_variance_ratio":
            multiplier = _xgb_news_multiplier(spec, news_features, h)
        else:
            return sigma_daily, False, "unsupported_model"
    except (KeyError, TypeError, ValueError, RuntimeError, OSError):
        return sigma_daily, False, "invalid"
    return sigma_daily * multiplier, True, supplied_quality


def _empty(symbol: str, h: int) -> RiskEstimate:
    nan = float("nan")
    return RiskEstimate(
        symbol=symbol,
        horizon=h,
        sigma_daily=nan,
        sigma_h=nan,
        var_95=nan,
        var_99=nan,
        es_95=nan,
        band_lo=nan,
        band_hi=nan,
        risk_level=nan,
        as_of=None,
        has_history=False,
        model_version=_model_version(),
    )


# --------------------------------------------------------------- per stock
def risk_estimate(
    symbol: str,
    ohlc: pd.DataFrame,
    h: int,
    news_features: Optional[Mapping[str, float]] = None,
) -> RiskEstimate:
    """Downside-risk suite for one stock at horizon h."""
    if _MODEL is None or str(h) not in _MODEL["horizons"]:
        return _empty(symbol, h)
    mh = _MODEL["horizons"][str(h)]
    fh = _MODEL["fhs"]["horizons"][str(h)]
    ff = _feature_frame(ohlc).dropna(subset=mh["features"])
    if len(ff) < _MIN_HISTORY:
        return _empty(symbol, h)

    sig = _sigma_series(ff, mh)
    base_sd = float(sig.iloc[-1])
    if news_features is None and _news_overlay_spec(h) is not None:
        news_features = _news_features_from_store(
            symbol, pd.Timestamp(ff.index[-1])
        )
    sd, news_applied, news_quality = _apply_news_overlay(
        base_sd, h, news_features
    )
    sh = sd * np.sqrt(h)
    sig_h = sig * np.sqrt(h)
    level = float((sig_h <= sh).mean() * 100)
    return RiskEstimate(
        symbol=symbol, horizon=h, sigma_daily=sd, sigma_h=sh,
        var_95=sh * fh["q05"], var_99=sh * fh["q01"], es_95=sh * fh["es05"],
        band_lo=sh * fh["q025"], band_hi=sh * fh["q975"], risk_level=level,
        as_of=str(ff.index[-1].date()), has_history=True,
        model_version=_model_version(), news_applied=news_applied,
        news_quality=news_quality)


def risk_estimates(
    ohlc_by_symbol: Dict[str, pd.DataFrame],
    horizons=HORIZONS,
    news_features_by_symbol: Optional[
        Mapping[str, Mapping[str, float]]
    ] = None,
) -> List[RiskEstimate]:
    news_features_by_symbol = news_features_by_symbol or {}
    return [
        risk_estimate(
            sym,
            ohlc,
            h,
            news_features=news_features_by_symbol.get(sym),
        )
        for sym, ohlc in ohlc_by_symbol.items()
        for h in horizons
    ]


# --------------------------------------------------------------- portfolio
def portfolio_risk(ohlc_by_symbol: Dict[str, pd.DataFrame],
                   weights: Dict[str, float], h: int) -> Optional[PortfolioRisk]:
    """Portfolio VaR/ES via HAR σ̂ diagonal + EWMA correlation (adapts the
    diversification benefit to the current regime) + the empirical portfolio tail."""
    if _MODEL is None:
        return None
    mh = _MODEL["horizons"]["5"]           # responsive per-day σ̂
    pf = _MODEL["fhs"]["portfolio"]
    one = _MODEL["fhs"]["one_day"]

    sig_cols, ret_cols = {}, {}
    for sym, ohlc in ohlc_by_symbol.items():
        if weights.get(sym, 0.0) <= 0:
            continue
        ff = _feature_frame(ohlc).dropna(subset=mh["features"])
        if len(ff) < _MIN_HISTORY:
            continue
        sig_cols[sym] = _sigma_series(ff, mh)
        ret_cols[sym] = ff["ret1"]
    syms = list(sig_cols)
    if len(syms) < 2:
        return None

    sigM = pd.DataFrame(sig_cols).dropna()
    retM = pd.DataFrame(ret_cols).reindex(sigM.index)
    common = sigM.dropna().index.intersection(retM.dropna().index)
    if len(common) < _MIN_HISTORY:
        return None
    sigM, retM = sigM.loc[common, syms], retM.loc[common, syms]

    w = np.array([weights[s] for s in syms], dtype=float)
    w = w / w.sum()
    R, SG = retM.to_numpy(), sigM.to_numpy()

    lam = float(pf.get("ewma_lambda", 0.94))
    S = np.cov(R[:60].T)
    for i in range(1, len(R)):
        r = R[i - 1]
        S = lam * S + (1 - lam) * np.outer(r, r)
    dv = np.sqrt(np.diag(S))
    Rt = S / np.outer(dv, dv)
    # The portfolio covariance uses the same formal five-session HAR-News
    # marginal risk for every holding; news therefore affects both per-stock
    # and portfolio outputs.
    D = np.asarray(
        [
            _apply_news_overlay(
                float(SG[-1, index]),
                5,
                _news_features_from_store(sym, pd.Timestamp(common[-1])),
            )[0]
            for index, sym in enumerate(syms)
        ],
        dtype=float,
    )
    sigp_1d = float(np.sqrt(max(w @ (np.outer(D, D) * Rt) @ w, 1e-12)))
    sigp_h = sigp_1d * np.sqrt(h)

    sum_var = float(np.sum(w * D * np.sqrt(h) * one["q05"]))   # undiversified Σ w·VaR_i
    var95 = sigp_h * pf["q05"]
    return PortfolioRisk(
        horizon=h, n_holdings=len(syms), sigma_h=sigp_h,
        var_95=var95, var_99=sigp_h * pf["q01"], es_95=sigp_h * pf["es05"],
        diversification_ratio=(var95 / sum_var if sum_var else float("nan")),
        as_of=str(common[-1].date()))
