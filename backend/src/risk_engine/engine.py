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

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import DATA_DIR

MODEL_PATH = DATA_DIR / "processed" / "risk_model.json"
EPS = 1e-6
HORIZONS = (5, 20)
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


def _empty(symbol: str, h: int) -> RiskEstimate:
    nan = float("nan")
    return RiskEstimate(symbol, h, nan, nan, nan, nan, nan, nan, nan, nan, None, False)


# --------------------------------------------------------------- per stock
def risk_estimate(symbol: str, ohlc: pd.DataFrame, h: int) -> RiskEstimate:
    """Downside-risk suite for one stock at horizon h."""
    if _MODEL is None or str(h) not in _MODEL["horizons"]:
        return _empty(symbol, h)
    mh = _MODEL["horizons"][str(h)]
    fh = _MODEL["fhs"]["horizons"][str(h)]
    ff = _feature_frame(ohlc).dropna(subset=mh["features"])
    if len(ff) < _MIN_HISTORY:
        return _empty(symbol, h)

    sig = _sigma_series(ff, mh)
    sd = float(sig.iloc[-1])
    sh = sd * np.sqrt(h)
    sig_h = sig * np.sqrt(h)
    level = float((sig_h <= sh).mean() * 100)
    return RiskEstimate(
        symbol=symbol, horizon=h, sigma_daily=sd, sigma_h=sh,
        var_95=sh * fh["q05"], var_99=sh * fh["q01"], es_95=sh * fh["es05"],
        band_lo=sh * fh["q025"], band_hi=sh * fh["q975"], risk_level=level,
        as_of=str(ff.index[-1].date()), has_history=True)


def risk_estimates(ohlc_by_symbol: Dict[str, pd.DataFrame],
                   horizons=HORIZONS) -> List[RiskEstimate]:
    return [risk_estimate(sym, ohlc, h)
            for sym, ohlc in ohlc_by_symbol.items() for h in horizons]


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
    D = SG[-1]
    sigp_1d = float(np.sqrt(max(w @ (np.outer(D, D) * Rt) @ w, 1e-12)))
    sigp_h = sigp_1d * np.sqrt(h)

    sum_var = float(np.sum(w * D * np.sqrt(h) * one["q05"]))   # undiversified Σ w·VaR_i
    var95 = sigp_h * pf["q05"]
    return PortfolioRisk(
        horizon=h, n_holdings=len(syms), sigma_h=sigp_h,
        var_95=var95, var_99=sigp_h * pf["q01"], es_95=sigp_h * pf["es05"],
        diversification_ratio=(var95 / sum_var if sum_var else float("nan")),
        as_of=str(common[-1].date()))
