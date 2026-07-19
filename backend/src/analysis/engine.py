"""Analysis engine — calendar alignment, constant-mix portfolio construction,
and risk/performance metrics.

Ported from frontendjs src/lib/metrics.ts. This is a standalone utility module
(NOT an engine in the interfaces.py sense — no contract functions, used
internally by the /analysis/explore router).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE = 0.04  # annual

ZERO_STATS = {
    "totalReturn": 0.0,
    "cagr": 0.0,
    "annVol": 0.0,
    "sharpe": 0.0,
    "sortino": 0.0,
    "maxDrawdown": 0.0,
    "calmar": 0.0,
    "bestDay": 0.0,
    "worstDay": 0.0,
    "winRate": 0.0,
}


# ------------------------------------------------------------------ alignment


def align_series(
    input_data: Dict[str, pd.DataFrame],
) -> Optional[Dict]:
    """Align several daily series on a common union calendar.

    Each value is a DataFrame with DatetimeIndex and a 'close' column (or a
    single-column Series). Forward-fills after the first observation; the
    window spans from the latest first-observation to the earliest
    last-observation so every series is defined across the whole window.

    Returns:
        {
            "dates": [str],               # ISO date strings
            "values": {symbol: [float]},  # aligned close prices
            "truncated": bool,
            "truncated_by": str | None,   # symbol that caused truncation
        }
        or None if alignment is impossible.
    """
    symbols = [s for s, df in input_data.items() if len(df) > 0]
    if not symbols:
        return None

    # Extract close series as pd.Series with DatetimeIndex
    series_map: Dict[str, pd.Series] = {}
    for s in symbols:
        df = input_data[s]
        if isinstance(df, pd.DataFrame):
            col = df["close"] if "close" in df.columns else df.iloc[:, 0]
        else:
            col = df
        series_map[s] = col.sort_index()

    # Find window bounds: latest first date, earliest last date
    start_dates = {s: sr.index[0] for s, sr in series_map.items()}
    end_dates = {s: sr.index[-1] for s, sr in series_map.items()}
    window_start: pd.Timestamp = max(start_dates.values())
    window_end: pd.Timestamp = min(end_dates.values())

    if window_start >= window_end:
        return None

    # Find which symbol caused truncation
    truncated_by = max(start_dates, key=lambda s: start_dates[s])
    global_first = min(start_dates.values())
    truncated = window_start > global_first

    # Build union date index within the window
    all_dates: set = set()
    for sr in series_map.values():
        for d in sr.index:
            if window_start <= d <= window_end:
                all_dates.add(d)
    dates = sorted(all_dates)
    if len(dates) < 2:
        return None

    date_index = pd.DatetimeIndex(dates)

    # Forward-fill each series onto the union index
    values: Dict[str, list] = {}
    for s in symbols:
        sr = series_map[s]
        # Reindex to union calendar, forward-fill, back-fill first value
        aligned = sr.reindex(date_index, method="ffill")
        # Back-fill any leading NaNs (instrument may not trade on window day 1)
        if pd.isna(aligned.iloc[0]):
            first_valid = aligned.first_valid_index()
            if first_valid is not None:
                aligned.loc[:first_valid] = aligned.loc[first_valid]
        # Drop any remaining NaN-only symbols
        if aligned.isna().all():
            continue
        values[s] = aligned.ffill().tolist()

    if not values:
        return None

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "values": values,
        "truncated": truncated,
        "truncated_by": truncated_by,
    }


# --------------------------------------------------------- portfolio construction


def index_series(values: List[float]) -> List[float]:
    """Index a price series to base 100."""
    base = values[0]
    if not base or base <= 0:
        return [100.0] * len(values)
    return [round(v / base * 100, 4) for v in values]


def build_portfolio_index(
    symbols: List[str],
    weights: Dict[str, float],
    values: Dict[str, List[float]],
    n: int,
) -> List[float]:
    """Build a daily-rebalanced constant-mix portfolio indexed to 100."""
    out = []
    for i in range(n):
        total = 0.0
        for s in symbols:
            if s not in values:
                continue
            base = values[s][0]
            if base > 0:
                total += weights.get(s, 0.0) * (values[s][i] / base)
        out.append(round(total * 100, 4))
    return out


# ------------------------------------------------------------------ statistics


def daily_returns(values: List[float]) -> List[float]:
    """Percent-change series from price values."""
    r = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        if prev > 0:
            r.append(values[i] / prev - 1)
    return r


def _mean(a: List[float]) -> float:
    if not a:
        return 0.0
    return sum(a) / len(a)


def _std(a: List[float], m: float) -> float:
    if len(a) < 2:
        return 0.0
    var = sum((x - m) ** 2 for x in a) / (len(a) - 1)
    return math.sqrt(var)


def compute_stats(values: List[float], dates: List[str]) -> dict:
    """Compute full SeriesStats from raw price values and date strings."""
    n = len(values)
    if n < 2 or len(dates) < 2:
        return dict(ZERO_STATS)

    rets = daily_returns(values)
    if not rets:
        return dict(ZERO_STATS)

    m = _mean(rets)
    sd = _std(rets, m)

    first = values[0]
    last = values[-1]
    total_return = (last / first - 1) if first > 0 else 0.0

    # Years spanned (exact)
    span_days = (
        pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])
    ).days
    years = max(span_days / 365.25, 1.0 / TRADING_DAYS)
    cagr = (
        math.pow(last / first, 1.0 / years) - 1
        if first > 0 and last > 0
        else 0.0
    )

    ann_vol = sd * math.sqrt(TRADING_DAYS)
    ann_ret_arith = m * TRADING_DAYS

    sharpe = (ann_ret_arith - RISK_FREE) / ann_vol if ann_vol > 1e-9 else 0.0

    # Sortino: downside deviation only
    rf_daily = RISK_FREE / TRADING_DAYS
    downside_sq = sum(min(r - rf_daily, 0) ** 2 for r in rets)
    downside_dev = math.sqrt(downside_sq / max(len(rets), 1)) * math.sqrt(TRADING_DAYS)
    sortino = (ann_ret_arith - RISK_FREE) / downside_dev if downside_dev > 1e-9 else 0.0

    # Max drawdown
    peak = values[0]
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = v / peak - 1
        if dd < max_dd:
            max_dd = dd

    calmar = cagr / abs(max_dd) if max_dd < -1e-9 else 0.0

    best = max(rets) if rets else 0.0
    worst = min(rets) if rets else 0.0
    wins = sum(1 for r in rets if r > 0)
    win_rate = wins / len(rets) if rets else 0.0

    return {
        "totalReturn": round(total_return, 4),
        "cagr": round(cagr, 4),
        "annVol": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "maxDrawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "bestDay": round(best, 4),
        "worstDay": round(worst, 4),
        "winRate": round(win_rate, 4),
    }


def beta_alpha(
    a_values: List[float], b_values: List[float]
) -> Optional[Dict[str, float]]:
    """Beta & Jensen's alpha (annualized) of a vs b using daily returns."""
    n = min(len(a_values), len(b_values))
    if n < 30:
        return None

    ra, rb = [], []
    for i in range(1, n):
        if a_values[i - 1] > 0 and b_values[i - 1] > 0:
            ra.append(a_values[i] / a_values[i - 1] - 1)
            rb.append(b_values[i] / b_values[i - 1] - 1)

    if len(ra) < 30:
        return None

    ma = _mean(ra)
    mb = _mean(rb)
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(len(ra)))
    var_b = sum((r - mb) ** 2 for r in rb)

    if var_b <= 1e-12:
        return None

    beta = cov / var_b
    alpha = (ma * TRADING_DAYS - RISK_FREE) - beta * (mb * TRADING_DAYS - RISK_FREE)
    return {"beta": round(beta, 3), "alpha": round(alpha, 4)}


def ytd_return(values: List[float], dates: List[str]) -> float:
    """Return from start of current calendar year to latest value."""
    if len(values) < 2:
        return 0.0
    last_date = dates[-1]
    year = last_date[:4]
    jan1 = f"{year}-01-01"
    base_idx = 0
    for i, d in enumerate(dates):
        if d < jan1:
            base_idx = i
        else:
            break
    base = values[base_idx]
    return round(values[-1] / base - 1, 4) if base > 0 else 0.0


def monthly_returns(values: List[float], dates: List[str]) -> List[dict]:
    """Break daily series into monthly return cells."""
    if len(values) < 2:
        return []

    last_idx_by_month: Dict[str, int] = {}
    for i, d in enumerate(dates):
        last_idx_by_month[d[:7]] = i

    months = sorted(last_idx_by_month.keys())
    cells = []
    prev_end_val = values[0]
    for k, month in enumerate(months):
        idx = last_idx_by_month[month]
        end_val = values[idx]
        r = (
            end_val / prev_end_val - 1
            if prev_end_val > 0
            else 0.0
        )
        cells.append({
            "year": int(month[:4]),
            "month": int(month[5:7]),
            "r": round(r if k > 0 else (end_val / values[0] - 1 if values[0] > 0 else 0.0), 4),
        })
        prev_end_val = end_val

    return cells


def round_n(x: float, n: int = 4) -> float:
    """Round to n decimal places."""
    p = 10 ** n
    return math.floor(x * p + 0.5) / p


def round_series(values: List[float], n: int = 4) -> List[float]:
    """Map round_n over a series."""
    return [round_n(v, n) for v in values]
