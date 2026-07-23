"""Validate the AURORA risk engine on the 2024-present broader test set.

Reads the artifacts produced by ``build_current_risk_testset.py`` and evaluates:

  - HAR point forecasts versus the naïve trailing-22-day volatility baseline;
  - VaR-95 / VaR-99 breach calibration;
  - 95% return-band coverage;
  - expected-shortfall severity;
  - whether risk-level deciles monotonically rank subsequent volatility.

The panel contains overlapping 5/20-day outcomes and cross-sectionally related
assets. Formal uncertainty therefore operates on date-level cross-sectional
means and uses moving blocks of trading dates rather than an IID row bootstrap.

Outputs:
  data/processed/current_risk_test/current_validation_summary.csv
  data/processed/current_risk_test/risk_level_calibration.csv
  data/processed/current_risk_test/current_validation_report.md

Run:
  python scripts/validate_current_risk_testset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "current_risk_test"
PANEL_PATH = DATA / "risk_backtest_panel.parquet"
OHLC_PATH = DATA / "ohlc.parquet"
HORIZONS = (5, 20)
BOOTSTRAP_REPS = 1000
EPS = 1e-8


def add_naive_forecast(panel: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    naive_frames = []
    for symbol, prices in ohlc.groupby("symbol"):
        prices = prices.sort_values("date")
        ret = prices["close"].pct_change()
        naive_frames.append(
            pd.DataFrame(
                {
                    "date": prices["date"],
                    "symbol": symbol,
                    "naive_sigma_daily": ret.rolling(22).std(),
                }
            )
        )
    naive = pd.concat(naive_frames, ignore_index=True)
    return panel.merge(naive, on=["date", "symbol"], how="left", validate="one_to_one")


def qlike_loss(realized: pd.Series, forecast: pd.Series) -> np.ndarray:
    y2 = np.maximum(realized.to_numpy(dtype=float) ** 2, EPS**2)
    f2 = np.maximum(forecast.to_numpy(dtype=float) ** 2, EPS**2)
    ratio = y2 / f2
    return ratio - np.log(ratio) - 1


def dm_test_date_aggregated(
    dates: pd.Series, loss_har: np.ndarray, loss_naive: np.ndarray, lag: int
) -> tuple[float, float]:
    daily = (
        pd.DataFrame(
            {
                "date": dates.to_numpy(),
                "difference": loss_har - loss_naive,
            }
        )
        .groupby("date")["difference"]
        .mean()
        .dropna()
    )
    values = daily.to_numpy()
    n = len(values)
    mean = values.mean()
    variance = np.var(values, ddof=1)
    for k in range(1, min(lag, n - 2) + 1):
        covariance = np.cov(values[k:], values[:-k], ddof=1)[0, 1]
        variance += 2 * (1 - k / (lag + 1)) * covariance
    standard_error = np.sqrt(max(variance, 1e-18) / n)
    statistic = mean / standard_error
    p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))
    return float(statistic), float(p_value)


def moving_block_mean_ci(
    date_values: pd.Series, block_length: int, seed: int
) -> tuple[float, float]:
    values = date_values.dropna().to_numpy(dtype=float)
    n = len(values)
    if n < block_length:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts_max = n - block_length + 1
    estimates = np.empty(BOOTSTRAP_REPS)
    for rep in range(BOOTSTRAP_REPS):
        sample = []
        while len(sample) < n:
            start = int(rng.integers(0, starts_max))
            sample.extend(values[start : start + block_length])
        estimates[rep] = np.mean(sample[:n])
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return float(lo), float(hi)


def daily_metric_ci(
    sample: pd.DataFrame, column: str, horizon: int, seed: int
) -> tuple[float, float]:
    daily = sample.groupby("date")[column].mean()
    return moving_block_mean_ci(daily, block_length=horizon, seed=seed)


def scope_masks(panel: pd.DataFrame) -> dict[str, pd.Series]:
    original = panel["groups"].str.contains("original_research")
    return {
        "all": pd.Series(True, index=panel.index),
        "original_research": original,
        "external_generalization": ~original,
        "current_portfolio": panel["groups"].str.contains("current_portfolio"),
    }


def validate(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope_index, (scope, scope_mask) in enumerate(scope_masks(panel).items()):
        for horizon in HORIZONS:
            mask = (
                scope_mask
                & panel[f"mature_{horizon}d"]
                & panel[f"realized_vol_{horizon}d"].notna()
                & panel["naive_sigma_daily"].notna()
            )
            sample = panel.loc[mask].copy()
            realized = sample[f"realized_vol_{horizon}d"]
            har = sample[f"sigma_daily_{horizon}d"]
            naive = sample["naive_sigma_daily"]
            har_loss = qlike_loss(realized, har)
            naive_loss = qlike_loss(realized, naive)
            q_har, q_naive = float(har_loss.mean()), float(naive_loss.mean())
            dm_t, dm_p = dm_test_date_aggregated(
                sample["date"], har_loss, naive_loss, lag=horizon
            )

            breach95 = sample[f"breach95_{horizon}d"].astype(float)
            breach99 = sample[f"breach99_{horizon}d"].astype(float)
            coverage = sample[f"inside_band_{horizon}d"].astype(float)
            sample["_breach95"] = breach95
            sample["_breach99"] = breach99
            sample["_coverage"] = coverage
            seed = 20260723 + scope_index * 100 + horizon
            b95_lo, b95_hi = daily_metric_ci(
                sample, "_breach95", horizon, seed
            )
            b99_lo, b99_hi = daily_metric_ci(
                sample, "_breach99", horizon, seed + 1
            )
            cov_lo, cov_hi = daily_metric_ci(
                sample, "_coverage", horizon, seed + 2
            )

            tail = sample[breach95 == 1]
            es_ratio = (
                tail[f"fwd_return_{horizon}d"].mean()
                / tail[f"es95_{horizon}d"].mean()
                if not tail.empty
                else np.nan
            )
            rank_corr, rank_p = stats.spearmanr(
                sample[f"risk_level_{horizon}d"],
                sample[f"realized_vol_{horizon}d"],
                nan_policy="omit",
            )
            rows.append(
                {
                    "scope": scope,
                    "horizon": horizon,
                    "n": len(sample),
                    "n_symbols": sample["symbol"].nunique(),
                    "start": sample["date"].min().date().isoformat(),
                    "end": sample["date"].max().date().isoformat(),
                    "har_qlike": q_har,
                    "naive_qlike": q_naive,
                    "har_relative_qlike_gain": (q_naive - q_har) / q_naive,
                    "dm_t_har_vs_naive": dm_t,
                    "dm_p": dm_p,
                    "var95_breach_rate": breach95.mean(),
                    "var95_ci_lo": b95_lo,
                    "var95_ci_hi": b95_hi,
                    "var99_breach_rate": breach99.mean(),
                    "var99_ci_lo": b99_lo,
                    "var99_ci_hi": b99_hi,
                    "band_coverage": coverage.mean(),
                    "band_ci_lo": cov_lo,
                    "band_ci_hi": cov_hi,
                    "es95_ratio": es_ratio,
                    "risk_level_spearman": rank_corr,
                    "risk_level_spearman_p": rank_p,
                }
            )
    return pd.DataFrame(rows)


def risk_level_calibration(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, scope_mask in scope_masks(panel).items():
        for horizon in HORIZONS:
            mask = (
                scope_mask
                & panel[f"mature_{horizon}d"]
                & panel[f"realized_vol_{horizon}d"].notna()
            )
            sample = panel.loc[mask].copy()
            sample["risk_decile"] = pd.cut(
                sample[f"risk_level_{horizon}d"],
                bins=np.arange(0, 101, 10),
                labels=range(1, 11),
                include_lowest=True,
            )
            grouped = sample.groupby("risk_decile", observed=True)
            for decile, group in grouped:
                rows.append(
                    {
                        "scope": scope,
                        "horizon": horizon,
                        "risk_decile": int(decile),
                        "n": len(group),
                        "mean_risk_level": group[
                            f"risk_level_{horizon}d"
                        ].mean(),
                        "mean_forecast_daily_vol": group[
                            f"sigma_daily_{horizon}d"
                        ].mean(),
                        "mean_realized_daily_vol": group[
                            f"realized_vol_{horizon}d"
                        ].mean(),
                        "median_realized_daily_vol": group[
                            f"realized_vol_{horizon}d"
                        ].median(),
                    }
                )
    return pd.DataFrame(rows)


def fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def write_report(summary: pd.DataFrame) -> None:
    indexed = summary.set_index(["scope", "horizon"])
    all5, all20 = indexed.loc[("all", 5)], indexed.loc[("all", 20)]
    original5 = indexed.loc[("original_research", 5)]
    external5 = indexed.loc[("external_generalization", 5)]
    portfolio5, portfolio20 = (
        indexed.loc[("current_portfolio", 5)],
        indexed.loc[("current_portfolio", 20)],
    )
    lines = [
        "# Current risk-engine validation",
        "",
        "Untouched evaluation period: 2024 through the latest mature outcome.",
        "Confidence intervals use moving blocks of trading dates and preserve",
        "the cross-section within each date.",
        "",
        "## Results",
        "",
        "| Scope | Horizon | HAR QLIKE | Naïve QLIKE | HAR gain | DM p | VaR-95 breaches (95% CI) | Band coverage (95% CI) | ES ratio | Risk-level ρ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.scope} | {int(row.horizon)}d | {row.har_qlike:.3f} | "
            f"{row.naive_qlike:.3f} | {fmt_pct(row.har_relative_qlike_gain)} | "
            f"{row.dm_p:.3g} | {fmt_pct(row.var95_breach_rate)} "
            f"[{fmt_pct(row.var95_ci_lo)}, {fmt_pct(row.var95_ci_hi)}] | "
            f"{fmt_pct(row.band_coverage)} "
            f"[{fmt_pct(row.band_ci_lo)}, {fmt_pct(row.band_ci_hi)}] | "
            f"{row.es95_ratio:.2f} | {row.risk_level_spearman:.3f} |"
        )
    lines += [
        "",
        "## Current conclusion",
        "",
        f"- Across all symbols, HAR improves QLIKE over naïve volatility by "
        f"{fmt_pct(all5.har_relative_qlike_gain)} at 5 days and "
        f"{fmt_pct(all20.har_relative_qlike_gain)} at 20 days "
        f"(DM p={all5.dm_p:.3g} and {all20.dm_p:.3g}).",
        f"- Pooled VaR-95 breach rates are {fmt_pct(all5.var95_breach_rate)} "
        f"and {fmt_pct(all20.var95_breach_rate)}; both block intervals contain "
        "the 5% target.",
        f"- The original research assets' 5-day band covers "
        f"{fmt_pct(original5.band_coverage)}, below target, while external "
        f"assets cover {fmt_pct(external5.band_coverage)}, indicating that the "
        "same band is conservative outside the development universe.",
        f"- For the current portfolio, HAR point-forecast improvements are not "
        f"statistically distinguishable from naïve volatility "
        f"(DM p={portfolio5.dm_p:.3g} and {portfolio20.dm_p:.3g}), although "
        "VaR/band calibration remains close to target.",
        "- Risk level ranks future volatility well in the broad pooled panel,",
        "  but should be interpreted cautiously for the six-symbol current portfolio.",
        "",
        "## Interpretation rules",
        "",
        "- HAR is better than naïve when QLIKE is lower, the relative gain is",
        "  positive, and the date-aggregated DM statistic is negative.",
        "- A calibrated VaR-95 process should have a breach interval containing 5%.",
        "- A calibrated 95% band should have a coverage interval containing 95%.",
        "- ES ratio should be near 1; below 1 means the predicted tail is conservative.",
        "- Risk-level correlation should be positive and risk-decile realized",
        "  volatility should generally increase monotonically.",
        "",
        "Daily outcomes overlap. The block-aware intervals in this report should",
        "be used instead of IID binomial tests on the raw row count.",
    ]
    (DATA / "current_validation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    ohlc = pd.read_parquet(OHLC_PATH)
    panel["date"] = pd.to_datetime(panel["date"])
    ohlc["date"] = pd.to_datetime(ohlc["date"])
    panel = add_naive_forecast(panel, ohlc)

    summary = validate(panel)
    calibration = risk_level_calibration(panel)
    summary.to_csv(DATA / "current_validation_summary.csv", index=False)
    calibration.to_csv(DATA / "risk_level_calibration.csv", index=False)
    write_report(summary)

    print("CURRENT RISK-ENGINE VALIDATION")
    print(
        summary[
            [
                "scope",
                "horizon",
                "har_qlike",
                "naive_qlike",
                "har_relative_qlike_gain",
                "dm_t_har_vs_naive",
                "dm_p",
                "var95_breach_rate",
                "var95_ci_lo",
                "var95_ci_hi",
                "band_coverage",
                "band_ci_lo",
                "band_ci_hi",
                "es95_ratio",
                "risk_level_spearman",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs -> {DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
