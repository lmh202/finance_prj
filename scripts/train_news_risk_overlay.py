"""Research a FNSPID news overlay on top of the frozen HAR risk forecast.

Instead of forcing news into the already-strong HAR point model, predict the
remaining log-volatility error:

    residual_t = log(realized_vol_5d / HAR_sigma_5d)
    sigma_final = HAR_sigma_5d * exp(news_overlay_t)

The overlay is neutral when all news inputs are zero. It is fitted as a Gamma
regression for the variance ratio

    realized_variance_5d / HAR_forecast_variance_5d

Gamma deviance with a log link is equivalent to twice the QLIKE loss, so model
training and final evaluation optimize the same objective. Two variants are compared:
  two_sided     news may raise or lower the forecast;
  amplify_only  negative overlay predictions are clipped to zero.

Feature families come from build_extended_news_features.py. Ridge strength and
the feature family are chosen on 2019-2020 validation only. 2021-2023 is shown
as a confirmation set, but is not claimed to be pristine because earlier
project experiments already inspected that period.

FNSPID coverage differs by symbol and year. Evaluation uses only dates inside
each symbol's observed coverage interval; absence outside that interval is
never treated as zero news.

Outputs:
  reports/news_risk_overlay/results.csv
  reports/news_risk_overlay/coefficients.csv
  reports/news_risk_overlay/report.md
  data/processed/news_risk_overlay_candidate.json

Run:
  python scripts/train_news_risk_overlay.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import TweedieRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_risk_engine import FEATS_BASE, log_feats  # noqa: E402

NEWS_PATH = ROOT / "data" / "processed" / "extended_news_features.parquet"
SPEC_PATH = ROOT / "data" / "processed" / "extended_news_feature_spec.json"
MODEL_PATH = ROOT / "data" / "processed" / "risk_model.json"
PRICES = ROOT / "FNSPID" / "final_dataset" / "prices"
OUT = ROOT / "reports" / "news_risk_overlay"
CANDIDATE = ROOT / "data" / "processed" / "news_risk_overlay_candidate.json"

TRAIN_END = pd.Timestamp("2018-12-01")  # 30-day embargo
VAL_START = pd.Timestamp("2019-01-01")
VAL_END = pd.Timestamp("2020-12-01")    # 30-day embargo
TEST_START = pd.Timestamp("2021-01-01")
HORIZON = 5
MIN_VAL_GAIN = 0.01
ALPHAS = (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)
MODES = ("two_sided", "amplify_only")
MAX_LOG_MULTIPLIER = np.log(2.0)
EPS = 1e-6


def build_price_panel(symbols: list[str], model: dict) -> pd.DataFrame:
    parts = []
    mh = model["horizons"][str(HORIZON)]
    for symbol in symbols:
        px = pd.read_csv(
            PRICES / f"{symbol}.csv",
            usecols=["date", "adj close", "high", "low"],
        )
        px["date"] = pd.to_datetime(px["date"])
        px = px.sort_values("date").set_index("date")
        close, high, low = px["adj close"], px["high"], px["low"]
        ret = close.pct_change()
        park = (np.log(high / low) ** 2) / (4 * np.log(2))
        frame = pd.DataFrame(index=close.index)
        frame["rv5"] = ret.rolling(5).std()
        frame["rv22"] = ret.rolling(22).std()
        frame["rv66"] = ret.rolling(66).std()
        frame["park5"] = np.sqrt(park.rolling(5).mean())
        frame["park22"] = np.sqrt(park.rolling(22).mean())
        frame["absret"] = ret.abs()
        frame["realized_vol_5d"] = ret.rolling(HORIZON).std().shift(-HORIZON)
        frame = log_feats(frame)
        log_sigma = mh["intercept"] + sum(
            frame[name] * mh["coef"][name] for name in mh["features"]
        )
        frame["base_sigma"] = np.exp(log_sigma) * mh["smearing"]
        frame["symbol"] = symbol
        frame.index.name = "date"
        parts.append(frame.reset_index())
    return pd.concat(parts, ignore_index=True)


def feature_families(spec: dict) -> dict[str, list[str]]:
    attention = spec["families"]["attention"]
    sentiment = spec["families"]["sentiment_distribution"]
    source = spec["families"]["source_diffusion_novelty"]
    events = spec["families"]["event_categories"]
    legacy = ["has_news", "log_count", "sent_std"]
    return {
        "legacy": legacy,
        "legacy+sentiment": list(dict.fromkeys(legacy + sentiment)),
        "legacy+source": list(dict.fromkeys(legacy + source)),
        "legacy+events": list(dict.fromkeys(legacy + events)),
        "legacy+source+events": list(
            dict.fromkeys(legacy + source + events)
        ),
        "legacy+expanded": list(
            dict.fromkeys(legacy + sentiment + source + events)
        ),
        "attention": attention,
        "attention+sentiment": list(dict.fromkeys(attention + sentiment)),
        "attention+source": list(dict.fromkeys(attention + source)),
        "attention+events": list(dict.fromkeys(attention + events)),
        "expanded_all": list(
            dict.fromkeys(attention + sentiment + source + events)
        ),
    }


def qlike_loss(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    y2 = np.maximum(np.asarray(realized, float) ** 2, EPS**2)
    f2 = np.maximum(np.asarray(forecast, float) ** 2, EPS**2)
    ratio = y2 / f2
    return ratio - np.log(ratio) - 1


def qlike(realized: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(qlike_loss(realized, forecast)))


def symbol_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("symbol")["symbol"].transform("size").to_numpy()
    weight = 1.0 / counts
    return weight / weight.mean()


def fit_overlay(
    train: pd.DataFrame, features: list[str], alpha: float
) -> tuple[TweedieRegressor, dict[str, float]]:
    scale = (
        train[features]
        .std(ddof=0)
        .replace(0, 1.0)
        .fillna(1.0)
        .to_dict()
    )
    x = train[features].fillna(0.0).div(pd.Series(scale))
    target = np.maximum(
        (
            train["realized_vol_5d"].to_numpy()
            / np.maximum(train["base_sigma"].to_numpy(), EPS)
        )
        ** 2,
        EPS,
    )
    model = TweedieRegressor(
        power=2,
        link="log",
        alpha=alpha,
        fit_intercept=False,
        max_iter=2000,
        tol=1e-8,
    )
    model.fit(
        x,
        target,
        sample_weight=symbol_equal_weights(train),
    )
    return model, {name: float(value) for name, value in scale.items()}


def overlay_prediction(
    model: TweedieRegressor,
    scale: dict[str, float],
    frame: pd.DataFrame,
    features: list[str],
    mode: str,
) -> np.ndarray:
    x = frame[features].fillna(0.0).div(pd.Series(scale))
    variance_ratio = np.clip(
        model.predict(x),
        np.exp(-2 * MAX_LOG_MULTIPLIER),
        np.exp(2 * MAX_LOG_MULTIPLIER),
    )
    prediction = 0.5 * np.log(variance_ratio)
    if mode == "amplify_only":
        return np.clip(prediction, 0.0, MAX_LOG_MULTIPLIER)
    return np.clip(prediction, -MAX_LOG_MULTIPLIER, MAX_LOG_MULTIPLIER)


def dm_test(
    dates: pd.Series,
    overlay_loss: np.ndarray,
    base_loss: np.ndarray,
) -> tuple[float, float]:
    daily = (
        pd.DataFrame(
            {
                "date": dates.to_numpy(),
                "difference": overlay_loss - base_loss,
            }
        )
        .groupby("date")["difference"]
        .mean()
        .dropna()
        .to_numpy()
    )
    n = len(daily)
    variance = np.var(daily, ddof=1)
    for lag in range(1, min(HORIZON, n - 2) + 1):
        covariance = np.cov(daily[lag:], daily[:-lag], ddof=1)[0, 1]
        variance += 2 * (1 - lag / (HORIZON + 1)) * covariance
    standard_error = np.sqrt(max(variance, 1e-18) / n)
    statistic = daily.mean() / standard_error
    p_value = 2 * (1 - stats.norm.cdf(abs(statistic)))
    return float(statistic), float(p_value)


def evaluate_candidate(
    split: pd.DataFrame,
    model: Ridge,
    scale: dict[str, float],
    features: list[str],
    mode: str,
) -> dict:
    overlay = overlay_prediction(model, scale, split, features, mode)
    forecast = split["base_sigma"].to_numpy() * np.exp(overlay)
    realized = split["realized_vol_5d"].to_numpy()
    base = split["base_sigma"].to_numpy()
    base_loss = qlike_loss(realized, base)
    candidate_loss = qlike_loss(realized, forecast)
    dm_t, dm_p = dm_test(split["date"], candidate_loss, base_loss)
    return {
        "qlike": float(candidate_loss.mean()),
        "base_qlike": float(base_loss.mean()),
        "relative_gain": float(
            (base_loss.mean() - candidate_loss.mean()) / base_loss.mean()
        ),
        "dm_t": dm_t,
        "dm_p": dm_p,
        "mean_multiplier": float(np.exp(overlay).mean()),
        "p95_multiplier": float(np.quantile(np.exp(overlay), 0.95)),
        "share_amplified": float((overlay > 1e-12).mean()),
    }


def main() -> None:
    news = pd.read_parquet(NEWS_PATH)
    news["date"] = pd.to_datetime(news["date"])
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    checkpoint = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    families = feature_families(spec)

    panel = build_price_panel(sorted(news["symbol"].unique()), checkpoint)
    panel = panel.merge(news, on=["date", "symbol"], how="inner")
    panel = panel[
        (panel["coverage_active"] == 1)
        & panel["base_sigma"].notna()
        & panel["realized_vol_5d"].notna()
    ].copy()
    panel = panel.replace([np.inf, -np.inf], np.nan)
    all_features = sorted(set(sum(families.values(), [])))
    panel[all_features] = panel[all_features].fillna(0.0)

    train = panel[panel["date"] <= TRAIN_END]
    validation = panel[
        (panel["date"] >= VAL_START) & (panel["date"] <= VAL_END)
    ]
    test = panel[panel["date"] >= TEST_START]
    print(
        f"FNSPID coverage-active panel: train {len(train):,} · "
        f"validation {len(validation):,} · confirmation {len(test):,}"
    )
    print(
        f"symbols: train {train.symbol.nunique()} · "
        f"validation {validation.symbol.nunique()} · test {test.symbol.nunique()}"
    )

    rows = []
    yearly_rows = []
    fitted = {}
    for family_name, features in families.items():
        for mode in MODES:
            best = None
            for alpha in ALPHAS:
                model, scale = fit_overlay(train, features, alpha)
                metrics = evaluate_candidate(
                    validation, model, scale, features, mode
                )
                yearly_validation = {}
                for year in (2019, 2020):
                    year_frame = validation[
                        validation["date"].dt.year == year
                    ]
                    yearly_validation[year] = evaluate_candidate(
                        year_frame, model, scale, features, mode
                    )["relative_gain"]
                candidate = {
                    "family": family_name,
                    "mode": mode,
                    "alpha": alpha,
                    "gain_2019": yearly_validation[2019],
                    "gain_2020": yearly_validation[2020],
                    "min_validation_year_gain": min(
                        yearly_validation.values()
                    ),
                    **metrics,
                }
                if (
                    best is None
                    or candidate["min_validation_year_gain"]
                    > best["min_validation_year_gain"]
                    or (
                        candidate["min_validation_year_gain"]
                        == best["min_validation_year_gain"]
                        and candidate["qlike"] < best["qlike"]
                    )
                ):
                    best = candidate
                    fitted[(family_name, mode)] = (model, scale, alpha)
            rows.append({"split": "VAL", **best})

            model, scale, alpha = fitted[(family_name, mode)]
            metrics = evaluate_candidate(test, model, scale, features, mode)
            rows.append(
                {
                    "split": "TEST_CONFIRMATION",
                    "family": family_name,
                    "mode": mode,
                    "alpha": alpha,
                    **metrics,
                }
            )
            for year in range(2019, 2024):
                year_frame = panel[panel["date"].dt.year == year]
                if year_frame.empty:
                    continue
                year_metrics = evaluate_candidate(
                    year_frame, model, scale, features, mode
                )
                yearly_rows.append(
                    {
                        "family": family_name,
                        "mode": mode,
                        "alpha": alpha,
                        "year": year,
                        "n": len(year_frame),
                        **year_metrics,
                    }
                )

    results = pd.DataFrame(rows)
    validation_results = results[results["split"] == "VAL"]
    best_row = validation_results.loc[
        validation_results["min_validation_year_gain"].idxmax()
    ]
    adopted = (
        best_row["relative_gain"] >= MIN_VAL_GAIN
        and best_row["min_validation_year_gain"] > 0
    )
    best_key = (best_row["family"], best_row["mode"])
    best_model, best_scale, best_alpha = fitted[best_key]
    best_features = families[best_row["family"]]

    coefficients = pd.DataFrame(
        {
            "feature": best_features,
            "scaled_coefficient": best_model.coef_,
            "raw_coefficient": [
                coefficient / best_scale[name]
                for name, coefficient in zip(
                    best_features, best_model.coef_
                )
            ],
        }
    ).sort_values("scaled_coefficient", key=lambda value: value.abs(), ascending=False)

    OUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT / "results.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(OUT / "yearly_results.csv", index=False)
    coefficients.to_csv(OUT / "coefficients.csv", index=False)

    confirmation = results[
        (results["split"] == "TEST_CONFIRMATION")
        & (results["family"] == best_row["family"])
        & (results["mode"] == best_row["mode"])
    ].iloc[0]
    candidate_artifact = {
        "model": "FNSPID FinBERT news risk overlay candidate",
        "status": (
            "research_pass_confirmation_positive"
            if (
                adopted
                and confirmation["relative_gain"] > 0
                and confirmation["dm_p"] < 0.05
            )
            else ("validation_pass" if adopted else "validation_rejected")
        ),
        "horizon": HORIZON,
        "target": "realized_variance_5d / frozen_HAR_variance_5d",
        "train_end": str(TRAIN_END.date()),
        "validation": [str(VAL_START.date()), str(VAL_END.date())],
        "confirmation_start": str(TEST_START.date()),
        "family": best_row["family"],
        "mode": best_row["mode"],
        "alpha": float(best_alpha),
        "min_validation_gain": MIN_VAL_GAIN,
        "validation_gain_2019": float(best_row["gain_2019"]),
        "validation_gain_2020": float(best_row["gain_2020"]),
        "min_validation_year_gain": float(
            best_row["min_validation_year_gain"]
        ),
        "features": best_features,
        "scale": best_scale,
        "coef": {
            name: float(coefficient)
            for name, coefficient in zip(best_features, best_model.coef_)
        },
        "max_log_multiplier": float(MAX_LOG_MULTIPLIER),
        "validation_relative_gain": float(best_row["relative_gain"]),
        "confirmation_relative_gain": float(confirmation["relative_gain"]),
        "confirmation_dm_p": float(confirmation["dm_p"]),
        "warning": (
            "2021-2023 is confirmation-only, not pristine: prior project "
            "experiments already inspected this period."
        ),
    }
    CANDIDATE.write_text(
        json.dumps(candidate_artifact, indent=2), encoding="utf-8"
    )
    write_report(results, best_row, confirmation, adopted, coefficients)

    print("\nVALIDATION-SCREENED NEWS OVERLAY")
    print(
        results[
            [
                "split",
                "family",
                "mode",
                "alpha",
                "qlike",
                "base_qlike",
                "relative_gain",
                "dm_t",
                "dm_p",
                "mean_multiplier",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nSelected on VAL: {best_row['family']} / {best_row['mode']} "
        f"alpha={best_alpha:g} · aggregate gain={best_row['relative_gain']:+.2%} · "
        f"worst-year gain={best_row['min_validation_year_gain']:+.2%} · "
        f"{'PASS' if adopted else 'REJECT'} (threshold {MIN_VAL_GAIN:.1%})"
    )
    print(
        f"Confirmation: gain={confirmation['relative_gain']:+.2%} · "
        f"DM t={confirmation['dm_t']:+.2f}, p={confirmation['dm_p']:.3g}"
    )
    print(f"Outputs -> {OUT.relative_to(ROOT)}")


def write_report(
    results: pd.DataFrame,
    best_validation: pd.Series,
    confirmation: pd.Series,
    adopted: bool,
    coefficients: pd.DataFrame,
) -> None:
    lines = [
        "# FNSPID + FinBERT news risk-overlay research",
        "",
        "The overlay predicts the five-day realized/HAR variance ratio with a",
        "Gamma-log model, whose deviance matches QLIKE. Feature family,",
        "regularization strength, and amplification",
        "constraint are selected on 2019-2020 validation only.",
        "",
        "## Result",
        "",
        f"- Selected family: `{best_validation['family']}`",
        f"- Mode: `{best_validation['mode']}`",
        f"- Validation QLIKE gain: {best_validation['relative_gain']:+.2%}",
        f"- Validation gain in 2019: {best_validation['gain_2019']:+.2%}",
        f"- Validation gain in 2020: {best_validation['gain_2020']:+.2%}",
        f"- Worst validation-year gain: "
        f"{best_validation['min_validation_year_gain']:+.2%}",
        f"- Validation decision: **{'PASS' if adopted else 'REJECT'}** "
        f"(minimum gain {MIN_VAL_GAIN:.1%})",
        f"- 2021-2023 confirmation gain: {confirmation['relative_gain']:+.2%}",
        f"- Confirmation DM: t={confirmation['dm_t']:+.2f}, "
        f"p={confirmation['dm_p']:.3g}",
        "",
        "The confirmation period is not described as pristine because earlier",
        "project experiments had already inspected 2021-2023.",
        "",
        "## Validation screen",
        "",
        "| Family | Mode | Alpha | 2019 gain | 2020 gain | Worst-year | Aggregate gain | DM p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in results[results["split"] == "VAL"].sort_values(
        "relative_gain", ascending=False
    ).iterrows():
        lines.append(
            f"| {row.family} | {row['mode']} | {row.alpha:g} | "
            f"{row.gain_2019:+.2%} | {row.gain_2020:+.2%} | "
            f"{row.min_validation_year_gain:+.2%} | "
            f"{row.relative_gain:+.2%} | {row.dm_p:.3g} |"
        )
    lines += [
        "",
        "## Largest selected coefficients",
        "",
        "| Feature | Scaled coefficient | Raw coefficient |",
        "|---|---:|---:|",
    ]
    for _, row in coefficients.head(15).iterrows():
        lines.append(
            f"| {row.feature} | {row.scaled_coefficient:+.4f} | "
            f"{row.raw_coefficient:+.4f} |"
        )
    lines += [
        "",
        "## Guardrails",
        "",
        "- FNSPID dates are shifted strictly to the next trading session.",
        "- Only symbol-dates inside actual FNSPID coverage are evaluated.",
        "- Intraday timing is excluded because >99% of timestamps are midnight.",
        "- The candidate does not overwrite the deployed HAR checkpoint.",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
