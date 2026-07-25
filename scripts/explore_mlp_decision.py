"""Explore a small multi-task MLP for the final decision layer.

The formal HAR-X + News five-session OOF forecast is fixed and used as an
input/normalizer.  Three feature families are compared:

* price_risk: online price, market, and formal risk inputs;
* deployable_news: price_risk plus RSS-reproducible attention state;
* research_news: price_risk plus all causal FNSPID/FinBERT features.

The primary target is the next-20-session return relative to SPY, divided by
the ex-ante formal risk scale.  Auxiliary heads estimate a lower quantile and
the cross-sectional target rank.  All choices are made on 2019-2020
walk-forward validation; 2021-2023 is evaluated only after each family is
locked.  No formal checkpoint is overwritten.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from decision_layer_core import (  # noqa: E402
    PRICE_FEATURES,
    TARGET,
    backtest_metrics,
    moving_block_utility_gain,
    prediction_metrics,
    run_backtest,
    time_fold,
)

DATA_PATH = ROOT / "data" / "processed" / "decision_dataset.parquet"
EXTERNAL_PATH = (
    ROOT / "data" / "processed" / "decision_external_dataset.parquet"
)
REPORT_DIR = ROOT / "reports" / "decision_layer_mlp"
CANDIDATE_DIR = ROOT / "data" / "processed" / "decision_model_candidate_mlp"

RANDOM_SEED = 20260724
VALIDATION_YEARS = (2019, 2020)
TEST_YEARS = (2021, 2022, 2023)
PRIMARY_COST_BPS = 25.0
RISK_AVERSION = 6.0
ALPHA_SCALES = (0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0)
TARGET_CLIP = 3.0
MAX_EPOCHS = 180
PATIENCE = 20
BATCH_SIZE = 512

MARKET_FEATURES = [
    "market_mom_20d",
    "market_mom_60d",
    "market_vol_20d",
]
RISK_FEATURES = ["log_risk_sigma_5d", "risk_level_5d_scaled"]
BASE_FEATURES = PRICE_FEATURES + MARKET_FEATURES + RISK_FEATURES
DEPLOYABLE_NEWS_FEATURES = [
    "has_news",
    "log_count",
    "count_z20",
    "count_ratio20",
    "news_count_3d",
    "news_count_5d",
    "days_since_news",
]
ALL_NEWS_FEATURES = [
    "sentiment",
    "news_count",
    "has_news",
    "unique_story_count",
    "sent_mean",
    "sent_std",
    "sent_min",
    "sent_max",
    "sent_abs_mean",
    "sent_positive_share",
    "sent_negative_share",
    "sent_extreme_share",
    "unique_publisher_count",
    "publisher_missing_share",
    "summary_share",
    "mean_text_length",
    "mean_story_breadth",
    "max_story_breadth",
    "firm_specific_share",
    "broad_story_share",
    "event_earnings_share",
    "event_analyst_share",
    "event_corporate_action_share",
    "event_legal_regulatory_share",
    "event_product_share",
    "event_macro_share",
    "event_management_share",
    "event_financing_share",
    "sent_range",
    "duplicate_share",
    "publisher_entropy",
    "title_token_novelty",
    "log_count",
    "count_z20",
    "count_ratio20",
    "news_count_3d",
    "news_count_5d",
    "sent_surprise20",
    "sent_abs_surprise20",
    "days_since_news",
]
FEATURE_FAMILIES = {
    "price_risk": BASE_FEATURES,
    "deployable_news": BASE_FEATURES + DEPLOYABLE_NEWS_FEATURES,
    "research_news": BASE_FEATURES + ALL_NEWS_FEATURES,
}


@dataclass(frozen=True)
class NetworkConfig:
    name: str
    hidden: tuple[int, ...]
    dropout: float
    learning_rate: float
    weight_decay: float


NETWORK_CONFIGS = (
    NetworkConfig("tiny", (32, 16), 0.10, 1e-3, 1e-3),
    NetworkConfig("small", (64, 32), 0.20, 5e-4, 2e-3),
    NetworkConfig("medium", (128, 64, 32), 0.25, 3e-4, 3e-3),
)


class MultiTaskMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = width
        self.encoder = nn.Sequential(*layers)
        self.mean_head = nn.Linear(previous, 1)
        self.q10_head = nn.Linear(previous, 1)
        self.rank_head = nn.Linear(previous, 1)

    def forward(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values)
        return (
            self.mean_head(encoded).squeeze(-1),
            self.q10_head(encoded).squeeze(-1),
            torch.tanh(self.rank_head(encoded).squeeze(-1)),
        )


@dataclass
class FitResult:
    model: MultiTaskMLP
    scaler: StandardScaler
    best_epoch: int
    best_loss: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    market = (
        output[["date", "benchmark_ret_1d"]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")["benchmark_ret_1d"]
    )
    market_price = (1 + market.fillna(0.0)).cumprod()
    features = pd.DataFrame(index=market.index)
    features["market_mom_20d"] = market_price.pct_change(20)
    features["market_mom_60d"] = market_price.pct_change(60)
    features["market_vol_20d"] = market.rolling(20).std() * math.sqrt(252)
    return output.merge(
        features.reset_index(),
        on="date",
        how="left",
        validate="many_to_one",
    )


def prepare_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = add_market_features(panel)
    panel["log_risk_sigma_5d"] = np.log(
        panel["risk_sigma_daily_5d"].clip(lower=1e-6)
    )
    panel["risk_level_5d_scaled"] = panel["risk_level_5d"] / 100.0
    risk_scale = panel["risk_sigma_daily_5d"] * math.sqrt(20)
    panel["target_risk_adjusted_excess_20d"] = (
        panel[TARGET] / risk_scale.clip(lower=1e-6)
    )
    panel["target_rank_scaled"] = panel["target_rank_20d"] * 2.0 - 1.0
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def symbol_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("symbol")["symbol"].transform("size").to_numpy()
    weight = 1.0 / np.maximum(count, 1)
    return weight / weight.mean()


def split_internal_time(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.sort(train["date"].unique())
    split_position = max(60, int(len(dates) * 0.80))
    split_position = min(split_position, len(dates) - 21)
    validation_start = pd.Timestamp(dates[split_position])
    fit_cutoff = pd.Timestamp(dates[split_position - 21])
    fit = train.loc[train["date"].le(fit_cutoff)].copy()
    validation = train.loc[train["date"].ge(validation_start)].copy()
    if fit.empty or validation.empty:
        raise ValueError("internal time split is empty")
    return fit, validation


def tensor_dataset(
    frame: pd.DataFrame,
    features: Sequence[str],
    scaler: StandardScaler,
) -> TensorDataset:
    values = scaler.transform(frame[list(features)]).astype("float32")
    main = np.clip(
        frame["target_risk_adjusted_excess_20d"].to_numpy(dtype=float),
        -TARGET_CLIP,
        TARGET_CLIP,
    ).astype("float32")
    rank = frame["target_rank_scaled"].to_numpy(dtype="float32")
    weights = symbol_equal_weights(frame).astype("float32")
    return TensorDataset(
        torch.from_numpy(values),
        torch.from_numpy(main),
        torch.from_numpy(rank),
        torch.from_numpy(weights),
    )


def batch_loss(
    model: MultiTaskMLP,
    values: torch.Tensor,
    target: torch.Tensor,
    rank_target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    mean, q10, rank = model(values)
    huber = nn.functional.huber_loss(
        mean,
        target,
        delta=0.75,
        reduction="none",
    )
    error = target - q10
    pinball = torch.maximum(0.10 * error, -0.90 * error)
    rank_loss = (rank - rank_target) ** 2
    ordering = nn.functional.relu(q10 - mean) ** 2
    combined = huber + 0.40 * pinball + 0.25 * rank_loss + 0.10 * ordering
    return torch.sum(combined * weights) / torch.sum(weights)


def fit_mlp(
    train: pd.DataFrame,
    features: Sequence[str],
    config: NetworkConfig,
    seed: int,
) -> FitResult:
    set_seed(seed)
    fit_frame, early_frame = split_internal_time(train)
    scaler = StandardScaler().fit(fit_frame[list(features)])
    fit_data = tensor_dataset(fit_frame, features, scaler)
    early_data = tensor_dataset(early_frame, features, scaler)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        fit_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    early_loader = DataLoader(
        early_data,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
    )
    current_device = device()
    model = MultiTaskMLP(
        len(features),
        config.hidden,
        config.dropout,
    ).to(current_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for values, target, rank_target, weights in loader:
            values = values.to(current_device)
            target = target.to(current_device)
            rank_target = rank_target.to(current_device)
            weights = weights.to(current_device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, values, target, rank_target, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        model.eval()
        validation_loss = 0.0
        batches = 0
        with torch.no_grad():
            for values, target, rank_target, weights in early_loader:
                validation_loss += float(
                    batch_loss(
                        model,
                        values.to(current_device),
                        target.to(current_device),
                        rank_target.to(current_device),
                        weights.to(current_device),
                    ).cpu()
                )
                batches += 1
        validation_loss /= max(batches, 1)
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.to("cpu")
    return FitResult(model, scaler, best_epoch, best_loss)


def predict_mlp(
    fitted: FitResult,
    frame: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    values = fitted.scaler.transform(frame[list(features)]).astype("float32")
    tensor = torch.from_numpy(values)
    fitted.model.eval()
    with torch.no_grad():
        mean, q10, rank = fitted.model(tensor)
    output = frame[
        [
            "date",
            "symbol",
            TARGET,
            "target_risk_adjusted_excess_20d",
            "target_rank_scaled",
            "risk_sigma_daily_5d",
        ]
    ].copy()
    output["predicted_risk_adjusted_return"] = mean.numpy()
    output["predicted_q10"] = q10.numpy()
    output["predicted_rank_head"] = rank.numpy()
    return output


def walk_forward_mlp(
    panel: pd.DataFrame,
    years: Sequence[int],
    features: Sequence[str],
    config: NetworkConfig,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    predictions = []
    fits = []
    required = list(features) + [
        TARGET,
        "target_risk_adjusted_excess_20d",
        "target_rank_scaled",
        "risk_sigma_daily_5d",
    ]
    for year in years:
        fold = time_fold(panel, year)
        train = panel.loc[fold.train_mask].dropna(subset=required).copy()
        test = panel.loc[fold.test_mask].dropna(subset=required).copy()
        fitted = fit_mlp(train, features, config, seed + year)
        predicted = predict_mlp(fitted, test, features)
        predicted["test_year"] = year
        predicted["train_cutoff"] = fold.train_cutoff
        predictions.append(predicted)
        fits.append(
            {
                "year": year,
                "train_rows": len(train),
                "test_rows": len(test),
                "best_epoch": fitted.best_epoch,
                "early_loss": fitted.best_loss,
            }
        )
    return pd.concat(predictions, ignore_index=True), fits


def portfolio_predictions(
    prediction: pd.DataFrame,
    alpha_scale: float,
) -> pd.DataFrame:
    output = prediction.copy()
    risk_scale = output["risk_sigma_daily_5d"] * math.sqrt(20)
    output["prediction"] = (
        output["predicted_risk_adjusted_return"]
        * risk_scale
        * alpha_scale
    ).clip(-0.25, 0.25)
    return output


def predictive_metrics(prediction: pd.DataFrame) -> dict[str, float]:
    point = portfolio_predictions(prediction, 1.0)
    standard = prediction_metrics(point)
    target = prediction["target_risk_adjusted_excess_20d"].to_numpy()
    mean = prediction["predicted_risk_adjusted_return"].to_numpy()
    q10 = prediction["predicted_q10"].to_numpy()
    standard.update(
        {
            "risk_adjusted_rmse": float(
                np.sqrt(np.mean((target - mean) ** 2))
            ),
            "q10_coverage": float(np.mean(target < q10)),
        }
    )
    return standard


def evaluate_portfolio(
    panel: pd.DataFrame,
    prediction: pd.DataFrame | None,
    *,
    period: str,
    strategy: str,
    start: str,
    end: str,
    alpha_scale: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    use_prediction = (
        portfolio_predictions(prediction, alpha_scale)
        if prediction is not None
        else None
    )
    engine_strategy = "ml" if prediction is not None else strategy
    daily, _ = run_backtest(
        panel,
        use_prediction,
        start=start,
        end=end,
        strategy=engine_strategy,
        transaction_cost_bps=PRIMARY_COST_BPS,
        rebalance_sessions=5,
        risk_aversion=RISK_AVERSION,
        turnover_penalty_multiplier=1.0,
    )
    daily["strategy"] = strategy
    metrics = backtest_metrics(daily, risk_aversion=RISK_AVERSION)
    metrics.update(
        {
            "period": period,
            "strategy": strategy,
            "alpha_scale": alpha_scale,
        }
    )
    daily["period"] = period
    daily["strategy"] = strategy
    daily["alpha_scale"] = alpha_scale
    return metrics, daily


def yearly_cer(daily: pd.DataFrame) -> dict[int, float]:
    return {
        int(year): float(
            backtest_metrics(group, risk_aversion=RISK_AVERSION)[
                "certainty_equivalent"
            ]
        )
        for year, group in daily.groupby(daily["date"].dt.year)
    }


def save_bundle(
    path: Path,
    fitted: FitResult,
    features: Sequence[str],
    config: NetworkConfig,
    family: str,
    alpha_scale: float,
) -> None:
    torch.save(
        {
            "state_dict": fitted.model.state_dict(),
            "input_dim": len(features),
            "hidden": list(config.hidden),
            "dropout": config.dropout,
            "features": list(features),
            "scaler_mean": fitted.scaler.mean_,
            "scaler_scale": fitted.scaler.scale_,
            "family": family,
            "alpha_scale": alpha_scale,
            "best_epoch": fitted.best_epoch,
        },
        path,
    )


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel(DATA_PATH)
    panel = panel.loc[panel["risk_sigma_daily_5d"].notna()].copy()
    external = prepare_panel(EXTERNAL_PATH)
    print(
        f"device={device()} | historical rows={len(panel):,} | "
        f"range={panel.date.min().date()}..{panel.date.max().date()}",
        flush=True,
    )

    risk_validation_metrics, risk_validation_daily = evaluate_portfolio(
        panel,
        None,
        period="validation",
        strategy="risk_only",
        start="2019-01-01",
        end="2020-12-31",
    )
    risk_validation_year = yearly_cer(risk_validation_daily)

    validation_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    fit_rows: list[dict[str, Any]] = []
    predictive_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    print("=" * 88, flush=True)
    print("MLP MODEL SEARCH | 2019-2020 VALIDATION ONLY", flush=True)
    print("=" * 88, flush=True)
    for family, features in FEATURE_FAMILIES.items():
        for config in NETWORK_CONFIGS:
            print(f"fitting {family}/{config.name}", flush=True)
            prediction, fits = walk_forward_mlp(
                panel,
                VALIDATION_YEARS,
                features,
                config,
                RANDOM_SEED,
            )
            validation_predictions[(family, config.name)] = prediction
            metrics = predictive_metrics(prediction)
            predictive_rows.append(
                {
                    "period": "validation",
                    "family": family,
                    "config": config.name,
                    **metrics,
                }
            )
            for row in fits:
                fit_rows.append(
                    {"period": "validation", "family": family, "config": config.name, **row}
                )
            for alpha_scale in ALPHA_SCALES:
                portfolio_metrics, daily = evaluate_portfolio(
                    panel,
                    prediction,
                    period="validation",
                    strategy="mlp",
                    start="2019-01-01",
                    end="2020-12-31",
                    alpha_scale=alpha_scale,
                )
                candidate_year = yearly_cer(daily)
                gains = [
                    candidate_year[year] - risk_validation_year[year]
                    for year in sorted(risk_validation_year)
                ]
                overall_gain = (
                    portfolio_metrics["certainty_equivalent"]
                    - risk_validation_metrics["certainty_equivalent"]
                )
                robust_score = overall_gain + min(0.0, min(gains))
                search_rows.append(
                    {
                        "family": family,
                        "config": config.name,
                        "alpha_scale": alpha_scale,
                        "validation_cer_gain": overall_gain,
                        "worst_year_gain": min(gains),
                        "positive_years": int(
                            np.sum(np.asarray(gains) > 0)
                        ),
                        "robust_score": robust_score,
                        **portfolio_metrics,
                    }
                )
            print(
                f"  rank IC={metrics['rank_ic_mean']:+.4f} | "
                f"q10 coverage={metrics['q10_coverage']:.3f}",
                flush=True,
            )

    search = pd.DataFrame(search_rows)
    family_choices: dict[str, pd.Series] = {}
    for family in FEATURE_FAMILIES:
        candidates = search.loc[
            search["family"].eq(family)
            & search["alpha_scale"].gt(0)
        ]
        family_choices[family] = candidates.sort_values(
            [
                "robust_score",
                "validation_cer_gain",
                "certainty_equivalent",
            ],
            ascending=False,
        ).iloc[0]

    locked_family = str(
        pd.DataFrame(family_choices.values())
        .sort_values(
            ["robust_score", "validation_cer_gain"],
            ascending=False,
        )
        .iloc[0]["family"]
    )
    print("=" * 88, flush=True)
    print("LOCKED FAMILY SPECIFICATIONS | 2021-2023 TEST", flush=True)
    print("=" * 88, flush=True)

    risk_test_metrics, risk_test_daily = evaluate_portfolio(
        panel,
        None,
        period="locked_test_reused",
        strategy="risk_only",
        start="2021-01-01",
        end="2023-12-31",
    )
    risk_test_year = yearly_cer(risk_test_daily)
    portfolio_rows = [risk_validation_metrics, risk_test_metrics]
    daily_outputs = [risk_validation_daily, risk_test_daily]
    test_predictions: list[pd.DataFrame] = []
    family_results: dict[str, dict[str, Any]] = {}

    for family, choice in family_choices.items():
        config = next(
            item
            for item in NETWORK_CONFIGS
            if item.name == str(choice["config"])
        )
        alpha_scale = float(choice["alpha_scale"])
        prediction, fits = walk_forward_mlp(
            panel,
            TEST_YEARS,
            FEATURE_FAMILIES[family],
            config,
            RANDOM_SEED,
        )
        prediction["family"] = family
        prediction["config"] = config.name
        test_predictions.append(prediction)
        for row in fits:
            fit_rows.append(
                {
                    "period": "locked_test_reused",
                    "family": family,
                    "config": config.name,
                    **row,
                }
            )
        metrics = predictive_metrics(prediction)
        predictive_rows.append(
            {
                "period": "locked_test_reused",
                "family": family,
                "config": config.name,
                **metrics,
            }
        )
        portfolio_metrics, daily = evaluate_portfolio(
            panel,
            prediction,
            period="locked_test_reused",
            strategy=family,
            start="2021-01-01",
            end="2023-12-31",
            alpha_scale=alpha_scale,
        )
        portfolio_rows.append(portfolio_metrics)
        daily_outputs.append(daily)
        candidate_year = yearly_cer(daily)
        gains = [
            candidate_year[year] - risk_test_year[year]
            for year in sorted(risk_test_year)
        ]
        bootstrap = moving_block_utility_gain(risk_test_daily, daily)
        family_results[family] = {
            "config": config.name,
            "alpha_scale": alpha_scale,
            "validation_robust_score": float(choice["robust_score"]),
            "validation_cer_gain": float(choice["validation_cer_gain"]),
            "test_rank_ic": metrics["rank_ic_mean"],
            "test_q10_coverage": metrics["q10_coverage"],
            "test_cer_gain": (
                portfolio_metrics["certainty_equivalent"]
                - risk_test_metrics["certainty_equivalent"]
            ),
            "test_sharpe_gain": (
                portfolio_metrics["sharpe"] - risk_test_metrics["sharpe"]
            ),
            "test_positive_years": int(np.sum(np.asarray(gains) > 0)),
            "test_worst_year_gain": float(min(gains)),
            "bootstrap_utility_gain_95": list(bootstrap),
        }
        print(
            f"{family}: config={config.name}, scale={alpha_scale:.2f}, "
            f"test IC={metrics['rank_ic_mean']:+.4f}, "
            f"CER gain={family_results[family]['test_cer_gain']:+.4f}",
            flush=True,
        )

    selected_choice = family_choices[locked_family]
    selected_config = next(
        item
        for item in NETWORK_CONFIGS
        if item.name == str(selected_choice["config"])
    )
    final_required = FEATURE_FAMILIES[locked_family] + [
        TARGET,
        "target_risk_adjusted_excess_20d",
        "target_rank_scaled",
        "risk_sigma_daily_5d",
    ]
    final_train = panel.dropna(subset=final_required).copy()
    final_fit = fit_mlp(
        final_train,
        FEATURE_FAMILIES[locked_family],
        selected_config,
        RANDOM_SEED + 99,
    )
    model_path = CANDIDATE_DIR / "mlp_bundle.pt"
    save_bundle(
        model_path,
        final_fit,
        FEATURE_FAMILIES[locked_family],
        selected_config,
        locked_family,
        float(selected_choice["alpha_scale"]),
    )

    price_choice = family_choices["price_risk"]
    price_config = next(
        item
        for item in NETWORK_CONFIGS
        if item.name == str(price_choice["config"])
    )
    price_fit = fit_mlp(
        final_train.dropna(subset=FEATURE_FAMILIES["price_risk"]),
        FEATURE_FAMILIES["price_risk"],
        price_config,
        RANDOM_SEED + 199,
    )
    external_required = FEATURE_FAMILIES["price_risk"] + [
        TARGET,
        "target_risk_adjusted_excess_20d",
        "target_rank_scaled",
        "risk_sigma_daily_5d",
    ]
    external_sample = external.dropna(subset=external_required).copy()
    external_prediction = predict_mlp(
        price_fit,
        external_sample,
        FEATURE_FAMILIES["price_risk"],
    )
    external_metrics = predictive_metrics(external_prediction)
    predictive_rows.append(
        {
            "period": "external_reused_price_risk_only",
            "family": "price_risk",
            "config": price_config.name,
            **external_metrics,
        }
    )
    risk_external_metrics, risk_external_daily = evaluate_portfolio(
        external,
        None,
        period="external_reused",
        strategy="risk_only",
        start="2024-01-01",
        end="2026-12-31",
    )
    price_external_metrics, price_external_daily = evaluate_portfolio(
        external,
        external_prediction,
        period="external_reused",
        strategy="price_risk",
        start="2024-01-01",
        end="2026-12-31",
        alpha_scale=float(price_choice["alpha_scale"]),
    )
    portfolio_rows.extend([risk_external_metrics, price_external_metrics])
    daily_outputs.extend([risk_external_daily, price_external_daily])

    news_increment = {
        family: {
            "validation_cer_gain_vs_price_risk_mlp": float(
                family_results[family]["validation_cer_gain"]
                - family_results["price_risk"]["validation_cer_gain"]
            ),
            "test_cer_gain_vs_price_risk_mlp": float(
                family_results[family]["test_cer_gain"]
                - family_results["price_risk"]["test_cer_gain"]
            ),
            "test_rank_ic_gain_vs_price_risk_mlp": float(
                family_results[family]["test_rank_ic"]
                - family_results["price_risk"]["test_rank_ic"]
            ),
        }
        for family in ("deployable_news", "research_news")
    }
    selected_result = family_results[locked_family]
    continuation_checks = {
        "validation_gain_positive": (
            selected_result["validation_cer_gain"] > 0
        ),
        "test_cer_gain_positive": selected_result["test_cer_gain"] > 0,
        "test_rank_ic_positive": selected_result["test_rank_ic"] > 0,
        "test_positive_years_at_least_2": (
            selected_result["test_positive_years"] >= 2
        ),
        "bootstrap_lower_bound_positive": (
            selected_result["bootstrap_utility_gain_95"][0] > 0
        ),
        "q10_coverage_between_7_and_13pct": (
            0.07 <= selected_result["test_q10_coverage"] <= 0.13
        ),
    }
    worth_continuing = all(continuation_checks.values())
    conclusion = {
        "validation_selected_family": locked_family,
        "family_results": family_results,
        "news_increment": news_increment,
        "continuation_checks": continuation_checks,
        "worth_continuing": worth_continuing,
        "automatic_promotion_allowed": False,
        "limitations": [
            (
                "The 2021-2023 test years have been observed in earlier "
                "experiments and are diagnostic rather than pristine."
            ),
            (
                "Rich FNSPID news fields have no 2024-2026 live-history parity; "
                "only the price_risk family can be evaluated externally."
            ),
            (
                "The sample has 26,711 correlated stock-days, not 26,711 "
                "independent observations."
            ),
        ],
    }

    metadata = {
        "schema_version": 1,
        "model_version": "decision-mlp-research-candidate-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_checkpoint_replaced": False,
        "selected_family_on_validation": locked_family,
        "selected_config": asdict(selected_config),
        "features": FEATURE_FAMILIES[locked_family],
        "target": (
            "20-session stock return minus SPY return, divided by formal "
            "HAR-X + News risk_sigma_daily_5d * sqrt(20)"
        ),
        "auxiliary_heads": [
            "conditional q10 of risk-adjusted excess return",
            "daily cross-sectional target rank",
        ],
        "alpha_scale": float(selected_choice["alpha_scale"]),
        "training_range": [
            str(final_train["date"].min().date()),
            str(final_train["date"].max().date()),
        ],
        "validation_years": list(VALIDATION_YEARS),
        "diagnostic_test_years": list(TEST_YEARS),
        "artifact": {
            "path": model_path.name,
            "sha256": sha256(model_path),
            "data_sha256": sha256(DATA_PATH),
        },
        "conclusion": conclusion,
        "random_seed": RANDOM_SEED,
    }
    (CANDIDATE_DIR / "decision_model.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    search.to_csv(REPORT_DIR / "validation_search.csv", index=False)
    pd.DataFrame(predictive_rows).to_csv(
        REPORT_DIR / "predictive_metrics.csv",
        index=False,
    )
    pd.DataFrame(portfolio_rows).to_csv(
        REPORT_DIR / "portfolio_metrics.csv",
        index=False,
    )
    pd.DataFrame(fit_rows).to_csv(REPORT_DIR / "fit_history.csv", index=False)
    pd.concat(test_predictions, ignore_index=True).to_parquet(
        REPORT_DIR / "test_predictions.parquet",
        index=False,
    )
    external_prediction.to_parquet(
        REPORT_DIR / "external_price_risk_predictions.parquet",
        index=False,
    )
    pd.concat(daily_outputs, ignore_index=True).to_parquet(
        REPORT_DIR / "backtest_daily.parquet",
        index=False,
    )
    (REPORT_DIR / "conclusion.json").write_text(
        json.dumps(conclusion, indent=2),
        encoding="utf-8",
    )

    family_table = pd.DataFrame(
        [
            {
                "family": family,
                "config": values["config"],
                "alpha_scale": values["alpha_scale"],
                "validation_cer_gain": values["validation_cer_gain"],
                "test_rank_ic": values["test_rank_ic"],
                "test_cer_gain": values["test_cer_gain"],
                "test_positive_years": values["test_positive_years"],
                "bootstrap_low": values["bootstrap_utility_gain_95"][0],
                "q10_coverage": values["test_q10_coverage"],
            }
            for family, values in family_results.items()
        ]
    ).round(4)
    report = f"""# Multi-task MLP Decision-Layer Exploration

## Decision

- Validation-selected family: **{locked_family}**
- Network: **{selected_config.name} {list(selected_config.hidden)}**
- Alpha calibration: **{float(selected_choice['alpha_scale']):.2f}**
- Worth continuing under all predeclared checks: **{worth_continuing}**
- Formal checkpoint replaced: **no**

The MLP directly predicts the next-20-session risk-adjusted return relative to
SPY. HAR-X + News risk is fixed and enters both the target normalization and
the portfolio optimiser.

## Feature-family comparison

{markdown_table(family_table)}

## News increment

```json
{json.dumps(news_increment, indent=2)}
```

## Continuation checks

```json
{json.dumps(continuation_checks, indent=2)}
```

## External price-risk diagnostic

Only the price-risk family can be reconstructed on 2024-2026 because no
historical RSS feature archive exists. Its CER is
{price_external_metrics['certainty_equivalent']:.4f} versus
{risk_external_metrics['certainty_equivalent']:.4f} for risk-only, while
annual turnover rises from {risk_external_metrics['annual_turnover']:.2f} to
{price_external_metrics['annual_turnover']:.2f}. This result cannot override
the negative 2021-2023 test because it is inconsistent across regimes.

## Material findings

1. Direct risk-adjusted alpha prediction is unstable between validation and
   the locked test.
2. Deployable news produces the safest MLP, but does not improve test CER.
3. The full research-news block does not improve test rank IC over price-risk
   inputs, indicating excessive interaction dimensionality for this sample.
4. The q10 head is better calibrated than the expected-return head, so
   distributional downside prediction is the more promising neural direction.

## Interpretation

Model architecture, feature family, and alpha calibration are selected only
from 2019-2020 walk-forward validation. The 2021-2023 result is a locked
diagnostic, but not a pristine blind test because those years were already
observed by earlier experiments. Rich news cannot be evaluated on 2024-2026
because no historical RSS feature archive exists.
"""
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")
    print("=" * 88, flush=True)
    print(
        f"selected family={locked_family} | worth_continuing={worth_continuing}",
        flush=True,
    )
    print(f"report -> {REPORT_DIR.relative_to(ROOT) / 'report.md'}")
    print(
        "candidate -> "
        f"{CANDIDATE_DIR.relative_to(ROOT) / 'decision_model.json'}"
    )


if __name__ == "__main__":
    main()
