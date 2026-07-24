"""Benchmark four non-trivial five-day volatility models for presentation.

The script reuses the leakage-safe 2018-2023 outer folds produced by the risk
engine optimisation and adds a neural candidate:

* HAR-X (the frozen price-only reference);
* HAR-X + News (the formal deployable news specification);
* XGBoost Gamma (the nonlinear tree candidate);
* Residual MLP (price + deployable news features).

The neural model predicts a bounded log-volatility residual around a fold-local
HAR forecast. Its architecture is fixed before testing; epoch selection uses
only the final mature year inside each outer training window. The outer test
year is scored once and all random seeds are fixed.

Outputs are written to ``reports/risk_engine_presentation``.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from risk_engine_optimization import (
    CAUSAL_NEWS_STATES,
    DEPLOYABLE_NEWS_BASE,
    OUTER_YEARS,
    PRICE_FEATURES,
    fit_har,
    qlike_loss,
    symbol_equal_weights,
    time_fold,
)

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data" / "processed" / "risk_optimization_panel.parquet"
OOF_PATH = (
    ROOT
    / "reports"
    / "risk_engine_optimization"
    / "oof_predictions.parquet"
)
OUTPUT = ROOT / "reports" / "risk_engine_presentation"
HORIZON = 5
TARGET = "realized_vol_5d"
EPS = 1e-8
BASE_SEED = 20260724

# Small dense layers are faster and more reproducible without 24-way BLAS
# oversubscription on the development workstation.
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

NEWS_FEATURES = list(
    dict.fromkeys(
        DEPLOYABLE_NEWS_BASE
        + CAUSAL_NEWS_STATES
        + ["news_quality_available"]
    )
)
MLP_FEATURES = PRICE_FEATURES + NEWS_FEATURES

MODEL_COLUMNS = {
    "HAR-X": "forecast__current_frozen_har",
    "HAR-X + News": "forecast__har_news_linear_deployable",
    "XGBoost Gamma": "forecast__xgb_price",
    "Residual MLP": "forecast__residual_mlp",
}


@dataclass(frozen=True)
class MLPConfig:
    name: str
    hidden: tuple[int, ...]
    dropout: float
    weight_decay: float
    learning_rate: float = 1e-3


CONFIGS = (
    MLPConfig("mlp_128_64", (128, 64), 0.15, 3e-4),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ResidualMLP(nn.Module):
    """Bounded nonlinear log-volatility adjustment around HAR."""

    def __init__(
        self,
        input_dim: int,
        hidden: tuple[int, ...],
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
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        base_log_sigma: torch.Tensor,
    ) -> torch.Tensor:
        residual = 0.70 * torch.tanh(self.network(features).squeeze(-1))
        return base_log_sigma + residual


def device() -> torch.device:
    # For this small tabular network, large CPU batches are materially faster
    # than repeated Windows/CUDA dispatches while remaining deterministic.
    return torch.device("cpu")


def feature_matrix(
    frame: pd.DataFrame,
    scaler: StandardScaler,
) -> np.ndarray:
    return scaler.transform(frame[MLP_FEATURES]).astype(np.float32)


def har_log_forecast(har, frame: pd.DataFrame) -> np.ndarray:
    return np.log(np.maximum(har.predict(frame), EPS)).astype(np.float32)


def tensors(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    har,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.from_numpy(feature_matrix(frame, scaler))
    base = torch.from_numpy(har_log_forecast(har, frame))
    target = torch.from_numpy(
        np.log(np.maximum(frame[TARGET].to_numpy(float), EPS)).astype(
            np.float32
        )
    )
    weight = torch.from_numpy(
        symbol_equal_weights(frame).astype(np.float32)
    )
    return features, base, target, weight


def qlike_from_logs(
    target_log: torch.Tensor,
    forecast_log: torch.Tensor,
) -> torch.Tensor:
    log_ratio = torch.clamp(
        2.0 * (target_log - forecast_log),
        min=-12.0,
        max=12.0,
    )
    return torch.exp(log_ratio) - log_ratio - 1.0


def weighted_loss(
    target_log: torch.Tensor,
    forecast_log: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    qlike = qlike_from_logs(target_log, forecast_log)
    robust_log_error = nn.functional.smooth_l1_loss(
        forecast_log,
        target_log,
        reduction="none",
        beta=0.25,
    )
    combined = qlike + 0.05 * robust_log_error
    return torch.sum(combined * weight) / torch.sum(weight)


@torch.no_grad()
def predict_logs(
    model: ResidualMLP,
    frame: pd.DataFrame,
    scaler: StandardScaler,
    har,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    features = torch.from_numpy(feature_matrix(frame, scaler))
    base = torch.from_numpy(har_log_forecast(har, frame))
    loader = DataLoader(
        TensorDataset(features, base),
        batch_size=batch_size,
        shuffle=False,
    )
    result = []
    model_device = next(model.parameters()).device
    for batch_features, batch_base in loader:
        prediction = model(
            batch_features.to(model_device),
            batch_base.to(model_device),
        )
        result.append(prediction.cpu().numpy())
    return np.concatenate(result)


def validation_qlike(
    model: ResidualMLP,
    frame: pd.DataFrame,
    scaler: StandardScaler,
    har,
) -> float:
    prediction = np.exp(predict_logs(model, frame, scaler, har))
    loss = qlike_loss(frame[TARGET].to_numpy(float), prediction)
    scored = pd.DataFrame(
        {"symbol": frame["symbol"].to_numpy(), "loss": loss}
    )
    return float(scored.groupby("symbol")["loss"].mean().mean())


def train_with_validation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: MLPConfig,
    seed: int,
    maximum_epochs: int = 60,
    patience: int = 3,
) -> tuple[dict[str, torch.Tensor], int, float]:
    set_seed(seed)
    model_device = device()
    scaler = StandardScaler().fit(train[MLP_FEATURES])
    har = fit_har(train, HORIZON)
    train_tensors = tensors(train, scaler, har)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=8192,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = ResidualMLP(
        len(MLP_FEATURES),
        config.hidden,
        config.dropout,
    ).to(model_device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )
    best_score = math.inf
    best_epoch = 1
    best_state: dict[str, torch.Tensor] = {}
    stale = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        for batch_features, batch_base, batch_target, batch_weight in loader:
            batch_features = batch_features.to(model_device)
            batch_base = batch_base.to(model_device)
            batch_target = batch_target.to(model_device)
            batch_weight = batch_weight.to(model_device)
            optimiser.zero_grad(set_to_none=True)
            forecast = model(batch_features, batch_base)
            loss = weighted_loss(batch_target, forecast, batch_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
        if epoch != 1 and epoch % 5:
            continue
        score = validation_qlike(model, validation, scaler, har)
        scheduler.step(score)
        if score < best_score - 2e-4:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best_state, best_epoch, best_score


def train_fixed_epochs(
    train: pd.DataFrame,
    config: MLPConfig,
    seed: int,
    epochs: int,
) -> tuple[ResidualMLP, StandardScaler, object]:
    set_seed(seed)
    model_device = device()
    scaler = StandardScaler().fit(train[MLP_FEATURES])
    har = fit_har(train, HORIZON)
    train_tensors = tensors(train, scaler, har)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=8192,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = ResidualMLP(
        len(MLP_FEATURES),
        config.hidden,
        config.dropout,
    ).to(model_device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for _ in range(max(epochs, 1)):
        model.train()
        for batch_features, batch_base, batch_target, batch_weight in loader:
            batch_features = batch_features.to(model_device)
            batch_base = batch_base.to(model_device)
            batch_target = batch_target.to(model_device)
            batch_weight = batch_weight.to(model_device)
            optimiser.zero_grad(set_to_none=True)
            forecast = model(batch_features, batch_base)
            loss = weighted_loss(batch_target, forecast, batch_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
    return model, scaler, har


def neural_oof(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    selection_rows = []
    for outer_year in OUTER_YEARS:
        outer_fold = time_fold(panel, outer_year, HORIZON)
        outer_train = panel.loc[outer_fold.train_mask].dropna(
            subset=MLP_FEATURES + [TARGET]
        )
        outer_test = panel.loc[outer_fold.test_mask].dropna(
            subset=MLP_FEATURES + [TARGET]
        )
        validation_year = outer_year - 1
        inner_fold = time_fold(panel, validation_year, HORIZON)
        inner_train = panel.loc[inner_fold.train_mask].dropna(
            subset=MLP_FEATURES + [TARGET]
        )
        inner_validation_mask = (
            panel["date"].dt.year.eq(validation_year).to_numpy()
            & outer_fold.train_mask
        )
        inner_validation = panel.loc[inner_validation_mask].dropna(
            subset=MLP_FEATURES + [TARGET]
        )

        trials = []
        for config_index, config in enumerate(CONFIGS):
            _, best_epoch, score = train_with_validation(
                inner_train,
                inner_validation,
                config,
                seed=BASE_SEED + outer_year * 10 + config_index,
            )
            trials.append((score, best_epoch, config))
            selection_rows.append(
                {
                    "outer_year": outer_year,
                    "validation_year": validation_year,
                    "config": config.name,
                    "hidden": "-".join(map(str, config.hidden)),
                    "dropout": config.dropout,
                    "weight_decay": config.weight_decay,
                    "best_epoch": best_epoch,
                    "validation_qlike": score,
                    "selected": False,
                }
            )
        score, best_epoch, selected = min(trials, key=lambda item: item[0])
        for row in selection_rows:
            if (
                row["outer_year"] == outer_year
                and row["config"] == selected.name
            ):
                row["selected"] = True

        seed_logs = []
        for ensemble_index in range(1):
            model, scaler, har = train_fixed_epochs(
                outer_train,
                selected,
                seed=BASE_SEED + outer_year * 100 + ensemble_index,
                epochs=best_epoch,
            )
            seed_logs.append(
                predict_logs(model, outer_test, scaler, har)
            )
        forecast = np.exp(np.mean(seed_logs, axis=0))
        prediction_rows.append(
            pd.DataFrame(
                {
                    "date": outer_test["date"].to_numpy(),
                    "symbol": outer_test["symbol"].to_numpy(),
                    "test_year": outer_year,
                    TARGET: outer_test[TARGET].to_numpy(float),
                    "forecast__residual_mlp": forecast,
                    "selected_config": selected.name,
                    "selected_epoch": best_epoch,
                    "inner_validation_qlike": score,
                }
            )
        )
        print(
            f"{outer_year}: {selected.name}, epoch={best_epoch}, "
            f"inner QLIKE={score:.6f}, test rows={len(outer_test)}",
            flush=True,
        )
    return (
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(selection_rows),
    )


def equal_weighted_mean(
    frame: pd.DataFrame,
    values: np.ndarray,
) -> float:
    scored = pd.DataFrame(
        {"symbol": frame["symbol"].to_numpy(), "value": values}
    )
    return float(scored.groupby("symbol")["value"].mean().mean())


def newey_west_dm(
    frame: pd.DataFrame,
    candidate_loss: np.ndarray,
    reference_loss: np.ndarray,
) -> tuple[float, float]:
    daily = (
        pd.DataFrame(
            {
                "date": frame["date"].to_numpy(),
                "difference": candidate_loss - reference_loss,
            }
        )
        .groupby("date")["difference"]
        .mean()
        .dropna()
        .to_numpy()
    )
    sample_size = len(daily)
    variance = np.var(daily, ddof=1)
    for lag in range(1, min(HORIZON, sample_size - 2) + 1):
        covariance = np.cov(daily[lag:], daily[:-lag], ddof=1)[0, 1]
        variance += 2 * (1 - lag / (HORIZON + 1)) * covariance
    standard_error = math.sqrt(max(variance, 1e-18) / sample_size)
    statistic = float(np.mean(daily) / standard_error)
    p_value = float(2 * (1 - stats.norm.cdf(abs(statistic))))
    return statistic, p_value


def model_comparison(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = frame[TARGET].to_numpy(float)
    reference_forecast = frame[MODEL_COLUMNS["HAR-X"]].to_numpy(float)
    reference_loss = qlike_loss(target, reference_forecast)
    reference_score = equal_weighted_mean(frame, reference_loss)
    high_cutoff = float(np.quantile(target, 2 / 3))
    high_mask = target >= high_cutoff
    rows = []
    yearly_rows = []
    symbol_rows = []
    for model_name, column in MODEL_COLUMNS.items():
        forecast = frame[column].to_numpy(float)
        loss = qlike_loss(target, forecast)
        qlike = equal_weighted_mean(frame, loss)
        log_error = np.log(np.maximum(target, EPS)) - np.log(
            np.maximum(forecast, EPS)
        )
        weights = symbol_equal_weights(frame)
        log_rmse = float(
            np.sqrt(np.average(log_error**2, weights=weights))
        )
        target_log = np.log(np.maximum(target, EPS))
        prediction_log = np.log(np.maximum(forecast, EPS))
        weighted_mean = float(np.average(target_log, weights=weights))
        denominator = float(
            np.sum(weights * (target_log - weighted_mean) ** 2)
        )
        r2_log = float(
            1
            - np.sum(weights * (target_log - prediction_log) ** 2)
            / denominator
        )
        bias_ratio = float(
            np.average(target / np.maximum(forecast, EPS), weights=weights)
        )
        high_frame = frame.loc[high_mask].reset_index(drop=True)
        high_qlike = equal_weighted_mean(high_frame, loss[high_mask])
        dm_t, dm_p = newey_west_dm(frame, loss, reference_loss)
        symbol_loss = pd.DataFrame(
            {
                "symbol": frame["symbol"].to_numpy(),
                "candidate": loss,
                "reference": reference_loss,
            }
        ).groupby("symbol").mean()
        symbol_gain = (
            symbol_loss["reference"] - symbol_loss["candidate"]
        ) / symbol_loss["reference"]
        rows.append(
            {
                "model": model_name,
                "qlike": qlike,
                "gain_vs_har_x": (reference_score - qlike)
                / reference_score,
                "log_rmse": log_rmse,
                "r2_log": r2_log,
                "realized_to_forecast": bias_ratio,
                "high_vol_qlike": high_qlike,
                "positive_symbol_share_vs_har_x": float(
                    symbol_gain.gt(0).mean()
                ),
                "dm_t_vs_har_x": dm_t,
                "dm_p_vs_har_x": dm_p,
                "rows": len(frame),
                "symbols": int(frame["symbol"].nunique()),
            }
        )
        for symbol, gain in symbol_gain.items():
            symbol_rows.append(
                {
                    "model": model_name,
                    "symbol": symbol,
                    "qlike_gain_vs_har_x": float(gain),
                }
            )
        for year, year_frame in frame.groupby("test_year"):
            year_target = year_frame[TARGET].to_numpy(float)
            year_loss = qlike_loss(
                year_target,
                year_frame[column].to_numpy(float),
            )
            year_reference = qlike_loss(
                year_target,
                year_frame[
                    MODEL_COLUMNS["HAR-X"]
                ].to_numpy(float),
            )
            year_score = equal_weighted_mean(year_frame, year_loss)
            year_reference_score = equal_weighted_mean(
                year_frame, year_reference
            )
            yearly_rows.append(
                {
                    "test_year": int(year),
                    "model": model_name,
                    "qlike": year_score,
                    "gain_vs_har_x": (
                        year_reference_score - year_score
                    )
                    / year_reference_score,
                }
            )
    return (
        pd.DataFrame(rows).sort_values("qlike").reset_index(drop=True),
        pd.DataFrame(yearly_rows),
        pd.DataFrame(symbol_rows),
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL_PATH)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    missing = [feature for feature in MLP_FEATURES if feature not in panel]
    if missing:
        raise ValueError(f"missing MLP features: {missing}")
    neural, selection = neural_oof(panel)
    existing = pd.read_parquet(OOF_PATH)
    existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
    existing = existing[existing["horizon"].eq(HORIZON)].copy()
    keep = [
        "date",
        "symbol",
        "test_year",
        TARGET,
        MODEL_COLUMNS["HAR-X"],
        MODEL_COLUMNS["HAR-X + News"],
        MODEL_COLUMNS["XGBoost Gamma"],
    ]
    comparison_frame = existing[keep].merge(
        neural[
            [
                "date",
                "symbol",
                "forecast__residual_mlp",
                "selected_config",
                "selected_epoch",
            ]
        ],
        on=["date", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    comparison, yearly, symbols = model_comparison(comparison_frame)

    neural.to_parquet(OUTPUT / "neural_oof_predictions.parquet", index=False)
    selection.to_csv(OUTPUT / "neural_model_selection.csv", index=False)
    comparison_frame.to_parquet(
        OUTPUT / "comparison_oof_predictions.parquet",
        index=False,
    )
    comparison.to_csv(OUTPUT / "model_comparison.csv", index=False)
    yearly.to_csv(OUTPUT / "yearly_qlike.csv", index=False)
    symbols.to_csv(OUTPUT / "symbol_results.csv", index=False)

    input_dim = len(MLP_FEATURES)
    parameter_counts = {}
    for config in CONFIGS:
        model = ResidualMLP(input_dim, config.hidden, config.dropout)
        parameter_counts[config.name] = int(
            sum(parameter.numel() for parameter in model.parameters())
        )
    payload = {
        "target": "future five-session daily realised volatility",
        "outer_years": list(OUTER_YEARS),
        "rows": len(comparison_frame),
        "symbols": int(comparison_frame["symbol"].nunique()),
        "price_features": len(PRICE_FEATURES),
        "deployable_news_features": len(NEWS_FEATURES),
        "mlp_input_features": len(MLP_FEATURES),
        "mlp_configs": [asdict(config) for config in CONFIGS],
        "mlp_parameter_counts": parameter_counts,
        "device": str(device()),
        "model_ranking": comparison.to_dict("records"),
    }
    (OUTPUT / "benchmark_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))
    print(f"\nOutputs -> {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
