"""Evaluate Daily Strategy vs the new guardrail Fusion from 2024 onward.

The model artifacts and Fusion parameters are frozen. Data before 2024 is used
only as causal warm-up for indicators, expanding stress percentiles, and the
initial portfolio state. No parameter is selected on the evaluation period.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
SCRIPTS = ROOT / "scripts"
for path in (BACKEND, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_wide_price_panel import LEGACY_21  # noqa: E402
from research_adaptive_fusion import (  # noqa: E402
    build_stress_series,
    metrics,
    paired_block_bootstrap,
    run_daily_strategy,
)
from research_guardrail_fusion import run_guardrail  # noqa: E402
from src.daily_strategy import engine as daily_strategy  # noqa: E402
from src.recommendation.gated_news import _runtime_features  # noqa: E402
from src.risk_engine import engine as risk_engine  # noqa: E402

TRADING_DAYS = 252
STEP = 5
WARMUP = 300
TRANSACTION_COST_BPS = 25
BENCHMARK = "SPY"


def normalise_index(frame: pd.DataFrame | pd.Series):
    output = frame.copy()
    output.index = pd.to_datetime(output.index).tz_localize(None)
    return output.sort_index()


def download_ohlc(
    symbols: list[str],
    *,
    start: str,
    end: str,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    requested = symbols + [BENCHMARK]
    raw = yf.download(
        requested,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data")

    ohlc: dict[str, pd.DataFrame] = {}
    missing = []
    for symbol in symbols:
        try:
            frame = raw[symbol][["Open", "High", "Low", "Close"]].dropna()
        except KeyError:
            missing.append(symbol)
            continue
        frame = normalise_index(frame)
        frame.columns = ["open", "high", "low", "close"]
        if len(frame) < WARMUP + 252:
            missing.append(symbol)
            continue
        ohlc[symbol] = frame
    if missing:
        raise RuntimeError(
            "Missing or insufficient OHLC data for: " + ", ".join(missing)
        )

    try:
        benchmark = normalise_index(raw[BENCHMARK]["Close"].dropna())
    except KeyError as exc:
        raise RuntimeError("SPY benchmark data is unavailable") from exc
    return ohlc, benchmark


def build_causal_cache(
    ohlc: dict[str, pd.DataFrame],
    benchmark: pd.Series,
    symbols: list[str],
) -> dict:
    close = pd.DataFrame(
        {symbol: ohlc[symbol]["close"] for symbol in symbols}
    ).dropna(how="any")
    benchmark = benchmark.reindex(close.index).ffill()
    history = close.join(benchmark.rename(BENCHMARK), how="left").ffill()
    returns = close.pct_change().fillna(0.0)
    holdings = pd.DataFrame(
        {
            "symbol": symbols,
            "name": symbols,
            "shares": 1.0,
            "buy_price": 1.0,
        }
    )
    dates = close.index
    cache = {}

    for index in range(WARMUP, len(dates), STEP):
        decision_date = dates[index - 1]
        causal_history = history.iloc[:index]
        signals = daily_strategy.score_assets(causal_history, holdings)
        if len(signals) < 5:
            continue
        features, _ = _runtime_features(
            symbols,
            causal_history,
            signals,
            [],
        )
        estimates = {
            estimate.symbol: estimate
            for estimate in risk_engine.risk_estimates(
                {
                    symbol: ohlc[symbol].loc[:decision_date]
                    for symbol in symbols
                },
                horizons=(5,),
            )
            if estimate.has_history
            and np.isfinite(float(estimate.sigma_daily))
        }
        if set(estimates) != set(symbols):
            continue

        available_benchmark = benchmark.loc[:decision_date]
        benchmark_drawdown = float(
            available_benchmark.iloc[-1]
            / available_benchmark.cummax().iloc[-1]
            - 1.0
        )
        cache[index] = (
            features["strategy_score"]
            .reindex(symbols)
            .fillna(0.0)
            .to_numpy(dtype=float),
            np.asarray(
                [estimates[symbol].sigma_daily for symbol in symbols],
                dtype=float,
            ),
            returns.iloc[max(0, index - TRADING_DAYS) : index],
            benchmark_drawdown,
        )

    if not cache:
        raise RuntimeError("No valid causal rebalance points were generated")
    expected = list(range(min(cache), len(dates), STEP))
    absent = [index for index in expected if index not in cache]
    if absent:
        raise RuntimeError(
            f"{len(absent)} rebalance points lack complete risk estimates"
        )
    return {
        "cache": cache,
        "rets": returns,
        "dates": dates,
        "SYMS": symbols,
        "WARMUP": WARMUP,
        "STEP": STEP,
        "spy": benchmark,
    }


def total_return(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def metric_block(returns: pd.Series) -> dict:
    output = metrics(returns)
    output["total_return"] = total_return(returns)
    output["observations"] = int(len(returns))
    return output


def render_report(payload: dict) -> str:
    overall = payload["overall"]
    daily = overall["daily_strategy"]
    fusion = overall["new_fusion"]
    lines = [
        "# Fusion Forward Evaluation: 2024 to Current",
        "",
        f"- Evaluation: **{payload['evaluation_start']} to "
        f"{payload['evaluation_end']}**",
        f"- Assets: **{len(payload['symbols'])}**",
        "- Rebalance frequency: **5 trading sessions**",
        f"- Transaction cost: **{TRANSACTION_COST_BPS} bps**",
        "- New Fusion: **75% neutral core + 25% Daily Strategy tilt, "
        "projected onto a causal dynamic volatility guardrail**",
        "- Status: **locked forward evaluation; no parameters re-tuned**",
        "",
        "## Overall results",
        "",
        "| Metric | Daily Strategy | New Fusion |",
        "|---|---:|---:|",
        f"| Total return | {daily['total_return']:+.2%} | "
        f"{fusion['total_return']:+.2%} |",
        f"| CAGR | {daily['cagr']:+.2%} | {fusion['cagr']:+.2%} |",
        f"| Sharpe | {daily['sharpe']:.3f} | {fusion['sharpe']:.3f} |",
        f"| Annual volatility | {daily['annual_volatility']:.1%} | "
        f"{fusion['annual_volatility']:.1%} |",
        f"| Maximum drawdown | {daily['max_drawdown']:.1%} | "
        f"{fusion['max_drawdown']:.1%} |",
        f"| Calmar | {daily['calmar']:.2f} | {fusion['calmar']:.2f} |",
        "",
        "## Calendar breakdown",
        "",
        "| Year | Policy | Return | Sharpe | Volatility | Max drawdown |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for year, block in payload["by_year"].items():
        for name, label in (
            ("daily_strategy", "Daily Strategy"),
            ("new_fusion", "New Fusion"),
        ):
            values = block[name]
            lines.append(
                f"| {year} | {label} | {values['total_return']:+.2%} | "
                f"{values['sharpe']:.3f} | "
                f"{values['annual_volatility']:.1%} | "
                f"{values['max_drawdown']:.1%} |"
            )
    bootstrap = payload["paired_bootstrap"]
    interval = bootstrap["bootstrap_delta_sharpe_95"]
    lines.extend(
        [
            "",
            "## Paired uncertainty",
            "",
            f"- New Fusion minus Daily Strategy Sharpe: "
            f"**{bootstrap['point_delta_sharpe']:+.3f}**",
            f"- 95% moving-block interval: **[{interval[0]:+.3f}, "
            f"{interval[2]:+.3f}]**",
            f"- Bootstrap probability of a positive difference: "
            f"**{bootstrap['probability_positive']:.1%}**",
            "",
            "## Scope",
            "",
            "The risk forecast uses the supported no-news historical path "
            "because a complete look-ahead-safe 2024-current news archive was "
            "not supplied to this evaluation. Live news can still affect the "
            "production risk estimate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-start", default="2021-01-01")
    parser.add_argument("--evaluation-start", default="2024-01-01")
    parser.add_argument(
        "--download-end",
        default=(datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat(),
        help="Exclusive yfinance end date.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports" / "fusion_forward_2024_current.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=ROOT / "reports" / "fusion_forward_2024_current.md",
    )
    parser.add_argument(
        "--cache-output",
        type=Path,
        help="Optional path for the causal rebalance cache used by research.",
    )
    args = parser.parse_args()

    symbols = list(LEGACY_21)
    ohlc, benchmark = download_ohlc(
        symbols,
        start=args.download_start,
        end=args.download_end,
    )
    cache_data = build_causal_cache(ohlc, benchmark, symbols)
    if args.cache_output is not None:
        args.cache_output.parent.mkdir(parents=True, exist_ok=True)
        with args.cache_output.open("wb") as handle:
            pickle.dump(cache_data, handle)
    stress = build_stress_series(cache_data)
    daily_returns = run_daily_strategy(cache_data, max_position=0.20)
    fusion_returns, decisions = run_guardrail(
        cache_data,
        stress,
        calm_base_target=0.25,
        stress_base_target=0.15,
        strategy_strength=0.25,
        max_position=0.20,
    )

    evaluation_start = pd.Timestamp(args.evaluation_start)
    daily_returns = daily_returns.loc[daily_returns.index >= evaluation_start]
    fusion_returns = fusion_returns.loc[fusion_returns.index >= evaluation_start]
    aligned = pd.concat(
        [
            daily_returns.rename("daily_strategy"),
            fusion_returns.rename("new_fusion"),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < TRADING_DAYS:
        raise RuntimeError("Forward evaluation has fewer than 252 observations")

    by_year = {}
    for year in sorted(aligned.index.year.unique()):
        block = aligned.loc[aligned.index.year == year]
        by_year[str(year)] = {
            column: metric_block(block[column]) for column in aligned.columns
        }

    relevant_decisions = decisions.loc[
        decisions["date"] >= evaluation_start
    ]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": "Yahoo Finance adjusted OHLC",
        "download_start": args.download_start,
        "download_end_exclusive": args.download_end,
        "evaluation_start": aligned.index.min().date().isoformat(),
        "evaluation_end": aligned.index.max().date().isoformat(),
        "symbols": symbols,
        "configuration": {
            "rebalance_sessions": STEP,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "neutral_core_share": 0.75,
            "daily_strategy_tilt_share": 0.25,
            "calm_base_volatility_target": 0.25,
            "stress_base_volatility_target": 0.15,
            "maximum_position": 0.20,
            "maximum_weekly_change": 0.05,
            "minimum_trade": 0.01,
            "historical_news_path": "supported_no_news_fallback",
        },
        "overall": {
            column: metric_block(aligned[column])
            for column in aligned.columns
        },
        "by_year": by_year,
        "paired_bootstrap": paired_block_bootstrap(
            aligned["daily_strategy"],
            aligned["new_fusion"],
            samples=5000,
            block=20,
            seed=20260726,
        ),
        "fusion_decisions": {
            "count": int(len(relevant_decisions)),
            "mean_base_target": float(
                relevant_decisions["base_target"].mean()
            ),
            "mean_effective_volatility_budget": float(
                relevant_decisions["volatility_budget"].mean()
            ),
            "mean_risky_gross": float(relevant_decisions["gross"].mean()),
            "risk_constraint_binding_share": float(
                relevant_decisions["risk_constraint_binding"].mean()
            ),
            "optimizer_success_rate": float(
                relevant_decisions["success"].mean()
            ),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    args.output_report.write_text(render_report(payload), encoding="utf-8")
    print(args.output_json)
    print(args.output_report)


if __name__ == "__main__":
    main()
