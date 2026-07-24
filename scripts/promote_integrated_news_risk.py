"""Promote the validated 5-day HAR-News model into the official checkpoint.

The promoted primary risk model is a single joint mapping:

    price HAR features + causal news attention -> 5-day predicted volatility

The news term is a regularised Gamma variance-ratio component.  It is promoted
only because its 2018-2023 nested OOF increment over the same price HAR base is
positive and statistically significant.  The 20-day price-only estimate is
retained as an explicitly auxiliary diagnostic because news did not improve it.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OFFICIAL = PROCESSED / "risk_model.json"
CANDIDATE = PROCESSED / "risk_model_candidate" / "manifest.json"
BACKUP = PROCESSED / "risk_model.pre_har_news_integrated.json"


def atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    official = json.loads(OFFICIAL.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    gate = candidate["gates"]["news_5d"]
    required_historical_checks = {
        name: passed
        for name, passed in gate["checks"].items()
        if name != "rss_shadow_60_mature"
    }
    if not required_historical_checks or not all(
        required_historical_checks.values()
    ):
        raise RuntimeError(
            "5-day news model did not pass every historical contribution gate"
        )
    if gate["aggregate_relative_gain"] <= 0 or gate["dm_p"] >= 0.05:
        raise RuntimeError("5-day news increment is not statistically significant")

    if not BACKUP.exists():
        shutil.copy2(OFFICIAL, BACKUP)

    result = copy.deepcopy(official)
    candidate_horizon = candidate["horizons"]["5"]
    har = candidate_horizon["har"]
    old_horizon = result["horizons"]["5"]
    result["horizons"]["5"] = {
        **old_horizon,
        "features": har["features"],
        "coef": har["coef"],
        "intercept": har["intercept"],
        "smearing": har["smearing"],
        "model_type": "har_news_integrated_gamma",
        "news_features": candidate_horizon["deployable_features"],
        "oof_news_increment": {
            "qlike_relative_gain": gate["aggregate_relative_gain"],
            "bootstrap_95": [
                gate["bootstrap_ci"][0],
                gate["bootstrap_ci"][2],
            ],
            "dm_p": gate["dm_p"],
            "positive_years": sum(
                value > 0 for value in gate["yearly_gains"].values()
            ),
            "positive_symbol_share": gate["positive_symbol_share"],
            "high_vol_relative_gain": gate["high_vol_relative_gain"],
        },
    }
    news_model = copy.deepcopy(
        candidate_horizon["news_linear_deployable"]
    )
    news_model["model_type"] = "linear_gamma_variance_ratio"
    news_model["required_for_primary_output"] = True
    news_model["source_contract"] = "FNSPID offline / RSS live"
    result["news_overlays"] = {"5": news_model}
    result["fhs"]["horizons"]["5"] = candidate_horizon[
        "fhs_by_component"
    ]["har_news_linear_deployable"]

    result["model"] = "HAR-News Integrated Risk Engine"
    result["version"] = "risk-har-news-5d-v1"
    result["created"] = datetime.now(timezone.utc).date().isoformat()
    result["primary_horizon"] = 5
    result["auxiliary_horizon"] = {
        "horizon": 20,
        "model": "price-only HAR-X",
        "reason": "20-day news increment was negative out of sample",
    }
    result["feature_spec"]["news_inputs"] = [
        "log_count",
    ]
    result["feature_spec"]["mapping"] = (
        "price HAR features + causal news attention -> predicted 5-day risk"
    )
    result["news_validation"] = {
        "outer_years": [2018, 2019, 2020, 2021, 2022, 2023],
        "observed_not_pristine": [2021, 2022, 2023],
        "gate": gate,
        "promotion_override": (
            "User requires news to be a core formal input; all historical "
            "significance and tail gates passed."
        ),
    }
    rejected = result.get("rejected_components", {}).get("5", [])
    result["rejected_components"]["5"] = [
        value for value in rejected if value not in {"+news", "+lev+news"}
    ]
    atomic_json(OFFICIAL, result)
    candidate["status"] = "news_5d_promoted"
    candidate["passed_components"] = ["news_5d"]
    candidate["promotion"] = {
        "official_version": result["version"],
        "promoted_utc": datetime.now(timezone.utc).isoformat(),
        "formal_mapping": (
            "price HAR features + causal news attention -> predicted 5-day risk"
        ),
        "decision": (
            "User-required core news input; every historical significance "
            "and tail gate passed."
        ),
    }
    atomic_json(CANDIDATE, candidate)
    print(f"Backup:   {BACKUP}")
    print(f"Official: {OFFICIAL}")
    print(
        "Promoted: 5-day HAR-News integrated Gamma | "
        f"OOF gain={gate['aggregate_relative_gain']:.2%} | "
        f"DM p={gate['dm_p']:.4g}"
    )


if __name__ == "__main__":
    main()
