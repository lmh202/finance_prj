"""Machine-check the rule fusion engine against the rules it claims to follow.

The four-step pipeline in backend/src/rule_fusion/ is only worth trusting if
its promises hold for EVERY input, not just the demo case. Each promise is
restated here as an executable invariant over the whole scenario grid:

    INV-1  health never changes direction
    INV-2  volatility never changes direction or confidence (only size)
    INV-3  only a GATED critical event may change direction away from Step 1
    INV-4  the confidence ledger is well-formed and telescopes to the answer
    INV-5  nothing is averaged: risk.level is the volatility percentile verbatim

Run: python scripts/fusion_selfcheck.py        (exit 0 = all invariants hold)
No network, no market data, no trained model — decide() is pure.
"""

import itertools
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.rule_fusion import engine as fx  # noqa: E402

# --------------------------------------------------------------- the grid

SIGNALS = [("BUY", 2.5), ("SELL", -2.5), ("HOLD", 0.0)]

# (label, sentiment, importance)
NEWS = [
    ("none", 0.0, 0.0),
    ("positive", 0.5, 70.0),
    ("negative", -0.5, 70.0),
    ("neutral", 0.05, 70.0),
]

# (category, keyword, sentiment, importance) — None means no critical story.
# The last two are the gate: a critical hit that is too quiet (importance 40)
# or too ambiguous (sentiment -0.20) must NOT flip the direction.
CRITICAL = [
    None,
    ("interest_rate", "rate hike", -0.70, 80.0),
    ("war_geopolitical", "invasion", -0.65, 90.0),
    ("earnings_corporate", "earnings beat", 0.72, 75.0),
    ("systemic_macro", "bank collapse", -0.80, 95.0),
    ("interest_rate", "rate cut", 0.66, 40.0),      # below the importance gate
    ("systemic_macro", "trading halt", -0.20, 88.0),  # below the sentiment gate
]

HEALTH = [None, 25.0, 55.0, 85.0]
VOLATILITY = [None, 10.0, 40.0, 65.0, 80.0, 95.0]
HELD = [True, False]

HEALTH_SWEEP = [None] + [float(v) for v in range(0, 101, 5)]
VOL_SWEEP = [None] + [float(v) for v in range(0, 101, 5)]


def make(signal, news, critical, health, vol, held, weight_pct=8.0) -> fx.FusionInputs:
    label, sentiment, importance = news
    category = keyword = None
    if critical is not None:
        category, keyword, sentiment, importance = critical
        label = fx.adapters.label_for(sentiment)
    return fx.FusionInputs(
        symbol="TEST",
        held=held,
        weight_pct=weight_pct,
        strategy_signal=signal[0],
        strategy_score=signal[1],
        strategy_reasons=["synthetic case"],
        news_label=label,
        news_sentiment=sentiment,
        news_importance=importance,
        news_headline="synthetic headline",
        critical_category=category,
        critical_keyword=keyword,
        health_score=health,
        volatility_pct=vol,
        volatility_source="synthetic" if vol is not None else "unavailable",
    )


def base_scenarios():
    """Every (signal, news, critical, held) combination — health and volatility
    are swept separately by the invariants that own them."""
    return itertools.product(SIGNALS, NEWS, CRITICAL, HELD)


def full_grid():
    return itertools.product(SIGNALS, NEWS, CRITICAL, HEALTH, VOLATILITY, HELD)


def override_applies(critical) -> bool:
    """Mirror of the engine's gate, written independently so the test does not
    simply restate the implementation."""
    return critical is not None and critical[3] >= fx.CRITICAL_MIN_IMPORTANCE


# ---------------------------------------------------------- the invariants

def inv1_health_never_changes_direction():
    failures = []
    for signal, news, critical, held in base_scenarios():
        seen = set()
        for health in HEALTH_SWEEP:
            d = fx.decide(make(signal, news, critical, health, 50.0, held))
            seen.add(d.direction)
        if len(seen) > 1:
            failures.append(f"{signal[0]}/{news[0]}/{critical}/held={held} -> {sorted(seen)}")
    return failures, len(list(base_scenarios())) * len(HEALTH_SWEEP)


def inv2_volatility_changes_only_size():
    failures = []
    for signal, news, critical, held in base_scenarios():
        directions, confidences, sizes = set(), set(), set()
        for vol in VOL_SWEEP:
            d = fx.decide(make(signal, news, critical, 55.0, vol, held))
            directions.add(d.direction)
            confidences.add(d.confidence)
            sizes.add(d.size.multiplier)
        tag = f"{signal[0]}/{news[0]}/{critical}/held={held}"
        if len(directions) > 1:
            failures.append(f"{tag}: direction moved -> {sorted(directions)}")
        if len(confidences) > 1:
            failures.append(f"{tag}: confidence moved -> {sorted(confidences)}")
        if len(sizes) < 2:
            failures.append(f"{tag}: size never responded to volatility")
    return failures, len(list(base_scenarios())) * len(VOL_SWEEP)


def inv3_only_critical_overrides_direction():
    failures, checked = [], 0
    for signal, news, critical, health, vol, held in full_grid():
        if override_applies(critical):
            continue
        checked += 1
        d = fx.decide(make(signal, news, critical, health, vol, held))
        expected = fx.DIRECTION_FROM_SIGNAL[signal[0]]
        if d.direction != expected or d.overridden:
            failures.append(
                f"{signal[0]}/{news[0]}/{critical}/h={health}/v={vol} -> "
                f"{d.direction} (overridden={d.overridden}), expected {expected}"
            )
    return failures, checked


def inv4_ledger_is_well_formed():
    failures, checked = [], 0
    for signal, news, critical, health, vol, held in full_grid():
        checked += 1
        d = fx.decide(make(signal, news, critical, health, vol, held))
        tag = f"{signal[0]}/{news[0]}/{critical}/h={health}/v={vol}/held={held}"
        if not 0.0 <= d.confidence <= 1.0:
            failures.append(f"{tag}: confidence {d.confidence} out of [0,1]")
        if len(d.adjustments) != 4:
            failures.append(f"{tag}: {len(d.adjustments)} ledger rows, expected 4")
            continue
        if [a.step for a in d.adjustments] != [1, 2, 3, 4]:
            failures.append(f"{tag}: ledger steps out of order")
        if abs(sum(a.delta for a in d.adjustments) - d.confidence) > 1e-9:
            failures.append(f"{tag}: deltas do not telescope to {d.confidence}")
        if d.adjustments[3].delta != 0.0:
            failures.append(f"{tag}: Step 4 moved confidence by {d.adjustments[3].delta}")
        if d.suggested_action not in fx.ACTIONS:
            failures.append(f"{tag}: unknown action {d.suggested_action}")
    return failures, checked


def inv5_risk_level_is_not_blended():
    failures, checked = [], 0
    for signal, news, critical, health, vol, held in full_grid():
        checked += 1
        d = fx.decide(make(signal, news, critical, health, vol, held))
        if d.risk.level != vol:
            failures.append(f"risk.level {d.risk.level} != volatility_pct {vol}")
    return failures, checked


INVARIANTS = [
    ("INV-1  health never changes direction", inv1_health_never_changes_direction),
    ("INV-2  volatility changes size only", inv2_volatility_changes_only_size),
    ("INV-3  only a gated critical event overrides", inv3_only_critical_overrides_direction),
    ("INV-4  confidence ledger is well-formed", inv4_ledger_is_well_formed),
    ("INV-5  risk.level is the percentile verbatim", inv5_risk_level_is_not_blended),
]


# ------------------------------------------------------------ truth table

SHOWCASE = [
    ("strategy BUY, no news, healthy, calm", ("BUY", 2.5), NEWS[0], None, 85.0, 15.0, False),
    ("strategy BUY, news agrees", ("BUY", 2.5), NEWS[1], None, 85.0, 40.0, False),
    ("strategy BUY, news disagrees", ("BUY", 2.5), NEWS[2], None, 55.0, 40.0, True),
    ("strategy BUY, fragile portfolio", ("BUY", 2.5), NEWS[0], None, 25.0, 40.0, True),
    ("strategy BUY, health unknown", ("BUY", 2.5), NEWS[0], None, None, 40.0, True),
    ("strategy BUY, extreme volatility", ("BUY", 3.5), NEWS[1], None, 85.0, 95.0, True),
    ("strategy SELL, held, moderate", ("SELL", -2.5), NEWS[0], None, 55.0, 40.0, True),
    ("strategy SELL, not held", ("SELL", -2.5), NEWS[0], None, 55.0, 40.0, False),
    ("strategy NEUTRAL", ("HOLD", 0.0), NEWS[1], None, 85.0, 40.0, True),
    ("CRITICAL rate hike overrides a BUY", ("BUY", 2.5), NEWS[0], CRITICAL[1], 36.0, 88.0, True),
    ("CRITICAL invasion, not held", ("BUY", 2.5), NEWS[0], CRITICAL[2], 55.0, 60.0, False),
    ("CRITICAL earnings beat overrides a SELL", ("SELL", -2.5), NEWS[0], CRITICAL[3], 85.0, 30.0, True),
    ("CRITICAL bank collapse", ("HOLD", 0.0), NEWS[0], CRITICAL[4], 55.0, 70.0, True),
    ("critical but too quiet (imp 40)", ("BUY", 2.5), NEWS[0], CRITICAL[5], 55.0, 40.0, True),
    ("critical but ambiguous (sent -0.20)", ("BUY", 2.5), NEWS[0], CRITICAL[6], 55.0, 40.0, True),
]


def truth_table() -> None:
    header = f"{'case':<40} {'dir':<8} {'conf':>5}  {'action':<9} {'size':>7}  {'risk':<9} ovr"
    print(header)
    print("-" * len(header))
    for name, signal, news, critical, health, vol, held in SHOWCASE:
        d = fx.decide(make(signal, news, critical, health, vol, held))
        if d.suggested_action in ("NEW_BUY", "ADD"):
            size = f"+{d.size.weight_points:.2f}pt"
        elif d.suggested_action in ("TRIM", "CLOSE"):
            size = f"-{d.size.trim_fraction:.0%}"
        else:
            size = "-"
        print(
            f"{name:<40} {d.direction:<8} {d.confidence:>5.2f}  "
            f"{d.suggested_action:<9} {size:>7}  {d.risk.band:<9} {'YES' if d.overridden else ''}"
        )


def main() -> int:
    print("=" * 78)
    print("RULE FUSION SELF-CHECK")
    print("=" * 78)
    print()
    truth_table()

    print()
    print("=" * 78)
    print("INVARIANTS")
    print("=" * 78)
    failed = 0
    for name, check in INVARIANTS:
        failures, checked = check()
        status = "PASS" if not failures else "FAIL"
        print(f"[{status}] {name:<48} {checked:>6} cases")
        for line in failures[:5]:
            print(f"         {line}")
        if len(failures) > 5:
            print(f"         ... and {len(failures) - 5} more")
        failed += bool(failures)

    print()
    if failed:
        print(f"{failed} invariant(s) VIOLATED — the engine does not obey its own rules.")
        return 1
    print("All invariants hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
