"""Rule Fusion page — one fused decision per holding, with its audit trail.

Strategy + news + health + volatility are never blended into a single score:
each input moves exactly one output dimension, in a fixed order, and every
step appends a row to a confidence ledger (backend/src/rule_fusion/). This
page renders that ledger, not just the final verdict, so a selection can be
traced back to the input that caused it — the same thing
scripts/fusion_selfcheck.py checks by machine.
"""

import pandas as pd
import streamlit as st

import api_client as api
from theme import apply_theme
from views._common import call, portfolio_key

ACTION_ICONS = {"NEW_BUY": "🟢", "ADD": "🔵", "HOLD": "⚪", "TRIM": "🟠", "CLOSE": "🔴"}
DIRECTION_ICONS = {"BUY": "📈", "SELL": "📉", "NEUTRAL": "➖"}
RISK_BAND_ICONS = {"low": "🟢", "moderate": "🟡", "elevated": "🟠", "high": "🔴", "extreme": "🟣"}


@st.cache_data(ttl=900, show_spinner="Fusing strategy, news, health and volatility…")
def _decisions(pkey: str, held_only: bool) -> dict:
    return api.fusion_decisions(held_only=held_only)


@st.cache_data(ttl=3600, show_spinner=False)
def _rules() -> dict:
    return api.fusion_rules()


def _size_text(d: dict) -> str:
    action, size = d["suggested_action"], d["size"]
    if action in ("NEW_BUY", "ADD"):
        return f"+{size['weight_points']:.1f} wt pt"
    if action in ("TRIM", "CLOSE"):
        return f"-{size['trim_fraction']:.0%}"
    return "—"


def _decisions_table(decisions: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": d["symbol"],
                "Held": "✅" if d["inputs"]["held"] else "",
                "Direction": f"{DIRECTION_ICONS.get(d['direction'], '')} {d['direction']}",
                "Action": f"{ACTION_ICONS.get(d['suggested_action'], '')} {d['action_label']}",
                "Confidence": round(d["confidence"] * 100, 1),
                "Risk": f"{RISK_BAND_ICONS.get(d['risk']['band'], '')} {d['risk']['band']}",
                "Size": _size_text(d),
                "News override": "Yes" if d["overridden"] else "",
            }
            for d in decisions
        ]
    )


def _render_ledger(d: dict) -> None:
    st.markdown(
        f"### {DIRECTION_ICONS.get(d['direction'], '')} {d['symbol']} — "
        f"{d['action_label']} "
        f"({d['confidence'] * 100:.0f}% confidence, {d['risk']['band']} risk)"
    )
    st.write(d["explanation"])

    st.markdown("**Step by step:**")
    for a in d["adjustments"]:
        arrow = "▲" if a["delta"] > 0 else ("▼" if a["delta"] < 0 else "–")
        st.markdown(f"**Step {a['step']} · {a['source'].title()}** — {a['observation']} {a['note']}")
        st.caption(
            f"confidence {a['confidence_before'] * 100:.0f}% {arrow} "
            f"{a['confidence_after'] * 100:.0f}%  (Δ {a['delta'] * 100:+.0f} pts)"
        )
    st.caption(
        "Each step above moves exactly one thing — nothing here is averaged, "
        "and those Δ values sum exactly to the final confidence."
    )

    if d["risk"]["drivers"]:
        st.markdown("**Risk band escalated because:**")
        for reason in d["risk"]["drivers"]:
            st.markdown(f"- {reason}")


def render() -> None:
    st.set_page_config(page_title="AURORA — Rule Fusion", page_icon="🧬", layout="wide")
    apply_theme()
    st.title("🧬 Rule Fusion")
    st.caption(
        "One decision per holding, fused from daily strategy, news, portfolio "
        "health and volatility — never averaged into a single score. Pick a row "
        "below to see exactly which input moved it."
    )

    pkey = call(portfolio_key)
    if pkey is None:
        return

    held_only = st.toggle(
        "Only my current holdings",
        value=False,
        help="Off also scans a candidate watchlist, so a NEW_BUY can surface on "
        "something you don't own yet.",
    )

    payload = call(_decisions, pkey, held_only)
    if payload is None:
        return
    decisions = payload["decisions"]
    ctx = payload["context"]

    if not decisions:
        st.info("Nothing scored yet — add a holding or turn off 'Only my current holdings'.")
        return

    # ------------------------------------------------------------- context strip
    regime = ctx.get("regime") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market regime", str(regime.get("regime", "—")).replace("_", " ").title())
    c2.metric(
        "Portfolio health",
        f"{ctx['health_score']:.0f}/100" if ctx.get("health_score") is not None else "—",
    )
    c3.metric("News events scanned", ctx.get("event_count", 0))
    c4.metric("Symbols scanned", len(ctx.get("scanned", [])))
    if "realized_fallback" in (ctx.get("volatility_sources") or []):
        st.caption(
            "⚠️ Some volatility readings fall back to a price-only percentile — "
            "the trained risk model isn't built for every symbol here yet."
        )

    # -------------------------------------------------------------------- table
    st.subheader("Fused decisions")
    st.caption("Most actionable first: closes and trims before adds, then by confidence.")
    table = _decisions_table(decisions)
    picked = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                format="%.0f%%", min_value=0, max_value=100
            ),
        },
    )

    # ---------------------------------------------------------------- drill-down
    st.divider()
    st.subheader("Why this decision")
    st.caption(
        "The fusion engine never blends its four inputs into one score — each "
        "step below moves exactly one thing (direction, confidence, or size) "
        "and records why, so any call traces back to the input that caused it."
    )
    rows = picked.selection.rows if picked is not None else []
    idx = rows[0] if rows else 0
    if not rows:
        st.caption(
            f"Showing the top-ranked decision ({decisions[idx]['symbol']}) by "
            "default — select a different row above to trace that one instead."
        )
    _render_ledger(decisions[idx])

    # -------------------------------------------------------------- rule table
    with st.expander("🛠️ Advanced: raw rule thresholds (for developers)"):
        st.caption(
            "The live GET /fusion/rules table, straight from the engine — not "
            "needed to understand a decision above, but useful if you're "
            "verifying or tuning the rules themselves."
        )
        rules = call(_rules)
        if rules is not None:
            st.json(rules)
