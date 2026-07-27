"""'Should I React?' page — Developer 4 owns everything rendered here."""

import pandas as pd
import streamlit as st

import api_client as api
from theme import apply_theme
from views._common import call, portfolio_key

SUGGESTION_LABELS = {
    "do_nothing": "🟥 Wait — do nothing for now",
    "moderate": "🟨 Moderate adjustment may be considered",
    "aggressive": "🟩 Acting now carries relatively low risk",
}

STRESS_LABELS = {
    "calm": (
        "🟩 Calm",
        "The benchmark's realised volatility is below the stress threshold, so "
        "the optimiser sits further from minimum variance.",
    ),
    "stressed": (
        "🟥 Stressed",
        "Realised volatility is at or above the 75th percentile, so the "
        "optimiser falls back toward minimum variance.",
    ),
    "unknown": (
        "⬜ Unavailable",
        "The market-stress signal could not be computed, so the conservative "
        "stressed-market setting was used.",
    ),
}

DEGRADED_NEWS_QUALITY = {"missing_store", "stale_store", "invalid_store"}


def _ordinal(value: float) -> str:
    """1 -> 1st, 53 -> 53rd, 11 -> 11th."""
    number = int(round(value))
    suffix = (
        "th"
        if 10 <= number % 100 <= 20
        else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    )
    return f"{number}{suffix}"


@st.cache_data(ttl=3600, show_spinner="Computing today's recommendation…")
def _daily(pkey: str) -> dict:
    return api.recommend_daily()


@st.cache_data(ttl=3600, show_spinner="Loading events…")
def _events(pkey: str) -> dict:
    return api.recommendation_events(5)


def _decision_strip(meta: dict) -> None:
    """Portfolio-level risk budget. Hidden on the fallback decision paths,
    which do not produce these numbers."""
    mode = meta.get("production_mode")
    if not mode or mode == "legacy_signal_fallback":
        return

    stress = meta.get("market_stress") or {}
    state = str(stress.get("state", "unknown")).lower()
    label, help_text = STRESS_LABELS.get(state, STRESS_LABELS["unknown"])
    percentile = stress.get("volatility_percentile")

    columns = st.columns(4)
    with columns[0]:
        st.metric(
            "Market state",
            label,
            delta=(
                f"{stress.get('benchmark', 'SPY')} "
                f"{stress.get('volatility_window_sessions', 60)}d vol at the "
                f"{_ordinal(percentile * 100)} percentile"
                if percentile is not None
                else "signal unavailable"
            ),
            delta_color="off",
            help=help_text,
        )
    with columns[1]:
        predicted = meta.get("predicted_annual_volatility")
        target = meta.get("target_annual_volatility")
        st.metric(
            "Predicted volatility",
            f"{predicted:.1%}" if predicted is not None else "—",
            delta=f"target {target:.1%}" if target is not None else None,
            delta_color="off",
            help="Annualised volatility of the proposed portfolio against the "
            "risk budget the health score set.",
        )
    with columns[2]:
        gross = meta.get("target_gross_pct")
        cash_after = meta.get("cash_after_pct")
        locked = meta.get("locked_weight_pct") or 0.0
        locked_names = ", ".join(sorted(meta.get("locked_positions") or {}))
        st.metric(
            "Managed exposure",
            f"{gross:.0f}%" if gross is not None else "—",
            delta=(
                f"{cash_after:.0f}% cash"
                + (f" · {locked:.0f}% unmanaged" if locked > 0.005 else "")
                if cash_after is not None
                else None
            ),
            delta_color="off",
            help="Weight the optimiser controls. The rest is cash plus any "
            "holding it does not manage"
            + (f" ({locked_names})." if locked_names else "."),
        )
    with columns[3]:
        base = meta.get("base_risk_aversion")
        effective = meta.get("effective_risk_aversion")
        st.metric(
            "Risk aversion",
            f"{base:.1f}" if base is not None else "—",
            delta=f"{effective:.1f} after health" if effective is not None else None,
            delta_color="off",
            help="Set by the market state, then scaled by portfolio health. "
            "Higher means closer to minimum variance.",
        )

    facts = [f"Mode: {mode}"]
    if meta.get("model_version"):
        facts.append(meta["model_version"])
    if meta.get("news_via_risk_share") is not None:
        facts.append(f"news reached risk for {meta['news_via_risk_share']:.0%} of holdings")
    if meta.get("strategy_information_coefficient") is not None:
        facts.append(f"strategy IC {meta['strategy_information_coefficient']:.2f}")
    if meta.get("optimizer_success") is False:
        facts.append("⚠ optimiser did not converge — portfolio left unchanged")
    locked_names = ", ".join(sorted(meta.get("locked_positions") or {}))
    if locked_names:
        facts.append(f"not managed: {locked_names} (held as-is, not spendable)")
    st.caption(" · ".join(facts))


def _feed_health_note(meta: dict) -> None:
    """A degraded news store is scored as 'no news', which understates risk.
    Surface it rather than letting a broken feed look like a calm market."""
    degraded = sorted(
        symbol
        for symbol, block in (meta.get("symbols") or {}).items()
        if str(block.get("risk_news_quality", "")).lower() in DEGRADED_NEWS_QUALITY
    )
    if degraded:
        st.warning(
            f"News feed degraded for {', '.join(degraded)} — the risk model "
            "scored these as having no news, so their risk figures are a "
            "lower bound until the feed recovers."
        )


def _trades_table(trades: list) -> None:
    st.dataframe(
        pd.DataFrame(
            [{"Ticker": t["symbol"], "Weight change": f"{t['weight_change_pct']:+.1f}%",
              "Reason": t["reason"]} for t in trades]
        ),
        hide_index=True,
        width="stretch",
    )


def render() -> None:
    st.set_page_config(page_title="AURORA — Should I React?", page_icon="🤔", layout="wide")
    apply_theme()
    st.title("🤔 Should I React?")
    st.caption("Engine 4 — Reaction Risk & Recommendation (Developer 4)")

    pkey = call(portfolio_key)
    if pkey is None:
        return

    # ---------------------------------------------------- daily recommendation
    st.subheader("Today's normal-day recommendation")
    daily = call(_daily, pkey)
    if daily is None:
        return
    rec = daily["recommendation"]
    st.write(rec["explanation"])

    decision_meta = daily.get("decision_meta") or {}
    _decision_strip(decision_meta)
    _feed_health_note(decision_meta)

    fusion_error = (daily.get("explanation_meta") or {}).get("fusion_error")
    if fusion_error:
        st.info(
            "The numeric decision below is valid, but its per-asset explanation "
            f"could not be rendered ({fusion_error})."
        )

    fusion_results = daily.get("fusion_results", [])
    if fusion_results:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Asset": item["symbol"],
                        "AURORA score": float(item["aurora_score"]),
                        "Outlook": item["outlook"],
                        "Risk": item["risk_level"],
                        "Action": item["action"],
                        "Δ weight": float(item.get("position_change_pct", 0.0)),
                        "Confidence": item["confidence_label"],
                        "News": (
                            f"{item['news_articles']} article(s), "
                            f"{item['news_confidence'].lower()} confidence"
                        ),
                        "Flags": " · ".join(
                            [f"stale:{name}" for name in item.get("stale_inputs", [])]
                            + [
                                f"n/a:{name}"
                                for name in item.get("unavailable_inputs", [])
                            ]
                        ),
                    }
                    for item in fusion_results
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "AURORA score": st.column_config.ProgressColumn(
                    "AURORA score",
                    min_value=0,
                    max_value=100,
                    format="%.0f",
                    help="Daily-strategy direction after risk attenuation. "
                    "50 is neutral.",
                ),
                "Δ weight": st.column_config.NumberColumn(
                    "Δ weight",
                    format="%+.2f pp",
                    help="Change in portfolio weight the optimiser proposes.",
                ),
            },
        )
        with st.expander("Why these recommendations?"):
            for item in fusion_results:
                change = item.get("position_change_pct", 0.0)
                st.markdown(
                    f"**{item['symbol']}: {item['outlook']} "
                    f"({item['aurora_score']:.0f}/100) · {change:+.2f}pp**"
                )
                for reason in item["why"]:
                    st.markdown(f"- {reason}")
                titles = item.get("news_titles") or []
                if titles:
                    st.caption("Headlines reviewed: " + " · ".join(titles[:3]))
                as_of = {
                    name: value
                    for name, value in (item.get("as_of") or {}).items()
                    if value
                }
                if as_of:
                    st.caption(
                        "As of — "
                        + ", ".join(f"{name}: {value[:10]}" for name, value in as_of.items())
                    )
    if rec["trades"]:
        _trades_table(rec["trades"])
        st.metric(
            "Portfolio health if applied",
            f"{daily['health_after']:.0f}/100",
            delta=f"{daily['health_after'] - daily['health_before']:+.1f} "
            f"vs current {daily['health_before']:.0f}",
        )

    # ---------------------------------------------------------- event reaction
    st.subheader("React to an event")
    payload = call(_events, pkey)
    if payload is None:
        return
    events = payload["events"]
    if payload["demo"]:
        st.caption(
            "No real events yet (news engine pending — Developer 3). "
            "Showing a demo event so the flow is testable."
        )

    picked = st.selectbox(
        "Event", options=range(len(events)), format_func=lambda i: events[i]["title"]
    )
    event = events[picked]

    reaction = call(api.react, event)
    if reaction is None:
        return
    risk = reaction["risk"]
    event_rec = reaction["recommendation"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Risk of reacting", f"{risk['risk_pct']:.0f}%")
        st.markdown(SUGGESTION_LABELS.get(risk["suggestion"], risk["suggestion"]))
    with c2:
        st.markdown("**Why:**")
        for r in risk["reasons"]:
            st.markdown(f"- {r}")

    st.markdown("**Factor breakdown** (0 = safe to act, 1 = risky):")
    st.dataframe(
        pd.Series(risk["factors"], name="value").round(2),
        width="stretch",
    )

    st.markdown(f"**Recommendation:** {event_rec['explanation']}")
    if event_rec["trades"]:
        _trades_table(event_rec["trades"])
    st.caption("AURORA never executes trades — the final decision is always yours.")
