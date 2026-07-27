"use client";

/** Should I React? — Engine 4 (port of frontend/views/recommendation.py). */

import { Fragment, useCallback, useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, ChevronUp, Loader2, RefreshCw, Scale } from "lucide-react";
import {
  ArcGauge,
  Chip,
  EngineShell,
  Metric,
  Note,
  Section,
  ThinBar,
  type ChipTone,
} from "@/components/EngineShell";
import {
  fetchDailyRecommendation,
  fetchRecommendationEvents,
  reactToEvent,
} from "@/lib/api-client";
import { useEngine } from "@/lib/use-engine";
import { fmtNum, fmtPct, signClass } from "@/lib/format";
import type {
  DecisionMeta,
  FusionResult,
  NewsEvent,
  ProposedTrade,
} from "@/lib/types";

const SUGGESTIONS: Record<string, { label: string; color: string; tone: ChipTone }> = {
  do_nothing: { label: "Wait — do nothing for now", color: "var(--color-loss)", tone: "loss" },
  moderate: { label: "A moderate adjustment may be considered", color: "var(--color-warn)", tone: "amber" },
  aggressive: { label: "Acting now carries relatively low risk", color: "var(--color-gain)", tone: "gain" },
};

const FACTOR_LABELS: Record<string, string> = {
  news_uncertainty: "News uncertainty",
  technical_disagreement: "Technical disagreement",
  market_volatility: "Market volatility",
  priced_in: "Already priced in",
  concentration: "Portfolio concentration",
  ambiguity: "Ambiguity",
};

function factorColor(v: number): string {
  if (v < 1 / 3) return "var(--color-gain)";
  if (v < 2 / 3) return "var(--color-warn)";
  return "var(--color-loss)";
}

const STRESS: Record<string, { label: string; tone: ChipTone; sub: string }> = {
  calm: {
    label: "Calm",
    tone: "gain",
    sub: "further from minimum variance",
  },
  stressed: {
    label: "Stressed",
    tone: "loss",
    sub: "falls back toward minimum variance",
  },
  unknown: {
    label: "Unavailable",
    tone: "mut",
    sub: "defaulted to the conservative setting",
  },
};

const RISK_TONE: Record<string, ChipTone> = {
  Low: "gain",
  Moderate: "accent",
  High: "amber",
  Extreme: "loss",
};

const CONFIDENCE_TONE: Record<string, ChipTone> = {
  High: "gain",
  Medium: "amber",
  Low: "mut",
};

/** A degraded store is scored as "no news", which understates risk exactly
 *  when the feed has failed — never let it look like a calm market. */
const DEGRADED_FEED = new Set(["missing_store", "stale_store", "invalid_store"]);

/** 1 -> 1st, 53 -> 53rd, 11 -> 11th. */
function ordinal(value: number): string {
  const n = Math.round(value);
  const suffix =
    n % 100 >= 10 && n % 100 <= 20
      ? "th"
      : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] ?? "th");
  return `${n}${suffix}`;
}

function scoreColor(score: number): string {
  if (score >= 60) return "var(--color-gain)";
  if (score >= 45) return "var(--color-warn)";
  return "var(--color-loss)";
}

function degradedSymbols(meta: DecisionMeta): string[] {
  return Object.entries(meta.symbols ?? {})
    .filter(([, block]) => DEGRADED_FEED.has(block.risk_news_quality ?? ""))
    .map(([symbol]) => symbol)
    .sort();
}

/** Portfolio-level risk budget. The fallback decision paths do not produce
 *  these numbers, so the strip hides itself rather than rendering dashes. */
function DecisionStrip({ meta }: { meta: DecisionMeta }) {
  if (!meta.production_mode || meta.production_mode === "legacy_signal_fallback") {
    return null;
  }
  const stress = meta.market_stress;
  const state = STRESS[stress?.state ?? "unknown"] ?? STRESS.unknown;
  const predicted = meta.predicted_annual_volatility;
  const target = meta.target_annual_volatility;

  return (
    <div className="mt-4 space-y-3">
      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          label="Market state"
          value={state.label}
          tone={
            state.tone === "gain"
              ? "text-gain"
              : state.tone === "loss"
                ? "text-loss"
                : "text-mut"
          }
          sub={
            stress?.volatility_percentile != null
              ? `${stress.benchmark} ${stress.volatility_window_sessions}d vol · ${ordinal(
                  stress.volatility_percentile * 100
                )} percentile`
              : (stress?.unavailable_reason ?? "signal unavailable")
          }
        />
        <div className="card p-4">
          <Metric
            label="Predicted volatility"
            value={predicted != null ? fmtPct(predicted, 1, false) : "—"}
            sub={target != null ? `target ${fmtPct(target, 1, false)}` : undefined}
          />
          {predicted != null && target != null && target > 0 && (
            <ThinBar
              fraction={predicted / target}
              color={predicted > target ? "var(--color-warn)" : "var(--color-gain)"}
              className="mt-2 w-full"
            />
          )}
        </div>
        <Metric
          label="Managed exposure"
          value={meta.target_gross_pct != null ? `${fmtNum(meta.target_gross_pct, 0)}%` : "—"}
          sub={
            meta.cash_after_pct != null
              ? `${fmtNum(meta.cash_after_pct, 0)}% cash${
                  meta.locked_weight_pct
                    ? ` · ${fmtNum(meta.locked_weight_pct, 0)}% unmanaged`
                    : ""
                }`
              : undefined
          }
        />
        <Metric
          label="Risk aversion"
          value={meta.base_risk_aversion != null ? fmtNum(meta.base_risk_aversion, 1) : "—"}
          sub={
            meta.effective_risk_aversion != null
              ? `${fmtNum(meta.effective_risk_aversion, 1)} after health · ${state.sub}`
              : state.sub
          }
        />
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <Chip tone="mut">{meta.production_mode.replace(/_/g, " ")}</Chip>
        {meta.news_via_risk_share != null && (
          <Chip tone="accent">
            news via risk {fmtPct(meta.news_via_risk_share, 0, false)}
          </Chip>
        )}
        {meta.strategy_information_coefficient != null && (
          <Chip tone="mut">IC {fmtNum(meta.strategy_information_coefficient, 2)}</Chip>
        )}
        {meta.optimizer_success === false && (
          <Chip tone="loss">optimiser did not converge</Chip>
        )}
        {meta.locked_positions &&
          Object.keys(meta.locked_positions).length > 0 && (
            <Chip tone="amber">
              not managed: {Object.keys(meta.locked_positions).sort().join(", ")}
            </Chip>
          )}
      </div>
    </div>
  );
}

function FusionTable({ results, meta }: { results: FusionResult[]; meta: DecisionMeta }) {
  const [open, setOpen] = useState<string | null>(null);

  return (
    <div className="scroll-slim overflow-x-auto">
      <table className="w-full min-w-[820px] border-collapse">
        <thead>
          <tr className="border-b border-line text-left font-mono text-[10px] uppercase tracking-[0.16em] text-mut">
            <th className="px-3 py-2.5 font-medium">Asset</th>
            <th className="px-3 py-2.5 text-right font-medium">AURORA score</th>
            <th className="px-3 py-2.5 font-medium">Outlook</th>
            <th className="px-3 py-2.5 font-medium">Risk</th>
            <th className="px-3 py-2.5 font-medium">Action</th>
            <th className="px-3 py-2.5 text-right font-medium">Δ weight</th>
            <th className="px-3 py-2.5 font-medium">Confidence</th>
            <th className="px-3 py-2.5 font-medium">News</th>
            <th className="px-3 py-2.5" />
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => {
            const expanded = open === r.symbol;
            const asOf = Object.entries(r.as_of).filter(([, v]) => v);
            return (
              <Fragment key={r.symbol}>
              <motion.tr
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.03 * i }}
                className="border-b border-overlay/[0.04] align-top transition-colors last:border-0 hover:bg-overlay/[0.025]"
              >
                <td className="px-3 py-3 font-mono text-sm font-semibold text-accent">
                  {r.symbol}
                </td>
                <td className="px-3 py-3">
                  <div className="flex items-center justify-end gap-2">
                    <ThinBar
                      fraction={r.aurora_score / 100}
                      color={scoreColor(r.aurora_score)}
                      className="w-14"
                    />
                    <span className="w-7 text-right font-mono text-xs font-medium tabular text-ink/90">
                      {Math.round(r.aurora_score)}
                    </span>
                  </div>
                </td>
                <td className="px-3 py-3 text-xs text-ink/85">{r.outlook}</td>
                <td className="px-3 py-3">
                  <Chip tone={RISK_TONE[r.risk_level] ?? "mut"}>{r.risk_level}</Chip>
                </td>
                <td className="px-3 py-3 text-xs text-ink/85">{r.action}</td>
                <td
                  className={`px-3 py-3 text-right font-mono text-xs font-medium tabular ${signClass(
                    r.position_change_pct
                  )}`}
                >
                  {r.position_change_pct > 0 ? "+" : ""}
                  {fmtNum(r.position_change_pct, 2)}pp
                </td>
                <td className="px-3 py-3">
                  <Chip tone={CONFIDENCE_TONE[r.confidence_label] ?? "mut"}>
                    {r.confidence_label}
                  </Chip>
                </td>
                <td className="px-3 py-3 text-xs text-mut">
                  {r.news_articles} · {r.news_confidence.toLowerCase()}
                </td>
                <td className="px-3 py-3 text-right">
                  <button
                    onClick={() => setOpen(expanded ? null : r.symbol)}
                    aria-label={expanded ? `Hide why ${r.symbol}` : `Why ${r.symbol}?`}
                    aria-expanded={expanded}
                    className="text-mut transition-colors hover:text-accent"
                  >
                    {expanded ? (
                      <ChevronUp className="size-3.5" />
                    ) : (
                      <ChevronDown className="size-3.5" />
                    )}
                  </button>
                </td>
              </motion.tr>
              {expanded && (
                <tr className="border-b border-overlay/[0.04]">
                  <td colSpan={9} className="bg-overlay/[0.02] px-3 py-4">
                    <ul className="space-y-1.5">
                      {r.why.map((reason, index) => (
                        <li
                          key={index}
                          className="flex gap-2 text-xs leading-relaxed text-mut"
                        >
                          <span className="mt-[7px] size-1 shrink-0 rounded-full bg-accent" />
                          <span>{reason}</span>
                        </li>
                      ))}
                    </ul>
                    {r.news_titles.length > 0 && (
                      <p className="mt-3 text-[11px] leading-relaxed text-mut/80">
                        Headlines reviewed: {r.news_titles.slice(0, 3).join(" · ")}
                      </p>
                    )}
                    <div className="mt-3 flex flex-wrap items-center gap-1.5">
                      {r.stale_inputs.map((name) => (
                        <Chip key={`stale-${name}`} tone="amber">
                          stale: {name}
                        </Chip>
                      ))}
                      {r.unavailable_inputs.map((name) => (
                        <Chip key={`na-${name}`} tone="loss">
                          n/a: {name}
                        </Chip>
                      ))}
                      {DEGRADED_FEED.has(
                        meta.symbols?.[r.symbol]?.risk_news_quality ?? ""
                      ) && <Chip tone="amber">feed degraded</Chip>}
                    </div>
                    {asOf.length > 0 && (
                      <p className="mt-3 font-mono text-[10px] uppercase tracking-wider text-mut/70">
                        As of{" "}
                        {asOf
                          .map(([name, value]) => `${name} ${String(value).slice(0, 10)}`)
                          .join(" · ")}
                      </p>
                    )}
                  </td>
                </tr>
              )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TradesTable({ trades }: { trades: ProposedTrade[] }) {
  return (
    <div className="scroll-slim overflow-x-auto">
      <table className="w-full min-w-[420px] border-collapse">
        <thead>
          <tr className="border-b border-line text-left font-mono text-[10px] uppercase tracking-[0.16em] text-mut">
            <th className="px-3 py-2.5 font-medium">Ticker</th>
            <th className="px-3 py-2.5 text-right font-medium">Weight change</th>
            <th className="px-3 py-2.5 font-medium">Reason</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((t) => (
            <tr
              key={`${t.symbol}-${t.weight_change_pct}`}
              className="border-b border-overlay/[0.04] last:border-0"
            >
              <td className="px-3 py-2.5 font-mono text-sm font-semibold text-accent">
                {t.symbol}
              </td>
              <td
                className={`px-3 py-2.5 text-right font-mono text-xs font-medium tabular ${signClass(
                  t.weight_change_pct
                )}`}
              >
                {t.weight_change_pct > 0 ? "+" : ""}
                {t.weight_change_pct.toFixed(1)}%
              </td>
              <td className="px-3 py-2.5 text-xs leading-relaxed text-mut">{t.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EventReaction({ events, demo }: { events: NewsEvent[]; demo: boolean }) {
  const [picked, setPicked] = useState(0);
  const event = events[Math.min(picked, events.length - 1)];

  const fetcher = useCallback(
    (signal: AbortSignal) => reactToEvent(event, signal),
    [event]
  );
  const reaction = useEngine(fetcher);
  const data = reaction.data;
  const suggestion = data ? SUGGESTIONS[data.risk.suggestion] : null;

  return (
    <div className="space-y-4">
      {demo && (
        <Note>
          No real events yet (news engine pending — Developer 3). Showing a demo
          event so the flow is testable.
        </Note>
      )}

      <div className="relative">
        <select
          value={picked}
          onChange={(e) => setPicked(Number(e.target.value))}
          aria-label="Event to react to"
          className="w-full cursor-pointer appearance-none rounded-xl border border-line bg-overlay/[0.04] py-2.5 pl-3.5 pr-10 text-sm text-ink outline-none transition-colors focus:border-accent/50 [&>option]:bg-panel"
        >
          {events.map((ev, i) => (
            <option key={`${ev.title}-${i}`} value={i}>
              {ev.title}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute right-3.5 top-1/2 size-4 -translate-y-1/2 text-mut" />
      </div>

      {reaction.status.phase === "error" || reaction.status.phase === "marker" ? (
        <Note>
          Could not evaluate this event
          {reaction.status.phase === "error" ? ` — ${reaction.status.message}` : ""}.{" "}
          <button onClick={reaction.reload} className="underline underline-offset-2 hover:text-ink">
            Retry
          </button>
        </Note>
      ) : data == null ? (
        <div className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
            <div className="skeleton h-52 rounded-2xl" />
            <div className="skeleton h-52 rounded-2xl" />
          </div>
          <div className="skeleton h-64 rounded-2xl" />
        </div>
      ) : (
        <div className="relative space-y-4">
          {reaction.loading && (
            <div className="absolute -top-1 right-0 z-10 flex items-center gap-1.5 rounded-full border border-accent/30 bg-bg0/90 px-3 py-1 font-mono text-[10px] uppercase tracking-wider text-accent backdrop-blur">
              <Loader2 className="size-3 animate-spin" />
              Updating
            </div>
          )}

          <div className="grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
            <Section title="Risk of reacting" className="flex flex-col">
              <div className="flex flex-1 flex-col items-center justify-center gap-4 py-2">
                <ArcGauge
                  value={data.risk.risk_pct}
                  color={suggestion?.color ?? "var(--color-warn)"}
                  display={`${Math.round(data.risk.risk_pct)}%`}
                  caption="0 = safe · 100 = risky"
                />
                <Chip tone={suggestion?.tone ?? "mut"}>
                  {suggestion?.label ?? data.risk.suggestion}
                </Chip>
              </div>
            </Section>
            <Section title="Why">
              <ul className="space-y-2.5">
                {data.risk.reasons.length === 0 ? (
                  <li className="text-sm text-mut">—</li>
                ) : (
                  data.risk.reasons.map((r) => (
                    <li key={r} className="flex items-start gap-2.5 text-sm leading-relaxed text-ink/90">
                      <span className="mt-[7px] size-1 shrink-0 rounded-full bg-accent" />
                      {r}
                    </li>
                  ))
                )}
              </ul>
            </Section>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Section
              title="Factor breakdown"
              action={
                <span className="font-mono text-[10px] uppercase tracking-wider text-mut/70">
                  0 safe · 1 risky
                </span>
              }
            >
              <div className="space-y-3">
                {Object.entries(data.risk.factors).map(([key, v]) => (
                  <div key={key} className="flex items-center gap-3">
                    <span className="w-44 shrink-0 font-mono text-[11px] text-mut">
                      {FACTOR_LABELS[key] ?? key.replace(/_/g, " ")}
                    </span>
                    <ThinBar fraction={v} color={factorColor(v)} className="flex-1" />
                    <span className="w-9 text-right font-mono text-xs tabular text-ink/85">
                      {fmtNum(v, 2)}
                    </span>
                  </div>
                ))}
              </div>
            </Section>
            <Section title="Recommendation">
              <p className="text-sm leading-relaxed text-ink/90">
                {data.recommendation.explanation}
              </p>
              {data.recommendation.trades.length > 0 && (
                <div className="mt-4">
                  <TradesTable trades={data.recommendation.trades} />
                </div>
              )}
            </Section>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ReactPage() {
  const fetcher = useCallback(async (signal: AbortSignal) => {
    const [daily, events] = await Promise.all([
      fetchDailyRecommendation(signal),
      fetchRecommendationEvents(5, signal),
    ]);
    return { daily, events };
  }, []);
  const engine = useEngine(fetcher);
  const data = engine.data;

  return (
    <EngineShell
      title="Should I React?"
      icon={Scale}
      caption="Engine 4 — Reaction Risk & Recommendation · Developer 4"
      engine={engine}
      hasData={data != null}
    >
      {data && (
        <>
          <Section title="Today's normal-day recommendation">
            <p className="text-sm leading-relaxed text-ink/90">
              {data.daily.recommendation.explanation}
            </p>
            <DecisionStrip meta={data.daily.decision_meta ?? {}} />
            {degradedSymbols(data.daily.decision_meta ?? {}).length > 0 && (
              <div className="mt-3">
                <Note tone="amber">
                  News feed degraded for{" "}
                  {degradedSymbols(data.daily.decision_meta).join(", ")} — the risk
                  model scored these as having no news, so their risk figures are a
                  lower bound until the feed recovers.
                </Note>
              </div>
            )}
            {data.daily.recommendation.trades.length > 0 && (
              <div className="mt-4 space-y-4">
                <TradesTable trades={data.daily.recommendation.trades} />
                {data.daily.health_after != null && data.daily.health_before != null && (
                  <div className="max-w-xs">
                    <Metric
                      label="Portfolio health if applied"
                      value={`${Math.round(data.daily.health_after)}/100`}
                      tone={signClass(data.daily.health_after - data.daily.health_before)}
                      sub={`${data.daily.health_after - data.daily.health_before >= 0 ? "+" : ""}${fmtNum(
                        data.daily.health_after - data.daily.health_before,
                        1
                      )} vs current ${Math.round(data.daily.health_before)}`}
                    />
                  </div>
                )}
              </div>
            )}
          </Section>

          {/*<Section title="Per-asset decision">
            {data.daily.fusion_results?.length > 0 ? (
              <FusionTable
                results={data.daily.fusion_results}
                meta={data.daily.decision_meta ?? {}}
              />
            ) : (
              <Note tone="mut">
                No per-asset explanation was produced for this run
                {data.daily.explanation_meta?.fusion_error
                  ? ` (${data.daily.explanation_meta.fusion_error})`
                  : ""}
                . The recommendation above is still valid — explanation is
                presentational and never blocks the decision.
              </Note>
            )}
          </Section>*/}

          <Section
            title="React to an event"
            action={
              <button onClick={engine.reload} className="text-mut transition-colors hover:text-accent" title="Reload events">
                <RefreshCw className="size-3.5" />
              </button>
            }
          >
            <EventReaction events={data.events.events} demo={data.events.demo} />
          </Section>

          <p className="text-center font-mono text-[10px] uppercase tracking-[0.18em] text-mut/70">
            AURORA never executes trades — the final decision is always yours
          </p>
        </>
      )}
    </EngineShell>
  );
}
