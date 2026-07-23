"use client";

/** Should I React? — Engine 4 (port of frontend/views/recommendation.py). */

import { useCallback, useState } from "react";
import { ChevronDown, Loader2, RefreshCw, Scale } from "lucide-react";
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
import { fmtNum, signClass } from "@/lib/format";
import type { NewsEvent, ProposedTrade } from "@/lib/types";

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
