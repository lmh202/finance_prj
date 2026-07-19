"use client";

/** Essential News — Engine 3 (port of frontend/views/news_intelligence.py). */

import { useCallback } from "react";
import { motion } from "framer-motion";
import { ExternalLink, Newspaper, Rss } from "lucide-react";
import { Chip, EngineShell, Section, ThinBar } from "@/components/EngineShell";
import { fetchEssentialNews, fetchNewsFeeds } from "@/lib/api-client";
import { useEngine } from "@/lib/use-engine";
import { fmtDate } from "@/lib/format";
import type { NewsEvent } from "@/lib/types";

function sentimentChip(sentiment: number) {
  const label = `${sentiment > 0 ? "+" : ""}${sentiment.toFixed(2)}`;
  if (sentiment > 0.15) return { tone: "gain" as const, label };
  if (sentiment < -0.15) return { tone: "loss" as const, label };
  return { tone: "mut" as const, label };
}

function EventCard({ event, index }: { event: NewsEvent; index: number }) {
  const sent = sentimentChip(event.sentiment);
  return (
    <motion.article
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.06 * index, ease: [0.16, 1, 0.3, 1] }}
      className="card p-5"
    >
      <div className="flex flex-wrap items-center gap-2">
        <Chip tone="accent">{event.category.replace(/_/g, " ")}</Chip>
        <Chip tone={sent.tone}>sentiment {sent.label}</Chip>
        <div className="ml-auto flex items-center gap-2 font-mono text-[10px] uppercase tracking-wider text-mut">
          <span>importance</span>
          <ThinBar fraction={event.importance / 100} className="w-14" />
          <span className="tabular text-ink/85">{Math.round(event.importance)}/100</span>
        </div>
      </div>

      <h2 className="mt-3 text-[15px] font-semibold leading-snug tracking-tight">
        {event.url ? (
          <a
            href={event.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-start gap-1.5 transition-colors hover:text-accent"
          >
            {event.title}
            <ExternalLink className="mt-1 size-3 shrink-0 text-mut" />
          </a>
        ) : (
          event.title
        )}
      </h2>

      <div className="mt-1.5 font-mono text-[10px] uppercase tracking-wider text-mut/80">
        {event.source}
        {event.published && <> · {fmtDate(event.published.slice(0, 10), true)}</>}
      </div>

      {event.summary && (
        <p className="mt-3 text-[13px] leading-relaxed text-mut">{event.summary}</p>
      )}

      {event.affected_symbols.length > 0 && (
        <div className="mt-3.5 flex flex-wrap items-center gap-1.5">
          <span className="font-mono text-[10px] uppercase tracking-wider text-mut/70">
            Affected holdings
          </span>
          {event.affected_symbols.map((s) => (
            <span
              key={s}
              className="rounded-md border border-line bg-white/[0.04] px-1.5 py-0.5 font-mono text-[11px] font-semibold text-accent"
            >
              {s}
            </span>
          ))}
        </div>
      )}
    </motion.article>
  );
}

export default function NewsPage() {
  const fetcher = useCallback(async (signal: AbortSignal) => {
    const [events, feeds] = await Promise.all([
      fetchEssentialNews(5, signal),
      fetchNewsFeeds(signal),
    ]);
    return { events, feeds };
  }, []);
  const engine = useEngine(fetcher);
  const data = engine.data;

  return (
    <EngineShell
      title="Essential News"
      icon={Newspaper}
      caption="Engine 3 — Event Intelligence · Developer 3"
      engine={engine}
      hasData={data != null}
    >
      {data &&
        (data.events.length === 0 ? (
          <Section title="Planned feeds">
            <p className="text-sm leading-relaxed text-mut">
              The news engine is not implemented yet — this page comes alive when
              Developer 3 ships{" "}
              <code className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[11px] text-ink/90">
                backend/src/news_intelligence/engine.py
              </code>
              . These are the planned RSS sources:
            </p>
            <ul className="mt-4 space-y-2">
              {data.feeds.map((feed) => (
                <li key={feed} className="flex items-center gap-2.5 font-mono text-xs text-ink/80">
                  <Rss className="size-3 shrink-0 text-amber-200/80" />
                  <span className="truncate">{feed}</span>
                </li>
              ))}
            </ul>
          </Section>
        ) : (
          <div className="space-y-4">
            {data.events.map((event, i) => (
              <EventCard key={`${event.title}-${i}`} event={event} index={i} />
            ))}
          </div>
        ))}
    </EngineShell>
  );
}
