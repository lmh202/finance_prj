"use client";

/** Essential News — Engine 3 (port of frontend/views/news_intelligence.py). */

import { useCallback, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ExternalLink, Loader2, Newspaper, RotateCcw, X } from "lucide-react";
import { Chip, EngineShell, Note, Section, ThinBar } from "@/components/EngineShell";
import { SearchBox } from "@/components/SearchBox";
import { fetchEssentialNews } from "@/lib/api-client";
import { readPortfolio } from "@/lib/portfolio-store";
import { useEngine } from "@/lib/use-engine";
import { fmtDate } from "@/lib/format";
import type { NewsEvent, StockInfo } from "@/lib/types";

// Fetch a generous candidate batch once per tracked-symbol set; the sliders
// below then filter/slice it client-side with no extra backend calls —
// mirrors frontend/views/news_intelligence.py's st.cache_data batching.
const BATCH_SIZE = 20;
const WINDOW_OPTIONS = [6, 12, 24, 48];
const DEFAULT_LOOKBACK_IDX = 2; // WINDOW_OPTIONS[2] = 24h

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
              className="rounded-md border border-line bg-overlay/[0.04] px-1.5 py-0.5 font-mono text-[11px] font-semibold text-accent"
            >
              {s}
            </span>
          ))}
        </div>
      )}

      {event.affected_symbols.length > 0 && (
        <p className="mt-2 text-[12px] leading-relaxed text-mut/80">
          <span className="text-mut/60">Potential impact: </span>
          {event.affected_symbols
            .map((s) => (event.impact[s] ? `${s} (${event.impact[s]})` : s))
            .join(" · ")}
        </p>
      )}
    </motion.article>
  );
}

function TickerChip({ symbol, onRemove }: { symbol: string; onRemove: () => void }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-overlay/[0.04] py-1 pl-3 pr-1.5 font-mono text-[11px] font-semibold tracking-wide text-ink/85">
      {symbol}
      <button
        type="button"
        onClick={onRemove}
        aria-label={`Stop tracking ${symbol}`}
        className="flex size-3.5 items-center justify-center rounded-full text-mut/70 transition-colors hover:bg-loss/15 hover:text-loss"
      >
        <X className="size-3" />
      </button>
    </span>
  );
}

function SliderField({
  label,
  display,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  display: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  const fraction = max === min ? 0 : (value - min) / (max - min);
  return (
    <label className="block">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-mut">{label}</span>
        <span className="font-mono text-[11px] font-semibold tabular text-ink/85">{display}</span>
      </div>
      <input
        type="range"
        className="slider mt-3 w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ "--fill": `${fraction * 100}%` } as React.CSSProperties}
      />
    </label>
  );
}

/** This browser's held symbols, deduped and sorted. */
function portfolioSymbols(): string[] {
  return Array.from(new Set(readPortfolio().holdings.map((h) => h.symbol))).sort();
}

export default function NewsPage() {
  // The tracked-ticker list defaults to this browser's portfolio (resolved
  // lazily inside the fetcher below, on first load only) but is editable in
  // the page body — remove a holding you don't care about, or add any symbol
  // you don't hold just to watch its news.
  const symbolsRef = useRef<string[] | null>(null);
  const [symbols, setSymbols] = useState<string[] | null>(null);
  const [resetting, setResetting] = useState(false);
  // Timestamp the batch was fetched, captured as state (not read from a ref
  // during render) — set once per fetch, then reused by the candidates
  // useMemo below so the age filter stays pure across renders instead of
  // calling Date.now() during render.
  const [fetchedAt, setFetchedAt] = useState(0);

  const [lookbackIdx, setLookbackIdx] = useState(DEFAULT_LOOKBACK_IDX);
  const [minImportance, setMinImportance] = useState(40);
  const [limit, setLimit] = useState(5);
  const lookbackHours = WINDOW_OPTIONS[lookbackIdx];

  // Stable identity on purpose: refetching should only ever happen on mount,
  // on an explicit reload()/applySymbols() call, or the Refresh button —
  // never as a side effect of a symbols-state change, since essential_news()
  // does a live RSS fetch per feed and isn't cheap to double up.
  const fetcher = useCallback(async (signal: AbortSignal) => {
    if (symbolsRef.current === null) {
      // The portfolio is local, so seeding the ticker list costs no request.
      const derived = portfolioSymbols();
      symbolsRef.current = derived;
      setSymbols(derived);
    }
    const events = await fetchEssentialNews(BATCH_SIZE, symbolsRef.current, signal);
    setFetchedAt(Date.now());
    return events;
  }, []);
  const engine = useEngine(fetcher);

  function applySymbols(next: string[]) {
    symbolsRef.current = next;
    setSymbols(next);
    engine.reload();
  }

  function addSymbol(info: StockInfo) {
    const cur = symbolsRef.current ?? [];
    const sym = info.symbol.toUpperCase();
    if (cur.includes(sym)) return;
    applySymbols([...cur, sym].sort());
  }

  function removeSymbol(sym: string) {
    applySymbols((symbolsRef.current ?? []).filter((s) => s !== sym));
  }

  function resetToPortfolio() {
    setResetting(true);
    try {
      applySymbols(portfolioSymbols());
    } finally {
      setResetting(false);
    }
  }

  const candidates = useMemo(() => {
    if (!engine.data) return [];
    return engine.data
      .filter((e) => {
        if (e.published) {
          const ageHours = (fetchedAt - new Date(e.published).getTime()) / 3_600_000;
          if (ageHours > lookbackHours) return false;
        }
        return e.importance >= minImportance;
      })
      .slice(0, limit);
  }, [engine.data, fetchedAt, lookbackHours, minImportance, limit]);

  return (
    <EngineShell
      title="Essential News"
      icon={Newspaper}
      caption="Engine 3 — Event Intelligence · Developer 3"
      engine={engine}
      hasData={engine.data != null}
    >
      {engine.data && (
        <>
          <p className="max-w-2xl text-sm leading-relaxed text-mut">
            Ranked by source credibility, corroboration, relevance to your holdings, severity,
            recency and expected market impact. Informational only — not investment advice.
          </p>

          <Section
            title="Tracking news for"
            action={
              <button
                type="button"
                onClick={resetToPortfolio}
                disabled={resetting}
                className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-mut transition-colors hover:text-accent disabled:pointer-events-none disabled:opacity-50"
              >
                {resetting ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : (
                  <RotateCcw className="size-3" />
                )}
                Reset to portfolio
              </button>
            }
          >
            <div className="flex flex-wrap gap-2">
              {(symbols ?? []).length === 0 ? (
                <span className="text-[13px] text-mut">
                  No tickers selected — showing general market news only.
                </span>
              ) : (
                (symbols ?? []).map((s) => (
                  <TickerChip key={s} symbol={s} onRemove={() => removeSymbol(s)} />
                ))
              )}
            </div>

            <div className="mt-3">
              <SearchBox
                onAdd={addSymbol}
                existing={new Set(symbols ?? [])}
                placeholder="Search a ticker or company name to add…"
              />
            </div>

            <div className="mt-6 grid gap-5 sm:grid-cols-3">
              <SliderField
                label="Lookback window"
                display={`last ${lookbackHours}h`}
                value={lookbackIdx}
                min={0}
                max={WINDOW_OPTIONS.length - 1}
                onChange={setLookbackIdx}
              />
              <SliderField
                label="Minimum importance"
                display={`${minImportance}`}
                value={minImportance}
                min={0}
                max={100}
                onChange={setMinImportance}
              />
              <SliderField
                label="Number of stories"
                display={`${limit}`}
                value={limit}
                min={1}
                max={BATCH_SIZE}
                onChange={setLimit}
              />
            </div>
          </Section>

          {candidates.length === 0 ? (
            <Note tone="mut">
              No stories met the current lookback window and importance threshold.
            </Note>
          ) : (
            <>
              <h3 className="px-1 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                Top {candidates.length} essential event{candidates.length === 1 ? "" : "s"}
              </h3>
              <div className="space-y-4">
                {candidates.map((event, i) => (
                  <EventCard key={`${event.title}-${i}`} event={event} index={i} />
                ))}
              </div>
            </>
          )}

          <p className="pt-1 text-center text-[11px] text-mut/60">
            Headlines remain attributed to their original publishers — follow the source link for
            full context.
          </p>
        </>
      )}
    </EngineShell>
  );
}
