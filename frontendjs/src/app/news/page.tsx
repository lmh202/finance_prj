"use client";

/** Essential News — Engine 3 (port of frontend/views/news_intelligence.py). */

import { useCallback, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ExternalLink, Loader2, Newspaper, RotateCcw, X } from "lucide-react";
import { Chip, EngineShell, Note, Section, ThinBar } from "@/components/EngineShell";
import { SearchBox } from "@/components/SearchBox";
import { fetchEssentialNews, fetchPortfolio } from "@/lib/api-client";
import { useEngine } from "@/lib/use-engine";
import { fmtDate } from "@/lib/format";
import type { NewsEvent, StockInfo } from "@/lib/types";

// Fetch a generous candidate batch once per tracked-symbol set; the sliders
// below then filter/slice it client-side with no extra backend calls —
// mirrors frontend/views/news_intelligence.py's st.cache_data batching.
const BATCH_SIZE = 20;
// The backend's essential_news() only ever considers the last 48h
// (news_intelligence/engine.py `_LOOKBACK_HOURS`), so that is the widest
// window worth offering here — past it the slider would filter nothing.
const MAX_AGE_HOURS = 48;
const DEFAULT_WINDOW: [number, number] = [0, 24];
// Narrowest selectable window, in hours — keeps the two handles from landing
// on the same value and collapsing the range to nothing.
const MIN_WINDOW_HOURS = 1;

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

// Thumb diameter in px — mirrors the `.range-dual` thumb size in globals.css.
// A native thumb's centre travels between THUMB_PX/2 and width − THUMB_PX/2
// rather than 0…width, so the highlighted band has to be laid out on that
// same inset scale or it drifts off the handles at either end of the track.
const THUMB_PX = 12;

/** `<percent>% ± <px>px` as a calc() with an explicit operator (no `- -6px`). */
function trackPos(percent: number, px: number) {
  return `calc(${percent.toFixed(3)}% ${px < 0 ? "-" : "+"} ${Math.abs(px).toFixed(2)}px)`;
}

/**
 * Two-handle range slider — the low and high ends are separate native range
 * inputs stacked over a shared rail, which keeps keyboard and screen-reader
 * behaviour native while looking like one control.
 */
function DualSliderField({
  label,
  display,
  lo,
  hi,
  min,
  max,
  step = 1,
  minGap = 1,
  loLabel,
  hiLabel,
  valueText,
  ends,
  onChange,
}: {
  label: string;
  display: string;
  lo: number;
  hi: number;
  min: number;
  max: number;
  step?: number;
  minGap?: number;
  loLabel: string;
  hiLabel: string;
  valueText: (value: number) => string;
  ends: [string, string];
  onChange: (lo: number, hi: number) => void;
}) {
  const span = max === min ? 1 : max - min;
  const loFraction = (lo - min) / span;
  const hiFraction = (hi - min) / span;

  // Clamps, not swaps: a handle stops one step short of its neighbour so the
  // two can never cross and `lo <= hi` holds for every reachable state.
  const setLo = (value: number) => onChange(Math.min(value, hi - minGap), hi);
  const setHi = (value: number) => onChange(lo, Math.max(value, lo + minGap));

  // With the pair squeezed against the right end the high handle covers the
  // low one and has nowhere left to go — lift the low handle so it stays
  // grabbable. Anywhere else the high handle sits on top.
  const loOnTop = hiFraction > 0.9;

  function jumpToPress(e: React.PointerEvent<HTMLDivElement>) {
    // Both inputs are pointer-events:none apart from their thumbs, so a press
    // landing on the container is a press on the rail. Move whichever handle
    // is nearer, the way a single native range input jumps on a track click.
    if (e.target !== e.currentTarget) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const travel = Math.max(1, rect.width - THUMB_PX);
    const raw = min + ((e.clientX - rect.left - THUMB_PX / 2) / travel) * span;
    const snapped = min + Math.round((raw - min) / step) * step;
    const value = Math.min(max, Math.max(min, snapped));
    if (Math.abs(value - lo) <= Math.abs(value - hi)) setLo(value);
    else setHi(value);
  }

  return (
    <div role="group" aria-label={label}>
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-mut">{label}</span>
        <span className="font-mono text-[11px] font-semibold tabular text-ink/85">{display}</span>
      </div>

      {/* mt-2 (not mt-3) so the 12px-tall rail box centres its track on the
          same line as the 3px single sliders sitting beside it in the grid. */}
      <div className="range-dual mt-2" onPointerDown={jumpToPress}>
        <div className="range-rail" />
        <div
          className="range-sel"
          style={{
            left: trackPos(loFraction * 100, (0.5 - loFraction) * THUMB_PX),
            right: trackPos((1 - hiFraction) * 100, (hiFraction - 0.5) * THUMB_PX),
          }}
        />
        <input
          type="range"
          aria-label={loLabel}
          aria-valuetext={valueText(lo)}
          min={min}
          max={max}
          step={step}
          value={lo}
          style={loOnTop ? { zIndex: 2 } : undefined}
          onChange={(e) => setLo(Number(e.target.value))}
        />
        <input
          type="range"
          aria-label={hiLabel}
          aria-valuetext={valueText(hi)}
          min={min}
          max={max}
          step={step}
          value={hi}
          onChange={(e) => setHi(Number(e.target.value))}
        />
      </div>

      <div className="mt-2 flex items-center justify-between font-mono text-[9px] uppercase tracking-wider text-mut/50">
        <span>{ends[0]}</span>
        <span>{ends[1]}</span>
      </div>
    </div>
  );
}

export default function NewsPage() {
  // The tracked-ticker list defaults to the saved portfolio (resolved lazily
  // inside the fetcher below, on first load only) but is editable in the
  // page body — remove a holding you don't care about, or add any symbol
  // you don't hold just to watch its news.
  const symbolsRef = useRef<string[] | null>(null);
  const [symbols, setSymbols] = useState<string[] | null>(null);
  const [resetting, setResetting] = useState(false);
  // Timestamp the batch was fetched, captured as state (not read from a ref
  // during render) — set once per fetch, then reused by the candidates
  // useMemo below so the age filter stays pure across renders instead of
  // calling Date.now() during render.
  const [fetchedAt, setFetchedAt] = useState(0);

  // Both ends of the age window, in hours before `fetchedAt`: [newest, oldest].
  const [ageWindow, setAgeWindow] = useState<[number, number]>(DEFAULT_WINDOW);
  const [minImportance, setMinImportance] = useState(40);
  const [limit, setLimit] = useState(5);
  const [newestHours, oldestHours] = ageWindow;

  // Stable identity on purpose: refetching should only ever happen on mount,
  // on an explicit reload()/applySymbols() call, or the Refresh button —
  // never as a side effect of a symbols-state change, since essential_news()
  // does a live RSS fetch per feed and isn't cheap to double up.
  const fetcher = useCallback(async (signal: AbortSignal) => {
    if (symbolsRef.current === null) {
      const holdings = await fetchPortfolio(signal);
      const derived = Array.from(new Set(holdings.map((h) => h.symbol))).sort();
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

  async function resetToPortfolio() {
    setResetting(true);
    try {
      const holdings = await fetchPortfolio();
      applySymbols(Array.from(new Set(holdings.map((h) => h.symbol))).sort());
    } catch {
      // backend unreachable — keep whatever is currently tracked
    } finally {
      setResetting(false);
    }
  }

  const candidates = useMemo(() => {
    if (!engine.data) return [];
    return engine.data
      .filter((e) => {
        if (e.published) {
          // Floored at 0: a feed timestamp running slightly ahead of our clock
          // is still "just now", not something outside the window.
          const ageHours = Math.max(
            0,
            (fetchedAt - new Date(e.published).getTime()) / 3_600_000,
          );
          if (ageHours < newestHours || ageHours > oldestHours) return false;
        }
        return e.importance >= minImportance;
      })
      .slice(0, limit);
  }, [engine.data, fetchedAt, newestHours, oldestHours, minImportance, limit]);

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
              <DualSliderField
                label="Lookback window"
                display={
                  newestHours === 0 ? `last ${oldestHours}h` : `${newestHours}–${oldestHours}h ago`
                }
                lo={newestHours}
                hi={oldestHours}
                min={0}
                max={MAX_AGE_HOURS}
                minGap={MIN_WINDOW_HOURS}
                loLabel="Lookback window — newest story age, in hours"
                hiLabel="Lookback window — oldest story age, in hours"
                valueText={(h) => (h === 0 ? "now" : `${h}h ago`)}
                ends={["now", `${MAX_AGE_HOURS}h ago`]}
                onChange={(lo, hi) => setAgeWindow([lo, hi])}
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
