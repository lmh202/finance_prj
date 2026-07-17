"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Eye, EyeOff, Loader2, TriangleAlert, Waypoints } from "lucide-react";
import { SearchBox } from "@/components/SearchBox";
import { HoldingsPanel, type Holding, type HoldingMeta } from "@/components/HoldingsPanel";
import { PerformanceChart, type ChartSeries } from "@/components/PerformanceChart";
import { StatsRow } from "@/components/StatsRow";
import { AllocationDonut, SectorBars } from "@/components/AllocationDonut";
import { HoldingsTable } from "@/components/HoldingsTable";
import { MonthlyHeatmap } from "@/components/MonthlyHeatmap";
import { Header, Footer, Hero, PRESETS, type Preset } from "@/components/chrome";
import { deriveRangeView } from "@/lib/view";
import { fmtPct, fmtDate } from "@/lib/format";
import { RANGES, type AnalyzeResponse, type InputMode, type RangeId, type StockInfo } from "@/lib/types";
import { analyze } from "@/lib/api-client";

const ACCENT = "#B3F34C";

export default function Page() {
  const [hydrated, setHydrated] = useState(false);
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [mode, setMode] = useState<InputMode>("weight");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [range, setRange] = useState<RangeId>("5Y");
  const [showSpy, setShowSpy] = useState(true);
  const [showQqq, setShowQqq] = useState(false);
  const lastSaved = useRef<string>("");

  /* ---------- hydrate from localStorage ---------- */
  useEffect(() => {
    try {
      const raw = localStorage.getItem("aurora_portfolio");
      if (raw) {
        const parsed = JSON.parse(raw) as {
          h: Holding[];
          m: InputMode;
        };
        if (Array.isArray(parsed.h) && parsed.h.length > 0) {
          setHoldings(parsed.h);
          if (parsed.m === "shares" || parsed.m === "weight") {
            setMode(parsed.m);
          }
          lastSaved.current = raw;
        }
      }
    } catch {
      /* corrupted — start fresh */
    }
    setHydrated(true);
  }, []);

  const holdingsKey = useMemo(
    () => JSON.stringify(holdings.map((h) => [h.symbol, h.value])),
    [holdings],
  );

  /* ---------- persist to localStorage ---------- */
  useEffect(() => {
    if (!hydrated) return;
    const snapshot = JSON.stringify({ h: holdings, m: mode });
    if (snapshot === lastSaved.current) return;
    const t = setTimeout(() => {
      lastSaved.current = snapshot;
      try {
        localStorage.setItem("aurora_portfolio", snapshot);
      } catch {
        /* localStorage full — ignore */
      }
    }, 700);
    return () => clearTimeout(t);
  }, [holdingsKey, mode, hydrated, holdings]);

  /* ---------- analysis ---------- */
  useEffect(() => {
    if (!hydrated) return;
    if (holdings.length === 0) {
      setResult(null);
      setLoading(false);
      return;
    }
    const ctrl = new AbortController();
    setLoading(true);
    setError(null);
    const t = setTimeout(async () => {
      try {
        const json = await analyze(
          holdings.map((h) => ({ symbol: h.symbol, value: h.value })),
          mode,
        );
        if (ctrl.signal.aborted) return;
        setResult(json);
      } catch (err) {
        if (!ctrl.signal.aborted) {
          setError(
            err instanceof Error
              ? err.message
              : "Network error — could not reach the analytics engine.",
          );
        }
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }, 350);
    return () => {
      ctrl.abort();
      clearTimeout(t);
    };
  }, [holdingsKey, mode, hydrated, holdings]);

  /* ---------- derived view ---------- */
  const months = RANGES.find((r) => r.id === range)!.months;
  const view = useMemo(() => (result ? deriveRangeView(result, months) : null), [result, months]);

  const existing = useMemo(() => new Set(holdings.map((h) => h.symbol)), [holdings]);

  const meta = useMemo(() => {
    if (!result) return null;
    const m = new Map<string, HoldingMeta>();
    for (const h of result.holdings) {
      m.set(h.symbol, {
        weight: h.weight,
        lastPrice: h.lastPrice,
        simulated: h.simulated,
        name: h.name,
        sector: h.sector,
      });
    }
    return m;
  }, [result]);

  /* ---------- actions ---------- */
  function addHolding(info: StockInfo) {
    if (existing.has(info.symbol)) return;
    const value = mode === "shares" ? 10 : holdings.length === 0 ? 100 : 10;
    setHoldings((hs) => [
      ...hs,
      { symbol: info.symbol, value, name: info.name, quoteType: info.quoteType },
    ]);
  }
  function changeValue(symbol: string, v: number) {
    setHoldings((hs) => hs.map((h) => (h.symbol === symbol ? { ...h, value: v } : h)));
  }
  function removeHolding(symbol: string) {
    setHoldings((hs) => hs.filter((h) => h.symbol !== symbol));
  }
  function clearAll() {
    setHoldings([]);
  }
  function equalize() {
    setHoldings((hs) =>
      hs.map((h) => ({
        ...h,
        value: mode === "shares" ? 10 : Math.round((1000 / hs.length)) / 10,
      })),
    );
  }
  function applyPreset(p: Preset) {
    setMode("weight");
    setHoldings(p.holdings.map((h) => ({ symbol: h.symbol, value: h.value, name: h.name, quoteType: h.quoteType })));
  }

  /* ---------- chart series ---------- */
  const chartSeries: ChartSeries[] = useMemo(() => {
    if (!view) return [];
    const out: ChartSeries[] = [
      { id: "PF", label: "Portfolio", color: ACCENT, values: view.portfolio, width: 2.5 },
    ];
    for (const b of view.benchmarks) {
      if (b.symbol === "SPY" && showSpy) out.push({ id: "SPY", label: b.name, color: b.color, values: b.values });
      if (b.symbol === "QQQ" && showQqq) out.push({ id: "QQQ", label: b.name, color: b.color, values: b.values });
    }
    return out;
  }, [view, showSpy, showQqq]);

  const spyView = view?.benchmarks.find((b) => b.symbol === "SPY") ?? null;
  const endReturn = view ? view.portfolio[view.portfolio.length - 1] / 100 - 1 : 0;

  const showDashboard = view != null;
  const showSkeleton = !showDashboard && loading && holdings.length > 0;

  return (
    <div className="flex min-h-screen flex-col">
      <Header source={result?.source ?? null} asOf={result ? fmtDate(result.asOf, true) : undefined} />

      <main className="flex-1">
        <AnimatePresence mode="wait">
          {/* ---------------- EMPTY STATE ---------------- */}
          {!showDashboard && !showSkeleton && (
            <motion.div key="hero" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, y: -12 }} transition={{ duration: 0.35 }}>
              <Hero
                searchSlot={<SearchBox big onAdd={addHolding} existing={existing} placeholder="Search any stock or ETF — try “NVDA” or “Apple”…" />}
                presetsSlot={
                  <div className="flex flex-wrap items-center justify-center gap-2">
                    <span className="font-mono text-[10px] uppercase tracking-widest text-mut/70">Try</span>
                    {PRESETS.map((p) => (
                      <button
                        key={p.label}
                        onClick={() => applyPreset(p)}
                        className="rounded-full border border-line bg-white/[0.04] px-3.5 py-1.5 text-xs text-ink/80 transition-all hover:border-accent/50 hover:bg-accent-dim hover:text-accent"
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                }
                error={error}
              />
            </motion.div>
          )}

          {/* ---------------- SKELETON ---------------- */}
          {showSkeleton && (
            <motion.div key="skeleton" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mx-auto max-w-[1480px] px-4 py-6 lg:px-8">
              <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
                <div className="space-y-4">
                  <div className="skeleton h-12 rounded-2xl" />
                  <div className="skeleton h-[420px] rounded-2xl" />
                </div>
                <div className="space-y-4">
                  <div className="skeleton h-[430px] rounded-2xl" />
                  <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 xl:grid-cols-5">
                    {Array.from({ length: 10 }).map((_, i) => (
                      <div key={i} className="skeleton h-[92px] rounded-2xl" />
                    ))}
                  </div>
                  <div className="grid gap-4 xl:grid-cols-2">
                    <div className="skeleton h-[300px] rounded-2xl" />
                    <div className="skeleton h-[300px] rounded-2xl" />
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* ---------------- DASHBOARD ---------------- */}
          {view && result && (
            <motion.div key="dash" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }} className="mx-auto max-w-[1480px] px-4 py-6 lg:px-8">
              <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
                {/* left rail */}
                <aside className="flex min-h-0 flex-col gap-3 lg:sticky lg:top-[72px] lg:h-[calc(100vh-96px)]">
                  <SearchBox onAdd={addHolding} existing={existing} />
                  <HoldingsPanel
                    holdings={holdings}
                    mode={mode}
                    meta={meta}
                    onModeChange={setMode}
                    onValueChange={changeValue}
                    onRemove={removeHolding}
                    onClear={clearAll}
                    onEqualize={equalize}
                  />
                </aside>

                {/* main column */}
                <div className="min-w-0 space-y-4">
                  {/* chart card */}
                  <section className="card relative overflow-hidden p-5 pb-7">
                    <div className="flex flex-wrap items-end justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                          <Waypoints className="size-3.5 text-accent" />
                          Constant-mix portfolio · indexed to 100
                        </div>
                        <div className="mt-1.5 flex items-baseline gap-3">
                          <span className={`text-4xl font-bold tracking-tight tabular ${endReturn >= 0 ? "text-gain" : "text-loss"}`}>
                            {fmtPct(endReturn)}
                          </span>
                          <span className="font-mono text-xs text-mut">
                            {fmtDate(view.dates[0], true)} → {fmtDate(view.dates[view.dates.length - 1], true)}
                          </span>
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center gap-2">
                        {view.benchmarks.map((b) => {
                          const on = b.symbol === "SPY" ? showSpy : showQqq;
                          const toggle = b.symbol === "SPY" ? () => setShowSpy((v) => !v) : () => setShowQqq((v) => !v);
                          return (
                            <button
                              key={b.symbol}
                              onClick={toggle}
                              className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider transition-all ${
                                on ? "border-line bg-white/[0.05] text-ink/85" : "border-line/50 text-mut/45"
                              }`}
                            >
                              <span className="size-1.5 rounded-full" style={{ backgroundColor: b.color, opacity: on ? 1 : 0.35 }} />
                              {b.symbol}
                              {on ? <Eye className="size-3" /> : <EyeOff className="size-3" />}
                            </button>
                          );
                        })}
                        <div className="ml-1 flex rounded-xl border border-line bg-white/[0.03] p-0.5">
                          {RANGES.map((r) => (
                            <button
                              key={r.id}
                              onClick={() => setRange(r.id)}
                              className={`rounded-[10px] px-2.5 py-1 font-mono text-[10px] tracking-wider transition-all ${
                                range === r.id ? "bg-accent font-semibold text-[#0a0f07]" : "text-mut hover:text-ink"
                              }`}
                            >
                              {r.id}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="mt-4">
                      <PerformanceChart dates={view.dates} series={chartSeries} height={380} />
                    </div>

                    {result.range.truncatedNote && (
                      <div className="mt-5 flex items-center gap-2 text-[11px] text-amber-200/80">
                        <TriangleAlert className="size-3.5 shrink-0" />
                        {result.range.truncatedNote}
                      </div>
                    )}
                    {result.source !== "live" && (
                      <div className="mt-2 flex items-center gap-2 text-[11px] text-amber-200/80">
                        <TriangleAlert className="size-3.5 shrink-0" />
                        Live market feed partially unreachable — some series are deterministic
                        simulations so the analysis remains explorable.
                      </div>
                    )}

                    <AnimatePresence>
                      {loading && (
                        <motion.div
                          initial={{ opacity: 0, y: -6 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0 }}
                          className="absolute right-4 top-4 flex items-center gap-1.5 rounded-full border border-accent/30 bg-bg0/90 px-3 py-1 font-mono text-[10px] uppercase tracking-wider text-accent backdrop-blur"
                        >
                          <Loader2 className="size-3 animate-spin" />
                          Updating
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </section>

                  {/* stats */}
                  <StatsRow metrics={view.metrics} spy={spyView?.stats ?? null} />

                  {/* allocation + heatmap */}
                  <div className="grid gap-4 xl:grid-cols-2">
                    <section className="card p-5">
                      <h3 className="mb-5 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                        Allocation
                      </h3>
                      <AllocationDonut
                        items={view.holdings.map((h) => ({
                          symbol: h.symbol,
                          name: h.name,
                          sector: h.sector,
                          weight: h.weight,
                        }))}
                      />
                      <SectorBars sectors={result.sectors} />
                    </section>
                    <section className="card p-5">
                      <h3 className="mb-4 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                        Monthly returns · portfolio
                      </h3>
                      <MonthlyHeatmap monthly={view.monthly} />
                    </section>
                  </div>

                  {/* table */}
                  <section className="card px-2 py-2">
                    <h3 className="px-3 pb-1 pt-3 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                      Position analytics
                    </h3>
                    <HoldingsTable holdings={view.holdings} />
                  </section>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <Footer />
    </div>
  );
}
