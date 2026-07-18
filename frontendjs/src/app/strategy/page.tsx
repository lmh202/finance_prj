"use client";

/** Daily Strategy — Engine 2 (port of frontend/views/daily_strategy.py). */

import { useCallback } from "react";
import { motion } from "framer-motion";
import { TrendingUp } from "lucide-react";
import { Chip, EngineShell, Note, Section, ThinBar, type ChipTone } from "@/components/EngineShell";
import { fetchRegime, fetchSignals } from "@/lib/api-client";
import { useEngine } from "@/lib/use-engine";
import { fmtDate, fmtNum, fmtPct, fmtPrice } from "@/lib/format";
import type { AssetSignal, RegimeState } from "@/lib/types";

const REGIME_META: Record<string, { label: string; dot: string; tone: string }> = {
  bullish: { label: "Bullish", dot: "#3fd9a4", tone: "text-gain" },
  bearish: { label: "Bearish", dot: "#ff7a7a", tone: "text-loss" },
  high_volatility: { label: "High volatility", dot: "#fcd34d", tone: "text-amber-200" },
  sideways: { label: "Sideways / uncertain", dot: "#86938a", tone: "text-ink/85" },
};

const INDICATOR_DEFS: Record<string, { label: string; fmt: (v: number) => string }> = {
  price: { label: "SPY price", fmt: (v) => `$${fmtPrice(v)}` },
  sma50: { label: "SMA 50", fmt: (v) => `$${fmtPrice(v)}` },
  sma200: { label: "SMA 200", fmt: (v) => `$${fmtPrice(v)}` },
  momentum_20d: { label: "Momentum 20d", fmt: (v) => fmtPct(v) },
  volatility_20d: { label: "Volatility 20d", fmt: (v) => fmtPct(v, 1, false) },
  volatility_median: { label: "Median volatility", fmt: (v) => fmtPct(v, 1, false) },
};

const ACTION_TONES: Record<string, ChipTone> = {
  increase: "gain",
  hold: "mut",
  reduce: "loss",
};

function RegimeCard({ regime }: { regime: RegimeState }) {
  const meta = REGIME_META[regime.regime] ?? {
    label: regime.regime,
    dot: "#86938a",
    tone: "text-ink/85",
  };
  return (
    <Section title="Market regime" className="flex flex-col">
      <div className="flex flex-1 flex-col justify-center gap-4 py-1">
        <div className="flex items-center gap-3">
          <span
            className="pulse-dot size-2.5 rounded-full"
            style={{ backgroundColor: meta.dot }}
          />
          <span className={`text-2xl font-bold tracking-tight ${meta.tone}`}>{meta.label}</span>
        </div>
        <div>
          <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-wider text-mut">
            <span>Confidence</span>
            <span className="tabular text-ink/85">{fmtPct(regime.confidence, 0, false)}</span>
          </div>
          <div className="mt-1.5">
            <ThinBar fraction={regime.confidence} color={meta.dot} className="w-full" />
          </div>
        </div>
        {regime.as_of && (
          <div className="font-mono text-[10px] uppercase tracking-wider text-mut/70">
            as of {fmtDate(regime.as_of.slice(0, 10), true)}
          </div>
        )}
      </div>
    </Section>
  );
}

function SignalsTable({ signals }: { signals: AssetSignal[] }) {
  return (
    <div className="scroll-slim overflow-x-auto">
      <table className="w-full min-w-[820px] border-collapse">
        <thead>
          <tr className="border-b border-line text-left font-mono text-[10px] uppercase tracking-[0.16em] text-mut">
            <th className="px-4 py-3 font-medium">Ticker</th>
            <th className="px-4 py-3 text-right font-medium">Score</th>
            <th className="px-4 py-3 text-center font-medium">Signal</th>
            <th className="px-4 py-3 text-right font-medium">Momentum</th>
            <th className="px-4 py-3 text-right font-medium">Sharpe</th>
            <th className="px-4 py-3 text-right font-medium">Volatility</th>
            <th className="px-4 py-3 font-medium">Why</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s, i) => (
            <motion.tr
              key={s.symbol}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.03 * i }}
              className="border-b border-white/[0.04] transition-colors last:border-0 hover:bg-white/[0.025]"
            >
              <td className="px-4 py-3 font-mono text-sm font-semibold text-accent">{s.symbol}</td>
              <td className="px-4 py-3">
                <div className="flex items-center justify-end gap-2">
                  <ThinBar fraction={s.score / 100} className="w-16" />
                  <span className="w-8 text-right font-mono text-xs font-medium tabular text-ink/90">
                    {Math.round(s.score)}
                  </span>
                </div>
              </td>
              <td className="px-4 py-3 text-center">
                <Chip tone={ACTION_TONES[s.action] ?? "mut"}>{s.action}</Chip>
              </td>
              <td className="px-4 py-3 text-right font-mono text-xs tabular text-ink/85">
                {s.indicators.momentum != null ? fmtPct(s.indicators.momentum) : "—"}
              </td>
              <td className="px-4 py-3 text-right font-mono text-xs tabular text-ink/85">
                {s.indicators.sharpe != null ? fmtNum(s.indicators.sharpe, 2) : "—"}
              </td>
              <td className="px-4 py-3 text-right font-mono text-xs tabular text-mut">
                {s.indicators.volatility != null ? fmtPct(s.indicators.volatility, 1, false) : "—"}
              </td>
              <td className="max-w-[340px] px-4 py-3 text-xs leading-relaxed text-mut">
                {s.rationale}
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function StrategyPage() {
  const fetcher = useCallback(async (signal: AbortSignal) => {
    const [regime, signals] = await Promise.all([fetchRegime(signal), fetchSignals(signal)]);
    return { regime, signals };
  }, []);
  const engine = useEngine(fetcher);
  const data = engine.data;

  return (
    <EngineShell
      title="Daily Strategy"
      icon={TrendingUp}
      caption="Engine 2 — Regime-Aware Momentum · Developer 2"
      engine={engine}
      hasData={data != null}
    >
      {data && (
        <>
          <div className="grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
            <RegimeCard regime={data.regime} />
            <Section title="Benchmark indicators (SPY)">
              <div className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-3">
                {Object.entries(data.regime.indicators).map(([key, v]) => (
                  <div key={key}>
                    <div className="font-mono text-[10px] uppercase tracking-wider text-mut/80">
                      {INDICATOR_DEFS[key]?.label ?? key.replace(/_/g, " ")}
                    </div>
                    <div className="mt-1 font-mono text-lg font-semibold tabular text-ink/90">
                      {v == null || !Number.isFinite(v)
                        ? "—"
                        : INDICATOR_DEFS[key]?.fmt(v) ?? fmtNum(v, 4)}
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          </div>

          <section className="card px-2 py-2">
            <h3 className="px-3 pb-1 pt-3 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
              Daily asset ranking
            </h3>
            {data.signals.length === 0 ? (
              <div className="px-3 py-3">
                <Note>No asset has enough history to score yet.</Note>
              </div>
            ) : (
              <SignalsTable signals={data.signals} />
            )}
          </section>
        </>
      )}
    </EngineShell>
  );
}
