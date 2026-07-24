"use client";

/** Daily Strategy — Engine 2 (port of frontend/views/daily_strategy.py). */

import { useCallback } from "react";
import { motion } from "framer-motion";
import { Check, TrendingUp } from "lucide-react";
import { Chip, EngineShell, Note, Section, ThinBar, type ChipTone } from "@/components/EngineShell";
import { fetchStrategyRecommendations } from "@/lib/api-client";
import { useEngine } from "@/lib/use-engine";
import { fmtNum, fmtPrice, signClass } from "@/lib/format";
import type { StrategyRecommendation } from "@/lib/types";

const ACTION_TONES: Record<string, ChipTone> = {
  BUY: "gain",
  ADD: "gain",
  TRIM: "amber",
  SELL: "loss",
  HOLD: "mut",
};

// |score| <= W_EMA + W_MACD + W_RSI = 1.5 + 1.0 + 1.0 (backend/src/daily_strategy/engine.py)
const MAX_SCORE = 3.5;

function scoreColor(score: number): string {
  if (Math.abs(score) < 1e-9) return "var(--color-mut)";
  return score > 0 ? "var(--color-gain)" : "var(--color-loss)";
}

function RecommendationsTable({ recs }: { recs: StrategyRecommendation[] }) {
  return (
    <div className="scroll-slim overflow-x-auto">
      <table className="w-full min-w-[860px] border-collapse">
        <thead>
          <tr className="border-b border-line text-left font-mono text-[10px] uppercase tracking-[0.16em] text-mut">
            <th className="px-4 py-3 font-medium">Ticker</th>
            <th className="px-4 py-3 font-medium">Action</th>
            <th className="px-4 py-3 text-center font-medium">Held</th>
            <th className="px-4 py-3 text-right font-medium">Score</th>
            <th className="px-4 py-3 text-right font-medium">P&L %</th>
            <th className="px-4 py-3 text-right font-medium">Price</th>
            <th className="px-4 py-3 font-medium">Why</th>
          </tr>
        </thead>
        <tbody>
          {recs.map((r, i) => (
            <motion.tr
              key={r.symbol}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.03 * i }}
              className="border-b border-overlay/[0.04] transition-colors last:border-0 hover:bg-overlay/[0.025]"
            >
              <td className="px-4 py-3 font-mono text-sm font-semibold text-accent">{r.symbol}</td>
              <td className="px-4 py-3">
                <Chip tone={ACTION_TONES[r.recommendation] ?? "mut"}>{r.recommendation}</Chip>
              </td>
              <td className="px-4 py-3 text-center">
                {r.held ? (
                  <Check className="mx-auto size-3.5 text-accent" />
                ) : (
                  <span className="text-mut/40">—</span>
                )}
              </td>
              <td className="px-4 py-3">
                <div className="flex items-center justify-end gap-2">
                  <ThinBar
                    fraction={Math.abs(r.score) / MAX_SCORE}
                    color={scoreColor(r.score)}
                    className="w-16"
                  />
                  <span
                    className={`w-10 text-right font-mono text-xs font-medium tabular ${signClass(r.score)}`}
                  >
                    {r.score >= 0 ? "+" : ""}
                    {r.score.toFixed(1)}
                  </span>
                </div>
              </td>
              <td
                className={`px-4 py-3 text-right font-mono text-xs tabular ${signClass(
                  r.unrealized_pnl_pct
                )}`}
              >
                {r.unrealized_pnl_pct == null
                  ? "—"
                  : `${r.unrealized_pnl_pct >= 0 ? "+" : ""}${fmtNum(r.unrealized_pnl_pct, 1)}%`}
              </td>
              <td className="px-4 py-3 text-right font-mono text-xs tabular text-ink/85">
                {r.current_price != null ? `$${fmtPrice(r.current_price)}` : "—"}
              </td>
              <td className="max-w-[360px] px-4 py-3 text-xs leading-relaxed text-mut">
                {r.reasons.join(" · ")}
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function StrategyPage() {
  const fetcher = useCallback((signal: AbortSignal) => fetchStrategyRecommendations(signal), []);
  const engine = useEngine(fetcher);
  const recs = engine.data;

  return (
    <EngineShell
      title="Daily Strategy"
      icon={TrendingUp}
      caption="Engine 2 — EMA/RSI/MACD confluence signals · Developer 2"
      engine={engine}
      hasData={recs != null}
    >
      {recs && (
        <>
          <p className="max-w-2xl text-sm leading-relaxed text-mut">
            Confluence score = EMA20/50 (±1.5) + fresh MACD crossover (±1.0) + RSI (±1.0). Raw
            BUY at score ≥ 2, SELL at ≤ −2. The portfolio layer then maps (what you hold + P&L)
            into BUY / ADD / TRIM / SELL / HOLD.
          </p>

          <Section title="Per-stock recommendations">
            {recs.length === 0 ? (
              <Note>No symbol has enough history to score yet.</Note>
            ) : (
              <RecommendationsTable recs={recs} />
            )}
          </Section>
        </>
      )}
    </EngineShell>
  );
}
