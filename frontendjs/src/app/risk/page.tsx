"use client";

/**
 * Risk — per-stock + portfolio downside risk (port of frontend/views/risk.py).
 *
 * Forecasts the SIZE of a move (HAR volatility forecast + Filtered Historical
 * Simulation), not its direction — distinct from the reaction-risk proxy on
 * the "Should I React?" page.
 */

import { useCallback, useState } from "react";
import { motion } from "framer-motion";
import { Loader2, Shield } from "lucide-react";
import { Chip, EngineShell, Metric, Note, Section, ThinBar } from "@/components/EngineShell";
import { fetchPortfolioRisk, fetchRiskEstimates } from "@/lib/api-client";
import { useEngine } from "@/lib/use-engine";
import { fmtPct, signClass } from "@/lib/format";
import type { RiskEstimate } from "@/lib/types";

const HORIZONS: { value: 5 | 20; label: string }[] = [
  { value: 5, label: "Next 5 days · ~1 week" },
  { value: 20, label: "Next 20 days · ~1 month" },
];

function pctSwing(x: number): string {
  return `±${fmtPct(x, 1, false)}`;
}

function dangerColor(level: number): string {
  if (level >= 70) return "var(--color-loss)";
  if (level >= 40) return "var(--color-warn)";
  return "var(--color-gain)";
}

/** A degraded store is scored as "no news", which understates risk exactly
 *  when the feed has failed — surface it rather than hiding it. */
const DEGRADED_FEED = new Set(["missing_store", "stale_store", "invalid_store"]);

function RiskTable({ estimates }: { estimates: RiskEstimate[] }) {
  return (
    <div className="scroll-slim overflow-x-auto">
      <table className="w-full min-w-[820px] border-collapse">
        <thead>
          <tr className="border-b border-line text-left font-mono text-[10px] uppercase tracking-[0.16em] text-mut">
            <th className="px-4 py-3 font-medium">Ticker</th>
            <th className="px-4 py-3 text-right font-medium">Danger (0–100)</th>
            <th className="px-4 py-3 text-right font-medium">Typical swing</th>
            <th className="px-4 py-3 text-right font-medium">Worst-case loss (95%)</th>
            <th className="px-4 py-3 text-right font-medium">Expected shortfall</th>
            <th className="px-4 py-3 text-right font-medium">Severe loss (1%)</th>
            <th className="px-4 py-3 font-medium">News</th>
          </tr>
        </thead>
        <tbody>
          {estimates.map((e, i) => (
            <motion.tr
              key={e.symbol}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.03 * i }}
              className="border-b border-overlay/[0.04] transition-colors last:border-0 hover:bg-overlay/[0.025]"
            >
              <td className="px-4 py-3 font-mono text-sm font-semibold text-accent">{e.symbol}</td>
              <td className="px-4 py-3">
                <div className="flex items-center justify-end gap-2">
                  <ThinBar
                    fraction={e.risk_level / 100}
                    color={dangerColor(e.risk_level)}
                    className="w-16"
                  />
                  <span className="w-8 text-right font-mono text-xs font-medium tabular text-ink/90">
                    {Math.round(e.risk_level)}
                  </span>
                </div>
              </td>
              <td className="px-4 py-3 text-right font-mono text-xs tabular text-ink/85">
                {pctSwing(e.sigma_h)}
              </td>
              <td className="px-4 py-3 text-right font-mono text-xs tabular text-ink/85">
                {fmtPct(e.var_95, 1, false)}
              </td>
              <td className="px-4 py-3 text-right font-mono text-xs tabular text-ink/85">
                {fmtPct(e.es_95, 1, false)}
              </td>
              <td className="px-4 py-3 text-right font-mono text-xs tabular text-mut">
                {fmtPct(e.var_99, 1, false)}
              </td>
              <td className="px-4 py-3">
                {DEGRADED_FEED.has(e.news_quality) ? (
                  <Chip tone="amber">{e.news_quality.replace(/_/g, " ")}</Chip>
                ) : e.news_applied ? (
                  <Chip tone="accent">live</Chip>
                ) : (
                  <span className="text-xs text-mut">—</span>
                )}
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 text-[11px] leading-relaxed text-mut/80">
        Danger = today&apos;s expected swing ranked against the stock&apos;s own history
        (90+ = unusually turbulent). Losses are downside thresholds, validated on 3 years
        of held-out data (the 95% line was breached ~5% of the time, as intended).
        Expected shortfall is the average loss on the days that breach the 95% line —
        the size of a bad day, not just its threshold. News shows whether live attention
        reached the forecast; a degraded feed means the model saw no news and the figures
        are a lower bound.
      </div>
    </div>
  );
}

export default function RiskPage() {
  const [horizon, setHorizon] = useState<5 | 20>(5);

  const fetcher = useCallback(
    async (signal: AbortSignal) => {
      const [estimates, portfolio] = await Promise.all([
        fetchRiskEstimates(horizon, signal),
        fetchPortfolioRisk(horizon, signal),
      ]);
      return { estimates, portfolio };
    },
    [horizon]
  );
  const engine = useEngine(fetcher);
  const data = engine.data;
  const rows = (data?.estimates ?? [])
    .filter((e) => e.has_history)
    .sort((a, b) => b.risk_level - a.risk_level);
  const degradedFeeds = rows
    .filter((e) => DEGRADED_FEED.has(e.news_quality))
    .map((e) => e.symbol)
    .sort();
  const diversificationOffset = data ? 1 - data.portfolio.diversification_ratio : 0;

  return (
    <EngineShell
      title="Risk"
      icon={Shield}
      caption="Downside Risk — HAR Volatility Forecast + Filtered Historical Simulation"
      engine={engine}
      hasData={data != null}
    >
      {data && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="max-w-2xl text-sm leading-relaxed text-mut">
              How much each holding could swing, and the worst it would normally lose —
              the size of a move, not its direction. The number a position-sizing or
              reaction decision should lean on.
            </p>
            <div className="flex shrink-0 items-center gap-2.5">
              {engine.loading && <Loader2 className="size-3.5 animate-spin text-accent" />}
              <div className="inline-flex items-center rounded-xl border border-line bg-overlay/[0.03] p-0.5">
                {HORIZONS.map((h) => (
                  <button
                    key={h.value}
                    onClick={() => setHorizon(h.value)}
                    className={`rounded-[10px] px-3 py-1.5 font-mono text-[11px] tracking-wide transition-all ${
                      horizon === h.value
                        ? "bg-accent font-semibold text-[#0a0f07]"
                        : "text-mut hover:text-ink"
                    }`}
                  >
                    {h.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <Section title="Your whole portfolio">
            <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
              <Metric
                label="Typical swing"
                value={pctSwing(data.portfolio.sigma_h)}
                sub="expected move over the window"
              />
              <Metric
                label="Worst-case loss (95%)"
                value={fmtPct(data.portfolio.var_95, 1, false)}
                sub="19 weeks out of 20, smaller than this"
                tone="text-loss"
              />
              <Metric
                label="Severe loss (worst 1%)"
                value={fmtPct(data.portfolio.var_99, 1, false)}
                sub="a rare bad-day figure"
                tone="text-loss"
              />
              <Metric
                label="Diversification benefit"
                value={`${Math.round(diversificationOffset * 100)}%`}
                sub="offset vs. holding each alone"
                tone={signClass(diversificationOffset)}
              />
            </div>
            {data.portfolio.as_of && (
              <div className="mt-3 font-mono text-[10px] uppercase tracking-wider text-mut/70">
                As of {data.portfolio.as_of} · {data.portfolio.n_holdings} holdings scored
              </div>
            )}
          </Section>

          <Section title="Each holding">
            {degradedFeeds.length > 0 && (
              <div className="mb-3">
                <Note tone="amber">
                  News feed degraded for {degradedFeeds.join(", ")} — the model scored
                  these as having no news, so their risk figures are a lower bound
                  until the feed recovers.
                </Note>
              </div>
            )}
            {rows.length === 0 ? (
              <Note>No holding has enough price history to score yet.</Note>
            ) : (
              <RiskTable estimates={rows} />
            )}
          </Section>
        </>
      )}
    </EngineShell>
  );
}
