"use client";

/** Portfolio Health — Engine 1 (port of frontend/views/portfolio_health.py). */

import { useCallback } from "react";
import { Check, HeartPulse, TriangleAlert } from "lucide-react";
import { ArcGauge, EngineShell, Metric, Section } from "@/components/EngineShell";
import { fetchHealthReport } from "@/lib/api-client";
import { useEngine } from "@/lib/use-engine";
import { fmtNum, fmtPct } from "@/lib/format";
import type { SplitFrame } from "@/lib/types";

const METRIC_DEFS: Record<string, { label: string; fmt: (v: number) => string }> = {
  annual_return: { label: "Annual return", fmt: (v) => fmtPct(v) },
  annual_volatility: { label: "Annual volatility", fmt: (v) => fmtPct(v, 1, false) },
  sharpe: { label: "Sharpe", fmt: (v) => fmtNum(v, 2) },
  sortino: { label: "Sortino", fmt: (v) => fmtNum(v, 2) },
  max_drawdown: { label: "Max drawdown", fmt: (v) => fmtPct(v) },
  beta: { label: "Beta vs SPY", fmt: (v) => fmtNum(v, 2) },
  diversification: { label: "Diversification (0–1)", fmt: (v) => fmtNum(v, 2) },
  largest_position: { label: "Largest position", fmt: (v) => fmtPct(v, 1, false) },
};

function metricLabel(key: string): string {
  return METRIC_DEFS[key]?.label ?? key.replace(/_/g, " ");
}

function metricValue(key: string, v: number | null): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return METRIC_DEFS[key]?.fmt(v) ?? fmtNum(v, 3);
}

function scoreColor(score: number): string {
  if (score >= 70) return "var(--color-gain)";
  if (score >= 45) return "var(--color-warn)";
  return "var(--color-loss)";
}

function CorrelationTable({ corr }: { corr: SplitFrame }) {
  return (
    <div className="scroll-slim overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="px-2 py-2" />
            {corr.columns.map((c) => (
              <th
                key={c}
                className="px-2 py-2 text-center font-mono text-[10px] uppercase tracking-wider text-mut"
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {corr.index.map((row, i) => (
            <tr key={row}>
              <td className="px-2 py-1 font-mono text-[11px] font-semibold text-accent">{row}</td>
              {corr.columns.map((col, j) => {
                const v = corr.data[i]?.[j] ?? null;
                const diag = row === col;
                const alpha = v == null || diag ? 0 : Math.min(Math.abs(v), 1) * 0.4;
                const bg =
                  alpha === 0
                    ? undefined
                    : v! >= 0
                      ? `color-mix(in srgb, var(--color-accent) ${alpha * 100}%, transparent)`
                      : `color-mix(in srgb, var(--color-spy) ${alpha * 100}%, transparent)`;
                return (
                  <td
                    key={col}
                    className={`min-w-[52px] rounded px-2 py-1 text-center font-mono text-[11px] tabular ${
                      diag ? "text-mut/40" : "text-ink/90"
                    }`}
                    style={{ backgroundColor: bg }}
                  >
                    {v == null ? "—" : fmtNum(v, 2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 font-mono text-[10px] uppercase tracking-wider text-mut/70">
        <span className="text-accent/80">green</span> = move together ·{" "}
        <span className="text-spy">periwinkle</span> = move opposite
      </div>
    </div>
  );
}

export default function HealthPage() {
  const fetcher = useCallback((signal: AbortSignal) => fetchHealthReport(signal), []);
  const engine = useEngine(fetcher);
  const report = engine.data;

  const hasMetrics = report != null && Object.keys(report.metrics).length > 0;

  return (
    <EngineShell
      title="Portfolio Health"
      icon={HeartPulse}
      caption="Engine 1 — Portfolio Intelligence · Developer 1"
      engine={engine}
      hasData={report != null}
    >
      {report && (
        <>
          <div className="grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
            <Section title="Health score" className="flex flex-col">
              <div className="flex flex-1 flex-col items-center justify-center py-2">
                <ArcGauge
                  value={report.score}
                  color={scoreColor(report.score)}
                  display={Math.round(report.score)}
                  caption="out of 100"
                />
              </div>
            </Section>
            {hasMetrics && (
              <div className="grid content-start gap-2.5 sm:grid-cols-3">
                <Metric
                  label="Sharpe"
                  value={metricValue("sharpe", report.metrics.sharpe ?? null)}
                  sub="risk-adjusted return"
                  tone={(report.metrics.sharpe ?? 0) >= 1 ? "text-gain" : "text-ink"}
                />
                <Metric
                  label="Max drawdown"
                  value={metricValue("max_drawdown", report.metrics.max_drawdown ?? null)}
                  sub="deepest peak-to-trough"
                  tone={(report.metrics.max_drawdown ?? 0) < -0.25 ? "text-loss" : "text-ink"}
                />
                <Metric
                  label="Volatility"
                  value={metricValue("annual_volatility", report.metrics.annual_volatility ?? null)}
                  sub="annualized"
                />
                <div className="card px-4 py-3.5 sm:col-span-3">
                  <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-mut">
                    All metrics
                  </div>
                  <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-3 md:grid-cols-4">
                    {Object.entries(report.metrics).map(([key, v]) => (
                      <div key={key}>
                        <div className="font-mono text-[10px] uppercase tracking-wider text-mut/80">
                          {metricLabel(key)}
                        </div>
                        <div className="mt-0.5 font-mono text-sm font-medium tabular text-ink/90">
                          {metricValue(key, v)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Section title="Strengths">
              {report.strengths.length === 0 ? (
                <p className="text-sm text-mut">—</p>
              ) : (
                <ul className="space-y-2.5">
                  {report.strengths.map((s) => (
                    <li key={s} className="flex items-start gap-2.5 text-sm leading-relaxed text-ink/90">
                      <Check className="mt-0.5 size-3.5 shrink-0 text-gain" />
                      {s}
                    </li>
                  ))}
                </ul>
              )}
            </Section>
            <Section title="Weaknesses">
              {report.weaknesses.length === 0 ? (
                <p className="text-sm text-mut">—</p>
              ) : (
                <ul className="space-y-2.5">
                  {report.weaknesses.map((w) => (
                    <li key={w} className="flex items-start gap-2.5 text-sm leading-relaxed text-ink/90">
                      <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-warn" />
                      {w}
                    </li>
                  ))}
                </ul>
              )}
            </Section>
          </div>

          {report.correlation && report.correlation.columns.length > 0 && (
            <Section title="Asset correlation">
              <CorrelationTable corr={report.correlation} />
            </Section>
          )}
        </>
      )}
    </EngineShell>
  );
}
