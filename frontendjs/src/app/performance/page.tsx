"use client";

/** Performance & Benchmark — presented by Developer 1 over Developer 2's
 *  backtest output (port of frontend/views/portfolio_health.py::render_performance). */

import { useCallback, useMemo } from "react";
import { Flag } from "lucide-react";
import { EngineShell, Note } from "@/components/EngineShell";
import { PerformanceChart, type ChartSeries } from "@/components/PerformanceChart";
import { fetchBacktest, type BacktestCurves } from "@/lib/api-client";
import { useEngine } from "@/lib/use-engine";
import { fmtDate, fmtNum, fmtPct, signClass } from "@/lib/format";

const SERIES_META: Record<string, { label: string; color: string }> = {
  buy_hold: { label: "Buy & hold", color: "#B3F34C" },
  equal_weight: { label: "Equal weight", color: "#8DA2FB" },
};

const EXTRA_COLORS = ["#CFA3FF", "#3FD9A4", "#FCD34D", "#FF7A7A"];

function seriesMeta(column: string, i: number): { label: string; color: string } {
  return (
    SERIES_META[column] ?? {
      label: column.replace(/_/g, " "),
      color: EXTRA_COLORS[i % EXTRA_COLORS.length],
    }
  );
}

/** Column values with nulls forward-filled, indexed to 100 for the chart. */
function chartValues(curves: BacktestCurves, col: number): number[] {
  const out: number[] = [];
  let last = 1;
  for (const row of curves.rows) {
    const v = row[col];
    if (v != null && Number.isFinite(v)) last = v;
    out.push(last * 100);
  }
  return out;
}

export default function PerformancePage() {
  const fetcher = useCallback((signal: AbortSignal) => fetchBacktest(signal), []);
  const engine = useEngine(fetcher);
  const curves = engine.data;

  const series: ChartSeries[] = useMemo(() => {
    if (!curves) return [];
    return curves.columns.map((c, i) => {
      const meta = seriesMeta(c, i);
      return {
        id: c,
        label: meta.label,
        color: meta.color,
        values: chartValues(curves, i),
        width: i === 0 ? 2.5 : 1.5,
      };
    });
  }, [curves]);

  const empty = curves != null && (curves.dates.length < 2 || curves.columns.length === 0);
  const tail = useMemo(() => {
    if (!curves) return [];
    const n = curves.dates.length;
    const start = Math.max(0, n - 10);
    return curves.dates.slice(start).map((date, k) => ({
      date,
      values: curves.rows[start + k] ?? [],
    }));
  }, [curves]);

  return (
    <EngineShell
      title="Performance & Benchmark"
      icon={Flag}
      caption="Presentation: Developer 1 · Backtest engine: Developer 2"
      engine={engine}
      hasData={curves != null}
    >
      {curves &&
        (empty ? (
          <Note>Nothing to backtest yet.</Note>
        ) : (
          <>
            <div
              className="grid gap-2.5"
              style={{
                gridTemplateColumns: `repeat(auto-fit, minmax(180px, 1fr))`,
              }}
            >
              {series.map((s) => {
                const final = s.values[s.values.length - 1] / 100;
                return (
                  <div key={s.id} className="card px-4 py-3.5">
                    <div className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-mut">
                      <span className="size-1.5 rounded-full" style={{ backgroundColor: s.color }} />
                      {s.label}
                    </div>
                    <div className="mt-1.5 font-mono text-[22px] font-semibold leading-none tabular text-ink">
                      ${fmtNum(final, 3)}
                    </div>
                    <div className={`mt-1.5 font-mono text-[10px] tabular ${signClass(final - 1)}`}>
                      {fmtPct(final - 1)} · growth of $1
                    </div>
                  </div>
                );
              })}
            </div>

            <section className="card p-5 pb-7">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <h3 className="font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                  Walk-forward backtest · growth of $1 · indexed to 100
                </h3>
                <span className="font-mono text-[10px] tabular text-mut/70">
                  {fmtDate(curves.dates[0], true)} → {fmtDate(curves.dates[curves.dates.length - 1], true)}
                </span>
              </div>
              <div className="mt-4">
                <PerformanceChart dates={curves.dates} series={series} height={380} />
              </div>
              <div className="mt-5 flex flex-wrap items-center gap-4">
                {series.map((s) => (
                  <span key={s.id} className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-mut">
                    <span className="size-1.5 rounded-full" style={{ backgroundColor: s.color }} />
                    {s.label}
                  </span>
                ))}
              </div>
            </section>

            <section className="card px-2 py-2">
              <h3 className="px-3 pb-1 pt-3 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                Last {tail.length} sessions
              </h3>
              <div className="scroll-slim overflow-x-auto">
                <table className="w-full min-w-[520px] border-collapse">
                  <thead>
                    <tr className="border-b border-line text-left font-mono text-[10px] uppercase tracking-[0.16em] text-mut">
                      <th className="px-3 py-2.5 font-medium">Date</th>
                      {curves.columns.map((c, i) => (
                        <th key={c} className="px-3 py-2.5 text-right font-medium">
                          {seriesMeta(c, i).label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tail.map((row) => (
                      <tr key={row.date} className="border-b border-white/[0.04] last:border-0">
                        <td className="px-3 py-2 font-mono text-xs tabular text-mut">
                          {fmtDate(row.date, true)}
                        </td>
                        {curves.columns.map((c, i) => (
                          <td key={c} className="px-3 py-2 text-right font-mono text-xs tabular text-ink/85">
                            {row.values[i] == null ? "—" : fmtNum(row.values[i], 4)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <Note tone="mut">
              On the roadmap — Developer 1: rolling metrics and the full metric table
              (return, Sharpe, Sortino, max drawdown, volatility, turnover).
              Developer 2: the ML strategy runs (price-only vs price+news) added to
              backtest() so the ablation shows up here.
            </Note>
          </>
        ))}
    </EngineShell>
  );
}
