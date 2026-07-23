"use client";

import type { MonthCell } from "@/lib/types";
import { fmtPct } from "@/lib/format";

const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

function cellBg(r: number): string {
  const t = Math.min(Math.abs(r) / 0.08, 1);
  const pct = (0.07 + t * 0.55) * 100;
  return r >= 0
    ? `color-mix(in srgb, var(--color-gain) ${pct.toFixed(1)}%, transparent)`
    : `color-mix(in srgb, var(--color-loss) ${pct.toFixed(1)}%, transparent)`;
}

export function MonthlyHeatmap({ monthly }: { monthly: MonthCell[] }) {
  if (monthly.length === 0) {
    return <p className="py-8 text-center text-xs text-mut">Not enough history for monthly breakdown.</p>;
  }

  const byYear = new Map<number, Map<number, number>>();
  for (const c of monthly) {
    if (!byYear.has(c.year)) byYear.set(c.year, new Map());
    byYear.get(c.year)!.set(c.month, c.r);
  }
  const years = [...byYear.keys()].sort((a, b) => b - a);

  return (
    <div className="scroll-slim overflow-x-auto">
      <table className="w-full min-w-[560px] border-separate" style={{ borderSpacing: "4px" }}>
        <thead>
          <tr>
            <th className="w-10" />
            {MONTHS.map((m) => (
              <th key={m} className="pb-1 text-center font-mono text-[9px] font-medium tracking-wider text-mut/80">
                {m}
              </th>
            ))}
            <th className="pl-2 text-right font-mono text-[9px] font-medium tracking-wider text-mut">
              YEAR
            </th>
          </tr>
        </thead>
        <tbody>
          {years.map((y) => {
            const monthsMap = byYear.get(y)!;
            let annual = 1;
            monthsMap.forEach((r) => {
              annual *= 1 + r;
            });
            annual -= 1;
            return (
              <tr key={y}>
                <td className="pr-1 text-right font-mono text-[10px] tabular text-mut">{y}</td>
                {MONTHS.map((_, mi) => {
                  const r = monthsMap.get(mi + 1);
                  if (r == null) {
                    return <td key={mi} className="h-8 rounded-md bg-overlay/[0.015]" />;
                  }
                  const strong = Math.abs(r) >= 0.04;
                  return (
                    <td
                      key={mi}
                      title={`${y}-${String(mi + 1).padStart(2, "0")}: ${fmtPct(r)}`}
                      className="h-8 rounded-md text-center font-mono text-[10px] tabular transition-transform hover:scale-[1.08]"
                      style={{
                        backgroundColor: cellBg(r),
                        color: strong ? "#0a0f07" : "var(--ink-soft)",
                      }}
                    >
                      {(r * 100).toFixed(1)}
                    </td>
                  );
                })}
                <td
                  className={`h-8 rounded-md pl-2 pr-1 text-right font-mono text-[10px] font-semibold tabular ${
                    annual >= 0 ? "text-gain" : "text-loss"
                  }`}
                >
                  {fmtPct(annual)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="mt-2 flex items-center justify-end gap-2 font-mono text-[9px] text-mut/70">
        <span>monthly return</span>
        <span
          className="inline-block h-2 w-16 rounded-full"
          style={{
            background:
              "linear-gradient(to right, color-mix(in srgb, var(--color-loss) 55%, transparent), color-mix(in srgb, var(--overlay-base) 6%, transparent), color-mix(in srgb, var(--color-gain) 55%, transparent))",
          }}
        />
        <span>−8% · +8%</span>
      </div>
    </div>
  );
}
