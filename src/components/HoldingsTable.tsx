"use client";

import { motion } from "framer-motion";
import type { ViewHolding } from "@/lib/view";
import { fmtPct, fmtNum, fmtPrice, signClass } from "@/lib/format";

function Sparkline({ values, positive }: { values: number[]; positive: boolean }) {
  const W = 96;
  const H = 26;
  const step = Math.max(1, Math.floor(values.length / 48));
  const pts: number[] = [];
  for (let i = 0; i < values.length; i += step) pts.push(values[i]);
  if (pts[pts.length - 1] !== values[values.length - 1]) pts.push(values[values.length - 1]);
  const lo = Math.min(...pts);
  const hi = Math.max(...pts);
  const span = Math.max(hi - lo, 1e-9);
  const d = pts
    .map((v, i) => {
      const x = (i / (pts.length - 1)) * W;
      const y = H - 3 - ((v - lo) / span) * (H - 6);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join("");
  const color = positive ? "#3fd9a4" : "#ff7a7a";
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-[26px] w-24 overflow-visible">
      <path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
        strokeLinecap="round"
        opacity={0.9}
      />
    </svg>
  );
}

export function HoldingsTable({ holdings }: { holdings: ViewHolding[] }) {
  const maxContrib = Math.max(...holdings.map((h) => Math.abs(h.contribution)), 1e-9);
  return (
    <div className="scroll-slim overflow-x-auto">
      <table className="w-full min-w-[760px] border-collapse">
        <thead>
          <tr className="border-b border-line text-left font-mono text-[10px] uppercase tracking-[0.16em] text-mut">
            <th className="px-4 py-3 font-medium">Position</th>
            <th className="px-4 py-3 text-right font-medium">Weight</th>
            <th className="px-4 py-3 text-right font-medium">Last</th>
            <th className="px-4 py-3 text-right font-medium">Period</th>
            <th className="px-4 py-3 text-right font-medium">CAGR</th>
            <th className="px-4 py-3 text-right font-medium">Vol</th>
            <th className="px-4 py-3 text-right font-medium">Sharpe</th>
            <th className="px-4 py-3 text-right font-medium">Contribution</th>
            <th className="px-4 py-3 text-right font-medium">Trend</th>
          </tr>
        </thead>
        <tbody>
          {holdings.map((h, i) => {
            const positive = h.stats.totalReturn >= 0;
            return (
              <motion.tr
                key={h.symbol}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.03 * i }}
                className="border-b border-white/[0.04] transition-colors last:border-0 hover:bg-white/[0.025]"
              >
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2.5">
                    <span className="font-mono text-sm font-semibold text-accent">{h.symbol}</span>
                    {h.simulated && (
                      <span className="rounded border border-amber-300/30 bg-amber-300/10 px-1 py-px font-mono text-[9px] uppercase tracking-wider text-amber-200/90">
                        sim
                      </span>
                    )}
                    <span className="max-w-[200px] truncate text-xs text-mut">{h.name}</span>
                  </div>
                </td>
                <td className="px-4 py-3 text-right">
                  <div className="flex items-center justify-end gap-2">
                    <div className="h-[3px] w-12 overflow-hidden rounded-full bg-white/[0.07]">
                      <div
                        className="h-full rounded-full bg-accent/80"
                        style={{ width: `${Math.min(h.weight * 100, 100)}%` }}
                      />
                    </div>
                    <span className="font-mono text-xs tabular text-ink/85">
                      {fmtPct(h.weight, 1, false)}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-right font-mono text-xs tabular text-mut">
                  {h.lastPrice != null ? `$${fmtPrice(h.lastPrice)}` : "—"}
                </td>
                <td className={`px-4 py-3 text-right font-mono text-xs font-medium tabular ${signClass(h.stats.totalReturn)}`}>
                  {fmtPct(h.stats.totalReturn)}
                </td>
                <td className={`px-4 py-3 text-right font-mono text-xs tabular ${signClass(h.stats.cagr)}`}>
                  {fmtPct(h.stats.cagr)}
                </td>
                <td className="px-4 py-3 text-right font-mono text-xs tabular text-mut">
                  {fmtPct(h.stats.annVol, 1, false)}
                </td>
                <td className="px-4 py-3 text-right font-mono text-xs tabular text-ink/85">
                  {fmtNum(h.stats.sharpe, 2)}
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center justify-end gap-2">
                    <div className="h-[3px] w-12 overflow-hidden rounded-full bg-white/[0.07]">
                      <div
                        className={`h-full rounded-full ${h.contribution >= 0 ? "bg-gain/80" : "bg-loss/80"}`}
                        style={{ width: `${Math.min((Math.abs(h.contribution) / maxContrib) * 100, 100)}%` }}
                      />
                    </div>
                    <span className={`font-mono text-xs tabular ${signClass(h.contribution)}`}>
                      {fmtPct(h.contribution)}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="flex justify-end">
                    <Sparkline values={h.values} positive={positive} />
                  </div>
                </td>
              </motion.tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
