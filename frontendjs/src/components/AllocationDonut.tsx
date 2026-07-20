"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { fmtPct } from "@/lib/format";

export interface AllocationItem {
  symbol: string;
  name: string;
  sector: string;
  weight: number; // 0..1
}

export const ALLOC_COLORS = [
  "#B3F34C",
  "#5EEAD4",
  "#7DD3FC",
  "#C4B5FD",
  "#FCD34D",
  "#FDA4AF",
  "#6EE7B7",
  "#93C5FD",
  "#F0ABFC",
  "#FB923C",
];

const OTHER_COLOR = "#5b6560";
const R = 64;
const C = 2 * Math.PI * R;
const LABEL_R = R + 24;
const MIN_LABEL_WEIGHT = 0.03; // don't label slices smaller than 3%

export function AllocationDonut({ items }: { items: AllocationItem[] }) {
  const [active, setActive] = useState<number | null>(null);

  const slices = useMemo(() => {
    const top = items.filter((i) => i.weight > 0.0001).slice(0, 9);
    const rest = items.slice(top.length);
    const restW = rest.reduce((s, r) => s + r.weight, 0);
    const list = top.map((t, i) => ({ ...t, color: ALLOC_COLORS[i % ALLOC_COLORS.length] }));
    if (restW > 0.0001) {
      list.push({
        symbol: "OTHER",
        name: `${rest.length} more positions`,
        sector: "—",
        weight: restW,
        color: OTHER_COLOR,
      });
    }
    const out: ((typeof list)[number] & { offset: number })[] = [];
    let cum = 0;
    for (const s of list) {
      out.push({ ...s, offset: cum });
      cum += s.weight;
    }
    return out;
  }, [items]);

  const activeSlice = active != null ? slices[active] : null;

  return (
    <div className="flex flex-col gap-5 sm:flex-row sm:items-center">
      <div className="relative mx-auto shrink-0 sm:mx-0" style={{ width: 220, height: 220 }}>
        <svg viewBox="0 0 200 200" className="h-full w-full -rotate-90">
          {slices.map((s, i) => {
            const len = Math.max(s.weight * C - 2.5, 0.5);
            const midAngle = (s.offset + s.weight / 2) * 2 * Math.PI;
            const lx = 100 + LABEL_R * Math.sin(midAngle);
            const ly = 100 - LABEL_R * Math.cos(midAngle);
            const showLabel = s.weight >= MIN_LABEL_WEIGHT;
            // text-anchor based on which side of center the label falls
            const anchor = lx > 105 ? "start" : lx < 95 ? "end" : "middle";

            return (
              <g key={s.symbol}>
                <motion.circle
                  cx="100"
                  cy="100"
                  r={R}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={active === i ? 28 : 20}
                  strokeDasharray={`${len} ${C - len}`}
                  strokeDashoffset={-s.offset * C}
                  strokeLinecap="butt"
                  initial={{ opacity: 0, scale: 0.92 }}
                  animate={{ opacity: active == null || active === i ? 1 : 0.35, scale: 1 }}
                  transition={{ duration: 0.35, delay: i * 0.03 }}
                  onMouseEnter={() => setActive(i)}
                  onMouseLeave={() => setActive(null)}
                  style={{ transformOrigin: "100px 100px", cursor: "pointer" }}
                />
                {showLabel && (
                  <>
                    {/* connector line */}
                    <line
                      x1={100 + (R + 10) * Math.sin(midAngle)}
                      y1={100 - (R + 10) * Math.cos(midAngle)}
                      x2={lx}
                      y2={ly}
                      stroke={s.color}
                      strokeWidth={1}
                      opacity={active == null || active === i ? 0.5 : 0.15}
                    />
                    {/* label text */}
                    <text
                      x={lx}
                      y={ly}
                      textAnchor={anchor}
                      dominantBaseline="central"
                      className="fill-ink/80 font-mono text-[10px] font-semibold pointer-events-none"
                      style={{
                        opacity: active == null || active === i ? 1 : 0.25,
                        transition: "opacity 0.2s",
                      }}
                    >
                      {s.symbol}
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </svg>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center text-center">
          {activeSlice ? (
            <>
              <span className="font-mono text-lg font-semibold tabular text-ink">
                {fmtPct(activeSlice.weight, 1, false)}
              </span>
              <span className="font-mono text-[10px] uppercase tracking-widest text-mut">
                {activeSlice.symbol}
              </span>
            </>
          ) : (
            <>
              <span className="font-mono text-2xl font-semibold tabular text-ink">
                {items.length}
              </span>
              <span className="font-mono text-[10px] uppercase tracking-widest text-mut">
                positions
              </span>
            </>
          )}
        </div>
      </div>

      <ul className="scroll-slim max-h-[200px] min-w-0 flex-1 space-y-0.5 overflow-y-auto pr-1">
        {slices.map((s, i) => (
          <li
            key={s.symbol}
            onMouseEnter={() => setActive(i)}
            onMouseLeave={() => setActive(null)}
            className={`flex cursor-default items-center gap-2.5 rounded-lg px-2 py-1.5 transition-colors ${
              active === i ? "bg-white/[0.06]" : ""
            }`}
          >
            <span
              className="size-2 shrink-0 rounded-full"
              style={{ backgroundColor: s.color }}
            />
            <span className="w-[68px] shrink-0 font-mono text-xs font-semibold text-ink/90">
              {s.symbol}
            </span>
            <span className="min-w-0 flex-1 truncate text-[11px] text-mut">{s.name}</span>
            <div className="relative h-[3px] w-14 shrink-0 overflow-hidden rounded-full bg-white/[0.07]">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{ width: `${s.weight * 100}%`, backgroundColor: s.color }}
              />
            </div>
            <span className="w-12 shrink-0 text-right font-mono text-xs tabular text-ink/80">
              {fmtPct(s.weight, 1, false)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function SectorBars({
  sectors,
}: {
  sectors: { sector: string; weight: number }[];
}) {
  const max = Math.max(...sectors.map((s) => s.weight), 0.0001);
  return (
    <div className="mt-5 space-y-2 border-t border-line pt-4">
      <div className="mb-2 font-mono text-[10px] uppercase tracking-widest text-mut">
        Sector exposure
      </div>
      {sectors.slice(0, 8).map((s, i) => (
        <div key={s.sector} className="flex items-center gap-3">
          <span className="w-40 shrink-0 truncate text-[11px] text-mut">{s.sector}</span>
          <div className="h-[5px] min-w-0 flex-1 overflow-hidden rounded-full bg-white/[0.05]">
            <motion.div
              className="h-full rounded-full bg-gradient-to-r from-accent/50 to-accent"
              initial={{ width: 0 }}
              animate={{ width: `${(s.weight / max) * 100}%` }}
              transition={{ duration: 0.7, delay: i * 0.05, ease: [0.16, 1, 0.3, 1] }}
            />
          </div>
          <span className="w-12 shrink-0 text-right font-mono text-[11px] tabular text-ink/80">
            {fmtPct(s.weight, 1, false)}
          </span>
        </div>
      ))}
    </div>
  );
}
