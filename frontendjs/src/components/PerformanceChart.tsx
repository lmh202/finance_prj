"use client";

import { useId, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { fmtDate, fmtPct, clamp } from "@/lib/format";

export interface ChartSeries {
  id: string;
  label: string;
  color: string;
  values: number[];
  width?: number;
}

interface Props {
  dates: string[];
  series: ChartSeries[];
  height?: number;
}

const W = 1000;
const H = 380;

export function PerformanceChart({ dates, series, height = 380 }: Props) {
  const gid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const ref = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<number | null>(null);

  const n = dates.length;
  const primary = series[0];

  const { yMin, yMax } = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of series) {
      for (const v of s.values) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    if (!Number.isFinite(lo)) {
      lo = 90;
      hi = 110;
    }
    const span = Math.max(hi - lo, 1);
    return { yMin: lo - span * 0.1, yMax: hi + span * 0.1 };
  }, [series]);

  const xPct = (i: number) => (n <= 1 ? 0 : (i / (n - 1)) * 100);
  const yPct = (v: number) => (1 - (v - yMin) / (yMax - yMin)) * 100;

  const pathFor = (vals: number[]) => {
    let d = "";
    for (let i = 0; i < vals.length; i++) {
      const x = (xPct(i) / 100) * W;
      const y = (yPct(vals[i]) / 100) * H;
      d += `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }
    return d;
  };

  const areaPath = useMemo(() => {
    if (!primary) return "";
    return `${pathFor(primary.values)}L${W},${H}L0,${H}Z`;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [primary?.values, yMin, yMax, n]);

  const gridLines = useMemo(() => {
    const lines: number[] = [];
    for (let k = 1; k <= 4; k++) lines.push(yMin + ((yMax - yMin) * k) / 5);
    return lines;
  }, [yMin, yMax]);

  const xLabels = useMemo(() => {
    const out: { i: number; label: string }[] = [];
    const count = 5;
    for (let k = 0; k < count; k++) {
      const i = Math.round(((n - 1) * k) / (count - 1));
      out.push({ i, label: fmtDate(dates[i], n > 400) });
    }
    return out;
  }, [dates, n]);

  function onMove(e: React.MouseEvent) {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect || n < 2) return;
    const frac = clamp((e.clientX - rect.left) / rect.width, 0, 1);
    setHover(Math.round(frac * (n - 1)));
  }

  if (!primary || n < 2) return null;

  const animKey = `${dates[0]}_${dates[n - 1]}_${series.length}`;
  const hoverX = hover != null ? xPct(hover) : 0;

  return (
    <div
      ref={ref}
      className="relative w-full select-none"
      style={{ height }}
      onMouseMove={onMove}
      onMouseLeave={() => setHover(null)}
    >
      {/* horizontal grid */}
      {gridLines.map((g, i) => (
        <div
          key={i}
          className="absolute left-0 right-0 border-t border-overlay/[0.05]"
          style={{ top: `${yPct(g)}%` }}
        >
          <span className="absolute -top-2 left-1 font-mono text-[10px] tabular text-mut/70">
            {fmtPct(g / 100 - 1, 0)}
          </span>
        </div>
      ))}

      {/* baseline = 100 */}
      {yMin < 100 && yMax > 100 && (
        <div
          className="absolute left-0 right-0 border-t border-dashed border-overlay/[0.12]"
          style={{ top: `${yPct(100)}%` }}
        />
      )}

      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full overflow-visible"
      >
        <defs>
          <linearGradient id={`g${gid}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={primary.color} stopOpacity="0.25" />
            <stop offset="100%" stopColor={primary.color} stopOpacity="0" />
          </linearGradient>
        </defs>

        <path d={areaPath} fill={`url(#g${gid})`} />

        {series.slice(1).map((s) => (
          <path
            key={s.id}
            d={pathFor(s.values)}
            fill="none"
            stroke={s.color}
            strokeWidth={s.width ?? 1.5}
            strokeOpacity={0.85}
            vectorEffect="non-scaling-stroke"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}

        <motion.path
          key={animKey}
          d={pathFor(primary.values)}
          fill="none"
          stroke={primary.color}
          strokeWidth={primary.width ?? 2.5}
          vectorEffect="non-scaling-stroke"
          strokeLinejoin="round"
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 8px color-mix(in srgb, ${primary.color} 33%, transparent))` }}
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 1.4, ease: [0.65, 0, 0.15, 1] }}
        />
      </svg>

      {/* x-axis labels */}
      {xLabels.map((l) => (
        <span
          key={l.i}
          className="absolute -bottom-1 -translate-x-1/2 font-mono text-[10px] tabular text-mut/70"
          style={{ left: `${xPct(l.i)}%` }}
        >
          {l.label}
        </span>
      ))}

      {/* crosshair + dots + tooltip */}
      {hover != null && hover < n && (
        <>
          <div
            className="pointer-events-none absolute bottom-0 top-0 w-px bg-overlay/15"
            style={{ left: `${hoverX}%` }}
          />
          {series.map((s) => {
            const last = s.values[s.values.length - 1];
            const v = s.values[hover];
            return (
              <div
                key={s.id}
                className="pointer-events-none absolute size-2 -translate-x-1/2 -translate-y-1/2 rounded-full border-2"
                style={{
                  left: `${hoverX}%`,
                  top: `${yPct(v)}%`,
                  backgroundColor: s.color,
                  borderColor: "var(--color-bg0)",
                }}
              />
            );
          })}
          <div
            className="pointer-events-none absolute top-2 z-10 -translate-x-1/2 rounded-xl border border-line bg-surface/95 px-3.5 py-2.5 shadow-2xl backdrop-blur-md"
            style={{
              left: `${clamp(hoverX, 12, 88)}%`,
            }}
          >
            <div className="mb-1.5 font-mono text-[10px] uppercase tracking-widest text-mut">
              {fmtDate(dates[hover], true)}
            </div>
            <div className="space-y-1">
              {series.map((s) => (
                <div key={s.id} className="flex items-center gap-2 whitespace-nowrap">
                  <span
                    className="size-1.5 rounded-full"
                    style={{ backgroundColor: s.color }}
                  />
                  <span className="text-[11px] text-mut">{s.label}</span>
                  <span
                    className={`ml-auto pl-4 font-mono text-xs font-medium tabular ${
                      s.values[hover] >= 100 ? "text-gain" : "text-loss"
                    }`}
                  >
                    {fmtPct(s.values[hover] / 100 - 1)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
