"use client";

/** Tiny inline trend line for table rows (indexed price series). */
export function Sparkline({ values, positive }: { values: number[]; positive: boolean }) {
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
