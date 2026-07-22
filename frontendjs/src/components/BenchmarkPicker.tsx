"use client";

/**
 * Multi-select dropdown for the "extra" benchmarks (anything beyond the
 * always-visible SPY/QQQ toggle buttons) — keeps the chart header from
 * growing a toggle button per benchmark as more get added server-side.
 */

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";

export interface BenchmarkOption {
  symbol: string;
  name: string;
  color: string;
}

interface Props {
  options: BenchmarkOption[];
  selected: Set<string>;
  onToggle: (symbol: string) => void;
}

export function BenchmarkPicker({ options, selected, onToggle }: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDocMouseDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [open]);

  if (options.length === 0) return null;
  const count = options.filter((o) => selected.has(o.symbol)).length;

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider transition-all ${
          count > 0
            ? "border-line bg-white/[0.05] text-ink/85"
            : "border-line/50 text-mut/45"
        }`}
      >
        More
        {count > 0 && (
          <span className="rounded-full bg-accent-dim px-1.5 text-accent">{count}</span>
        )}
        <ChevronDown className={`size-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ duration: 0.14, ease: [0.16, 1, 0.3, 1] }}
            className="absolute right-0 top-full z-50 mt-2 w-52 rounded-2xl border border-line bg-[#0b1014]/95 p-1.5 shadow-[0_24px_60px_-12px_rgba(0,0,0,0.8)] backdrop-blur-xl"
          >
            {options.map((b) => {
              const on = selected.has(b.symbol);
              return (
                <label
                  key={b.symbol}
                  className="flex cursor-pointer items-center gap-2.5 rounded-xl px-3 py-2 text-left transition-colors hover:bg-white/[0.06]"
                >
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={() => onToggle(b.symbol)}
                    className="size-3.5 accent-accent"
                  />
                  <span
                    className="size-1.5 shrink-0 rounded-full"
                    style={{ backgroundColor: b.color, opacity: on ? 1 : 0.5 }}
                  />
                  <span className="shrink-0 font-mono text-[10px] uppercase tracking-wider text-ink/85">
                    {b.symbol}
                  </span>
                  <span className="ml-auto truncate text-[11px] text-mut">{b.name}</span>
                </label>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
