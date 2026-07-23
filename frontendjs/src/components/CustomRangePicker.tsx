"use client";

/**
 * Free-form window picker for the performance chart's range control —
 * sits alongside the 1M/6M/1Y/3Y/5Y pills and lets the user type any
 * amount + unit (days/months/years) instead of a fixed preset.
 */

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import type { RangeUnit } from "@/lib/types";

interface Props {
  active: boolean;
  amount: number;
  unit: RangeUnit;
  onApply: (amount: number, unit: RangeUnit) => void;
}

const UNIT_OPTIONS: { id: RangeUnit; label: string }[] = [
  { id: "days", label: "Days" },
  { id: "months", label: "Months" },
  { id: "years", label: "Years" },
];

const UNIT_ABBREV: Record<RangeUnit, string> = {
  days: "D",
  months: "M",
  years: "Y",
};

export function CustomRangePicker({ active, amount, unit, onApply }: Props) {
  const [open, setOpen] = useState(false);
  const [draftAmount, setDraftAmount] = useState(String(amount));
  const [draftUnit, setDraftUnit] = useState<RangeUnit>(unit);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDocMouseDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [open]);

  function toggleOpen() {
    setOpen((o) => {
      if (!o) {
        // Reset the draft to the last-applied values each time the popover opens.
        setDraftAmount(String(amount));
        setDraftUnit(unit);
      }
      return !o;
    });
  }

  function submit() {
    const n = Math.floor(Number(draftAmount));
    if (!Number.isFinite(n) || n <= 0) return;
    onApply(n, draftUnit);
    setOpen(false);
  }

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={toggleOpen}
        className={`flex items-center gap-1 rounded-[10px] px-2.5 py-1 font-mono text-[10px] tracking-wider transition-all ${
          active ? "bg-accent font-semibold text-[#0a0f07]" : "text-mut hover:text-ink"
        }`}
      >
        {active ? `${amount}${UNIT_ABBREV[unit]}` : "Custom"}
        <ChevronDown className={`size-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.98 }}
            transition={{ duration: 0.14, ease: [0.16, 1, 0.3, 1] }}
            className="absolute right-0 top-full z-50 mt-2 w-56 rounded-2xl border border-line bg-surface/95 p-3 shadow-[0_24px_60px_-12px_rgba(0,0,0,0.8)] backdrop-blur-xl"
          >
            <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-mut/70">
              Custom range
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={1}
                step={1}
                value={draftAmount}
                onChange={(e) => setDraftAmount(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && submit()}
                className="w-16 rounded-lg border border-line bg-overlay/[0.04] px-2 py-1.5 font-mono text-sm tabular text-ink outline-none focus:border-accent/50"
              />
              <select
                value={draftUnit}
                onChange={(e) => setDraftUnit(e.target.value as RangeUnit)}
                className="flex-1 rounded-lg border border-line bg-overlay/[0.04] px-2 py-1.5 font-mono text-xs uppercase tracking-wider text-ink outline-none focus:border-accent/50"
              >
                {UNIT_OPTIONS.map((o) => (
                  <option key={o.id} value={o.id} className="bg-surface text-ink">
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
            <button
              type="button"
              onClick={submit}
              className="mt-2.5 w-full rounded-lg bg-accent py-1.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-[#0a0f07] transition-opacity hover:opacity-90"
            >
              Apply
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
