"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Scale, Trash2, X, PackageOpen } from "lucide-react";
import type { InputMode } from "@/lib/types";
import { fmtPct, fmtPrice } from "@/lib/format";

export interface Holding {
  symbol: string;
  value: number;
  name?: string;
  quoteType?: string;
}

export interface HoldingMeta {
  weight: number;
  lastPrice: number | null;
  simulated: boolean;
  name: string;
  sector: string;
}

interface Props {
  holdings: Holding[];
  mode: InputMode;
  meta: Map<string, HoldingMeta> | null;
  onModeChange: (m: InputMode) => void;
  onValueChange: (symbol: string, v: number) => void;
  onRemove: (symbol: string) => void;
  onClear: () => void;
  onEqualize: () => void;
}

export function HoldingsPanel({
  holdings,
  mode,
  meta,
  onModeChange,
  onValueChange,
  onRemove,
  onClear,
  onEqualize,
}: Props) {
  const totalInput = holdings.reduce((s, h) => s + h.value, 0);
  const totalValue =
    meta != null
      ? holdings.reduce((s, h) => {
          const px = meta.get(h.symbol)?.lastPrice ?? 0;
          return s + (mode === "shares" ? h.value * px : 0);
        }, 0)
      : 0;

  return (
    <div className="card flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex items-center gap-3 border-b border-line px-4 py-3">
        <h2 className="text-sm font-semibold tracking-wide">Positions</h2>
        <span className="rounded-full border border-accent/30 bg-accent-dim px-2 py-0.5 font-mono text-[11px] tabular text-accent">
          {holdings.length}
        </span>
        <div className="ml-auto flex items-center gap-1.5">
          <div className="flex rounded-lg border border-line bg-white/[0.03] p-0.5">
            {(["weight", "shares"] as const).map((m) => (
              <button
                key={m}
                onClick={() => onModeChange(m)}
                className={`rounded-md px-2 py-1 font-mono text-[10px] uppercase tracking-wider transition-all ${
                  mode === m
                    ? "bg-accent text-[#0a0f07] font-semibold"
                    : "text-mut hover:text-ink"
                }`}
              >
                {m === "weight" ? "% Wt" : "Shares"}
              </button>
            ))}
          </div>
          <button
            onClick={onEqualize}
            title="Equal weight all positions"
            className="rounded-lg border border-line bg-white/[0.03] p-1.5 text-mut transition-colors hover:border-accent/40 hover:text-accent"
          >
            <Scale className="size-3.5" />
          </button>
          <button
            onClick={onClear}
            title="Clear all"
            className="rounded-lg border border-line bg-white/[0.03] p-1.5 text-mut transition-colors hover:border-loss/40 hover:text-loss"
          >
            <Trash2 className="size-3.5" />
          </button>
        </div>
      </div>

      <div className="scroll-slim min-h-0 flex-1 overflow-y-auto p-2">
        {holdings.length === 0 ? (
          <div className="flex h-full min-h-[180px] flex-col items-center justify-center gap-2 text-center">
            <PackageOpen className="size-6 text-mut/60" />
            <p className="max-w-[200px] text-xs leading-relaxed text-mut">
              Search above to add the stocks you own, or load a preset below.
            </p>
          </div>
        ) : (
          <ul className="space-y-1">
            <AnimatePresence initial={false}>
              {holdings.map((h) => {
                const m = meta?.get(h.symbol);
                const isWeight = mode === "weight";
                const sliderMax = 100;
                const fill = isWeight
                  ? Math.min((h.value / sliderMax) * 100, 100)
                  : 0;
                return (
                  <motion.li
                    key={h.symbol}
                    layout
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, x: -16, height: 0, marginBottom: 0 }}
                    transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
                    className="group rounded-xl border border-transparent bg-white/[0.025] px-3 py-2.5 transition-colors hover:border-line hover:bg-white/[0.045]"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-semibold tracking-wide text-accent">
                        {h.symbol}
                      </span>
                      {m?.simulated && (
                        <span className="rounded border border-amber-300/30 bg-amber-300/10 px-1 py-px font-mono text-[9px] uppercase tracking-wider text-amber-200/90">
                          sim
                        </span>
                      )}
                      <span className="min-w-0 flex-1 truncate text-[11px] text-mut">
                        {m?.name ?? h.name ?? ""}
                      </span>
                      <input
                        type="number"
                        min={0}
                        step={isWeight ? 0.5 : 1}
                        value={Number.isFinite(h.value) ? h.value : 0}
                        onChange={(e) =>
                          onValueChange(h.symbol, Math.max(0, Number(e.target.value) || 0))
                        }
                        className="w-[72px] rounded-md border border-line bg-black/30 px-2 py-1 text-right font-mono text-xs tabular outline-none transition-colors focus:border-accent/60"
                        aria-label={`${h.symbol} ${isWeight ? "weight percent" : "shares"}`}
                      />
                      <span className="w-6 font-mono text-[10px] text-mut">
                        {isWeight ? "%" : "sh"}
                      </span>
                      <button
                        onClick={() => onRemove(h.symbol)}
                        className="rounded p-1 text-mut/50 opacity-0 transition-all hover:bg-loss/10 hover:text-loss group-hover:opacity-100"
                        aria-label={`Remove ${h.symbol}`}
                      >
                        <X className="size-3.5" />
                      </button>
                    </div>
                    <div className="mt-2 flex items-center gap-3">
                      {isWeight ? (
                        <input
                          type="range"
                          min={0}
                          max={sliderMax}
                          step={0.5}
                          value={Math.min(h.value, sliderMax)}
                          onChange={(e) => onValueChange(h.symbol, Number(e.target.value))}
                          className="slider min-w-0 flex-1"
                          style={{ ["--fill" as string]: `${fill}%` }}
                          aria-label={`${h.symbol} weight slider`}
                        />
                      ) : (
                        <div className="h-[3px] min-w-0 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
                          <div
                            className="h-full rounded-full bg-accent/70 transition-all duration-300"
                            style={{ width: `${Math.min((m?.weight ?? 0) * 100, 100)}%` }}
                          />
                        </div>
                      )}
                      <span className="flex shrink-0 items-baseline gap-2 font-mono text-[10px] tabular text-mut">
                        {m?.lastPrice != null && (
                          <span>${fmtPrice(m.lastPrice)}</span>
                        )}
                        {m && (
                          <span className="text-accent/90">
                            {fmtPct(m.weight, 1, false)}
                          </span>
                        )}
                      </span>
                    </div>
                  </motion.li>
                );
              })}
            </AnimatePresence>
          </ul>
        )}
      </div>

      {holdings.length > 0 && (
        <div className="border-t border-line px-4 py-2.5">
          {mode === "weight" ? (
            <div className="flex items-center justify-between font-mono text-[11px] tabular">
              <span className="text-mut">
                Σ {totalInput.toFixed(1)}%
                {Math.abs(totalInput - 100) > 0.05 && (
                  <span className="text-amber-200/80"> · normalized to 100%</span>
                )}
              </span>
              <div className="h-1 w-24 overflow-hidden rounded-full bg-white/[0.07]">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    Math.abs(totalInput - 100) <= 0.05 ? "bg-gain" : "bg-amber-300"
                  }`}
                  style={{ width: `${Math.min(totalInput, 100)}%` }}
                />
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-between font-mono text-[11px] tabular">
              <span className="text-mut">Portfolio value</span>
              <span className="text-ink">
                {totalValue > 0
                  ? `$${totalValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}`
                  : "—"}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
