"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, CornerDownLeft, Globe, Loader2, Search } from "lucide-react";
import type { StockInfo } from "@/lib/types";
import { searchStocks } from "@/lib/api-client";

interface Props {
  onAdd: (info: StockInfo) => void;
  existing: Set<string>;
  big?: boolean;
  placeholder?: string;
}

const TICKER_RE = /^[A-Za-z][A-Za-z.\-]{0,11}$/;

export function SearchBox({ onAdd, existing, big, placeholder }: Props) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<StockInfo[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const query = q.trim();
    if (query.length === 0) {
      setResults([]);
      setOpen(false);
      return;
    }
    setLoading(true);
    const t = setTimeout(async () => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      try {
        const results = await searchStocks(query, ctrl.signal);
        if (!ctrl.signal.aborted) {
          setResults(results);
          setActive(0);
          setOpen(true);
        }
      } catch {
        /* aborted */
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }, 200);
    return () => clearTimeout(t);
  }, [q]);

  const exact = results.some(
    (r) => r.symbol.toUpperCase() === q.trim().toUpperCase(),
  );
  const canCustomAdd =
    !exact && TICKER_RE.test(q.trim()) && !existing.has(q.trim().toUpperCase().replace(/\./g, "-"));
  const rows = results.length + (canCustomAdd ? 1 : 0);

  function pick(info: StockInfo) {
    onAdd(info);
    setQ("");
    setResults([]);
    setOpen(false);
    inputRef.current?.focus();
  }

  function customAdd() {
    const symbol = q.trim().toUpperCase().replace(/\./g, "-");
    pick({
      symbol,
      name: symbol,
      exchange: "",
      sector: "Other",
      quoteType: "EQUITY",
      live: true,
    });
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => (rows > 0 ? (a + 1) % rows : 0));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => (rows > 0 ? (a - 1 + rows) % rows : 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (active < results.length && results[active]) {
        const r = results[active];
        if (!existing.has(r.symbol)) pick(r);
      } else if (canCustomAdd) {
        customAdd();
      }
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div className="relative w-full">
      <div
        className={`group flex items-center gap-3 rounded-2xl border border-line bg-white/[0.04] transition-all focus-within:border-accent/50 focus-within:bg-white/[0.06] focus-within:shadow-[0_0_0_4px_rgba(179,243,76,0.08),0_0_40px_-8px_rgba(179,243,76,0.25)] ${
          big ? "px-5 py-4" : "px-3.5 py-2.5"
        }`}
      >
        {loading ? (
          <Loader2 className={`${big ? "size-5" : "size-4"} animate-spin text-accent`} />
        ) : (
          <Search className={`${big ? "size-5" : "size-4"} text-mut transition-colors group-focus-within:text-accent`} />
        )}
        <input
          ref={inputRef}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onFocus={() => q.trim() && setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={onKeyDown}
          placeholder={placeholder ?? "Search any stock or ETF — Apple, NVDA, SPY…"}
          className={`w-full bg-transparent outline-none placeholder:text-mut/60 ${
            big ? "text-lg" : "text-sm"
          }`}
          aria-label="Search stocks"
        />
        {open && rows > 0 && (
          <kbd className="hidden items-center gap-1 rounded-md border border-line bg-white/5 px-1.5 py-0.5 font-mono text-[10px] text-mut sm:flex">
            <CornerDownLeft className="size-3" />
          </kbd>
        )}
      </div>

      <AnimatePresence>
        {open && rows > 0 && (
          <motion.ul
            initial={{ opacity: 0, y: -6, scale: 0.99 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.99 }}
            transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
            className="scroll-slim absolute left-0 right-0 top-full z-50 mt-2 max-h-[340px] overflow-y-auto rounded-2xl border border-line bg-[#0b1014]/95 p-1.5 shadow-[0_24px_60px_-12px_rgba(0,0,0,0.8)] backdrop-blur-xl"
          >
            {results.map((r, i) => {
              const added = existing.has(r.symbol);
              return (
                <li key={`${r.symbol}-${i}`}>
                  <button
                    type="button"
                    disabled={added}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => pick(r)}
                    onMouseEnter={() => setActive(i)}
                    className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors ${
                      active === i ? "bg-white/[0.07]" : ""
                    } ${added ? "cursor-default opacity-45" : "cursor-pointer"}`}
                  >
                    <span className="w-[76px] shrink-0 font-mono text-sm font-semibold tracking-wide text-accent">
                      {r.symbol}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-sm text-ink/90">
                      {r.name}
                    </span>
                    {r.quoteType === "ETF" && (
                      <span className="rounded-md border border-amber-300/25 bg-amber-300/10 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-amber-200">
                        ETF
                      </span>
                    )}
                    <span className="hidden shrink-0 font-mono text-[11px] text-mut sm:block">
                      {r.sector}
                    </span>
                    {added ? (
                      <Check className="size-3.5 shrink-0 text-gain" />
                    ) : r.live ? (
                      <Globe className="size-3.5 shrink-0 text-mut" />
                    ) : null}
                  </button>
                </li>
              );
            })}
            {canCustomAdd && (
              <li>
                <button
                  type="button"
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={customAdd}
                  onMouseEnter={() => setActive(results.length)}
                  className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors ${
                    active === results.length ? "bg-white/[0.07]" : ""
                  }`}
                >
                  <span className="w-[76px] shrink-0 font-mono text-sm font-semibold text-accent">
                    {q.trim().toUpperCase().replace(/\./g, "-")}
                  </span>
                  <span className="flex-1 text-sm text-mut">
                    Add as custom ticker — we&apos;ll try to resolve live data
                  </span>
                </button>
              </li>
            )}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}
