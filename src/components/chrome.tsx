"use client";

import type { ReactNode } from "react";
import { motion } from "framer-motion";
import {
  LineChart,
  Rss,
  Search,
  SlidersHorizontal,
  Activity,
} from "lucide-react";

/* ------------------------------------------------------------------ */
/* Logo                                                                */
/* ------------------------------------------------------------------ */

export function PrismMark({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden>
      <defs>
        <linearGradient id="prism-g" x1="4" y1="26" x2="28" y2="26" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#B3F34C" />
          <stop offset="0.6" stopColor="#5EEAD4" />
          <stop offset="1" stopColor="#8DA2FB" />
        </linearGradient>
      </defs>
      <path d="M7 25L16 6L25 25H7Z" stroke="url(#prism-g)" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M2 16.5H9.5" stroke="#B3F34C" strokeWidth="1.4" strokeLinecap="round" opacity="0.9" />
      <path d="M13.6 16.5L21 21.5" stroke="#5EEAD4" strokeWidth="1.3" strokeLinecap="round" opacity="0.75" />
      <path d="M13.6 16.5L23.5 13" stroke="#8DA2FB" strokeWidth="1.3" strokeLinecap="round" opacity="0.75" />
      <path d="M13.6 16.5L25 17.5" stroke="#CFA3FF" strokeWidth="1.1" strokeLinecap="round" opacity="0.55" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Header                                                              */
/* ------------------------------------------------------------------ */

export function Header({
  source,
  asOf,
}: {
  source: "live" | "mixed" | "simulated" | null;
  asOf?: string;
}) {
  const chip =
    source === "live"
      ? { dot: "bg-gain", text: `Live feed · ${asOf ?? ""}`, cls: "border-gain/25 bg-gain/10 text-gain" }
      : source === "mixed"
        ? { dot: "bg-amber-300", text: "Partial live feed", cls: "border-amber-300/25 bg-amber-300/10 text-amber-200" }
        : source === "simulated"
          ? { dot: "bg-amber-300", text: "Simulated feed · offline", cls: "border-amber-300/25 bg-amber-300/10 text-amber-200" }
          : null;

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg0/80 backdrop-blur-xl">
      <div className="mx-auto flex h-14 max-w-[1480px] items-center gap-4 px-4 lg:px-8">
        <a href="#" className="flex items-center gap-2.5" aria-label="Prism home">
          <PrismMark />
          <span className="text-[17px] font-bold tracking-tight">Prism</span>
          <span className="mt-0.5 hidden font-mono text-[9px] uppercase tracking-[0.22em] text-mut sm:block">
            portfolio intelligence
          </span>
        </a>

        <nav className="ml-6 hidden items-center gap-1 md:flex">
          <span className="flex items-center gap-1.5 rounded-full border border-accent/30 bg-accent-dim px-3 py-1 text-xs font-medium text-accent">
            <Activity className="size-3" />
            Analyzer
          </span>
          <span
            className="flex cursor-not-allowed items-center gap-1.5 rounded-full px-3 py-1 text-xs text-mut/70"
            title="News-driven rebalancing signals — next milestone"
          >
            <Rss className="size-3" />
            News signals
            <span className="rounded-full border border-amber-300/25 bg-amber-300/10 px-1.5 py-px font-mono text-[9px] uppercase tracking-wider text-amber-200/80">
              RSS · soon
            </span>
          </span>
        </nav>

        <div className="ml-auto">
          {chip && (
            <div className={`flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-[10px] uppercase tracking-wider ${chip.cls}`}>
              <span className={`pulse-dot size-1.5 rounded-full ${chip.dot}`} />
              {chip.text}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

/* ------------------------------------------------------------------ */
/* Presets                                                             */
/* ------------------------------------------------------------------ */

export interface PresetHolding {
  symbol: string;
  value: number;
  name?: string;
  quoteType?: string;
}

export interface Preset {
  label: string;
  holdings: PresetHolding[];
}

export const PRESETS: Preset[] = [
  {
    label: "Magnificent 7",
    holdings: [
      { symbol: "AAPL", value: 15, name: "Apple Inc." },
      { symbol: "MSFT", value: 15, name: "Microsoft Corporation" },
      { symbol: "NVDA", value: 15, name: "NVIDIA Corporation" },
      { symbol: "AMZN", value: 14, name: "Amazon.com Inc." },
      { symbol: "META", value: 14, name: "Meta Platforms Inc." },
      { symbol: "GOOGL", value: 13, name: "Alphabet Inc." },
      { symbol: "TSLA", value: 14, name: "Tesla Inc." },
    ],
  },
  {
    label: "AI & Semis",
    holdings: [
      { symbol: "NVDA", value: 20, name: "NVIDIA Corporation" },
      { symbol: "AVGO", value: 14, name: "Broadcom Inc." },
      { symbol: "TSM", value: 14, name: "Taiwan Semiconductor" },
      { symbol: "ASML", value: 13, name: "ASML Holding" },
      { symbol: "AMD", value: 13, name: "Advanced Micro Devices" },
      { symbol: "MU", value: 9, name: "Micron Technology" },
      { symbol: "MRVL", value: 9, name: "Marvell Technology" },
      { symbol: "QCOM", value: 8, name: "QUALCOMM" },
    ],
  },
  {
    label: "Buffett Core",
    holdings: [
      { symbol: "AAPL", value: 32, name: "Apple Inc." },
      { symbol: "AXP", value: 14, name: "American Express" },
      { symbol: "BAC", value: 12, name: "Bank of America" },
      { symbol: "KO", value: 11, name: "Coca-Cola" },
      { symbol: "CVX", value: 10, name: "Chevron" },
      { symbol: "OXY", value: 7, name: "Occidental Petroleum" },
      { symbol: "MCO", value: 5, name: "Moody's" },
      { symbol: "CB", value: 5, name: "Chubb" },
      { symbol: "V", value: 4, name: "Visa" },
    ],
  },
  {
    label: "Dividend Fortress",
    holdings: [
      { symbol: "SCHD", value: 14, name: "Schwab U.S. Dividend Equity ETF", quoteType: "ETF" },
      { symbol: "KO", value: 11, name: "Coca-Cola" },
      { symbol: "PEP", value: 11, name: "PepsiCo" },
      { symbol: "PG", value: 11, name: "Procter & Gamble" },
      { symbol: "JNJ", value: 11, name: "Johnson & Johnson" },
      { symbol: "MCD", value: 10, name: "McDonald's" },
      { symbol: "MO", value: 8, name: "Altria" },
      { symbol: "VZ", value: 8, name: "Verizon" },
      { symbol: "O", value: 8, name: "Realty Income" },
      { symbol: "OKE", value: 8, name: "ONEOK" },
    ],
  },
  {
    label: "Classic 60/40",
    holdings: [
      { symbol: "VTI", value: 60, name: "Vanguard Total Stock Market ETF", quoteType: "ETF" },
      { symbol: "AGG", value: 40, name: "iShares Core U.S. Aggregate Bond ETF", quoteType: "ETF" },
    ],
  },
];

/* ------------------------------------------------------------------ */
/* Hero (empty state)                                                  */
/* ------------------------------------------------------------------ */

const STEPS = [
  {
    icon: Search,
    title: "Find your stocks",
    body: "Search the live NASDAQ / NYSE universe — every ticker you own, from Apple to a 60/40 ETF mix.",
  },
  {
    icon: SlidersHorizontal,
    title: "Mirror your positions",
    body: "Dial in weights or exact share counts. We normalize and price them against the latest close.",
  },
  {
    icon: LineChart,
    title: "Read the truth",
    body: "Blended trend vs SPY & QQQ, annual return, Sharpe, drawdowns and monthly heat — one honest picture.",
  },
];

export function Hero({
  searchSlot,
  presetsSlot,
  error,
}: {
  searchSlot: ReactNode;
  presetsSlot: ReactNode;
  error: string | null;
}) {
  return (
    <div className="relative overflow-hidden">
      {/* backdrop art */}
      <div
        aria-hidden
        className="absolute inset-0 opacity-[0.55]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)",
          backgroundSize: "72px 72px",
          maskImage: "radial-gradient(ellipse 80% 60% at 50% 30%, black 30%, transparent 75%)",
          WebkitMaskImage: "radial-gradient(ellipse 80% 60% at 50% 30%, black 30%, transparent 75%)",
        }}
      />
      <svg
        aria-hidden
        className="absolute inset-x-0 top-24 mx-auto w-[1200px] max-w-none opacity-[0.16]"
        viewBox="0 0 1200 400"
        fill="none"
      >
        <path
          d="M0 320 L80 300 L160 310 L240 260 L320 280 L400 220 L480 250 L560 190 L640 220 L720 160 L800 190 L880 140 L960 170 L1040 110 L1120 140 L1200 90"
          stroke="#B3F34C"
          strokeWidth="1.5"
          strokeDasharray="6 8"
          style={{ animation: "dash-drift 30s linear infinite" }}
        />
        <path
          d="M0 350 L100 340 L200 345 L300 320 L400 330 L500 300 L600 315 L700 285 L800 300 L900 270 L1000 285 L1100 255 L1200 265"
          stroke="#8DA2FB"
          strokeWidth="1.2"
          strokeDasharray="3 10"
          style={{ animation: "dash-drift 40s linear infinite" }}
        />
      </svg>

      <div className="relative mx-auto max-w-3xl px-4 pb-20 pt-20 text-center sm:pt-28">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          className="mx-auto mb-6 inline-flex items-center gap-2 rounded-full border border-accent/25 bg-accent-dim px-4 py-1.5 font-mono text-[10px] uppercase tracking-[0.22em] text-accent"
        >
          <span className="pulse-dot size-1.5 rounded-full bg-accent" />
          Live market data · 5Y history · SPY / QQQ benchmarks
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.08, ease: [0.16, 1, 0.3, 1] }}
          className="text-balance text-5xl font-bold leading-[1.02] tracking-tight sm:text-7xl"
        >
          See your portfolio
          <br />
          in a{" "}
          <span className="bg-gradient-to-r from-accent via-[#5eead4] to-[#8da2fb] bg-clip-text text-transparent">
            new light
          </span>
          .
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.16, ease: [0.16, 1, 0.3, 1] }}
          className="mx-auto mt-5 max-w-xl text-pretty text-[15px] leading-relaxed text-mut"
        >
          Bring the stocks you actually own. Prism refracts them into allocation,
          risk and blended performance — measured honestly against the market.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.24, ease: [0.16, 1, 0.3, 1] }}
          className="mx-auto mt-9 max-w-xl"
        >
          {searchSlot}
          <div className="mt-4">{presetsSlot}</div>
          {error && (
            <div className="mt-4 rounded-xl border border-loss/30 bg-loss/10 px-4 py-3 text-sm text-loss">
              {error}
            </div>
          )}
        </motion.div>

        <div className="mx-auto mt-16 grid max-w-2xl gap-3 text-left sm:grid-cols-3">
          {STEPS.map((s, i) => (
            <motion.div
              key={s.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.34 + i * 0.08, ease: [0.16, 1, 0.3, 1] }}
              className="card px-4 py-4"
            >
              <s.icon className="size-4 text-accent" />
              <div className="mt-2.5 text-sm font-semibold">{s.title}</div>
              <p className="mt-1 text-[11.5px] leading-relaxed text-mut">{s.body}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Footer                                                              */
/* ------------------------------------------------------------------ */

export function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-[1480px] flex-col items-center justify-between gap-3 px-4 py-6 text-center sm:flex-row sm:text-left lg:px-8">
        <div className="flex items-center gap-2 text-xs text-mut">
          <PrismMark size={16} />
          <span>Prism — portfolio intelligence. Split-adjusted daily closes · Nasdaq &amp; Yahoo Finance.</span>
        </div>
        <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-wider text-mut/70">
          <span>Not investment advice</span>
          <span className="size-1 rounded-full bg-mut/40" />
          <span className="flex items-center gap-1">
            <Rss className="size-3 text-amber-200/70" />
            next: RSS news signals → rebalancing
          </span>
        </div>
      </div>
    </footer>
  );
}
