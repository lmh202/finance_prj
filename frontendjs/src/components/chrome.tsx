"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import {
  Activity,
  Briefcase,
  Flag,
  HeartPulse,
  Home,
  LineChart,
  Moon,
  Rss,
  Scale,
  Search,
  Sun,
  TrendingUp,
  type LucideIcon,
} from "lucide-react";
import { useTheme } from "@/components/ThemeProvider";

/* ------------------------------------------------------------------ */
/* Logo                                                                */
/* ------------------------------------------------------------------ */

export function AuroraMark({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden>
      <defs>
        <linearGradient id="aurora-g" x1="4" y1="26" x2="28" y2="26" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#B3F34C" />
          <stop offset="0.6" stopColor="#5EEAD4" />
          <stop offset="1" stopColor="#8DA2FB" />
        </linearGradient>
      </defs>
      <path d="M7 25L16 6L25 25H7Z" stroke="url(#aurora-g)" strokeWidth="1.6" strokeLinejoin="round" />
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

const NAV: { href: string; label: string; icon: LucideIcon }[] = [
  { href: "/", label: "Home", icon: Home },
  { href: "/portfolio", label: "Portfolio", icon: Briefcase },
  { href: "/health", label: "Health", icon: HeartPulse },
  { href: "/strategy", label: "Strategy", icon: TrendingUp },
  { href: "/news", label: "News", icon: Rss },
  { href: "/react", label: "React?", icon: Scale },
  { href: "/performance", label: "Performance", icon: Flag },
];

function NavLink({
  href,
  label,
  icon: Icon,
  active,
  alwaysLabel,
}: {
  href: string;
  label: string;
  icon: LucideIcon;
  active: boolean;
  alwaysLabel?: boolean;
}) {
  return (
    <Link
      href={href}
      title={label}
      className={`flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors ${
        active
          ? "border-accent/30 bg-accent-dim font-medium text-accent"
          : "border-transparent text-mut hover:bg-overlay/[0.04] hover:text-ink"
      }`}
    >
      <Icon className="size-3" />
      <span className={alwaysLabel ? "" : "hidden xl:block"}>{label}</span>
    </Link>
  );
}

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();
  const toLight = theme === "dark";
  return (
    <button
      type="button"
      onClick={toggleTheme}
      title={toLight ? "Switch to light mode" : "Switch to dark mode"}
      aria-label={toLight ? "Switch to light mode" : "Switch to dark mode"}
      className="flex size-7 shrink-0 items-center justify-center rounded-full border border-line text-mut transition-colors hover:border-accent/40 hover:text-accent"
    >
      {toLight ? <Sun className="size-3.5" /> : <Moon className="size-3.5" />}
    </button>
  );
}

export function Header({
  source,
  asOf,
}: {
  source?: "live" | "mixed" | "simulated" | null;
  asOf?: string;
}) {
  const pathname = usePathname();
  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  const chip =
    source === "live"
      ? { dot: "bg-gain", text: `Live feed · ${asOf ?? ""}`, cls: "border-gain/25 bg-gain/10 text-gain" }
      : source === "mixed"
        ? { dot: "bg-warn", text: "Partial live feed", cls: "border-warn/25 bg-warn/10 text-warn" }
        : source === "simulated"
          ? { dot: "bg-warn", text: "Simulated feed · offline", cls: "border-warn/25 bg-warn/10 text-warn" }
          : null;

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg0/80 backdrop-blur-xl">
      <div className="mx-auto max-w-[1480px] px-4 lg:px-8">
        <div className="flex h-14 items-center gap-3">
          <Link href="/" className="flex items-center gap-2.5" aria-label="Aurora home">
            <AuroraMark />
            <span className="text-[17px] font-bold tracking-tight">Aurora</span>
            <span className="mt-0.5 hidden font-mono text-[9px] uppercase tracking-[0.22em] text-mut sm:block">
              portfolio intelligence
            </span>
          </Link>

          <nav className="ml-3 hidden items-center gap-0.5 md:flex">
            {NAV.map((item) => (
              <NavLink key={item.href} {...item} active={isActive(item.href)} />
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2.5">
            {chip && (
              <div className={`flex items-center gap-2 rounded-full border px-3 py-1 font-mono text-[10px] uppercase tracking-wider ${chip.cls}`}>
                <span className={`pulse-dot size-1.5 rounded-full ${chip.dot}`} />
                {chip.text}
              </div>
            )}
            <ThemeToggle />
          </div>
        </div>

        {/* small screens: scrollable nav row */}
        <nav className="scroll-slim -mx-1 flex items-center gap-0.5 overflow-x-auto px-1 pb-2 md:hidden">
          {NAV.map((item) => (
            <NavLink key={item.href} {...item} active={isActive(item.href)} alwaysLabel />
          ))}
        </nav>
      </div>
    </header>
  );
}

/* ------------------------------------------------------------------ */
/* Hero (home entry)                                                   */
/* ------------------------------------------------------------------ */

const STEPS = [
  {
    icon: Search,
    title: "Find your stocks",
    body: "Search the live NASDAQ / NYSE universe — every ticker you own, from Apple to a 60/40 ETF mix.",
  },
  {
    icon: Briefcase,
    title: "Build your portfolio",
    body: "Picking a stock opens the portfolio builder — enter your share count; average cost is optional.",
  },
  {
    icon: LineChart,
    title: "Read the truth",
    body: "Blended trend vs SPY & QQQ, Sharpe, drawdowns and monthly heat — all on the Portfolio page.",
  },
];

export function Hero({
  searchSlot,
  presetsSlot,
  error,
}: {
  searchSlot: ReactNode;
  presetsSlot?: ReactNode;
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
            "linear-gradient(color-mix(in srgb, var(--overlay-base) 2.5%, transparent) 1px, transparent 1px), linear-gradient(90deg, color-mix(in srgb, var(--overlay-base) 2.5%, transparent) 1px, transparent 1px)",
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
          stroke="var(--color-accent)"
          strokeWidth="1.5"
          strokeDasharray="6 8"
          style={{ animation: "dash-drift 30s linear infinite" }}
        />
        <path
          d="M0 350 L100 340 L200 345 L300 320 L400 330 L500 300 L600 315 L700 285 L800 300 L900 270 L1000 285 L1100 255 L1200 265"
          stroke="var(--color-spy)"
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
          <span className="bg-gradient-to-r from-accent via-gain to-spy bg-clip-text text-transparent">
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
          Bring the stocks you actually own. Aurora refracts them into allocation,
          risk and blended performance — measured honestly against the market.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.24, ease: [0.16, 1, 0.3, 1] }}
          className="mx-auto mt-9 max-w-xl"
        >
          {searchSlot}
          {presetsSlot && <div className="mt-4">{presetsSlot}</div>}
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
          <AuroraMark size={16} />
          <span>Aurora — portfolio intelligence. Split-adjusted daily closes · Nasdaq &amp; Yahoo Finance.</span>
        </div>
        <div className="flex items-center gap-3 font-mono text-[10px] uppercase tracking-wider text-mut/70">
          <span>Not investment advice</span>
          <span className="size-1 rounded-full bg-mut/40" />
          <span className="flex items-center gap-1">
            <Activity className="size-3 text-accent/70" />
            powered by the AURORA engine API
          </span>
        </div>
      </div>
    </footer>
  );
}
