"use client";

/**
 * Shared scaffolding for the AURORA engine pages (/portfolio, /health,
 * /strategy, /news, /react, /performance): page chrome + title block,
 * the three expected non-ready states (backend down, empty portfolio,
 * no price history) and small UI primitives reused across those pages.
 */

import { useEffect, useState, type ReactNode } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Briefcase,
  Loader2,
  PackageOpen,
  RefreshCw,
  ServerCrash,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";
import { Footer, Header } from "@/components/chrome";
import { EMPTY_PORTFOLIO, loadSamplePortfolio } from "@/lib/api-client";
import type { EngineStatus } from "@/lib/use-engine";
import { clamp } from "@/lib/format";

export const BTN_PRIMARY =
  "inline-flex items-center gap-2 rounded-xl bg-accent px-4 py-2 text-sm font-semibold text-[#0a0f07] transition-all hover:brightness-110 disabled:pointer-events-none disabled:opacity-50";
export const BTN_GHOST =
  "inline-flex items-center gap-2 rounded-xl border border-line bg-white/[0.04] px-4 py-2 text-sm text-ink/85 transition-colors hover:border-accent/40 hover:text-accent disabled:pointer-events-none disabled:opacity-50";

const EASE = [0.16, 1, 0.3, 1] as const;

/* ------------------------------------------------------------------ */
/* Page chrome                                                         */
/* ------------------------------------------------------------------ */

interface PageShellProps {
  title: string;
  icon: LucideIcon;
  caption: string;
  actions?: ReactNode;
  wide?: boolean;
  children: ReactNode;
}

export function PageShell({
  title,
  icon: Icon,
  caption,
  actions,
  wide,
  children,
}: PageShellProps) {
  useEffect(() => {
    document.title = `Aurora — ${title}`;
  }, [title]);

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1">
        <div
          className={`mx-auto w-full px-4 py-8 lg:px-8 ${
            wide ? "max-w-[1480px]" : "max-w-[1200px]"
          }`}
        >
          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: EASE }}
            className="flex flex-wrap items-center gap-4"
          >
            <div className="flex size-11 shrink-0 items-center justify-center rounded-2xl border border-accent/25 bg-accent-dim">
              <Icon className="size-5 text-accent" />
            </div>
            <div className="min-w-0 flex-1">
              <h1 className="text-2xl font-bold tracking-tight sm:text-[28px]">
                {title}
              </h1>
              <div className="mt-1 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                {caption}
              </div>
            </div>
            {actions}
          </motion.div>
          <div className="mt-6">{children}</div>
        </div>
      </main>
      <Footer />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Engine shell: PageShell + status handling                           */
/* ------------------------------------------------------------------ */

interface EngineShellProps {
  title: string;
  icon: LucideIcon;
  caption: string;
  engine: { status: EngineStatus; loading: boolean; reload: () => void };
  /** render children (instead of the skeleton) once this is true */
  hasData: boolean;
  wide?: boolean;
  children: ReactNode;
}

export function EngineShell({
  title,
  icon,
  caption,
  engine,
  hasData,
  wide,
  children,
}: EngineShellProps) {
  const { status, loading, reload } = engine;

  let body: ReactNode;
  if (status.phase === "marker") {
    body =
      status.marker === EMPTY_PORTFOLIO ? (
        <EmptyPortfolioState onReload={reload} />
      ) : (
        <StateCard
          icon={TriangleAlert}
          iconTone="text-amber-200"
          title="Price history unavailable"
          body="Could not load market history — check your connection and retry."
          actions={
            <button onClick={reload} className={BTN_GHOST}>
              <RefreshCw className="size-3.5" />
              Retry
            </button>
          }
        />
      );
  } else if (status.phase === "error") {
    body = status.backendDown ? (
      <StateCard
        icon={ServerCrash}
        iconTone="text-loss"
        title="Backend not running"
        body={
          <>
            Start it with{" "}
            <code className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[11px] text-ink/90">
              scripts\dev.ps1
            </code>{" "}
            or{" "}
            <code className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[11px] text-ink/90">
              uvicorn main:app --app-dir backend --port 8000
            </code>
            , then retry.
          </>
        }
        actions={
          <button onClick={reload} className={BTN_PRIMARY}>
            <RefreshCw className="size-3.5" />
            Retry
          </button>
        }
      />
    ) : (
      <StateCard
        icon={TriangleAlert}
        iconTone="text-loss"
        title="Something went wrong"
        body={status.message}
        actions={
          <button onClick={reload} className={BTN_GHOST}>
            <RefreshCw className="size-3.5" />
            Retry
          </button>
        }
      />
    );
  } else if (!hasData) {
    body = <EngineSkeleton />;
  } else {
    body = (
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: EASE }}
        className="space-y-4"
      >
        {children}
      </motion.div>
    );
  }

  return (
    <PageShell
      title={title}
      icon={icon}
      caption={caption}
      wide={wide}
      actions={
        <button onClick={reload} disabled={loading} className={BTN_GHOST} title="Refresh">
          <RefreshCw className={`size-3.5 ${loading ? "animate-spin text-accent" : ""}`} />
          Refresh
        </button>
      }
    >
      {body}
    </PageShell>
  );
}

function EngineSkeleton() {
  return (
    <div className="space-y-4">
      <div className="skeleton h-32 rounded-2xl" />
      <div className="grid gap-4 md:grid-cols-2">
        <div className="skeleton h-56 rounded-2xl" />
        <div className="skeleton h-56 rounded-2xl" />
      </div>
      <div className="skeleton h-72 rounded-2xl" />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Status cards                                                        */
/* ------------------------------------------------------------------ */

export function StateCard({
  icon: Icon,
  iconTone,
  title,
  body,
  actions,
}: {
  icon: LucideIcon;
  iconTone: string;
  title: string;
  body: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: EASE }}
      className="card mx-auto max-w-xl px-6 py-12 text-center"
    >
      <div className="mx-auto flex size-12 items-center justify-center rounded-2xl border border-line bg-white/[0.04]">
        <Icon className={`size-5 ${iconTone}`} />
      </div>
      <h2 className="mt-4 text-lg font-semibold tracking-tight">{title}</h2>
      <div className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-mut">{body}</div>
      {actions && <div className="mt-6 flex flex-wrap justify-center gap-2.5">{actions}</div>}
    </motion.div>
  );
}

function EmptyPortfolioState({ onReload }: { onReload: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadSample() {
    setBusy(true);
    setError(null);
    try {
      await loadSamplePortfolio();
      onReload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the sample portfolio.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <StateCard
      icon={PackageOpen}
      iconTone="text-accent"
      title="Your AURORA portfolio is empty"
      body={
        <>
          The engine pages read the saved portfolio on the backend. Build yours on
          the Portfolio page — or load the sample portfolio to explore right away.
          {error && <span className="mt-2 block text-loss">{error}</span>}
        </>
      }
      actions={
        <>
          <Link href="/portfolio" className={BTN_PRIMARY}>
            <Briefcase className="size-3.5" />
            Open portfolio builder
          </Link>
          <button onClick={loadSample} disabled={busy} className={BTN_GHOST}>
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : <PackageOpen className="size-3.5" />}
            Load sample portfolio
          </button>
        </>
      }
    />
  );
}

/* ------------------------------------------------------------------ */
/* Small shared primitives                                             */
/* ------------------------------------------------------------------ */

export function Section({
  title,
  action,
  className = "",
  children,
}: {
  title: ReactNode;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`card p-5 ${className}`}>
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="font-mono text-[10px] uppercase tracking-[0.22em] text-mut">{title}</h3>
        {action}
      </div>
      {children}
    </section>
  );
}

export function Metric({
  label,
  value,
  sub,
  tone = "text-ink",
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: string;
}) {
  return (
    <div className="card px-4 py-3.5">
      <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-mut">{label}</div>
      <div className={`mt-1.5 font-mono text-[22px] font-semibold leading-none tabular ${tone}`}>
        {value}
      </div>
      {sub && <div className="mt-1.5 font-mono text-[10px] tabular text-mut">{sub}</div>}
    </div>
  );
}

export function ThinBar({
  fraction,
  color = "var(--color-accent)",
  className = "w-16",
}: {
  fraction: number; // 0..1
  color?: string;
  className?: string;
}) {
  return (
    <div className={`h-[3px] overflow-hidden rounded-full bg-white/[0.07] ${className}`}>
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{ width: `${clamp(fraction, 0, 1) * 100}%`, backgroundColor: color, opacity: 0.85 }}
      />
    </div>
  );
}

const CHIP_TONES = {
  accent: "border-accent/30 bg-accent-dim text-accent",
  gain: "border-gain/25 bg-gain/10 text-gain",
  loss: "border-loss/30 bg-loss/10 text-loss",
  amber: "border-amber-300/25 bg-amber-300/10 text-amber-200",
  mut: "border-line bg-white/[0.04] text-mut",
} as const;

export type ChipTone = keyof typeof CHIP_TONES;

export function Chip({ tone = "mut", children }: { tone?: ChipTone; children: ReactNode }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider ${CHIP_TONES[tone]}`}
    >
      {children}
    </span>
  );
}

/** Semicircular gauge for 0–100 scores (health score, reaction risk). */
export function ArcGauge({
  value,
  color,
  display,
  caption,
}: {
  value: number; // 0..100
  color: string;
  display: ReactNode;
  caption?: ReactNode;
}) {
  const frac = clamp(value / 100, 0, 1);
  return (
    <div className="relative mx-auto w-[190px] max-w-full">
      <svg viewBox="0 0 100 57" className="w-full overflow-visible">
        <path
          d="M8 52 A 42 42 0 0 1 92 52"
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={7}
          strokeLinecap="round"
        />
        <motion.path
          d="M8 52 A 42 42 0 0 1 92 52"
          fill="none"
          stroke={color}
          strokeWidth={7}
          strokeLinecap="round"
          pathLength={100}
          initial={{ strokeDasharray: "0 100" }}
          animate={{ strokeDasharray: `${frac * 100} 100` }}
          transition={{ duration: 1.1, ease: [0.65, 0, 0.15, 1] }}
          style={{ filter: `drop-shadow(0 0 10px ${color}55)` }}
        />
      </svg>
      <div className="absolute inset-x-0 bottom-0 text-center">
        <div className="font-mono text-[30px] font-semibold leading-none tabular" style={{ color }}>
          {display}
        </div>
        {caption && (
          <div className="mt-1 font-mono text-[10px] uppercase tracking-widest text-mut">
            {caption}
          </div>
        )}
      </div>
    </div>
  );
}

/** Muted warning strip used for expected soft conditions. */
export function Note({ tone = "amber", children }: { tone?: "amber" | "mut"; children: ReactNode }) {
  return (
    <div
      className={`flex items-start gap-2 rounded-xl border px-4 py-3 text-[12.5px] leading-relaxed ${
        tone === "amber"
          ? "border-amber-300/25 bg-amber-300/[0.07] text-amber-200/90"
          : "border-line bg-white/[0.03] text-mut"
      }`}
    >
      <TriangleAlert className="mt-0.5 size-3.5 shrink-0" />
      <div>{children}</div>
    </div>
  );
}
