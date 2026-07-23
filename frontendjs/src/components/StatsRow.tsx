"use client";

import { useEffect } from "react";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import type { PortfolioMetrics, SeriesStats } from "@/lib/types";
import { fmtPct, fmtNum, signClass } from "@/lib/format";

function AnimatedValue({
  value,
  format,
  className,
}: {
  value: number;
  format: (v: number) => string;
  className?: string;
}) {
  const mv = useMotionValue(0);
  const spring = useSpring(mv, { stiffness: 80, damping: 22, mass: 0.9 });
  const text = useTransform(spring, (v) => format(v));
  useEffect(() => {
    mv.set(value);
  }, [value, mv]);
  return (
    <motion.span className={className} style={{ display: "inline-block" }}>
      {text}
    </motion.span>
  );
}

interface CardDef {
  label: string;
  hint: string;
  render: (m: PortfolioMetrics) => number;
  format: (v: number) => string;
  tone: (m: PortfolioMetrics) => string;
  sub: (m: PortfolioMetrics, spy: SeriesStats | null) => string | null;
}

const CARDS: CardDef[] = [
  {
    label: "Total Return",
    hint: "Cumulative gain of the blended portfolio over the selected window, dividends adjusted.",
    render: (m) => m.totalReturn,
    format: (v) => fmtPct(v),
    tone: (m) => signClass(m.totalReturn),
    sub: (_, spy) => (spy ? `SPY ${fmtPct(spy.totalReturn)}` : null),
  },
  {
    label: "CAGR",
    hint: "Compound annual growth rate — the smooth yearly pace of this portfolio.",
    render: (m) => m.cagr,
    format: (v) => fmtPct(v),
    tone: (m) => signClass(m.cagr),
    sub: (_, spy) => (spy ? `SPY ${fmtPct(spy.cagr)}` : null),
  },
  {
    label: "Volatility",
    hint: "Annualized standard deviation of daily returns. Lower means a steadier ride.",
    render: (m) => m.annVol,
    format: (v) => fmtPct(v, 1, false),
    tone: () => "text-ink",
    sub: (_, spy) => (spy ? `SPY ${fmtPct(spy.annVol, 1, false)}` : null),
  },
  {
    label: "Sharpe",
    hint: "Return per unit of risk, from the Portfolio Health report (trailing ~2y, no risk-free adjustment) — matches the Sharpe on the Health page. Above 1 is strong, above 2 is elite.",
    render: (m) => m.sharpe,
    format: (v) => fmtNum(v, 2),
    tone: (m) => (m.sharpe >= 1 ? "text-gain" : m.sharpe < 0 ? "text-loss" : "text-ink"),
    sub: () => "from Health report",
  },
  {
    label: "Sortino",
    hint: "Like Sharpe, but only penalizes downside volatility — the risk that actually hurts.",
    render: (m) => m.sortino,
    format: (v) => fmtNum(v, 2),
    tone: (m) => (m.sortino >= 1 ? "text-gain" : m.sortino < 0 ? "text-loss" : "text-ink"),
    sub: (_, spy) => (spy ? `SPY ${fmtNum(spy.sortino, 2)}` : null),
  },
  {
    label: "Max Drawdown",
    hint: "Deepest peak-to-trough loss over the window. Pain you would have had to stomach.",
    render: (m) => m.maxDrawdown,
    format: (v) => fmtPct(v),
    tone: (m) => (m.maxDrawdown < -0.25 ? "text-loss" : "text-ink"),
    sub: (_, spy) => (spy ? `SPY ${fmtPct(spy.maxDrawdown)}` : null),
  },
  {
    label: "Beta",
    hint: "Sensitivity to the S&P 500. 1.0 = moves with the market; above 1 amplifies it.",
    render: (m) => m.beta ?? 0,
    format: (v) => fmtNum(v, 2),
    tone: () => "text-ink",
    sub: () => "vs SPY",
  },
  {
    label: "Alpha",
    hint: "Annualized excess return beyond what beta exposure alone would explain.",
    render: (m) => m.alpha ?? 0,
    format: (v) => fmtPct(v),
    tone: (m) => signClass(m.alpha),
    sub: () => "annualized, vs SPY",
  },
  {
    label: "YTD",
    hint: "Return since the first trading day of the current calendar year.",
    render: (m) => m.ytd,
    format: (v) => fmtPct(v),
    tone: (m) => signClass(m.ytd),
    sub: () => "year to date",
  },
  {
    label: "Win Rate",
    hint: "Share of trading days that closed higher than the previous day.",
    render: (m) => m.winRate,
    format: (v) => fmtPct(v, 1, false),
    tone: () => "text-ink",
    sub: () => "of up days",
  },
];

export function StatsRow({
  metrics,
  spy,
}: {
  metrics: PortfolioMetrics;
  spy: SeriesStats | null;
}) {
  return (
    <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 xl:grid-cols-5">
      {CARDS.map((c, i) => {
        const sub = c.sub(metrics, spy);
        return (
          <motion.div
            key={c.label}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.05 + i * 0.04, ease: [0.16, 1, 0.3, 1] }}
            className="card group relative px-4 py-3.5"
          >
            <div className="flex items-center gap-1.5">
              <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-mut">
                {c.label}
              </span>
              <span className="size-1 cursor-help rounded-full bg-mut/40 transition-colors group-hover:bg-accent" />
            </div>
            <div className="mt-1.5">
              <AnimatedValue
                value={c.render(metrics)}
                format={c.format}
                className={`font-mono text-[22px] font-semibold leading-none tabular ${c.tone(metrics)}`}
              />
            </div>
            {sub && <div className="mt-1.5 font-mono text-[10px] tabular text-mut">{sub}</div>}

            <div className="pointer-events-none absolute bottom-[calc(100%+8px)] left-1/2 z-30 w-52 -translate-x-1/2 rounded-xl border border-line bg-surface/97 p-3 text-[11px] leading-relaxed text-ink/80 opacity-0 shadow-2xl backdrop-blur-md transition-all duration-200 group-hover:opacity-100">
              {c.hint}
              <div className="absolute -bottom-1 left-1/2 size-2 -translate-x-1/2 rotate-45 border-b border-r border-line bg-surface" />
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
