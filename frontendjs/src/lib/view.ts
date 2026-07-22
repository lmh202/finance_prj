/**
 * Client-side view derivation: slice the full ~5y analysis down to the
 * selected range (1M/6M/1Y/… or a custom days/months/years window) and
 * recompute every metric on that window so charts, stats and tables always
 * speak about the same period. No refetch.
 */
import {
  computeStats,
  betaAlpha,
  ytdReturn,
  monthlyReturns,
  roundSeries,
} from "./metrics";
import type {
  AnalyzeResponse,
  MonthCell,
  PortfolioMetrics,
  RangeSpec,
  SeriesStats,
} from "./types";

export interface ViewHolding {
  symbol: string;
  name: string;
  sector: string;
  weight: number;
  lastPrice: number | null;
  simulated: boolean;
  stats: SeriesStats;
  contribution: number;
  values: number[];
}

export interface ViewBenchmark {
  symbol: string;
  name: string;
  color: string;
  values: number[];
  stats: SeriesStats;
  ytd: number;
}

export interface RangeView {
  dates: string[];
  portfolio: number[];
  metrics: PortfolioMetrics;
  benchmarks: ViewBenchmark[];
  holdings: ViewHolding[];
  monthly: MonthCell[];
}

/** Subtract a {@link RangeSpec} window from an ISO date, calendar-correct
 *  for months/years (variable month length) and calendar-day-exact for
 *  the custom "days" unit. */
function subtractSpec(iso: string, spec: RangeSpec): string {
  const y = Number(iso.slice(0, 4));
  const m = Number(iso.slice(5, 7)) - 1;
  const d = Number(iso.slice(8, 10));
  if (spec.unit === "days") {
    return new Date(Date.UTC(y, m, d - spec.amount)).toISOString().slice(0, 10);
  }
  const months = spec.unit === "years" ? spec.amount * 12 : spec.amount;
  return new Date(Date.UTC(y, m - months, d)).toISOString().slice(0, 10);
}

function slice(values: number[], from: number): number[] {
  return values.slice(from);
}

function rebase(values: number[]): number[] {
  const b = values[0];
  if (!b || b <= 0) return values.map(() => 100);
  return roundSeries(values.map((v) => (v / b) * 100));
}

export function deriveRangeView(a: AnalyzeResponse, spec: RangeSpec): RangeView {
  const n = a.dates.length;
  let from = 0;
  if (n > 2) {
    const cutoff = subtractSpec(a.dates[n - 1], spec);
    let i = 0;
    while (i < n - 26 && a.dates[i] < cutoff) i++;
    from = Math.max(0, i);
  }

  const dates = a.dates.slice(from);
  const portfolio = rebase(slice(a.portfolio, from));

  const benchmarks: ViewBenchmark[] = a.benchmarks.map((b) => {
    const vals = rebase(slice(b.values, from));
    return {
      symbol: b.symbol,
      name: b.name,
      color: b.color,
      values: vals,
      stats: computeStats(vals, dates),
      ytd: ytdReturn(vals, dates),
    };
  });

  const base = computeStats(portfolio, dates);
  const spyB = benchmarks.find((b) => b.symbol === "SPY");
  const spyRaw = a.benchmarks.find((b) => b.symbol === "SPY");
  const ba =
    spyRaw && spyB ? betaAlpha(slice(a.portfolio, from), slice(spyRaw.values, from)) : null;

  const metrics: PortfolioMetrics = {
    ...base,
    beta: ba ? ba.beta : null,
    alpha: ba ? ba.alpha : null,
    ytd: ytdReturn(portfolio, dates),
  };

  const holdings: ViewHolding[] = a.holdings.map((h) => {
    const vals = rebase(slice(h.values, from));
    const stats = computeStats(vals, dates);
    return {
      symbol: h.symbol,
      name: h.name,
      sector: h.sector,
      weight: h.weight,
      lastPrice: h.lastPrice,
      simulated: h.simulated,
      stats,
      contribution: h.weight * stats.totalReturn,
      values: vals,
    };
  });
  holdings.sort((x, y) => y.weight - x.weight);

  return {
    dates,
    portfolio,
    metrics,
    benchmarks,
    holdings,
    monthly: monthlyReturns(portfolio, dates),
  };
}
