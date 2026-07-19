/**
 * Quant toolkit: calendar alignment, portfolio construction (daily-rebalanced
 * constant-mix), and risk/performance analytics.
 */
import type { SeriesStats, MonthCell } from "./types";
import type { Bar } from "./prices";

const TRADING_DAYS = 252;
export const RISK_FREE = 0.04; // annual, used for Sharpe/Sortino/alpha

const ZERO_STATS: SeriesStats = {
  totalReturn: 0,
  cagr: 0,
  annVol: 0,
  sharpe: 0,
  sortino: 0,
  maxDrawdown: 0,
  calmar: 0,
  bestDay: 0,
  worstDay: 0,
  winRate: 0,
};

/* ------------------------------------------------------------------ */
/* alignment                                                           */
/* ------------------------------------------------------------------ */

export interface Aligned {
  dates: string[];
  values: Record<string, number[]>;
  truncated: boolean;
  truncatedBy?: string;
}

/**
 * Align several daily series on a common union calendar. Each series is
 * forward-filled after its first observation; the window spans from the
 * latest first-observation to the earliest last-observation so every series
 * is defined across the whole window.
 */
export function alignSeries(input: Record<string, Bar[]>): Aligned | null {
  const symbols = Object.keys(input).filter((s) => input[s].length > 0);
  if (symbols.length === 0) return null;

  const maps: Record<string, Map<string, number>> = {};
  let start = "";
  let end = "9999-12-31";
  let truncatedBy: string | undefined;

  for (const s of symbols) {
    const bars = input[s];
    maps[s] = new Map(bars.map((b) => [b.date, b.close]));
    const first = bars[0].date;
    const last = bars[bars.length - 1].date;
    if (first > start) {
      start = first;
      truncatedBy = s;
    }
    if (last < end) end = last;
  }

  const dateSet = new Set<string>();
  for (const s of symbols) for (const d of maps[s].keys()) dateSet.add(d);
  const dates = [...dateSet].filter((d) => d >= start && d <= end).sort();
  if (dates.length < 2) return null;

  const values: Record<string, number[]> = {};
  for (const s of symbols) {
    const m = maps[s];
    const out: number[] = new Array(dates.length);
    let lastSeen: number | undefined = m.get(dates[0]);
    if (lastSeen == null) {
      // The instrument may not trade exactly on the window's first day —
      // backfill with its first available observation inside the window.
      for (let i = 1; i < dates.length; i++) {
        const v = m.get(dates[i]);
        if (v != null) {
          lastSeen = v;
          break;
        }
      }
    }
    for (let i = 0; i < dates.length; i++) {
      const v = m.get(dates[i]);
      if (v != null) lastSeen = v;
      out[i] = lastSeen ?? NaN;
    }
    values[s] = out;
  }

  const firstDates = symbols.map((s) => input[s][0].date);
  const globalFirst = firstDates.reduce((a, b) => (a < b ? a : b));
  const truncated = start > globalFirst;

  return { dates, values, truncated, truncatedBy };
}

/* ------------------------------------------------------------------ */
/* portfolio construction                                              */
/* ------------------------------------------------------------------ */

/** Indexed (base 100) constant-mix portfolio from aligned price values. */
export function buildPortfolioIndex(
  symbols: string[],
  weights: Record<string, number>,
  values: Record<string, number[]>,
  n: number,
): number[] {
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (const s of symbols) {
      const base = values[s][0];
      sum += weights[s] * (values[s][i] / base);
    }
    out[i] = sum * 100;
  }
  return out;
}

/** Indexed (base 100) single series. */
export function indexSeries(values: number[]): number[] {
  const base = values[0];
  return values.map((v) => (v / base) * 100);
}

/* ------------------------------------------------------------------ */
/* statistics                                                          */
/* ------------------------------------------------------------------ */

export function dailyReturns(values: number[]): number[] {
  const r: number[] = [];
  for (let i = 1; i < values.length; i++) {
    const prev = values[i - 1];
    if (prev > 0) r.push(values[i] / prev - 1);
  }
  return r;
}

function mean(a: number[]): number {
  if (a.length === 0) return 0;
  let s = 0;
  for (const x of a) s += x;
  return s / a.length;
}

function std(a: number[], m: number): number {
  if (a.length < 2) return 0;
  let s = 0;
  for (const x of a) s += (x - m) * (x - m);
  return Math.sqrt(s / (a.length - 1));
}

export function computeStats(values: number[], dates: string[]): SeriesStats {
  if (values.length < 2 || dates.length < 2) return ZERO_STATS;

  const rets = dailyReturns(values);
  const m = mean(rets);
  const sd = std(rets, m);

  const first = values[0];
  const last = values[values.length - 1];
  const totalReturn = first > 0 ? last / first - 1 : 0;

  const spanMs =
    new Date(dates[dates.length - 1]).getTime() - new Date(dates[0]).getTime();
  const years = Math.max(spanMs / (365.25 * 86400000), 1 / TRADING_DAYS);
  const cagr = first > 0 && last > 0 ? Math.pow(last / first, 1 / years) - 1 : 0;

  const annVol = sd * Math.sqrt(TRADING_DAYS);
  const annRetArith = m * TRADING_DAYS;

  const sharpe = annVol > 1e-9 ? (annRetArith - RISK_FREE) / annVol : 0;

  const rfDaily = RISK_FREE / TRADING_DAYS;
  let downsideSq = 0;
  for (const r of rets) {
    const d = Math.min(r - rfDaily, 0);
    downsideSq += d * d;
  }
  const downsideDev =
    Math.sqrt(downsideSq / Math.max(rets.length, 1)) * Math.sqrt(TRADING_DAYS);
  const sortino = downsideDev > 1e-9 ? (annRetArith - RISK_FREE) / downsideDev : 0;

  let peak = values[0];
  let maxDrawdown = 0;
  for (const v of values) {
    if (v > peak) peak = v;
    const dd = v / peak - 1;
    if (dd < maxDrawdown) maxDrawdown = dd;
  }

  const calmar = maxDrawdown < -1e-9 ? cagr / Math.abs(maxDrawdown) : 0;

  let best = -Infinity;
  let worst = Infinity;
  let wins = 0;
  for (const r of rets) {
    if (r > best) best = r;
    if (r < worst) worst = r;
    if (r > 0) wins++;
  }

  return {
    totalReturn,
    cagr,
    annVol,
    sharpe,
    sortino,
    maxDrawdown,
    calmar,
    bestDay: rets.length ? best : 0,
    worstDay: rets.length ? worst : 0,
    winRate: rets.length ? wins / rets.length : 0,
  };
}

/** Beta & Jensen's alpha (annualized) of a vs b using daily returns. */
export function betaAlpha(
  aValues: number[],
  bValues: number[],
): { beta: number; alpha: number } | null {
  const n = Math.min(aValues.length, bValues.length);
  if (n < 30) return null;
  const ra: number[] = [];
  const rb: number[] = [];
  for (let i = 1; i < n; i++) {
    if (aValues[i - 1] > 0 && bValues[i - 1] > 0) {
      ra.push(aValues[i] / aValues[i - 1] - 1);
      rb.push(bValues[i] / bValues[i - 1] - 1);
    }
  }
  if (ra.length < 30) return null;
  const ma = mean(ra);
  const mb = mean(rb);
  let cov = 0;
  let varB = 0;
  for (let i = 0; i < ra.length; i++) {
    cov += (ra[i] - ma) * (rb[i] - mb);
    varB += (rb[i] - mb) * (rb[i] - mb);
  }
  if (varB <= 1e-12) return null;
  const beta = cov / varB;
  const alpha =
    (ma * TRADING_DAYS - RISK_FREE) - beta * (mb * TRADING_DAYS - RISK_FREE);
  return { beta, alpha };
}

export function ytdReturn(values: number[], dates: string[]): number {
  if (values.length < 2) return 0;
  const lastDate = dates[dates.length - 1];
  const year = lastDate.slice(0, 4);
  const jan1 = `${year}-01-01`;
  let baseIdx = 0;
  for (let i = 0; i < dates.length; i++) {
    if (dates[i] < jan1) baseIdx = i;
    else break;
  }
  const base = values[baseIdx];
  return base > 0 ? values[values.length - 1] / base - 1 : 0;
}

export function monthlyReturns(values: number[], dates: string[]): MonthCell[] {
  if (values.length < 2) return [];
  const lastIdxByMonth = new Map<string, number>();
  for (let i = 0; i < dates.length; i++) {
    lastIdxByMonth.set(dates[i].slice(0, 7), i);
  }
  const months = [...lastIdxByMonth.keys()].sort();
  const cells: MonthCell[] = [];
  let prevEndVal = values[0];
  for (let k = 0; k < months.length; k++) {
    const idx = lastIdxByMonth.get(months[k])!;
    const endVal = values[idx];
    const r = prevEndVal > 0 ? endVal / prevEndVal - 1 : 0;
    cells.push({
      year: Number(months[k].slice(0, 4)),
      month: Number(months[k].slice(5, 7)),
      r: k === 0 ? (values[0] > 0 ? endVal / values[0] - 1 : 0) : r,
    });
    prevEndVal = endVal;
  }
  return cells;
}

/* ------------------------------------------------------------------ */
/* rounding for wire transfer                                          */
/* ------------------------------------------------------------------ */

export function roundN(x: number, n = 4): number {
  const p = Math.pow(10, n);
  return Math.round(x * p) / p;
}

export function roundSeries(values: number[], n = 4): number[] {
  return values.map((v) => roundN(v, n));
}
