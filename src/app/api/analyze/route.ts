/**
 * POST /api/analyze
 * Body: { holdings: {symbol, value}[], mode: "weight" | "shares" }
 *
 * Builds a daily-rebalanced constant-mix portfolio over ~5y of daily data,
 * compares it against SPY / QQQ, and returns full risk & performance
 * analytics plus allocation structure.
 */
import { NextRequest, NextResponse } from "next/server";
import { getManyPrices, normalizeSymbol } from "@/lib/prices";
import {
  alignSeries,
  buildPortfolioIndex,
  indexSeries,
  computeStats,
  betaAlpha,
  ytdReturn,
  monthlyReturns,
  roundN,
  roundSeries,
} from "@/lib/metrics";
import { getStockInfos } from "@/lib/stocks";
import type {
  AnalyzeResponse,
  AnalyzeResult,
  BenchmarkSeries,
  HoldingAnalysis,
  HoldingInput,
  InputMode,
} from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const MAX_HOLDINGS = 24;
const BENCHMARKS = [
  { symbol: "SPY", name: "S&P 500", color: "#8DA2FB" },
  { symbol: "QQQ", name: "NASDAQ 100", color: "#CF9FFF" },
] as const;

export async function POST(req: NextRequest): Promise<NextResponse<AnalyzeResult>> {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body" }, { status: 400 });
  }

  const mode: InputMode =
    (body as { mode?: string })?.mode === "shares" ? "shares" : "weight";
  const rawHoldings = (body as { holdings?: HoldingInput[] })?.holdings;
  if (!Array.isArray(rawHoldings)) {
    return NextResponse.json({ ok: false, error: "holdings must be an array" }, { status: 400 });
  }

  // sanitize + merge duplicates
  const merged = new Map<string, number>();
  for (const h of rawHoldings) {
    if (!h || typeof h.symbol !== "string" || typeof h.value !== "number") continue;
    if (!Number.isFinite(h.value) || h.value <= 0) continue;
    const s = normalizeSymbol(h.symbol);
    if (!s || s.length > 14) continue;
    merged.set(s, (merged.get(s) ?? 0) + h.value);
  }
  const symbols = [...merged.keys()].slice(0, MAX_HOLDINGS);
  if (symbols.length === 0) {
    return NextResponse.json({ ok: false, error: "No valid holdings provided" }, { status: 400 });
  }

  /* ---------- fetch prices (cached / live / simulated) ---------- */
  const priceMap = await getManyPrices([...symbols, ...BENCHMARKS.map((b) => b.symbol)]);

  const usable = symbols.filter((s) => (priceMap.get(s)?.bars.length ?? 0) >= 20);
  if (usable.length === 0) {
    return NextResponse.json(
      { ok: false, error: "No usable price history found for these symbols" },
      { status: 422 },
    );
  }

  /* ---------- resolve weights ---------- */
  const infos = await getStockInfos(usable);
  const weights: Record<string, number> = {};
  const lastPrices: Record<string, number | null> = {};
  for (const s of usable) {
    const bars = priceMap.get(s)!.bars;
    lastPrices[s] = bars.length > 0 ? bars[bars.length - 1].close : null;
  }
  let rawSum = 0;
  for (const s of usable) {
    const inp = merged.get(s)!;
    const raw = mode === "shares" ? inp * (lastPrices[s] ?? 0) : inp;
    weights[s] = raw;
    rawSum += raw;
  }
  if (!(rawSum > 0)) {
    return NextResponse.json(
      { ok: false, error: "Could not resolve position values" },
      { status: 422 },
    );
  }
  for (const s of usable) weights[s] = weights[s] / rawSum;

  /* ---------- align calendars ---------- */
  const alignInput: Record<string, { date: string; close: number }[]> = {};
  for (const s of usable) alignInput[s] = priceMap.get(s)!.bars;
  for (const b of BENCHMARKS) {
    const ps = priceMap.get(b.symbol);
    if (ps) alignInput[b.symbol] = ps.bars;
  }
  const aligned = alignSeries(alignInput);
  if (!aligned || aligned.dates.length < 30) {
    return NextResponse.json(
      { ok: false, error: "Insufficient overlapping history across holdings" },
      { status: 422 },
    );
  }
  const { dates, values } = aligned;
  const n = dates.length;

  /* ---------- portfolio + benchmark curves ---------- */
  const portfolio = buildPortfolioIndex(usable, weights, values, n);
  const portfolioMetricsBase = computeStats(portfolio, dates);

  const benchmarks: BenchmarkSeries[] = [];
  for (const b of BENCHMARKS) {
    if (!values[b.symbol]) continue;
    const idx = indexSeries(values[b.symbol]);
    benchmarks.push({
      symbol: b.symbol,
      name: b.name,
      color: b.color,
      values: roundSeries(idx),
      stats: computeStats(idx, dates),
    });
  }

  const spyIdx = benchmarks.find((b) => b.symbol === "SPY");
  const ba = spyIdx ? betaAlpha(portfolio, indexSeries(values["SPY"])) : null;

  /* ---------- per-holding analytics ---------- */
  const holdings: HoldingAnalysis[] = usable.map((s) => {
    const idx = indexSeries(values[s]);
    const stats = computeStats(values[s], dates);
    const info = infos.get(s);
    return {
      symbol: s,
      name: info?.name ?? s,
      sector: info?.sector ?? "Other",
      weight: roundN(weights[s], 4),
      inputValue: roundN(merged.get(s)!, 4),
      lastPrice: lastPrices[s],
      simulated: priceMap.get(s)?.simulated ?? false,
      stats,
      contribution: roundN(weights[s] * stats.totalReturn, 5),
      values: roundSeries(idx),
    };
  });
  holdings.sort((a, b) => b.weight - a.weight);

  /* ---------- allocation by sector ---------- */
  const sectorMap = new Map<string, number>();
  for (const h of holdings) {
    sectorMap.set(h.sector, (sectorMap.get(h.sector) ?? 0) + h.weight);
  }
  const sectors = [...sectorMap.entries()]
    .map(([sector, weight]) => ({ sector, weight: roundN(weight, 4) }))
    .sort((a, b) => b.weight - a.weight);

  /* ---------- source flags ---------- */
  const anySim = usable.some((s) => priceMap.get(s)?.simulated) ||
    BENCHMARKS.some((b) => priceMap.get(b.symbol)?.simulated);
  const allSim = usable.every((s) => priceMap.get(s)?.simulated);
  const source = allSim ? "simulated" : anySim ? "mixed" : "live";

  const response: AnalyzeResponse = {
    ok: true,
    mode,
    source,
    asOf: dates[n - 1],
    range: {
      start: dates[0],
      end: dates[n - 1],
      days: n,
      truncated: aligned.truncated,
      truncatedNote: aligned.truncated
        ? `History starts ${dates[0]} — ${
            aligned.truncatedBy && infos.get(aligned.truncatedBy)
              ? `${aligned.truncatedBy} (${infos.get(aligned.truncatedBy)!.name})`
              : aligned.truncatedBy ?? "one holding"
          } has the shortest track record in this portfolio.`
        : undefined,
    },
    dates,
    portfolio: roundSeries(portfolio),
    portfolioMetrics: {
      ...portfolioMetricsBase,
      beta: ba ? roundN(ba.beta, 3) : null,
      alpha: ba ? roundN(ba.alpha, 4) : null,
      ytd: roundN(ytdReturn(portfolio, dates), 4),
    },
    benchmarks,
    holdings,
    sectors,
    monthly: monthlyReturns(portfolio, dates).map((m) => ({ ...m, r: roundN(m.r, 4) })),
  };

  return NextResponse.json(response, {
    headers: { "Cache-Control": "private, max-age=120" },
  });
}
