/**
 * Tradable universe search: seeded PostgreSQL catalog ranked by match quality
 * and popularity, extended live from Yahoo Finance search on cache-miss so
 * effectively every listed ticker is discoverable.
 */
import { ilike, or, inArray } from "drizzle-orm";
import { db } from "@/db";
import { stocks } from "@/db/schema";
import type { StockInfo } from "./types";
import { normalizeSymbol, httpGetText, upstream } from "./prices";

const US_EXCHANGES = new Set([
  "NMS",
  "NYQ",
  "NGM",
  "NYS",
  "ASE",
  "PCX",
  "BTS",
  "CBO",
]);

interface StockRow {
  symbol: string;
  name: string;
  exchange: string;
  sector: string;
  quoteType: string;
  popularity: number;
}

function toInfo(r: StockRow, live = false): StockInfo {
  return {
    symbol: r.symbol,
    name: r.name,
    exchange: r.exchange,
    sector: r.sector,
    quoteType: r.quoteType,
    live,
  };
}

function rankRows(rows: StockRow[], q: string): StockRow[] {
  const qq = q.toUpperCase();
  const scored = rows.map((r) => {
    const sym = r.symbol.toUpperCase();
    const name = r.name.toUpperCase();
    let score = r.popularity / 20;
    if (sym === qq) score += 1000;
    else if (sym.startsWith(qq)) score += 500 - sym.length;
    else if (name.startsWith(qq)) score += 250;
    else if (name.includes(qq)) score += 120;
    else if (sym.includes(qq)) score += 60;
    return { r, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.map((s) => s.r);
}

export async function searchStocks(qRaw: string): Promise<StockInfo[]> {
  const q = qRaw.replace(/[%_\\"']/g, "").trim();
  if (q.length < 1) return [];

  let rows: StockRow[] = [];
  try {
    rows = await db
      .select({
        symbol: stocks.symbol,
        name: stocks.name,
        exchange: stocks.exchange,
        sector: stocks.sector,
        quoteType: stocks.quoteType,
        popularity: stocks.popularity,
      })
      .from(stocks)
      .where(or(ilike(stocks.symbol, `%${q}%`), ilike(stocks.name, `%${q}%`)))
      .limit(40);
  } catch {
    rows = [];
  }

  const ranked = rankRows(rows, q).slice(0, 9);

  // Augment live when the local catalog is thin (or for symbol-like queries).
  if (q.length >= 2 && ranked.length < 6) {
    const live = await liveSearch(q);
    if (live.length > 0) {
      const known = new Set(ranked.map((r) => r.symbol));
      const fresh = live.filter((l) => !known.has(l.symbol));
      void persistLive(fresh);
      return [...ranked.map((r) => toInfo(r)), ...fresh.map((f) => toInfo(f, true))].slice(0, 10);
    }
  }

  return ranked.map((r) => toInfo(r));
}

interface YahooQuote {
  symbol?: string;
  longname?: string;
  shortname?: string;
  exchange?: string;
  quoteType?: string;
  sector?: string;
}

interface NasdaqHit {
  symbol?: string;
  name?: string;
  exchange?: string;
  asset?: string;
  industry?: string | null;
  nasdaq100?: string | null;
}

const INDUSTRY_SECTORS: ReadonlyArray<readonly [RegExp, string]> = [
  [/semiconductor|software|technology|computer|internet|cloud|cybersec|telecom equipment/i, "Technology"],
  [/bank|finance|financial|investment|insurance|capital|broker|credit|mortgage/i, "Financials"],
  [/biotech|pharma|health|medical|diagnostic|life science|drug/i, "Health Care"],
  [/oil|gas|energy|pipeline|drilling|coal|refining/i, "Energy"],
  [/food|beverage|tobacco|household|personal products|staples/i, "Consumer Staples"],
  [/retail|restaurant|auto|leisure|apparel|consumer|hotel|casino|travel|commerce|footwear|home improvement/i, "Consumer Discretionary"],
  [/aerospace|industrial|transport|construct|machinery|logistic|railroad|defense|freight|engineering/i, "Industrials"],
  [/telecom|media|advertis|entertainment|broadcast|gaming|publish/i, "Communication Services"],
  [/utilit|electric|water/i, "Utilities"],
  [/real estate|reit/i, "Real Estate"],
  [/chemical|mining|metal|paper|material|steel|gold|silver|lumber|aluminum/i, "Materials"],
];

function mapIndustry(industry: string | null | undefined, asset: string | undefined): string {
  if (asset === "ETFS") return "ETF";
  if (!industry) return "Other";
  for (const [re, sector] of INDUSTRY_SECTORS) {
    if (re.test(industry)) return sector;
  }
  return "Other";
}

function cleanNasdaqName(name: string): string {
  return name
    .replace(
      /\s+(common stock|common shares|ordinary shares|american depositary shares|ads|adr|preferred stock|units?|warrants?|shares|class [a-z] common stock)\.?$/i,
      "",
    )
    .trim()
    .slice(0, 90);
}

/** Live instrument search via the official Nasdaq autocomplete API. */
async function nasdaqLookup(q: string): Promise<StockRow[]> {
  try {
    const text = await upstream(() =>
      httpGetText(
        `https://api.nasdaq.com/api/autocomplete/slookup/10?search=${encodeURIComponent(q)}`,
        6000,
        { Referer: "https://www.nasdaq.com/", Origin: "https://www.nasdaq.com" },
      ),
    );
    const json = JSON.parse(text) as { data?: NasdaqHit[] | NasdaqHit | null };
    const raw = json.data;
    const hits: NasdaqHit[] = Array.isArray(raw) ? raw : raw ? [raw] : [];
    const seen = new Set<string>();
    const out: StockRow[] = [];
    for (const h of hits) {
      if (!h.symbol) continue;
      if (!/^[A-Z][A-Z0-9/.\-]{0,11}$/.test(h.symbol)) continue;
      if (h.asset !== "STOCKS" && h.asset !== "ETFS") continue;
      const symbol = normalizeSymbol(h.symbol);
      if (seen.has(symbol)) continue;
      seen.add(symbol);
      out.push({
        symbol,
        name: cleanNasdaqName(h.name ?? h.symbol) || h.symbol,
        exchange: (h.exchange ?? "").includes("NASDAQ") ? "NASDAQ" : (h.exchange ?? "").split(" ")[0],
        sector: mapIndustry(h.industry, h.asset),
        quoteType: h.asset === "ETFS" ? "ETF" : "EQUITY",
        popularity: h.nasdaq100 === "Y" ? 80 : 4,
      });
      if (out.length >= 8) break;
    }
    return out;
  } catch {
    return [];
  }
}

/** Live search quorum: Nasdaq autocomplete → Yahoo search. */
async function liveSearch(q: string): Promise<StockRow[]> {
  const ndq = await nasdaqLookup(q);
  if (ndq.length > 0) return ndq;
  return yahooSearch(q);
}

async function yahooSearch(q: string): Promise<StockRow[]> {
  try {
    const text = await httpGetText(
      `https://query2.finance.yahoo.com/v1/finance/search?q=${encodeURIComponent(
        q,
      )}&quotesCount=8&newsCount=0&enableFuzzyQuery=true`,
      4000,
    );
    const json = JSON.parse(text) as { quotes?: YahooQuote[] };
    const quotes = json.quotes ?? [];
    return quotes
      .filter((x) => {
        if (!x.symbol) return false;
        if (x.quoteType !== "EQUITY" && x.quoteType !== "ETF") return false;
        if (!/^[A-Z][A-Z0-9.\-]{0,11}$/.test(x.symbol)) return false;
        if (x.exchange && !US_EXCHANGES.has(x.exchange)) return false;
        return true;
      })
      .slice(0, 6)
      .map((x) => ({
        symbol: normalizeSymbol(x.symbol!),
        name: (x.longname || x.shortname || x.symbol!).slice(0, 90),
        exchange: x.exchange === "NYQ" || x.exchange === "NYS" ? "NYSE" : x.exchange === "NMS" || x.exchange === "NGM" ? "NASDAQ" : x.exchange ?? "",
        sector: x.quoteType === "ETF" ? "ETF" : x.sector ?? "Other",
        quoteType: x.quoteType!,
        popularity: 1,
      }));
  } catch {
    return [];
  }
}

async function persistLive(rows: StockRow[]): Promise<void> {
  if (rows.length === 0) return;
  try {
    await db
      .insert(stocks)
      .values(rows)
      .onConflictDoNothing();
  } catch {
    /* best-effort */
  }
}

/** Ensure catalog rows exist for symbols (used to label holdings). */
export async function getStockInfos(symbols: string[]): Promise<Map<string, StockInfo>> {
  const norm = [...new Set(symbols.map(normalizeSymbol))];
  const out = new Map<string, StockInfo>();
  if (norm.length === 0) return out;

  let rows: StockRow[] = [];
  try {
    rows = await db
      .select({
        symbol: stocks.symbol,
        name: stocks.name,
        exchange: stocks.exchange,
        sector: stocks.sector,
        quoteType: stocks.quoteType,
        popularity: stocks.popularity,
      })
      .from(stocks)
      .where(inArray(stocks.symbol, norm));
  } catch {
    rows = [];
  }

  const found = new Set(rows.map((r) => r.symbol));
  for (const r of rows) out.set(r.symbol, toInfo(r));

  const missing = norm.filter((s) => !found.has(s));
  if (missing.length > 0) {
    const synthesized: StockRow[] = [];
    for (const s of missing) {
      const live = await liveSearch(s);
      const exact = live.find((l) => l.symbol === s);
      if (exact) {
        out.set(s, toInfo(exact, true));
        synthesized.push(exact);
      } else {
        out.set(s, {
          symbol: s,
          name: s,
          exchange: "",
          sector: "Other",
          quoteType: "EQUITY",
        });
      }
    }
    void persistLive(synthesized);
  }

  return out;
}
