/**
 * Market-data engine.
 *
 * Resolution order for 5y of daily adjusted closes:
 *   1. in-process cache (per server instance)
 *   2. PostgreSQL `price_cache` table (shared, persistent)
 *   3. live fetch from Yahoo Finance chart API (write-through cached)
 *   4. deterministic synthetic random walk (offline-safe fallback,
 *      always flagged `simulated: true` so the UI stays honest)
 */
import { execFile } from "node:child_process";
import { and, asc, eq, gte } from "drizzle-orm";
import { db } from "@/db";
import { priceCache } from "@/db/schema";

export interface Bar {
  date: string; // YYYY-MM-DD
  close: number; // adjusted close
}

export interface PriceSeries {
  symbol: string;
  bars: Bar[];
  simulated: boolean;
}

const YEARS = 5;
const MIN_BARS = 40;
export const UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36";

/**
 * Upstream APIs fingerprint-block plain Node fetch (429) but permit curl.
 * Prefer curl with browser-like headers; fall back to fetch where curl is
 * unavailable.
 */
export function httpGetText(
  url: string,
  timeoutMs: number,
  headers: Record<string, string> = {},
): Promise<string> {
  const headerArgs: string[] = [];
  for (const [k, v] of Object.entries({ "User-Agent": UA, Accept: "application/json", ...headers })) {
    headerArgs.push("-H", `${k}: ${v}`);
  }
  return new Promise((resolve, reject) => {
    execFile(
      "curl",
      [
        "-sS",
        "--compressed",
        "--max-time",
        String(Math.ceil(timeoutMs / 1000)),
        ...headerArgs,
        "-w",
        "\n%{http_code}",
        url,
      ],
      { maxBuffer: 24 * 1024 * 1024 },
      (err, stdout) => {
        if (err) {
          fetch(url, {
            headers: { "User-Agent": UA, Accept: "application/json", ...headers },
            cache: "no-store",
          })
            .then(async (r) => {
              if (!r.ok) throw new Error(`HTTP ${r.status}`);
              resolve(await r.text());
            })
            .catch(reject);
          return;
        }
        const body = stdout.trimEnd();
        const nl = body.lastIndexOf("\n");
        const status = nl >= 0 ? Number(body.slice(nl + 1)) : 0;
        const payload = nl >= 0 ? body.slice(0, nl) : body;
        if (status !== 200) {
          reject(new Error(`HTTP ${status || "?"}`));
          return;
        }
        resolve(payload);
      },
    );
  });
}

/* ---------- polite upstream scheduler ---------- */

const schedGlobal = globalThis as unknown as {
  __prismSched?: { inFlight: number; queue: Array<() => void>; lastStart: number };
};
const sched =
  schedGlobal.__prismSched ??
  (schedGlobal.__prismSched = { inFlight: 0, queue: [], lastStart: 0 });

function acquireSlot(): Promise<void> {
  if (sched.inFlight < 3) {
    sched.inFlight++;
    return Promise.resolve();
  }
  return new Promise((res) => sched.queue.push(res));
}
function releaseSlot(): void {
  sched.inFlight--;
  const next = sched.queue.shift();
  if (next) {
    sched.inFlight++;
    next();
  }
}

/** Concurrency-limited, gap-spaced upstream call to stay under rate limits. */
export async function upstream<T>(fn: () => Promise<T>): Promise<T> {
  await acquireSlot();
  const gap = 140;
  const wait = Math.max(0, gap - (Date.now() - sched.lastStart));
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  sched.lastStart = Date.now();
  try {
    return await fn();
  } finally {
    releaseSlot();
  }
}

const g = globalThis as unknown as {
  __prismPrices?: Map<string, Promise<PriceSeries>>;
};
const mem = g.__prismPrices ?? (g.__prismPrices = new Map());

export function normalizeSymbol(raw: string): string {
  return raw.trim().toUpperCase().replace(/\./g, "-").slice(0, 14);
}

function cutoffDate(): string {
  return new Date(Date.now() - YEARS * 365.25 * 86400000)
    .toISOString()
    .slice(0, 10);
}

function isFresh(bars: Bar[]): boolean {
  if (bars.length < 500) return false;
  const last = bars[bars.length - 1].date;
  const staleMs = Date.now() - new Date(last + "T16:00:00Z").getTime();
  return staleMs < 10 * 86400000; // last bar within ~10 days
}

async function readCache(symbol: string): Promise<Bar[] | null> {
  try {
    const rows = await db
      .select({ d: priceCache.d, close: priceCache.close })
      .from(priceCache)
      .where(and(gte(priceCache.d, cutoffDate()), eq(priceCache.symbol, symbol)))
      .orderBy(asc(priceCache.d));
    if (rows.length === 0) return null;
    return rows.map((r) => ({ date: r.d, close: r.close }));
  } catch {
    return null;
  }
}

async function writeCache(symbol: string, bars: Bar[]): Promise<void> {
  try {
    for (let i = 0; i < bars.length; i += 400) {
      const chunk = bars
        .slice(i, i + 400)
        .map((b) => ({ symbol, d: b.date, close: b.close }));
      if (chunk.length > 0) {
        await db.insert(priceCache).values(chunk).onConflictDoNothing();
      }
    }
  } catch {
    /* cache is best-effort */
  }
}

interface NasdaqRow {
  date?: string;
  close?: string;
}

/**
 * Provider 1 — official Nasdaq API (split-adjusted closes).
 * Class shares translate to Nasdaq notation (BRK-B → BRK/B).
 */
async function fetchNasdaq(symbolRaw: string): Promise<Bar[] | null> {
  const symbol = symbolRaw.replace(/-/g, "/");
  const from = cutoffDate();
  const to = new Date().toISOString().slice(0, 10);
  for (const assetclass of ["stocks", "etf"] as const) {
    try {
      const url = `https://api.nasdaq.com/api/quote/${encodeURIComponent(
        symbol,
      )}/historical?assetclass=${assetclass}&fromdate=${from}&todate=${to}&limit=1600`;
      const text = await upstream(() =>
        httpGetText(url, 11000, { Referer: "https://www.nasdaq.com/", Origin: "https://www.nasdaq.com" }),
      );
      const json = JSON.parse(text) as {
        data?: { tradesTable?: { rows?: NasdaqRow[] } };
      };
      const rows = json.data?.tradesTable?.rows;
      if (!rows || rows.length < MIN_BARS) continue;
      const bars: Bar[] = [];
      for (let i = rows.length - 1; i >= 0; i--) {
        const r = rows[i];
        if (!r.date || !r.close) continue;
        const px = Number(String(r.close).replace(/[$,\s]/g, ""));
        if (!Number.isFinite(px) || px <= 0) continue;
        const parts = String(r.date).split("/");
        if (parts.length !== 3) continue;
        const date = `${parts[2]}-${parts[0].padStart(2, "0")}-${parts[1].padStart(2, "0")}`;
        if (bars.length > 0 && bars[bars.length - 1].date >= date) continue;
        bars.push({ date, close: Math.round(px * 10000) / 10000 });
      }
      if (bars.length >= MIN_BARS) return bars;
    } catch {
      continue;
    }
  }
  return null;
}

/**
 * Provider 2 — Yahoo Finance chart API (dividend-adjusted closes).
 */
async function fetchYahoo(symbol: string): Promise<Bar[] | null> {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(
    symbol,
  )}?range=${YEARS}y&interval=1d&includePrePost=false&events=div%2Csplit`;

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const text = await upstream(() =>
        httpGetText(url, 10000, { Referer: "https://finance.yahoo.com/" }),
      );
      const json = JSON.parse(text) as {
        chart?: {
          result?: Array<{
            timestamp?: number[];
            indicators?: {
              quote?: Array<{ close?: Array<number | null> }>;
              adjclose?: Array<{ adjclose?: Array<number | null> }>;
            };
          }>;
          error?: unknown;
        };
      };
      const result = json.chart?.result?.[0];
      const ts = result?.timestamp;
      const adj =
        result?.indicators?.adjclose?.[0]?.adjclose ??
        result?.indicators?.quote?.[0]?.close;
      if (!ts || !adj || ts.length === 0) return null;
      const bars: Bar[] = [];
      for (let i = 0; i < ts.length; i++) {
        const c = adj[i];
        if (c == null || !Number.isFinite(c) || c <= 0) continue;
        const date = new Date(ts[i] * 1000).toISOString().slice(0, 10);
        // guard against duplicate dates from any source quirk
        if (bars.length > 0 && bars[bars.length - 1].date >= date) continue;
        bars.push({ date, close: Math.round(c * 10000) / 10000 });
      }
      return bars.length >= MIN_BARS ? bars : null;
    } catch {
      if (attempt === 0) {
        await new Promise((r) => setTimeout(r, 350));
        continue;
      }
      return null;
    }
  }
  return null;
}

/* ---------- deterministic synthetic fallback ---------- */

function hashSeed(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller standard normal from a uniform PRNG. */
function gauss(rand: () => number): number {
  const u = Math.max(rand(), 1e-9);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function syntheticBars(symbol: string): Bar[] {
  const rand = mulberry32(hashSeed(symbol));
  const mu = 0.02 + rand() * 0.26; // annual drift 2%..28%
  const sigma = 0.16 + rand() * 0.42; // annual vol 16%..58%
  let price = 12 + rand() * 480;
  const bars: Bar[] = [];
  const dt = 1 / 252;
  const start = Date.now() - YEARS * 365.25 * 86400000;
  for (let t = start; t <= Date.now(); t += 86400000) {
    const day = new Date(t).getUTCDay();
    if (day === 0 || day === 6) continue;
    const shock = gauss(rand);
    const jump = rand() < 0.006 ? (rand() - 0.45) * 0.18 : 0;
    price *= Math.exp((mu - 0.5 * sigma * sigma) * dt + sigma * Math.sqrt(dt) * shock + jump);
    price = Math.max(price, 0.5);
    bars.push({
      date: new Date(t).toISOString().slice(0, 10),
      close: Math.round(price * 100) / 100,
    });
  }
  return bars;
}

async function resolvePrices(symbolRaw: string): Promise<PriceSeries> {
  const symbol = normalizeSymbol(symbolRaw);

  const cached = await readCache(symbol);
  if (cached && isFresh(cached)) {
    return { symbol, bars: cached, simulated: false };
  }

  const live = (await fetchNasdaq(symbol)) ?? (await fetchYahoo(symbol));
  if (live) {
    void writeCache(symbol, live);
    return { symbol, bars: live, simulated: false };
  }

  if (cached && cached.length >= MIN_BARS) {
    return { symbol, bars: cached, simulated: false };
  }

  return { symbol, bars: syntheticBars(symbol), simulated: true };
}

/** Fetch a single instrument's series (deduped, cached). */
export function getPrices(symbol: string): Promise<PriceSeries> {
  const key = normalizeSymbol(symbol);
  const existing = mem.get(key);
  if (existing) return existing;
  const p = resolvePrices(key).finally(() => {
    // keep resolution for a while; drop errors from cache after settling slowly
    setTimeout(() => mem.delete(key), 5 * 60 * 1000).unref?.();
  });
  mem.set(key, p);
  return p;
}

export async function getManyPrices(
  symbols: string[],
): Promise<Map<string, PriceSeries>> {
  const unique = [...new Set(symbols.map(normalizeSymbol))];
  const results = await Promise.all(unique.map((s) => getPrices(s)));
  const map = new Map<string, PriceSeries>();
  for (const r of results) map.set(r.symbol, r);
  return map;
}
