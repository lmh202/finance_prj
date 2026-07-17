/**
 * Typed HTTP client for the AURORA FastAPI backend.
 *
 * Every data operation that used to hit a local Next.js API route or
 * PostgreSQL now calls the backend at NEXT_PUBLIC_BACKEND_URL instead.
 */
import type { AnalyzeResponse, HoldingInput, InputMode, StockInfo } from "./types";

const BASE = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

async function get(path: string, signal?: AbortSignal): Promise<unknown> {
  const res = await fetch(`${BASE}${path}`, { signal });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(
      (detail as { detail?: string })?.detail ?? `HTTP ${res.status}`
    );
  }
  return res.json();
}

async function post(path: string, body: unknown): Promise<unknown> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(
      (detail as { detail?: string })?.detail ?? `HTTP ${res.status}`
    );
  }
  return res.json();
}

/* ------------------------------------------------------------------ */
/* Stock search                                                        */
/* ------------------------------------------------------------------ */

interface SearchResult {
  symbol: string;
  name: string;
  exchange: string;
  sector: string;
  quoteType: string;
}

export async function searchStocks(
  q: string,
  signal?: AbortSignal
): Promise<StockInfo[]> {
  const results = (await get(
    `/market/search?q=${encodeURIComponent(q)}&limit=10`,
    signal
  )) as SearchResult[];
  return results.map((r) => ({
    symbol: r.symbol,
    name: r.name,
    exchange: r.exchange,
    sector: r.sector,
    quoteType: r.quoteType,
    live: false,
  }));
}

/* ------------------------------------------------------------------ */
/* Portfolio analysis                                                  */
/* ------------------------------------------------------------------ */

export async function analyze(
  holdings: HoldingInput[],
  mode: InputMode
): Promise<AnalyzeResponse> {
  const json = (await post("/analysis/explore", {
    holdings: holdings.map((h) => ({ symbol: h.symbol, value: h.value })),
    mode,
  })) as { ok: boolean; detail?: string } & AnalyzeResponse;

  if (!json.ok) {
    throw new Error(json.detail ?? "Analysis failed");
  }

  return json as AnalyzeResponse;
}
