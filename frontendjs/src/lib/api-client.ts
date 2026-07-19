/**
 * Typed HTTP client for the AURORA FastAPI backend.
 *
 * Every data operation that used to hit a local Next.js API route or
 * PostgreSQL now calls the backend at NEXT_PUBLIC_BACKEND_URL instead.
 * The engine pages (/health, /strategy, /news, /react, /performance)
 * consume the four engine routers; the home page consumes /market/search
 * and /portfolio consumes /analysis/explore for its analytics.
 */
import type {
  AnalyzeResponse,
  AssetSignal,
  BackendHolding,
  DailyRecommendation,
  EventsPayload,
  HealthReport,
  HoldingInput,
  InputMode,
  NewsEvent,
  PortfolioTotals,
  PortfolioViewRow,
  ReactResponse,
  RegimeState,
  SplitFrame,
  StockInfo,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

/* ------------------------------------------------------------------ */
/* Errors                                                              */
/* ------------------------------------------------------------------ */

/** Expected backend conditions signalled via 409/502 marker details. */
export const EMPTY_PORTFOLIO = "empty_portfolio";
export const NO_HISTORY = "no_history";
export type ApiMarker = typeof EMPTY_PORTFOLIO | typeof NO_HISTORY;

export class ApiMarkerError extends Error {
  constructor(public readonly marker: ApiMarker) {
    super(marker);
    this.name = "ApiMarkerError";
  }
}

/** The backend process is not reachable at all. */
export class BackendDownError extends Error {
  constructor() {
    super("Backend not reachable");
    this.name = "BackendDownError";
  }
}

/* ------------------------------------------------------------------ */
/* Request plumbing                                                    */
/* ------------------------------------------------------------------ */

async function request(path: string, init?: RequestInit): Promise<unknown> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new BackendDownError();
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
    if (
      (res.status === 409 || res.status === 502) &&
      (body.detail === EMPTY_PORTFOLIO || body.detail === NO_HISTORY)
    ) {
      throw new ApiMarkerError(body.detail);
    }
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

function get(path: string, signal?: AbortSignal): Promise<unknown> {
  return request(path, { signal });
}

function send(
  method: "POST" | "PUT",
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<unknown> {
  return request(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
}

/** "2024-05-17T00:00:00.000" → "2024-05-17" (what fmtDate expects). */
function isoDay(iso: string): string {
  return iso.slice(0, 10);
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

/** Latest close per symbol; a symbol that could not be priced is absent. */
export async function fetchLatestPrices(
  symbols: string[],
  signal?: AbortSignal
): Promise<Record<string, number>> {
  if (symbols.length === 0) return {};
  return (await get(
    `/market/prices?symbols=${encodeURIComponent(symbols.join(","))}`,
    signal
  )) as Record<string, number>;
}

/* ------------------------------------------------------------------ */
/* Portfolio analysis (analyzer home page)                             */
/* ------------------------------------------------------------------ */

export async function analyze(
  holdings: HoldingInput[],
  mode: InputMode,
  signal?: AbortSignal
): Promise<AnalyzeResponse> {
  const json = (await send(
    "POST",
    "/analysis/explore",
    {
      holdings: holdings.map((h) => ({ symbol: h.symbol, value: h.value })),
      mode,
    },
    signal
  )) as { ok: boolean; detail?: string } & AnalyzeResponse;

  if (!json.ok) {
    throw new Error(json.detail ?? "Analysis failed");
  }

  return json as AnalyzeResponse;
}

/* ------------------------------------------------------------------ */
/* Saved portfolio (backend data/portfolio.csv — used by the engines)  */
/* ------------------------------------------------------------------ */

export async function fetchPortfolio(
  signal?: AbortSignal
): Promise<BackendHolding[]> {
  const json = (await get("/portfolio", signal)) as { holdings: BackendHolding[] };
  return json.holdings;
}

export async function savePortfolio(
  holdings: BackendHolding[]
): Promise<{ holdings: BackendHolding[]; problems: string[] }> {
  return (await send("PUT", "/portfolio", { holdings })) as {
    holdings: BackendHolding[];
    problems: string[];
  };
}

export async function addHolding(
  holding: BackendHolding
): Promise<BackendHolding[]> {
  const json = (await send("POST", "/portfolio/holdings", holding)) as {
    holdings: BackendHolding[];
  };
  return json.holdings;
}

/** Parse only — nothing is saved until the result is PUT back. */
export async function parsePortfolioCsv(
  csv: string
): Promise<{ holdings: BackendHolding[]; problems: string[] }> {
  return (await send("POST", "/portfolio/parse-csv", { csv })) as {
    holdings: BackendHolding[];
    problems: string[];
  };
}

export async function loadSamplePortfolio(): Promise<{
  holdings: BackendHolding[];
  cash: number;
}> {
  return (await send("POST", "/portfolio/load-sample")) as {
    holdings: BackendHolding[];
    cash: number;
  };
}

export async function fetchCash(signal?: AbortSignal): Promise<number> {
  return ((await get("/portfolio/cash", signal)) as { cash: number }).cash;
}

export async function saveCash(cash: number): Promise<void> {
  await send("PUT", "/portfolio/cash", { cash });
}

export async function fetchPortfolioView(
  signal?: AbortSignal
): Promise<{ view: PortfolioViewRow[]; totals: PortfolioTotals }> {
  return (await get("/portfolio/view", signal)) as {
    view: PortfolioViewRow[];
    totals: PortfolioTotals;
  };
}

/* ------------------------------------------------------------------ */
/* Engine 1 — Portfolio Health                                         */
/* ------------------------------------------------------------------ */

export async function fetchHealthReport(
  signal?: AbortSignal
): Promise<HealthReport> {
  return (await get("/health/report", signal)) as HealthReport;
}

/* ------------------------------------------------------------------ */
/* Engine 2 — Daily Strategy                                           */
/* ------------------------------------------------------------------ */

export async function fetchRegime(signal?: AbortSignal): Promise<RegimeState> {
  return (await get("/strategy/regime", signal)) as RegimeState;
}

export async function fetchSignals(
  signal?: AbortSignal
): Promise<AssetSignal[]> {
  return (await get("/strategy/signals", signal)) as AssetSignal[];
}

/** Walk-forward backtest curves — growth of $1 per strategy column. */
export interface BacktestCurves {
  dates: string[]; // "YYYY-MM-DD"
  columns: string[]; // buy_hold, equal_weight, + strategy runs
  rows: (number | null)[][]; // aligned to dates
}

export async function fetchBacktest(
  signal?: AbortSignal
): Promise<BacktestCurves> {
  const split = (await get("/strategy/backtest", signal)) as SplitFrame;
  return {
    dates: (split.index ?? []).map(isoDay),
    columns: split.columns ?? [],
    rows: split.data ?? [],
  };
}

/* ------------------------------------------------------------------ */
/* Engine 3 — Essential News                                           */
/* ------------------------------------------------------------------ */

export async function fetchEssentialNews(
  maxEvents = 5,
  signal?: AbortSignal
): Promise<NewsEvent[]> {
  return (await get(`/news/essential?max_events=${maxEvents}`, signal)) as NewsEvent[];
}

export async function fetchNewsFeeds(signal?: AbortSignal): Promise<string[]> {
  return ((await get("/news/feeds", signal)) as { feeds: string[] }).feeds;
}

/* ------------------------------------------------------------------ */
/* Engine 4 — Reaction Risk & Recommendation                           */
/* ------------------------------------------------------------------ */

export async function fetchDailyRecommendation(
  signal?: AbortSignal
): Promise<DailyRecommendation> {
  return (await get("/recommendation/daily", signal)) as DailyRecommendation;
}

export async function fetchRecommendationEvents(
  maxEvents = 5,
  signal?: AbortSignal
): Promise<EventsPayload> {
  return (await get(
    `/recommendation/events?max_events=${maxEvents}`,
    signal
  )) as EventsPayload;
}

export async function reactToEvent(
  event: NewsEvent,
  signal?: AbortSignal
): Promise<ReactResponse> {
  return (await send("POST", "/recommendation/react", { event }, signal)) as ReactResponse;
}
