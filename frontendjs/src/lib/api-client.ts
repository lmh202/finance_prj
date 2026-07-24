/**
 * Typed HTTP client for the AURORA FastAPI backend.
 *
 * Every data operation that used to hit a local Next.js API route or
 * PostgreSQL now calls the backend at NEXT_PUBLIC_BACKEND_URL instead.
 * The engine pages (/health, /strategy, /news, /react, /performance, /risk)
 * consume the backend routers; the home page consumes /market/search
 * and /portfolio consumes /analysis/explore for its analytics.
 */
import type {
  AnalyzeResponse,
  BackendHolding,
  DailyRecommendation,
  EventsPayload,
  HealthReport,
  HoldingInput,
  InputMode,
  NewsEvent,
  PortfolioRisk,
  PortfolioTotals,
  PortfolioViewRow,
  ReactResponse,
  RiskEstimate,
  SplitFrame,
  StockInfo,
  StrategyRecommendation,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

/* ------------------------------------------------------------------ */
/* Errors                                                              */
/* ------------------------------------------------------------------ */

/** Expected backend conditions signalled via 409/502/503 marker details. */
export const EMPTY_PORTFOLIO = "empty_portfolio";
export const NO_HISTORY = "no_history";
/** Risk engine artifact (data/processed/risk_model.json) not built yet. */
export const NO_MODEL = "no_model";
export type ApiMarker = typeof EMPTY_PORTFOLIO | typeof NO_HISTORY | typeof NO_MODEL;

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
      (res.status === 409 && body.detail === EMPTY_PORTFOLIO) ||
      (res.status === 502 && body.detail === NO_HISTORY) ||
      (res.status === 503 && body.detail === NO_MODEL)
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

/** Per-stock confluence (EMA/MACD/RSI) action, cross-referenced with holdings. */
export async function fetchStrategyRecommendations(
  signal?: AbortSignal
): Promise<StrategyRecommendation[]> {
  return (await get("/strategy/recommendations", signal)) as StrategyRecommendation[];
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

/**
 * `symbols`, if given, OVERRIDES the saved portfolio as the tracked-ticker
 * list — including `[]`, which means "no tickers selected" (general market
 * news only), not "fall back to the portfolio". Omit it to keep the old
 * default of deriving tickers from the saved portfolio server-side.
 */
export async function fetchEssentialNews(
  maxEvents = 5,
  symbols?: string[],
  signal?: AbortSignal
): Promise<NewsEvent[]> {
  const params = new URLSearchParams({ max_events: String(maxEvents) });
  if (symbols !== undefined) params.set("symbols", symbols.join(","));
  return (await get(`/news/essential?${params.toString()}`, signal)) as NewsEvent[];
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

/* ------------------------------------------------------------------ */
/* Risk Engine — HAR volatility forecast + Filtered Historical         */
/* Simulation (backend/src/risk_engine)                                */
/* ------------------------------------------------------------------ */

/** Per-held-symbol downside risk at one horizon (5 or 20 trading days). */
export async function fetchRiskEstimates(
  horizon: 5 | 20,
  signal?: AbortSignal
): Promise<RiskEstimate[]> {
  return (await get(`/risk/estimates?horizon=${horizon}`, signal)) as RiskEstimate[];
}

/** Aggregate portfolio downside risk (EWMA-correlation VaR/ES) at one horizon. */
export async function fetchPortfolioRisk(
  horizon: 5 | 20,
  signal?: AbortSignal
): Promise<PortfolioRisk> {
  return (await get(`/risk/portfolio?horizon=${horizon}`, signal)) as PortfolioRisk;
}
