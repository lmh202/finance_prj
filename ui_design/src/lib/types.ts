/** Shared types between API routes and client components. */

export type InputMode = "weight" | "shares";

export interface HoldingInput {
  symbol: string;
  /** weight % (mode=weight) or share count (mode=shares) */
  value: number;
}

export interface StockInfo {
  symbol: string;
  name: string;
  exchange: string;
  sector: string;
  quoteType: string;
  /** true when the instrument came from live lookup vs seed universe */
  live?: boolean;
}

export interface SeriesStats {
  totalReturn: number; // 0.42 = +42%
  cagr: number;
  annVol: number;
  sharpe: number;
  sortino: number;
  maxDrawdown: number; // negative
  calmar: number;
  bestDay: number;
  worstDay: number;
  winRate: number; // 0..1
}

export interface PortfolioMetrics extends SeriesStats {
  beta: number | null; // vs SPY
  alpha: number | null; // annualized, vs SPY
  ytd: number;
}

export interface HoldingAnalysis {
  symbol: string;
  name: string;
  sector: string;
  weight: number; // resolved 0..1
  inputValue: number;
  lastPrice: number | null;
  simulated: boolean;
  stats: SeriesStats;
  contribution: number; // share of portfolio total return attributable
  /** indexed series aligned to dates (start = 100) */
  values: number[];
}

export interface BenchmarkSeries {
  symbol: string;
  name: string;
  color: string;
  values: number[]; // aligned to dates, indexed 100
  stats: SeriesStats;
}

export interface MonthCell {
  year: number;
  month: number; // 1-12
  r: number; // return within month
}

export interface AnalysisRange {
  start: string;
  end: string;
  days: number;
  truncated: boolean;
  truncatedNote?: string;
}

export interface AnalyzeResponse {
  ok: true;
  mode: InputMode;
  source: "live" | "mixed" | "simulated";
  asOf: string;
  range: AnalysisRange;
  dates: string[];
  portfolio: number[]; // indexed 100
  portfolioMetrics: PortfolioMetrics;
  benchmarks: BenchmarkSeries[];
  holdings: HoldingAnalysis[];
  sectors: { sector: string; weight: number }[];
  monthly: MonthCell[];
}

export interface AnalyzeError {
  ok: false;
  error: string;
}

export type AnalyzeResult = AnalyzeResponse | AnalyzeError;

/* ---------- range slicing (client) ---------- */

export const RANGES = [
  { id: "6M", months: 6 },
  { id: "1Y", months: 12 },
  { id: "2Y", months: 24 },
  { id: "3Y", months: 36 },
  { id: "5Y", months: 63 },
] as const;

export type RangeId = (typeof RANGES)[number]["id"];
