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

/* ------------------------------------------------------------------ */
/* AURORA engine API (FastAPI backend) — see backend/src/interfaces.py */
/* ------------------------------------------------------------------ */

/** pandas DataFrame serialized with orient="split". */
export interface SplitFrame {
  columns: string[];
  index: string[];
  data: (number | null)[][];
}

/** GET /health/report — Engine 1, portfolio_health.compute_health(). */
export interface HealthReport {
  score: number;
  /** annual_return, annual_volatility, sharpe, sortino, max_drawdown,
   *  beta, diversification, largest_position — NaN arrives as null */
  metrics: Record<string, number | null>;
  strengths: string[];
  weaknesses: string[];
  correlation: SplitFrame | null; // symbols × symbols
}

/** GET /strategy/recommendations — Engine 2, daily_strategy.recommend_signals().
 *  One per-stock action from the EMA/MACD/RSI confluence layer, cross-referenced
 *  with what's held. `score` is the raw confluence score, roughly ±3.5
 *  (EMA ±1.5, MACD ±1.0, RSI ±1.0). */
export interface StrategyRecommendation {
  symbol: string;
  recommendation: "BUY" | "ADD" | "TRIM" | "SELL" | "HOLD" | string;
  held: boolean;
  raw_signal: "BUY" | "SELL" | "HOLD" | string;
  score: number;
  quantity: number;
  avg_buy_price: number | null;
  current_price: number | null;
  unrealized_pnl_pct: number | null; // already a percentage (12.3 = +12.3%)
  reasons: string[];
}

/** GET /news/essential — Engine 3, one classified news story. */
export interface NewsEvent {
  title: string;
  source: string;
  url: string;
  published: string | null; // ISO datetime
  category: string;
  sentiment: number; // -1..+1
  importance: number; // 0..100
  affected_symbols: string[];
  impact: Record<string, string>;
  summary: string;
}

/** Engine 4 — how risky it is to trade on an event. */
export interface ReactionRisk {
  risk_pct: number; // 0..100
  factors: Record<string, number>; // each 0..1, 0 = safe to act
  reasons: string[];
  suggestion: "do_nothing" | "moderate" | "aggressive" | string;
}

export interface ProposedTrade {
  symbol: string;
  weight_change_pct: number; // +2.0 = increase weight by 2 points
  reason: string;
}

export interface Recommendation {
  kind: "daily" | "event" | string;
  trades: ProposedTrade[];
  confidence: number; // 0..1
  explanation: string;
  reaction_risk: ReactionRisk | null;
}

/** Timestamps behind one fusion result; any input may be missing. */
export interface FusionAsOf {
  strategy: string | null;
  news: string | null;
  health: string | null;
  risk: string | null;
}

/** One per-asset explanation of the risk-controlled allocation.
 *  NOTE: aurora_score is 0..100 and every *_pct is already in percentage
 *  points — do NOT pass them through fmtPct(), which multiplies by 100. */
export interface FusionResult {
  symbol: string;
  aurora_score: number; // 0..100, 50 = neutral
  outlook: string;
  risk_level: "Low" | "Moderate" | "High" | "Extreme" | string;
  action: string;
  confidence: number; // 0..1
  confidence_label: "High" | "Medium" | "Low" | string;
  strategy_score: number; // -1..1
  news_score: number; // -1..1, context only
  health_score: number; // 0..100
  health_normalized: number;
  risk_percentile: number; // 0..100
  risk_factor: number;
  raw_score: number;
  adjusted_score: number;
  component_weights: Record<string, number>;
  news_articles: number;
  news_confidence: string;
  news_titles: string[];
  position_change_pct: number; // percentage points
  conflict: boolean;
  extreme_volatility: boolean;
  stale_inputs: string[];
  unavailable_inputs: string[];
  as_of: FusionAsOf;
  why: string[];
}

/** Causal market-stress state driving the adaptive risk budget. */
export interface MarketStress {
  state: "calm" | "stressed" | "unknown" | string;
  stressed: boolean;
  benchmark: string;
  volatility_percentile: number | null; // 0..1 fraction
  realised_volatility_annualised: number | null; // fraction
  stress_threshold: number; // 0..1 fraction
  volatility_window_sessions: number;
  percentile_window_sessions: number;
  percentile_min_periods: number;
  observations: number;
  as_of: string | null;
  unavailable_reason: string | null;
}

export interface DecisionSymbolMeta {
  strategy_direction?: number;
  direction_signal?: number;
  expected_return_5d_proxy: number;
  risk_sigma_daily_5d: number;
  risk_level_5d: number;
  risk_news_applied?: boolean;
  risk_news_quality?: string;
  news_available?: boolean;
  weight_before_pct: number;
  weight_after_pct: number;
}

/** Audit trail for the numeric decision. Only production_mode is guaranteed —
 *  the fallback decision paths emit a much smaller block. */
export interface DecisionMeta {
  production_mode: string;
  model_version?: string;
  fallback_reason?: string | null;
  optimizer_success?: boolean;
  direct_news_residual_applied?: boolean;
  news_via_risk_share?: number; // 0..1 fraction
  alpha_scaling?: string;
  strategy_information_coefficient?: number;
  risk_aversion_policy?: string;
  calm_risk_aversion?: number;
  stressed_risk_aversion?: number;
  base_risk_aversion?: number;
  effective_risk_aversion?: number;
  cash_before_pct?: number; // TRUE cash, percentage points
  cash_after_pct?: number;
  /** Held weight the optimiser does not manage — the benchmark, or a holding
   *  with no risk estimate. Not cash: it caps the gross budget instead. */
  locked_weight_pct?: number;
  locked_positions?: Record<string, number>;
  target_gross_pct?: number;
  predicted_annual_volatility?: number; // fraction
  target_annual_volatility?: number; // fraction
  market_stress?: MarketStress;
  symbols?: Record<string, DecisionSymbolMeta>;
}

export interface ExplanationMeta {
  source: string;
  reason: string | null;
  /** Set when the explanation layer failed. The recommendation is still
   *  valid — explanation is presentational and never blocks the decision. */
  fusion_error?: string;
}

/** GET /recommendation/daily */
export interface DailyRecommendation {
  recommendation: Recommendation;
  fusion_results: FusionResult[];
  decision_meta: DecisionMeta;
  explanation_meta: ExplanationMeta;
  health_before: number | null;
  health_after: number | null;
}

/** GET /recommendation/events */
export interface EventsPayload {
  events: NewsEvent[];
  demo: boolean;
}

/** POST /recommendation/react */
export interface ReactResponse {
  risk: ReactionRisk;
  recommendation: Recommendation;
}

/** GET /risk/estimates — one symbol × horizon downside-risk estimate
 *  (HAR volatility forecast + Filtered Historical Simulation). */
export interface RiskEstimate {
  symbol: string;
  horizon: number;
  sigma_daily: number;
  sigma_h: number; // forecast vol over the horizon
  var_95: number; // downside VaR, horizon return threshold (negative)
  var_99: number;
  es_95: number; // expected shortfall / CVaR
  band_lo: number;
  band_hi: number;
  risk_level: number; // 0..100, sigma percentile vs. the stock's own history
  as_of: string | null;
  has_history: boolean;
  model_version: string;
  /** Whether live news attention reached this forecast. False on a degraded
   *  store — an observed zero is a real state, a broken feed is not. */
  news_applied: boolean;
  news_quality: string; // fresh | no_news | missing_store | stale_store | ...
}

/** GET /risk/portfolio — aggregate portfolio downside risk for one horizon. */
export interface PortfolioRisk {
  horizon: number;
  n_holdings: number;
  sigma_h: number;
  var_95: number;
  var_99: number;
  es_95: number;
  diversification_ratio: number; // portfolio VaR / sum(w * VaR_i); <1 = diversified
  as_of: string | null;
}

/* ---------- backend-persisted portfolio (data/portfolio.csv) ---------- */

export interface BackendHolding {
  symbol: string;
  name: string;
  shares: number;
  buy_price: number;
}

/** One row of GET /portfolio/view — holdings joined with live prices. */
export interface PortfolioViewRow extends BackendHolding {
  current_price: number | null;
  has_price: boolean;
  cost_value: number;
  market_value: number;
  pnl: number;
  pnl_pct: number | null; // null when cost basis is 0
  weight_pct: number;
}

export interface PortfolioTotals {
  market_value: number; // incl. cash
  invested: number;
  cost: number;
  pnl: number;
  cash: number;
}

/* ---------- range slicing (client) ---------- */

export type RangeUnit = "days" | "months" | "years";

export interface RangeSpec {
  unit: RangeUnit;
  amount: number;
}

export const RANGES = [
  { id: "1M", spec: { unit: "months", amount: 1 } },
  { id: "6M", spec: { unit: "months", amount: 6 } },
  { id: "1Y", spec: { unit: "months", amount: 12 } },
  { id: "3Y", spec: { unit: "months", amount: 36 } },
  { id: "5Y", spec: { unit: "months", amount: 63 } },
] as const;

export type RangeId = (typeof RANGES)[number]["id"];
