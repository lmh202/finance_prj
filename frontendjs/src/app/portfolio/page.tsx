"use client";

/**
 * Portfolio builder + analytics — mirrors the Streamlit Home page
 * (frontend/app.py) and absorbs the retired "/" analyzer.
 *
 * This page edits the SAVED portfolio on the backend (data/portfolio.csv +
 * cash) — the portfolio every engine page reads — and analyzes it via
 * POST /analysis/explore in "shares" mode (constant-mix curve, stats,
 * allocation donut, monthly heatmap, per-position Sharpe/contribution).
 * The home page hands off searched symbols through the ?add= query param.
 */

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  Briefcase,
  ChevronDown,
  ChevronUp,
  Download,
  Eye,
  EyeOff,
  FileUp,
  Loader2,
  PackageOpen,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  RefreshCw,
  ServerCrash,
  Trash2,
  TriangleAlert,
  Undo2,
  Waypoints,
  X,
} from "lucide-react";
import {
  BTN_GHOST,
  BTN_PRIMARY,
  Metric,
  Note,
  PageShell,
  Section,
  StateCard,
  ThinBar,
} from "@/components/EngineShell";
import { AllocationDonut, SectorBars } from "@/components/AllocationDonut";
import { MonthlyHeatmap } from "@/components/MonthlyHeatmap";
import { PerformanceChart, type ChartSeries } from "@/components/PerformanceChart";
import { SearchBox } from "@/components/SearchBox";
import { Sparkline } from "@/components/Sparkline";
import { StatsRow } from "@/components/StatsRow";
import {
  addHolding,
  analyze,
  BackendDownError,
  fetchCash,
  fetchLatestPrices,
  fetchPortfolio,
  fetchPortfolioView,
  loadSamplePortfolio,
  parsePortfolioCsv,
  saveCash,
  savePortfolio,
} from "@/lib/api-client";
import { fmtDate, fmtNum, fmtPct, signClass } from "@/lib/format";
import {
  RANGES,
  type AnalyzeResponse,
  type BackendHolding,
  type PortfolioTotals,
  type PortfolioViewRow,
  type RangeId,
  type StockInfo,
} from "@/lib/types";
import { deriveRangeView } from "@/lib/view";

const ACCENT = "#B3F34C";

const INPUT =
  "w-full rounded-lg border border-line bg-white/[0.04] px-2.5 py-1.5 font-mono text-sm tabular text-ink outline-none transition-colors focus:border-accent/50";
const LABEL = "font-mono text-[10px] uppercase tracking-wider text-mut";

function fmtMoney(v: number): string {
  if (v)
    return (
      "$" +
      v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    );
  else
    return "";
}

/** Editable string-typed row so number inputs stay free to type into. */
interface EditRow {
  symbol: string;
  name: string;
  shares: string;
  buy_price: string;
}

/** ≤4 decimals so CSV float noise (331.8699951172) doesn't flood the inputs. */
function trimNum(v: number): string {
  return String(Math.round(v * 1e4) / 1e4);
}

/** Saved symbols may use directory dots (BRK.B); /analysis/explore normalizes
 *  to dashes (BRK-B). Match its _normalize_symbol so lookups line up. */
function analysisKey(symbol: string): string {
  return symbol.trim().toUpperCase().replace(/\./g, "-").slice(0, 14);
}

function toEdit(h: BackendHolding): EditRow {
  return {
    symbol: h.symbol,
    name: h.name ?? "",
    shares: trimNum(h.shares),
    buy_price: trimNum(h.buy_price),
  };
}

function toHolding(r: EditRow): BackendHolding {
  return {
    symbol: r.symbol,
    name: r.name,
    shares: Number.parseFloat(r.shares) || 0,
    buy_price: Number.parseFloat(r.buy_price) || 0,
  };
}

function sameRows(a: EditRow[], b: EditRow[]): boolean {
  if (a.length !== b.length) return false;
  return a.every(
    (r, i) =>
      r.symbol === b[i].symbol &&
      r.shares === b[i].shares &&
      r.buy_price === b[i].buy_price
  );
}

function downloadText(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function holdingsCsv(rows: BackendHolding[]): string {
  const lines = ["symbol,name,shares,buy_price"];
  for (const r of rows) {
    const name = /[",]/.test(r.name) ? `"${r.name.replace(/"/g, '""')}"` : r.name;
    lines.push(`${r.symbol},${name},${r.shares},${r.buy_price}`);
  }
  return lines.join("\n") + "\n";
}

interface PendingAdd {
  info: StockInfo;
  shares: string;
  buyPrice: string;
  error: string | null;
  /** true when the symbol arrived from the home-page search (?add=) —
   *  shares start empty and the panel prompts for them. */
  fromSearch?: boolean;
}

/** useSearchParams needs a Suspense boundary during prerendering. */
export default function PortfolioPage() {
  return (
    <Suspense fallback={null}>
      <PortfolioPageInner />
    </Suspense>
  );
}

function PortfolioPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [boot, setBoot] = useState<"loading" | "down" | "error" | "ready">("loading");
  const [bootMsg, setBootMsg] = useState("");

  const [saved, setSaved] = useState<BackendHolding[]>([]);
  const [rows, setRows] = useState<EditRow[]>([]);
  const [cash, setCash] = useState(0);
  const [cashText, setCashText] = useState("0");
  const [view, setView] = useState<PortfolioViewRow[]>([]);
  const [totals, setTotals] = useState<PortfolioTotals | null>(null);

  const [busy, setBusy] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const [collapsedAlloc, setCollapsedAlloc] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [problems, setProblems] = useState<string[]>([]);
  const [pending, setPending] = useState<PendingAdd | null>(null);
  const [confirmSample, setConfirmSample] = useState(false);
  const [importPrev, setImportPrev] = useState<{
    holdings: BackendHolding[];
    problems: string[];
  } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  /* ---- analytics (the retired analyzer, fed by the saved portfolio) ---- */
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  /** Bump to retry the analysis after a transient failure — the effect below
   *  re-runs whenever this changes, even if saved/boot stay the same. */
  const [analysisAttempt, setAnalysisAttempt] = useState(0);
  const [range, setRange] = useState<RangeId>("5Y");
  const [showSpy, setShowSpy] = useState(true);
  const [showQqq, setShowQqq] = useState(false);

  /* ---- symbol handed off from the home-page search (?add=NVDA) ---- */
  const consumedAdd = useRef(false);
  useEffect(() => {
    const raw = searchParams.get("add");
    if (!raw) return;
    const name = searchParams.get("name")?.trim();
    // Deferred so no setState runs synchronously inside the effect body;
    // the ref survives strict-mode double fires, the timer their cleanup.
    const timer = setTimeout(() => {
      if (consumedAdd.current) return;
      consumedAdd.current = true;
      const symbol = raw.trim().toUpperCase();
      if (symbol) {
        setPending({
          info: { symbol, name: name || symbol, exchange: "", sector: "", quoteType: "" },
          shares: "",
          buyPrice: "",
          error: null,
          fromSearch: true,
        });
      }
      router.replace("/portfolio", { scroll: false });
    }, 0);
    return () => clearTimeout(timer);
  }, [searchParams, router]);

  const applySaved = useCallback((h: BackendHolding[]) => {
    setSaved(h);
    setRows(h.map(toEdit));
  }, []);

  const refreshView = useCallback(async () => {
    const { view, totals } = await fetchPortfolioView();
    setView(view);
    setTotals(totals);
  }, []);

  // No synchronous setState before the first await — safe to call from the
  // mount effect; retry() flips boot back to "loading" from its own handler.
  const bootstrap = useCallback(async () => {
    try {
      const [holdings, cashValue, viewData] = await Promise.all([
        fetchPortfolio(),
        fetchCash(),
        fetchPortfolioView(),
      ]);
      applySaved(holdings);
      setCash(cashValue);
      setCashText(String(cashValue));
      setView(viewData.view);
      setTotals(viewData.totals);
      setBoot("ready");
    } catch (err) {
      setBootMsg(err instanceof Error ? err.message : String(err));
      setBoot(err instanceof BackendDownError ? "down" : "error");
    }
  }, [applySaved]);

  useEffect(() => {
    // Deferred so no setState runs synchronously inside the effect body
    // (react-hooks/set-state-in-effect); also skips strict-mode double fires.
    const timer = setTimeout(() => void bootstrap(), 0);
    return () => clearTimeout(timer);
  }, [bootstrap]);

  /* Re-analyze whenever the SAVED holdings change (shares mode: weights
     derive from share count × latest close, like the engines see them).
     All setState lives inside the debounce timer so nothing fires
     synchronously in the effect body. */
  useEffect(() => {
    if (boot !== "ready") return;
    const positions = saved.filter((h) => h.shares > 0);
    const ctrl = new AbortController();
    const t = setTimeout(
      async () => {
        if (positions.length === 0) {
          setAnalysis(null);
          setAnalysisError(null);
          setAnalyzing(false);
          return;
        }
        setAnalyzing(true);
        setAnalysisError(null);
        try {
          const json = await analyze(
            positions.map((h) => ({ symbol: h.symbol, value: h.shares })),
            "shares",
            ctrl.signal
          );
          if (!ctrl.signal.aborted) setAnalysis(json);
        } catch (err) {
          if (!ctrl.signal.aborted) {
            setAnalysisError(
              err instanceof BackendDownError
                ? "Backend not reachable."
                : err instanceof Error
                  ? err.message
                  : "Could not analyze the portfolio."
            );
          }
        } finally {
          if (!ctrl.signal.aborted) setAnalyzing(false);
        }
      },
      positions.length === 0 ? 0 : 300
    );
    return () => {
      ctrl.abort();
      clearTimeout(t);
    };
  }, [boot, saved, analysisAttempt]);

  async function run(name: string, fn: () => Promise<void>) {
    setBusy(name);
    setProblems([]);
    try {
      await fn();
    } catch (err) {
      setProblems([err instanceof Error ? err.message : String(err)]);
    } finally {
      setBusy(null);
    }
  }

  /* ------------------------------- derived ------------------------------- */

  const savedRows = useMemo(() => saved.map(toEdit), [saved]);
  const dirty = boot === "ready" && !sameRows(rows, savedRows);
  const existing = useMemo(() => new Set(saved.map((h) => h.symbol)), [saved]);
  const viewBy = useMemo(() => new Map(view.map((r) => [r.symbol, r])), [view]);
  const unpriced = useMemo(
    () => view.filter((r) => !r.has_price).map((r) => r.symbol),
    [view]
  );

  const displayRows = useMemo(() => {
    const value = (r: EditRow) => viewBy.get(r.symbol)?.market_value ?? 0;
    return [...rows].sort((a, b) => value(b) - value(a));
  }, [rows, viewBy]);

  const alloc = useMemo(() => {
    if (!totals || totals.market_value <= 0) return [];
    const items = view
      .map((r) => ({ label: r.symbol, value: r.market_value, weight: r.weight_pct, cash: false }))
      .sort((a, b) => b.weight - a.weight);
    if (totals.cash > 0) {
      items.push({
        label: "Cash",
        value: totals.cash,
        weight: (totals.cash / totals.market_value) * 100,
        cash: true,
      });
    }
    return items;
  }, [view, totals]);

  /* ---- analysis view (range-sliced client-side, no refetch) ---- */
  const months = RANGES.find((r) => r.id === range)!.months;
  const rangeView = useMemo(
    () => (analysis ? deriveRangeView(analysis, months) : null),
    [analysis, months]
  );
  const analysisBy = useMemo(
    () => new Map((rangeView?.holdings ?? []).map((h) => [h.symbol, h])),
    [rangeView]
  );
  /* Holdings the analysis dropped (no usable history / backend cap).
   * Only show after the analysis settled — during the debounce + round-trip
   * a newly added symbol is absent from the stale analysis object and would
   * flash a false "no history" label. */
  const excluded = useMemo(() => {
    if (!analysis || analyzing) return [];
    const have = new Set(analysis.holdings.map((h) => h.symbol));
    return saved
      .filter((h) => h.shares > 0 && !have.has(analysisKey(h.symbol)))
      .map((h) => h.symbol);
  }, [analysis, analyzing, saved]);
  const maxContrib = useMemo(
    () =>
      Math.max(...(rangeView?.holdings ?? []).map((h) => Math.abs(h.contribution)), 1e-9),
    [rangeView]
  );
  const chartSeries: ChartSeries[] = useMemo(() => {
    if (!rangeView) return [];
    const out: ChartSeries[] = [
      { id: "PF", label: "Portfolio", color: ACCENT, values: rangeView.portfolio, width: 2.5 },
    ];
    for (const b of rangeView.benchmarks) {
      if (b.symbol === "SPY" && showSpy) out.push({ id: "SPY", label: b.name, color: b.color, values: b.values });
      if (b.symbol === "QQQ" && showQqq) out.push({ id: "QQQ", label: b.name, color: b.color, values: b.values });
    }
    return out;
  }, [rangeView, showSpy, showQqq]);
  const spyStats = rangeView?.benchmarks.find((b) => b.symbol === "SPY")?.stats ?? null;
  const endReturn = rangeView
    ? rangeView.portfolio[rangeView.portfolio.length - 1] / 100 - 1
    : 0;

  /* ------------------------------- actions ------------------------------- */

  function startAdd(info: StockInfo) {
    setPending({ info, shares: "1", buyPrice: "", error: null });
  }

  async function confirmAdd() {
    if (!pending) return;
    const shares = Number.parseFloat(pending.shares) || 0;
    if (shares <= 0) {
      setPending({ ...pending, error: "Shares must be greater than zero." });
      return;
    }
    await run("add", async () => {
      let price = Number.parseFloat(pending.buyPrice) || 0;
      if (price <= 0) {
        const prices = await fetchLatestPrices([pending.info.symbol]);
        const live = prices[pending.info.symbol];
        if (live == null) {
          setPending({
            ...pending,
            error: `No live price available for ${pending.info.symbol} — enter your buy price manually.`,
          });
          return;
        }
        price = live;
      }
      const holdings = await addHolding({
        symbol: pending.info.symbol,
        name: pending.info.name,
        shares,
        buy_price: price,
      });
      applySaved(holdings);
      setPending(null);
      await refreshView();
    });
  }

  function editRow(symbol: string, field: "shares" | "buy_price", value: string) {
    setRows((rs) => rs.map((r) => (r.symbol === symbol ? { ...r, [field]: value } : r)));
  }

  function removeRow(symbol: string) {
    setRows((rs) => rs.filter((r) => r.symbol !== symbol));
  }

  async function saveRows() {
    await run("save", async () => {
      const { holdings, problems } = await savePortfolio(rows.map(toHolding));
      applySaved(holdings);
      setProblems(problems);
      await refreshView();
    });
  }

  async function commitCash() {
    const value = Number.parseFloat(cashText) || 0;
    if (value === cash) return;
    await run("cash", async () => {
      await saveCash(value);
      setCash(value);
      setCashText(String(value));
      await refreshView();
    });
  }

  async function loadSample() {
    await run("sample", async () => {
      const { holdings, cash: sampleCash } = await loadSamplePortfolio();
      applySaved(holdings);
      setCash(sampleCash);
      setCashText(String(sampleCash));
      await refreshView();
    });
  }

  /** Loading the sample REPLACES the saved portfolio — two-step confirm
   *  whenever there is something to lose. */
  function onLoadSampleClick() {
    if (saved.length > 0 && !confirmSample) {
      setConfirmSample(true);
      setTimeout(() => setConfirmSample(false), 4000);
      return;
    }
    setConfirmSample(false);
    void loadSample();
  }

  async function onCsvPicked(file: File) {
    const text = await file.text();
    await run("parse", async () => {
      setImportPrev(await parsePortfolioCsv(text));
    });
  }

  async function confirmImport() {
    if (!importPrev) return;
    await run("import", async () => {
      const { holdings, problems } = await savePortfolio(importPrev.holdings);
      applySaved(holdings);
      setProblems(problems);
      setImportPrev(null);
      await refreshView();
    });
  }

  /* ------------------------------- render -------------------------------- */

  if (boot !== "ready") {
    return (
      <PageShell
        title="Portfolio"
        icon={Briefcase}
        caption="The saved AURORA portfolio — what every engine page reads"
        wide
      >
        {boot === "loading" ? (
          <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
            <div className="space-y-4">
              <div className="skeleton h-40 rounded-2xl" />
              <div className="skeleton h-64 rounded-2xl" />
            </div>
            <div className="space-y-4">
              <div className="skeleton h-24 rounded-2xl" />
              <div className="skeleton h-80 rounded-2xl" />
            </div>
          </div>
        ) : (
          <StateCard
            icon={boot === "down" ? ServerCrash : TriangleAlert}
            iconTone="text-loss"
            title={boot === "down" ? "Backend not running" : "Something went wrong"}
            body={
              boot === "down" ? (
                <>
                  Start it with{" "}
                  <code className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[11px] text-ink/90">
                    scripts\dev.ps1
                  </code>{" "}
                  or{" "}
                  <code className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[11px] text-ink/90">
                    uvicorn main:app --app-dir backend --port 8000
                  </code>
                  , then retry.
                </>
              ) : (
                bootMsg
              )
            }
            actions={
              <button
                onClick={() => {
                  setBoot("loading");
                  void bootstrap();
                }}
                className={BTN_PRIMARY}
              >
                <RefreshCw className="size-3.5" />
                Retry
              </button>
            }
          />
        )}
      </PageShell>
    );
  }

  const pnlPct =
    totals && totals.cost > 0 ? (totals.pnl / totals.cost) * 100 : 0;

  return (
    <PageShell
      title="Portfolio"
      icon={Briefcase}
      caption="The saved AURORA portfolio — what every engine page reads"
      wide
      actions={
        <button
          onClick={() => void run("refresh", refreshView)}
          disabled={busy != null}
          className={BTN_GHOST}
          title="Refresh prices"
        >
          <RefreshCw
            className={`size-3.5 ${busy === "refresh" ? "animate-spin text-accent" : ""}`}
          />
          Refresh prices
        </button>
      }
    >
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
        className={`grid gap-4 ${sidebarOpen ? "lg:grid-cols-[360px_minmax(0,1fr)]" : ""}`}
      >
        {/* ------------------------------ left rail ------------------------------ */}
        <AnimatePresence initial={false}>
          {sidebarOpen ? (
            <motion.aside
              key="sidebar-open"
              initial={{ x: -380, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -380, opacity: 0 }}
              transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
              className="flex flex-col gap-4 lg:sticky lg:top-[72px] lg:self-start"
            >
          <Section
            title="Add a holding"
            action={
              <button
                onClick={() => setSidebarOpen((o) => !o)}
                className="rounded-md p-1 text-mut transition-colors hover:text-ink/80"
                title={sidebarOpen ? "Hide panel" : "Show panel"}
                aria-label={sidebarOpen ? "Hide panel" : "Show panel"}
              >
                {sidebarOpen ? (
                  <PanelLeftClose className="size-3.5" />
                ) : (
                  <PanelLeftOpen className="size-3.5" />
                )}
              </button>
            }
          >
            <SearchBox onAdd={startAdd} existing={existing} />
            {pending ? (
              <div
                className={`mt-3 rounded-xl border bg-white/[0.03] p-3 ${
                  pending.fromSearch ? "border-accent/40" : "border-line"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm font-semibold text-accent">
                    {pending.info.symbol}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-xs text-mut">
                    {pending.info.name}
                  </span>
                  <button
                    onClick={() => setPending(null)}
                    className="text-mut transition-colors hover:text-loss"
                    aria-label="Cancel"
                  >
                    <X className="size-3.5" />
                  </button>
                </div>
                {pending.fromSearch && (
                  <p className="mt-2 text-xs leading-relaxed text-accent">
                    How many shares of {pending.info.symbol} do you own? Avg cost is
                    optional.
                  </p>
                )}
                {existing.has(pending.info.symbol) && (
                  <p className="mt-2 text-[11px] leading-relaxed text-amber-200/80">
                    You already hold {pending.info.symbol} — adding merges into the
                    existing position at weighted-average cost.
                  </p>
                )}
                <div className="mt-3 grid grid-cols-2 gap-2.5">
                  <label className="block">
                    <span className={LABEL}>Shares</span>
                    <input
                      type="number"
                      min={0}
                      step="any"
                      autoFocus={pending.fromSearch}
                      placeholder={pending.fromSearch ? "required" : undefined}
                      value={pending.shares}
                      onChange={(e) => setPending({ ...pending, shares: e.target.value, error: null })}
                      className={`mt-1 ${INPUT}`}
                    />
                  </label>
                  <label className="block">
                    <span className={LABEL}>Buy price ($)</span>
                    <input
                      type="number"
                      min={0}
                      step="any"
                      placeholder="market"
                      value={pending.buyPrice}
                      onChange={(e) => setPending({ ...pending, buyPrice: e.target.value, error: null })}
                      className={`mt-1 ${INPUT}`}
                    />
                  </label>
                </div>
                <p className="mt-1.5 text-[11px] leading-relaxed text-mut/80">
                  Leave the price empty to use the latest market price as your cost basis.
                </p>
                {pending.error && (
                  <p className="mt-2 text-xs leading-relaxed text-loss">{pending.error}</p>
                )}
                <button
                  onClick={() => void confirmAdd()}
                  disabled={busy != null}
                  className={`${BTN_PRIMARY} mt-3 w-full justify-center`}
                >
                  {busy === "add" ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <Plus className="size-3.5" />
                  )}
                  Add holding
                </button>
              </div>
            ) : (
              <p className="mt-2.5 text-[11px] leading-relaxed text-mut/80">
                Search the full NASDAQ / NYSE universe, then enter shares and cost basis.
              </p>
            )}
          </Section>

          <Section title="Cash balance">
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={0}
                step="any"
                value={cashText}
                onChange={(e) => setCashText(e.target.value)}
                onBlur={() => void commitCash()}
                onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
                className={INPUT}
                aria-label="Cash balance in dollars"
              />
              {busy === "cash" && <Loader2 className="size-3.5 shrink-0 animate-spin text-accent" />}
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-mut/80">
              Counted in total value and allocation, excluded from position weights.
            </p>
          </Section>

          <Section title="Import / export">
            <input
              ref={fileRef}
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void onCsvPicked(f);
                e.target.value = "";
              }}
            />
            <div className="space-y-2">
              <button
                onClick={() => fileRef.current?.click()}
                disabled={busy != null}
                className={`${BTN_GHOST} w-full justify-center`}
              >
                {busy === "parse" ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <FileUp className="size-3.5" />
                )}
                Upload portfolio CSV
              </button>
              <button
                onClick={() => downloadText("my_portfolio.csv", holdingsCsv(saved))}
                disabled={saved.length === 0}
                className={`${BTN_GHOST} w-full justify-center`}
              >
                <Download className="size-3.5" />
                Download current (CSV)
              </button>
              <button
                onClick={() =>
                  downloadText(
                    "portfolio_template.csv",
                    "symbol,shares,buy_price\nAAPL,10,250\nSPY,5,600\n"
                  )
                }
                className={`${BTN_GHOST} w-full justify-center`}
              >
                <Download className="size-3.5" />
                Download CSV template
              </button>
            </div>
            <p className="mt-2.5 text-[11px] leading-relaxed text-mut/80">
              Columns: <span className="font-mono">symbol, shares</span> and optionally{" "}
              <span className="font-mono">buy_price, name</span>.
            </p>

            {importPrev && (
              <div className="mt-3 rounded-xl border border-amber-300/25 bg-amber-300/[0.06] p-3">
                <div className="text-xs text-amber-200/90">
                  {importPrev.holdings.length} holdings found in file.
                </div>
                {importPrev.problems.map((p) => (
                  <div key={p} className="mt-1.5 text-[11px] leading-relaxed text-amber-200/70">
                    {p}
                  </div>
                ))}
                <div className="mt-2.5 flex gap-2">
                  <button
                    onClick={() => void confirmImport()}
                    disabled={busy != null || importPrev.holdings.length === 0}
                    className={`${BTN_PRIMARY} flex-1 justify-center`}
                  >
                    {busy === "import" && <Loader2 className="size-3.5 animate-spin" />}
                    Import (replaces current)
                  </button>
                  <button onClick={() => setImportPrev(null)} className={BTN_GHOST}>
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </Section>

          <button
            onClick={onLoadSampleClick}
            disabled={busy != null}
            className={`${
              confirmSample
                ? "inline-flex items-center gap-2 rounded-xl border border-loss/40 bg-loss/10 px-4 py-2 text-sm font-medium text-loss transition-colors hover:bg-loss/20"
                : BTN_GHOST
            } w-full justify-center`}
          >
            {busy === "sample" ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : confirmSample ? (
              <TriangleAlert className="size-3.5" />
            ) : (
              <PackageOpen className="size-3.5" />
            )}
            {confirmSample ? "Replace current portfolio?" : "Load sample portfolio"}
          </button>
            </motion.aside>
          ) : (
            <motion.button
              key="sidebar-closed"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSidebarOpen(true)}
              className="flex items-center gap-2 rounded-xl border border-line bg-white/[0.04] px-3 py-2 text-sm text-mut transition-colors hover:border-accent/40 hover:text-accent"
              title="Show sidebar"
            >
              <PanelLeftOpen className="size-3.5" />
              <span className="font-mono text-[10px] uppercase tracking-wider">Panel</span>
            </motion.button>
          )}
        </AnimatePresence>

        {/* ------------------------------ main column ---------------------------- */}
        <div className="min-w-0 space-y-4">
          {problems.map((p) => (
            <Note key={p}>{p}</Note>
          ))}

          {saved.length === 0 && !dirty ? (
            <StateCard
              icon={PackageOpen}
              iconTone="text-accent"
              title="Your portfolio is empty"
              body="Add your first holding with the search box, upload a CSV, or load the sample portfolio — then explore the engine pages."
              actions={
                <button onClick={() => void loadSample()} disabled={busy != null} className={BTN_PRIMARY}>
                  {busy === "sample" ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <PackageOpen className="size-3.5" />
                  )}
                  Load sample portfolio
                </button>
              }
            />
          ) : (
            <>
              {totals && (
                <div className="grid grid-cols-2 gap-2.5 xl:grid-cols-4">
                  <Metric label="Total value" value={fmtMoney(totals.market_value)} sub="incl. cash" />
                  <Metric label="Invested" value={fmtMoney(totals.invested)} />
                  <Metric label="Cash" value={fmtMoney(totals.cash)} />
                  <Metric
                    label="Unrealized P/L"
                    value={fmtMoney(totals.pnl)}
                    tone={signClass(totals.pnl)}
                    sub={`${pnlPct >= 0 ? "+" : ""}${fmtNum(pnlPct, 2)}% on cost`}
                  />
                </div>
              )}

              {unpriced.length > 0 && (
                <Note>
                  No live price for: {unpriced.join(", ")} — showing cost basis instead.
                </Note>
              )}

              {dirty && (
                <div className="flex flex-wrap items-center gap-3 rounded-xl border border-accent/30 bg-accent-dim px-4 py-3">
                  <span className="text-sm text-ink/90">
                    Unsaved changes — values and weights refresh after you save.
                  </span>
                  <div className="ml-auto flex gap-2">
                    <button
                      onClick={() => void saveRows()}
                      disabled={busy != null}
                      className={BTN_PRIMARY}
                    >
                      {busy === "save" && <Loader2 className="size-3.5 animate-spin" />}
                      Save changes
                    </button>
                    <button
                      onClick={() => setRows(saved.map(toEdit))}
                      disabled={busy != null}
                      className={BTN_GHOST}
                    >
                      <Undo2 className="size-3.5" />
                      Discard
                    </button>
                  </div>
                </div>
              )}

              <section className="card px-2 py-2">
                <div className="flex items-center justify-between gap-3 px-3 pb-1 pt-3">
                  <div className="flex min-w-0 items-center gap-1.5">
                    <button
                      onClick={() => setCollapsed((c) => !c)}
                      className="shrink-0 rounded-md p-0.5 text-mut transition-colors hover:text-ink/80"
                      title={collapsed ? "Expand holdings table" : "Collapse holdings table"}
                      aria-label={collapsed ? "Expand holdings table" : "Collapse holdings table"}
                    >
                      {collapsed ? (
                        <ChevronDown className="size-4" />
                      ) : (
                        <ChevronUp className="size-4" />
                      )}
                    </button>
                    <h3 className="shrink-0 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                      Holdings
                    </h3>
                    {collapsed && totals && (
                      <span className="min-w-0 truncate font-mono text-[10px] tracking-wider text-mut/70">
                        <span className="mx-1.5 text-mut/30">·</span>
                        {saved.length} position{saved.length !== 1 ? "s" : ""}
                        <span className="mx-1.5 text-mut/30">·</span>
                        Invested{" "}
                        <span className="text-ink/80">{fmtMoney(totals.invested)}</span>
                        <span className="mx-1.5 text-mut/30">·</span>
                        Value{" "}
                        <span className="text-ink/80">{fmtMoney(totals.market_value)}</span>
                        <span className="mx-1.5 text-mut/30">·</span>
                        <span className={signClass(totals.pnl)}>
                          P/L {totals.pnl >= 0 ? "+" : ""}
                          {fmtMoney(totals.pnl)} ({pnlPct >= 0 ? "+" : ""}
                          {fmtNum(pnlPct, 2)}%)
                        </span>
                      </span>
                    )}
                  </div>
                  {!collapsed && rangeView && (
                    <span className="shrink-0 font-mono text-[10px] uppercase tracking-wider text-mut/60">
                      Sharpe · Contribution · Trend over {range}
                    </span>
                  )}
                </div>

                {!collapsed && (
                  <div className="scroll-slim overflow-x-auto">
                    <table className="w-full min-w-[1160px] border-collapse">
                      <thead>
                        <tr className="border-b border-line text-left font-mono text-[10px] uppercase tracking-[0.16em] text-mut">
                          <th className="px-3 py-3 font-medium">Position</th>
                          <th className="px-3 py-3 text-right font-medium">Shares</th>
                          <th className="px-3 py-3 text-right font-medium">Avg cost</th>
                          <th className="px-3 py-3 text-right font-medium">Price</th>
                          <th className="px-3 py-3 text-right font-medium">Value</th>
                          <th className="px-3 py-3 text-right font-medium">P/L</th>
                          <th className="px-3 py-3 text-right font-medium">P/L %</th>
                          <th className="px-3 py-3 text-right font-medium">Weight</th>
                          <th className="px-3 py-3 text-right font-medium">Sharpe</th>
                          <th className="px-3 py-3 text-right font-medium">Contribution</th>
                          <th className="px-3 py-3 text-right font-medium">Trend</th>
                          <th className="px-3 py-3" />
                        </tr>
                      </thead>
                      <tbody>
                        {displayRows.map((r) => {
                          const v = viewBy.get(r.symbol);
                          const a = analysisBy.get(analysisKey(r.symbol));
                          return (
                            <tr
                              key={r.symbol}
                              className="border-b border-white/[0.04] transition-colors last:border-0 hover:bg-white/[0.02]"
                            >
                              <td className="px-3 py-2.5">
                                <div className="flex items-center gap-2.5">
                                  <span className="font-mono text-sm font-semibold text-accent">
                                    {r.symbol}
                                  </span>
                                  <span className="max-w-[180px] truncate text-xs text-mut">
                                    {r.name}
                                  </span>
                                </div>
                              </td>
                              <td className="px-3 py-2.5 text-right">
                                <input
                                  type="number"
                                  min={0}
                                  step="any"
                                  value={r.shares}
                                  onChange={(e) => editRow(r.symbol, "shares", e.target.value)}
                                  className={`${INPUT} w-24 text-right`}
                                  aria-label={`${r.symbol} shares`}
                                />
                              </td>
                              <td className="px-3 py-2.5 text-right">
                                <input
                                  type="number"
                                  min={0}
                                  step="any"
                                  value={r.buy_price}
                                  onChange={(e) => editRow(r.symbol, "buy_price", e.target.value)}
                                  className={`${INPUT} w-24 text-right`}
                                  aria-label={`${r.symbol} average cost`}
                                />
                              </td>
                              <td className="px-3 py-2.5 text-right font-mono text-xs tabular text-mut">
                                {v?.has_price && v.current_price != null
                                  ? fmtMoney(v.current_price)
                                  : "—"}
                              </td>
                              <td className="px-3 py-2.5 text-right font-mono text-xs tabular text-ink/90">
                                {v ? fmtMoney(v.market_value) : "—"}
                              </td>
                              <td
                                className={`px-3 py-2.5 text-right font-mono text-xs font-medium tabular ${signClass(
                                  v?.pnl ?? null
                                )}`}
                              >
                                {v ? fmtMoney(v.pnl) : "—"}
                              </td>
                              <td
                                className={`px-3 py-2.5 text-right font-mono text-xs tabular ${signClass(
                                  v?.pnl_pct ?? null
                                )}`}
                              >
                                {v?.pnl_pct == null
                                  ? "—"
                                  : `${v.pnl_pct >= 0 ? "+" : ""}${fmtNum(v.pnl_pct, 2)}%`}
                              </td>
                              <td className="px-3 py-2.5">
                                <div className="flex items-center justify-end gap-2">
                                  <ThinBar fraction={(v?.weight_pct ?? 0) / 100} className="w-12" />
                                  <span className="w-11 text-right font-mono text-xs tabular text-ink/85">
                                    {v ? `${fmtNum(v.weight_pct, 1)}%` : "—"}
                                  </span>
                                </div>
                              </td>
                              <td className="px-3 py-2.5 text-right font-mono text-xs tabular text-ink/85">
                                {a ? fmtNum(a.stats.sharpe, 2) : "—"}
                              </td>
                              <td className="px-3 py-2.5">
                                {a ? (
                                  <div className="flex items-center justify-end gap-2">
                                    <div className="h-[3px] w-12 overflow-hidden rounded-full bg-white/[0.07]">
                                      <div
                                        className={`h-full rounded-full ${
                                          a.contribution >= 0 ? "bg-gain/80" : "bg-loss/80"
                                        }`}
                                        style={{
                                          width: `${Math.min(
                                            (Math.abs(a.contribution) / maxContrib) * 100,
                                            100
                                          )}%`,
                                        }}
                                      />
                                    </div>
                                    <span
                                      className={`font-mono text-xs tabular ${signClass(a.contribution)}`}
                                    >
                                      {fmtPct(a.contribution)}
                                    </span>
                                  </div>
                                ) : (
                                  <div className="text-right font-mono text-xs text-mut">—</div>
                                )}
                              </td>
                              <td className="px-3 py-2.5">
                                {a ? (
                                  <div className="flex justify-end">
                                    <Sparkline
                                      values={a.values}
                                      positive={a.stats.totalReturn >= 0}
                                    />
                                  </div>
                                ) : (
                                  <div className="text-right font-mono text-xs text-mut">—</div>
                                )}
                              </td>
                              <td className="px-3 py-2.5 text-right">
                                <button
                                  onClick={() => removeRow(r.symbol)}
                                  title={`Remove ${r.symbol}`}
                                  className="rounded-lg border border-transparent p-1.5 text-mut transition-colors hover:border-loss/40 hover:text-loss"
                                >
                                  <Trash2 className="size-3.5" />
                                </button>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>

              {rangeView && analysis && (
                <section className="card px-2 py-2">
                  <div className="flex items-center justify-between gap-3 px-3 pb-1 pt-3">
                    <div className="flex min-w-0 items-center gap-1.5">
                      <button
                        onClick={() => setCollapsedAlloc((c) => !c)}
                        className="shrink-0 rounded-md p-0.5 text-mut transition-colors hover:text-ink/80"
                        title={collapsedAlloc ? "Expand allocation" : "Collapse allocation"}
                        aria-label={collapsedAlloc ? "Expand allocation" : "Collapse allocation"}
                      >
                        {collapsedAlloc ? (
                          <ChevronDown className="size-4" />
                        ) : (
                          <ChevronUp className="size-4" />
                        )}
                      </button>
                      <h3 className="shrink-0 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                        Allocation
                      </h3>
                      {collapsedAlloc && (
                        <span className="min-w-0 truncate font-mono text-[10px] tracking-wider text-mut/70">
                          <span className="mx-1.5 text-mut/30">·</span>
                          Sector exposure
                          {analysis.sectors
                            .sort((a, b) => b.weight - a.weight)
                            .slice(0, 5)
                            .map((s) => (
                              <span key={s.sector}>
                                <span className="mx-1.5 text-mut/30">·</span>
                                {s.sector}{" "}
                                <span className="text-ink/80">{fmtNum(s.weight * 100, 1)}%</span>
                              </span>
                            ))}
                          {analysis.sectors.length > 5 && (
                            <span className="text-mut/50">
                              {" "}
                              +{analysis.sectors.length - 5} more
                            </span>
                          )}
                        </span>
                      )}
                    </div>
                  </div>
                  {!collapsedAlloc && (
                    <div className="p-5">
                      <AllocationDonut
                        items={rangeView.holdings.map((h) => ({
                          symbol: h.symbol,
                          name: h.name,
                          sector: h.sector,
                          weight: h.weight,
                        }))}
                      />
                      <SectorBars sectors={analysis.sectors} />
                    </div>
                  )}
                </section>
              )}

              {/* ---- analytics migrated from the retired "/" analyzer ---- */}
              {analysisError && (
                <Note>
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      Portfolio analytics unavailable — {analysisError}
                      {analysis ? " Showing the last successful analysis below." : ""}
                    </div>
                    <button
                      onClick={() => setAnalysisAttempt((n) => n + 1)}
                      disabled={analyzing}
                      className="shrink-0 rounded-lg border border-line bg-white/[0.05] px-3 py-1 font-mono text-[10px] uppercase tracking-wider text-accent transition-colors hover:border-accent/40 hover:text-accent disabled:opacity-50"
                    >
                      {analyzing ? (
                        <>
                          <Loader2 className="mr-1 inline size-3 animate-spin" />
                          Retrying…
                        </>
                      ) : (
                        <>
                          <RefreshCw className="mr-1 inline size-3" />
                          Retry
                        </>
                      )}
                    </button>
                  </div>
                </Note>
              )}
              {excluded.length > 0 && (
                <Note>
                  Excluded from the analytics below (no usable price history):{" "}
                  {excluded.join(", ")}.
                </Note>
              )}

              {!rangeView && analyzing && (
                <div className="space-y-4">
                  <div className="skeleton h-[430px] rounded-2xl" />
                  <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 xl:grid-cols-5">
                    {Array.from({ length: 10 }).map((_, i) => (
                      <div key={i} className="skeleton h-[92px] rounded-2xl" />
                    ))}
                  </div>
                  <div className="grid gap-4 xl:grid-cols-2">
                    <div className="skeleton h-[300px] rounded-2xl" />
                    <div className="skeleton h-[300px] rounded-2xl" />
                  </div>
                </div>
              )}

              {rangeView && analysis && (
                <>
                  <section className="card relative overflow-hidden p-5 pb-7">
                    <div className="flex flex-wrap items-end justify-between gap-3">
                      <div>
                        <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                          <Waypoints className="size-3.5 text-accent" />
                          Constant-mix portfolio · indexed to 100
                        </div>
                        <div className="mt-1.5 flex items-baseline gap-3">
                          <span
                            className={`text-4xl font-bold tracking-tight tabular ${
                              endReturn >= 0 ? "text-gain" : "text-loss"
                            }`}
                          >
                            {fmtPct(endReturn)}
                          </span>
                          <span className="font-mono text-xs text-mut">
                            {fmtDate(rangeView.dates[0], true)} →{" "}
                            {fmtDate(rangeView.dates[rangeView.dates.length - 1], true)}
                          </span>
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center gap-2">
                        {rangeView.benchmarks.map((b) => {
                          const on = b.symbol === "SPY" ? showSpy : showQqq;
                          const toggle =
                            b.symbol === "SPY"
                              ? () => setShowSpy((x) => !x)
                              : () => setShowQqq((x) => !x);
                          return (
                            <button
                              key={b.symbol}
                              onClick={toggle}
                              className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider transition-all ${
                                on
                                  ? "border-line bg-white/[0.05] text-ink/85"
                                  : "border-line/50 text-mut/45"
                              }`}
                            >
                              <span
                                className="size-1.5 rounded-full"
                                style={{ backgroundColor: b.color, opacity: on ? 1 : 0.35 }}
                              />
                              {b.symbol}
                              {on ? <Eye className="size-3" /> : <EyeOff className="size-3" />}
                            </button>
                          );
                        })}
                        <div className="ml-1 flex rounded-xl border border-line bg-white/[0.03] p-0.5">
                          {RANGES.map((r) => (
                            <button
                              key={r.id}
                              onClick={() => setRange(r.id)}
                              className={`rounded-[10px] px-2.5 py-1 font-mono text-[10px] tracking-wider transition-all ${
                                range === r.id
                                  ? "bg-accent font-semibold text-[#0a0f07]"
                                  : "text-mut hover:text-ink"
                              }`}
                            >
                              {r.id}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="mt-4">
                      <PerformanceChart
                        dates={rangeView.dates}
                        series={chartSeries}
                        height={380}
                      />
                    </div>

                    {analysis.range.truncatedNote && (
                      <div className="mt-5 flex items-center gap-2 text-[11px] text-amber-200/80">
                        <TriangleAlert className="size-3.5 shrink-0" />
                        {analysis.range.truncatedNote}
                      </div>
                    )}
                    {analysis.source !== "live" && (
                      <div className="mt-2 flex items-center gap-2 text-[11px] text-amber-200/80">
                        <TriangleAlert className="size-3.5 shrink-0" />
                        Live market feed partially unreachable — some series are
                        deterministic simulations so the analysis remains explorable.
                      </div>
                    )}

                    <AnimatePresence>
                      {analyzing && (
                        <motion.div
                          initial={{ opacity: 0, y: -6 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0 }}
                          className="absolute right-4 top-4 flex items-center gap-1.5 rounded-full border border-accent/30 bg-bg0/90 px-3 py-1 font-mono text-[10px] uppercase tracking-wider text-accent backdrop-blur"
                        >
                          <Loader2 className="size-3 animate-spin" />
                          Updating
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </section>

                  <StatsRow metrics={rangeView.metrics} spy={spyStats} />

                  <section className="card p-5">
                    <h3 className="mb-4 font-mono text-[10px] uppercase tracking-[0.22em] text-mut">
                      Monthly returns · portfolio
                    </h3>
                    <MonthlyHeatmap monthly={rangeView.monthly} />
                  </section>
                </>
              )}
            </>
          )}
        </div>
      </motion.div>
    </PageShell>
  );
}
