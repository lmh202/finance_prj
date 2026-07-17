# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
npm run dev          # Next.js dev server (Turbopack)
npm run build        # Production build
npm run start        # Production server
npm run lint         # ESLint (flat config, Next.js core-web-vitals)
npm run typecheck    # TypeScript check (tsc --noEmit)
```

Seeding the stocks catalog:
```bash
npx tsx src/db/seed.ts
```

## Architecture

**Aurora** is a portfolio analytics platform — a single-page Next.js 16 App Router app that lets visitors compose a stock/ETF portfolio and see 5 years of risk/performance analytics benchmarked against SPY and QQQ.

### Tech stack

Next.js 16 (React 19) · TypeScript 5.9 strict · PostgreSQL + Drizzle ORM · Tailwind CSS 4 · Framer Motion · lucide-react icons

### Key architectural decisions

- **Anonymous client identity** — a UUID is generated and stored in `localStorage` on first visit. No auth.
- **Portfolios stored as JSON** in the `portfolios` table (`holdings` column) for MVP simplicity. Client auto-saves via `PUT /api/portfolio` with a 700ms debounce; hydration reads back from `GET /api/portfolio?client=<id>`.
- **Constant-mix portfolio** — the analysis engine builds a daily-rebalanced portfolio (assumes positions are reset to target weights each day). No transaction-cost modeling.
- **Market data resolution order** (in `src/lib/prices.ts`):
  1. In-process memory cache (5-minute TTL)
  2. PostgreSQL `price_cache` table (shared, persistent, write-through)
  3. Live fetch: Nasdaq API → Yahoo Finance fallback (both via `curl` — Yahoo blocks Node `fetch` 429)
  4. Deterministic synthetic random walk (per-symbol seeded PRNG, flagged `simulated: true`)
- **Concurrent upstream scheduling** — max 3 in-flight requests, 140ms minimum gap between requests, to stay under upstream rate limits.
- **Calendar alignment** — `alignSeries()` unions the calendars of all holdings, forward-fills missing bars, and truncates to the latest-starting / earliest-ending instrument so every series covers the full window.
- **Context-prop drilling, not state library** — page.tsx owns all state and passes it down. No Redux/Zustand.

### Request flow

1. User types ticker → `GET /api/stocks/search?q=` → seeded PostgreSQL catalog (ILIKE, ranked by match quality + popularity), live-augmented from Nasdaq/Yahoo if local results are thin.
2. User adds holdings → `PUT /api/portfolio` persists.
3. Analysis triggers via `POST /api/analyze` with `{mode, holdings}` → fetches prices for all symbols + SPY/QQQ → aligns calendars → builds constant-mix index → computes 14 metrics per holding → returns `AnalyzeResponse`.
4. Client runs `deriveRangeView()` to slice all series to the selected range (6M/1Y/2Y/3Y/5Y) and recompute every stat on that window — no refetch.

### Database schema

Three tables, all managed by Drizzle (`src/db/schema.ts`):
- **`stocks`** — tradable universe (symbol PK, name, exchange, sector, quoteType, popularity). Seeded from `src/db/stock-universe.ts` (~400 US equities + ETFs), grown at runtime via live search.
- **`price_cache`** — daily adjusted closes (symbol, date, close), unique on `(symbol, date)`.
- **`portfolios`** — one row per client (`client_id` unique, `holdings` JSON, `mode`, `name`).

### Directory map

```
src/
├── app/
│   ├── api/
│   │   ├── analyze/route.ts    # POST — the core analytics engine
│   │   ├── portfolio/route.ts  # GET/PUT — portfolio persistence
│   │   ├── stocks/search/route.ts  # GET — instrument search
│   │   └── health/route.ts     # GET — DB connectivity check
│   ├── layout.tsx              # Root layout (fonts, metadata)
│   ├── page.tsx                # Single client page (hero → dashboard)
│   └── globals.css             # Tailwind theme tokens + utility classes
├── components/
│   ├── chrome.tsx              # Header, Footer, Hero, presets, logo
│   ├── SearchBox.tsx           # Autocomplete search with keyboard nav
│   ├── HoldingsPanel.tsx       # Editable position list (weight/shares toggle)
│   ├── PerformanceChart.tsx    # SVG area chart with crosshair tooltip
│   ├── StatsRow.tsx            # Animated metric cards (Sharpe, beta, etc.)
│   ├── AllocationDonut.tsx     # Donut chart + sector breakdown bars
│   ├── HoldingsTable.tsx       # Per-position analytics table with sparklines
│   └── MonthlyHeatmap.tsx      # Calendar heatmap of monthly returns
├── db/
│   ├── schema.ts               # Drizzle schema (stocks, price_cache, portfolios)
│   ├── index.ts                # Drizzle client + pg Pool singleton
│   ├── seed.ts                 # DB seeder
│   └── stock-universe.ts       # Static seed data (~400 instruments)
└── lib/
    ├── types.ts                # Shared TypeScript types + RANGES constant
    ├── format.ts               # Client formatting (fmtPct, fmtPrice, fmtDate, etc.)
    ├── metrics.ts              # Quant toolkit (alignment, indexing, stats)
    ├── prices.ts               # Market data engine (cache, fetch, synthetic)
    ├── stocks.ts               # Search across seeded + live instrument universe
    └── view.ts                 # Client-side range slicing & metric recomputation
```

### Important conventions

- **Ticker normalization** — `normalizeSymbol()` uppercases, replaces `.` with `-`, truncates to 14 chars. Always normalize before DB lookups or cache keys.
- **API route boilerplate** — every route exports `runtime = "nodejs"` and `dynamic = "force-dynamic"` (no static generation for data routes).
- **All path aliases** are `@/*` → `./src/*`.
- **The `curl` binary is required at runtime** for live price fetching. The app falls back to `fetch` if curl is absent, but Yahoo Finance blocks Node fetch with 429.
- **The seed script** expects `DATABASE_URL` in the environment. Production uses the same environment variable.
- **Range views are pure client-side derivation** — `deriveRangeView()` can be called on any `AnalyzeResponse` to window it differently without a network round-trip.
