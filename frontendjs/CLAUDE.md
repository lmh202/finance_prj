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

This app has no database anymore — there is nothing to seed or migrate.
`drizzle.config.json` is a leftover from the removed database layer; there's
no `DATABASE_URL` and nothing runs migrations.

The Python FastAPI backend at the repo root must be running for search or
analysis to work (`uvicorn main:app --app-dir ../backend --reload --port
8000`, or `..\scripts\dev.ps1` from the repo root) — see the top-level
`CLAUDE.md`.

## Architecture

**Aurora** is a portfolio analytics platform — a Next.js 16 App Router app
with two kinds of pages:

- **`/` (Home)** — a clean entry point: a hero search box over the symbol
  universe. Picking a result routes to `/portfolio?add=SYMBOL&name=…`,
  where the builder prompts for the share count (avg cost optional). The
  old localStorage analyzer that used to live here is gone — its analytics
  were merged into `/portfolio`.
- **Engine pages** — Next.js ports of the Streamlit pages in the repo-root
  `frontend/`, all reading the **backend-persisted** AURORA portfolio
  (`data/portfolio.csv` + cash):
  - `/portfolio` — builder for the saved portfolio (add/edit/delete
    holdings, cash, CSV import/export, sample load with replace-confirm)
    plus the analytics absorbed from the old analyzer: constant-mix
    performance chart vs SPY/QQQ with 6M–5Y ranges, stat cards, allocation
    donut + sector bars, monthly-returns heatmap (all fed by
    `POST /analysis/explore` in `"shares"` mode), and per-position
    Sharpe / Contribution / Trend columns in the holdings table
  - `/health` — Engine 1 health report (score gauge, metrics, correlation)
  - `/strategy` — Engine 2 market regime + daily asset ranking
  - `/news` — Engine 3 essential news (planned-feeds empty state while the
    engine is a stub)
  - `/react` — Engine 4 daily recommendation + event reaction risk
  - `/performance` — Dev 2's backtest curves rendered with PerformanceChart
  - `/risk` — per-stock + portfolio downside risk (HAR volatility forecast +
    Filtered Historical Simulation) via `/risk/{estimates,portfolio}`; a
    5d/20d horizon toggle, a whole-portfolio VaR/ES card and a per-holding
    danger table — distinct from Engine 4's reaction-risk proxy

Engine pages share scaffolding: `useEngine` (src/lib/use-engine.ts) fetches
with abort + classifies errors; `EngineShell` (src/components/EngineShell.tsx)
renders chrome, loading skeletons and the four expected non-ready states —
backend down, `empty_portfolio` (409 → onboarding card with sample-load),
`no_history` (502 → retry) and `no_model` (503 → the risk engine's offline
artifact hasn't been built). Small primitives (Section, Metric, Chip,
ThinBar, ArcGauge, Note, StateCard) live in EngineShell.tsx too.

This app used to be fully self-contained (its own PostgreSQL database,
Drizzle schema, and Next.js API routes doing price fetching and analytics
locally). **That has been removed.** It is now a pure client: every data
operation calls out to the AURORA Python FastAPI backend that lives at the
repo root (`../backend/`). There are no `src/app/api/*` routes and no
`src/db/` anymore.

### Tech stack

Next.js 16 (React 19) · TypeScript 5.9 strict · Tailwind CSS 4 · Framer Motion · lucide-react icons — no database, no ORM.

### Key architectural decisions

- **The backend is the Python FastAPI service in `../backend/`**, not this
  app. `src/lib/api-client.ts` is the only place that talks to it, via
  `NEXT_PUBLIC_BACKEND_URL` (set in the tracked `.env.local` to
  `http://localhost:8000` for local dev). Endpoints used:
  - `GET /market/search?q=` + `GET /market/prices` — symbol universe search
    and latest closes
  - `POST /analysis/explore` — /portfolio's full analytics engine
    (calendar alignment, constant-mix construction, risk stats), backed by
    `backend/src/analysis/engine.py`; called in `"shares"` mode with the
    saved holdings
  - `/portfolio*` — saved-portfolio CRUD, cash, CSV parse, sample, view
  - `/health/report`, `/strategy/{regime,signals,backtest}`,
    `/news/{essential,feeds}`, `/recommendation/{daily,events,react}` —
    the four engine routers consumed by the engine pages
  - `/risk/{estimates,portfolio}` — the risk engine (HAR volatility +
    Filtered Historical Simulation), consumed by `/risk`
  Expected conditions arrive as marker details (`empty_portfolio` 409,
  `no_history` 502, `no_model` 503) and are thrown as `ApiMarkerError`; an
  unreachable backend throws `BackendDownError`.
  Backend CORS (`backend/main.py`) allow-lists `http://localhost:3000`
  specifically so this app can call it directly from the browser.
- **No auth. One portfolio store** — every page reads/writes the backend's
  saved portfolio (`data/portfolio.csv`), shared with the Streamlit
  frontend. The analyzer's old localStorage portfolio (`aurora_portfolio`
  key) no longer exists. There is no per-client UUID.
- **Constant-mix portfolio** — the analysis engine builds a daily-rebalanced
  portfolio (assumes positions are reset to target weights each day). No
  transaction-cost modeling.
- **Calendar alignment** happens server-side now, in
  `backend/src/analysis/engine.py`'s `align_series()`: it unions the
  calendars of all holdings, forward-fills missing bars, and truncates to
  the latest-starting / earliest-ending instrument so every series covers
  the full window.
- **Context-prop drilling, not a state library** — `page.tsx` owns all
  state and passes it down. No Redux/Zustand.

### Request flow

1. User types a ticker on `/` (or in /portfolio's "Add a holding" box) →
   `searchStocks()` in `src/lib/api-client.ts` → `GET /market/search?q=`
   on the Python backend.
2. On `/`, picking a result routes to `/portfolio?add=SYMBOL&name=…`; the
   builder opens its pending-add panel asking for shares (buy price
   optional — falls back to the latest close) and POSTs
   `/portfolio/holdings`.
3. Whenever the saved holdings change, /portfolio calls `analyze()` →
   `POST /analysis/explore` with `{holdings: [{symbol, value: shares}],
   mode: "shares"}` → the backend fetches 5y of prices for all symbols +
   SPY/QQQ, aligns calendars, builds the constant-mix index, computes
   per-holding metrics, and returns an `AnalyzeResponse`.
4. Client runs `deriveRangeView()` to slice all series to the selected
   range (6M/1Y/2Y/3Y/5Y) and recompute every stat on that window — no
   refetch. The holdings table's Sharpe / Contribution / Trend columns
   follow the selected range.

### Directory map

```
src/
├── app/
│   ├── layout.tsx              # Root layout (fonts, metadata)
│   ├── page.tsx                # Home — hero search, routes picks to /portfolio?add=
│   ├── portfolio/page.tsx      # Saved-portfolio builder + analytics (CRUD + /analysis/explore)
│   ├── health/page.tsx         # Engine 1 — health report
│   ├── strategy/page.tsx       # Engine 2 — regime + asset ranking
│   ├── news/page.tsx           # Engine 3 — essential news
│   ├── react/page.tsx          # Engine 4 — should-I-react flow
│   ├── performance/page.tsx    # Backtest curves (Dev 2 engine, Dev 1 page)
│   ├── risk/page.tsx           # Risk engine — HAR + FHS downside risk
│   └── globals.css             # Tailwind theme tokens + utility classes
├── components/
│   ├── chrome.tsx              # Header (real nav, active route), Footer, Hero
│   ├── EngineShell.tsx         # Engine-page shell + status states + UI primitives
│   ├── SearchBox.tsx           # Autocomplete search with keyboard nav
│   ├── PerformanceChart.tsx    # SVG area chart with crosshair tooltip
│   ├── StatsRow.tsx            # Animated metric cards (Sharpe, beta, etc.)
│   ├── AllocationDonut.tsx     # Donut chart + sector breakdown bars
│   ├── Sparkline.tsx           # Tiny trend line for table rows
│   └── MonthlyHeatmap.tsx      # Calendar heatmap of monthly returns
└── lib/
    ├── types.ts                # Shared types incl. engine API types + RANGES
    ├── format.ts                # Client formatting (fmtPct, fmtPrice, fmtDate, etc.)
    ├── api-client.ts            # Typed fetch client over the FastAPI backend (all endpoints)
    ├── use-engine.ts           # Fetch hook: abort, reload, marker/error classification
    ├── metrics.ts              # Legacy quant toolkit; superseded by backend/src/analysis/engine.py
    └── view.ts                  # Client-side range slicing & metric recomputation
```

### Important conventions

- **All path aliases** are `@/*` → `./src/*`.
- **`NEXT_PUBLIC_BACKEND_URL`** must be set and the Python backend must be
  running for search or analysis to work — there's no local fallback.
- **Range views are pure client-side derivation** — `deriveRangeView()` can
  be called on any `AnalyzeResponse` to window it differently without a
  network round-trip.
