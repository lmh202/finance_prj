"use client";

/**
 * The portfolio lives in this browser, not on the backend.
 *
 * AURORA has no accounts, so there is nothing to key server-side storage on:
 * every visitor to a deployed instance would otherwise read and edit the same
 * data/portfolio.csv. Holdings + cash live in localStorage instead, and every
 * engine request ships them in its body. That keeps the backend stateless
 * (no disk, no session, no database) and means we never hold anyone's
 * positions.
 *
 * The trade-off is deliberate and must stay visible in the UI: clearing site
 * data or switching browser/device loses the portfolio. CSV export on
 * /portfolio is the backup path.
 */

import { useCallback, useEffect, useState } from "react";
import type { BackendHolding } from "./types";

const KEY = "aurora_portfolio";

export interface StoredPortfolio {
  holdings: BackendHolding[];
  cash: number;
}

export const EMPTY_PORTFOLIO_STATE: StoredPortfolio = { holdings: [], cash: 0 };

/** Reject anything that isn't a usable position so one corrupt entry written
 *  by an older build can't break every engine page. */
function coerce(raw: unknown): StoredPortfolio {
  if (!raw || typeof raw !== "object") return EMPTY_PORTFOLIO_STATE;
  const obj = raw as { holdings?: unknown; cash?: unknown };
  const holdings = Array.isArray(obj.holdings)
    ? obj.holdings.flatMap((h): BackendHolding[] => {
        if (!h || typeof h !== "object") return [];
        const r = h as Record<string, unknown>;
        const symbol = typeof r.symbol === "string" ? r.symbol.trim().toUpperCase() : "";
        const shares = Number(r.shares);
        if (!symbol || !Number.isFinite(shares) || shares <= 0) return [];
        const buyPrice = Number(r.buy_price);
        return [
          {
            symbol,
            name: typeof r.name === "string" ? r.name : "",
            shares,
            buy_price: Number.isFinite(buyPrice) ? buyPrice : 0,
          },
        ];
      })
    : [];
  const cash = Number(obj.cash);
  return { holdings, cash: Number.isFinite(cash) ? cash : 0 };
}

/** Safe on the server (returns empty) and in private mode (localStorage throws). */
export function readPortfolio(): StoredPortfolio {
  if (typeof window === "undefined") return EMPTY_PORTFOLIO_STATE;
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? coerce(JSON.parse(raw)) : EMPTY_PORTFOLIO_STATE;
  } catch {
    return EMPTY_PORTFOLIO_STATE;
  }
}

/* Same-tab subscribers. The `storage` event only fires in OTHER tabs, so
 * without this a page that both edits and reads the portfolio would not see
 * its own writes. */
type Listener = (p: StoredPortfolio) => void;
const listeners = new Set<Listener>();

export function writePortfolio(next: StoredPortfolio): StoredPortfolio {
  const clean = coerce(next);
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(clean));
    } catch {
      /* private mode / quota — the in-memory state below still works for
         this session, it just won't survive a reload. */
    }
  }
  listeners.forEach((fn) => fn(clean));
  return clean;
}

/**
 * The portfolio for the current browser.
 *
 * `ready` is false until the first client-side read completes — engine pages
 * must not fire a request before then or a visitor with a saved portfolio
 * would flash the "no holdings yet" state on every navigation. Reading in a
 * useState initialiser instead would desync server and client HTML.
 */
export function usePortfolio() {
  const [portfolio, setState] = useState<StoredPortfolio>(EMPTY_PORTFOLIO_STATE);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    // Deferred so no setState runs synchronously in the effect body, matching
    // the pattern the engine pages already use.
    const timer = setTimeout(() => {
      setState(readPortfolio());
      setReady(true);
    }, 0);

    const onLocal: Listener = (p) => setState(p);
    listeners.add(onLocal);
    const onStorage = (e: StorageEvent) => {
      if (e.key === KEY || e.key === null) setState(readPortfolio());
    };
    window.addEventListener("storage", onStorage);

    return () => {
      clearTimeout(timer);
      listeners.delete(onLocal);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  const save = useCallback((next: StoredPortfolio) => writePortfolio(next), []);

  const setHoldings = useCallback(
    (holdings: BackendHolding[]) =>
      writePortfolio({ holdings, cash: readPortfolio().cash }),
    []
  );

  const setCash = useCallback(
    (cash: number) => writePortfolio({ holdings: readPortfolio().holdings, cash }),
    []
  );

  return { portfolio, ready, save, setHoldings, setCash };
}
