"use client";

/**
 * Home — the clean entry point into Aurora.
 *
 * The old analyzer (hypothetical localStorage portfolio) was merged into
 * /portfolio, which now owns all analytics. This page only searches the
 * symbol universe; picking a result hands the symbol to the portfolio
 * builder, which asks for the share count (average cost optional).
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Briefcase, Rss, Scale } from "lucide-react";
import { SearchBox } from "@/components/SearchBox";
import { Footer, Header, Hero } from "@/components/chrome";
import { BTN_GHOST } from "@/components/EngineShell";
import { BackendDownError, fetchCash } from "@/lib/api-client";
import type { StockInfo } from "@/lib/types";

const NONE = new Set<string>();

export default function Page() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  /* Probe the backend once so a silently dead search box isn't a mystery. */
  useEffect(() => {
    const ctrl = new AbortController();
    fetchCash(ctrl.signal).catch((err: unknown) => {
      if (!ctrl.signal.aborted && err instanceof BackendDownError) {
        setError(
          "Backend not reachable — start it with scripts\\dev.ps1 (or uvicorn main:app --app-dir backend --port 8000) and reload."
        );
      }
    });
    return () => ctrl.abort();
  }, []);

  function goToPortfolio(info: StockInfo) {
    const q = new URLSearchParams({ add: info.symbol });
    if (info.name && info.name !== info.symbol) q.set("name", info.name);
    router.push(`/portfolio?${q.toString()}`);
  }

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex-1">
        <Hero
          searchSlot={
            <SearchBox
              big
              onAdd={goToPortfolio}
              existing={NONE}
              placeholder="Search any stock or ETF — try “NVDA” or “Apple”…"
            />
          }
          presetsSlot={
            <div className="flex flex-col items-center gap-3">
              <span className="font-mono text-[10px] uppercase tracking-widest text-mut/70">
                Pick a result to add it to your portfolio — you enter the shares next
              </span>
              <div className="flex flex-wrap items-center justify-center gap-2.5">
                <Link href="/portfolio" className={BTN_GHOST}>
                  <Briefcase className="size-3.5" />
                  Open my portfolio
                </Link>
                <Link href="/news" className={BTN_GHOST}>
                  <Rss className="size-3.5" />
                  Get real-time news
                </Link>
                <Link href="/react" className={BTN_GHOST}>
                  <Scale className="size-3.5" />
                  Should I react?
                </Link>
              </div>
            </div>
          }
          error={error}
        />
      </main>
      <Footer />
    </div>
  );
}
