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
import { Briefcase, Rss, Scale, type LucideIcon } from "lucide-react";
import { SearchBox } from "@/components/SearchBox";
import { Footer, Header, Hero } from "@/components/chrome";
import { BackendDownError, fetchCash } from "@/lib/api-client";
import type { StockInfo } from "@/lib/types";

const NONE = new Set<string>();

/* The hero's route trio — the three things a visitor most likely came for. */
const QUICK_ACTIONS: { href: string; label: string; icon: LucideIcon }[] = [
  { href: "/portfolio", label: "Open my portfolio", icon: Briefcase },
  { href: "/news", label: "Get real-time news", icon: Rss },
  { href: "/react", label: "Should I react?", icon: Scale },
];

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
            <div className="flex flex-col items-center gap-5">
              <span className="font-mono text-[10px] uppercase tracking-widest text-mut/70">
                Pick a result to add it to your portfolio — you enter the shares next
              </span>

              <div className="quick-actions w-full px-4 py-4 sm:px-5 sm:py-5">
                <div className="mb-4 flex items-center justify-center gap-2.5">
                  <span className="h-px w-6 bg-line" />
                  <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-mut/80">
                    Quick actions
                  </span>
                  <span className="h-px w-6 bg-line" />
                </div>

                <div className="flex flex-wrap items-center justify-center gap-3.5">
                  {QUICK_ACTIONS.map(({ href, label, icon: Icon }) => (
                    <Link key={href} href={href} className="quick-action">
                      <Icon className="qa-icon size-[18px]" />
                      {label}
                    </Link>
                  ))}
                </div>
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
