/**
 * GET  /api/portfolio?client=xxx  — load a visitor's saved portfolio
 * PUT  /api/portfolio             — upsert { client, mode, holdings[] }
 */
import { NextRequest, NextResponse } from "next/server";
import { eq } from "drizzle-orm";
import { db } from "@/db";
import { portfolios } from "@/db/schema";
import { normalizeSymbol } from "@/lib/prices";
import type { HoldingInput, InputMode } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const CLIENT_RE = /^[A-Za-z0-9-]{6,64}$/;

function parseHoldings(raw: unknown): HoldingInput[] {
  if (!Array.isArray(raw)) return [];
  const out: HoldingInput[] = [];
  const seen = new Set<string>();
  for (const h of raw) {
    if (!h || typeof (h as HoldingInput).symbol !== "string") continue;
    const value = (h as HoldingInput).value;
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) continue;
    const symbol = normalizeSymbol((h as HoldingInput).symbol);
    if (!symbol || seen.has(symbol)) continue;
    seen.add(symbol);
    out.push({ symbol, value });
    if (out.length >= 40) break;
  }
  return out;
}

export async function GET(req: NextRequest) {
  const client = req.nextUrl.searchParams.get("client") ?? "";
  if (!CLIENT_RE.test(client)) {
    return NextResponse.json({ found: false, holdings: [], mode: "weight" });
  }
  try {
    const rows = await db
      .select()
      .from(portfolios)
      .where(eq(portfolios.clientId, client))
      .limit(1);
    if (rows.length === 0) {
      return NextResponse.json({ found: false, holdings: [], mode: "weight" });
    }
    const row = rows[0];
    let holdings: HoldingInput[] = [];
    try {
      holdings = parseHoldings(JSON.parse(row.holdings));
    } catch {
      holdings = [];
    }
    return NextResponse.json({
      found: true,
      name: row.name,
      mode: row.mode === "shares" ? "shares" : "weight",
      holdings,
    });
  } catch {
    return NextResponse.json({ found: false, holdings: [], mode: "weight" });
  }
}

export async function PUT(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false }, { status: 400 });
  }
  const client = (body as { client?: string })?.client ?? "";
  if (!CLIENT_RE.test(client)) {
    return NextResponse.json({ ok: false, error: "invalid client id" }, { status: 400 });
  }
  const mode: InputMode = (body as { mode?: string })?.mode === "shares" ? "shares" : "weight";
  const holdings = parseHoldings((body as { holdings?: unknown })?.holdings);
  const name =
    typeof (body as { name?: string })?.name === "string"
      ? (body as { name: string }).name.slice(0, 60)
      : "My Portfolio";

  try {
    await db
      .insert(portfolios)
      .values({
        clientId: client,
        name,
        mode,
        holdings: JSON.stringify(holdings),
        updatedAt: new Date(),
      })
      .onConflictDoUpdate({
        target: portfolios.clientId,
        set: {
          mode,
          name,
          holdings: JSON.stringify(holdings),
          updatedAt: new Date(),
        },
      });
    return NextResponse.json({ ok: true, saved: holdings.length });
  } catch {
    return NextResponse.json({ ok: false, error: "db error" }, { status: 500 });
  }
}
