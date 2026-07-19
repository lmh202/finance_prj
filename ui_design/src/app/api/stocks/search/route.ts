/**
 * GET /api/stocks/search?q=apple
 * Ranked instrument search across the seeded catalog, live-augmented.
 */
import { NextRequest, NextResponse } from "next/server";
import { searchStocks } from "@/lib/stocks";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const q = req.nextUrl.searchParams.get("q") ?? "";
  if (q.trim().length === 0 || q.trim().length > 40) {
    return NextResponse.json({ results: [] });
  }
  const results = await searchStocks(q);
  return NextResponse.json(
    { results },
    { headers: { "Cache-Control": "public, max-age=300" } },
  );
}
