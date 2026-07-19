import "dotenv/config";
import { db } from "./index";
import { stocks } from "./schema";
import { STOCK_UNIVERSE } from "./stock-universe";

async function main() {
  const seen = new Set<string>();
  const rows = STOCK_UNIVERSE.filter(([symbol]) => {
    if (seen.has(symbol)) return false;
    seen.add(symbol);
    return true;
  }).map(([symbol, name, exchange, sector], i) => ({
    symbol,
    name,
    exchange,
    sector,
    quoteType: sector === "ETF" ? "ETF" : "EQUITY",
    popularity: STOCK_UNIVERSE.length - i,
  }));

  for (let i = 0; i < rows.length; i += 100) {
    await db
      .insert(stocks)
      .values(rows.slice(i, i + 100))
      .onConflictDoNothing();
  }
  console.log(`Seeded ${rows.length} instruments into stocks table.`);
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
