import {
  pgTable,
  text,
  integer,
  real,
  date,
  timestamp,
  serial,
  uniqueIndex,
  index,
} from "drizzle-orm/pg-core";

/**
 * Prism — portfolio intelligence platform.
 *
 * Data model notes:
 * - `stocks`       : searchable tradable universe (seeded w/ popular NASDAQ/NYSE
 *                    names + ETFs, augmented at runtime from live search).
 * - `price_cache`  : persistent daily adjusted-close cache so repeat analysis
 *                    is instant and resilient to upstream outages.
 * - `portfolios`   : anonymous-client persisted portfolio (holdings stored as
 *                    JSON for MVP simplicity).
 *
 * Roadmap (deferred): `news_sources` / `news_items` tables will hold RSS feeds
 * mapped to tickers/sectors for rebalancing signals.
 */

export const stocks = pgTable(
  "stocks",
  {
    symbol: text("symbol").primaryKey(),
    name: text("name").notNull(),
    exchange: text("exchange").notNull().default(""),
    sector: text("sector").notNull().default("Other"),
    quoteType: text("quote_type").notNull().default("EQUITY"),
    popularity: integer("popularity").notNull().default(0),
    createdAt: timestamp("created_at").notNull().defaultNow(),
  },
  (t) => [index("stocks_name_idx").on(t.name)],
);

export const priceCache = pgTable(
  "price_cache",
  {
    symbol: text("symbol").notNull(),
    d: date("d", { mode: "string" }).notNull(),
    close: real("close").notNull(),
  },
  (t) => [uniqueIndex("price_cache_symbol_date_idx").on(t.symbol, t.d)],
);

export const portfolios = pgTable("portfolios", {
  id: serial("id").primaryKey(),
  clientId: text("client_id").notNull().unique(),
  name: text("name").notNull().default("My Portfolio"),
  mode: text("mode").notNull().default("weight"), // 'weight' | 'shares'
  holdings: text("holdings").notNull().default("[]"), // JSON: {symbol, value}[]
  updatedAt: timestamp("updated_at").notNull().defaultNow(),
});
