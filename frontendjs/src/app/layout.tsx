import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Space_Grotesk, IBM_Plex_Mono } from "next/font/google";
import { ThemeProvider, THEME_INIT_SCRIPT } from "@/components/ThemeProvider";
import "./globals.css";

const display = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  weight: ["400", "500", "600", "700"],
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-plex-mono",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Aurora — Portfolio Intelligence",
  description:
    "Reconstruct your real portfolio, refract it into allocations, risk metrics and benchmark comparisons. Live market data, Sharpe, drawdowns, and trend vs SPY & QQQ.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${mono.variable}`} suppressHydrationWarning>
      {/* No manual <head>: the App Router owns it and injects metadata there,
          so rendering one here made the server and client trees disagree —
          that was the hydration failure. The theme bootstrap runs as the
          first thing in <body> instead, which is still before any painted
          content and still sets data-theme ahead of first paint. */}
      <body className="grain bg-bg0 font-display text-ink antialiased">
        <script
          suppressHydrationWarning
          dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }}
        />
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
