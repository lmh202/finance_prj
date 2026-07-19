import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Space_Grotesk, IBM_Plex_Mono } from "next/font/google";
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
  title: "Prism — Portfolio Intelligence",
  description:
    "Reconstruct your real portfolio, refract it into allocations, risk metrics and benchmark comparisons. Live market data, Sharpe, drawdowns, and trend vs SPY & QQQ.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${mono.variable}`}>
      <body className="grain bg-bg0 font-display text-ink antialiased">
        {children}
      </body>
    </html>
  );
}
