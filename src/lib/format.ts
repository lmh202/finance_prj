/** Client-safe formatting helpers. */

export function fmtPct(
  x: number | null | undefined,
  digits = 1,
  sign = true,
): string {
  if (x == null || !Number.isFinite(x)) return "—";
  const v = x * 100;
  const s = sign && v > 0 ? "+" : "";
  return `${s}${v.toFixed(digits)}%`;
}

export function fmtNum(x: number | null | undefined, digits = 2): string {
  if (x == null || !Number.isFinite(x)) return "—";
  return x.toFixed(digits);
}

export function fmtPrice(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return "—";
  if (x >= 1000) return x.toLocaleString("en-US", { maximumFractionDigits: 2 });
  return x.toFixed(2);
}

export function fmtDate(iso: string, withYear = false): string {
  const d = new Date(iso + "T12:00:00Z");
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    ...(withYear ? { year: "2-digit" as const } : {}),
    timeZone: "UTC",
  });
}

export function signClass(x: number | null | undefined): string {
  if (x == null || Math.abs(x) < 1e-9) return "text-mut";
  return x > 0 ? "text-gain" : "text-loss";
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(Math.max(x, lo), hi);
}
