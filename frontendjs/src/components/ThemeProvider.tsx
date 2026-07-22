"use client";

/**
 * Light/dark theme toggle. The actual color swap happens entirely through
 * CSS custom properties scoped to `[data-theme]` on <html> (see
 * globals.css) — this provider only owns *which* theme is active so
 * ThemeToggle (chrome.tsx) can render the right icon and flip it.
 *
 * The inline script in layout.tsx sets the data-theme attribute before
 * React hydrates (reading localStorage, falling back to system
 * preference) so there is no dark->light flash on load. This provider's
 * initial state mirrors whatever that script already applied to the DOM.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type Theme = "light" | "dark";

export const THEME_STORAGE_KEY = "aurora-theme";

/** Must stay in sync with the inline bootstrap script in layout.tsx. */
export const THEME_INIT_SCRIPT = `(function(){try{var s=localStorage.getItem(${JSON.stringify(
  THEME_STORAGE_KEY
)});var t=s==="light"||s==="dark"?s:(window.matchMedia("(prefers-color-scheme: light)").matches?"light":"dark");document.documentElement.setAttribute("data-theme",t);}catch(e){}})();`;

function readDomTheme(): Theme {
  if (typeof document === "undefined") return "dark";
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

const ThemeContext = createContext<{ theme: Theme; toggleTheme: () => void } | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  // Lazy initializer reads what THEME_INIT_SCRIPT already applied, so this
  // never disagrees with the DOM on first client render.
  const [theme, setTheme] = useState<Theme>(readDomTheme);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => {
      const next: Theme = t === "dark" ? "light" : "dark";
      try {
        window.localStorage.setItem(THEME_STORAGE_KEY, next);
      } catch {
        /* localStorage unavailable (private browsing, etc.) — theme just won't persist */
      }
      return next;
    });
  }, []);

  const value = useMemo(() => ({ theme, toggleTheme }), [theme, toggleTheme]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
