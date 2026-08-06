"use client";

/**
 * Data-fetching hook for the engine pages.
 *
 * Wraps one backend call (or a Promise.all of several) and translates the
 * expected AURORA conditions — backend down, empty portfolio (409),
 * no price history (502) — into a discriminated status the EngineShell
 * renders. Pass a stable fetcher (wrap it in useCallback); reload() refires
 * it while keeping the previous data on screen.
 *
 * Pass `enabled: false` while a prerequisite is still resolving — engine
 * pages use it to hold off until the portfolio has been read out of
 * localStorage, which would otherwise report "no holdings yet" for a frame
 * on every navigation.
 */

import { useCallback, useEffect, useState } from "react";
import { ApiMarkerError, BackendDownError, type ApiMarker } from "./api-client";

export type EngineStatus =
  | { phase: "loading" }
  | { phase: "ready" }
  | { phase: "marker"; marker: ApiMarker }
  | { phase: "error"; message: string; backendDown: boolean };

export interface Engine<T> {
  data: T | null;
  status: EngineStatus;
  /** true whenever a request is in flight (also during background reloads) */
  loading: boolean;
  reload: () => void;
}

export function useEngine<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  enabled = true
): Engine<T> {
  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<EngineStatus>({ phase: "loading" });
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    const ctrl = new AbortController();
    // Deferred kick-off keeps setState out of the synchronous effect body
    // (react-hooks/set-state-in-effect) and skips strict-mode double fires.
    const timer = setTimeout(() => {
      setLoading(true);
      fetcher(ctrl.signal)
        .then((result) => {
          if (ctrl.signal.aborted) return;
          setData(result);
          setStatus({ phase: "ready" });
          setLoading(false);
        })
        .catch((err: unknown) => {
          if (ctrl.signal.aborted) return;
          if (err instanceof DOMException && err.name === "AbortError") return;
          setData(null);
          if (err instanceof ApiMarkerError) {
            setStatus({ phase: "marker", marker: err.marker });
          } else {
            setStatus({
              phase: "error",
              message: err instanceof Error ? err.message : String(err),
              backendDown: err instanceof BackendDownError,
            });
          }
          setLoading(false);
        });
    }, 0);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [fetcher, nonce, enabled]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  return { data, status, loading, reload };
}
