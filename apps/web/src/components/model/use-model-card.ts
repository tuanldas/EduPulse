"use client";

import { useCallback, useEffect, useState } from "react";

import { api, call, requestSignal, toFailure } from "@/lib/api/client";
import { isAbortError, type ApiFailure } from "@/lib/api/errors";
import type { ModelCard } from "@/lib/api/types";

export type ModelCardState =
  | { status: "loading" }
  | { status: "ready"; card: ModelCard }
  | { status: "error"; failure: ApiFailure };

/** Nạp model card của bundle đang phục vụ (`/api/model-card`); mọi số trên trang 4 lấy từ đây. */
export function useModelCard(): { state: ModelCardState; retry: () => void } {
  const [state, setState] = useState<ModelCardState>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const abort = new AbortController();
    call(api.GET("/api/model-card", { signal: requestSignal(abort.signal) }))
      .then((card) => {
        if (!abort.signal.aborted) setState({ status: "ready", card });
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) return;
        setState({ status: "error", failure: toFailure(error) });
      });
    return () => abort.abort();
  }, [attempt]);

  const retry = useCallback(() => {
    setState({ status: "loading" });
    setAttempt((value) => value + 1);
  }, []);

  return { state, retry };
}
