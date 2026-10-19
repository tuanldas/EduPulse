"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api, call, requestSignal, toFailure } from "@/lib/api/client";
import { isAbortError, type ApiFailure } from "@/lib/api/errors";
import type { AnalyzeResponse, BaselineComparison, ExplainResponse } from "@/lib/api/types";

/** Giải thích (occlusion) tải sau nhãn: mỗi trạng thái là một cách hiển thị riêng. */
export type ExplainState =
  | { status: "loading" }
  | { status: "ready"; data: ExplainResponse }
  /** Câu dài hơn giới hạn từ của giải thích: không gọi, chỉ báo "không tô từ để giữ tốc độ". */
  | { status: "skipped" }
  | { status: "error"; failure: ApiFailure };

export type BaselineState =
  | { status: "off" }
  | { status: "loading" }
  | { status: "ready"; data: BaselineComparison }
  | { status: "error"; failure: ApiFailure };

export type AnalysisState =
  | { phase: "idle" }
  | { phase: "loading"; text: string }
  | { phase: "error"; text: string; failure: ApiFailure }
  | { phase: "done"; text: string; analysis: AnalyzeResponse; explain: ExplainState };

/** Kết quả đã nhận của mô hình cơ sở, gắn với câu đã hỏi để biết còn khớp câu đang hiển thị không. */
interface BaselineResult {
  text: string;
  state: Extract<BaselineState, { status: "ready" | "error" }>;
}

export interface UseAnalysis {
  state: AnalysisState;
  baseline: BaselineState;
  run: (text: string) => void;
  retryExplain: () => void;
  reset: () => void;
}

/**
 * Phân tích một câu theo hai pha: `/api/analyze` trước để nhãn hiện NGAY, rồi `/api/explain` (bỏ qua
 * khi câu quá dài). Yêu cầu cũ bị huỷ khi người dùng đổi câu, nên kết quả muộn không ghi đè kết quả mới.
 * Bật `compare` thì lấy thêm mô hình cơ sở bằng một yêu cầu riêng, không làm chậm hay làm hỏng kết quả chính.
 */
export function useAnalysis(compare: boolean): UseAnalysis {
  const [state, setState] = useState<AnalysisState>({ phase: "idle" });
  const [baselineResult, setBaselineResult] = useState<BaselineResult | null>(null);
  const controller = useRef<AbortController | null>(null);

  const explain = useCallback(async (text: string, signal: AbortSignal) => {
    setState((prev) =>
      prev.phase === "done" && prev.text === text ? { ...prev, explain: { status: "loading" } } : prev,
    );
    let next: ExplainState;
    try {
      const data = await call(api.POST("/api/explain", { body: { text }, signal: requestSignal(signal) }));
      next = { status: "ready", data };
    } catch (error) {
      if (isAbortError(error)) return;
      const failure = toFailure(error);
      next = failure.code === "xai_too_long" ? { status: "skipped" } : { status: "error", failure };
    }
    if (signal.aborted) return;
    setState((prev) => (prev.phase === "done" && prev.text === text ? { ...prev, explain: next } : prev));
  }, []);

  const run = useCallback(
    (rawText: string) => {
      const text = rawText.trim();
      if (!text) return;
      controller.current?.abort();
      const abort = new AbortController();
      controller.current = abort;
      setState({ phase: "loading", text });
      void (async () => {
        let analysis: AnalyzeResponse;
        try {
          analysis = await call(
            api.POST("/api/analyze", { body: { text, compare: false }, signal: requestSignal(abort.signal) }),
          );
        } catch (error) {
          if (isAbortError(error)) return;
          setState({ phase: "error", text, failure: toFailure(error) });
          return;
        }
        if (abort.signal.aborted) return;
        if (!analysis.xai_available) {
          setState({ phase: "done", text, analysis, explain: { status: "skipped" } });
          return;
        }
        setState({ phase: "done", text, analysis, explain: { status: "loading" } });
        await explain(text, abort.signal);
      })();
    },
    [explain],
  );

  const retryExplain = useCallback(() => {
    if (state.phase !== "done" || !controller.current) return;
    void explain(state.text, controller.current.signal);
  }, [explain, state]);

  const reset = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
    setState({ phase: "idle" });
    setBaselineResult(null);
  }, []);

  // Mô hình cơ sở: chạy khi công tắc bật và đã có kết quả của câu hiện tại. Trạng thái hiển thị suy ra
  // từ công tắc + câu hiện tại + kết quả đã nhận, nên không cần setState đồng bộ trong effect.
  const doneText = state.phase === "done" ? state.text : null;
  const baseline: BaselineState =
    !compare || doneText === null
      ? { status: "off" }
      : baselineResult && baselineResult.text === doneText
        ? baselineResult.state
        : { status: "loading" };

  useEffect(() => {
    if (!compare || doneText === null) return;
    const abort = new AbortController();
    call(api.POST("/api/analyze", { body: { text: doneText, compare: true }, signal: requestSignal(abort.signal) }))
      .then((analysis) => {
        if (abort.signal.aborted) return;
        setBaselineResult({
          text: doneText,
          state: analysis.baseline
            ? { status: "ready", data: analysis.baseline }
            : { status: "error", failure: toFailure(new Error("thiếu mô hình cơ sở")) },
        });
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) return;
        setBaselineResult({ text: doneText, state: { status: "error", failure: toFailure(error) } });
      });
    return () => abort.abort();
  }, [compare, doneText]);

  return { state, baseline, run, retryExplain, reset };
}
