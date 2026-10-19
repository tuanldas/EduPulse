import { useCallback, useEffect, useState } from "react";

import { isAbortError } from "./failure";
import { useLatest } from "./use-latest";

export type AsyncState<T> =
  /** `key` là null: chưa cần tải. */
  | { status: "idle" }
  /** Đang tải; `stale` là kết quả gần nhất của một khoá trước (để màn hình không nhảy). */
  | { status: "loading"; stale: T | null }
  | { status: "ready"; data: T }
  | { status: "error"; failure: unknown };

export type UseAsyncData<T> = AsyncState<T> & { reload: () => void };

type Outcome<T> = { ok: true; data: T } | { ok: false; failure: unknown };

interface Result<T> {
  key: string;
  attempt: number;
  outcome: Outcome<T>;
  /** Dữ liệu thành công gần nhất, giữ qua các khoá để làm `stale`. */
  lastData: T | null;
}

/**
 * Tải dữ liệu theo `key`: đổi khoá thì huỷ yêu cầu cũ và tải lại; `reload()` tải lại cùng khoá.
 * Trạng thái được gắn với khoá nên kết quả của khoá cũ không bao giờ hiện cho khoá mới.
 * `load` có thể là closure mới ở mỗi lần render: hook luôn gọi bản mới nhất.
 */
export function useAsyncData<T>(
  key: string | null,
  load: (signal: AbortSignal) => Promise<T>,
): UseAsyncData<T> {
  const [attempt, setAttempt] = useState(0);
  const [result, setResult] = useState<Result<T> | null>(null);
  const latestLoad = useLatest(load);

  useEffect(() => {
    if (key === null) return;
    const controller = new AbortController();
    const { signal } = controller;
    latestLoad.current(signal).then(
      (data) => {
        if (signal.aborted) return;
        setResult({ key, attempt, outcome: { ok: true, data }, lastData: data });
      },
      (failure: unknown) => {
        if (signal.aborted || isAbortError(failure)) return;
        setResult((previous) => ({
          key,
          attempt,
          outcome: { ok: false, failure },
          lastData: previous?.lastData ?? null,
        }));
      },
    );
    return () => controller.abort();
  }, [key, attempt, latestLoad]);

  const reload = useCallback(() => setAttempt((previous) => previous + 1), []);

  let state: AsyncState<T>;
  if (key === null) {
    state = { status: "idle" };
  } else if (result !== null && result.key === key && result.attempt === attempt) {
    state = result.outcome.ok
      ? { status: "ready", data: result.outcome.data }
      : { status: "error", failure: result.outcome.failure };
  } else {
    state = { status: "loading", stale: result?.lastData ?? null };
  }
  return { ...state, reload };
}
