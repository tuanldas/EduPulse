import { useCallback, useEffect, useState } from "react";

import { isAbortError, isNotFound } from "./failure";
import { isActiveState, type BatchStatus } from "./types";

export type PollingState =
  /** Chưa có batch để hỏi. */
  | { phase: "idle" }
  /** Đang hỏi tiến độ; `status` là null cho tới khi có câu trả lời đầu tiên. */
  | { phase: "polling"; status: BatchStatus | null }
  /** Đã biết trạng thái dừng: `uploaded`, `done` hoặc `failed`. */
  | { phase: "settled"; status: BatchStatus }
  /** 404: batch không còn (hết hạn, đã đóng). */
  | { phase: "gone" }
  /** Mất kết nối liên tục; người dùng có thể thử lại bằng `restart`. */
  | { phase: "error"; failure: unknown };

export interface UseBatchPollingOptions {
  /** null: không có batch, hook ở trạng thái `idle`. */
  batchId: string | null;
  /** Phải có danh tính ổn định giữa các lần render (hàm cấp module hoặc `useCallback`). */
  fetchStatus: (batchId: string, signal: AbortSignal) => Promise<BatchStatus>;
  /** Chu kỳ hỏi tiến độ; mặc định 1 giây (1 Hz). */
  intervalMs?: number;
  /** Số lần lỗi mạng liên tiếp trước khi bỏ cuộc. */
  maxConsecutiveErrors?: number;
}

export interface UseBatchPolling {
  state: PollingState;
  /**
   * Hỏi lại từ đầu, vd. ngay sau khi gọi `run`. Truyền trạng thái do `run` trả về (nếu có) để hiện
   * ngay, không phải chờ lượt hỏi đầu tiên.
   */
  restart: (initial?: BatchStatus) => void;
}

type Settled = Exclude<PollingState, { phase: "idle" }>;

const DEFAULT_INTERVAL_MS = 1000;
const DEFAULT_MAX_ERRORS = 5;
const MAX_BACKOFF_STEPS = 5;

function fromStatus(status: BatchStatus): Settled {
  return isActiveState(status.state)
    ? { phase: "polling", status }
    : { phase: "settled", status };
}

/**
 * Hỏi `GET /batches/{id}` mỗi giây khi batch đang chờ/chạy; dừng khi xong, lỗi hoặc 404. Kết quả
 * được gắn với (batchId, lần chạy) nên đổi batch không để lộ trạng thái của batch cũ.
 */
export function useBatchPolling(options: UseBatchPollingOptions): UseBatchPolling {
  const {
    batchId,
    fetchStatus,
    intervalMs = DEFAULT_INTERVAL_MS,
    maxConsecutiveErrors = DEFAULT_MAX_ERRORS,
  } = options;

  const [control, setControl] = useState<{ run: number; initial: BatchStatus | null }>({
    run: 0,
    initial: null,
  });
  const [snapshot, setSnapshot] = useState<{ key: string; value: Settled } | null>(null);
  const key = `${batchId ?? ""}#${control.run}`;

  useEffect(() => {
    if (batchId === null) return;
    const controller = new AbortController();
    const { signal } = controller;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let consecutiveErrors = 0;

    const publish = (value: Settled) => {
      if (!signal.aborted) setSnapshot({ key, value });
    };

    const tick = async () => {
      try {
        const status = await fetchStatus(batchId, signal);
        if (signal.aborted) return;
        consecutiveErrors = 0;
        publish(fromStatus(status));
        if (isActiveState(status.state)) timer = setTimeout(() => void tick(), intervalMs);
      } catch (failure) {
        if (signal.aborted || isAbortError(failure)) return;
        if (isNotFound(failure)) {
          publish({ phase: "gone" });
          return;
        }
        consecutiveErrors += 1;
        if (consecutiveErrors >= maxConsecutiveErrors) {
          publish({ phase: "error", failure });
          return;
        }
        const backoff = Math.min(consecutiveErrors + 1, MAX_BACKOFF_STEPS);
        timer = setTimeout(() => void tick(), intervalMs * backoff);
      }
    };

    void tick();
    return () => {
      controller.abort();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [batchId, key, fetchStatus, intervalMs, maxConsecutiveErrors]);

  const restart = useCallback((initial?: BatchStatus) => {
    setControl((previous) => ({ run: previous.run + 1, initial: initial ?? null }));
  }, []);

  let state: PollingState;
  if (batchId === null) state = { phase: "idle" };
  else if (snapshot !== null && snapshot.key === key) state = snapshot.value;
  else if (control.initial !== null) state = fromStatus(control.initial);
  else state = { phase: "polling", status: null };

  return { state, restart };
}
