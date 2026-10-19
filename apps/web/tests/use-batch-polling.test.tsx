import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { BatchStatus } from "@/lib/batch/types";
import { useBatchPolling } from "@/lib/batch/use-batch-polling";

class HttpError extends Error {
  constructor(readonly status: number) {
    super(`http ${status}`);
  }
}

function status(overrides: Partial<BatchStatus> = {}): BatchStatus {
  return {
    state: "running",
    step: "predict",
    done: 10,
    total: 100,
    progress: 0.1,
    eta_s: 20,
    n_rows: 100,
    pii_rows: 3,
    error_code: null,
    illustrative_groups: false,
    ...overrides,
  };
}

type Fetcher = (batchId: string, signal: AbortSignal) => Promise<BatchStatus>;

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe("useBatchPolling", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("không hỏi gì khi chưa có batch", async () => {
    const fetchStatus = vi.fn<Fetcher>();
    const { result } = renderHook(() => useBatchPolling({ batchId: null, fetchStatus }));
    await advance(5000);
    expect(result.current.state).toEqual({ phase: "idle" });
    expect(fetchStatus).not.toHaveBeenCalled();
  });

  it("hỏi mỗi giây khi đang chạy rồi dừng khi xong", async () => {
    const fetchStatus = vi
      .fn<Fetcher>()
      .mockResolvedValueOnce(status({ state: "queued", step: null, done: 0 }))
      .mockResolvedValueOnce(status({ done: 50, progress: 0.5 }))
      .mockResolvedValueOnce(status({ state: "done", step: null, done: 100, progress: 1 }));

    const { result } = renderHook(() => useBatchPolling({ batchId: "b1", fetchStatus }));
    expect(result.current.state).toEqual({ phase: "polling", status: null });

    await advance(0);
    expect(fetchStatus).toHaveBeenCalledTimes(1);
    expect(result.current.state).toMatchObject({ phase: "polling", status: { state: "queued" } });

    await advance(1000);
    expect(fetchStatus).toHaveBeenCalledTimes(2);
    expect(result.current.state).toMatchObject({ phase: "polling", status: { done: 50 } });

    await advance(1000);
    expect(fetchStatus).toHaveBeenCalledTimes(3);
    expect(result.current.state).toMatchObject({ phase: "settled", status: { state: "done" } });

    await advance(10_000);
    expect(fetchStatus).toHaveBeenCalledTimes(3);
  });

  it("chỉ hỏi một lần khi batch chưa chạy (uploaded) hoặc đã lỗi", async () => {
    const fetchStatus = vi
      .fn<Fetcher>()
      .mockResolvedValue(status({ state: "failed", step: null, error_code: "internal_error" }));
    const { result } = renderHook(() => useBatchPolling({ batchId: "b1", fetchStatus }));
    await advance(5000);
    expect(fetchStatus).toHaveBeenCalledTimes(1);
    expect(result.current.state).toMatchObject({
      phase: "settled",
      status: { state: "failed", error_code: "internal_error" },
    });
  });

  it("dừng và báo gone khi nhận 404", async () => {
    const fetchStatus = vi
      .fn<Fetcher>()
      .mockResolvedValueOnce(status())
      .mockRejectedValueOnce(new HttpError(404));
    const { result } = renderHook(() => useBatchPolling({ batchId: "b1", fetchStatus }));
    await advance(0);
    await advance(1000);
    expect(result.current.state).toEqual({ phase: "gone" });
    await advance(10_000);
    expect(fetchStatus).toHaveBeenCalledTimes(2);
  });

  it("thử lại khi lỗi mạng rồi bỏ cuộc sau số lần liên tiếp tối đa", async () => {
    const failure = new TypeError("Failed to fetch");
    const fetchStatus = vi.fn<Fetcher>().mockRejectedValue(failure);
    const { result } = renderHook(() =>
      useBatchPolling({ batchId: "b1", fetchStatus, maxConsecutiveErrors: 3 }),
    );
    await advance(0);
    expect(result.current.state).toMatchObject({ phase: "polling", status: null });
    await advance(60_000);
    expect(fetchStatus).toHaveBeenCalledTimes(3);
    expect(result.current.state).toEqual({ phase: "error", failure });
  });

  it("lỗi mạng thoáng qua không làm mất tiến độ", async () => {
    const fetchStatus = vi
      .fn<Fetcher>()
      .mockResolvedValueOnce(status({ done: 20 }))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(status({ state: "done", step: null, progress: 1 }));
    const { result } = renderHook(() => useBatchPolling({ batchId: "b1", fetchStatus }));
    await advance(0);
    await advance(1000);
    await advance(5000);
    expect(result.current.state).toMatchObject({ phase: "settled", status: { state: "done" } });
  });

  it("restart hỏi lại từ đầu và hiện ngay trạng thái do run trả về", async () => {
    const fetchStatus = vi
      .fn<Fetcher>()
      .mockResolvedValueOnce(status({ state: "uploaded", step: null, done: 0, total: 0 }))
      .mockResolvedValueOnce(status({ done: 5 }))
      .mockResolvedValueOnce(status({ state: "done", step: null, progress: 1 }));
    const { result } = renderHook(() => useBatchPolling({ batchId: "b1", fetchStatus }));
    await advance(0);
    expect(result.current.state).toMatchObject({ phase: "settled", status: { state: "uploaded" } });

    act(() => result.current.restart(status({ state: "queued", step: null, done: 0 })));
    expect(result.current.state).toMatchObject({ phase: "polling", status: { state: "queued" } });

    await advance(0);
    expect(result.current.state).toMatchObject({ phase: "polling", status: { done: 5 } });
    await advance(1000);
    expect(result.current.state).toMatchObject({ phase: "settled", status: { state: "done" } });
  });

  it("không để lộ trạng thái của batch cũ khi đổi batch", async () => {
    const fetchStatus = vi
      .fn<Fetcher>()
      .mockImplementation(async (id) =>
        id === "a" ? status({ state: "done", step: null, progress: 1 }) : status({ done: 1 }),
      );
    const { result, rerender } = renderHook(
      ({ batchId }) => useBatchPolling({ batchId, fetchStatus }),
      { initialProps: { batchId: "a" as string | null } },
    );
    await advance(0);
    expect(result.current.state).toMatchObject({ phase: "settled" });

    rerender({ batchId: "b" });
    expect(result.current.state).toEqual({ phase: "polling", status: null });
    await advance(0);
    expect(result.current.state).toMatchObject({ phase: "polling", status: { done: 1 } });
  });

  it("huỷ yêu cầu đang bay và ngừng hỏi khi gỡ hook", async () => {
    let received: AbortSignal | undefined;
    const fetchStatus = vi.fn<Fetcher>().mockImplementation(
      (_id, signal) =>
        new Promise<BatchStatus>(() => {
          received = signal;
        }),
    );
    const { unmount } = renderHook(() => useBatchPolling({ batchId: "b1", fetchStatus }));
    await advance(0);
    expect(received?.aborted).toBe(false);
    unmount();
    expect(received?.aborted).toBe(true);
    await advance(10_000);
    expect(fetchStatus).toHaveBeenCalledTimes(1);
  });
});
