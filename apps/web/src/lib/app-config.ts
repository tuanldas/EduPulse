"use client";

// Cấu hình giao diện từ `/api/config`, nạp một lần cho cả ứng dụng (giới hạn, ngân sách duyệt,
// câu mẫu, trạng thái demo, liên kết mã nguồn, chu kỳ heartbeat). Một store mức module thay cho
// Provider: nhiều component gọi `useAppConfig()` nhưng chỉ có một yêu cầu mạng.
import { useEffect, useSyncExternalStore } from "react";

import { api, call, requestSignal, toFailure } from "@/lib/api/client";
import type { ApiFailure } from "@/lib/api/errors";
import type { AppConfig } from "@/lib/api/types";

export type AppConfigState =
  | { status: "loading" }
  | { status: "ready"; config: AppConfig }
  | { status: "error"; failure: ApiFailure };

const LOADING: AppConfigState = { status: "loading" };

let state: AppConfigState = LOADING;
let inflight: Promise<void> | null = null;
const listeners = new Set<() => void>();

function publish(next: AppConfigState): void {
  state = next;
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function fetchConfig(): Promise<void> {
  if (inflight) return inflight;
  inflight = call(api.GET("/api/config", { signal: requestSignal() }))
    .then((config) => publish({ status: "ready", config }))
    .catch((error: unknown) => publish({ status: "error", failure: toFailure(error) }))
    .finally(() => {
      inflight = null;
    });
  return inflight;
}

/** Nạp cấu hình nếu chưa có. Đang lỗi thì không tự thử lại; dùng `retryAppConfig`. */
export function loadAppConfig(): Promise<void> {
  if (state.status !== "loading") return Promise.resolve();
  return fetchConfig();
}

/** Thử lại sau lỗi (nút "Thử lại"). */
export function retryAppConfig(): Promise<void> {
  if (state.status === "error") publish(LOADING);
  return fetchConfig();
}

/** Chỉ dùng trong test: về trạng thái ban đầu. */
export function resetAppConfig(): void {
  inflight = null;
  publish(LOADING);
}

export function useAppConfig(): AppConfigState {
  useEffect(() => {
    void loadAppConfig();
  }, []);
  return useSyncExternalStore(
    subscribe,
    () => state,
    () => LOADING,
  );
}
