import { api, apiUrl, call, requestSignal } from "@/lib/api/client";
import {
  failureFromResponse,
  isAbortError,
  isTimeoutError,
  networkFailure,
} from "@/lib/api/errors";

import type {
  BatchStatus,
  BatchUpload,
  ExplainResponse,
  ExportRequest,
  InsightsResponse,
  ResultsResponse,
  RunRequest,
} from "./types";

/** Tải file lên có thể chậm trên mạng yếu (tới 10 MB), nên cho thời gian dài hơn mặc định. */
const UPLOAD_TIMEOUT_MS = 120_000;
const EXPORT_TIMEOUT_MS = 60_000;

function pathOf(batchId: string) {
  return { params: { path: { batch_id: batchId } } };
}

function isBatchUpload(value: unknown): value is BatchUpload {
  if (typeof value !== "object" || value === null) return false;
  const upload = value as Partial<BatchUpload>;
  return (
    typeof upload.batch_id === "string" &&
    typeof upload.filename === "string" &&
    typeof upload.n_rows === "number" &&
    Array.isArray(upload.columns) &&
    Array.isArray(upload.warnings)
  );
}

/**
 * `POST /api/batches` (multipart). Không dùng client sinh từ OpenAPI vì kiểu của trường tệp là
 * `string`; tự gửi `FormData` và đọc lỗi theo cùng khuôn `{detail:{code,params}}`.
 */
export async function uploadBatch(file: File, signal?: AbortSignal): Promise<BatchUpload> {
  const form = new FormData();
  form.append("file", file, file.name);

  let response: Response;
  try {
    response = await fetch(apiUrl("/api/batches"), {
      method: "POST",
      body: form,
      signal: requestSignal(signal, UPLOAD_TIMEOUT_MS),
    });
  } catch (error) {
    // Cùng quy ước với `call`: huỷ chủ động được ném nguyên, còn lại là lỗi mạng hoặc hết giờ.
    if (isAbortError(error)) throw error;
    throw networkFailure(isTimeoutError(error) ? "timeout" : "network");
  }

  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) throw failureFromResponse(response.status, body);
  if (!isBatchUpload(body)) throw failureFromResponse(502, null);
  return body;
}

export async function createSampleBatch(signal?: AbortSignal): Promise<BatchUpload> {
  return call(api.POST("/api/batches/sample", { signal: requestSignal(signal) }));
}

export async function runBatch(
  batchId: string,
  body: RunRequest,
  signal?: AbortSignal,
): Promise<BatchStatus> {
  return call(
    api.POST("/api/batches/{batch_id}/run", { ...pathOf(batchId), body, signal: requestSignal(signal) }),
  );
}

export async function getBatchStatus(batchId: string, signal?: AbortSignal): Promise<BatchStatus> {
  return call(api.GET("/api/batches/{batch_id}", { ...pathOf(batchId), signal: requestSignal(signal) }));
}

export async function getBatchResults(
  batchId: string,
  signal?: AbortSignal,
): Promise<ResultsResponse> {
  return call(
    api.GET("/api/batches/{batch_id}/results", { ...pathOf(batchId), signal: requestSignal(signal) }),
  );
}

export async function explainRow(
  batchId: string,
  index: number,
  signal?: AbortSignal,
): Promise<ExplainResponse> {
  return call(
    api.GET("/api/batches/{batch_id}/rows/{index}/explain", {
      params: { path: { batch_id: batchId, index } },
      signal: requestSignal(signal),
    }),
  );
}

/** `groups` rỗng hoặc null là mọi nhóm. Nhóm đi trong thân yêu cầu, không đi trong URL. */
export async function getInsights(
  batchId: string,
  groups: readonly string[] | null,
  signal?: AbortSignal,
): Promise<InsightsResponse> {
  const selected = groups !== null && groups.length > 0 ? [...groups] : null;
  return call(
    api.POST("/api/batches/{batch_id}/insights", {
      ...pathOf(batchId),
      body: { groups: selected },
      signal: requestSignal(signal),
    }),
  );
}

/** CSV UTF-8 có BOM; trả về `Blob` để giữ nguyên byte (BOM giúp Excel đọc đúng dấu tiếng Việt). */
export async function exportBatch(
  batchId: string,
  body: ExportRequest,
  signal?: AbortSignal,
): Promise<Blob> {
  return call(
    api.POST("/api/batches/{batch_id}/export", {
      ...pathOf(batchId),
      body,
      parseAs: "blob",
      signal: requestSignal(signal, EXPORT_TIMEOUT_MS),
    }),
  );
}

export async function sendHeartbeat(batchId: string, signal?: AbortSignal): Promise<void> {
  await call(
    api.POST("/api/batches/{batch_id}/heartbeat", { ...pathOf(batchId), signal: requestSignal(signal) }),
  );
}

export async function deleteBatch(batchId: string, signal?: AbortSignal): Promise<void> {
  await call(
    api.DELETE("/api/batches/{batch_id}", { ...pathOf(batchId), signal: requestSignal(signal) }),
  );
}

/**
 * Báo máy chủ xoá batch khi trang đang đóng. `navigator.sendBeacon` sống sót qua việc đóng tab và chỉ
 * gửi được POST, nên dùng `POST …/close`. Không có beacon thì rơi về `fetch` với `keepalive`.
 */
export function sendClose(batchId: string): void {
  const url = apiUrl(`/api/batches/${encodeURIComponent(batchId)}/close`);
  if (typeof navigator !== "undefined" && typeof navigator.sendBeacon === "function") {
    if (navigator.sendBeacon(url)) return;
  }
  void fetch(url, { method: "POST", keepalive: true }).catch(() => undefined);
}
